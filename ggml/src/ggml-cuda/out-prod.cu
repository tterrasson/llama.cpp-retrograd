#include "out-prod.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "../ggml-retro-quant.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

// Native quantized OUT_PROD (OPTIMS_V4 Q2).
//
// Keep the reduction in F32 and in increasing-k order, but decode src0 directly
// into a small shared-memory tile instead of materializing an F32 copy in the
// CUDA pool. A 256-row tile is intentional: the K/IQ decoders already expose a
// cooperative 256-value super-block contract, so the same kernel covers every
// type in GGML_RETRO_OUT_PROD_TYPES without duplicating quantization formulae.
// Each block produces 256 x 32 output values. The 32 accumulators per thread
// are a better trade than re-reading/dequantizing the frozen weight for every
// skinny output tile (the Vulkan reference uses 64 x 16).
static constexpr int OUT_PROD_Q_BM = 256;
static constexpr int OUT_PROD_Q_BN = 32;
static constexpr int OUT_PROD_Q_BK = 4;
static constexpr int OUT_PROD_Q_TM = 16;
static constexpr int OUT_PROD_Q_TN = 2;
static constexpr int OUT_PROD_Q_THREADS = 256;

struct out_prod_f16_loader {
    static __device__ __forceinline__ void load(
            const void * row, const int64_t m0, const int64_t ne0, float * tile) {
        const int m = threadIdx.x;
        if (m0 + m < ne0) {
            tile[m] = __half2float(((const half *) row)[m0 + m]);
        }
    }
};

template<int qk, int qr, dequantize_kernel_t dequantize_kernel>
struct out_prod_pair_loader {
    static __device__ __forceinline__ void load(
            const void * row, const int64_t m0, const int64_t ne0, float * tile) {
        const int pair = threadIdx.x;
        if (pair >= OUT_PROD_Q_BM/2 || m0 + 2*pair >= ne0) {
            return;
        }

        const int i00 = 2*pair;
        const int ib  = (m0 + i00)/qk;
        const int iqs = ((m0 + i00)%qk)/qr;
        const int block_start = i00 - i00%qk;
        const int second = qr == 1 ? 1 : qk/2;

        float2 v;
        dequantize_kernel(row, ib, iqs, v);
        tile[block_start + iqs]          = v.x;
        tile[block_start + iqs + second] = v.y;
    }
};

template<int dequant_threads, dequantize_kq_t<float> dequantize_block>
struct out_prod_superblock_loader {
    static __device__ __forceinline__ void load(
            const void * row, const int64_t m0, const int64_t ne0, float * tile) {
        GGML_UNUSED(ne0);
        if (threadIdx.x < dequant_threads) {
            dequantize_block(row, m0/QK_K, tile, threadIdx.x);
        }
    }
};

struct out_prod_nvfp4_loader {
    static __device__ __forceinline__ void load(
            const void * row, const int64_t m0, const int64_t ne0, float * tile) {
        const int local = threadIdx.x;
        const int64_t m = m0 + local;
        if (m >= ne0) {
            return;
        }
        const block_nvfp4 * blocks = (const block_nvfp4 *) row;
        const block_nvfp4 & block = blocks[m/QK_NVFP4];
        const int in_block = m%QK_NVFP4;
        const int sub = in_block/QK_NVFP4_SUB;
        const int in_sub = in_block%QK_NVFP4_SUB;
        const uint8_t q = block.qs[sub*(QK_NVFP4_SUB/2) + in_sub%(QK_NVFP4_SUB/2)];
        const int nibble = in_sub < QK_NVFP4_SUB/2 ? q & 0x0f : q >> 4;
        tile[local] = ggml_cuda_ue4m3_to_fp32(block.d[sub])*kvalues_mxfp4[nibble];
    }
};

// The type-to-loader mapping is kept declarative.  The dispatch below expands
// GGML_RETRO_OUT_PROD_TYPES, so adding a CPU/Vulkan/CUDA OUT_PROD type without a
// CUDA loader is a compile error rather than a silent trip through the fallback.
template<ggml_type type> struct out_prod_native_traits;

#define OUT_PROD_F16_TRAITS(type) \
    template<> struct out_prod_native_traits<type> { \
        using loader = out_prod_f16_loader; \
        static constexpr int64_t alignment = 1; \
    }
#define OUT_PROD_PAIR_TRAITS(type, qk, qr, decoder) \
    template<> struct out_prod_native_traits<type> { \
        using loader = out_prod_pair_loader<qk, qr, decoder>; \
        static constexpr int64_t alignment = qk; \
    }
#define OUT_PROD_SUPER_TRAITS(type, threads, decoder) \
    template<> struct out_prod_native_traits<type> { \
        using loader = out_prod_superblock_loader<threads, decoder>; \
        static constexpr int64_t alignment = QK_K; \
    }

OUT_PROD_F16_TRAITS(GGML_TYPE_F16);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q1_0, QK1_0, QR1_0, dequantize_q1_0);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q2_0, QK2_0, QR2_0, dequantize_q2_0);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q4_0, QK4_0, QR4_0, dequantize_q4_0);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q4_1, QK4_1, QR4_1, dequantize_q4_1);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q5_0, QK5_0, QR5_0, dequantize_q5_0);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q5_1, QK5_1, QR5_1, dequantize_q5_1);
OUT_PROD_PAIR_TRAITS(GGML_TYPE_Q8_0, QK8_0, QR8_0, dequantize_q8_0);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_MXFP4,   32, dequantize_mxfp4<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_Q2_K,    64, dequantize_q2_K<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_Q3_K,    64, dequantize_q3_K<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_Q4_K,    32, dequantize_q4_K<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_Q5_K,    64, dequantize_q5_K<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_Q6_K,    64, dequantize_q6_K<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ2_XXS, 32, dequantize_iq2_xxs<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ2_XS,  32, dequantize_iq2_xs<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ2_S,   32, dequantize_iq2_s<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ3_XXS, 32, dequantize_iq3_xxs<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ3_S,   32, dequantize_iq3_s<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ1_S,   32, dequantize_iq1_s<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ1_M,   32, dequantize_iq1_m<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ4_NL,  32, dequantize_iq4_nl<float>);
OUT_PROD_SUPER_TRAITS(GGML_TYPE_IQ4_XS,  32, dequantize_iq4_xs<float>);
template<> struct out_prod_native_traits<GGML_TYPE_NVFP4> {
    using loader = out_prod_nvfp4_loader;
    static constexpr int64_t alignment = QK_NVFP4;
};

#undef OUT_PROD_F16_TRAITS
#undef OUT_PROD_PAIR_TRAITS
#undef OUT_PROD_SUPER_TRAITS

template<typename loader>
static __global__ void k_out_prod_quant(
        const void * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne10,
        const int64_t ne2, const int64_t ne3, const int64_t dps2, const int64_t dps3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    __shared__ float tile_a[OUT_PROD_Q_BK*OUT_PROD_Q_BM];
    __shared__ float tile_b[OUT_PROD_Q_BK*OUT_PROD_Q_BN];

    const int tid = threadIdx.x;
    const int thread_m = tid / 16;
    const int thread_n = tid % 16;
    const int64_t m0 = (int64_t) blockIdx.x*OUT_PROD_Q_BM;
    const int64_t n0 = (int64_t) blockIdx.y*OUT_PROD_Q_BN;

    for (int64_t batch = blockIdx.z; batch < ne2*ne3; batch += gridDim.z) {
        const int64_t i2 = batch % ne2;
        const int64_t i3 = batch / ne2;
        const int64_t a_i2 = i2/dps2;
        const int64_t a_i3 = i3/dps3;
        const char * a_plane = (const char *) src0 + a_i3*nb03 + a_i2*nb02;
        const char * b_plane = (const char *) src1 + i3*nb13 + i2*nb12;
        char * d_plane = (char *) dst + i3*nb3 + i2*nb2;

        float acc[OUT_PROD_Q_TM][OUT_PROD_Q_TN] = {};

        for (int64_t k0 = 0; k0 < ne01; k0 += OUT_PROD_Q_BK) {
            for (int l = tid; l < OUT_PROD_Q_BK*OUT_PROD_Q_BM; l += OUT_PROD_Q_THREADS) {
                tile_a[l] = 0.0f;
            }
            for (int l = tid; l < OUT_PROD_Q_BK*OUT_PROD_Q_BN; l += OUT_PROD_Q_THREADS) {
                tile_b[l] = 0.0f;
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < OUT_PROD_Q_BK; ++kk) {
                const int64_t k = k0 + kk;
                if (k < ne01) {
                    loader::load(a_plane + k*nb01, m0, ne00, tile_a + kk*OUT_PROD_Q_BM);
                }
            }
            for (int l = tid; l < OUT_PROD_Q_BK*OUT_PROD_Q_BN; l += OUT_PROD_Q_THREADS) {
                const int kk = l/OUT_PROD_Q_BN;
                const int nn = l%OUT_PROD_Q_BN;
                const int64_t k = k0 + kk;
                const int64_t n = n0 + nn;
                if (k < ne01 && n < ne10) {
                    tile_b[l] = *(const float *) (b_plane + n*nb10 + k*nb11);
                }
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < OUT_PROD_Q_BK; ++kk) {
#pragma unroll
                for (int r = 0; r < OUT_PROD_Q_TM; ++r) {
                    const float av = tile_a[kk*OUT_PROD_Q_BM + thread_m*OUT_PROD_Q_TM + r];
#pragma unroll
                    for (int c = 0; c < OUT_PROD_Q_TN; ++c) {
                        const float bv = tile_b[kk*OUT_PROD_Q_BN + thread_n*OUT_PROD_Q_TN + c];
                        // Explicit round-to-nearest operations make the contract
                        // independent of -use_fast_math's FMA contraction.
                        acc[r][c] = __fadd_rn(acc[r][c], __fmul_rn(av, bv));
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int r = 0; r < OUT_PROD_Q_TM; ++r) {
            const int64_t m = m0 + thread_m*OUT_PROD_Q_TM + r;
#pragma unroll
            for (int c = 0; c < OUT_PROD_Q_TN; ++c) {
                const int64_t n = n0 + thread_n*OUT_PROD_Q_TN + c;
                if (m < ne00 && n < ne10) {
                    *(float *) (d_plane + n*nb1 + m*sizeof(float)) = acc[r][c];
                }
            }
        }
    }
}

template<typename loader>
static void launch_out_prod_quant_native(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const int64_t batches = ne2*ne3;
    GGML_ASSERT(batches > 0);
    const dim3 blocks(
        (src0->ne[0] + OUT_PROD_Q_BM - 1)/OUT_PROD_Q_BM,
        (src1->ne[0] + OUT_PROD_Q_BN - 1)/OUT_PROD_Q_BN,
        std::min<int64_t>(batches, 65535));

    k_out_prod_quant<loader><<<blocks, OUT_PROD_Q_THREADS, 0, ctx.stream()>>>(
        src0->data, (const float *) src1->data, (float *) dst->data,
        src0->ne[0], src0->ne[1], src1->ne[0], ne2, ne3,
        ne2/src0->ne[2], ne3/src0->ne[3],
        src0->nb[1], src0->nb[2], src0->nb[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

template<ggml_type type>
static bool launch_out_prod_quant_type(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = out_prod_native_traits<type>;
    if (src0->ne[0] % traits::alignment != 0) {
        return false;
    }
    launch_out_prod_quant_native<typename traits::loader>(ctx, src0, src1, dst);
    return true;
}

// Returns false only for a partial 256-value super-block (K/IQ/MXFP4). Those
// uncommon geometries retain bounded dequantize+SGEMM as a correctness fallback.
static bool ggml_cuda_out_prod_quant_native(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#define OUT_PROD_NATIVE_CASE(TYPE, BLK, NL, NAME, VKNAME) \
        case TYPE: return launch_out_prod_quant_type<TYPE>(ctx, src0, src1, dst);
    switch (src0->type) {
        GGML_RETRO_OUT_PROD_TYPES(OUT_PROD_NATIVE_CASE)
        default: return false;
    }
#undef OUT_PROD_NATIVE_CASE
}

static __global__ void k_compute_out_prod_ptrs(
        const float * src0_d, const float * src1_d, float * dst_d,
        const float ** ptrs_a, const float ** ptrs_b, float ** ptrs_c,
        const int64_t ne2, const int64_t ne3,
        const int64_t dps2, const int64_t dps3,
        const size_t s02, const size_t s03,
        const size_t s12, const size_t s13,
        const size_t s2,  const size_t s3) {
    const int64_t i2 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i3 = blockIdx.y*blockDim.y + threadIdx.y;

    if (i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t idx = i3*ne2 + i2;

    ptrs_a[idx] = src0_d + (i3/dps3)*s03 + (i2/dps2)*s02;
    ptrs_b[idx] = src1_d +  i3      *s13 +  i2      *s12;
    ptrs_c[idx] = dst_d  +  i3      *s3  +  i2      *s2;
}

// retro delta: ceiling on the F32 scratch used to dequantize a quantized src0.
// Dequantizing the whole weight at once costs ne00*ne01*ne02*ne03*4 bytes, which
// the CUDA pool then holds at its high-water mark for the rest of the process --
// about 225 MiB for a 12B `ffn_up`, paid on every LoRA backward. Slicing the
// reduction axis keeps that scratch bounded. A budget that already covers the
// whole weight produces exactly one slice, hence the same single GEMM and the
// same arithmetic as an unsliced run: small weights are bit-identical either way.
// GGML_CUDA_DEQUANT_BUDGET_MB overrides the default; 0 or less means "do not
// slice". Read on every call rather than cached in a static: the value is only
// consulted once per OUT_PROD node, which is negligible next to that node's GEMM,
// and a cache would freeze on whatever the environment held when the first out
// product of the process ran -- making the knob untestable from inside a suite.
static int64_t ggml_cuda_dequant_budget_bytes() {
    const char * env = getenv("GGML_CUDA_DEQUANT_BUDGET_MB");
    const int64_t mb = env ? std::atoll(env) : 64;
    return mb > 0 ? mb*1024*1024 : std::numeric_limits<int64_t>::max();
}

// One reduction slice of the out product: dst = alpha*src0*src1^T + beta*dst over
// `k` reduction elements. `beta` is 0 for the first slice (overwrite) and 1 for
// every later one (accumulate), so slicing only changes how the k-sum is grouped.
static void ggml_cuda_out_prod_gemm(
        ggml_backend_cuda_context & ctx,
        const float * src0_d, int64_t lda, size_t s02, size_t s03,
        const float * src1_d, int64_t ldb, cublasOperation_t src1_op, size_t s12, size_t s13,
        float * dst_d, int64_t ldc, size_t s2, size_t s3,
        int64_t ne0, int64_t ne1, int64_t k,
        int64_t ne2, int64_t ne3, int64_t dps2, int64_t dps3,
        float beta) {
    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    // retro delta: OUT_PROD is a *gradient* (dW for a LoRA factor, dX through a
    // frozen projection), so it runs in true F32 rather than the TF32 mode
    // common.cuh installs on every handle for inference mul_mat. Two reasons, and
    // the second is why this is not merely a nicety:
    //   - TF32 keeps 10 mantissa bits, i.e. ~1e-3 relative. A gradient that wrong
    //     does not fail a loss test, it slowly degrades training.
    //   - cuBLAS picks TF32 tensor-core kernels per shape, so with TF32 on, the
    //     result depends on `k` -- and `k` is the reduction slice width chosen from
    //     a scratch budget below. That made a pure regrouping of the k-sum move the
    //     result by 1.3e-3 (measured, RTX 4090) and would make the budget, a memory
    //     knob, silently change numerics.
    // Same set/restore idiom as solve_tri.cu, which needs it for the same reason.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f;

    if (dps2 == 1 && ne2 > 1) {
        // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
        GGML_ASSERT(ne2 <= std::numeric_limits<int>::max());
        const int batch_count = (int) ne2;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            CUBLAS_CHECK(
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_op,
                        ne0, ne1, k,
                        &alpha, src0_d + (i3/dps3)*s03, lda, s02,
                                src1_d +  i3     *s13, ldb, s12,
                        &beta,  dst_d  +  i3     *s3,  ldc, s2,
                        batch_count));
        }
    } else if (ne2 > 1 || ne3 > 1) {
        // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
        // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
        GGML_ASSERT(ne3 > 0);
        GGML_ASSERT(ne2 <= (int64_t) std::numeric_limits<int>::max() / ne3);
        const int batch_count = (int) (ne2 * ne3);

        ggml_cuda_pool_alloc<const float *> ptrs_a(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<const float *> ptrs_b(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<      float *> ptrs_c(ctx.pool(), batch_count);

        const dim3 block_dims(16, 16);
        const dim3 grid_dims((ne2 + block_dims.x - 1)/block_dims.x, (ne3 + block_dims.y - 1)/block_dims.y);
        k_compute_out_prod_ptrs<<<grid_dims, block_dims, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ptrs_a.get(), ptrs_b.get(), ptrs_c.get(),
            ne2, ne3, dps2, dps3, s02, s03, s12, s13, s2, s3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
            cublasSgemmBatched(handle, CUBLAS_OP_N, src1_op,
                    ne0, ne1, k,
                    &alpha, ptrs_a.get(), lda,
                            ptrs_b.get(), ldb,
                    &beta,  ptrs_c.get(), ldc,
                    batch_count));
    } else {
        // ne2 == 1 && ne3 == 1: single GEMM
        CUBLAS_CHECK(
            cublasSgemm(handle, CUBLAS_OP_N, src1_op,
                    ne0, ne1, k,
                    &alpha, src0_d, lda,
                            src1_d, ldb,
                    &beta,  dst_d,  ldc));
    }

    // revert to standard mode from common.cuh
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // retro delta: OUT_PROD with a quantized src0 is the input-gradient of a
    // quantized (frozen base) projection during LoRA training. Upstream CUDA
    // only handles F32 x F32; the CPU/Vulkan forks dequantize src0 and accumulate
    // in F32. Here src0 is dequantized with ggml-cuda's existing per-type kernels
    // (bit-identical to the CPU oracle's dequant) and the proven cuBLAS path is
    // reused, so every type with a to_fp32 kernel is covered with F32 accumulation
    // (F16 included) -- but in slices along the reduction axis, so the F32 scratch
    // never scales with the whole weight (see ggml_cuda_dequant_budget_bytes).
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const int64_t ldc = nb1 / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb     = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                      (src1_T ?        nb11 :        nb10) == sizeof(float));

    // Distance in floats between two consecutive reduction elements of src1, used
    // to offset a reduction slice: with CUBLAS_OP_N cuBLAS reads src1 as [k, ne1]
    // so k advances by one element, with CUBLAS_OP_T it reads [ne1, k] so k
    // advances by one leading dimension.
    const int64_t src1_k_stride = src1_T ? 1 : ldb;

    // data strides in dimensions 2/3
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_out_prod_gemm(ctx,
            (const float *) src0->data, nb01 / sizeof(float), nb02 / sizeof(float), nb03 / sizeof(float),
            src1_d, ldb, src1_op, s12, s13,
            dst_d, ldc, s2, s3,
            ne0, ne1, ne01, ne2, ne3, dps2, dps3,
            /*beta =*/ 0.0f);
        return;
    }

    // Q2: every supported training type decodes inside the tiled kernel and
    // never allocates dequantization scratch. The legacy path remains below only
    // for partial K/IQ/MXFP4 super-blocks.
    if (ggml_cuda_out_prod_quant_native(ctx, src0, src1, dst)) {
        return;
    }

    // The dequant kernels consume a contiguous run of quantized blocks, matching
    // the packed weight layout (nb00 == type size, rows tightly packed).
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_is_contiguous(src0));

    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
    GGML_ASSERT(to_fp32 != nullptr);

    // A src0 column spans a whole number of quantization blocks, so any slice of
    // whole columns starts and ends on a block boundary and can be handed to the
    // dequant kernel as-is.
    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    const int64_t n_planes = ne02*ne03;

    // Columns of src0 per slice, from the scratch budget. At least one column per
    // plane is always dequantized, even if that exceeds the budget: below that
    // there is nothing left to slice.
    const int64_t budget_cols = ggml_cuda_dequant_budget_bytes()/((int64_t) sizeof(float)*ne00*n_planes);
    const int64_t slice_cols  = std::max<int64_t>(1, std::min<int64_t>(ne01, budget_cols));

    ggml_cuda_pool_alloc<float> src0_f32(ctx.pool(), (size_t) ne00*slice_cols*n_planes);
    cudaStream_t stream = ctx.stream();

    for (int64_t j0 = 0; j0 < ne01; j0 += slice_cols) {
        const int64_t nj = std::min(slice_cols, ne01 - j0);

        // Dequantize columns [j0, j0+nj) of every src0 plane into a contiguous
        // [ne00, nj, ne02, ne03] scratch, so the GEMM sees the packed row layout.
        for (int64_t p = 0; p < n_planes; ++p) {
            const int64_t i2 = p % ne02;
            const int64_t i3 = p / ne02;
            const char * q = (const char *) src0->data + i3*nb03 + i2*nb02 + j0*nb01;
            to_fp32(q, src0_f32.get() + p*ne00*nj, ne00*nj, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_out_prod_gemm(ctx,
            src0_f32.get(), ne00, (size_t) ne00*nj, (size_t) ne00*nj*ne02,
            src1_d + j0*src1_k_stride, ldb, src1_op, s12, s13,
            dst_d, ldc, s2, s3,
            ne0, ne1, nj, ne2, ne3, dps2, dps3,
            /*beta =*/ j0 == 0 ? 0.0f : 1.0f);
    }
}
