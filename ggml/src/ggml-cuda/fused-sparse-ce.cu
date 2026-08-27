#include "fused-sparse-ce.cuh"

#include "convert.cuh"
#include "retro-quant-loader.cuh"
#include "sum.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// retro delta: tiled CUDA FUSED_SPARSE_CE[_BACK].  cuBLAS materializes at most
// [min(tile, 1024), n_tokens] logits, never [n_vocab, n_tokens].  Each tile is
// folded into an online F32 log-sum-exp; backward recomputes it, converts the
// tile to scaled probabilities, and accumulates grad_h with another GEMM.
//
// retro delta: optional fixed per-vocab bias (src[4] forward / src[5]
// backward, may be null). Added to every GEMM tile right after cuBLAS and
// before it is folded into the log-sum-exp / turned into probabilities, so it
// participates in the softmax exactly like the CPU oracle. The bias never
// receives a gradient, so the grad_h GEMMs are untouched.

static __global__ void fused_sparse_ce_count_active(
        const int32_t * targets, const float * weights, int32_t * n_active,
        int64_t n_tokens, int64_t n_vocab) {
    const int64_t t = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (t < n_tokens && targets[t] >= 0 && targets[t] < n_vocab && weights[t] != 0.0f) {
        atomicAdd(n_active, 1);
    }
}

static __global__ void fused_sparse_ce_init(float * maxima, float * sums, float * target_logits, int64_t n_tokens) {
    const int64_t t = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (t < n_tokens) {
        maxima[t] = -INFINITY;
        sums[t] = 0.0f;
        target_logits[t] = 0.0f;
    }
}

static __global__ void fused_sparse_ce_add_bias(
        float * logits, const float * bias, int64_t v0,
        int64_t tile_size, int64_t tile_stride, int64_t n_tokens) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = tile_size*n_tokens;
    if (i >= n) {
        return;
    }
    const int64_t t = i/tile_size;
    const int64_t v = i - t*tile_size;
    logits[t*tile_stride + v] += bias[v0 + v];
}

// retro delta (plan rl/OPTIMIZE feature 1): logits holds only the current token
// chunk [tile_size, nt_tok], so it is indexed by the local token lt while the
// per-token state (maxima/sums/target_logits/targets) is indexed by the global
// token t0_tok + lt.
static __global__ void fused_sparse_ce_update_lse(
        const float * logits, const int32_t * targets, float * maxima,
        float * sums, float * target_logits, int64_t tile_start,
        int64_t tile_size, int64_t tile_stride, int64_t t0_tok, int64_t nt_tok) {
    const int64_t lt = blockIdx.x;
    if (lt >= nt_tok || threadIdx.x != 0) {
        return;
    }
    const int64_t t = t0_tok + lt;
    float running_max = maxima[t];
    float running_sum = sums[t];
    const float * column = logits + lt*tile_stride;
    for (int64_t i = 0; i < tile_size; ++i) {
        const float z = column[i];
        if (z > running_max) {
            running_sum = running_sum*expf(running_max - z) + 1.0f;
            running_max = z;
        } else {
            running_sum += expf(z - running_max);
        }
    }
    const int64_t target = targets[t];
    if (target >= tile_start && target < tile_start + tile_size) {
        target_logits[t] = column[target - tile_start];
    }
    maxima[t] = running_max;
    sums[t] = running_sum;
}

static __global__ void fused_sparse_ce_finish_loss(
        const int32_t * targets, const float * weights, const int32_t * n_active,
        const float * maxima, const float * sums, const float * target_logits,
        float * losses, int64_t n_tokens, int64_t n_vocab) {
    const int64_t t = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (t >= n_tokens) {
        return;
    }
    const bool active = targets[t] >= 0 && targets[t] < n_vocab && weights[t] != 0.0f && *n_active > 0;
    losses[t] = active
        ? weights[t]*(maxima[t] + logf(sums[t]) - target_logits[t])/(float) *n_active
        : 0.0f;
}

static __global__ void fused_sparse_ce_make_probs(
        float * logits, const float * grad, const int32_t * targets,
        const float * weights, const int32_t * n_active, const float * maxima,
        const float * sums, int64_t tile_size, int64_t tile_stride,
        int64_t t0_tok, int64_t nt_tok, int64_t n_vocab) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = tile_size*nt_tok;
    if (i >= n) {
        return;
    }
    const int64_t lt = i/tile_size;
    const int64_t v = i - lt*tile_size;
    const int64_t t = t0_tok + lt;
    const bool active = targets[t] >= 0 && targets[t] < n_vocab && weights[t] != 0.0f && *n_active > 0;
    const float coef = active ? *grad*weights[t]/(float) *n_active : 0.0f;
    const int64_t offset = lt*tile_stride + v;
    logits[offset] = active ? coef*expf(logits[offset] - maxima[t])/sums[t] : 0.0f;
}

// Subtracts the target row from one token chunk's accumulated grad_h. `out` is the
// chunk's [n_embd, nt] output. `gathered` selects where the target row comes from:
// a [n_embd, nt] capture indexed by the chunk-local token (quantized head, see
// fused_sparse_ce_capture_target_rows), or the F32 head itself indexed by target
// row (F32 head, unchanged behaviour). Either way this is per-element the same
// arithmetic on the same fully accumulated value as before.
template <bool gathered>
static __global__ void fused_sparse_ce_subtract_target(
        const float * grad, const float * rows, const int32_t * targets,
        const float * weights, const int32_t * n_active, float * out,
        int64_t n_embd, int64_t t0, int64_t nt, int64_t n_vocab) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n_embd*nt) {
        return;
    }
    const int64_t lt = i/n_embd;
    const int64_t e  = i - lt*n_embd;
    const int64_t t  = t0 + lt;
    const int32_t target = targets[t];
    if (target >= 0 && target < n_vocab && weights[t] != 0.0f && *n_active > 0) {
        const float coef = *grad*weights[t]/(float) *n_active;
        out[i] -= coef*rows[(gathered ? lt : int64_t(target))*n_embd + e];
    }
}

// retro delta: with a quantized head the target subtraction can no longer index an
// F32 copy of the whole head. Each token's target row is captured here, out of the
// one vocab tile that contains it, into the chunk's [n_embd, nt] F32 buffer, so the
// subtraction keeps operating on a plain F32 row. Bounded by the token chunk like
// every other intermediate of this operator.
static __global__ void fused_sparse_ce_capture_target_rows(
        const float * w_tile, const int32_t * targets, float * target_rows,
        int64_t n_embd, int64_t t0, int64_t nt, int64_t v0, int64_t nv) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n_embd*nt) {
        return;
    }
    const int64_t lt = i/n_embd;
    const int64_t e  = i - lt*n_embd;
    const int64_t target = targets[t0 + lt];
    if (target < v0 || target >= v0 + nv) {
        return;
    }
    target_rows[lt*n_embd + e] = w_tile[(target - v0)*n_embd + e];
}

// retro delta: bounded per-tile F32 view of a possibly quantized output head.
// Dequantizing the whole head up front costs n_embd*n_vocab*4 bytes of pool --
// 3.75 GiB for a 262k-vocab, 3840-wide head, more than the 4-bit model itself --
// while every consumer below already walks the head one vocab tile at a time. A
// tile of whole vocab rows is a contiguous, block-aligned run of the packed
// weight, so one dequant call per tile suffices and the scratch is bounded by the
// op's own tile capacity. An F32 head is handed out in place: that path allocates
// nothing and keeps its previous behaviour exactly.
struct fused_sparse_ce_head {
    const ggml_tensor * w;
    const int64_t n_embd;
    to_fp32_cuda_t to_fp32 = nullptr;
    ggml_cuda_pool_alloc<float> tile;

    fused_sparse_ce_head(
            ggml_cuda_pool & pool, const ggml_tensor * head, int64_t n_embd, int64_t tile_capacity)
            : w(head), n_embd(n_embd), tile(pool) {
        if (w->type == GGML_TYPE_F32) {
            return;
        }
        // The dequant kernels consume a contiguous run of quantized blocks, and a
        // vocab row spans a whole number of them, so a tile boundary is always a
        // block boundary.
        GGML_ASSERT(w->nb[0] == ggml_type_size(w->type));
        GGML_ASSERT(n_embd % ggml_blck_size(w->type) == 0);
        to_fp32 = ggml_get_to_fp32_cuda(w->type);
        GGML_ASSERT(to_fp32 != nullptr);
        tile.alloc(n_embd*tile_capacity);
    }

    // [n_embd, nv] F32 view of vocab rows [v0, v0+nv). Valid until the next call:
    // consumers must finish with a tile before asking for the following one.
    const float * rows(int64_t v0, int64_t nv, cudaStream_t stream) {
        if (!to_fp32) {
            return (const float *) w->data + v0*n_embd;
        }
        to_fp32((const char *) w->data + v0*w->nb[1], tile.get(), nv*n_embd, stream);
        CUDA_CHECK(cudaGetLastError());
        return tile.get();
    }
};

struct fused_sparse_ce_work {
    ggml_cuda_pool_alloc<int32_t> n_active;
    ggml_cuda_pool_alloc<float> maxima;
    ggml_cuda_pool_alloc<float> sums;
    ggml_cuda_pool_alloc<float> target_logits;

    fused_sparse_ce_work(ggml_cuda_pool & pool, int64_t n_tokens) :
        n_active(pool, 1), maxima(pool, n_tokens), sums(pool, n_tokens), target_logits(pool, n_tokens) {}
};

static void fused_sparse_ce_prepare(
        cudaStream_t stream, const ggml_tensor * targets, const ggml_tensor * weights,
        int64_t n_vocab, fused_sparse_ce_work & work) {
    const int64_t n_tokens = ggml_nelements(targets);
    const int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(work.n_active.get(), 0, sizeof(int32_t), stream));
    fused_sparse_ce_count_active<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        (const int32_t *) targets->data, (const float *) weights->data,
        work.n_active.get(), n_tokens, n_vocab);
    fused_sparse_ce_init<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        work.maxima.get(), work.sums.get(), work.target_logits.get(), n_tokens);
    CUDA_CHECK(cudaGetLastError());
}

static int64_t fused_sparse_ce_tile_size(const ggml_tensor * dst, int64_t n_vocab) {
    const int32_t requested_tiles = std::max(1, ggml_get_op_params_i32(dst, 0));
    return std::min<int64_t>(1024, (n_vocab + requested_tiles - 1)/requested_tiles);
}

// retro delta (plan rl/OPTIMIZE feature 1): op_params[1] caps how many tokens of
// the flattened (batch x seq) axis are processed per pass. 0 means "all tokens"
// (unchanged). The logits intermediate is then [tile_capacity, seq_chunk] instead
// of [tile_capacity, n_tokens], so its peak no longer grows with sequence length.
static int64_t fused_sparse_ce_seq_chunk(const ggml_tensor * dst, int64_t n_tokens) {
    const int32_t seq_chunk = ggml_get_op_params_i32(dst, 1);
    return seq_chunk > 0 ? std::min<int64_t>(seq_chunk, n_tokens) : n_tokens;
}

// Online log-sum-exp for one token chunk [t0, t0+nt). `logits` is a scratch of
// [tile_capacity, nt]; `h` points at the full [n_embd, n_tokens] hidden states
// (offset to the chunk here). maxima/sums/target_logits are the global per-token
// state, indexed by the absolute token id inside the kernels.
static void fused_sparse_ce_lse(
        ggml_backend_cuda_context & ctx, const float * h, fused_sparse_ce_head & head,
        const ggml_tensor * targets, const float * bias, int64_t n_embd,
        int64_t t0, int64_t nt, int64_t n_vocab, int64_t tile_capacity,
        float * logits, fused_sparse_ce_work & work) {
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
    const int threads = 256;
    for (int64_t v0 = 0; v0 < n_vocab; v0 += tile_capacity) {
        const int64_t nv = std::min(tile_capacity, n_vocab - v0);
        const float * w_tile = head.rows(v0, nv, stream);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            nv, nt, n_embd, &alpha, w_tile, n_embd,
            h + t0*n_embd, n_embd, &beta, logits, tile_capacity));
        if (bias) {
            const int64_t n_bias = nv*nt;
            fused_sparse_ce_add_bias<<<(n_bias + threads - 1)/threads, threads, 0, stream>>>(
                logits, bias, v0, nv, tile_capacity, nt);
            CUDA_CHECK(cudaGetLastError());
        }
        fused_sparse_ce_update_lse<<<nt, 1, 0, stream>>>(
            logits, (const int32_t *) targets->data, work.maxima.get(), work.sums.get(),
            work.target_logits.get(), v0, nv, tile_capacity, t0, nt);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// retro delta (OPTIM_V3 O5): in-kernel decode of a quantized head.
//
// The path above bounds its scratch by the vocab tile, but it still *writes*
// every head element it needs as F32 and reads it back through cuBLAS: one
// dequant pass per tile in the forward, two in the backward (the recomputed
// log-sum-exp, then the grad_h accumulation), each of them n_embd*tile floats of
// store-then-load traffic. Metal and Vulkan never pay that -- they read the head
// through the per-type decoders inside the kernel (see
// kernels/retro.metal, fused_sparse_ce{,_back}.comp) -- and this is the CUDA
// version of the same thing, over the shared loader table of
// retro-quant-loader.cuh so no second list of formats appears (O5 points 1-3).
//
// One block per token, 256 threads. The [n_vocab, n_tokens] logits are never
// materialized and neither is any F32 view of the head: a group of CE_Q_ROWS
// vocabulary rows is decoded 256 elements at a time into shared memory and
// consumed immediately. The forward keeps this operator's existing tail
// (fused_sparse_ce_finish_loss then sum_f32_cuda) by writing the same per-token
// maxima/sums/target_logits it always did.
//
// Because logits are never stored, chunked_ce_tiles / chunked_ce_seq_chunk have
// nothing to size here: both are ignored on this path, which is what makes the
// result independent of them rather than merely tolerant of them.
//
// Opt-in through GGML_CUDA_CE_QHEAD=1. The dequantize+cuBLAS path stays the
// default and the oracle: it uses tensor-core SGEMM for the two products, this
// one does not, so which is faster at a given (n_embd, n_vocab, n_tokens) is a
// measurement and not a claim (OPTIM_V3 O5 point 4).
#define CE_Q_ROWS  8   // vocabulary rows decoded per pass; 8 KiB of shared tile
#define CE_Q_WARPS (RETRO_QUANT_THREADS/WARP_SIZE)
// Embedding chunks one thread accumulates in the backward. 32*256 = 8192 is the
// n_embd ceiling ggml_backend_cuda_device_supports_op already enforces for
// FUSED_SPARSE_CE_BACK, so this array cannot be the binding constraint.
#define CE_Q_MAXC  32

// z[v0 .. v0+nrows) = dot(w[:, v], h[:, t]) for one token, block-wide. Leaves the
// per-row dot products in `sz`; every thread of the block must call it.
template<typename loader>
static __device__ __forceinline__ void ce_q_row_dots(
        const char * __restrict__ w, const float * __restrict__ h_col,
        int64_t v0, int nrows, int64_t n_embd, int64_t n_chunks, size_t nb_w,
        float * __restrict__ sw, float * __restrict__ swarp, float * __restrict__ sz) {
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    // The row loops that touch `part` are unrolled over the constant CE_Q_ROWS
    // rather than run to `nrows`: a dynamic index would put the accumulator in
    // local memory, which is the mistake the Flash Attention backward already
    // paid for once. The decode loop keeps its dynamic bound -- it indexes
    // shared memory, and eight inlined copies of a K/IQ decoder are not free.
    float part[CE_Q_ROWS] = {};
    for (int64_t c = 0; c < n_chunks; ++c) {
        __syncthreads();
        for (int r = 0; r < nrows; ++r) {
            loader::load(w + (v0 + r)*nb_w, c*RETRO_QUANT_TILE, n_embd,
                         sw + r*RETRO_QUANT_TILE);
        }
        __syncthreads();
        const float hv = h_col[c*RETRO_QUANT_TILE + tid];
#pragma unroll
        for (int r = 0; r < CE_Q_ROWS; ++r) {
            if (r < nrows) {
                // Explicit round-to-nearest, as in k_out_prod_quant: the contract
                // must not depend on -use_fast_math's FMA contraction.
                part[r] = __fadd_rn(part[r], __fmul_rn(sw[r*RETRO_QUANT_TILE + tid], hv));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < CE_Q_ROWS; ++r) {
        const float s = warp_reduce_sum(part[r]);
        if (lane == 0 && r < nrows) {
            swarp[warp*CE_Q_ROWS + r] = s;
        }
    }
    __syncthreads();
    if (tid < nrows) {
        float z = 0.0f;
        for (int wp = 0; wp < CE_Q_WARPS; ++wp) {
            z += swarp[wp*CE_Q_ROWS + tid];
        }
        sz[tid] = z;
    }
    __syncthreads();
}

template<typename loader>
static __global__ void k_fused_sparse_ce_decode(
        const float * __restrict__ h, const char * __restrict__ w,
        const int32_t * __restrict__ targets, const float * __restrict__ weights,
        const float * __restrict__ bias, float * __restrict__ maxima,
        float * __restrict__ sums, float * __restrict__ target_logits,
        const int64_t n_embd, const int64_t n_vocab, const size_t nb_w, const bool has_bias) {
    __shared__ float sw[CE_Q_ROWS*RETRO_QUANT_TILE];
    __shared__ float swarp[CE_Q_WARPS*CE_Q_ROWS];
    __shared__ float sz[CE_Q_ROWS];

    const int64_t t = blockIdx.x;
    const int32_t target = targets[t];
    // An inactive token keeps the values fused_sparse_ce_init wrote, which is
    // exactly what fused_sparse_ce_finish_loss discards for it.
    if (!(target >= 0 && target < n_vocab) || weights[t] == 0.0f) {
        return;
    }

    const float * h_col = h + t*n_embd;
    const int64_t n_chunks = n_embd/RETRO_QUANT_TILE;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float logit_target = 0.0f;

    for (int64_t v0 = 0; v0 < n_vocab; v0 += CE_Q_ROWS) {
        const int nrows = (int) min((int64_t) CE_Q_ROWS, n_vocab - v0);
        ce_q_row_dots<loader>(w, h_col, v0, nrows, n_embd, n_chunks, nb_w, sw, swarp, sz);
        if (threadIdx.x == 0) {
            for (int r = 0; r < nrows; ++r) {
                const float z = has_bias ? sz[r] + bias[v0 + r] : sz[r];
                if (z > running_max) {
                    running_sum = running_sum*expf(running_max - z) + 1.0f;
                    running_max = z;
                } else {
                    running_sum += expf(z - running_max);
                }
                if (v0 + r == target) {
                    logit_target = z;
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        maxima[t] = running_max;
        sums[t] = running_sum;
        target_logits[t] = logit_target;
    }
}

// grad_h[:, t] = coef*(sum_v softmax(z)[v] * w[:, v] - w[:, target]).
//
// The softmax denominator is not known until the whole vocabulary has been seen,
// so the accumulator is kept in the online form the forward already uses: acc
// holds sum_v exp(z_v - m) w[:, v] for the running maximum m, and is rescaled by
// exp(m_old - m_new) whenever m grows. That is one decode pass for the dot
// products and one for the accumulation -- the same two the dequantize+cuBLAS
// backward pays per tile, without the F32 round trip between them.
template<typename loader>
static __global__ void k_fused_sparse_ce_back_decode(
        const float * __restrict__ grad, const float * __restrict__ h,
        const char * __restrict__ w, const int32_t * __restrict__ targets,
        const float * __restrict__ weights, const float * __restrict__ bias,
        const int32_t * __restrict__ n_active, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_vocab, const size_t nb_w, const bool has_bias) {
    __shared__ float sw[CE_Q_ROWS*RETRO_QUANT_TILE];
    __shared__ float swarp[CE_Q_WARPS*CE_Q_ROWS];
    __shared__ float sz[CE_Q_ROWS];
    __shared__ float sp[CE_Q_ROWS];
    __shared__ float srescale;
    __shared__ float ssum;

    const int tid = threadIdx.x;
    const int64_t t = blockIdx.x;
    const int64_t n_chunks = n_embd/RETRO_QUANT_TILE;
    // dst may be h's own buffer (the fused CE's offload_h option makes the graph
    // allocator reuse it). This block reads column t of h and writes column t of
    // dst and nothing else, so the aliasing is safe per block -- the reason this
    // path needs no staging buffer, unlike the tiled one above.
    float * dst_col = dst + t*n_embd;

    const int32_t target = targets[t];
    const float weight = weights[t];
    if (!(target >= 0 && target < n_vocab) || weight == 0.0f || *n_active <= 0) {
        for (int64_t c = 0; c < n_chunks; ++c) {
            dst_col[c*RETRO_QUANT_TILE + tid] = 0.0f;
        }
        return;
    }

    const float * h_col = h + t*n_embd;
    float acc[CE_Q_MAXC] = {};
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int64_t v0 = 0; v0 < n_vocab; v0 += CE_Q_ROWS) {
        const int nrows = (int) min((int64_t) CE_Q_ROWS, n_vocab - v0);
        ce_q_row_dots<loader>(w, h_col, v0, nrows, n_embd, n_chunks, nb_w, sw, swarp, sz);
        if (tid == 0) {
            float new_max = running_max;
            for (int r = 0; r < nrows; ++r) {
                const float z = has_bias ? sz[r] + bias[v0 + r] : sz[r];
                sz[r] = z;
                new_max = fmaxf(new_max, z);
            }
            const float rescale = expf(running_max - new_max);
            running_sum *= rescale;
            for (int r = 0; r < nrows; ++r) {
                const float p = expf(sz[r] - new_max);
                sp[r] = p;
                running_sum += p;
            }
            running_max = new_max;
            srescale = rescale;
        }
        __syncthreads();
        const float rescale = srescale;

        for (int64_t c = 0; c < n_chunks; ++c) {
            __syncthreads();
            for (int r = 0; r < nrows; ++r) {
                loader::load(w + (v0 + r)*nb_w, c*RETRO_QUANT_TILE, n_embd,
                             sw + r*RETRO_QUANT_TILE);
            }
            __syncthreads();
            float a = 0.0f;
            for (int r = 0; r < nrows; ++r) {
                a = __fadd_rn(a, __fmul_rn(sp[r], sw[r*RETRO_QUANT_TILE + tid]));
            }
            // acc is indexed by the chunk, so it lives in local memory: 128 bytes
            // per thread at the n_embd ceiling, touched once per vocabulary group.
            // Keeping it in registers would mean unrolling a decode loop 32 times.
            acc[c] = __fadd_rn(acc[c]*rescale, a);
        }
    }

    if (tid == 0) {
        ssum = running_sum;
    }
    __syncthreads();
    const float coef = grad[0]*weight/(float) *n_active;
    const float inv_sum = 1.0f/ssum;
    for (int64_t c = 0; c < n_chunks; ++c) {
        __syncthreads();
        loader::load(w + (int64_t) target*nb_w, c*RETRO_QUANT_TILE, n_embd, sw);
        __syncthreads();
        dst_col[c*RETRO_QUANT_TILE + tid] = coef*(acc[c]*inv_sum - sw[tid]);
    }
}

// The head must be one of the types the shared loader table covers -- a superset
// of what Metal and Vulkan decode, but not of what ggml_cuda_can_decode_frozen
// accepts, which is every type with a to_fp32 kernel. A type outside it, an F32
// head (cheaper without any decoding) or an embedding that is not a whole number
// of 256-element tiles falls back to the default path.
template<ggml_type type>
static bool fused_sparse_ce_decode_fits(int64_t n_embd) {
    using traits = retro_quant_traits<type>;
    return n_embd % RETRO_QUANT_TILE == 0 && RETRO_QUANT_TILE % traits::alignment == 0;
}

static bool fused_sparse_ce_decode_enabled() {
    const char * env = getenv("GGML_CUDA_CE_QHEAD");
    return env != nullptr && std::atoi(env) != 0;
}

template<ggml_type type>
static bool launch_fused_sparse_ce_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * h = dst->src[0], * w = dst->src[1];
    const ggml_tensor * targets = dst->src[2], * weights = dst->src[3];
    const ggml_tensor * bias = dst->src[4];
    const int64_t n_embd = h->ne[0], n_tokens = h->ne[1], n_vocab = w->ne[1];
    if (!fused_sparse_ce_decode_fits<type>(n_embd)) {
        return false;
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    fused_sparse_ce_work work(pool, n_tokens);
    fused_sparse_ce_prepare(stream, targets, weights, n_vocab, work);
    k_fused_sparse_ce_decode<typename retro_quant_traits<type>::loader>
        <<<(unsigned) n_tokens, RETRO_QUANT_THREADS, 0, stream>>>(
            (const float *) h->data, (const char *) w->data,
            (const int32_t *) targets->data, (const float *) weights->data,
            bias ? (const float *) bias->data : nullptr,
            work.maxima.get(), work.sums.get(), work.target_logits.get(),
            n_embd, n_vocab, w->nb[1], bias != nullptr);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<float> losses(pool, n_tokens);
    const int threads = 256;
    fused_sparse_ce_finish_loss<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        (const int32_t *) targets->data, (const float *) weights->data, work.n_active.get(),
        work.maxima.get(), work.sums.get(), work.target_logits.get(), losses.get(), n_tokens, n_vocab);
    CUDA_CHECK(cudaGetLastError());
    sum_f32_cuda(pool, losses.get(), (float *) dst->data, n_tokens, stream);
    return true;
}

template<ggml_type type>
static bool launch_fused_sparse_ce_back_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad = dst->src[0], * h = dst->src[1], * w = dst->src[2];
    const ggml_tensor * targets = dst->src[3], * weights = dst->src[4];
    const ggml_tensor * bias = dst->src[5];
    const int64_t n_embd = h->ne[0], n_tokens = h->ne[1], n_vocab = w->ne[1];
    if (!fused_sparse_ce_decode_fits<type>(n_embd) || n_embd > CE_Q_MAXC*RETRO_QUANT_TILE) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    fused_sparse_ce_work work(ctx.pool(), n_tokens);
    fused_sparse_ce_prepare(stream, targets, weights, n_vocab, work);
    k_fused_sparse_ce_back_decode<typename retro_quant_traits<type>::loader>
        <<<(unsigned) n_tokens, RETRO_QUANT_THREADS, 0, stream>>>(
            (const float *) grad->data, (const float *) h->data, (const char *) w->data,
            (const int32_t *) targets->data, (const float *) weights->data,
            bias ? (const float *) bias->data : nullptr, work.n_active.get(),
            (float *) dst->data, n_embd, n_vocab, w->nb[1], bias != nullptr);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Returns false when the default dequantize+cuBLAS path has to run. Also the one
// place that publishes which regime executed: O5's numbers are only comparable
// between two runs that decoded the head the same way, so a measurement must be
// able to say so without reading this file (same rule as O3's gemm= line).
static bool fused_sparse_ce_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool back) {
    const ggml_tensor * h = dst->src[back ? 1 : 0];
    const ggml_tensor * w = dst->src[back ? 2 : 1];
    bool decoded = false;
    if (fused_sparse_ce_decode_enabled() && w->type != GGML_TYPE_F32) {
        GGML_ASSERT(w->nb[0] == ggml_type_size(w->type));
        switch (w->type) {
#define FSCE_DECODE_CASE(TYPE, BLK, NL, NAME, VKNAME)                              \
            case TYPE:                                                             \
                decoded = back ? launch_fused_sparse_ce_back_decode<TYPE>(ctx, dst) \
                               : launch_fused_sparse_ce_decode<TYPE>(ctx, dst);     \
                break;
            GGML_RETRO_OUT_PROD_TYPES(FSCE_DECODE_CASE)
#undef FSCE_DECODE_CASE
            default:
                break;
        }
    }
    if (getenv("GGML_CUDA_CE_DEBUG")) {
        fprintf(stderr, "fused_sparse_ce%s: n_embd=%lld n_vocab=%lld n_tokens=%lld w=%s head=%s\n",
                back ? "_back" : "", (long long) h->ne[0], (long long) w->ne[1],
                (long long) h->ne[1], ggml_type_name(w->type),
                decoded ? "decode" : "dequant+sgemm");
    }
    return decoded;
}

void ggml_cuda_fused_sparse_ce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * h = dst->src[0], * w = dst->src[1];
    const ggml_tensor * targets = dst->src[2], * weights = dst->src[3];
    const ggml_tensor * bias = dst->src[4];
    const int64_t n_embd = h->ne[0], n_tokens = h->ne[1], n_vocab = w->ne[1];
    GGML_ASSERT(h->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(targets->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(h) && ggml_is_contiguous(w));
    GGML_ASSERT(!bias || (bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias)));

    if (fused_sparse_ce_decode(ctx, dst, false)) {
        return;
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const float * bias_f32 = bias ? (const float *) bias->data : nullptr;
    fused_sparse_ce_work work(pool, n_tokens);
    fused_sparse_ce_prepare(stream, targets, weights, n_vocab, work);
    const int64_t tile  = fused_sparse_ce_tile_size(dst, n_vocab);
    const int64_t chunk = fused_sparse_ce_seq_chunk(dst, n_tokens);
    fused_sparse_ce_head head(pool, w, n_embd, tile);
    ggml_cuda_pool_alloc<float> logits(pool, tile*chunk);
    for (int64_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        const int64_t nt = std::min(chunk, n_tokens - t0);
        fused_sparse_ce_lse(ctx, (const float *) h->data, head, targets, bias_f32,
            n_embd, t0, nt, n_vocab, tile, logits.get(), work);
    }

    ggml_cuda_pool_alloc<float> losses(pool, n_tokens);
    const int threads = 256;
    fused_sparse_ce_finish_loss<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        (const int32_t *) targets->data, (const float *) weights->data, work.n_active.get(),
        work.maxima.get(), work.sums.get(), work.target_logits.get(), losses.get(), n_tokens, n_vocab);
    CUDA_CHECK(cudaGetLastError());
    sum_f32_cuda(pool, losses.get(), (float *) dst->data, n_tokens, stream);
}

void ggml_cuda_fused_sparse_ce_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad = dst->src[0], * h = dst->src[1], * w = dst->src[2];
    const ggml_tensor * targets = dst->src[3], * weights = dst->src[4];
    const ggml_tensor * bias = dst->src[5];
    const int64_t n_embd = h->ne[0], n_tokens = h->ne[1], n_vocab = w->ne[1];
    GGML_ASSERT(grad->type == GGML_TYPE_F32 && h->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(h) && ggml_is_contiguous(w) && ggml_is_contiguous(dst));
    GGML_ASSERT(!bias || (bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias)));

    if (fused_sparse_ce_decode(ctx, dst, true)) {
        return;
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const float * bias_f32 = bias ? (const float *) bias->data : nullptr;
    fused_sparse_ce_work work(pool, n_tokens);
    fused_sparse_ce_prepare(stream, targets, weights, n_vocab, work);
    const int64_t tile  = fused_sparse_ce_tile_size(dst, n_vocab);
    const int64_t chunk = fused_sparse_ce_seq_chunk(dst, n_tokens);
    fused_sparse_ce_head head(pool, w, n_embd, tile);
    ggml_cuda_pool_alloc<float> logits(pool, tile*chunk);

    // A quantized head is only ever materialized one vocab tile at a time, so the
    // rows the target subtraction needs are captured while their tile is live.
    const bool gather_targets = w->type != GGML_TYPE_F32;
    ggml_cuda_pool_alloc<float> target_rows(pool);
    if (gather_targets) {
        target_rows.alloc(n_embd*chunk);
    }

    // retro delta (plan rl/OPTIMIZE feature 3): with offload_h the graph allocator
    // gives grad_h the buffer of `h`, so writing a column would destroy the hidden
    // state the very next vocab tile still has to read. Accumulate the chunk into a
    // [n_embd, chunk] staging buffer instead and copy it back once the chunk is
    // done; chunks own disjoint columns, so the eviction is exact. The staging
    // buffer only bounds the peak when seq_chunk > 0 -- that is why the flag is
    // gated on it upstream.
    const bool inplace = dst->data == h->data;
    ggml_cuda_pool_alloc<float> grad_stage(pool);
    if (inplace) {
        grad_stage.alloc(n_embd*chunk);
    }

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;
    const int threads = 256;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        const int64_t nt = std::min(chunk, n_tokens - t0);
        float * out = inplace ? grad_stage.get() : (float *) dst->data + t0*n_embd;
        if (gather_targets) {
            // A masked token (target < 0) is captured by no tile, and CUDA scratch
            // -- unlike host memory -- does not come zeroed.
            CUDA_CHECK(cudaMemsetAsync(target_rows.get(), 0, n_embd*nt*sizeof(float), stream));
        }
        // Recompute this chunk's online log-sum-exp (checkpointing over the
        // sequence axis), then accumulate grad_h tile by tile.
        fused_sparse_ce_lse(ctx, (const float *) h->data, head, targets, bias_f32,
            n_embd, t0, nt, n_vocab, tile, logits.get(), work);
        bool first = true;
        for (int64_t v0 = 0; v0 < n_vocab; v0 += tile) {
            const int64_t nv = std::min(tile, n_vocab - v0);
            const float * w_tile = head.rows(v0, nv, stream);
            if (gather_targets) {
                const int64_t n_rows = n_embd*nt;
                fused_sparse_ce_capture_target_rows<<<(n_rows + threads - 1)/threads, threads, 0, stream>>>(
                    w_tile, (const int32_t *) targets->data, target_rows.get(),
                    n_embd, t0, nt, v0, nv);
                CUDA_CHECK(cudaGetLastError());
            }
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nv, nt, n_embd, &alpha, w_tile, n_embd,
                (const float *) h->data + t0*n_embd, n_embd, &beta0, logits.get(), tile));
            if (bias_f32) {
                const int64_t n_bias = nv*nt;
                fused_sparse_ce_add_bias<<<(n_bias + threads - 1)/threads, threads, 0, stream>>>(
                    logits.get(), bias_f32, v0, nv, tile, nt);
                CUDA_CHECK(cudaGetLastError());
            }
            const int64_t n_probs = nv*nt;
            fused_sparse_ce_make_probs<<<(n_probs + threads - 1)/threads, threads, 0, stream>>>(
                logits.get(), (const float *) grad->data, (const int32_t *) targets->data,
                (const float *) weights->data, work.n_active.get(), work.maxima.get(), work.sums.get(),
                nv, tile, t0, nt, n_vocab);
            CUDA_CHECK(cudaGetLastError());
            const float * beta = first ? &beta0 : &beta1;
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n_embd, nt, nv, &alpha, w_tile, n_embd,
                logits.get(), tile, beta, out, n_embd));
            first = false;
        }
        // grad_h for this chunk is fully accumulated over the vocabulary, so the
        // target row can be subtracted now -- on `out`, before the in-place
        // eviction below, and while the chunk's captured rows are still valid.
        const int64_t n_sub = n_embd*nt;
        const int64_t sub_blocks = (n_sub + threads - 1)/threads;
        if (gather_targets) {
            fused_sparse_ce_subtract_target<true><<<sub_blocks, threads, 0, stream>>>(
                (const float *) grad->data, target_rows.get(), (const int32_t *) targets->data,
                (const float *) weights->data, work.n_active.get(), out,
                n_embd, t0, nt, n_vocab);
        } else {
            fused_sparse_ce_subtract_target<false><<<sub_blocks, threads, 0, stream>>>(
                (const float *) grad->data, (const float *) w->data, (const int32_t *) targets->data,
                (const float *) weights->data, work.n_active.get(), out,
                n_embd, t0, nt, n_vocab);
        }
        CUDA_CHECK(cudaGetLastError());
        if (inplace) {
            CUDA_CHECK(cudaMemcpyAsync((float *) dst->data + t0*n_embd, out,
                n_embd*nt*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }
}
