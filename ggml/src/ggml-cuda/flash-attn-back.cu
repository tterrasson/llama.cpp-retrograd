#include "flash-attn-back.cuh"

#include <cstdint>
#include <cstdlib>

#include <algorithm>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(TURING_MMA_AVAILABLE)
#include <mma.h>
namespace wmma = nvcuda::wmma;
#endif

// retro delta: streaming Flash Attention backward. The query kernel runs one
// warp per (batch, head, query), reuses O from the forward pass for
// D = dot(dO, O), and only recomputes the row LSE before accumulating dQ. The KV
// kernel runs one warp per (batch, KV head, gradient-window row) and reduces all
// contributing query rows in a fixed order before writing dK/dV once. This
// removes the redundant forward-output reconstruction (OPTIMS_V4 F1) and the
// contended, non-deterministic atomics (F3).
//
// dK/dV only cover the gradient window (the KV rows written at this step) when
// the caller declares one. dQ still reads the whole cache -- every key
// contributes to it -- which is why the gradient passes are split in two.
//
// The head dimension is split across the 32 lanes (lane `l` owns d = l, l+32,
// l+64, ...), which is what makes this affordable:
//   - the per-row accumulators are 2*REGS registers per lane instead of
//     o_acc[D] + dq[D] = 1 KiB per thread, which spilled to local memory;
//   - the q.k and dO.v dot products become warp shuffle reductions instead of
//     D-iteration serial loops;
//   - the dK/dV stores are issued 32-wide to consecutive addresses (coalesced)
//     after a deterministic reduction instead of serialized atomics.
//
// An earlier revision ran one *thread* per query row. On a 0.6B model at
// nq=256/nkv=1024 that is ~5e8 serialized atomicAdds per launch on top of the
// spill traffic: 25 ms per launch, 73% of training wall-clock.
//
// REGS = ceil(max(hsk,hsv)/32) is a template parameter rather than a single
// compile-time cap: the warp split means the head dimension only costs 4*REGS
// registers per lane (qv, dov, o_acc, dq), so wide-head models (Qwen3.5/GDN at
// head_dim=256, Gemma-4's global-attention layers at 512, MLA at 576) fit fine
// — but instantiating everything at the widest bucket would make the common
// head_dim<=128 models pay the extra registers and occupancy for nothing.
// Dispatch picks the smallest bucket that covers the shape. Note that 4*REGS is
// the real number: 64 registers of accumulator at the 512 bucket, which costs
// occupancy but stays far from the 255-register/thread ceiling.

// Head-dimension elements owned by each lane, per bucket. Adding a bucket for a
// wider head dimension (MLA at 576 would be REGS 18) is a two-line change here
// plus a matching FA_BACK_MAX_D bump — but do it with a parity test at that
// head dimension, never on the strength of "it's the same code".
#define FA_BACK_REGS_SMALL  4   // head dim <= 128 (most dense models)
#define FA_BACK_REGS_WIDE   8   // head dim <= 256 (Qwen3.5/GDN, Gemma2)
#define FA_BACK_REGS_XWIDE 16   // head dim <= 512 (Gemma-4 global-attn layers)

// The supports check in ggml-cuda.cu gates on FA_BACK_MAX_D, so the widest
// bucket must cover exactly that or shapes get accepted and then assert.
static_assert(FA_BACK_REGS_XWIDE*WARP_SIZE == FA_BACK_MAX_D,
              "widest kernel bucket must match the advertised head-dim cap");
static_assert(FA_BACK_MAX_D <= GGML_FLASH_ATTN_BACK_MAX_HEAD_DIM,
              "advertised head-dim cap exceeds what the probe harness can exercise");

static __device__ __forceinline__ float fab_load(const char * base, int64_t idx,
                                                  int64_t stride0, bool is_f16) {
    if (is_f16) {
        return __half2float(*(const half *) (base + idx*stride0));
    }
    return *(const float *) (base + idx*stride0);
}

// Maps a gradient-window row to its row in the KV cache view. `kv_idxs` holds
// the same global indices the forward ggml_set_rows() used, so the stream plane
// has to be subtracted back out (see ggml_flash_attn_ext_set_grad_window).
struct fa_back_window {
    const int32_t * idxs;   // [nwin, nbatch], stream-major; null when dense
    int64_t         nwin;   // rows carrying a gradient, per stream
    int64_t         stride; // KV rows per stream in the underlying cache
    int64_t         stream0;
};

static __device__ __forceinline__ int64_t fab_window_key(
        const fa_back_window & w, int64_t ib, int64_t j) {
    if (!w.idxs) {
        return j;
    }
    return (int64_t) w.idxs[ib*w.nwin + j] - w.stride*(w.stream0 + ib);
}

// F2 fast path. One block owns a 16x16 (query x KV) tile. Q and dO are rounded
// to F16 while being staged next to the native F16 K/V cache; both score
// products use tensor cores and accumulate in F32. The nonlinear attention
// algebra and the gradient accumulators remain F32. The scalar kernels below
// remain the reference/fallback for F32 caches, unusual head shapes, non-NVIDIA
// devices, and when GGML_CUDA_FA_BACK_MMA=0.
#define FA_BACK_MMA_TILE 16
#define FA_BACK_MMA_MAX_D 256
#define FA_BACK_MMA_THREADS 256
#define FA_BACK_MMA_OWNED ((FA_BACK_MMA_TILE*FA_BACK_MMA_MAX_D)/FA_BACK_MMA_THREADS)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(TURING_MMA_AVAILABLE)

static __device__ __forceinline__ void fab_mma_scores(
        const half * a, const half * b_rows, float * scores, int d) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int id = 0; id < d; id += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
        wmma::load_matrix_sync(af, a + id, d);
        // b_rows is [16, d] row-major, hence B = b_rows^T is column-major.
        wmma::load_matrix_sync(bf, b_rows + id, d);
        wmma::mma_sync(c, af, bf, c);
    }
    wmma::store_matrix_sync(scores, c, 16, wmma::mem_row_major);
}

static __device__ __forceinline__ float fab_mma_score(
        float dot, const half * mask, int64_t key, size_t nbm0,
        bool has_mask, float scale, float softcap) {
    float score = softcap != 0.0f ? softcap*tanhf(dot*scale/softcap) : dot*scale;
    if (has_mask) {
        score += __half2float(*(const half *) ((const char *) mask + key*nbm0));
    }
    return score;
}

static __device__ __forceinline__ float fab_mma_deriv(float dot, float scale, float softcap) {
    if (softcap == 0.0f) {
        return scale;
    }
    const float t = tanhf(dot*scale/softcap);
    return scale*(1.0f - t*t);
}

template<int>
static __global__ void flash_attn_back_mma_q_kernel(
        const char * __restrict__ q, const half * __restrict__ k,
        const half * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ out, const char * __restrict__ dO,
        float * __restrict__ dst, const fa_back_mma_args a) {
    const int64_t d = a.d, nq = a.nq, nkv = a.nkv, nhead = a.nhead, ratio = a.ratio;
    const size_t nbq0 = a.nbq0, nbq1 = a.nbq1, nbq2 = a.nbq2, nbq3 = a.nbq3;
    const size_t nbk1 = a.nbk1, nbk2 = a.nbk2, nbk3 = a.nbk3;
    const size_t nbv1 = a.nbv1, nbv2 = a.nbv2, nbv3 = a.nbv3;
    const size_t nbm0 = a.nbm0, nbm1 = a.nbm1, nbm3 = a.nbm3;
    const size_t nbo0 = a.nbo0, nbo1 = a.nbo1, nbo2 = a.nbo2, nbo3 = a.nbo3;
    const size_t nbd0 = a.nbd0, nbd1 = a.nbd1, nbd2 = a.nbd2, nbd3 = a.nbd3;
    const size_t off_s = a.off_s;
    const bool has_mask = a.has_mask, want_dq = a.want_dq;
    const float scale = a.scale, softcap = a.softcap;
    const int tid = threadIdx.x;
    const int64_t q0 = int64_t(blockIdx.x)*FA_BACK_MMA_TILE;
    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;
    const int64_t ikh = ih/ratio;

    extern __shared__ char raw[];
    half * sq = (half *) raw;
    half * sd = sq + FA_BACK_MMA_TILE*d;
    half * sk = sd + FA_BACK_MMA_TILE*d;
    half * sv = sk + FA_BACK_MMA_TILE*d;
    float * qk = (float *) (sv + FA_BACK_MMA_TILE*d);
    float * dov = qk + FA_BACK_MMA_TILE*FA_BACK_MMA_TILE;
    float * row_m = dov + FA_BACK_MMA_TILE*FA_BACK_MMA_TILE;
    float * row_l = row_m + FA_BACK_MMA_TILE;
    float * row_delta = row_l + FA_BACK_MMA_TILE;

    for (int idx = tid; idx < FA_BACK_MMA_TILE*d; idx += blockDim.x) {
        const int qr = idx/d;
        const int id = idx - qr*d;
        const int64_t iq = q0 + qr;
        if (iq < nq) {
            const char * qr_ptr = q + iq*nbq1 + ih*nbq2 + ib*nbq3;
            const char * do_ptr = dO + ih*nbd1 + iq*nbd2 + ib*nbd3;
            sq[idx] = __float2half(*(const float *) (qr_ptr + id*nbq0));
            sd[idx] = __float2half(*(const float *) (do_ptr + id*nbd0));
        } else {
            sq[idx] = __float2half(0.0f);
            sd[idx] = __float2half(0.0f);
        }
    }
    if (tid < FA_BACK_MMA_TILE) {
        const int64_t iq = q0 + tid;
        float delta = 0.0f;
        if (iq < nq) {
            const char * out_row = out + ih*nbo1 + iq*nbo2 + ib*nbo3;
            const char * do_row = dO + ih*nbd1 + iq*nbd2 + ib*nbd3;
            for (int id = 0; id < d; ++id) {
                delta += *(const float *) (out_row + id*nbo0) *
                         *(const float *) (do_row + id*nbd0);
            }
        }
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
        row_delta[tid] = delta;
    }
    __syncthreads();

    // Pass 1: tensor-core Q.K^T and online LSE.
    for (int64_t kv0 = 0; kv0 < nkv; kv0 += FA_BACK_MMA_TILE) {
        for (int idx = tid; idx < FA_BACK_MMA_TILE*d; idx += blockDim.x) {
            const int kr = idx/d;
            const int id = idx - kr*d;
            const int64_t ik = kv0 + kr;
            sk[idx] = ik < nkv ? *(const half *) ((const char *) k + ik*nbk1 + ikh*nbk2 + ib*nbk3 + id*sizeof(half))
                               : __float2half(0.0f);
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            fab_mma_scores(sq, sk, qk, (int) d);
        }
        __syncthreads();
        if (tid < FA_BACK_MMA_TILE && q0 + tid < nq) {
            float m = row_m[tid];
            float l = row_l[tid];
            const half * mask_row = has_mask ? (const half *) (mask + (q0 + tid)*nbm1 + ib*nbm3) : nullptr;
            const int n = (int) min((int64_t) FA_BACK_MMA_TILE, nkv - kv0);
            for (int kr = 0; kr < n; ++kr) {
                const float score = fab_mma_score(qk[tid*FA_BACK_MMA_TILE + kr], mask_row,
                                                  kv0 + kr, nbm0, has_mask, scale, softcap);
                if (score == -INFINITY) {
                    continue;
                }
                const float mn = fmaxf(m, score);
                l = l*expf(m - mn) + expf(score - mn);
                m = mn;
            }
            row_m[tid] = m;
            row_l[tid] = l;
        }
        __syncthreads();
    }

    if (tid < FA_BACK_MMA_TILE && q0 + tid < nq) {
        const int64_t stat = (ib*nhead + ih)*nq + q0 + tid;
        const int64_t nstats = gridDim.z*nhead*nq;
        float * stats = (float *) ((char *) dst + off_s);
        // A fully masked row gets +INFINITY, not -INFINITY, and the scalar
        // kernel does the same: the KV kernel reads this back as
        // exp(score - lse), so the sentinel has to drive that to zero for *any*
        // score it might see. -INFINITY would drive it to +inf, i.e. NaN in
        // dK/dV, the moment the two kernels ever disagreed about which entries
        // are masked.
        stats[stat] = row_l[tid] > 0.0f ? row_m[tid] + logf(row_l[tid]) : INFINITY;
        stats[nstats + stat] = row_delta[tid];
    }
    if (!want_dq) {
        return;
    }

    float dq[FA_BACK_MMA_OWNED];
#pragma unroll
    for (int i = 0; i < FA_BACK_MMA_OWNED; ++i) { dq[i] = 0.0f; }

    // Pass 2: both tensor-core products, followed by F32 softmax algebra and dQ.
    for (int64_t kv0 = 0; kv0 < nkv; kv0 += FA_BACK_MMA_TILE) {
        for (int idx = tid; idx < FA_BACK_MMA_TILE*d; idx += blockDim.x) {
            const int kr = idx/d;
            const int id = idx - kr*d;
            const int64_t ik = kv0 + kr;
            if (ik < nkv) {
                sk[idx] = *(const half *) ((const char *) k + ik*nbk1 + ikh*nbk2 + ib*nbk3 + id*sizeof(half));
                sv[idx] = *(const half *) ((const char *) v + ik*nbv1 + ikh*nbv2 + ib*nbv3 + id*sizeof(half));
            } else {
                sk[idx] = __float2half(0.0f);
                sv[idx] = __float2half(0.0f);
            }
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            fab_mma_scores(sq, sk, qk, (int) d);
            fab_mma_scores(sd, sv, dov, (int) d);
        }
        __syncthreads();
        const int owned = (FA_BACK_MMA_TILE*(int) d + blockDim.x - 1)/blockDim.x;
        const int n = (int) min((int64_t) FA_BACK_MMA_TILE, nkv - kv0);
        for (int oi = 0; oi < owned; ++oi) {
            const int idx = tid + oi*blockDim.x;
            if (idx >= FA_BACK_MMA_TILE*d) { continue; }
            const int qr = idx/d;
            const int id = idx - qr*d;
            if (q0 + qr >= nq || row_l[qr] <= 0.0f) { continue; }
            const half * mask_row = has_mask ? (const half *) (mask + (q0 + qr)*nbm1 + ib*nbm3) : nullptr;
            for (int kr = 0; kr < n; ++kr) {
                const float dot = qk[qr*FA_BACK_MMA_TILE + kr];
                const float score = fab_mma_score(dot, mask_row, kv0 + kr, nbm0, has_mask, scale, softcap);
                if (score == -INFINITY) { continue; }
                const float p = expf(score - row_m[qr])/row_l[qr];
                const float ds0 = p*(dov[qr*FA_BACK_MMA_TILE + kr] - row_delta[qr]) *
                                  fab_mma_deriv(dot, scale, softcap);
                dq[oi] += ds0*__half2float(sk[kr*d + id]);
            }
        }
        __syncthreads();
    }
    const int owned = (FA_BACK_MMA_TILE*(int) d + blockDim.x - 1)/blockDim.x;
    for (int oi = 0; oi < owned; ++oi) {
        const int idx = tid + oi*blockDim.x;
        if (idx >= FA_BACK_MMA_TILE*d) { continue; }
        const int qr = idx/d;
        const int id = idx - qr*d;
        const int64_t iq = q0 + qr;
        if (iq < nq) {
            dst[((ib*nhead + ih)*nq + iq)*d + id] = dq[oi];
        }
    }
}

template<int>
static __global__ void flash_attn_back_mma_kv_kernel(
        const char * __restrict__ q, const half * __restrict__ k,
        const half * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ dO, float * __restrict__ dst, const fa_back_mma_args a) {
    const int64_t d = a.d, nq = a.nq, nkv = a.nkv, nhead = a.nhead, nheadk = a.nheadk, ratio = a.ratio;
    const size_t nbq0 = a.nbq0, nbq1 = a.nbq1, nbq2 = a.nbq2, nbq3 = a.nbq3;
    const size_t nbk1 = a.nbk1, nbk2 = a.nbk2, nbk3 = a.nbk3;
    const size_t nbv1 = a.nbv1, nbv2 = a.nbv2, nbv3 = a.nbv3;
    const size_t nbm0 = a.nbm0, nbm1 = a.nbm1, nbm3 = a.nbm3;
    const size_t nbd0 = a.nbd0, nbd1 = a.nbd1, nbd2 = a.nbd2, nbd3 = a.nbd3;
    const size_t off_k = a.off_k, off_v = a.off_v, off_s = a.off_s;
    const bool has_mask = a.has_mask, want_dk = a.want_dk, want_dv = a.want_dv;
    const fa_back_window window = {a.window_idxs, a.window_nwin, a.window_stride, a.window_stream0};
    const float scale = a.scale, softcap = a.softcap;
    const int tid = threadIdx.x;
    const int64_t j0 = int64_t(blockIdx.x)*FA_BACK_MMA_TILE;
    const int64_t ikh = blockIdx.y;
    const int64_t ib = blockIdx.z;

    extern __shared__ char raw[];
    half * sq = (half *) raw;
    half * sd = sq + FA_BACK_MMA_TILE*d;
    half * sk = sd + FA_BACK_MMA_TILE*d;
    half * sv = sk + FA_BACK_MMA_TILE*d;
    float * qk = (float *) (sv + FA_BACK_MMA_TILE*d);
    float * dov = qk + FA_BACK_MMA_TILE*FA_BACK_MMA_TILE;

    for (int idx = tid; idx < FA_BACK_MMA_TILE*d; idx += blockDim.x) {
        const int kr = idx/d;
        const int id = idx - kr*d;
        const int64_t j = j0 + kr;
        const int64_t ik = j < window.nwin ? fab_window_key(window, ib, j) : -1;
        if (ik >= 0 && ik < nkv) {
            sk[idx] = *(const half *) ((const char *) k + ik*nbk1 + ikh*nbk2 + ib*nbk3 + id*sizeof(half));
            sv[idx] = *(const half *) ((const char *) v + ik*nbv1 + ikh*nbv2 + ib*nbv3 + id*sizeof(half));
        } else {
            sk[idx] = __float2half(0.0f);
            sv[idx] = __float2half(0.0f);
        }
    }
    __syncthreads();

    float dk[FA_BACK_MMA_OWNED];
    float dv[FA_BACK_MMA_OWNED];
#pragma unroll
    for (int i = 0; i < FA_BACK_MMA_OWNED; ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }

    const float * stats = (const float *) ((const char *) dst + off_s);
    const int64_t nstats = gridDim.z*nhead*nq;
    for (int64_t rhead = 0; rhead < ratio; ++rhead) {
        const int64_t ih = ikh*ratio + rhead;
        for (int64_t q0 = 0; q0 < nq; q0 += FA_BACK_MMA_TILE) {
            for (int idx = tid; idx < FA_BACK_MMA_TILE*d; idx += blockDim.x) {
                const int qr = idx/d;
                const int id = idx - qr*d;
                const int64_t iq = q0 + qr;
                if (iq < nq) {
                    const char * qr_ptr = q + iq*nbq1 + ih*nbq2 + ib*nbq3;
                    const char * do_ptr = dO + ih*nbd1 + iq*nbd2 + ib*nbd3;
                    sq[idx] = __float2half(*(const float *) (qr_ptr + id*nbq0));
                    sd[idx] = __float2half(*(const float *) (do_ptr + id*nbd0));
                } else {
                    sq[idx] = __float2half(0.0f);
                    sd[idx] = __float2half(0.0f);
                }
            }
            __syncthreads();
            if (tid < WARP_SIZE) {
                fab_mma_scores(sq, sk, qk, (int) d);
                fab_mma_scores(sd, sv, dov, (int) d);
            }
            __syncthreads();
            const int owned = (FA_BACK_MMA_TILE*(int) d + blockDim.x - 1)/blockDim.x;
            for (int oi = 0; oi < owned; ++oi) {
                const int idx = tid + oi*blockDim.x;
                if (idx >= FA_BACK_MMA_TILE*d) { continue; }
                const int kr = idx/d;
                const int id = idx - kr*d;
                const int64_t j = j0 + kr;
                if (j >= window.nwin) { continue; }
                const int64_t ik = fab_window_key(window, ib, j);
                if (ik < 0 || ik >= nkv) { continue; }
                for (int qr = 0; qr < FA_BACK_MMA_TILE && q0 + qr < nq; ++qr) {
                    const int64_t iq = q0 + qr;
                    const half * mask_row = has_mask ? (const half *) (mask + iq*nbm1 + ib*nbm3) : nullptr;
                    const float dot = qk[qr*FA_BACK_MMA_TILE + kr];
                    const float score = fab_mma_score(dot, mask_row, ik, nbm0, has_mask, scale, softcap);
                    const int64_t stat = (ib*nhead + ih)*nq + iq;
                    const float p = score == -INFINITY ? 0.0f : expf(score - stats[stat]);
                    const float ds0 = p*(dov[qr*FA_BACK_MMA_TILE + kr] - stats[nstats + stat]) *
                                      fab_mma_deriv(dot, scale, softcap);
                    dk[oi] += ds0*__half2float(sq[qr*d + id]);
                    dv[oi] += p*__half2float(sd[qr*d + id]);
                }
            }
            __syncthreads();
        }
    }

    float * dK = (float *) ((char *) dst + off_k);
    float * dV = (float *) ((char *) dst + off_v);
    const int owned = (FA_BACK_MMA_TILE*(int) d + blockDim.x - 1)/blockDim.x;
    for (int oi = 0; oi < owned; ++oi) {
        const int idx = tid + oi*blockDim.x;
        if (idx >= FA_BACK_MMA_TILE*d) { continue; }
        const int kr = idx/d;
        const int id = idx - kr*d;
        const int64_t j = j0 + kr;
        if (j < window.nwin) {
            const int64_t base = ((ib*nheadk + ikh)*window.nwin + j)*d + id;
            if (want_dk) { dK[base] = dk[oi]; }
            if (want_dv) { dV[base] = dv[oi]; }
        }
    }
}

#elif !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Host and pre-Turing compilation passes still need declarations for nvcc's
// generated launch stubs. These bodies are unreachable because dispatch also
// checks turing_mma_available().
template<int>
static __global__ void flash_attn_back_mma_q_kernel(
        const char * __restrict__ q, const half * __restrict__ k,
        const half * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ out, const char * __restrict__ dO,
        float * __restrict__ dst, const fa_back_mma_args a) {
    GGML_UNUSED_VARS(q, k, v, mask, out, dO, dst, a);
    NO_DEVICE_CODE;
}

template<int>
static __global__ void flash_attn_back_mma_kv_kernel(
        const char * __restrict__ q, const half * __restrict__ k,
        const half * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ dO, float * __restrict__ dst, const fa_back_mma_args a) {
    GGML_UNUSED_VARS(q, k, v, mask, dO, dst, a);
    NO_DEVICE_CODE;
}

#endif // NVIDIA WMMA

template<int REGS>
static __global__ void flash_attn_back_kernel(
        const char * __restrict__ q, const char * __restrict__ k,
        const char * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ out, const char * __restrict__ dO,
        float * __restrict__ dst,
        const int64_t hsk, const int64_t hsv, const int64_t nq, const int64_t nkv,
        const int64_t nhead, const int64_t nheadk, const int64_t nbatch,
        const int64_t ratio,
        // element strides (bytes) for each operand's non-fastest dims
        const int64_t nbq0, const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk0, const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv0, const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t nbm0, const int64_t nbm1, const int64_t nbm3,
        const int64_t nbo0, const int64_t nbo1, const int64_t nbo2, const int64_t nbo3,
        const int64_t nbd0, const int64_t nbd1, const int64_t nbd2, const int64_t nbd3,
        const size_t off_s,
        const bool kv_f16, const bool has_mask,
        const bool want_dq,
        const int  tkv,
        const float scale, const float softcap) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    // retro delta (P1b): the grid is (query tile, head, batch) rather than one flat
    // row index. That is what lets the warps of a block cooperate: they now share ih
    // and ib by construction, so they need the *same* K/V rows and can read them
    // once into shared memory instead of each re-streaming the cache from global.
    // A flat row index only happened to group query rows of one head when nq was an
    // exact multiple of the warps per block.
    const int64_t iq  = int64_t(blockIdx.x)*(blockDim.x/WARP_SIZE) + warp;
    const int64_t ih  = blockIdx.y;
    const int64_t ib  = blockIdx.z;
    const int64_t ikh = ih / ratio;

    // Every warp must reach every __syncthreads() and take part in every
    // cooperative load, so an out-of-range query row is carried as a predicate
    // instead of returning. A return here would strand the remaining warps on the
    // next barrier, and the tail block has such rows whenever nq is not a multiple
    // of the warps per block.
    bool active = iq < nq;

    const char * q_row  = active ? q  + iq*nbq1 + ih*nbq2 + ib*nbq3 : q;
    const char * out_row = active ? out + ih*nbo1 + iq*nbo2 + ib*nbo3 : out;
    const char * dO_row = active ? dO + ih*nbd1 + iq*nbd2 + ib*nbd3 : dO;
    const char * mask_row = (has_mask && active) ? (mask + iq*nbm1 + ib*nbm3) : nullptr;

    // K/V tile, converted to F32 once per tile instead of on every read. Sized by
    // the host so tkv*(hsk+hsv) floats fit the shared-memory budget at this head
    // dimension; lane `l` reads element l + r*32 of a row, so consecutive lanes hit
    // consecutive banks.
    extern __shared__ float fab_smem[];
    float * sk = fab_smem;
    float * sv = fab_smem + (size_t) tkv*hsk;

    // dst packed layout: [ dQ | dK | dV | scalars ], all contiguous F32.
    float * dQ = dst;
    float * stats = (float *) ((char *) dst + off_s);
    const int64_t qbase = ((ib*nhead + ih)*nq + iq)*hsk;
    const int64_t stat_index = (ib*nhead + ih)*nq + iq;
    const int64_t nstats = nbatch*nhead*nq;

    // Lane `lane` owns head-dim elements d = lane + r*WARP_SIZE. Consecutive
    // lanes therefore hold consecutive d, so every global load/atomic below is
    // coalesced across the warp.
    float qv[REGS];   // q[d], cached across both passes
    float dov[REGS];  // dO[d], cached across both passes
    float dq[REGS];
#pragma unroll
    for (int r = 0; r < REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        qv[r]    = (active && d < hsk) ? *(const float *) (q_row  + d*nbq0) : 0.0f;
        dov[r]   = (active && d < hsv) ? *(const float *) (dO_row + d*nbd0) : 0.0f;
        dq[r]    = 0.0f;
    }

    // retro delta (P1b): stage one K/V tile in shared memory, cooperatively, for all
    // warps of the block. Global traffic on K/V is divided by the number of query
    // rows per block; before this, each warp streamed the whole cache itself, twice
    // over (pass 1 and pass 2a) plus a windowed third time.
    //
    // t is looped outside d so the address arithmetic stays a multiply-add per
    // element: hsk/hsv are runtime values, and folding both into one flat index
    // would put a 64-bit division on every element of every tile.
    auto load_k_tile = [&](int64_t kv0, int n) {
        for (int t = 0; t < n; ++t) {
            const char * kr = k + (kv0 + t)*nbk1 + ikh*nbk2 + ib*nbk3;
            for (int64_t d = threadIdx.x; d < hsk; d += blockDim.x) {
                sk[(size_t) t*hsk + d] = fab_load(kr, d, nbk0, kv_f16);
            }
        }
    };
    auto load_v_tile = [&](int64_t kv0, int n) {
        for (int t = 0; t < n; ++t) {
            const char * vr = v + (kv0 + t)*nbv1 + ikh*nbv2 + ib*nbv3;
            for (int64_t d = threadIdx.x; d < hsv; d += blockDim.x) {
                sv[(size_t) t*hsv + d] = fab_load(vr, d, nbv0, kv_f16);
            }
        }
    };

    // F1: only recompute the row normalization. O is already an input of the
    // backward node, so reconstructing it by streaming V a second time was pure
    // duplicate work. The local LSE pass is the compatibility fallback described
    // in OPTIMS_V4 B.4; it scans K only and preserves the existing graph ABI.
    float m = -INFINITY;
    float l = 0.0f;
    for (int64_t kv0 = 0; kv0 < nkv; kv0 += tkv) {
        const int n = (int) min((int64_t) tkv, nkv - kv0);
        __syncthreads();
        load_k_tile(kv0, n);
        __syncthreads();
        for (int t = 0; t < n; ++t) {
            const int64_t ik = kv0 + t;
            float dot = 0.0f;
#pragma unroll
            for (int r = 0; r < REGS; ++r) {
                const int64_t d = lane + r*WARP_SIZE;
                if (d < hsk) { dot += qv[r] * sk[(size_t) t*hsk + d]; }
            }
            dot = warp_reduce_sum(dot);
            float score = softcap != 0.0f ? softcap*tanhf(dot*scale/softcap) : dot*scale;
            if (has_mask && active) {
                score += __half2float(*(const half *) (mask_row + ik*nbm0));
            }
            // score is identical in every lane, so this is warp-uniform. Inactive
            // warps fall through the same control flow; only their results are
            // discarded, so the barriers above stay collective.
            if (score == -INFINITY) {
                continue;
            }
            const float m_new = fmaxf(m, score);
            const float corr = expf(m - m_new); // m=-inf on first hit -> corr=0
            const float p = expf(score - m_new);
            l = l*corr + p;
            m = m_new;
        }
    }
    float delta = 0.0f;
#pragma unroll
    for (int r = 0; r < REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        if (active && d < hsv) {
            delta += dov[r] * *(const float *) (out_row + d*nbo0);
        }
    }
    delta = warp_reduce_sum(delta);

    if (l <= 0.0f) {
        // Fully masked row: zero gradients for dQ; dK/dV get no contribution. This
        // used to return; it cannot any more, because the passes below synchronize.
        if (want_dq && active) {
#pragma unroll
            for (int r = 0; r < REGS; ++r) {
                const int64_t d = lane + r*WARP_SIZE;
                if (d < hsk) { dQ[qbase + d] = 0.0f; }
            }
        }
        active = false;
        l = 1.0f; // keep the arithmetic below finite; its results are discarded
        m = INFINITY;
        delta = 0.0f;
    }
    const float inv_l = 1.0f/l;
    if (lane == 0 && iq < nq) {
        stats[stat_index] = m + logf(l);
        stats[nstats + stat_index] = delta;
    }

    // prob[ik] = exp(score - m)/l, recomputed rather than stored.
    // `ds_out` is the softmax-input gradient; the two passes below share it.
    //
    // retro delta (P1b): takes the K/V elements this lane owns as registers rather
    // than a row pointer, so the caller decides where they came from — the shared
    // tile in the dQ pass, global memory in the windowed dK/dV pass. It also removes
    // a redundant read: pass 2a used to fetch k_row once here for the dot product
    // and again for the dq accumulation.
    auto score_grads = [&](int64_t ik, const float (&kd)[REGS], const float (&vd)[REGS],
                           float & prob_out, float & ds_out) -> bool {
        float dot    = 0.0f;
        float dot_dv = 0.0f;
#pragma unroll
        for (int r = 0; r < REGS; ++r) {
            const int64_t d = lane + r*WARP_SIZE;
            if (d < hsk) { dot    += qv[r]  * kd[r]; }
            if (d < hsv) { dot_dv += dov[r] * vd[r]; }
        }
        dot    = warp_reduce_sum(dot);
        dot_dv = warp_reduce_sum(dot_dv);
        float score = softcap != 0.0f ? softcap*tanhf(dot*scale/softcap) : dot*scale;
        if (has_mask && active) {
            score += __half2float(*(const half *) (mask_row + ik*nbm0));
        }
        if (score == -INFINITY) {
            return false;
        }
        const float p = expf(score - m)*inv_l;
        float deriv = scale;
        if (softcap != 0.0f) {
            const float t = tanhf(dot*scale/softcap);
            deriv *= 1.0f - t*t;
        }
        prob_out = p;
        ds_out   = p*(dot_dv - delta)*deriv;
        return true;
    };

    // Pass 2a: dQ. Every key contributes, so this one still spans the cache -- but
    // it now spans it through the shared tile, and it writes a row this warp owns
    // exclusively, with no atomics. ik ascends exactly as before, so dq accumulates
    // its terms in the same order.
    if (want_dq) {
        for (int64_t kv0 = 0; kv0 < nkv; kv0 += tkv) {
            const int n = (int) min((int64_t) tkv, nkv - kv0);
            __syncthreads();
            load_k_tile(kv0, n);
            load_v_tile(kv0, n);
            __syncthreads();
            for (int t = 0; t < n; ++t) {
                float kd[REGS];
                float vd[REGS];
#pragma unroll
                for (int r = 0; r < REGS; ++r) {
                    const int64_t d = lane + r*WARP_SIZE;
                    kd[r] = d < hsk ? sk[(size_t) t*hsk + d] : 0.0f;
                    vd[r] = d < hsv ? sv[(size_t) t*hsv + d] : 0.0f;
                }
                float prob;
                float ds;
                if (!score_grads(kv0 + t, kd, vd, prob, ds)) {
                    continue;
                }
#pragma unroll
                for (int r = 0; r < REGS; ++r) {
                    const int64_t d = lane + r*WARP_SIZE;
                    if (d < hsk) { dq[r] += ds*kd[r]; }
                }
            }
        }
        if (active) {
#pragma unroll
            for (int r = 0; r < REGS; ++r) {
                const int64_t d = lane + r*WARP_SIZE;
                if (d < hsk) { dQ[qbase + d] = dq[r]; }
            }
        }
    }

}

// F3: one warp owns one dK/dV row. It visits grouped query heads and query rows
// in lexical order, so no two blocks ever write the same address and repeated
// executions are bit-identical. Vulkan already uses this ownership scheme; the
// CUDA implementation intentionally mirrors it.
template<int REGS>
static __global__ void flash_attn_back_kv_kernel(
        const char * __restrict__ q, const char * __restrict__ k,
        const char * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ dO, float * __restrict__ dst,
        const int64_t hsk, const int64_t hsv, const int64_t nq, const int64_t nkv,
        const int64_t nhead, const int64_t nheadk, const int64_t nbatch,
        const int64_t ratio,
        const int64_t nbq0, const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk0, const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv0, const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t nbm0, const int64_t nbm1, const int64_t nbm3,
        const int64_t nbd0, const int64_t nbd1, const int64_t nbd2, const int64_t nbd3,
        const size_t off_k, const size_t off_v, const size_t off_s,
        const bool kv_f16, const bool has_mask,
        const bool want_dk, const bool want_dv,
        const fa_back_window window,
        const float scale, const float softcap) {
    const int lane = threadIdx.x;
    const int64_t j   = blockIdx.x;
    const int64_t ikh = blockIdx.y;
    const int64_t ib  = blockIdx.z;
    const int64_t ik  = fab_window_key(window, ib, j);
    const bool in_cache = ik >= 0 && ik < nkv;

    float dk[REGS];
    float dv[REGS];
#pragma unroll
    for (int r = 0; r < REGS; ++r) {
        dk[r] = 0.0f;
        dv[r] = 0.0f;
    }

    if (in_cache) {
        const char * k_row = k + ik*nbk1 + ikh*nbk2 + ib*nbk3;
        const char * v_row = v + ik*nbv1 + ikh*nbv2 + ib*nbv3;
        float kd[REGS];
        float vd[REGS];
#pragma unroll
        for (int r = 0; r < REGS; ++r) {
            const int64_t id = lane + r*WARP_SIZE;
            kd[r] = id < hsk ? fab_load(k_row, id, nbk0, kv_f16) : 0.0f;
            vd[r] = id < hsv ? fab_load(v_row, id, nbv0, kv_f16) : 0.0f;
        }

        const float * stats = (const float *) ((const char *) dst + off_s);
        const int64_t nstats = nbatch*nhead*nq;
        for (int64_t rhead = 0; rhead < ratio; ++rhead) {
            const int64_t ih = ikh*ratio + rhead;
            for (int64_t iq = 0; iq < nq; ++iq) {
                const char * q_row = q + iq*nbq1 + ih*nbq2 + ib*nbq3;
                const char * dO_row = dO + ih*nbd1 + iq*nbd2 + ib*nbd3;
                const char * mask_row = has_mask ? mask + iq*nbm1 + ib*nbm3 : nullptr;

                float dot_qk = 0.0f;
                float dot_dv = 0.0f;
#pragma unroll
                for (int r = 0; r < REGS; ++r) {
                    const int64_t id = lane + r*WARP_SIZE;
                    if (id < hsk) {
                        dot_qk += *(const float *) (q_row + id*nbq0) * kd[r];
                    }
                    if (id < hsv) {
                        dot_dv += *(const float *) (dO_row + id*nbd0) * vd[r];
                    }
                }
                dot_qk = warp_reduce_sum(dot_qk);
                dot_dv = warp_reduce_sum(dot_dv);

                float score = softcap != 0.0f
                    ? softcap*tanhf(dot_qk*scale/softcap) : dot_qk*scale;
                if (has_mask) {
                    score += __half2float(*(const half *) (mask_row + ik*nbm0));
                }
                const int64_t stat_index = (ib*nhead + ih)*nq + iq;
                const float lse = stats[stat_index];
                const float delta = stats[nstats + stat_index];
                const float prob = score == -INFINITY ? 0.0f : expf(score - lse);
                float deriv = scale;
                if (softcap != 0.0f) {
                    const float t = tanhf(dot_qk*scale/softcap);
                    deriv *= 1.0f - t*t;
                }
                const float ds = prob*(dot_dv - delta)*deriv;
#pragma unroll
                for (int r = 0; r < REGS; ++r) {
                    const int64_t id = lane + r*WARP_SIZE;
                    if (id < hsk) {
                        dk[r] += ds * *(const float *) (q_row + id*nbq0);
                    }
                    if (id < hsv) {
                        dv[r] += prob * *(const float *) (dO_row + id*nbd0);
                    }
                }
            }
        }
    }

    float * dK = (float *) ((char *) dst + off_k);
    float * dV = (float *) ((char *) dst + off_v);
    const int64_t kbase = ((ib*nheadk + ikh)*window.nwin + j)*hsk;
    const int64_t vbase = ((ib*nheadk + ikh)*window.nwin + j)*hsv;
#pragma unroll
    for (int r = 0; r < REGS; ++r) {
        const int64_t id = lane + r*WARP_SIZE;
        if (want_dk && id < hsk) { dK[kbase + id] = dk[r]; }
        if (want_dv && id < hsv) { dV[vbase + id] = dv[r]; }
    }
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * out = dst->src[4];
    const ggml_tensor * d = dst->src[5];
    const ggml_tensor * sinks = dst->src[6];
    const ggml_tensor * kv_idxs = dst->src[9];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    GGML_ASSERT(d->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == v->type);
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!sinks); // attention sinks are not part of the training path yet

    const int64_t hsk    = q->ne[0];
    const int64_t nq     = q->ne[1];
    const int64_t nhead  = q->ne[2];
    const int64_t nbatch = q->ne[3];
    const int64_t nkv    = k->ne[1];
    const int64_t nheadk = k->ne[2];
    const int64_t hsv    = v->ne[0];
    GGML_ASSERT(hsk <= FA_BACK_MAX_D && hsv <= FA_BACK_MAX_D);
    GGML_ASSERT(nhead % nheadk == 0);
    // hsk and hsv may differ (MLA); one bucket has to cover both.
    const int64_t hs_max = MAX(hsk, hsv);
    const int64_t ratio = nhead / nheadk;

    float params[3];
    memcpy(params, dst->op_params, sizeof(params));
    const float scale   = params[0];
    const float softcap = params[2];

    const bool kv_f16 = k->type == GGML_TYPE_F16;
    const bool has_mask = mask != nullptr;

    const int32_t grad_mask = ggml_get_op_params_i32(dst, 3);
    const bool want_dq = (grad_mask & GGML_FLASH_ATTN_BACK_GRAD_Q) != 0;
    const bool want_dk = (grad_mask & GGML_FLASH_ATTN_BACK_GRAD_K) != 0;
    const bool want_dv = (grad_mask & GGML_FLASH_ATTN_BACK_GRAD_V) != 0;

    // Packed output offsets: derived from ggml so the layout lives in one place.
    size_t off_k = 0;
    size_t off_v = 0;
    size_t off_s = 0;
    ggml_flash_attn_back_offsets(dst, nullptr, &off_k, &off_v, &off_s);

    fa_back_window window = {};
    if (kv_idxs) {
        GGML_ASSERT(kv_idxs->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(kv_idxs));
        window.idxs    = (const int32_t *) kv_idxs->data;
        window.nwin    = ggml_flash_attn_back_grad_k(dst)->ne[1];
        window.stride  = ggml_get_op_params_i32(dst, 4);
        window.stream0 = ggml_get_op_params_i32(dst, 5);
        GGML_ASSERT(ggml_nelements(kv_idxs) == window.nwin*nbatch);
    } else {
        window.nwin = nkv;
    }

    cudaStream_t stream = ctx.stream();
    float * dst_d = (float *) dst->data;
    // Define alignment padding and segments omitted by grad_mask. The kernels
    // themselves write every requested gradient/statistics element exactly once.
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const char * mma_env = getenv("GGML_CUDA_FA_BACK_MMA");
    const bool mma_enabled = mma_env == nullptr || std::atoi(mma_env) != 0;
    const bool use_mma = mma_enabled && kv_f16 && hsk == hsv && hsk >= 16 &&
        hsk <= FA_BACK_MMA_MAX_D && hsk % 16 == 0 &&
        k->nb[0] == sizeof(half) && v->nb[0] == sizeof(half) &&
        GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc);
    if (use_mma) {
        const fa_back_mma_args mma = {
            hsk, nq, nkv, nhead, nheadk, ratio,
            q->nb[0], q->nb[1], q->nb[2], q->nb[3],
            k->nb[1], k->nb[2], k->nb[3],
            v->nb[1], v->nb[2], v->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0,
            out->nb[0], out->nb[1], out->nb[2], out->nb[3],
            d->nb[0], d->nb[1], d->nb[2], d->nb[3],
            off_k, off_v, off_s,
            window.idxs, window.nwin, window.stride, window.stream0,
            scale, softcap,
            has_mask, want_dq, want_dk, want_dv,
        };
        const size_t mma_smem = 4*(size_t) FA_BACK_MMA_TILE*hsk*sizeof(half) +
            2*(size_t) FA_BACK_MMA_TILE*FA_BACK_MMA_TILE*sizeof(float) +
            3*(size_t) FA_BACK_MMA_TILE*sizeof(float);
        const dim3 grid_q((unsigned) ((nq + FA_BACK_MMA_TILE - 1)/FA_BACK_MMA_TILE),
                          (unsigned) nhead, (unsigned) nbatch);
        flash_attn_back_mma_q_kernel<0><<<grid_q, FA_BACK_MMA_THREADS, mma_smem, stream>>>(
            (const char *) q->data, (const half *) k->data, (const half *) v->data,
            mask ? (const char *) mask->data : nullptr,
            (const char *) out->data, (const char *) d->data, dst_d, mma);
        if (want_dk || want_dv) {
            const dim3 grid_kv((unsigned) ((window.nwin + FA_BACK_MMA_TILE - 1)/FA_BACK_MMA_TILE),
                               (unsigned) nheadk, (unsigned) nbatch);
            flash_attn_back_mma_kv_kernel<0><<<grid_kv, FA_BACK_MMA_THREADS, mma_smem, stream>>>(
                (const char *) q->data, (const half *) k->data, (const half *) v->data,
                mask ? (const char *) mask->data : nullptr,
                (const char *) d->data, dst_d, mma);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif

    // One warp per query row, and one block per (query tile, head, batch) so the
    // warps of a block share ih/ib and can therefore share a K/V tile.
    const int block = 256;
    const int rows_per_block = block/WARP_SIZE;
    const dim3 grid(
        (unsigned) ((nq + rows_per_block - 1)/rows_per_block),
        (unsigned) nhead,
        (unsigned) nbatch);

    // retro delta (P1b): K/V tile depth, chosen so the tile fits a conservative
    // shared-memory budget at this head dimension. 32 KiB rather than the full
    // 48 KiB default so two blocks per SM remain resident -- occupancy is the whole
    // point of this kernel, and trading it back for a deeper tile would be
    // self-defeating. At head_dim 128 that is a 32-row tile; at Gemma-4's 512 it
    // falls to 8, which still amortizes each K/V read across the block's query rows.
    const size_t row_floats = (size_t) (hsk + hsv);
    const int tkv = (int) std::max<size_t>(1, std::min<size_t>(32, (32*1024)/(row_floats*sizeof(float))));
    const size_t smem = (size_t) tkv*row_floats*sizeof(float);

    // Smallest bucket that covers the shape: head_dim only costs registers here,
    // so a wide model works, but narrow models must not pay for it.
#define FA_BACK_LAUNCH(REGS)                                                        \
    flash_attn_back_kernel<REGS><<<grid, block, smem, stream>>>(                     \
        (const char *) q->data, (const char *) k->data, (const char *) v->data,     \
        mask ? (const char *) mask->data : nullptr,                                 \
        (const char *) out->data, (const char *) d->data, dst_d,                    \
        hsk, hsv, nq, nkv, nhead, nheadk, nbatch, ratio,                            \
        q->nb[0], q->nb[1], q->nb[2], q->nb[3],                                     \
        k->nb[0], k->nb[1], k->nb[2], k->nb[3],                                     \
        v->nb[0], v->nb[1], v->nb[2], v->nb[3],                                     \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0,     \
        out->nb[0], out->nb[1], out->nb[2], out->nb[3],                             \
        d->nb[0], d->nb[1], d->nb[2], d->nb[3],                                     \
        off_s, kv_f16, has_mask, want_dq, tkv, scale, softcap);                      \
    if (want_dk || want_dv) {                                                       \
        const dim3 grid_kv((unsigned) window.nwin, (unsigned) nheadk,                \
                           (unsigned) nbatch);                                       \
        flash_attn_back_kv_kernel<REGS><<<grid_kv, WARP_SIZE, 0, stream>>>(          \
            (const char *) q->data, (const char *) k->data, (const char *) v->data, \
            mask ? (const char *) mask->data : nullptr,                             \
            (const char *) d->data, dst_d,                                          \
            hsk, hsv, nq, nkv, nhead, nheadk, nbatch, ratio,                        \
            q->nb[0], q->nb[1], q->nb[2], q->nb[3],                                 \
            k->nb[0], k->nb[1], k->nb[2], k->nb[3],                                 \
            v->nb[0], v->nb[1], v->nb[2], v->nb[3],                                 \
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, \
            d->nb[0], d->nb[1], d->nb[2], d->nb[3],                                 \
            off_k, off_v, off_s, kv_f16, has_mask, want_dk, want_dv, window,        \
            scale, softcap);                                                        \
    }

    if (hs_max <= FA_BACK_REGS_SMALL*WARP_SIZE) {
        FA_BACK_LAUNCH(FA_BACK_REGS_SMALL);
    } else if (hs_max <= FA_BACK_REGS_WIDE*WARP_SIZE) {
        FA_BACK_LAUNCH(FA_BACK_REGS_WIDE);
    } else {
        // hs_max <= FA_BACK_MAX_D by the assert above, so this is the XWIDE case.
        FA_BACK_LAUNCH(FA_BACK_REGS_XWIDE);
    }
#undef FA_BACK_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
