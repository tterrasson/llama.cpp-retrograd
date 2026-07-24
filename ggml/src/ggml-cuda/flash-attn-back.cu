#include "flash-attn-back.cuh"

#include <cstdint>

// retro delta: streaming Flash Attention backward, one *warp* per query row
// (batch ib, head ih, query iq). The warp recomputes scores/softmax/O exactly as
// the fork CPU reference does, then accumulates dQ/dK/dV. dQ rows are unique per
// warp and written directly; dK/dV are shared across query rows and GQA head
// groups, so they use atomicAdd into the zero-initialized result.
//
// The head dimension is split across the 32 lanes (lane `l` owns d = l, l+32,
// l+64, ...), which is what makes this affordable:
//   - the per-row accumulators are 2*FA_BACK_REGS registers per lane instead of
//     o_acc[128] + dq[128] = 1 KiB per thread, which spilled to local memory;
//   - the q.k and dO.v dot products become warp shuffle reductions instead of
//     128-iteration serial loops;
//   - the dK/dV atomics are issued 32-wide to consecutive addresses (coalesced)
//     instead of hsk+hsv = 256 of them serialized inside one thread.
//
// An earlier revision ran one *thread* per query row. On a 0.6B model at
// nq=256/nkv=1024 that is ~5e8 serialized atomicAdds per launch on top of the
// spill traffic: 25 ms per launch, 73% of training wall-clock.

#define FA_BACK_MAX_D 128
// Head-dimension elements owned by each lane (FA_BACK_MAX_D split over a warp).
#define FA_BACK_REGS  (FA_BACK_MAX_D/32)

static __device__ __forceinline__ float fab_load(const char * base, int64_t idx,
                                                  int64_t stride0, bool is_f16) {
    if (is_f16) {
        return __half2float(*(const half *) (base + idx*stride0));
    }
    return *(const float *) (base + idx*stride0);
}

static __global__ void flash_attn_back_kernel(
        const char * __restrict__ q, const char * __restrict__ k,
        const char * __restrict__ v, const char * __restrict__ mask,
        const char * __restrict__ dO, float * __restrict__ dst,
        const int64_t hsk, const int64_t hsv, const int64_t nq, const int64_t nkv,
        const int64_t nhead, const int64_t nheadk, const int64_t nbatch,
        const int64_t ratio,
        // element strides (bytes) for each operand's non-fastest dims
        const int64_t nbq0, const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk0, const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv0, const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t nbm0, const int64_t nbm1, const int64_t nbm3,
        const int64_t nbd0, const int64_t nbd1, const int64_t nbd2, const int64_t nbd3,
        const size_t off_k, const size_t off_v,
        const bool kv_f16, const bool has_mask,
        const float scale, const float softcap) {
    const int     lane   = threadIdx.x % WARP_SIZE;
    const int64_t row    = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x) / WARP_SIZE;
    const int64_t n_rows = nq*nhead*nbatch;
    // Warp-uniform: a whole warp owns one query row, so the guard never splits
    // a warp and every shuffle reduction below sees all 32 lanes.
    if (row >= n_rows) {
        return;
    }
    const int64_t iq = row % nq;
    const int64_t ih = (row / nq) % nhead;
    const int64_t ib = row / (nq*nhead);
    const int64_t ikh = ih / ratio;

    const char * q_row  = q  + iq*nbq1 + ih*nbq2 + ib*nbq3;
    const char * dO_row = dO + ih*nbd1 + iq*nbd2 + ib*nbd3;
    const char * mask_row = has_mask ? (mask + iq*nbm1 + ib*nbm3) : nullptr;

    // dst packed layout: [ dQ | dK | dV | scalars ], all contiguous F32.
    float * dQ = dst;
    float * dK = (float *) ((char *) dst + off_k);
    float * dV = (float *) ((char *) dst + off_v);
    const int64_t qbase = ((ib*nhead + ih)*nq + iq)*hsk;

    // Lane `lane` owns head-dim elements d = lane + r*WARP_SIZE. Consecutive
    // lanes therefore hold consecutive d, so every global load/atomic below is
    // coalesced across the warp.
    float qv[FA_BACK_REGS];   // q[d], cached across both passes
    float dov[FA_BACK_REGS];  // dO[d], cached across both passes
    float o_acc[FA_BACK_REGS];
    float dq[FA_BACK_REGS];
#pragma unroll
    for (int r = 0; r < FA_BACK_REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        qv[r]    = d < hsk ? *(const float *) (q_row  + d*nbq0) : 0.0f;
        dov[r]   = d < hsv ? *(const float *) (dO_row + d*nbd0) : 0.0f;
        o_acc[r] = 0.0f;
        dq[r]    = 0.0f;
    }

    // Pass 1: online softmax over K, accumulating the prob-weighted output O.
    float m = -INFINITY;
    float l = 0.0f;
    for (int64_t ik = 0; ik < nkv; ++ik) {
        const char * k_row = k + ik*nbk1 + ikh*nbk2 + ib*nbk3;
        float dot = 0.0f;
#pragma unroll
        for (int r = 0; r < FA_BACK_REGS; ++r) {
            const int64_t d = lane + r*WARP_SIZE;
            if (d < hsk) { dot += qv[r] * fab_load(k_row, d, nbk0, kv_f16); }
        }
        dot = warp_reduce_sum(dot);
        float score = softcap != 0.0f ? softcap*tanhf(dot*scale/softcap) : dot*scale;
        if (has_mask) {
            score += __half2float(*(const half *) (mask_row + ik*nbm0));
        }
        // score is identical in every lane, so this is warp-uniform.
        if (score == -INFINITY) {
            continue;
        }
        const float m_new = fmaxf(m, score);
        const float corr = expf(m - m_new); // m=-inf on first hit -> corr=0
        const float p = expf(score - m_new);
        l = l*corr + p;
        const char * v_row = v + ik*nbv1 + ikh*nbv2 + ib*nbv3;
#pragma unroll
        for (int r = 0; r < FA_BACK_REGS; ++r) {
            const int64_t d = lane + r*WARP_SIZE;
            if (d < hsv) { o_acc[r] = o_acc[r]*corr + p*fab_load(v_row, d, nbv0, kv_f16); }
        }
        m = m_new;
    }
    if (l <= 0.0f) {
        // Fully masked row: zero gradients for dQ; dK/dV get no contribution.
#pragma unroll
        for (int r = 0; r < FA_BACK_REGS; ++r) {
            const int64_t d = lane + r*WARP_SIZE;
            if (d < hsk) { dQ[qbase + d] = 0.0f; }
        }
        return;
    }
    const float inv_l = 1.0f/l;
    float delta = 0.0f;
#pragma unroll
    for (int r = 0; r < FA_BACK_REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        if (d < hsv) {
            o_acc[r] *= inv_l; // O[d]
            delta += dov[r] * o_acc[r];
        }
    }
    delta = warp_reduce_sum(delta);

    // Pass 2: gradients. prob[ik] = exp(score - m)/l (recomputed).
    for (int64_t ik = 0; ik < nkv; ++ik) {
        const char * k_row = k + ik*nbk1 + ikh*nbk2 + ib*nbk3;
        const char * v_row = v + ik*nbv1 + ikh*nbv2 + ib*nbv3;
        float dot    = 0.0f;
        float dot_dv = 0.0f;
#pragma unroll
        for (int r = 0; r < FA_BACK_REGS; ++r) {
            const int64_t d = lane + r*WARP_SIZE;
            if (d < hsk) { dot    += qv[r]  * fab_load(k_row, d, nbk0, kv_f16); }
            if (d < hsv) { dot_dv += dov[r] * fab_load(v_row, d, nbv0, kv_f16); }
        }
        dot    = warp_reduce_sum(dot);
        dot_dv = warp_reduce_sum(dot_dv);
        float score = softcap != 0.0f ? softcap*tanhf(dot*scale/softcap) : dot*scale;
        if (has_mask) {
            score += __half2float(*(const half *) (mask_row + ik*nbm0));
        }
        if (score == -INFINITY) {
            continue;
        }
        const float prob = expf(score - m)*inv_l;
        float deriv = scale;
        if (softcap != 0.0f) {
            const float t = tanhf(dot*scale/softcap);
            deriv *= 1.0f - t*t;
        }
        const float ds = prob*(dot_dv - delta)*deriv;
        const int64_t kbase = ((ib*nheadk + ikh)*nkv + ik)*hsk;
        const int64_t vbase = ((ib*nheadk + ikh)*nkv + ik)*hsv;
#pragma unroll
        for (int r = 0; r < FA_BACK_REGS; ++r) {
            const int64_t d = lane + r*WARP_SIZE;
            if (d < hsk) {
                dq[r] += ds*fab_load(k_row, d, nbk0, kv_f16);
                atomicAdd(&dK[kbase + d], ds*qv[r]);
            }
            if (d < hsv) {
                atomicAdd(&dV[vbase + d], prob*dov[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < FA_BACK_REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        if (d < hsk) { dQ[qbase + d] = dq[r]; }
    }
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * d = dst->src[5];
    const ggml_tensor * sinks = dst->src[6];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
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
    const int64_t ratio = nhead / nheadk;

    float params[3];
    memcpy(params, dst->op_params, sizeof(params));
    const float scale   = params[0];
    const float softcap = params[2];

    const bool kv_f16 = k->type == GGML_TYPE_F16;
    const bool has_mask = mask != nullptr;

    // Packed output offsets (must match ggml_flash_attn_ext_back).
    const int64_t elem_q = ggml_nelements(q);
    const int64_t elem_k = ggml_nelements(k);
    const size_t off_k = GGML_PAD(elem_q*sizeof(float), GGML_MEM_ALIGN);
    const size_t off_v = off_k + GGML_PAD(elem_k*sizeof(float), GGML_MEM_ALIGN);

    cudaStream_t stream = ctx.stream();
    float * dst_d = (float *) dst->data;
    // dK/dV accumulate with atomics and the scalar tail is unused; zero all of it.
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

    const int64_t n_rows = nq*nhead*nbatch;
    // One warp per query row.
    const int block = 256;
    const int rows_per_block = block/WARP_SIZE;
    const int grid = (int) ((n_rows + rows_per_block - 1)/rows_per_block);
    flash_attn_back_kernel<<<grid, block, 0, stream>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        mask ? (const char *) mask->data : nullptr,
        (const char *) d->data, dst_d,
        hsk, hsv, nq, nkv, nhead, nheadk, nbatch, ratio,
        q->nb[0], q->nb[1], q->nb[2], q->nb[3],
        k->nb[0], k->nb[1], k->nb[2], k->nb[3],
        v->nb[0], v->nb[1], v->nb[2], v->nb[3],
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0,
        d->nb[0], d->nb[1], d->nb[2], d->nb[3],
        off_k, off_v, kv_f16, has_mask, scale, softcap);
    CUDA_CHECK(cudaGetLastError());
}
