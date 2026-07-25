#include "flash-attn-back.cuh"

#include <cstdint>

#include <algorithm>

// retro delta: streaming Flash Attention backward, one *warp* per query row
// (batch ib, head ih, query iq). The warp recomputes scores/softmax/O exactly as
// the fork CPU reference does, then accumulates dQ/dK/dV. dQ rows are unique per
// warp and written directly; dK/dV are shared across query rows and GQA head
// groups, so they use atomicAdd into the zero-initialized result.
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
//   - the dK/dV atomics are issued 32-wide to consecutive addresses (coalesced)
//     instead of hsk+hsv of them serialized inside one thread.
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

template<int REGS>
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
        const bool want_dq, const bool want_dk, const bool want_dv,
        const fa_back_window window,
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
    float * dK = (float *) ((char *) dst + off_k);
    float * dV = (float *) ((char *) dst + off_v);
    const int64_t qbase = ((ib*nhead + ih)*nq + iq)*hsk;

    // Lane `lane` owns head-dim elements d = lane + r*WARP_SIZE. Consecutive
    // lanes therefore hold consecutive d, so every global load/atomic below is
    // coalesced across the warp.
    float qv[REGS];   // q[d], cached across both passes
    float dov[REGS];  // dO[d], cached across both passes
    float o_acc[REGS];
    float dq[REGS];
#pragma unroll
    for (int r = 0; r < REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        qv[r]    = (active && d < hsk) ? *(const float *) (q_row  + d*nbq0) : 0.0f;
        dov[r]   = (active && d < hsv) ? *(const float *) (dO_row + d*nbd0) : 0.0f;
        o_acc[r] = 0.0f;
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
    auto load_tile = [&](int64_t kv0, int n) {
        for (int t = 0; t < n; ++t) {
            const char * kr = k + (kv0 + t)*nbk1 + ikh*nbk2 + ib*nbk3;
            for (int64_t d = threadIdx.x; d < hsk; d += blockDim.x) {
                sk[(size_t) t*hsk + d] = fab_load(kr, d, nbk0, kv_f16);
            }
            const char * vr = v + (kv0 + t)*nbv1 + ikh*nbv2 + ib*nbv3;
            for (int64_t d = threadIdx.x; d < hsv; d += blockDim.x) {
                sv[(size_t) t*hsv + d] = fab_load(vr, d, nbv0, kv_f16);
            }
        }
    };

    // Pass 1: online softmax over K, accumulating the prob-weighted output O.
    // ik still ascends one at a time within a tile and the tiles ascend, so the
    // online-softmax recurrence sees the same sequence of scores as before and the
    // result is bit-identical to the untiled kernel and to the CPU reference.
    float m = -INFINITY;
    float l = 0.0f;
    for (int64_t kv0 = 0; kv0 < nkv; kv0 += tkv) {
        const int n = (int) min((int64_t) tkv, nkv - kv0);
        __syncthreads();
        load_tile(kv0, n);
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
#pragma unroll
            for (int r = 0; r < REGS; ++r) {
                const int64_t d = lane + r*WARP_SIZE;
                if (d < hsv) { o_acc[r] = o_acc[r]*corr + p*sv[(size_t) t*hsv + d]; }
            }
            m = m_new;
        }
    }
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
    }
    const float inv_l = 1.0f/l;
    float delta = 0.0f;
#pragma unroll
    for (int r = 0; r < REGS; ++r) {
        const int64_t d = lane + r*WARP_SIZE;
        if (d < hsv) {
            o_acc[r] *= inv_l; // O[d]
            delta += dov[r] * o_acc[r];
        }
    }
    delta = warp_reduce_sum(delta);

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
            load_tile(kv0, n);
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

    // Pass 2b: dK/dV, over the gradient window only. Keys outside it were
    // written by earlier steps and are constants -- computing and atomicAdd-ing
    // over the whole cache is the n_kv/n_window waste this window exists to
    // avoid. Without a window, nwin == nkv and this degenerates to the dense
    // case exactly.
    // Not tiled: the window maps j to arbitrary cache rows through kv_idxs, so a
    // contiguous shared tile does not apply, and the window is already the small
    // axis (it is why this pass exists). No barrier here, so the early `continue`s
    // are safe — but the atomics must respect `active`, since dK/dV are shared
    // across query rows and an inactive warp holds no gradient to contribute.
    if ((want_dk || want_dv) && active) {
        for (int64_t j = 0; j < window.nwin; ++j) {
            const int64_t ik = fab_window_key(window, ib, j);
            if (ik < 0 || ik >= nkv) {
                continue;
            }
            const char * k_row = k + ik*nbk1 + ikh*nbk2 + ib*nbk3;
            const char * v_row = v + ik*nbv1 + ikh*nbv2 + ib*nbv3;
            float kd[REGS];
            float vd[REGS];
#pragma unroll
            for (int r = 0; r < REGS; ++r) {
                const int64_t d = lane + r*WARP_SIZE;
                kd[r] = d < hsk ? fab_load(k_row, d, nbk0, kv_f16) : 0.0f;
                vd[r] = d < hsv ? fab_load(v_row, d, nbv0, kv_f16) : 0.0f;
            }
            float prob;
            float ds;
            if (!score_grads(ik, kd, vd, prob, ds)) {
                continue;
            }
            const int64_t kbase = ((ib*nheadk + ikh)*window.nwin + j)*hsk;
            const int64_t vbase = ((ib*nheadk + ikh)*window.nwin + j)*hsv;
#pragma unroll
            for (int r = 0; r < REGS; ++r) {
                const int64_t d = lane + r*WARP_SIZE;
                if (want_dk && d < hsk) { atomicAdd(&dK[kbase + d], ds*qv[r]); }
                if (want_dv && d < hsv) { atomicAdd(&dV[vbase + d], prob*dov[r]); }
            }
        }
    }
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * d = dst->src[5];
    const ggml_tensor * sinks = dst->src[6];
    const ggml_tensor * kv_idxs = dst->src[9];

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
    ggml_flash_attn_back_offsets(dst, nullptr, &off_k, &off_v, nullptr);

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
    // dK/dV accumulate with atomics and the scalar tail is unused; zero all of it.
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

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
#define FA_BACK_LAUNCH(REGS)                                                       \
    flash_attn_back_kernel<REGS><<<grid, block, smem, stream>>>(                     \
        (const char *) q->data, (const char *) k->data, (const char *) v->data,     \
        mask ? (const char *) mask->data : nullptr,                                 \
        (const char *) d->data, dst_d,                                              \
        hsk, hsv, nq, nkv, nhead, nheadk, nbatch, ratio,                            \
        q->nb[0], q->nb[1], q->nb[2], q->nb[3],                                     \
        k->nb[0], k->nb[1], k->nb[2], k->nb[3],                                     \
        v->nb[0], v->nb[1], v->nb[2], v->nb[3],                                     \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0,     \
        d->nb[0], d->nb[1], d->nb[2], d->nb[3],                                     \
        off_k, off_v, kv_f16, has_mask,                                             \
        want_dq, want_dk, want_dv, window, tkv, scale, softcap)

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
