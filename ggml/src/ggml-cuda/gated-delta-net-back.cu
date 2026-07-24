#include "gated-delta-net-back.cuh"

#include <cstdint>

// retro delta: analytic backward for GGML_OP_GATED_DELTA_NET, mirroring the
// CPU reference (ggml-cpu/ops.cpp: ggml_compute_forward_gated_delta_net_back_f32).
// One *block* per (head, sequence) unit recomputes the S_prev trajectory
// forward, then reverse-scans it. The token scan is inherently sequential, but
// every step inside it is O(S_v^2) and is spread across the block: the state
// matrices are walked flat (coalesced, thread-stride), the reductions over the
// contiguous `i` axis use one warp per column `j`, and the reductions over `j`
// give each thread a whole row `i` (also coalesced, since threads then differ
// only in the contiguous index).
//
// An earlier revision ran one *thread* per (head, sequence). For a typical
// LoRA ubatch that is H_v*n_seqs threads in total -- 64 for Qwen3.5 at
// n_seqs=4 -- i.e. two warps for the whole GPU, each serially grinding
// n_tokens * S_v^2 scalar FLOPs with fully uncoalesced access. That kernel
// measured 3.7 s per launch and 99% of training wall-clock.
//
// grad_q/grad_k can be shared by several v-heads when q/k are GQA-broadcast
// (H_v % H_qk == 0), so those two outputs use atomicAdd; every other output
// (grad_v/grad_g/grad_beta/grad_state) is unique per (head, seq) and written
// directly.
//
// Memory note: like the CPU reference, each unit keeps the full per-token
// state trajectory (n_tokens * S_v * S_v floats) live at once, so the pool
// scratch buffer scales with n_tokens * S_v^2 * H * n_seqs. Fine for the LoRA
// ubatch sizes Retroback targets; a chunked/checkpointed backward would be
// needed for very long contexts.

#define GDN_BACK_BLOCK 256

// Sum `val` across the block. `red` holds one float per warp. Safe to call
// repeatedly: the trailing barrier keeps a later call from overwriting `red`
// while earlier readers are still using it.
static __device__ __forceinline__ float gdn_block_reduce_sum(float val, float * red) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        red[wid] = val;
    }
    __syncthreads();

    float total = 0.0f;
    for (int i = 0; i < nw; ++i) {
        total += red[i];
    }
    __syncthreads();
    return total;
}

// out[j] = sum_i M[i + j*S_v] * vec[i], one warp per column so the reduced
// axis is the contiguous one and every warp read is coalesced.
static __device__ __forceinline__ void gdn_col_dot(
        const float * __restrict__ M, const float * __restrict__ vec,
        float * __restrict__ out, const int64_t S_v) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

    for (int64_t j = wid; j < S_v; j += nw) {
        float acc = 0.0f;
        for (int64_t i = lane; i < S_v; i += WARP_SIZE) {
            acc += M[i + j*S_v] * vec[i];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            out[j] = acc;
        }
    }
}

static __global__ __launch_bounds__(GDN_BACK_BLOCK) void gated_delta_net_back_kernel(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta, const float * __restrict__ state0,
        const float * __restrict__ grad,
        float * __restrict__ dst,
        float * __restrict__ scratch, const int64_t scratch_stride,
        const int64_t S_v, const int64_t H, const int64_t n_tokens, const int64_t n_seqs,
        const int64_t neq1, const int64_t nek1, const int64_t rq3, const int64_t rk3,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const bool kda, const float scale, const int K,
        const int64_t n_q, const int64_t n_k, const int64_t n_v, const int64_t n_g, const int64_t n_beta,
        const int64_t state_seq_stride) {
    const int64_t unit = blockIdx.x;
    if (unit >= H * n_seqs) {
        return;
    }
    const int     tid  = threadIdx.x;
    const int     nthr = blockDim.x;

    const int64_t iv1 = unit % H;
    const int64_t iv3 = unit / H;

    const int64_t iq1 = iv1 % neq1;
    const int64_t ik1 = iv1 % nek1;
    const int64_t iq3 = iv3 / rq3;
    const int64_t ik3 = iv3 / rk3;

    float * g_q     = dst;
    float * g_k     = g_q + n_q;
    float * g_v     = g_k + n_k;
    float * g_g     = g_v + n_v;
    float * g_beta  = g_g + n_g;
    float * g_state = g_beta + n_beta;

    const int64_t SS                  = S_v * S_v;
    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = SS * H * n_seqs;
    const float * grad_attn_base  = grad;
    const float * grad_state_base = grad + attn_score_elems;

    const float * s0 = state0 + iv3 * state_seq_stride + iv1 * SS;

    float * base = scratch + unit * scratch_stride;
    float * traj = base;                  // n_tokens*S_v*S_v : S_prev per token
    float * S1   = traj + n_tokens * SS;  // S_v*S_v
    float * Snew = S1 + SS;               // S_v*S_v
    float * dS   = Snew + SS;             // S_v*S_v
    float * dS1  = dS + SS;               // S_v*S_v

    extern __shared__ float smem[];
    float * s_k      = smem;              // S_v
    float * s_v      = s_k + S_v;         // S_v
    float * s_q      = s_v + S_v;         // S_v
    float * s_do     = s_q + S_v;         // S_v
    float * s_gexp   = s_do + S_v;        // S_v
    float * s_pre    = s_gexp + S_v;      // S_v
    float * s_delta  = s_pre + S_v;       // S_v
    float * s_dpre   = s_delta + S_v;     // S_v
    float * s_ddelta = s_dpre + S_v;      // S_v
    float * s_red    = s_ddelta + S_v;    // blockDim.x / WARP_SIZE

    // Flat walk of an S_v*S_v matrix keeping (i, j) in step without a modulo
    // in the inner loop.
    const int64_t i0     = tid % S_v;
    const int64_t j0     = tid / S_v;
    const int64_t i_step = nthr % S_v;
    const int64_t j_step = nthr / S_v;

    // ---- forward recompute: fill the S_prev trajectory ----
    for (int64_t n = tid; n < SS; n += nthr) {
        traj[n] = s0[n];
    }
    __syncthreads();

    for (int64_t t = 0; t < n_tokens - 1; ++t) {
        const float * k_d = k + ik3 * sk3 + t * sk2 + ik1 * sk1;
        const float * v_d = v + iv3 * sv3 + t * sv2 + iv1 * sv1;
        const int64_t gb_off = iv3 * sb3 + t * sb2 + iv1 * sb1;
        const float beta_val = beta[gb_off];
        const float * g_d = g + gb_off * (kda ? S_v : 1);

        const float * S_prev = traj + t * SS;
        float *       S_next = traj + (t + 1) * SS;

        for (int64_t i = tid; i < S_v; i += nthr) {
            s_k[i]    = k_d[i];
            s_v[i]    = v_d[i];
            s_gexp[i] = expf(kda ? g_d[i] : g_d[0]);
        }
        __syncthreads();

        for (int64_t n = tid, i = i0; n < SS; n += nthr) {
            S1[n] = S_prev[n] * s_gexp[i];
            i += i_step;
            if (i >= S_v) i -= S_v;
        }
        __syncthreads();

        gdn_col_dot(S1, s_k, s_pre, S_v);
        __syncthreads();

        for (int64_t j = tid; j < S_v; j += nthr) {
            s_delta[j] = (s_v[j] - s_pre[j]) * beta_val;
        }
        __syncthreads();

        for (int64_t n = tid, i = i0, j = j0; n < SS; n += nthr) {
            S_next[n] = S1[n] + s_k[i] * s_delta[j];
            i += i_step;
            j += j_step;
            if (i >= S_v) { i -= S_v; ++j; }
        }
        __syncthreads();
    }

    // ---- reverse scan ----
    for (int64_t n = tid; n < SS; n += nthr) {
        dS[n] = 0.0f;
    }
    __syncthreads();

    for (int64_t t = n_tokens - 1; t >= 0; --t) {
        const float * q_d = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_d = k + ik3 * sk3 + t * sk2 + ik1 * sk1;
        const float * v_d = v + iv3 * sv3 + t * sv2 + iv1 * sv1;
        const int64_t gb_off = iv3 * sb3 + t * sb2 + iv1 * sb1;
        const float beta_val = beta[gb_off];
        const float * g_d = g + gb_off * (kda ? S_v : 1);

        const float * S_prev = traj + t * SS;
        const float * d_out  = grad_attn_base + (iv3 * n_tokens * H + t * H + iv1) * S_v;

        for (int64_t i = tid; i < S_v; i += nthr) {
            s_k[i]    = k_d[i];
            s_v[i]    = v_d[i];
            s_q[i]    = q_d[i];
            s_do[i]   = d_out[i];
            s_gexp[i] = expf(kda ? g_d[i] : g_d[0]);
        }
        __syncthreads();

        for (int64_t n = tid, i = i0; n < SS; n += nthr) {
            S1[n] = S_prev[n] * s_gexp[i];
            i += i_step;
            if (i >= S_v) i -= S_v;
        }
        __syncthreads();

        gdn_col_dot(S1, s_k, s_pre, S_v);
        __syncthreads();

        for (int64_t j = tid; j < S_v; j += nthr) {
            s_delta[j] = (s_v[j] - s_pre[j]) * beta_val;
        }
        __syncthreads();

        for (int64_t n = tid, i = i0, j = j0; n < SS; n += nthr) {
            Snew[n] = S1[n] + s_k[i] * s_delta[j];
            i += i_step;
            j += j_step;
            if (i >= S_v) { i -= S_v; ++j; }
        }
        __syncthreads();

        // dS += scale * outer(q, d_out); grad_q = scale * Snew . d_out
        float * gq_out = g_q + S_v * (iq1 + neq1 * (t + n_tokens * iq3));
        for (int64_t i = tid; i < S_v; i += nthr) {
            const float q_i = s_q[i];
            float dqi = 0.0f;
            for (int64_t j = 0; j < S_v; ++j) {
                dS[i + j*S_v] += scale * s_do[j] * q_i;
                dqi           += Snew[i + j*S_v] * s_do[j];
            }
            atomicAdd(&gq_out[i], scale * dqi);
        }
        __syncthreads();

        const int64_t target_slot = n_tokens - 1 - t;
        if (target_slot < K) {
            const float * d_snap = grad_state_base + target_slot * state_size_per_snap
                    + (iv3 * H + iv1) * SS;
            for (int64_t n = tid; n < SS; n += nthr) {
                dS[n] += d_snap[n];
            }
            __syncthreads();
        }

        // step 3 backward: S_new = S1 + outer(k, delta)
        for (int64_t n = tid; n < SS; n += nthr) {
            dS1[n] = dS[n];
        }
        gdn_col_dot(dS, s_k, s_ddelta, S_v);
        float * gk_out = g_k + S_v * (ik1 + nek1 * (t + n_tokens * ik3));
        for (int64_t i = tid; i < S_v; i += nthr) {
            float dki = 0.0f;
            for (int64_t j = 0; j < S_v; ++j) {
                dki += dS[i + j*S_v] * s_delta[j];
            }
            atomicAdd(&gk_out[i], dki);
        }
        __syncthreads();

        // step 2 backward: delta[j] = beta*(v[j] - pre[j])
        float dbeta_partial = 0.0f;
        for (int64_t j = tid; j < S_v; j += nthr) {
            const float dd = s_ddelta[j];
            s_dpre[j] = -dd * beta_val;
            dbeta_partial += dd * (s_v[j] - s_pre[j]);
            g_v[j + S_v * (iv1 + H * (t + n_tokens * iv3))] += dd * beta_val;
        }
        const float dbeta_t = gdn_block_reduce_sum(dbeta_partial, s_red);
        if (tid == 0) {
            g_beta[iv1 + H * (t + n_tokens * iv3)] += dbeta_t;
        }
        __syncthreads();

        for (int64_t n = tid, i = i0, j = j0; n < SS; n += nthr) {
            dS1[n] += s_dpre[j] * s_k[i];
            i += i_step;
            j += j_step;
            if (i >= S_v) { i -= S_v; ++j; }
        }
        __syncthreads();

        for (int64_t i = tid; i < S_v; i += nthr) {
            float dki2 = 0.0f;
            for (int64_t j = 0; j < S_v; ++j) {
                dki2 += s_dpre[j] * S1[i + j*S_v];
            }
            atomicAdd(&gk_out[i], dki2);
        }
        __syncthreads();

        // step 1 backward: S1[i,j] = S_prev[i,j] * gexp[i]
        if (kda) {
            float * gg_out = g_g + S_v * (iv1 + H * (t + n_tokens * iv3));
            for (int64_t i = tid; i < S_v; i += nthr) {
                const float ge = s_gexp[i];
                float dgexp_i = 0.0f;
                for (int64_t j = 0; j < S_v; ++j) {
                    const int64_t idx = i + j*S_v;
                    dgexp_i += dS1[idx] * S_prev[idx];
                    dS[idx]  = dS1[idx] * ge;
                }
                gg_out[i] += dgexp_i * ge;
            }
        } else {
            const float ge = s_gexp[0];
            float dgexp_partial = 0.0f;
            for (int64_t n = tid; n < SS; n += nthr) {
                dgexp_partial += dS1[n] * S_prev[n];
                dS[n] = dS1[n] * ge;
            }
            const float dgexp_sum = gdn_block_reduce_sum(dgexp_partial, s_red);
            if (tid == 0) {
                g_g[iv1 + H * (t + n_tokens * iv3)] += dgexp_sum * ge;
            }
        }
        __syncthreads();
    }

    float * gs_out = g_state + iv3 * state_seq_stride + iv1 * SS;
    for (int64_t n = tid; n < SS; n += nthr) {
        gs_out[n] += dS[n];
    }
}

void ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];
    const ggml_tensor * src_grad  = dst->src[6];

    for (const ggml_tensor * t : {src_q, src_k, src_v, src_g, src_beta, src_state, src_grad}) {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
    }
    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(ggml_is_contiguous(src_grad));

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    const bool kda = (src_g->ne[0] == S_v);
    const int K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, src_k, nb);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t rq3 = n_seqs / neq3;
    const int64_t rk3 = n_seqs / nek3;

    const float scale = 1.0f / sqrtf((float) S_v);

    const int64_t n_q    = ggml_nelements(src_q);
    const int64_t n_k    = ggml_nelements(src_k);
    const int64_t n_v    = ggml_nelements(src_v);
    const int64_t n_g    = ggml_nelements(src_g);
    const int64_t n_beta = ggml_nelements(src_beta);

    const int64_t state_seq_stride = src_state->nb[3] / sizeof(float);

    cudaStream_t stream = ctx.stream();
    float * dst_d = (float *) dst->data;
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

    // The trajectory dominates; S1/Snew/dS/dS1 are the four working matrices.
    // pre/delta/gexp/dpre/ddelta moved to shared memory.
    const int64_t scratch_stride = n_tokens * S_v * S_v + 4 * S_v * S_v;
    const int64_t n_units = H * n_seqs;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) (n_units * scratch_stride));

    const int block  = GDN_BACK_BLOCK;
    const int grid   = (int) n_units;
    const size_t shmem = (9 * (size_t) S_v + block / WARP_SIZE) * sizeof(float);
    gated_delta_net_back_kernel<<<grid, block, shmem, stream>>>(
        (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
        (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
        (const float *) src_grad->data, dst_d,
        scratch.get(), scratch_stride,
        S_v, H, n_tokens, n_seqs,
        neq1, nek1, rq3, rk3,
        nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
        nbk1 / sizeof(float), nbk2 / sizeof(float), nbk3 / sizeof(float),
        nbv1 / sizeof(float), nbv2 / sizeof(float), nbv3 / sizeof(float),
        nbb1 / sizeof(float), nbb2 / sizeof(float), nbb3 / sizeof(float),
        kda, scale, K,
        n_q, n_k, n_v, n_g, n_beta,
        state_seq_stride);
    CUDA_CHECK(cudaGetLastError());
}
