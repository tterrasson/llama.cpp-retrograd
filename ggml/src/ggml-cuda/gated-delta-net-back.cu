#include "gated-delta-net-back.cuh"

#include <cstdint>

// retro delta: analytic backward for GGML_OP_GATED_DELTA_NET, mirroring the
// CPU reference (ggml-cpu/ops.cpp: ggml_compute_forward_gated_delta_net_back_f32).
// Correctness-first, like ssm-back.cu: one thread per (head, sequence) unit
// recomputes the S_prev trajectory forward, then reverse-scans it, exactly as
// the CPU does. grad_q/grad_k can be shared by several v-heads when q/k are
// GQA-broadcast (H_v % H_qk == 0), so those two outputs use atomicAdd; every
// other output (grad_v/grad_g/grad_beta/grad_state) is unique per (head, seq)
// and written directly.
//
// Memory note: like the CPU reference, each thread keeps the full per-token
// state trajectory (n_tokens * S_v * S_v floats) live at once, so the pool
// scratch buffer scales with n_tokens * S_v^2 * H * n_seqs. Fine for the LoRA
// ubatch sizes Retroback targets; a chunked/checkpointed backward would be
// needed for very long contexts.

static __global__ void gated_delta_net_back_kernel(
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
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= H * n_seqs) {
        return;
    }
    const int64_t iv1 = tid % H;
    const int64_t iv3 = tid / H;

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

    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    const float * grad_attn_base  = grad;
    const float * grad_state_base = grad + attn_score_elems;

    const float * s0 = state0 + iv3 * state_seq_stride + iv1 * S_v * S_v;

    float * base   = scratch + tid * scratch_stride;
    float * traj   = base;                       // n_tokens*S_v*S_v : S_prev per token
    float * S1     = traj + n_tokens * S_v * S_v; // S_v*S_v
    float * Snew   = S1 + S_v * S_v;              // S_v*S_v
    float * dS     = Snew + S_v * S_v;            // S_v*S_v
    float * dS1    = dS + S_v * S_v;              // S_v*S_v
    float * pre    = dS1 + S_v * S_v;             // S_v
    float * delta  = pre + S_v;                   // S_v
    float * gexp   = delta + S_v;                 // S_v
    float * dpre   = gexp + S_v;                  // S_v
    float * ddelta = dpre + S_v;                  // S_v

    // ---- forward recompute: fill the S_prev trajectory ----
    for (int64_t n = 0; n < S_v * S_v; ++n) {
        traj[n] = s0[n];
    }

    for (int64_t t = 0; t < n_tokens - 1; ++t) {
        const float * k_d = k + ik3 * sk3 + t * sk2 + ik1 * sk1;
        const float * v_d = v + iv3 * sv3 + t * sv2 + iv1 * sv1;
        const int64_t gb_off = iv3 * sb3 + t * sb2 + iv1 * sb1;
        const float beta_val = beta[gb_off];
        const float * g_d = g + gb_off * (kda ? S_v : 1);

        float * S_prev = traj + t * S_v * S_v;
        float * S_next = traj + (t + 1) * S_v * S_v;

        if (kda) {
            for (int64_t i = 0; i < S_v; ++i) { gexp[i] = expf(g_d[i]); }
        } else {
            const float gv = expf(g_d[0]);
            for (int64_t i = 0; i < S_v; ++i) { gexp[i] = gv; }
        }
        for (int64_t j = 0; j < S_v; ++j) {
            for (int64_t i = 0; i < S_v; ++i) {
                S1[i + j * S_v] = S_prev[i + j * S_v] * gexp[i];
            }
        }
        for (int64_t j = 0; j < S_v; ++j) {
            float sum = 0.0f;
            for (int64_t i = 0; i < S_v; ++i) { sum += S1[i + j * S_v] * k_d[i]; }
            delta[j] = (v_d[j] - sum) * beta_val;
        }
        for (int64_t j = 0; j < S_v; ++j) {
            for (int64_t i = 0; i < S_v; ++i) {
                S_next[i + j * S_v] = S1[i + j * S_v] + k_d[i] * delta[j];
            }
        }
    }

    // ---- reverse scan ----
    for (int64_t n = 0; n < S_v * S_v; ++n) { dS[n] = 0.0f; }

    for (int64_t t = n_tokens - 1; t >= 0; --t) {
        const float * q_d = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_d = k + ik3 * sk3 + t * sk2 + ik1 * sk1;
        const float * v_d = v + iv3 * sv3 + t * sv2 + iv1 * sv1;
        const int64_t gb_off = iv3 * sb3 + t * sb2 + iv1 * sb1;
        const float beta_val = beta[gb_off];
        const float * g_d = g + gb_off * (kda ? S_v : 1);

        const float * S_prev = traj + t * S_v * S_v;

        if (kda) {
            for (int64_t i = 0; i < S_v; ++i) { gexp[i] = expf(g_d[i]); }
        } else {
            const float gv = expf(g_d[0]);
            for (int64_t i = 0; i < S_v; ++i) { gexp[i] = gv; }
        }
        for (int64_t j = 0; j < S_v; ++j) {
            for (int64_t i = 0; i < S_v; ++i) {
                S1[i + j * S_v] = S_prev[i + j * S_v] * gexp[i];
            }
        }
        for (int64_t j = 0; j < S_v; ++j) {
            float sum = 0.0f;
            for (int64_t i = 0; i < S_v; ++i) { sum += S1[i + j * S_v] * k_d[i]; }
            pre[j]   = sum;
            delta[j] = (v_d[j] - sum) * beta_val;
        }
        for (int64_t j = 0; j < S_v; ++j) {
            for (int64_t i = 0; i < S_v; ++i) {
                Snew[i + j * S_v] = S1[i + j * S_v] + k_d[i] * delta[j];
            }
        }

        const float * d_out = grad_attn_base + (iv3 * n_tokens * H + t * H + iv1) * S_v;

        float * gq_out = g_q + S_v * (iq1 + neq1 * (t + n_tokens * iq3));
        for (int64_t i = 0; i < S_v; ++i) {
            float dqi = 0.0f;
            for (int64_t j = 0; j < S_v; ++j) {
                dS[i + j * S_v] += scale * d_out[j] * q_d[i];
                dqi += Snew[i + j * S_v] * d_out[j];
            }
            atomicAdd(&gq_out[i], scale * dqi);
        }

        const int64_t target_slot = n_tokens - 1 - t;
        if (target_slot < K) {
            const float * d_snap = grad_state_base + target_slot * state_size_per_snap
                    + (iv3 * H + iv1) * S_v * S_v;
            for (int64_t n = 0; n < S_v * S_v; ++n) {
                dS[n] += d_snap[n];
            }
        }

        // step 3 backward: S_new = S1 + outer(k, delta)
        for (int64_t n = 0; n < S_v * S_v; ++n) { dS1[n] = dS[n]; }
        for (int64_t j = 0; j < S_v; ++j) { ddelta[j] = 0.0f; }
        float * gk_out = g_k + S_v * (ik1 + nek1 * (t + n_tokens * ik3));
        for (int64_t i = 0; i < S_v; ++i) {
            float dki = 0.0f;
            for (int64_t j = 0; j < S_v; ++j) {
                dki       += dS[i + j * S_v] * delta[j];
                ddelta[j] += dS[i + j * S_v] * k_d[i];
            }
            atomicAdd(&gk_out[i], dki);
        }

        // step 2 backward: delta[j] = beta*(v[j] - pre[j])
        float dbeta_t = 0.0f;
        for (int64_t j = 0; j < S_v; ++j) { dpre[j] = -ddelta[j] * beta_val; }
        for (int64_t j = 0; j < S_v; ++j) {
            dbeta_t += ddelta[j] * (v_d[j] - pre[j]);
            g_v[j + S_v * (iv1 + H * (t + n_tokens * iv3))] += ddelta[j] * beta_val;
        }
        g_beta[iv1 + H * (t + n_tokens * iv3)] += dbeta_t;
        for (int64_t j = 0; j < S_v; ++j) {
            for (int64_t i = 0; i < S_v; ++i) {
                dS1[i + j * S_v] += dpre[j] * k_d[i];
            }
        }
        for (int64_t i = 0; i < S_v; ++i) {
            float dki2 = 0.0f;
            for (int64_t j = 0; j < S_v; ++j) { dki2 += dpre[j] * S1[i + j * S_v]; }
            atomicAdd(&gk_out[i], dki2);
        }

        // step 1 backward: S1[i,j] = S_prev[i,j] * gexp[i]
        if (kda) {
            float * gg_out = g_g + S_v * (iv1 + H * (t + n_tokens * iv3));
            for (int64_t i = 0; i < S_v; ++i) {
                float dgexp_i = 0.0f;
                for (int64_t j = 0; j < S_v; ++j) {
                    dgexp_i += dS1[i + j * S_v] * S_prev[i + j * S_v];
                    dS[i + j * S_v] = dS1[i + j * S_v] * gexp[i];
                }
                gg_out[i] += dgexp_i * gexp[i];
            }
        } else {
            float dgexp_sum = 0.0f;
            for (int64_t n = 0; n < S_v * S_v; ++n) { dgexp_sum += dS1[n] * S_prev[n]; }
            for (int64_t n = 0; n < S_v * S_v; ++n) { dS[n] = dS1[n] * gexp[0]; }
            g_g[iv1 + H * (t + n_tokens * iv3)] += dgexp_sum * gexp[0];
        }
    }

    float * gs_out = g_state + iv3 * state_seq_stride + iv1 * S_v * S_v;
    for (int64_t n = 0; n < S_v * S_v; ++n) { gs_out[n] += dS[n]; }
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

    const int64_t scratch_stride = n_tokens * S_v * S_v + 4 * S_v * S_v + 5 * S_v;
    const int64_t n_threads = H * n_seqs;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) (n_threads * scratch_stride));

    const int block = 32;
    const int grid = (int) ((n_threads + block - 1) / block);
    gated_delta_net_back_kernel<<<grid, block, 0, stream>>>(
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
