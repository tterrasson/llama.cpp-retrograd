// retro delta: the CPU kernels this fork adds to ggml, kept out of ops.cpp so
// upstream's file stays upstream's. Same headers as ops.cpp, same dispatch:
// ops.h declares these, ggml-cpu.c calls them, nothing else changes.
//
// Only whole added functions live here. Where the fork rewrites an upstream
// function in place -- gated delta net backward, flash attention backward --
// the code stays in ops.cpp, because there is no seam to cut along.
//
// Adding a CPU kernel to the fork belongs here (docs/RETRO_FORK.md).

#include "ops.h"
#include "retro-ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "binary-ops.h"
#include "simd-gemm.h"
#include "ggml.h"
#include "unary-ops.h"
#include "vec.h"

#include <algorithm>
#include <numeric>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <vector>

// ---- l2_norm_back ----

// retro delta: analytic backward for GGML_OP_L2_NORM.
//
// forward: y = x * scale, scale = 1/max(norm, eps), norm = sqrt(sum(x*x))
//
// Unlike RMS norm, l2_norm floors the scale at 1/eps instead of adding eps
// under the sqrt, so the two regimes have different gradients:
//   - norm > eps: scale depends on x, so
//         dx = (dz - x * (sum(x*dz) / (norm*norm))) * scale
//   - norm <= eps: scale == 1/eps is locally constant (subgradient), so
//         dx = dz * scale
static void ggml_compute_forward_l2_norm_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src0 (x) from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const int64_t i11 = i01;
                const int64_t i12 = i02;
                const int64_t i13 = i03;

                const float * dz = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                const float * x  = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx = 0.0;
                ggml_float sum_xdz = 0.0;

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                float * dx = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                const float norm = sqrtf((float) sum_xx);

                if (norm > eps) {
                    const float scale   = 1.0f / norm;
                    const float scale_x = (float) (-sum_xdz) / (float) sum_xx;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dx[i00] = (dz[i00] + x[i00] * scale_x) * scale;
                    }
                } else {
                    const float scale = 1.0f / eps;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dx[i00] = dz[i00] * scale;
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_l2_norm_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ---- ssm_conv_back / ssm_scan_back ----

// ggml_compute_forward_ssm_conv_back

static void ggml_compute_forward_ssm_conv_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // sx {d_conv - 1 + n_t, d_inner, n_s}
    const ggml_tensor * src1 = dst->src[1]; // conv1d.weight {d_conv, d_inner}
    const ggml_tensor * src2 = dst->src[2]; // dy {d_inner, n_t, n_s} (grad of the conv output)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t d_conv  = src1->ne[0];
    const int64_t ncs     = src0->ne[0]; // d_conv - 1 + n_t
    const int64_t d_inner = src0->ne[1];
    const int64_t n_s     = src0->ne[2];
    const int64_t n_t     = src2->ne[1]; // tokens per sequence

    GGML_ASSERT(ncs == d_conv - 1 + n_t);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));

    // packed output: [ grad_sx (nelements(src0)) | grad_c (nelements(src1)) ]
    float * grad_sx = (float *) dst->data;                       // {ncs, d_inner, n_s}
    float * grad_c  = grad_sx + ggml_nelements(src0);            // {d_conv, d_inner}

    // channels per thread (each channel is independent, no cross-thread accumulation)
    const int64_t dc  = (d_inner + nth - 1)/nth;
    const int64_t ic0 = dc*ith;
    const int64_t ic1 = MIN(ic0 + dc, d_inner);

    for (int64_t ch = ic0; ch < ic1; ++ch) {
        const float * c = (const float *) ((const char *) src1->data + ch*src1->nb[1]); // {d_conv}

        // grad_c[k, ch] = sum_{t,s} dy[ch,t,s] * sx[k+t, ch, s]
        for (int64_t k = 0; k < d_conv; ++k) {
            float gck = 0.0f;
            for (int64_t s = 0; s < n_s; ++s) {
                const float * sx = (const float *) ((const char *) src0->data + ch*src0->nb[1] + s*src0->nb[2]);
                for (int64_t t = 0; t < n_t; ++t) {
                    const float dyv = *(const float *) ((const char *) src2->data +
                        ch*src2->nb[0] + t*src2->nb[1] + s*src2->nb[2]);
                    gck += dyv * sx[k + t];
                }
            }
            grad_c[k + ch*d_conv] = gck;
        }

        // grad_sx[j, ch, s] = sum_{k} dy[ch, j-k, s] * c[k, ch]   (valid taps only)
        for (int64_t s = 0; s < n_s; ++s) {
            float * gsx = grad_sx + ch*ncs + s*ncs*d_inner;
            for (int64_t j = 0; j < ncs; ++j) {
                const int64_t kmin = (j >= n_t) ? (j - (n_t - 1)) : 0;
                const int64_t kmax = (j < d_conv) ? j : (d_conv - 1);
                float g = 0.0f;
                for (int64_t k = kmin; k <= kmax; ++k) {
                    const int64_t t = j - k;
                    const float dyv = *(const float *) ((const char *) src2->data +
                        ch*src2->nb[0] + t*src2->nb[1] + s*src2->nb[2]);
                    g += dyv * c[k];
                }
                gsx[j] = g;
            }
        }
    }
}

void ggml_compute_forward_ssm_conv_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan_back

static void ggml_compute_forward_ssm_scan_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int ith = params->ith;
    const int nth = params->nth;

    const ggml_tensor * src0 = dst->src[0]; // s   {d_state, head_dim, n_head, n_kv}
    const ggml_tensor * src1 = dst->src[1]; // x   {head_dim, n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src2 = dst->src[2]; // dt  {n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src3 = dst->src[3]; // A   {d_state, n_head} or {1, n_head}
    const ggml_tensor * src4 = dst->src[4]; // B   {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src5 = dst->src[5]; // C   {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}
    const ggml_tensor * src7 = dst->src[7]; // ds  grad of the scan output (y ++ final states)

    const int64_t nc  = src0->ne[0]; // d_state
    const int64_t nr  = src1->ne[0]; // head_dim
    const int64_t nh  = src1->ne[1]; // n_head
    const int64_t ng  = src4->ne[1]; // n_group
    const int64_t nt  = src1->ne[2]; // tokens per sequence
    const int64_t ns  = src1->ne[3]; // number of sequences
    const int64_t nA0 = src3->ne[0]; // 1 (Mamba-2) or d_state (Mamba-1)

    const int32_t * ids = (const int32_t *) src6->data;

    const int64_t n_x = nr*nh*nt*ns; // element offset of the final states inside ds

    // packed output blocks: [ grad_x | grad_dt | grad_A | grad_B | grad_C | grad_s ]
    float * g_x  = (float *) dst->data;
    float * g_dt = g_x  + nr*nh*nt*ns;
    float * g_A  = g_dt + nh*nt*ns;
    float * g_B  = g_A  + nA0*nh;
    float * g_C  = g_B  + nc*ng*nt*ns;
    float * g_s  = g_C  + nc*ng*nt*ns;

    // Each state-space group owns disjoint B/C cells and a disjoint range of
    // heads (therefore disjoint x/dt/A/s cells). Partitioning by group keeps
    // every reduction in its original sequence/head/token order without
    // atomics or floating-point reassociation.
    if (ith == 0) {
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    // scratch reused across heads/sequences
    std::vector<float> traj((size_t) nt*nr*nc); // forward states: traj[t*nr*nc + p*nc + n]
    std::vector<float> dtsp(nt);                 // softplus(dt)
    std::vector<float> sig(nt);                  // softplus'(dt) = sigmoid(dt)
    std::vector<float> aA((size_t) nt*((nA0 == 1) ? 1 : nc)); // decay per token (per state for Mamba-1)
    std::vector<float> lam((size_t) nr*nc);      // state adjoint λ_t[p*nc + n]
    std::vector<float> gda(nc);                  // dL/d(dA) accumulated over head_dim

    const int64_t heads_per_group = nh / ng;
    for (int64_t g = ith; g < ng; g += nth) {
        for (int64_t i3 = 0; i3 < ns; ++i3) {
            const int64_t slot = ids[i3];
            const float * s0 = (const float *) ((const char *) src0->data + slot*src0->nb[3]);

            const int64_t h0 = g*heads_per_group;
            const int64_t h1 = h0 + heads_per_group;
            for (int64_t h = h0; h < h1; ++h) {

            // ---- forward recompute: fill traj, dtsp, sig, aA ----
            for (int64_t t = 0; t < nt; ++t) {
                const float * xt = (const float *) ((const char *) src1->data + t*src1->nb[2] + i3*src1->nb[3]);
                const float * Bt = (const float *) ((const char *) src4->data + t*src4->nb[2] + i3*src4->nb[3]);
                const float   dtv = *(const float *) ((const char *) src2->data + h*src2->nb[0] + t*src2->nb[1] + i3*src2->nb[2]);

                const float dsp = ggml_compute_softplus_f32(dtv);
                dtsp[t] = dsp;
                sig[t]  = (dtv > 20.0f) ? 1.0f : 1.0f/(1.0f + expf(-dtv));

                if (nA0 == 1) {
                    aA[t] = expf(dsp * ((const float *) src3->data)[h]);
                } else {
                    for (int64_t n = 0; n < nc; ++n) {
                        aA[t*nc + n] = expf(dsp * ((const float *) src3->data)[n + h*nc]);
                    }
                }

                for (int64_t p = 0; p < nr; ++p) {
                    const float x_dt = xt[p + h*nr] * dsp;
                    for (int64_t n = 0; n < nc; ++n) {
                        const float prev = (t == 0) ? s0[n + p*nc + h*nc*nr]
                                                    : traj[(t-1)*nr*nc + p*nc + n];
                        const float aval = (nA0 == 1) ? aA[t] : aA[t*nc + n];
                        traj[t*nr*nc + p*nc + n] = prev*aval + Bt[n + g*nc]*x_dt;
                    }
                }
            }

            // ---- reverse scan ----
            std::fill(lam.begin(), lam.end(), 0.0f);

            const float * dfinal = (const float *) ((const char *) src7->data) + n_x + i3*(nc*nr*nh) + h*nc*nr;

            for (int64_t t = nt - 1; t >= 0; --t) {
                const float * xt = (const float *) ((const char *) src1->data + t*src1->nb[2] + i3*src1->nb[3]);
                const float * Bt = (const float *) ((const char *) src4->data + t*src4->nb[2] + i3*src4->nb[3]);
                const float * Ct = (const float *) ((const char *) src5->data + t*src5->nb[2] + i3*src5->nb[3]);
                const float * dyt = (const float *) ((const char *) src7->data) + t*nh*nr + i3*nt*nh*nr;

                // λ_t[p,n] = dy_t[p]*C_t[n] + (t==last ? dfinal[p,n] : a_{t+1}*λ_{t+1}[p,n])
                for (int64_t p = 0; p < nr; ++p) {
                    const float dyv = dyt[p + h*nr];
                    for (int64_t n = 0; n < nc; ++n) {
                        float future;
                        if (t == nt - 1) {
                            future = dfinal[n + p*nc];
                        } else {
                            const float anext = (nA0 == 1) ? aA[(t+1)] : aA[(t+1)*nc + n];
                            future = anext * lam[p*nc + n];
                        }
                        lam[p*nc + n] = dyv*Ct[n + g*nc] + future;
                    }
                }

                // accumulate gradients using λ_t
                const float dsp = dtsp[t];
                for (int64_t n = 0; n < nc; ++n) {
                    gda[n] = 0.0f;
                }
                float b_path = 0.0f; // dL/d(dt_soft_plus) contribution from x_dt
                for (int64_t p = 0; p < nr; ++p) {
                    const float xp = xt[p + h*nr];
                    float gxp = 0.0f;
                    for (int64_t n = 0; n < nc; ++n) {
                        const float l = lam[p*nc + n];
                        gxp += l * Bt[n + g*nc];
                        g_B[n + g*nc + t*nc*ng + i3*nc*ng*nt] += l * xp * dsp;
                        const float prev = (t == 0) ? s0[n + p*nc + h*nc*nr]
                                                    : traj[(t-1)*nr*nc + p*nc + n];
                        gda[n] += l * prev;
                    }
                    g_x[p + h*nr + t*nr*nh + i3*nr*nh*nt] = dsp * gxp;
                    b_path += xp * gxp;
                }

                // grad_C[n] += sum_p dy_t[p] * state_t[n,p]
                for (int64_t n = 0; n < nc; ++n) {
                    float gc = 0.0f;
                    for (int64_t p = 0; p < nr; ++p) {
                        gc += dyt[p + h*nr] * traj[t*nr*nc + p*nc + n];
                    }
                    g_C[n + g*nc + t*nc*ng + i3*nc*ng*nt] += gc;
                }

                // dt (and A) gradients
                float a_path = 0.0f;
                if (nA0 == 1) {
                    float dLda = 0.0f;
                    for (int64_t n = 0; n < nc; ++n) {
                        dLda += gda[n];
                    }
                    const float Ah = ((const float *) src3->data)[h];
                    a_path = dLda * Ah * aA[t];
                    g_A[h] += dLda * dsp * aA[t];
                } else {
                    for (int64_t n = 0; n < nc; ++n) {
                        const float Anh = ((const float *) src3->data)[n + h*nc];
                        a_path += gda[n] * Anh * aA[t*nc + n];
                        g_A[n + h*nc] += gda[n] * dsp * aA[t*nc + n];
                    }
                }
                g_dt[h + t*nh + i3*nh*nt] = (a_path + b_path) * sig[t];
            }

            // grad_s0: state_0 = a_0*s0 + b_0  =>  dL/ds0 = a_0 * λ_0
            for (int64_t p = 0; p < nr; ++p) {
                for (int64_t n = 0; n < nc; ++n) {
                    const float a0 = (nA0 == 1) ? aA[0] : aA[n];
                    g_s[n + p*nc + h*nc*nr + slot*nc*nr*nh] += a0 * lam[p*nc + n];
                }
            }
            }
        }
    }
}

void ggml_compute_forward_ssm_scan_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ---- fused sparse cross-entropy ----

// ggml_compute_forward_fused_sparse_ce
//
// retro delta: fused sparse cross-entropy over a (possibly quantized) projection
// head. Streams the vocabulary tile by tile, computing each logit z[v,t] on
// demand as dot(w[:,v], h[:,t]) with an online (running) log-sum-exp in F32, so
// the [n_vocab, n_tokens] logits are never materialized. Reproduces exactly the
// weighted, active-row-normalized loss of ggml_cross_entropy_loss applied to
// mul_mat(w, h) with labels = weights[t]*onehot(targets[t]). See
// docs/memory/03-vocab-logits-chunked-ce.md.
//
// retro delta: an optional fixed per-vocab bias (src[4] forward / src[5]
// backward, may be NULL) is added to every z[v,t] before the log-sum-exp and
// target-logit terms, matching ADD(MUL_MAT(w,h), bias) output heads (e.g.
// gemma4's suppressed-token logits bias). The bias never receives a gradient.
//
// retro delta (plan rl/OPTIMIZE feature 1): op_params[0] (vocab tile count) and
// op_params[1] (seq_chunk, the flattened-token chunk size) are honored only by
// the CUDA kernel, where they bound the materialized logits intermediate. The
// CPU reference already streams one token and one vocab row at a time with only
// O(n_embd) scratch, so it is exact and invariant to both parameters and reads
// neither. See docs/rl/OPTIMIZE.md.

static inline void ggml_fused_ce_row_to_f32(
        const ggml_tensor * w, int64_t v, ggml_to_float_t to_float,
        float * scratch, const float ** out_row) {
    const void * w_row = (const char *) w->data + v*w->nb[1];
    if (w->type == GGML_TYPE_F32) {
        *out_row = (const float *) w_row;
    } else {
        to_float(w_row, scratch, w->ne[0]);
        *out_row = scratch;
    }
}

static void ggml_compute_forward_fused_sparse_ce_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * h       = dst->src[0]; // [n_embd, n_tokens] F32
    const ggml_tensor * w       = dst->src[1]; // [n_embd, n_vocab]  any type
    const ggml_tensor * targets = dst->src[2]; // [n_tokens] I32
    const ggml_tensor * weights = dst->src[3]; // [n_tokens] F32
    const ggml_tensor * bias    = dst->src[4]; // [n_vocab] F32, may be NULL

    GGML_ASSERT(ggml_is_scalar(dst) && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(h->type == GGML_TYPE_F32);
    GGML_ASSERT(targets->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    const int64_t n_embd   = h->ne[0];
    const int64_t n_tokens = h->ne[1];
    const int64_t n_vocab  = w->ne[1];
    GGML_ASSERT(w->ne[0] == n_embd);

    const int ith = params->ith;
    const int nth = params->nth;

    ggml_to_float_t const to_float = ggml_get_type_traits(w->type)->to_float;
    GGML_ASSERT(w->type == GGML_TYPE_F32 || to_float);

    GGML_ASSERT(params->wsize >= sizeof(float) * (size_t)(nth + nth*n_embd));
    float * sums = (float *) params->wdata;                 // [nth]
    float * deq  = (float *) params->wdata + nth + ith*n_embd; // per-thread [n_embd]

    const int32_t * tgt = (const int32_t *) targets->data;
    const float   * wts = (const float   *) weights->data;
    const float   * bs  = bias ? (const float *) bias->data : NULL;

    // Active tokens: real target and non-zero coefficient. Scanned per thread
    // (cheap over n_tokens) so no cross-thread reduction of the count is needed.
    int64_t n_active = 0;
    for (int64_t t = 0; t < n_tokens; ++t) {
        if (tgt[t] >= 0 && tgt[t] < n_vocab && wts[t] != 0.0f) {
            ++n_active;
        }
    }

    const int64_t dt = (n_tokens + nth - 1)/nth;
    const int64_t t0 = dt*ith;
    const int64_t t1 = MIN(t0 + dt, n_tokens);

    double sum_thread = 0.0;
    for (int64_t t = t0; t < t1; ++t) {
        if (!(tgt[t] >= 0 && tgt[t] < n_vocab && wts[t] != 0.0f)) {
            continue;
        }
        const float * h_col = (const float *)((const char *) h->data + t*h->nb[1]);
        float running_max = -INFINITY;
        float running_sum = 0.0f;
        float z_target    = 0.0f;
        for (int64_t v = 0; v < n_vocab; ++v) {
            const float * wv;
            ggml_fused_ce_row_to_f32(w, v, to_float, deq, &wv);
            float z = 0.0f;
            for (int64_t e = 0; e < n_embd; ++e) {
                z += wv[e]*h_col[e];
            }
            if (bs) { z += bs[v]; }
            if (z > running_max) {
                running_sum = running_sum*expf(running_max - z) + 1.0f;
                running_max = z;
            } else {
                running_sum += expf(z - running_max);
            }
            if (v == tgt[t]) {
                z_target = z;
            }
        }
        const float lse = running_max + logf(running_sum);
        sum_thread += (double) wts[t] * ((double) lse - (double) z_target);
    }
    sums[ith] = (float) sum_thread;
    ggml_barrier(params->threadpool);

    if (ith == 0) {
        double total = 0.0;
        for (int i = 0; i < nth; ++i) {
            total += sums[i];
        }
        ((float *) dst->data)[0] = n_active > 0 ? (float) (total / (double) n_active) : 0.0f;
    }
}

void ggml_compute_forward_fused_sparse_ce(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_fused_sparse_ce_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_fused_sparse_ce_back

static void ggml_compute_forward_fused_sparse_ce_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * grad    = dst->src[0]; // scalar gradient of the loss
    const ggml_tensor * h       = dst->src[1]; // [n_embd, n_tokens] F32
    const ggml_tensor * w       = dst->src[2]; // [n_embd, n_vocab]  any type
    const ggml_tensor * targets = dst->src[3]; // [n_tokens] I32
    const ggml_tensor * weights = dst->src[4]; // [n_tokens] F32
    const ggml_tensor * bias    = dst->src[5]; // [n_vocab] F32, may be NULL

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(h->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(h, dst));

    const int64_t n_embd   = h->ne[0];
    const int64_t n_tokens = h->ne[1];
    const int64_t n_vocab  = w->ne[1];
    GGML_ASSERT(w->ne[0] == n_embd);

    const int ith = params->ith;
    const int nth = params->nth;

    ggml_to_float_t const to_float = ggml_get_type_traits(w->type)->to_float;
    GGML_ASSERT(w->type == GGML_TYPE_F32 || to_float);

    // retro delta (plan rl/OPTIMIZE feature 3): with offload_h the allocator may
    // hand `dst` the very buffer of `h`, so grad_col and h_col alias. Copy the
    // column out before the first write; done unconditionally (O(n_embd) next to
    // the O(n_vocab*n_embd) body) so there is a single arithmetic path and the
    // CPU stays a bit-identical oracle whether or not the flag is set.
    GGML_ASSERT(params->wsize >= sizeof(float) * (size_t)(2*nth*n_embd));
    float * deq   = (float *) params->wdata + ith*n_embd;             // per-thread [n_embd]
    float * h_loc = (float *) params->wdata + (nth + ith)*n_embd;     // per-thread [n_embd]

    const int32_t * tgt = (const int32_t *) targets->data;
    const float   * wts = (const float   *) weights->data;
    const float   * bs  = bias ? (const float *) bias->data : NULL;
    const float     g   = ((const float *) grad->data)[0];

    int64_t n_active = 0;
    for (int64_t t = 0; t < n_tokens; ++t) {
        if (tgt[t] >= 0 && tgt[t] < n_vocab && wts[t] != 0.0f) {
            ++n_active;
        }
    }

    const int64_t dt = (n_tokens + nth - 1)/nth;
    const int64_t t0 = dt*ith;
    const int64_t t1 = MIN(t0 + dt, n_tokens);

    for (int64_t t = t0; t < t1; ++t) {
        float * grad_col = (float *)((char *) dst->data + t*dst->nb[1]);
        const bool active = tgt[t] >= 0 && tgt[t] < n_vocab && wts[t] != 0.0f;
        if (!active || n_active == 0) {
            for (int64_t e = 0; e < n_embd; ++e) {
                grad_col[e] = 0.0f;
            }
            continue;
        }
        memcpy(h_loc, (const char *) h->data + t*h->nb[1], n_embd*sizeof(float));
        const float * h_col = h_loc;

        // Pass 1: online log-sum-exp, identical to the forward.
        float running_max = -INFINITY;
        float running_sum = 0.0f;
        for (int64_t v = 0; v < n_vocab; ++v) {
            const float * wv;
            ggml_fused_ce_row_to_f32(w, v, to_float, deq, &wv);
            float z = 0.0f;
            for (int64_t e = 0; e < n_embd; ++e) {
                z += wv[e]*h_col[e];
            }
            if (bs) { z += bs[v]; }
            if (z > running_max) {
                running_sum = running_sum*expf(running_max - z) + 1.0f;
                running_max = z;
            } else {
                running_sum += expf(z - running_max);
            }
        }
        const float lse = running_max + logf(running_sum);

        // Pass 2: grad_h = coef * ( sum_v softmax[v]*w[:,v] - w[:,target] ),
        // coef = g * weight / n_active, recomputing logits tile by tile.
        for (int64_t e = 0; e < n_embd; ++e) {
            grad_col[e] = 0.0f;
        }
        for (int64_t v = 0; v < n_vocab; ++v) {
            const float * wv;
            ggml_fused_ce_row_to_f32(w, v, to_float, deq, &wv);
            float z = 0.0f;
            for (int64_t e = 0; e < n_embd; ++e) {
                z += wv[e]*h_col[e];
            }
            if (bs) { z += bs[v]; }
            const float p = expf(z - lse);
            for (int64_t e = 0; e < n_embd; ++e) {
                grad_col[e] += p*wv[e];
            }
        }
        const float * w_target;
        ggml_fused_ce_row_to_f32(w, tgt[t], to_float, deq, &w_target);
        const float coef = g * wts[t] / (float) n_active;
        for (int64_t e = 0; e < n_embd; ++e) {
            grad_col[e] = coef*(grad_col[e] - w_target[e]);
        }
    }
}

void ggml_compute_forward_fused_sparse_ce_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_fused_sparse_ce_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ---- F16 AdamW with stochastic rounding ----

void ggml_compute_forward_opt_step_adamw_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_tensor * w_tensor       = dst->src[0];
    const ggml_tensor * g_tensor = dst->src[1];
    ggml_tensor * m_tensor       = dst->src[2];
    ggml_tensor * v_tensor       = dst->src[3];
    const ggml_tensor * pars     = dst->src[4];

    GGML_ASSERT(w_tensor->type == GGML_TYPE_F16);
    GGML_ASSERT(g_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(m_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(v_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w_tensor));
    GGML_ASSERT(ggml_is_contiguous(g_tensor));
    GGML_ASSERT(ggml_is_contiguous(m_tensor));
    GGML_ASSERT(ggml_is_contiguous(v_tensor));
    GGML_ASSERT(ggml_nelements(pars) == 9);

    const int64_t n = ggml_nelements(w_tensor);
    const int64_t chunk = (n + params->nth - 1) / params->nth;
    const int64_t begin = chunk * params->ith;
    const int64_t end = MIN(begin + chunk, n);
    const float * p = ggml_get_data_f32(pars);
    const float alpha = p[0];
    const float beta1 = p[1];
    const float beta2 = p[2];
    const float eps = p[3];
    const float keep = 1.0f - alpha * p[4];
    const float beta1h = p[5];
    const float beta2h = p[6];
    const uint32_t seed = (uint32_t) p[7];
    const float gscale = p[8];

    ggml_fp16_t * w = static_cast<ggml_fp16_t *>(w_tensor->data);
    const float * g = static_cast<const float *>(g_tensor->data);
    float * m = static_cast<float *>(m_tensor->data);
    float * v = static_cast<float *>(v_tensor->data);
    for (int64_t i = begin; i < end; ++i) {
        const float gi = g[i] * gscale;
        m[i] = m[i] * beta1 + gi * (1.0f - beta1);
        v[i] = v[i] * beta2 + gi * gi * (1.0f - beta2);
        const float mh = m[i] * beta1h;
        const float vh = sqrtf(v[i] * beta2h) + eps;
        const float updated = ggml_fp16_to_fp32(w[i]) * keep - alpha * mh / vh;
        w[i] = ggml_fp32_to_fp16(
                ggml_stochastic_round_f16(updated, ggml_sr_uniform(seed, (uint32_t) i)));
    }
}
