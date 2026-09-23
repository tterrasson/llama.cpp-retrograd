// retro delta: the CPU kernels this fork adds to ggml, kept out of ops.cpp so
// upstream's file stays upstream's. Same headers as ops.cpp, same dispatch:
// ops.h declares these, ggml-cpu.c calls them, nothing else changes.
//
// Only whole added functions live here. Where the fork rewrites an upstream
// function in place -- gated delta net backward, flash attention backward --
// the code stays in ops.cpp, because there is no seam to cut along.
//
// Adding a CPU kernel to the fork belongs here.

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
    // single-threaded: B and C gradients accumulate across heads within a group,
    // and A gradients accumulate across sequences, so avoid cross-thread races here.
    if (params->ith != 0) {
        return;
    }

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

    // zero everything (g_A, g_B, g_C, g_s accumulate; g_x, g_dt are fully written)
    memset(dst->data, 0, ggml_nbytes(dst));

    // scratch reused across heads/sequences
    std::vector<float> traj((size_t) nt*nr*nc); // forward states: traj[t*nr*nc + p*nc + n]
    std::vector<float> dtsp(nt);                 // softplus(dt)
    std::vector<float> sig(nt);                  // softplus'(dt) = sigmoid(dt)
    std::vector<float> aA((size_t) nt*((nA0 == 1) ? 1 : nc)); // decay per token (per state for Mamba-1)
    std::vector<float> lam((size_t) nr*nc);      // state adjoint λ_t[p*nc + n]
    std::vector<float> gda(nc);                  // dL/d(dA) accumulated over head_dim

    for (int64_t i3 = 0; i3 < ns; ++i3) {
        const int64_t slot = ids[i3];
        const float * s0 = (const float *) ((const char *) src0->data + slot*src0->nb[3]);

        for (int64_t h = 0; h < nh; ++h) {
            const int64_t g = h / (nh / ng);

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


// ---- F16 AdamW with stochastic rounding ----

