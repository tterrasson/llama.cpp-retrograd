#include "ssm-back.cuh"

#include <cstdint>

// retro delta: reference SSM backward kernels for recurrent LoRA training,
// mirroring the fork's CPU references (ggml-cpu/ops.cpp). Correctness-first:
// each independent unit (a conv channel; a scan (group, sequence, head) triple)
// is handled by one thread. Cross-unit reductions that the CPU keeps sequential
// per group (grad_A/B/C/s) use atomicAdd here; the probe tolerances account for
// the F32 reassociation.

// ---------------------------------------------------------------------------
// SSM_CONV_BACK: grad_c[k,ch] = sum_{t,s} dy[ch,t,s]*sx[k+t,ch,s]
//                grad_sx[j,ch,s] = sum_k dy[ch,j-k,s]*c[k,ch]  (valid taps)
// Layouts are contiguous: sx {ncs,d_inner,n_s}, c {d_conv,d_inner},
// dy {d_inner,n_t,n_s}; output [ grad_sx | grad_c ].
// ---------------------------------------------------------------------------
static __global__ void ssm_conv_back_kernel(
        const float * __restrict__ sx, const float * __restrict__ c,
        const float * __restrict__ dy, float * __restrict__ dst,
        const int64_t ncs, const int64_t d_conv, const int64_t d_inner,
        const int64_t n_t, const int64_t n_s) {
    const int64_t ch = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (ch >= d_inner) {
        return;
    }
    float * grad_sx = dst;
    float * grad_c  = dst + ncs*d_inner*n_s;

    // grad_c[k, ch]
    for (int64_t k = 0; k < d_conv; ++k) {
        float gck = 0.0f;
        for (int64_t s = 0; s < n_s; ++s) {
            const float * sxs = sx + ch*ncs + s*ncs*d_inner;
            for (int64_t t = 0; t < n_t; ++t) {
                const float dyv = dy[ch + t*d_inner + s*d_inner*n_t];
                gck += dyv * sxs[k + t];
            }
        }
        grad_c[k + ch*d_conv] = gck;
    }

    // grad_sx[j, ch, s]
    for (int64_t s = 0; s < n_s; ++s) {
        float * gsx = grad_sx + ch*ncs + s*ncs*d_inner;
        for (int64_t j = 0; j < ncs; ++j) {
            const int64_t kmin = (j >= n_t) ? (j - (n_t - 1)) : 0;
            const int64_t kmax = (j < d_conv) ? j : (d_conv - 1);
            float g = 0.0f;
            for (int64_t k = kmin; k <= kmax; ++k) {
                const int64_t t = j - k;
                const float dyv = dy[ch + t*d_inner + s*d_inner*n_t];
                g += dyv * c[k + ch*d_conv];
            }
            gsx[j] = g;
        }
    }
}

void ggml_cuda_ssm_conv_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sx = dst->src[0]; // {ncs, d_inner, n_s}
    const ggml_tensor * c  = dst->src[1]; // {d_conv, d_inner}
    const ggml_tensor * dy = dst->src[2]; // {d_inner, n_t, n_s}

    GGML_ASSERT(sx->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32 && dy->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(sx) && ggml_is_contiguous(c) && ggml_is_contiguous(dy));

    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];
    const int64_t ncs     = sx->ne[0];
    const int64_t n_s     = sx->ne[2];
    const int64_t n_t     = dy->ne[1];
    GGML_ASSERT(ncs == d_conv - 1 + n_t);

    cudaStream_t stream = ctx.stream();
    const int block = 64;
    const int grid = (int) ((d_inner + block - 1)/block);
    ssm_conv_back_kernel<<<grid, block, 0, stream>>>(
        (const float *) sx->data, (const float *) c->data, (const float *) dy->data,
        (float *) dst->data, ncs, d_conv, d_inner, n_t, n_s);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// SSM_SCAN_BACK: one thread per (group g, sequence i3, head h). Recomputes the
// forward state trajectory, then a reverse scan producing grad_x/dt/A/B/C/s.
// grad_x/grad_dt are unique per (h,t,i3) and written directly; grad_A/B/C/s are
// shared across heads-in-group or sequences-per-slot and use atomicAdd.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float ssm_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static __global__ void ssm_scan_back_kernel(
        const float * __restrict__ s0_all, const float * __restrict__ x_all,
        const float * __restrict__ dt_all, const float * __restrict__ A_all,
        const float * __restrict__ B_all, const float * __restrict__ C_all,
        const int32_t * __restrict__ ids, const float * __restrict__ ds_all,
        float * __restrict__ dst,
        float * __restrict__ scratch, const int64_t scratch_stride,
        const int64_t nc, const int64_t nr, const int64_t nh, const int64_t ng,
        const int64_t nt, const int64_t ns, const int64_t nA0,
        const int64_t heads_per_group) {
    const int64_t tid = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (tid >= nh*ns) {
        return;
    }
    const int64_t i3 = tid / nh;
    const int64_t h  = tid % nh;
    const int64_t g  = h / heads_per_group;
    const int64_t slot = ids[i3];

    // Packed output blocks.
    const int64_t n_x = nr*nh*nt*ns;
    float * g_x  = dst;
    float * g_dt = g_x  + nr*nh*nt*ns;
    float * g_A  = g_dt + nh*nt*ns;
    float * g_B  = g_A  + nA0*nh;
    float * g_C  = g_B  + nc*ng*nt*ns;
    float * g_s  = g_C  + nc*ng*nt*ns;

    // Per-thread scratch carve-out.
    float * base   = scratch + tid*scratch_stride;
    float * traj   = base;                       // nt*nr*nc
    float * aA     = traj + nt*nr*nc;             // nt*(nA0==1?1:nc)
    const int64_t aA_len = nt*((nA0 == 1) ? 1 : nc);
    float * dtsp   = aA + aA_len;                 // nt
    float * sig    = dtsp + nt;                   // nt
    float * lam    = sig + nt;                    // nr*nc
    float * gda    = lam + nr*nc;                 // nc

    const float * s0 = s0_all + slot*(nc*nr*nh);

    // ---- forward recompute ----
    for (int64_t t = 0; t < nt; ++t) {
        const float * xt = x_all + t*(nr*nh) + i3*(nr*nh*nt);
        const float * Bt = B_all + t*(nc*ng) + i3*(nc*ng*nt);
        const float dtv = dt_all[h + t*nh + i3*(nh*nt)];
        const float dsp = ssm_softplus(dtv);
        dtsp[t] = dsp;
        sig[t]  = (dtv > 20.0f) ? 1.0f : 1.0f/(1.0f + expf(-dtv));
        if (nA0 == 1) {
            aA[t] = expf(dsp * A_all[h]);
        } else {
            for (int64_t n = 0; n < nc; ++n) {
                aA[t*nc + n] = expf(dsp * A_all[n + h*nc]);
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
    for (int64_t i = 0; i < nr*nc; ++i) { lam[i] = 0.0f; }
    const float * dfinal = ds_all + n_x + i3*(nc*nr*nh) + h*nc*nr;

    for (int64_t t = nt - 1; t >= 0; --t) {
        const float * xt = x_all + t*(nr*nh) + i3*(nr*nh*nt);
        const float * Bt = B_all + t*(nc*ng) + i3*(nc*ng*nt);
        const float * Ct = C_all + t*(nc*ng) + i3*(nc*ng*nt);
        const float * dyt = ds_all + t*(nh*nr) + i3*(nt*nh*nr);

        for (int64_t p = 0; p < nr; ++p) {
            const float dyv = dyt[p + h*nr];
            for (int64_t n = 0; n < nc; ++n) {
                float future;
                if (t == nt - 1) {
                    future = dfinal[n + p*nc];
                } else {
                    const float anext = (nA0 == 1) ? aA[t+1] : aA[(t+1)*nc + n];
                    future = anext * lam[p*nc + n];
                }
                lam[p*nc + n] = dyv*Ct[n + g*nc] + future;
            }
        }

        const float dsp = dtsp[t];
        for (int64_t n = 0; n < nc; ++n) { gda[n] = 0.0f; }
        float b_path = 0.0f;
        for (int64_t p = 0; p < nr; ++p) {
            const float xp = xt[p + h*nr];
            float gxp = 0.0f;
            for (int64_t n = 0; n < nc; ++n) {
                const float l = lam[p*nc + n];
                gxp += l * Bt[n + g*nc];
                atomicAdd(&g_B[n + g*nc + t*nc*ng + i3*nc*ng*nt], l * xp * dsp);
                const float prev = (t == 0) ? s0[n + p*nc + h*nc*nr]
                                            : traj[(t-1)*nr*nc + p*nc + n];
                gda[n] += l * prev;
            }
            g_x[p + h*nr + t*nr*nh + i3*nr*nh*nt] = dsp * gxp;
            b_path += xp * gxp;
        }

        for (int64_t n = 0; n < nc; ++n) {
            float gc = 0.0f;
            for (int64_t p = 0; p < nr; ++p) {
                gc += dyt[p + h*nr] * traj[t*nr*nc + p*nc + n];
            }
            atomicAdd(&g_C[n + g*nc + t*nc*ng + i3*nc*ng*nt], gc);
        }

        float a_path = 0.0f;
        if (nA0 == 1) {
            float dLda = 0.0f;
            for (int64_t n = 0; n < nc; ++n) { dLda += gda[n]; }
            const float Ah = A_all[h];
            a_path = dLda * Ah * aA[t];
            atomicAdd(&g_A[h], dLda * dsp * aA[t]);
        } else {
            for (int64_t n = 0; n < nc; ++n) {
                const float Anh = A_all[n + h*nc];
                a_path += gda[n] * Anh * aA[t*nc + n];
                atomicAdd(&g_A[n + h*nc], gda[n] * dsp * aA[t*nc + n]);
            }
        }
        g_dt[h + t*nh + i3*nh*nt] = (a_path + b_path) * sig[t];
    }

    // grad_s0 = a_0 * λ_0
    for (int64_t p = 0; p < nr; ++p) {
        for (int64_t n = 0; n < nc; ++n) {
            const float a0 = (nA0 == 1) ? aA[0] : aA[n];
            atomicAdd(&g_s[n + p*nc + h*nc*nr + slot*nc*nr*nh], a0 * lam[p*nc + n]);
        }
    }
}

void ggml_cuda_ssm_scan_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * s   = dst->src[0]; // {nc, nr, nh, nslot}
    const ggml_tensor * x   = dst->src[1]; // {nr, nh, nt, ns}
    const ggml_tensor * dt  = dst->src[2]; // {nh, nt, ns}
    const ggml_tensor * A   = dst->src[3]; // {nA0, nh}
    const ggml_tensor * B   = dst->src[4]; // {nc, ng, nt, ns}
    const ggml_tensor * C   = dst->src[5]; // {nc, ng, nt, ns}
    const ggml_tensor * ids = dst->src[6]; // {ns} i32
    const ggml_tensor * ds  = dst->src[7]; // grad of scan output

    for (const ggml_tensor * t : {s, x, dt, A, B, C, ds}) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    }
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int64_t nc  = s->ne[0];
    const int64_t nr  = x->ne[0];
    const int64_t nh  = x->ne[1];
    const int64_t nt  = x->ne[2];
    const int64_t ns  = x->ne[3];
    const int64_t ng  = B->ne[1];
    const int64_t nA0 = A->ne[0];
    GGML_ASSERT(nh % ng == 0);
    const int64_t heads_per_group = nh / ng;

    cudaStream_t stream = ctx.stream();
    float * dst_d = (float *) dst->data;
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

    const int64_t aA_len = nt*((nA0 == 1) ? 1 : nc);
    const int64_t scratch_stride = nt*nr*nc + aA_len + 2*nt + nr*nc + nc;
    const int64_t n_threads = nh*ns;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) (n_threads*scratch_stride));

    const int block = 32;
    const int grid = (int) ((n_threads + block - 1)/block);
    ssm_scan_back_kernel<<<grid, block, 0, stream>>>(
        (const float *) s->data, (const float *) x->data, (const float *) dt->data,
        (const float *) A->data, (const float *) B->data, (const float *) C->data,
        (const int32_t *) ids->data, (const float *) ds->data, dst_d,
        scratch.get(), scratch_stride,
        nc, nr, nh, ng, nt, ns, nA0, heads_per_group);
    CUDA_CHECK(cudaGetLastError());
}
