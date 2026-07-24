#include "ssm-back.cuh"

#include <cstdint>

// retro delta: SSM backward kernels for recurrent LoRA training, mirroring the
// fork's CPU references (ggml-cpu/ops.cpp). Each independent unit (a conv
// channel; a scan (sequence, head) pair) is handled by one *block*, with the
// work inside it spread across the block along whichever axis is contiguous for
// the tensor being touched -- see the per-kernel notes below. Cross-unit
// reductions that the CPU keeps sequential per group (grad_A/B/C/s) use
// atomicAdd here; the probe tolerances account for the F32 reassociation.

// Sum `val` across the block. `red` holds one float per warp. The trailing
// barrier makes repeated calls safe.
static __device__ __forceinline__ float ssm_block_reduce_sum(float val, float * red) {
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

// ---------------------------------------------------------------------------
// SSM_CONV_BACK: grad_c[k,ch] = sum_{t,s} dy[ch,t,s]*sx[k+t,ch,s]
//                grad_sx[j,ch,s] = sum_k dy[ch,j-k,s]*c[k,ch]  (valid taps)
// Layouts are contiguous: sx {ncs,d_inner,n_s}, c {d_conv,d_inner},
// dy {d_inner,n_t,n_s}; output [ grad_sx | grad_c ].
//
// One block per channel, parallel over the *time* axis. That is the axis along
// which sx and grad_sx are contiguous, so both the sx reads and the grad_sx
// writes coalesce across the block. dy is contiguous along the channel axis
// instead, so it is staged into shared memory once per (channel, sequence)
// rather than re-read d_conv+1 times with a d_inner stride.
//
// An earlier revision ran one thread per channel. Every sx read and grad_sx
// write then had an `ncs` stride between neighbouring threads (one memory
// transaction each), and a Falcon-H1 conv (d_inner+2*n_group*d_state = 896
// channels) filled only 14 blocks of 64 threads: 130 us per launch against
// 6 us for the corresponding forward.
// ---------------------------------------------------------------------------
static __global__ void ssm_conv_back_kernel(
        const float * __restrict__ sx, const float * __restrict__ c,
        const float * __restrict__ dy, float * __restrict__ dst,
        const int64_t ncs, const int64_t d_conv, const int64_t d_inner,
        const int64_t n_t, const int64_t n_s, const bool use_smem) {
    const int64_t ch   = blockIdx.x;
    const int     tid  = threadIdx.x;
    const int     nthr = blockDim.x;
    if (ch >= d_inner) {
        return;
    }
    float * grad_sx = dst;
    float * grad_c  = dst + ncs*d_inner*n_s;

    extern __shared__ float smem[];
    float * dy_sh = smem;                            // n_t (only if use_smem)
    float * gc    = dy_sh + (use_smem ? n_t : 0);    // d_conv
    float * red   = gc + d_conv;                     // blockDim.x / WARP_SIZE

    for (int64_t k = tid; k < d_conv; k += nthr) {
        gc[k] = 0.0f;
    }
    __syncthreads();

    for (int64_t s = 0; s < n_s; ++s) {
        const float * dy_s = dy + s*d_inner*n_t;
        if (use_smem) {
            for (int64_t t = tid; t < n_t; t += nthr) {
                dy_sh[t] = dy_s[ch + t*d_inner];
            }
            __syncthreads();
        }

        // grad_c[k, ch]: reduce over t (and accumulate across s in shared).
        const float * sxs = sx + ch*ncs + s*ncs*d_inner;
        for (int64_t k = 0; k < d_conv; ++k) {
            float part = 0.0f;
            for (int64_t t = tid; t < n_t; t += nthr) {
                const float dyv = use_smem ? dy_sh[t] : dy_s[ch + t*d_inner];
                part += dyv * sxs[k + t];
            }
            const float total = ssm_block_reduce_sum(part, red);
            if (tid == 0) {
                gc[k] += total;
            }
        }
        __syncthreads();

        // grad_sx[j, ch, s]: one thread per j, contiguous store.
        float * gsx = grad_sx + ch*ncs + s*ncs*d_inner;
        for (int64_t j = tid; j < ncs; j += nthr) {
            const int64_t kmin = (j >= n_t) ? (j - (n_t - 1)) : 0;
            const int64_t kmax = (j < d_conv) ? j : (d_conv - 1);
            float g = 0.0f;
            for (int64_t k = kmin; k <= kmax; ++k) {
                const int64_t t = j - k;
                const float dyv = use_smem ? dy_sh[t] : dy_s[ch + t*d_inner];
                g += dyv * c[k + ch*d_conv];
            }
            gsx[j] = g;
        }
        __syncthreads();
    }

    for (int64_t k = tid; k < d_conv; k += nthr) {
        grad_c[k + ch*d_conv] = gc[k];
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
    // One block per channel. dy is staged in shared memory when the time axis
    // fits the usual 48 KiB budget; past that the kernel reads it from global
    // (correct, just with the d_inner stride it had before).
    const int block = 256;
    const int grid  = (int) d_inner;
    const size_t smem_fixed = (size_t) (d_conv + block/WARP_SIZE)*sizeof(float);
    const bool use_smem = (size_t) n_t*sizeof(float) + smem_fixed <= 48*1024;
    const size_t smem = (use_smem ? (size_t) n_t*sizeof(float) : 0) + smem_fixed;
    ssm_conv_back_kernel<<<grid, block, smem, stream>>>(
        (const float *) sx->data, (const float *) c->data, (const float *) dy->data,
        (float *) dst->data, ncs, d_conv, d_inner, n_t, n_s, use_smem);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// SSM_SCAN_BACK: one *block* per (sequence i3, head h). Recomputes the forward
// state trajectory, then a reverse scan producing grad_x/dt/A/B/C/s.
// grad_x/grad_dt are unique per (h,t,i3) and written directly; grad_A/B/C/s are
// shared across heads-in-group or sequences-per-slot and use atomicAdd.
//
// The token scan is sequential, but each step is O(nr*nc) and is spread over the
// block. The state matrices are indexed [p*nc + n] with n contiguous, so:
//   - element-wise sweeps walk them flat, coalesced;
//   - reductions over n (grad_x's lam.B) get one warp per row p, so the reduced
//     axis is the contiguous one;
//   - reductions over p (grad_B, gda, grad_C) give each thread a column n, which
//     is also coalesced because neighbouring threads then differ only in n.
//
// grad_B additionally folds its sum over p *before* the atomic instead of doing
// one atomicAdd per (p, n): same value, nr times fewer atomics.
//
// An earlier revision ran one *thread* per (i3, h). At a Mamba2 training shape
// (nc=128, nr=64, nh=24, nt=256) that is 24 threads for the whole GPU, each
// serially walking nt*nr*nc = 2.1M state elements: 742 ms per call.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float ssm_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

// out[p] = sum_n M[p*nc + n] * vec[n], one warp per row.
static __device__ __forceinline__ void ssm_row_dot(
        const float * __restrict__ M, const float * __restrict__ vec,
        float * __restrict__ out, const int64_t nr, const int64_t nc) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

    for (int64_t p = wid; p < nr; p += nw) {
        float acc = 0.0f;
        for (int64_t n = lane; n < nc; n += WARP_SIZE) {
            acc += M[p*nc + n] * vec[n];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            out[p] = acc;
        }
    }
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
    const int64_t unit = blockIdx.x;
    if (unit >= nh*ns) {
        return;
    }
    const int     tid  = threadIdx.x;
    const int     nthr = blockDim.x;
    const int64_t i3 = unit / nh;
    const int64_t h  = unit % nh;
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

    // Per-unit scratch carve-out.
    float * base   = scratch + unit*scratch_stride;
    float * traj   = base;                        // nt*nr*nc
    float * aA     = traj + nt*nr*nc;             // nt*(nA0==1?1:nc)
    const int64_t aA_len = nt*((nA0 == 1) ? 1 : nc);
    float * dtsp   = aA + aA_len;                 // nt
    float * sig    = dtsp + nt;                   // nt
    float * lam    = sig + nt;                    // nr*nc

    extern __shared__ float smem[];
    float * s_B   = smem;             // nc
    float * s_C   = s_B + nc;         // nc
    float * s_x   = s_C + nc;         // nr
    float * s_dy  = s_x + nr;         // nr
    float * s_gxp = s_dy + nr;        // nr
    float * s_gda = s_gxp + nr;       // nc
    float * red   = s_gda + nc;       // blockDim.x / WARP_SIZE

    const float * s0 = s0_all + slot*(nc*nr*nh);
    const int64_t nrc = nr*nc;

    // Flat walk of [p*nc + n] keeping (p, n) in step without a modulo.
    const int64_t n0     = tid % nc;
    const int64_t p0     = tid / nc;
    const int64_t n_step = nthr % nc;
    const int64_t p_step = nthr / nc;

    // ---- forward recompute ----
    for (int64_t t = 0; t < nt; ++t) {
        const float * xt = x_all + t*(nr*nh) + i3*(nr*nh*nt);
        const float * Bt = B_all + t*(nc*ng) + i3*(nc*ng*nt);
        const float dtv = dt_all[h + t*nh + i3*(nh*nt)];
        const float dsp = ssm_softplus(dtv);
        if (tid == 0) {
            dtsp[t] = dsp;
            sig[t]  = (dtv > 20.0f) ? 1.0f : 1.0f/(1.0f + expf(-dtv));
        }
        for (int64_t n = tid; n < nc; n += nthr) {
            s_B[n] = Bt[n + g*nc];
        }
        for (int64_t p = tid; p < nr; p += nthr) {
            s_x[p] = xt[p + h*nr];
        }
        if (nA0 == 1) {
            if (tid == 0) { aA[t] = expf(dsp * A_all[h]); }
        } else {
            for (int64_t n = tid; n < nc; n += nthr) {
                aA[t*nc + n] = expf(dsp * A_all[n + h*nc]);
            }
        }
        __syncthreads();

        const float aval_scalar = (nA0 == 1) ? aA[t] : 0.0f;
        const float * traj_prev = (t == 0) ? traj : traj + (t-1)*nrc;
        float * traj_t = traj + t*nrc;
        for (int64_t i = tid, n = n0, p = p0; i < nrc; i += nthr) {
            const float prev = (t == 0) ? s0[i + h*nc*nr] : traj_prev[i];
            const float aval = (nA0 == 1) ? aval_scalar : aA[t*nc + n];
            traj_t[i] = prev*aval + s_B[n]*(s_x[p]*dsp);
            n += n_step;
            p += p_step;
            if (n >= nc) { n -= nc; ++p; }
        }
        __syncthreads();
    }

    // ---- reverse scan ----
    for (int64_t i = tid; i < nrc; i += nthr) { lam[i] = 0.0f; }
    const float * dfinal = ds_all + n_x + i3*(nc*nr*nh) + h*nc*nr;
    __syncthreads();

    for (int64_t t = nt - 1; t >= 0; --t) {
        const float * xt = x_all + t*(nr*nh) + i3*(nr*nh*nt);
        const float * Bt = B_all + t*(nc*ng) + i3*(nc*ng*nt);
        const float * Ct = C_all + t*(nc*ng) + i3*(nc*ng*nt);
        const float * dyt = ds_all + t*(nh*nr) + i3*(nt*nh*nr);

        for (int64_t n = tid; n < nc; n += nthr) {
            s_B[n]   = Bt[n + g*nc];
            s_C[n]   = Ct[n + g*nc];
            s_gda[n] = 0.0f;
        }
        for (int64_t p = tid; p < nr; p += nthr) {
            s_x[p]  = xt[p + h*nr];
            s_dy[p] = dyt[p + h*nr];
        }
        __syncthreads();

        const float dsp = dtsp[t];
        const float anext_scalar = (t == nt - 1 || nA0 != 1) ? 0.0f : aA[t+1];
        for (int64_t i = tid, n = n0, p = p0; i < nrc; i += nthr) {
            float future;
            if (t == nt - 1) {
                future = dfinal[i];
            } else {
                const float anext = (nA0 == 1) ? anext_scalar : aA[(t+1)*nc + n];
                future = anext * lam[i];
            }
            lam[i] = s_dy[p]*s_C[n] + future;
            n += n_step;
            p += p_step;
            if (n >= nc) { n -= nc; ++p; }
        }
        __syncthreads();

        // grad_x: reduce lam over n (contiguous) for each row p.
        ssm_row_dot(lam, s_B, s_gxp, nr, nc);
        __syncthreads();

        float b_part = 0.0f;
        for (int64_t p = tid; p < nr; p += nthr) {
            g_x[p + h*nr + t*nr*nh + i3*nr*nh*nt] = dsp * s_gxp[p];
            b_part += s_x[p] * s_gxp[p];
        }
        const float b_path = ssm_block_reduce_sum(b_part, red);

        // Reductions over p, one column n per thread.
        const float * traj_prev = (t == 0) ? traj : traj + (t-1)*nrc;
        const float * traj_t    = traj + t*nrc;
        for (int64_t n = tid; n < nc; n += nthr) {
            float gb = 0.0f;
            float gda = 0.0f;
            float gc = 0.0f;
            for (int64_t p = 0; p < nr; ++p) {
                const float l = lam[p*nc + n];
                gb   += l * s_x[p];
                gda  += l * ((t == 0) ? s0[n + p*nc + h*nc*nr] : traj_prev[p*nc + n]);
                gc   += s_dy[p] * traj_t[p*nc + n];
            }
            atomicAdd(&g_B[n + g*nc + t*nc*ng + i3*nc*ng*nt], gb * dsp);
            atomicAdd(&g_C[n + g*nc + t*nc*ng + i3*nc*ng*nt], gc);
            s_gda[n] = gda;
        }
        __syncthreads();

        float a_path;
        if (nA0 == 1) {
            float part = 0.0f;
            for (int64_t n = tid; n < nc; n += nthr) { part += s_gda[n]; }
            const float dLda = ssm_block_reduce_sum(part, red);
            const float Ah = A_all[h];
            a_path = dLda * Ah * aA[t];
            if (tid == 0) {
                atomicAdd(&g_A[h], dLda * dsp * aA[t]);
            }
        } else {
            float part = 0.0f;
            for (int64_t n = tid; n < nc; n += nthr) {
                const float Anh = A_all[n + h*nc];
                part += s_gda[n] * Anh * aA[t*nc + n];
                atomicAdd(&g_A[n + h*nc], s_gda[n] * dsp * aA[t*nc + n]);
            }
            a_path = ssm_block_reduce_sum(part, red);
        }
        if (tid == 0) {
            g_dt[h + t*nh + i3*nh*nt] = (a_path + b_path) * sig[t];
        }
        __syncthreads();
    }

    // grad_s0 = a_0 * λ_0
    for (int64_t i = tid, n = n0; i < nrc; i += nthr) {
        const float a0 = (nA0 == 1) ? aA[0] : aA[n];
        atomicAdd(&g_s[i + h*nc*nr + slot*nc*nr*nh], a0 * lam[i]);
        n += n_step;
        if (n >= nc) { n -= nc; }
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
    // gda moved to shared memory; the rest stays in the pool.
    const int64_t scratch_stride = nt*nr*nc + aA_len + 2*nt + nr*nc;
    const int64_t n_units = nh*ns;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) (n_units*scratch_stride));

    // One block per (sequence, head).
    const int block = 256;
    const int grid  = (int) n_units;
    const size_t smem = (2*(size_t) nc + 3*(size_t) nr + (size_t) nc + block/WARP_SIZE)*sizeof(float);
    ssm_scan_back_kernel<<<grid, block, smem, stream>>>(
        (const float *) s->data, (const float *) x->data, (const float *) dt->data,
        (const float *) A->data, (const float *) B->data, (const float *) C->data,
        (const int32_t *) ids->data, (const float *) ds->data, dst_d,
        scratch.get(), scratch_stride,
        nc, nr, nh, ng, nt, ns, nA0, heads_per_group);
    CUDA_CHECK(cudaGetLastError());
}
