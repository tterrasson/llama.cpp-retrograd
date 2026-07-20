#include "common.h"
#include "dequantize.h" // retro delta: quantized training kernels
// retro delta: RMS-norm backward for LoRA training.
// src0 = dz (grad of output), src1 = x (forward input), same shape. Per row:
//   sum_xx = Σ x², sum_xdz = Σ x·dz, rrms = 1/sqrt(sum_xx/n + eps)
//   dx = (dz + x*(-sum_xdz/(sum_xx + eps*n))) * rrms
// One threadgroup per row; two simdgroup reductions (mirrors kernel_rms_norm).
kernel void kernel_rms_norm_back_f32(
        constant ggml_metal_kargs_rms_norm_back & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float sh_xx [32];
    threadgroup float sh_xdz[32];

    if (sgitg == 0) {
        sh_xx [tiisg] = 0.0f;
        sh_xdz[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float * dz = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float * x  = (device const float *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float * dx = (device       float *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sum_xx  = 0.0f;
    float sum_xdz = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float xv = x[i00];
        sum_xx  += xv * xv;
        sum_xdz += xv * dz[i00];
    }
    sum_xx  = simd_sum(sum_xx);
    sum_xdz = simd_sum(sum_xdz);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        sh_xx [sgitg] = sum_xx;
        sh_xdz[sgitg] = sum_xdz;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum_xx  = simd_sum(sh_xx [tiisg]);
    sum_xdz = simd_sum(sh_xdz[tiisg]);

    const float mean_eps = sum_xx / args.ne00 + args.eps;
    const float sum_eps  = sum_xx + args.eps * args.ne00;
    const float rrms     = 1.0f / sqrt(mean_eps);
    const float scale_x  = -sum_xdz / sum_eps;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
    }
}

// retro delta: SSM convolution backward. The packed output contains grad_sx
// followed by grad_c. Every thread owns exactly one output element, so neither
// a clearing pass nor atomics are needed.
kernel void kernel_ssm_conv_back_f32(
        constant ggml_metal_kargs_ssm_conv_back & args,
        device const char  * src0,
        device const char  * src1,
        device const char  * src2,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    const int64_t n_sx = args.ncs * args.d_inner * args.n_s;
    const int64_t n_c  = args.d_conv * args.d_inner;
    if ((int64_t) gid >= n_sx + n_c) {
        return;
    }

    if ((int64_t) gid < n_sx) {
        int64_t r = gid;
        const int64_t j  = r % args.ncs;
        r /= args.ncs;
        const int64_t ch = r % args.d_inner;
        const int64_t s  = r / args.d_inner;

        const int64_t kmin = j >= args.n_t ? j - (args.n_t - 1) : 0;
        const int64_t kmax = min(j, args.d_conv - 1);
        float acc = 0.0f;
        for (int64_t k = kmin; k <= kmax; ++k) {
            const int64_t t = j - k;
            const float dy = *(device const float *) (src2 +
                    ch*args.nb20 + t*args.nb21 + s*args.nb22);
            const float c = *(device const float *) (src1 +
                    k*args.nb10 + ch*args.nb11);
            acc += dy*c;
        }
        dst[gid] = acc;
        return;
    }

    int64_t r = (int64_t) gid - n_sx;
    const int64_t k  = r % args.d_conv;
    const int64_t ch = r / args.d_conv;
    float acc = 0.0f;
    for (int64_t s = 0; s < args.n_s; ++s) {
        for (int64_t t = 0; t < args.n_t; ++t) {
            const float sx = *(device const float *) (src0 +
                    (k + t)*args.nb00 + ch*args.nb01 + s*args.nb02);
            const float dy = *(device const float *) (src2 +
                    ch*args.nb20 + t*args.nb21 + s*args.nb22);
            acc += dy*sx;
        }
    }
    dst[gid] = acc;
}

// retro delta: SSM scan backward, pass 0. One threadgroup per (head,
// sequence) chain replays the forward recurrence once over the whole sequence
// and snapshots the state at every chunk boundary (checkpoints), so the grad
// pass can recompute each chunk independently. It also seeds the lambda carry
// with the gradient of the final state. Same sequential recurrence as the CPU
// reference: bit-identical states, O(T) work instead of the former O(T^2)
// per-element recomputation.
kernel void kernel_ssm_scan_back_ckpt_f32(
        constant ggml_metal_kargs_ssm_scan_back & args,
        device const char  * src0,
        device const char  * src1,
        device const char  * src2,
        device const char  * src3,
        device const char  * src4,
        device const char  * src6,
        device const float * src7,
        device       float * ckpt,
        device       float * carry,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]],
        uint2 ntg [[threads_per_threadgroup]]) {
    const uint tptg = ntg.x;
    const int64_t nc = args.d_state;
    const int64_t nr = args.head_dim;
    const int64_t nh = args.n_head;
    const int64_t nt = args.n_seq_tokens;
    const int64_t tc = args.tc;

    const int64_t h = tgpig.x;
    const int64_t s = tgpig.y;
    const int64_t g = h/(args.n_head/args.n_group);
    const int64_t cs = nc*nr;
    const int64_t ch = s*nh + h;

    const int32_t slot = *(device const int32_t *) (src6 + s*args.nb60);

    device float * ckpt_c = ckpt + ch*cs;   // chunk-major: [chunk][chain][cs]
    device float * carry_c = carry + ch*cs;

    for (int64_t pair = tid; pair < cs; pair += tptg) {
        const int64_t p = pair / nc;
        const int64_t n = pair % nc;

        const float A = *(device const float *) (src3 +
                (args.n_A0 == 1 ? 0 : n*args.nb30) + h*args.nb31);

        // the reverse adjoint of the last token starts from the gradient of
        // the final state
        carry_c[pair] = src7[args.off_dt + s*(cs*nh) + h*cs + pair];

        float state = *(device const float *) (src0 + n*args.nb00 +
                p*args.nb01 + h*args.nb02 + (int64_t) slot*args.nb03);

        for (int64_t t = 0; t < nt; ++t) {
            if (t % tc == 0) {
                ckpt_c[(t/tc)*nh*args.n_seqs*cs + pair] = state;
            }
            const float dtv = *(device const float *) (src2 +
                    h*args.nb20 + t*args.nb21 + s*args.nb22);
            const float dsp = dtv <= 20.0f ? log(1.0f + exp(dtv)) : dtv;
            const float x = *(device const float *) (src1 +
                    p*args.nb10 + h*args.nb11 + t*args.nb12 + s*args.nb13);
            const float B = *(device const float *) (src4 +
                    n*args.nb40 + g*args.nb41 + t*args.nb42 + s*args.nb43);
            state = state*exp(dsp*A) + B*x*dsp;
        }
    }
}

// retro delta: SSM scan backward, pass 1. One threadgroup per (head,
// sequence) chain, dispatched once per chunk of args.tc tokens from the last
// chunk to the first (encoder memory barriers carry the lambda between
// chunks). The threadgroup recomputes the chunk states from its checkpoint
// (same sequential recurrence, hence identical values), runs the reverse
// adjoint inside the chunk, then emits the packed gradients. Cross-pair sums
// are gathered with one writer per output cell (threadgroup atomic floats do
// not support fetch-add in MSL): grad_x/grad_dt are exclusive to the chain
// and written directly, grad_B/grad_C combine the per-cell partial across
// heads of a group through a single device atomic per cell and chunk.
kernel void kernel_ssm_scan_back_grad_f32(
        constant ggml_metal_kargs_ssm_scan_back & args,
        device const char  * src1,
        device const char  * src2,
        device const char  * src3,
        device const char  * src4,
        device const char  * src5,
        device const char  * src6,
        device const float * src7,
        device       float * dst,
        device       float * traj,
        device       float * lam,
        device const float * ckpt,
        device       float * carry,
        threadgroup  float * red [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]],
        uint2 ntg [[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint tptg = ntg.x;
    const int64_t nc = args.d_state;
    const int64_t nr = args.head_dim;
    const int64_t nh = args.n_head;
    const int64_t ng = args.n_group;
    const int64_t nt = args.n_seq_tokens;
    const int64_t tc = args.tc;
    const int64_t lo = args.chunk_lo;
    const int64_t hi = args.chunk_hi;

    const int64_t h = tgpig.x;
    const int64_t s = tgpig.y;
    const int64_t g = h/(nh/ng);
    const int64_t cs = nc*nr;
    const int64_t chains = nh*args.n_seqs;
    const int64_t ch = s*nh + h;
    const int64_t ich = lo/tc;

    const int32_t slot = *(device const int32_t *) (src6 + s*args.nb60);

    device       float * traj_c  = traj  + ch*cs;  // token-major: [tc][chain][cs]
    device       float * lam_c   = lam   + ch*cs;
    device const float * ckpt_c  = ckpt  + (ich*chains + ch)*cs;
    device       float * carry_c = carry + ch*cs;

    // phase A: recompute the chunk states from the checkpoint
    for (int64_t pair = tid; pair < cs; pair += tptg) {
        const int64_t p = pair / nc;
        const int64_t n = pair % nc;

        const float A = *(device const float *) (src3 +
                (args.n_A0 == 1 ? 0 : n*args.nb30) + h*args.nb31);

        float state = ckpt_c[pair];
        for (int64_t t = lo; t < hi; ++t) {
            const float dtv = *(device const float *) (src2 +
                    h*args.nb20 + t*args.nb21 + s*args.nb22);
            const float dsp = dtv <= 20.0f ? log(1.0f + exp(dtv)) : dtv;
            const float x = *(device const float *) (src1 +
                    p*args.nb10 + h*args.nb11 + t*args.nb12 + s*args.nb13);
            const float B = *(device const float *) (src4 +
                    n*args.nb40 + g*args.nb41 + t*args.nb42 + s*args.nb43);
            state = state*exp(dsp*A) + B*x*dsp;
            traj_c[(t - lo)*chains*cs + pair] = state;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // phase B: reverse adjoint over the chunk, seeded by the lambda carry of
    // the following chunk; per-pair A gradient and initial-state gradient
    for (int64_t pair = tid; pair < cs; pair += tptg) {
        const int64_t p = pair / nc;
        const int64_t n = pair % nc;

        const float A = *(device const float *) (src3 +
                (args.n_A0 == 1 ? 0 : n*args.nb30) + h*args.nb31);

        float f = carry_c[pair];
        float accA = 0.0f;
        for (int64_t t = hi - 1; t >= lo; --t) {
            const float dtv = *(device const float *) (src2 +
                    h*args.nb20 + t*args.nb21 + s*args.nb22);
            const float dsp = dtv <= 20.0f ? log(1.0f + exp(dtv)) : dtv;
            const float aval = exp(dsp*A);
            const float dy = src7[p + h*nr + t*nr*nh + s*nt*nr*nh];
            const float C = *(device const float *) (src5 +
                    n*args.nb50 + g*args.nb51 + t*args.nb52 + s*args.nb53);
            const float l = dy*C + f;
            lam_c[(t - lo)*chains*cs + pair] = l;
            const float prev = t == lo ? ckpt_c[pair]
                                       : traj_c[(t - 1 - lo)*chains*cs + pair];
            accA += l*prev*dsp*aval;
            f = aval*l;
            if (t == 0) {
                const int64_t is0 = args.off_s + n + p*nc + h*cs +
                                    (int64_t) slot*cs*nh;
                atomic_fetch_add_explicit((device atomic_float *) dst + is0,
                        f, memory_order_relaxed);
            }
        }
        carry_c[pair] = f;
        const int64_t iA = args.off_A + (args.n_A0 == 1 ? h : n + h*nc);
        atomic_fetch_add_explicit((device atomic_float *) dst + iA,
                accA, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // phase C: per-token outputs. grad_dt sums over every (n, p) pair, so each
    // thread accumulates a partial over its pairs which is then reduced across
    // simdgroups; grad_x/grad_B/grad_C are gathered per output cell with a
    // single writer each.
    const int64_t n_sg = (tptg + 31)/32;
    for (int64_t t = lo; t < hi; ++t) {
        const float dtv = *(device const float *) (src2 +
                h*args.nb20 + t*args.nb21 + s*args.nb22);
        const float dsp = dtv <= 20.0f ? log(1.0f + exp(dtv)) : dtv;
        const float sig = dtv > 20.0f ? 1.0f : 1.0f/(1.0f + exp(-dtv));
        const int64_t toff = (t - lo)*chains*cs;

        float part = 0.0f;
        for (int64_t pair = tid; pair < cs; pair += tptg) {
            const int64_t p = pair / nc;
            const int64_t n = pair % nc;

            const float A = *(device const float *) (src3 +
                    (args.n_A0 == 1 ? 0 : n*args.nb30) + h*args.nb31);
            const float aval = exp(dsp*A);
            const float x = *(device const float *) (src1 +
                    p*args.nb10 + h*args.nb11 + t*args.nb12 + s*args.nb13);
            const float B = *(device const float *) (src4 +
                    n*args.nb40 + g*args.nb41 + t*args.nb42 + s*args.nb43);
            const float l = lam_c[toff + pair];
            const float prev = t == lo ? ckpt_c[pair]
                                       : traj_c[toff - chains*cs + pair];
            part += l*prev*A*aval + x*l*B;
        }
        part = simd_sum(part);
        if (tiisg == 0) {
            red[sgitg] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float acc = 0.0f;
            for (int64_t i = 0; i < n_sg; ++i) {
                acc += red[i];
            }
            dst[args.off_dt + h + t*nh + s*nh*nt] = acc*sig;
        }

        for (int64_t i = tid; i < nr + 2*nc; i += tptg) {
            if (i < nr) {
                const int64_t p = i;
                float acc = 0.0f;
                for (int64_t n = 0; n < nc; ++n) {
                    const float B = *(device const float *) (src4 +
                            n*args.nb40 + g*args.nb41 + t*args.nb42 + s*args.nb43);
                    acc += lam_c[toff + p*nc + n]*B;
                }
                dst[p + h*nr + t*nr*nh + s*nt*nr*nh] = dsp*acc;
            } else if (i < nr + nc) {
                const int64_t n = i - nr;
                float acc = 0.0f;
                for (int64_t p = 0; p < nr; ++p) {
                    const float x = *(device const float *) (src1 +
                            p*args.nb10 + h*args.nb11 + t*args.nb12 + s*args.nb13);
                    acc += lam_c[toff + p*nc + n]*x;
                }
                atomic_fetch_add_explicit((device atomic_float *) dst +
                        args.off_B + n + g*nc + t*nc*ng + s*nc*ng*nt,
                        acc*dsp, memory_order_relaxed);
            } else {
                const int64_t n = i - nr - nc;
                float acc = 0.0f;
                for (int64_t p = 0; p < nr; ++p) {
                    const float dy = src7[p + h*nr + t*nr*nh + s*nt*nr*nh];
                    acc += dy*traj_c[toff + p*nc + n];
                }
                atomic_fetch_add_explicit((device atomic_float *) dst +
                        args.off_C + n + g*nc + t*nc*ng + s*nc*ng*nt,
                        acc, memory_order_relaxed);
            }
        }
        // red is reused by the next token's grad_dt reduction
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// retro delta: out-prod (weight-gradient GEMM) for LoRA training.
// dst[i0,i1,i2,i3] = Σ_k src0[i0,k,i02,i03] * src1[i1,k,i2,i3]
// with GQA broadcast i02 = i2/dps2, i03 = i3/dps3. Correctness-first: one thread
// per dst element, sequential reduction over the contraction dim.
kernel void kernel_out_prod_f32(
        constant ggml_metal_kargs_out_prod & args,
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tid[[thread_position_in_threadgroup]]) {
    constexpr int TILE = 8;
    threadgroup float tile0[TILE];
    threadgroup float tile1[TILE];
    const int64_t i0 = (int64_t) tgpig.x*TILE + tid.x;
    const int64_t i1 = (int64_t) tgpig.y*TILE + tid.y;
    const int64_t i2 = (int64_t) tgpig.z % args.ne2;
    const int64_t i3 = (int64_t) tgpig.z / args.ne2;
    const bool valid = i0 < args.ne0 && i1 < args.ne1 && i3 < args.ne3;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;

    const int64_t off0 = i0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float acc = 0.0f;
    for (int64_t k = 0; k < args.ne01; ++k) {
        if (tid.y == 0) {
            tile0[tid.x] = i0 < args.ne0 ? src0[off0 + k*args.s01] : 0.0f;
        }
        if (tid.x == 0) {
            tile1[tid.y] = i1 < args.ne1 ? src1[off1 + k*args.s11] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            acc += tile0[tid.x] * tile1[tid.y];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        dst[i0 + i1*args.s1 + i2*args.s2 + i3*args.s3] = acc;
    }
}

// retro delta: out-prod with a Q8_0-quantized src0 (the activation-gradient
// dx = out_prod(W, dy) case, where W is a quantized model weight). Same flat
// dispatch as kernel_out_prod_f32, but src0 strides s01/s02/s03 are in BYTES
// and each element is dequantized in place: within row k, element i0 lives in
// block i0/QK8_0 at lane i0%QK8_0, value d * qs.
kernel void kernel_out_prod_q8_0(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tid[[thread_position_in_threadgroup]]) {
    constexpr int TILE = 8;
    threadgroup float tile0[TILE];
    threadgroup float tile1[TILE];
    const int64_t i0 = (int64_t) tgpig.x*TILE + tid.x;
    const int64_t i1 = (int64_t) tgpig.y*TILE + tid.y;
    const int64_t i2 = (int64_t) tgpig.z % args.ne2;
    const int64_t i3 = (int64_t) tgpig.z / args.ne2;
    const bool valid = i0 < args.ne0 && i1 < args.ne1 && i3 < args.ne3;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;

    const int64_t ib = i0 / QK8_0; // block index within a src0 row
    const short   iq = i0 % QK8_0; // lane within the block

    device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float acc = 0.0f;
    for (int64_t k = 0; k < args.ne01; ++k) {
        if (tid.y == 0) {
            if (i0 < args.ne0) {
                device const block_q8_0 * blk = (device const block_q8_0 *)(base0 + k*args.s01) + ib;
                tile0[tid.x] = (float) blk->d * blk->qs[iq];
            } else {
                tile0[tid.x] = 0.0f;
            }
        }
        if (tid.x == 0) {
            tile1[tid.y] = i1 < args.ne1 ? src1[off1 + k*args.s11] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            acc += tile0[tid.x] * tile1[tid.y];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        dst[i0 + i1*args.s1 + i2*args.s2 + i3*args.s3] = acc;
    }
}

// Same activation-gradient path for legacy Q5_0 model weights. Q5_0 stores
// 32 signed five-bit values per block: low nibbles in qs and the fifth bits in
// qh, with a -16 zero point.
kernel void kernel_out_prod_q5_0(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tid[[thread_position_in_threadgroup]]) {
    constexpr int TILE = 8;
    threadgroup float tile0[TILE];
    threadgroup float tile1[TILE];
    const int64_t i0 = (int64_t) tgpig.x*TILE + tid.x;
    const int64_t i1 = (int64_t) tgpig.y*TILE + tid.y;
    const int64_t i2 = (int64_t) tgpig.z % args.ne2;
    const int64_t i3 = (int64_t) tgpig.z / args.ne2;
    const bool valid = i0 < args.ne0 && i1 < args.ne1 && i3 < args.ne3;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;
    const int64_t ib  = i0 / QK5_0;
    const short   iq  = i0 % QK5_0;
    const short   il  = iq & 15;

    device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float acc = 0.0f;
    for (int64_t k = 0; k < args.ne01; ++k) {
        if (tid.y == 0) {
            if (i0 < args.ne0) {
                device const block_q5_0 * blk = (device const block_q5_0 *) (base0 + k*args.s01) + ib;
                const uint qh = *((device const uint *) blk->qh);
                const int low = iq < 16 ? (blk->qs[il] & 0x0f) : (blk->qs[il] >> 4);
                const int q = low | (int) (((qh >> iq) & 1u) << 4);
                tile0[tid.x] = (float) blk->d * (float) (q - 16);
            } else {
                tile0[tid.x] = 0.0f;
            }
        }
        if (tid.x == 0) {
            tile1[tid.y] = i1 < args.ne1 ? src1[off1 + k*args.s11] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            acc += tile0[tid.x] * tile1[tid.y];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        dst[i0 + i1*args.s1 + i2*args.s2 + i3*args.s3] = acc;
    }
}

template<typename block_q, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_out_prod_k(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tid[[thread_position_in_threadgroup]]) {
    constexpr int TILE = 8;
    threadgroup float4x4 tile0[TILE];
    threadgroup float tile1[TILE];

    const int64_t ic = ((int64_t) tgpig.x*TILE + tid.x);
    const int64_t i1 =  (int64_t) tgpig.y*TILE + tid.y;
    const int64_t i2 =  (int64_t) tgpig.z % args.ne2;
    const int64_t i3 =  (int64_t) tgpig.z / args.ne2;
    const int64_t i0 = ic * 16;
    const bool valid0 = i0 < args.ne0;
    const bool valid = valid0 && i1 < args.ne1 && i3 < args.ne3;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;
    const int64_t ib  = i0 / QK_K;
    const short   il  = (i0 % QK_K) / 16;

    device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float4x4 acc(0.0f);
    for (int64_t k = 0; k < args.ne01; ++k) {
        // One row of the group cooperatively dequantizes eight adjacent
        // 16-value chunks. The other seven rows reuse those values for their
        // independent dst rows instead of repeating the same block decode.
        if (tid.y == 0) {
            float4x4 values(0.0f);
            if (valid0) {
                dequantize_func((device const block_q *)(base0 + k*args.s01) + ib, il, values);
            }
            tile0[tid.x] = values;
        }
        if (tid.x == 0) {
            tile1[tid.y] = i1 < args.ne1 ? src1[off1 + k*args.s11] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            acc += tile0[tid.x] * tile1[tid.y];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        *((device float4x4 *)(dst + i0 + i1*args.s1 + i2*args.s2 + i3*args.s3)) = acc;
    }
}

typedef decltype(kernel_out_prod_k<block_q2_K, dequantize_q2_K>) out_prod_k_t;

template [[host_name("kernel_out_prod_q2_K")]] kernel out_prod_k_t kernel_out_prod_k<block_q2_K, dequantize_q2_K>;
template [[host_name("kernel_out_prod_q3_K")]] kernel out_prod_k_t kernel_out_prod_k<block_q3_K, dequantize_q3_K>;
template [[host_name("kernel_out_prod_q4_K")]] kernel out_prod_k_t kernel_out_prod_k<block_q4_K, dequantize_q4_K>;
template [[host_name("kernel_out_prod_q5_K")]] kernel out_prod_k_t kernel_out_prod_k<block_q5_K, dequantize_q5_K>;
template [[host_name("kernel_out_prod_q6_K")]] kernel out_prod_k_t kernel_out_prod_k<block_q6_K, dequantize_q6_K>;

// retro delta: threadgroup-wide sum/max over per-simdgroup partials. Safe for any
// threadgroup size (including a partial trailing simdgroup): only simdgroup 0
// combines the partials, then the total is re-broadcast through shared memory.
// `sh` must hold 32 floats; all threads of the threadgroup must call this.
static float retro_tg_sum(float partial, threadgroup float * sh,
                          ushort sgitg, ushort tiisg, ushort ntg) {
    const ushort nsg = (ntg + 31) / 32;
    partial = simd_sum(partial);
    if (tiisg == 0) {
        sh[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float v = tiisg < nsg ? sh[tiisg] : 0.0f;
        v = simd_sum(v);
        if (tiisg == 0) {
            sh[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

static float retro_tg_max(float partial, threadgroup float * sh,
                          ushort sgitg, ushort tiisg, ushort ntg) {
    const ushort nsg = (ntg + 31) / 32;
    partial = simd_max(partial);
    if (tiisg == 0) {
        sh[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float v = tiisg < nsg ? sh[tiisg] : -INFINITY;
        v = simd_max(v);
        if (tiisg == 0) {
            sh[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// retro delta: soft-max backward for LoRA training.
// src0 = dy (grad of softmax output), src1 = y (softmax output), same shape.
// Per row: dx = (dy - dot(y, dy)) * y * scale. One threadgroup per row.
kernel void kernel_soft_max_back_f32(
        constant ggml_metal_kargs_soft_max_back & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float sh[32];

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float * dy = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float * y  = (device const float *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float * dx = (device       float *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float dot_y_dy = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dot_y_dy += y[i00] * dy[i00];
    }
    dot_y_dy = retro_tg_sum(dot_y_dy, sh, sgitg, tiisg, ntg.x);

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dx[i00] = (dy[i00] - dot_y_dy) * y[i00] * args.scale;
    }
}

// retro delta: flat F32 fill; zero-initialises accumulator outputs before the
// atomic-add stages of cross-entropy loss and get-rows backward.
kernel void kernel_retro_fill_f32(
        constant ggml_metal_kargs_retro_fill & args,
        device float * dst,
        uint gid[[thread_position_in_grid]]) {
    if ((int64_t) gid >= args.np) {
        return;
    }
    dst[gid] = args.val;
}

// retro delta: cross-entropy loss forward for LoRA training.
// src0 = logits, src1 = labels, both contiguous [ne00, nrows]; dst = scalar [1].
// One threadgroup per row: log-sum-exp over the row, then the row's loss
// contribution -Σ(label·log_softmax(logit))/nrows is atomically added to dst[0]
// (dst is zero-filled by a preceding dispatch).
kernel void kernel_cross_entropy_loss_f32(
        constant ggml_metal_kargs_cross_entropy_loss & args,
        device const float  * logits,
        device const float  * labels,
        device atomic_float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float sh[32];

    const int64_t i1 = tgpig.x;

    // A batch containing only ignored labels has a zero loss.  Avoid dividing
    // by zero while leaving the zero-filled destination unchanged.
    if (args.nactive == 0) {
        return;
    }

    device const float * s0 = logits + i1*args.ne00;
    device const float * s1 = labels + i1*args.ne00;

    float lmax = -INFINITY;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        lmax = MAX(lmax, s0[i00]);
    }
    const float max_val = retro_tg_max(lmax, sh, sgitg, tiisg, ntg.x);

    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        lsum += exp(s0[i00] - max_val);
    }
    const float log_sum = log(retro_tg_sum(lsum, sh, sgitg, tiisg, ntg.x));

    // loss contribution: Σ label·(logit - max - log_sum)
    float lloss = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        lloss += (s0[i00] - max_val - log_sum) * s1[i00];
    }
    const float loss = retro_tg_sum(lloss, sh, sgitg, tiisg, ntg.x);

    if (tpitg.x == 0) {
        atomic_fetch_add_explicit(dst, -loss / (float) args.nactive, memory_order_relaxed);
    }
}

// retro delta: cross-entropy loss backward for LoRA training.
// src0 = grad of the loss (scalar), src1 = logits, src2 = labels (contiguous
// [ne00, nrows]); dst = (softmax(logits) - labels) * grad/nrows, same shape.
kernel void kernel_cross_entropy_loss_back_f32(
        constant ggml_metal_kargs_cross_entropy_loss_back & args,
        device const float * grad,
        device const float * logits,
        device const float * labels,
        device       float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float sh[32];

    const int64_t i1 = tgpig.x;

    device const float * s0 = logits + i1*args.ne00;
    device const float * s1 = labels + i1*args.ne00;
    device       float * d  = dst    + i1*args.ne00;

    float lmax = -INFINITY;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        lmax = MAX(lmax, s0[i00]);
    }
    const float max_val = retro_tg_max(lmax, sh, sgitg, tiisg, ntg.x);

    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        lsum += exp(s0[i00] - max_val);
    }
    const float sum = retro_tg_sum(lsum, sh, sgitg, tiisg, ntg.x);

    const float sm_scale = 1.0f / sum;
    float label_sum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) label_sum += s1[i00];
    // The row's label mass scales the softmax term so the gradient stays exact
    // for weighted labels (Σlabels != 1), mirroring the CPU op:
    // dst = (Σlabels·softmax - labels) * grad / nactive. One-hot rows unchanged.
    const float label_mass = retro_tg_sum(label_sum, sh, sgitg, tiisg, ntg.x);
    const bool active = label_mass != 0.0f;
    const float d_by_nr  = active ? grad[0] / (float) args.nactive : 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float sm = exp(s0[i00] - max_val) * sm_scale;
        d[i00] = active ? (label_mass * sm - s1[i00]) * d_by_nr : 0.0f;
    }
}

// retro delta: get-rows backward for LoRA training.
// src0 = grad rows [ne00, nr], src1 = I32 row indices [nr]; dst [ne00, n_vocab]
// is zero-filled by a preceding dispatch, then each grad row is scatter-added
// into dst row idx[i]. Duplicate indices accumulate, hence the atomic add.
// One threadgroup per source row; threads stride the row.
kernel void kernel_get_rows_back_f32(
        constant ggml_metal_kargs_get_rows_back & args,
        device const char    * src0,
        device const int32_t * src1,
        device       char    * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int64_t i = tgpig.x;
    if (i >= args.nr) {
        return;
    }

    const int64_t r = src1[i];

    device const float  * s = (device const float  *) (src0 + i*args.nb01);
    device atomic_float * d = (device atomic_float *) (dst  + r*args.nb1);

    for (int64_t i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        atomic_fetch_add_explicit(d + i00, s[i00], memory_order_relaxed);
    }
}
