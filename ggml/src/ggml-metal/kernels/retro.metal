#include "common.h"
#include "dequantize.h" // retro delta: quantized training kernels

// retro delta: RMS-norm backward for LoRA training.
// src0 = dz (grad of output), src1 = x (forward input), same shape. Per row:
//   sum_xx = sum x^2, sum_xdz = sum x*dz, rrms = 1/sqrt(sum_xx/n + eps)
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
// followed by grad_c. Every thread owns exactly one output element.
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
            const float dy = *(device const float *) (src2 + ch*args.nb20 + t*args.nb21 + s*args.nb22);
            const float c = *(device const float *) (src1 + k*args.nb10 + ch*args.nb11);
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
            const float sx = *(device const float *) (src0 + (k + t)*args.nb00 + ch*args.nb01 + s*args.nb02);
            const float dy = *(device const float *) (src2 + ch*args.nb20 + t*args.nb21 + s*args.nb22);
            acc += dy*sx;
        }
    }
    dst[gid] = acc;
}

// retro delta: SSM scan backward. One thread handles one (sequence, token,
// head, channel, state) tuple. It recomputes the scalar forward trajectory and
// reverse adjoint, then atomically contributes to the packed gradients.
kernel void kernel_ssm_scan_back_f32(
        constant ggml_metal_kargs_ssm_scan_back & args,
        device const char  * src0,
        device const char  * src1,
        device const char  * src2,
        device const char  * src3,
        device const char  * src4,
        device const char  * src5,
        device const char  * src6,
        device const float * src7,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    const int64_t total = args.d_state * args.head_dim * args.n_head * args.n_seq_tokens * args.n_seqs;
    if ((int64_t) gid >= total) {
        return;
    }

    int64_t r = gid;
    const int64_t n = r % args.d_state;
    r /= args.d_state;
    const int64_t p = r % args.head_dim;
    r /= args.head_dim;
    const int64_t h = r % args.n_head;
    r /= args.n_head;
    const int64_t t = r % args.n_seq_tokens;
    const int64_t s = r / args.n_seq_tokens;

    const int64_t g = h / (args.n_head / args.n_group);
    const int32_t slot = *(device const int32_t *) (src6 + s*args.nb60);
    const float A = *(device const float *) (src3 + (args.n_A0 == 1 ? 0 : n*args.nb30) + h*args.nb31);

    float state = *(device const float *) (src0 + n*args.nb00 + p*args.nb01 + h*args.nb02 + (int64_t) slot*args.nb03);
    float prev = state;
    for (int64_t u = 0; u <= t; ++u) {
        const float dtv = *(device const float *) (src2 + h*args.nb20 + u*args.nb21 + s*args.nb22);
        const float dsp = dtv <= 20.0f ? log(1.0f + exp(dtv)) : dtv;
        const float aval = exp(dsp*A);
        const float x = *(device const float *) (src1 + p*args.nb10 + h*args.nb11 + u*args.nb12 + s*args.nb13);
        const float B = *(device const float *) (src4 + n*args.nb40 + g*args.nb41 + u*args.nb42 + s*args.nb43);
        prev = state;
        state = prev*aval + B*x*dsp;
    }

    const int64_t n_x = args.off_dt;
    const int64_t final_off = n_x + s*(args.d_state*args.head_dim*args.n_head) + h*(args.d_state*args.head_dim) + p*args.d_state + n;
    float lambda = 0.0f;
    for (int64_t u = args.n_seq_tokens - 1; u >= t; --u) {
        const float dy = src7[p + h*args.head_dim + u*args.head_dim*args.n_head + s*args.n_seq_tokens*args.head_dim*args.n_head];
        const float C = *(device const float *) (src5 + n*args.nb50 + g*args.nb51 + u*args.nb52 + s*args.nb53);
        float future;
        if (u == args.n_seq_tokens - 1) {
            future = src7[final_off];
        } else {
            const float dt_next = *(device const float *) (src2 + h*args.nb20 + (u + 1)*args.nb21 + s*args.nb22);
            const float dsp_next = dt_next <= 20.0f ? log(1.0f + exp(dt_next)) : dt_next;
            future = exp(dsp_next*A)*lambda;
        }
        lambda = dy*C + future;
    }

    const float dtv = *(device const float *) (src2 + h*args.nb20 + t*args.nb21 + s*args.nb22);
    const float dsp = dtv <= 20.0f ? log(1.0f + exp(dtv)) : dtv;
    const float sig = dtv > 20.0f ? 1.0f : 1.0f/(1.0f + exp(-dtv));
    const float aval = exp(dsp*A);
    const float x = *(device const float *) (src1 + p*args.nb10 + h*args.nb11 + t*args.nb12 + s*args.nb13);
    const float B = *(device const float *) (src4 + n*args.nb40 + g*args.nb41 + t*args.nb42 + s*args.nb43);
    const float dy = src7[p + h*args.head_dim + t*args.head_dim*args.n_head + s*args.n_seq_tokens*args.head_dim*args.n_head];

    const int64_t ix  = p + h*args.head_dim + t*args.head_dim*args.n_head + s*args.n_seq_tokens*args.head_dim*args.n_head;
    const int64_t idt = args.off_dt + h + t*args.n_head + s*args.n_head*args.n_seq_tokens;
    const int64_t iA  = args.off_A + (args.n_A0 == 1 ? h : n + h*args.d_state);
    const int64_t iB  = args.off_B + n + g*args.d_state + t*args.d_state*args.n_group + s*args.d_state*args.n_group*args.n_seq_tokens;
    const int64_t iC  = args.off_C + n + g*args.d_state + t*args.d_state*args.n_group + s*args.d_state*args.n_group*args.n_seq_tokens;

    atomic_fetch_add_explicit((device atomic_float *) dst + ix, dsp*lambda*B, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float *) dst + idt, (lambda*prev*A*aval + x*lambda*B)*sig, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float *) dst + iA, lambda*prev*dsp*aval, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float *) dst + iB, lambda*x*dsp, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float *) dst + iC, dy*state, memory_order_relaxed);

    if (t == 0) {
        const int64_t is0 = args.off_s + n + p*args.d_state + h*args.d_state*args.head_dim + (int64_t) slot*args.d_state*args.head_dim*args.n_head;
        atomic_fetch_add_explicit((device atomic_float *) dst + is0, aval*lambda, memory_order_relaxed);
    }
}

// retro delta: out-prod (weight-gradient GEMM) for LoRA training.
// dst[i0,i1,i2,i3] = sum_k src0[i0,k,i02,i03] * src1[i1,k,i2,i3]
// with GQA broadcast i02 = i2/dps2, i03 = i3/dps3. Correctness-first: one thread
// per dst element, sequential reduction over the contraction dim.
kernel void kernel_out_prod_f32(
        constant ggml_metal_kargs_out_prod & args,
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    const int64_t total = args.ne0 * args.ne1 * args.ne2 * args.ne3;
    if ((int64_t) gid >= total) {
        return;
    }

    const int64_t i0 = (int64_t) gid % args.ne0;
    int64_t r        = (int64_t) gid / args.ne0;
    const int64_t i1 = r % args.ne1;
    r /= args.ne1;
    const int64_t i2 = r % args.ne2;
    const int64_t i3 = r / args.ne2;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;

    const int64_t off0 = i0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float acc = 0.0f;
    for (int64_t k = 0; k < args.ne01; ++k) {
        acc += src0[off0 + k*args.s01] * src1[off1 + k*args.s11];
    }

    dst[i0 + i1*args.s1 + i2*args.s2 + i3*args.s3] = acc;
}

// retro delta: out-prod with a Q8_0-quantized src0 (the activation-gradient
// dx = out_prod(W, dy) case, where W is a quantized model weight). Same flat
// dispatch as kernel_out_prod_f32, but src0 strides s01/s02/s03 are in bytes.
kernel void kernel_out_prod_q8_0(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    const int64_t total = args.ne0 * args.ne1 * args.ne2 * args.ne3;
    if ((int64_t) gid >= total) {
        return;
    }

    const int64_t i0 = (int64_t) gid % args.ne0;
    int64_t r        = (int64_t) gid / args.ne0;
    const int64_t i1 = r % args.ne1;
    r /= args.ne1;
    const int64_t i2 = r % args.ne2;
    const int64_t i3 = r / args.ne2;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;
    const int64_t ib = i0 / QK8_0;
    const short   iq = i0 % QK8_0;

    device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float acc = 0.0f;
    for (int64_t k = 0; k < args.ne01; ++k) {
        device const block_q8_0 * blk = (device const block_q8_0 *)(base0 + k*args.s01) + ib;
        acc += ((float) blk->d * blk->qs[iq]) * src1[off1 + k*args.s11];
    }

    dst[i0 + i1*args.s1 + i2*args.s2 + i3*args.s3] = acc;
}

// retro delta: activation-gradient path for legacy Q5_0 model weights.
kernel void kernel_out_prod_q5_0(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    const int64_t total = args.ne0 * args.ne1 * args.ne2 * args.ne3;
    if ((int64_t) gid >= total) {
        return;
    }

    const int64_t i0 = (int64_t) gid % args.ne0;
    int64_t r        = (int64_t) gid / args.ne0;
    const int64_t i1 = r % args.ne1;
    r /= args.ne1;
    const int64_t i2 = r % args.ne2;
    const int64_t i3 = r / args.ne2;
    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;
    const int64_t ib  = i0 / QK5_0;
    const short   iq  = i0 % QK5_0;
    const short   il  = iq & 15;

    device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;
    float acc = 0.0f;
    for (int64_t k = 0; k < args.ne01; ++k) {
        device const block_q5_0 * blk = (device const block_q5_0 *) (base0 + k*args.s01) + ib;
        const uint qh = *((device const uint *) blk->qh);
        const int low = iq < 16 ? (blk->qs[il] & 0x0f) : (blk->qs[il] >> 4);
        const int q = low | (int) (((qh >> iq) & 1u) << 4);
        acc += ((float) blk->d * (float) (q - 16)) * src1[off1 + k*args.s11];
    }
    dst[i0 + i1*args.s1 + i2*args.s2 + i3*args.s3] = acc;
}

// retro delta: K-quant out-prod specialization.
template<typename block_q, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_out_prod_k(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    const int64_t ne0_chunks = args.ne0 / 16;
    const int64_t total = ne0_chunks * args.ne1 * args.ne2 * args.ne3;
    if ((int64_t) gid >= total) {
        return;
    }

    const int64_t ic = (int64_t) gid % ne0_chunks;
    int64_t r        = (int64_t) gid / ne0_chunks;
    const int64_t i1 = r % args.ne1;
    r /= args.ne1;
    const int64_t i2 = r % args.ne2;
    const int64_t i3 = r / args.ne2;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;
    const int64_t i0  = ic * 16;
    const int64_t ib  = i0 / QK_K;
    const short   il  = (i0 % QK_K) / 16;

    device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
    const int64_t off1 = i1*args.s10 + i2*args.s12 + i3*args.s13;

    float4x4 acc(0.0f);
    for (int64_t k = 0; k < args.ne01; ++k) {
        float4x4 values;
        dequantize_func((device const block_q *)(base0 + k*args.s01) + ib, il, values);
        acc += values * src1[off1 + k*args.s11];
    }

    *((device float4x4 *)(dst + i0 + i1*args.s1 + i2*args.s2 + i3*args.s3)) = acc;
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
// contribution is atomically added to dst[0] (dst is zero-filled first).
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

    // A batch containing only ignored labels has a zero loss. Avoid dividing
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
    // for weighted labels, mirroring the CPU op. One-hot rows are unchanged.
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
