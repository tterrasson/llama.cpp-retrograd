#include "common.h"

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
