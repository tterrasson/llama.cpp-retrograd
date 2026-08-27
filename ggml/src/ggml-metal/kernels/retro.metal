#include "common.h"
#include "dequantize.h" // retro delta: quantized training kernels
#include "ggml-retro-quant.h" // retro delta: shared fork dequantization table
// retro delta: repeat backward. dst is the smaller tensor; every element sums
// the src0 elements that ggml_repeat would have copied onto it, i.e. src0 is
// walked with a stride of the dst extent along each broadcast axis.
// Emitted by the autodiff of ADD/MUL/REPEAT when one input is broadcast.
//
// The nesting order is the CPU op's (outermost axis first, ascending), so the
// F32 reassociation matches the reference instead of merely being close to it.
kernel void kernel_repeat_back_f32(
        constant ggml_metal_kargs_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    device char * dst_ptr = dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        float acc = 0.0f;
        for (int k3 = i3; k3 < args.ne03; k3 += args.ne3) {
            for (int k2 = i2; k2 < args.ne02; k2 += args.ne2) {
                for (int k1 = i1; k1 < args.ne01; k1 += args.ne1) {
                    for (int k0 = i0; k0 < args.ne00; k0 += args.ne0) {
                        acc += *((device const float *)(src0 + k3*args.nb03 + k2*args.nb02 + k1*args.nb01 + k0*args.nb00));
                    }
                }
            }
        }
        *((device float *)(dst_ptr + i0*args.nb0)) = acc;
    }
}


// retro delta: streaming Flash Attention backward, ported line for line from the
// Vulkan shaders (flash_attn_back_{q,kv}.comp) and mirroring the CUDA kernels
// (flash-attn-back.cu). This is what makes an F16 KV cache differentiable
// (`kv_dtype = "f16"`), so it must agree with the analytic reference in
// retro_probe.cpp to the same tolerance the other two backends hold.
//
// Two passes over one threadgroup of 32 threads == one SIMD group, so every
// cross-thread reduction is a plain simd_sum:
//
//   pass q  -- one threadgroup per (query row, head, batch). Recomputes the row
//              logsumexp and delta = dot(dO, O) into the statistics segment of
//              the packed result, then accumulates dQ.
//   pass kv -- one threadgroup per (gradient-window row, kv head, batch). Reads
//              the statistics the q pass wrote and accumulates dK/dV.
//
// The q pass therefore always runs, even when dQ itself is not requested: the kv
// pass depends on its statistics. The caller inserts a memory barrier between
// the two dispatches.
//
// NSLOT is the per-thread head-dim accumulator depth: 32*NSLOT must cover both
// head dimensions, so 4 -> <= 128 and 8 -> <= 256. Only registers scale with the
// head dimension; narrow-head models keep the shallow variant. Mirrors the CUDA
// and Vulkan bucket dispatch.
//
// KVT is the cache element type. Both F16 and F32 are instantiated (like CUDA, and
// unlike Vulkan): `kv_dtype = "f16"` is only reached *through* the F32 capability
// probe in retro_backend.cpp, which builds the op with F32 K/V, so an F16-only
// kernel leaves the whole path dark.

// Bits 2..4 carry ggml_flash_attn_back_grad shifted left by 2.
#define RETRO_FA_BACK_FLAG_MASK   1
#define RETRO_FA_BACK_FLAG_SINKS  2
#define RETRO_FA_BACK_FLAG_GRAD_Q 4
#define RETRO_FA_BACK_FLAG_GRAD_K 8
#define RETRO_FA_BACK_FLAG_GRAD_V 16
#define RETRO_FA_BACK_FLAG_WINDOW 32

static inline float retro_fa_back_alibi_slope(
        constant ggml_metal_kargs_flash_attn_back & args,
        uint head) {
    if (args.max_bias <= 0.0f) {
        return 1.0f;
    }
    const uint n = 1u << uint(floor(log2(float(args.n_head))));
    const float base = head < n
        ? pow(2.0f, -args.max_bias / float(n))
        : pow(2.0f, -args.max_bias / (2.0f * float(n)));
    const int exponent = head < n ? int(head + 1u) : int(2u * (head - n) + 1u);
    return pow(base, float(exponent));
}

static inline float retro_fa_back_mask_value(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const half * data_mask,
        uint key, uint row, uint head, uint batch) {
    if ((args.flags & RETRO_FA_BACK_FLAG_MASK) == 0) {
        return 0.0f;
    }
    const uint mh = head  % (uint) args.mask_ne2;
    const uint mb = batch % (uint) args.mask_ne3;
    const uint index = key + (uint) args.KV*(row + (uint) args.mask_ne1*(mh + (uint) args.mask_ne2*mb));
    return (float) data_mask[index];
}

static inline float retro_fa_back_score_from_dot(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const half * data_mask,
        float dot, uint key, uint row, uint head, uint batch) {
    const float score = args.logit_softcap != 0.0f
        ? args.logit_softcap * precise::tanh(dot * args.scale / args.logit_softcap)
        : dot * args.scale;
    return score + retro_fa_back_alibi_slope(args, head)
        * retro_fa_back_mask_value(args, data_mask, key, row, head, batch);
}

static inline float retro_fa_back_score_derivative(
        constant ggml_metal_kargs_flash_attn_back & args,
        float dot) {
    if (args.logit_softcap == 0.0f) {
        return args.scale;
    }
    const float t = precise::tanh(dot * args.scale / args.logit_softcap);
    return args.scale * (1.0f - t * t);
}

template<typename KVT, short NSLOT>
kernel void kernel_flash_attn_back_q_impl(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const float * data_q     [[buffer(1)]],
        device const KVT   * data_k     [[buffer(2)]],
        device const KVT   * data_v     [[buffer(3)]],
        device const half  * data_mask  [[buffer(4)]],
        device const float * data_out   [[buffer(5)]],
        device const float * data_dout  [[buffer(6)]],
        device const float * data_sinks [[buffer(7)]],
        device       float * data_grad  [[buffer(8)]],
        device const int   * data_kvidx [[buffer(9)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    const uint lid   = tiisg;
    const uint row   = tgpig.x;
    const uint head  = tgpig.y;
    const uint batch = tgpig.z;
    if (row >= (uint) args.N || head >= (uint) args.n_head || batch >= (uint) args.n_batch) {
        return;
    }

    const uint HSK = (uint) args.HSK;
    const uint HSV = (uint) args.HSV;
    const uint KV  = (uint) args.KV;
    const uint N   = (uint) args.N;

    const uint kv_head = head / ((uint) args.n_head / (uint) args.n_head_kv);
    const uint q_base  = batch*(uint) args.q_nb3 + head*(uint) args.q_nb2 + row*(uint) args.q_nb1;
    const uint k_head_base = (batch*(uint) args.n_head_kv + kv_head)*KV*HSK;
    const uint v_head_base = (batch*(uint) args.n_head_kv + kv_head)*KV*HSV;
    const uint out_base   = ((batch*N + row)*(uint) args.n_head + head)*HSV;
    const uint stat_index = (batch*(uint) args.n_head + head)*N + row;

    float delta_part = 0.0f;
    for (uint id = lid; id < HSV; id += 32) {
        delta_part += data_dout[out_base + id] * data_out[out_base + id];
    }
    const float delta = simd_sum(delta_part);

    float row_max = -3.402823466e+38f;
    float row_sum = 0.0f;
    for (uint key = 0; key < KV; ++key) {
        float part = 0.0f;
        for (uint id = lid; id < HSK; id += 32) {
            part += data_q[q_base + id] * (float) data_k[k_head_base + key*HSK + id];
        }
        const float dot   = simd_sum(part);
        const float score = retro_fa_back_score_from_dot(args, data_mask, dot, key, row, head, batch);
        const float new_max = max(row_max, score);
        row_sum = row_sum * exp(row_max - new_max) + exp(score - new_max);
        row_max = new_max;
    }
    if ((args.flags & RETRO_FA_BACK_FLAG_SINKS) != 0) {
        const float score = data_sinks[head];
        const float new_max = max(row_max, score);
        row_sum = row_sum * exp(row_max - new_max) + exp(score - new_max);
        row_max = new_max;
    }

    const float lse = row_max + log(row_sum);
    if (lid == 0) {
        data_grad[args.off_stats + stat_index] = lse;
        data_grad[args.off_stats + N*(uint) args.n_head*(uint) args.n_batch + stat_index] = delta;
    }

    // The dK/dV pass consumes the statistics written above, so they are produced
    // unconditionally; the dQ accumulation below is the expensive part and is
    // skipped when the packed result has no dQ segment. The branch is uniform
    // across the threadgroup (kernel argument), so the simd_sum reductions inside
    // the loop stay well formed.
    if ((args.flags & RETRO_FA_BACK_FLAG_GRAD_Q) == 0) {
        return;
    }

    float dq[NSLOT];
    for (short slot = 0; slot < NSLOT; ++slot) {
        dq[slot] = 0.0f;
    }
    for (uint key = 0; key < KV; ++key) {
        float qk_part = 0.0f;
        for (uint id = lid; id < HSK; id += 32) {
            qk_part += data_q[q_base + id] * (float) data_k[k_head_base + key*HSK + id];
        }
        const float dot_qk = simd_sum(qk_part);

        float dv_part = 0.0f;
        for (uint id = lid; id < HSV; id += 32) {
            dv_part += data_dout[out_base + id] * (float) data_v[v_head_base + key*HSV + id];
        }
        const float dot_dv = simd_sum(dv_part);

        const float probability = exp(retro_fa_back_score_from_dot(args, data_mask, dot_qk, key, row, head, batch) - lse);
        const float ds = probability * (dot_dv - delta) * retro_fa_back_score_derivative(args, dot_qk);
        for (short slot = 0; slot < NSLOT; ++slot) {
            const uint id = lid + 32*slot;
            if (id < HSK) {
                dq[slot] += ds * (float) data_k[k_head_base + key*HSK + id];
            }
        }
    }

    for (short slot = 0; slot < NSLOT; ++slot) {
        const uint id = lid + 32*slot;
        if (id < HSK) {
            data_grad[args.off_q + (((batch*(uint) args.n_head + head)*N + row)*HSK + id)] = dq[slot];
        }
    }
}

typedef decltype(kernel_flash_attn_back_q_impl<half,  4>) kernel_flash_attn_back_q_f16_t;
typedef decltype(kernel_flash_attn_back_q_impl<float, 4>) kernel_flash_attn_back_q_f32_t;

template [[host_name("kernel_flash_attn_back_q_f32_f16_d128")]] kernel kernel_flash_attn_back_q_f16_t kernel_flash_attn_back_q_impl<half,  4>;
template [[host_name("kernel_flash_attn_back_q_f32_f16_d256")]] kernel kernel_flash_attn_back_q_f16_t kernel_flash_attn_back_q_impl<half,  8>;
template [[host_name("kernel_flash_attn_back_q_f32_f32_d128")]] kernel kernel_flash_attn_back_q_f32_t kernel_flash_attn_back_q_impl<float, 4>;
template [[host_name("kernel_flash_attn_back_q_f32_f32_d256")]] kernel kernel_flash_attn_back_q_f32_t kernel_flash_attn_back_q_impl<float, 8>;

template<typename KVT, short NSLOT>
kernel void kernel_flash_attn_back_kv_impl(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const float * data_q     [[buffer(1)]],
        device const KVT   * data_k     [[buffer(2)]],
        device const KVT   * data_v     [[buffer(3)]],
        device const half  * data_mask  [[buffer(4)]],
        device const float * data_out   [[buffer(5)]],
        device const float * data_dout  [[buffer(6)]],
        device const float * data_sinks [[buffer(7)]],
        device       float * data_grad  [[buffer(8)]],
        device const int   * data_kvidx [[buffer(9)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    const uint lid     = tiisg;
    const uint win_row = tgpig.x;
    const uint kv_head = tgpig.y;
    const uint batch   = tgpig.z;
    if (win_row >= (uint) args.KV_GRAD || kv_head >= (uint) args.n_head_kv || batch >= (uint) args.n_batch) {
        return;
    }

    const uint HSK = (uint) args.HSK;
    const uint HSV = (uint) args.HSV;
    const uint KV  = (uint) args.KV;
    const uint N   = (uint) args.N;
    const uint KVG = (uint) args.KV_GRAD;

    // retro delta: one threadgroup per *gradient window* row rather than per
    // cache key. Keys outside the window were written by earlier steps and are
    // constants; without a window KV_GRAD == KV and this is the dense case.
    uint key = win_row;
    bool in_cache = true;
    if ((args.flags & RETRO_FA_BACK_FLAG_WINDOW) != 0) {
        const int idx = data_kvidx[batch*KVG + win_row] - args.kv_stride*(args.kv_stream0 + (int) batch);
        in_cache = idx >= 0 && (uint) idx < KV;
        key = in_cache ? (uint) idx : 0;
    }

    const uint ratio = (uint) args.n_head / (uint) args.n_head_kv;
    // Read the cache at `key`, write the gradient at `win_row`.
    const uint k_base  = ((batch*(uint) args.n_head_kv + kv_head)*KV  + key)*HSK;
    const uint v_base  = ((batch*(uint) args.n_head_kv + kv_head)*KV  + key)*HSV;
    const uint dk_base = ((batch*(uint) args.n_head_kv + kv_head)*KVG + win_row)*HSK;
    const uint dv_base = ((batch*(uint) args.n_head_kv + kv_head)*KVG + win_row)*HSV;
    const uint n_stats = N*(uint) args.n_head*(uint) args.n_batch;

    float dk[NSLOT];
    float dv[NSLOT];
    for (short slot = 0; slot < NSLOT; ++slot) {
        dk[slot] = 0.0f;
        dv[slot] = 0.0f;
    }

    // A window row pointing outside the cache view carries no gradient, but its
    // slot in the packed result still has to be defined: fall through with zero
    // accumulators instead of returning early.
    for (uint rhead = 0; in_cache && rhead < ratio; ++rhead) {
        const uint head = kv_head*ratio + rhead;
        for (uint row = 0; row < N; ++row) {
            const uint q_base     = batch*(uint) args.q_nb3 + head*(uint) args.q_nb2 + row*(uint) args.q_nb1;
            const uint out_base   = ((batch*N + row)*(uint) args.n_head + head)*HSV;
            const uint stat_index = (batch*(uint) args.n_head + head)*N + row;

            float qk_part = 0.0f;
            for (uint id = lid; id < HSK; id += 32) {
                qk_part += data_q[q_base + id] * (float) data_k[k_base + id];
            }
            const float dot_qk = simd_sum(qk_part);

            float dv_part = 0.0f;
            for (uint id = lid; id < HSV; id += 32) {
                dv_part += data_dout[out_base + id] * (float) data_v[v_base + id];
            }
            const float dot_dv = simd_sum(dv_part);

            const float lse   = data_grad[args.off_stats + stat_index];
            const float delta = data_grad[args.off_stats + n_stats + stat_index];
            const float probability = exp(retro_fa_back_score_from_dot(args, data_mask, dot_qk, key, row, head, batch) - lse);
            const float ds = probability * (dot_dv - delta) * retro_fa_back_score_derivative(args, dot_qk);
            for (short slot = 0; slot < NSLOT; ++slot) {
                const uint id = lid + 32*slot;
                if (id < HSK) {
                    dk[slot] += ds * data_q[q_base + id];
                }
                if (id < HSV) {
                    dv[slot] += probability * data_dout[out_base + id];
                }
            }
        }
    }

    for (short slot = 0; slot < NSLOT; ++slot) {
        const uint id = lid + 32*slot;
        if ((args.flags & RETRO_FA_BACK_FLAG_GRAD_K) != 0 && id < HSK) {
            data_grad[args.off_k + dk_base + id] = dk[slot];
        }
        if ((args.flags & RETRO_FA_BACK_FLAG_GRAD_V) != 0 && id < HSV) {
            data_grad[args.off_v + dv_base + id] = dv[slot];
        }
    }
}

typedef decltype(kernel_flash_attn_back_kv_impl<half,  4>) kernel_flash_attn_back_kv_f16_t;
typedef decltype(kernel_flash_attn_back_kv_impl<float, 4>) kernel_flash_attn_back_kv_f32_t;

template [[host_name("kernel_flash_attn_back_kv_f32_f16_d128")]] kernel kernel_flash_attn_back_kv_f16_t kernel_flash_attn_back_kv_impl<half,  4>;
template [[host_name("kernel_flash_attn_back_kv_f32_f16_d256")]] kernel kernel_flash_attn_back_kv_f16_t kernel_flash_attn_back_kv_impl<half,  8>;
template [[host_name("kernel_flash_attn_back_kv_f32_f32_d128")]] kernel kernel_flash_attn_back_kv_f32_t kernel_flash_attn_back_kv_impl<float, 4>;
template [[host_name("kernel_flash_attn_back_kv_f32_f32_d256")]] kernel kernel_flash_attn_back_kv_f32_t kernel_flash_attn_back_kv_impl<float, 8>;

// retro delta: analytic backward for GATED_DELTA_NET (Qwen3-Next / KDA),
// mirroring the CPU reference (ggml-cpu/ops.cpp) and the CUDA/Vulkan ports
// (gated-delta-net-back.cu / gated_delta_net_back.comp). Correctness-first:
// one thread per (head, sequence) unit recomputes the S_prev trajectory into
// a scratch region appended after the packed destination buffer (sized by
// ggml_metal_op_gated_delta_net_back_extra_tmp), then reverse-scans it.
// grad_q/grad_k can be shared by several v-heads under GQA broadcast, so
// those two outputs use atomic_fetch_add_explicit; every other output is
// unique per (head, seq) and written directly. The caller zeroes `dst`
// before dispatch.
//
// One *threadgroup* per (head, sequence) unit. The token scan is inherently
// sequential, but every step inside it is O(S_v^2) and is spread across the
// threadgroup: the state matrices are walked flat (coalesced, thread-stride),
// the reductions over the contiguous `i` axis use one SIMD group per column
// `j`, and the reductions over `j` give each thread a whole row `i` (also
// coalesced, since threads then differ only in the contiguous index). The nine
// per-token vectors live in threadgroup memory, sized dynamically by the
// caller, so there is no cap on S_v.
//
// An earlier revision ran one *thread* per (head, sequence) -- 64 threads for
// Qwen3.5 at n_seqs=4, each serially grinding n_tokens * S_v^2 scalar FLOPs.
// That is the shape the CUDA port measured at 3.7 s per launch and 99% of
// training wall-clock before it was reparallelised the same way.
//
// Validated against the CPU reference on Apple M1 for scalar-gate/K=1 and
// KDA/K=3 cases (tests/metal_ops.rs).
kernel void kernel_gated_delta_net_back_f32(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const float * data_q     [[buffer(1)]],
        device const float * data_k     [[buffer(2)]],
        device const float * data_v     [[buffer(3)]],
        device const float * data_g     [[buffer(4)]],
        device const float * data_beta  [[buffer(5)]],
        device const float * data_state [[buffer(6)]],
        device const float * data_grad  [[buffer(7)]],
        device       float * data_dst   [[buffer(8)]],
        device       float * data_scratch [[buffer(9)]],
        threadgroup  float * smem       [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 ntg  [[threads_per_threadgroup]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  nsg  [[simdgroups_per_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    const uint unit = tgpig.x;
    if (unit >= (uint) (args.H*args.n_seqs)) {
        return;
    }
    const uint tid  = tpitg.x;
    const uint nthr = ntg.x;

    const uint iv1 = unit % (uint) args.H;
    const uint iv3 = unit / (uint) args.H;

    const uint iq1 = iv1 % (uint) args.neq1;
    const uint ik1 = iv1 % (uint) args.nek1;
    const uint iq3 = iv3 / (uint) args.rq3;
    const uint ik3 = iv3 / (uint) args.rk3;

    const uint S_v = (uint) args.S_v;
    const uint SS  = S_v*S_v;
    const uint H   = (uint) args.H;
    const uint n_tokens = (uint) args.n_tokens;
    const uint n_seqs   = (uint) args.n_seqs;
    const uint neq1 = (uint) args.neq1;
    const uint nek1 = (uint) args.nek1;
    const bool kda  = args.kda != 0;

    const uint n_q    = S_v * neq1 * n_tokens * (n_seqs / (uint) args.rq3);
    const uint n_k    = S_v * nek1 * n_tokens * (n_seqs / (uint) args.rk3);
    const uint n_v    = S_v * H * n_tokens * n_seqs;
    const uint n_g    = (kda ? S_v : 1u) * H * n_tokens * n_seqs;
    const uint n_beta = H * n_tokens * n_seqs;

    const uint attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const uint state_size_per_snap = SS * H * n_seqs;
    const uint state_seq_stride    = SS * H;

    const uint g_q_off     = 0u;
    const uint g_k_off     = g_q_off + n_q;
    const uint g_v_off     = g_k_off + n_k;
    const uint g_g_off     = g_v_off + n_v;
    const uint g_beta_off  = g_g_off + n_g;
    const uint g_state_off = g_beta_off + n_beta;

    // The trajectory dominates; S1/Snew/dS/dS1 are the four working matrices.
    // pre/delta/gexp/dpre/ddelta moved to threadgroup memory.
    const uint traj_stride = n_tokens * SS;
    const uint per_unit    = traj_stride + 4u*SS;
    const uint base = unit * per_unit;
    const uint traj = base;
    const uint S1   = traj + traj_stride;
    const uint Snew = S1 + SS;
    const uint dS   = Snew + SS;
    const uint dS1  = dS + SS;

    threadgroup float * s_k      = smem;
    threadgroup float * s_v      = s_k + S_v;
    threadgroup float * s_q      = s_v + S_v;
    threadgroup float * s_do     = s_q + S_v;
    threadgroup float * s_gexp   = s_do + S_v;
    threadgroup float * s_pre    = s_gexp + S_v;
    threadgroup float * s_delta  = s_pre + S_v;
    threadgroup float * s_dpre   = s_delta + S_v;
    threadgroup float * s_ddelta = s_dpre + S_v;
    threadgroup float * s_red    = s_ddelta + S_v;

    // Flat walk of an S_v*S_v matrix keeping (i, j) in step without a modulo in
    // the inner loop. i_step is nthr % S_v, so it is always < S_v and a single
    // correction per step suffices.
    const uint i0     = tid % S_v;
    const uint j0     = tid / S_v;
    const uint i_step = nthr % S_v;
    const uint j_step = nthr / S_v;

    device atomic_float * atomic_dst = (device atomic_float *) data_dst;

    // ---- forward recompute: fill the S_prev trajectory ----
    const uint s0 = iv3*state_seq_stride + iv1*SS;
    for (uint n = tid; n < SS; n += nthr) {
        data_scratch[traj + n] = data_state[s0 + n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint t = 0u; t + 1u < n_tokens; ++t) {
        const uint k_off  = ik3*(uint)args.sk3 + t*(uint)args.sk2 + ik1*(uint)args.sk1;
        const uint v_off  = iv3*(uint)args.sv3 + t*(uint)args.sv2 + iv1*(uint)args.sv1;
        const uint gb_off = iv3*(uint)args.sb3 + t*(uint)args.sb2 + iv1*(uint)args.sb1;
        const float beta_val = data_beta[gb_off];
        const uint g_off = gb_off * (kda ? S_v : 1u);

        const uint S_prev = traj + t*SS;
        const uint S_next = traj + (t+1u)*SS;

        for (uint i = tid; i < S_v; i += nthr) {
            s_k[i]    = data_k[k_off + i];
            s_v[i]    = data_v[v_off + i];
            s_gexp[i] = exp(data_g[g_off + (kda ? i : 0u)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint n = tid, i = i0; n < SS; n += nthr) {
            data_scratch[S1 + n] = data_scratch[S_prev + n] * s_gexp[i];
            i += i_step;
            if (i >= S_v) { i -= S_v; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // s_pre[j] = sum_i S1[i + j*S_v] * s_k[i], one SIMD group per column.
        for (uint j = sgitg; j < S_v; j += nsg) {
            float acc = 0.0f;
            for (uint i = tiisg; i < S_v; i += 32u) {
                acc += data_scratch[S1 + i + j*S_v] * s_k[i];
            }
            acc = simd_sum(acc);
            if (tiisg == 0u) {
                s_pre[j] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < S_v; j += nthr) {
            s_delta[j] = (s_v[j] - s_pre[j]) * beta_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint n = tid, i = i0, j = j0; n < SS; n += nthr) {
            data_scratch[S_next + n] = data_scratch[S1 + n] + s_k[i]*s_delta[j];
            i += i_step;
            j += j_step;
            if (i >= S_v) { i -= S_v; ++j; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // ---- reverse scan ----
    for (uint n = tid; n < SS; n += nthr) {
        data_scratch[dS + n] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (int ti = (int) n_tokens - 1; ti >= 0; --ti) {
        const uint t = (uint) ti;
        const uint q_off  = iq3*(uint)args.sq3 + t*(uint)args.sq2 + iq1*(uint)args.sq1;
        const uint k_off  = ik3*(uint)args.sk3 + t*(uint)args.sk2 + ik1*(uint)args.sk1;
        const uint v_off  = iv3*(uint)args.sv3 + t*(uint)args.sv2 + iv1*(uint)args.sv1;
        const uint gb_off = iv3*(uint)args.sb3 + t*(uint)args.sb2 + iv1*(uint)args.sb1;
        const float beta_val = data_beta[gb_off];
        const uint g_off = gb_off * (kda ? S_v : 1u);

        const uint S_prev = traj + t*SS;
        const uint d_out  = (iv3*n_tokens*H + t*H + iv1) * S_v;

        for (uint i = tid; i < S_v; i += nthr) {
            s_k[i]    = data_k[k_off + i];
            s_v[i]    = data_v[v_off + i];
            s_q[i]    = data_q[q_off + i];
            s_do[i]   = data_grad[d_out + i];
            s_gexp[i] = exp(data_g[g_off + (kda ? i : 0u)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint n = tid, i = i0; n < SS; n += nthr) {
            data_scratch[S1 + n] = data_scratch[S_prev + n] * s_gexp[i];
            i += i_step;
            if (i >= S_v) { i -= S_v; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint j = sgitg; j < S_v; j += nsg) {
            float acc = 0.0f;
            for (uint i = tiisg; i < S_v; i += 32u) {
                acc += data_scratch[S1 + i + j*S_v] * s_k[i];
            }
            acc = simd_sum(acc);
            if (tiisg == 0u) {
                s_pre[j] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < S_v; j += nthr) {
            s_delta[j] = (s_v[j] - s_pre[j]) * beta_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint n = tid, i = i0, j = j0; n < SS; n += nthr) {
            data_scratch[Snew + n] = data_scratch[S1 + n] + s_k[i]*s_delta[j];
            i += i_step;
            j += j_step;
            if (i >= S_v) { i -= S_v; ++j; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // dS += scale * outer(q, d_out); grad_q = scale * Snew . d_out
        const uint gq_out = g_q_off + S_v*(iq1 + neq1*(t + n_tokens*iq3));
        for (uint i = tid; i < S_v; i += nthr) {
            const float q_i = s_q[i];
            float dqi = 0.0f;
            for (uint j = 0u; j < S_v; ++j) {
                data_scratch[dS + i + j*S_v] += args.scale * s_do[j] * q_i;
                dqi += data_scratch[Snew + i + j*S_v] * s_do[j];
            }
            atomic_fetch_add_explicit(&atomic_dst[gq_out + i], args.scale * dqi, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        const uint target_slot = n_tokens - 1u - t;
        if (target_slot < (uint) args.K) {
            const uint d_snap = attn_score_elems + target_slot*state_size_per_snap + (iv3*H+iv1)*SS;
            for (uint n = tid; n < SS; n += nthr) {
                data_scratch[dS + n] += data_grad[d_snap + n];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }

        // step 3 backward: S_new = S1 + outer(k, delta)
        for (uint n = tid; n < SS; n += nthr) {
            data_scratch[dS1 + n] = data_scratch[dS + n];
        }
        for (uint j = sgitg; j < S_v; j += nsg) {
            float acc = 0.0f;
            for (uint i = tiisg; i < S_v; i += 32u) {
                acc += data_scratch[dS + i + j*S_v] * s_k[i];
            }
            acc = simd_sum(acc);
            if (tiisg == 0u) {
                s_ddelta[j] = acc;
            }
        }
        const uint gk_out = g_k_off + S_v*(ik1 + nek1*(t + n_tokens*ik3));
        for (uint i = tid; i < S_v; i += nthr) {
            float dki = 0.0f;
            for (uint j = 0u; j < S_v; ++j) {
                dki += data_scratch[dS + i + j*S_v] * s_delta[j];
            }
            atomic_fetch_add_explicit(&atomic_dst[gk_out + i], dki, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // step 2 backward: delta[j] = beta*(v[j] - pre[j])
        float dbeta_partial = 0.0f;
        for (uint j = tid; j < S_v; j += nthr) {
            const float dd = s_ddelta[j];
            s_dpre[j] = -dd * beta_val;
            dbeta_partial += dd * (s_v[j] - s_pre[j]);
            data_dst[g_v_off + j + S_v*(iv1 + H*(t + n_tokens*iv3))] += dd * beta_val;
        }
        {
            float v = simd_sum(dbeta_partial);
            if (tiisg == 0u) {
                s_red[sgitg] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dbeta_t = 0.0f;
            for (uint i = 0u; i < nsg; ++i) {
                dbeta_t += s_red[i];
            }
            if (tid == 0u) {
                data_dst[g_beta_off + iv1 + H*(t + n_tokens*iv3)] += dbeta_t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint n = tid, i = i0, j = j0; n < SS; n += nthr) {
            data_scratch[dS1 + n] += s_dpre[j] * s_k[i];
            i += i_step;
            j += j_step;
            if (i >= S_v) { i -= S_v; ++j; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint i = tid; i < S_v; i += nthr) {
            float dki2 = 0.0f;
            for (uint j = 0u; j < S_v; ++j) {
                dki2 += s_dpre[j] * data_scratch[S1 + i + j*S_v];
            }
            atomic_fetch_add_explicit(&atomic_dst[gk_out + i], dki2, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // step 1 backward: S1[i,j] = S_prev[i,j] * gexp[i]
        if (kda) {
            const uint gg_out = g_g_off + S_v*(iv1 + H*(t + n_tokens*iv3));
            for (uint i = tid; i < S_v; i += nthr) {
                const float ge = s_gexp[i];
                float dgexp_i = 0.0f;
                for (uint j = 0u; j < S_v; ++j) {
                    const uint idx = i + j*S_v;
                    dgexp_i += data_scratch[dS1 + idx] * data_scratch[S_prev + idx];
                    data_scratch[dS + idx] = data_scratch[dS1 + idx] * ge;
                }
                data_dst[gg_out + i] += dgexp_i * ge;
            }
        } else {
            const float ge = s_gexp[0];
            float dgexp_partial = 0.0f;
            for (uint n = tid; n < SS; n += nthr) {
                dgexp_partial += data_scratch[dS1 + n] * data_scratch[S_prev + n];
                data_scratch[dS + n] = data_scratch[dS1 + n] * ge;
            }
            float v = simd_sum(dgexp_partial);
            if (tiisg == 0u) {
                s_red[sgitg] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dgexp_sum = 0.0f;
            for (uint i = 0u; i < nsg; ++i) {
                dgexp_sum += s_red[i];
            }
            if (tid == 0u) {
                data_dst[g_g_off + iv1 + H*(t + n_tokens*iv3)] += dgexp_sum * ge;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    const uint gs_out = g_state_off + iv3*state_seq_stride + iv1*SS;
    for (uint n = tid; n < SS; n += nthr) {
        data_dst[gs_out + n] += data_scratch[dS + n];
    }
}

// retro delta: gathers the K overlapping causal-conv rollback snapshot windows
// used by the shared-prefix GRPO scorer's recurrent-state rollback (see
// build_conv_state / [TAG_RECURRENT_ROLLBACK_SPLITS] in delta-net-base.cpp) in
// a single dispatch, replacing a host-side loop that previously built K
// separate view+cpy graph nodes per call. One thread per output element; every
// element is an independent gather. Slot 0 is the window ending at the last
// token, slot s reads s tokens further back, clamped to the start of the
// new-token region when n_seq_tokens < K -- exactly the original loop's
// `std::max<int64_t>(0, ...)`.
//
// Validated bit-for-bit against the CPU reference on Apple M1 for K=1, K>1,
// and n_seq_tokens < K (tests/metal_ops.rs).
kernel void kernel_conv_rs_gather_f32(
        constant ggml_metal_kargs_conv_rs_gather & args,
        device const char  * src0,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    if ((int64_t) gid >= args.total) {
        return;
    }

    uint r = gid;
    const uint k    = r % (uint) args.kernel_m1; r /= (uint) args.kernel_m1;
    const uint c    = r % (uint) args.n_channels; r /= (uint) args.n_channels;
    const uint s    = r % (uint) args.n_seqs;
    const uint slot = r / (uint) args.n_seqs;

    const int64_t back  = args.base - (int64_t) slot;
    const int64_t s_idx = back > 0 ? back : 0;

    dst[gid] = *(device const float *) (src0 +
            (s_idx + (int64_t) k)*args.nb00 + (int64_t) c*args.nb01 + (int64_t) s*args.nb02);
}

// retro delta: stochastic rounding for F16 parameters. Mirrors
// ggml_sr_uniform / ggml_stochastic_round_f16 in ggml-impl.h bit for bit -- an
// exact CPU-vs-GPU equality test covers the three implementations.
static inline float retro_sr_uniform(uint seed, uint index) {
    uint h = seed ^ (index * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return float(h >> 8) * (1.0f / 16777216.0f);
}

static inline ushort retro_f16_neighbour(ushort bits, bool up) {
    const ushort sign = bits & 0x8000;
    const ushort mag  = bits & 0x7FFF;
    if (mag == 0) {
        return up ? 0x0001 : 0x8001;
    }
    const bool grow = up ? (sign == 0) : (sign != 0);
    return ushort(sign | ushort(grow ? mag + 1 : mag - 1));
}

static inline float retro_stochastic_round_f16(float x, float u) {
    const half  nearest   = half(x);
    const float nearest_f = float(nearest);
    const float residual  = x - nearest_f;
    if (residual == 0.0f) {
        return nearest_f;
    }
    const ushort other   = retro_f16_neighbour(as_type<ushort>(nearest), residual > 0.0f);
    const float  other_f = float(as_type<half>(other));
    const float  span    = other_f - nearest_f;
    const float  p = (span != 0.0f && isfinite(span)) ? residual / span : 0.0f;
    return u < p ? other_f : nearest_f;
}

kernel void kernel_opt_step_adamw_f16(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       half * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {
    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const float gi = g[gid] * pars[8];
    const float gmi = g_m[gid] * beta1 + gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);
    g_m[gid] = gmi;
    g_v[gid] = gvi;
    const float updated = float(x[gid]) * (1.0f - alpha * wd)
            - alpha * (gmi * beta1h) / (sqrt(gvi * beta2h) + eps);
    x[gid] = half(retro_stochastic_round_f16(
            updated, retro_sr_uniform(uint(pars[7]), gid)));
}

// retro delta: `kernel_rms_norm_back_f32` and `kernel_l2_norm_back_f32` stood
// here. Both are retired (docs/INT_RIR_V4.md §P6): their (op, backend) pairs
// declare no restriction on the ggml domain, so the generated RIR variants —
// `rir_rms_norm_back*` and `rir_l2_norm_back*` in kernels/rir.metal — are the
// only implementation, and `ggml_rir_supports_op` is the whole answer Metal
// gives for them. Rebuilding one for a differential means checking out the
// commit that removed them.

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
// with GQA broadcast i02 = i2/dps2, i03 = i3/dps3.
//
// Tiled GEMM over the contraction axis ne01, ported from the Vulkan P2 shader
// (docs/optims/VRAM_2.md 3.2, docs/backends/UNIFY.md 6.1). The previous
// shape was one dst element per thread with the reduction advanced one k at a
// time, which paid two threadgroup barriers per k, loaded 16 of 64 threads'
// worth of operands per step, and yielded a single fused multiply-add per pair
// of threadgroup reads. Its threadgroup was also 64 threads -- two SIMD groups,
// under the occupancy Apple silicon needs.
//
// The tiling fixes all of it: one barrier pair per BK-slice instead of per k
// (BK times fewer), every one of the 256 threads participating in both
// cooperative loads, and OUT_PROD_TM dst rows per thread so each src1 value
// read from threadgroup memory feeds TM FMAs rather than one.
//
// BM is 64 while BN stays 16 on purpose: dst is ne00 x n_tokens, tall and thin
// in the training regime (ne1 is the token count, and n_ubatch is small), so a
// square tile would spend most of itself on columns that do not exist.
//
// The accumulation order is deliberately unchanged -- k ascends within a slice
// and the slices ascend -- so every dst element sums the same terms in the same
// sequence, with the same operations, as the scalar kernel did: bit-exact
// against it by construction. The probes in tests/metal_ops.rs consequently pass
// at their original tolerances against the CPU oracle; widening one would mean
// the arithmetic had changed.
#define OUT_PROD_BM  64
#define OUT_PROD_BN  16
#define OUT_PROD_BK  16
#define OUT_PROD_TM   4
#define OUT_PROD_NTH 256

// The only part that differs per src0 type: filling one BK x BM slice of src0
// into threadgroup memory, laid out [kk][mm]. F32 reads a float, the legacy
// quants decode one element at a time, the K-quants decode 16 at a time and so
// run one thread per 16-value chunk. Out-of-range lanes store zero rather than
// skipping, so the inner product below needs no per-element predicate.
//
// `s01/s02/s03` are element strides for the F32 variant and byte strides for
// the quantized ones (see ggml_metal_kargs_out_prod); each loader owns that
// convention, which is why the base pointer is computed here and not by the
// caller.
struct out_prod_tile_f32 {
    static void load(
            threadgroup float * tile0,
            device const char * src0,
            constant ggml_metal_kargs_out_prod & args,
            int64_t i02, int64_t i03, int64_t m0, int64_t k0, ushort tiitg) {
        device const float * base0 = (device const float *) src0 + i02*args.s02 + i03*args.s03;
        for (ushort l = tiitg; l < OUT_PROD_BK*OUT_PROD_BM; l += OUT_PROD_NTH) {
            const int64_t m = m0 + (l % OUT_PROD_BM);
            const int64_t k = k0 + (l / OUT_PROD_BM);
            tile0[l] = (m < args.ne0 && k < args.ne01) ? base0[m + k*args.s01] : 0.0f;
        }
    }
};

// Every non-F32 type, through the one decoding contract the whole file already
// relies on: dequantize_<T>(blk, il, reg) fills the 16 *consecutive* elements at
// il*16 within a block of nl*16 values (the same (block, nl, dequantize) triple
// kernel_mul_mm is instantiated with). So one thread owns one 16-value chunk of
// the tile row rather than one element, and the type only enters as template
// arguments -- no per-format bit twiddling here. F16 is the nl == 1 case;
// nothing about it is special.
//
// BM is a multiple of 16, and supports_op rejects a src0 whose row width is not,
// so a chunk never straddles the end of the tensor and never crosses a block
// boundary. Out-of-range lanes store zero rather than skipping, so the inner
// product below needs no per-element predicate.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
struct out_prod_tile_dq {
    static void load(
            threadgroup float * tile0,
            device const char * src0,
            constant ggml_metal_kargs_out_prod & args,
            int64_t i02, int64_t i03, int64_t m0, int64_t k0, ushort tiitg) {
        constexpr ushort n_chunks = OUT_PROD_BM / 16;
        constexpr short  qk       = nl*16; // elements per block
        device const char * base0 = src0 + i02*args.s02 + i03*args.s03;
        for (ushort l = tiitg; l < OUT_PROD_BK*n_chunks; l += OUT_PROD_NTH) {
            const ushort kk = l / n_chunks;
            const ushort cc = l % n_chunks;
            const int64_t m = m0 + cc*16;
            const int64_t k = k0 + kk;
            float4x4 values(0.0f);
            if (m < args.ne0 && k < args.ne01) {
                dequantize_func(
                        (device const block_q *)(base0 + k*args.s01) + m/qk,
                        (short) ((m % qk) / 16), values);
            }
            threadgroup float * out = tile0 + kk*OUT_PROD_BM + cc*16;
            for (ushort j = 0; j < 16; ++j) {
                out[j] = values[j/4][j%4];
            }
        }
    }
};

template<typename loader>
kernel void kernel_out_prod_impl(
        constant ggml_metal_kargs_out_prod & args,
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    threadgroup float tile0[OUT_PROD_BK*OUT_PROD_BM];
    threadgroup float tile1[OUT_PROD_BK*OUT_PROD_BN];

    // tn varies fastest, so neighbouring threads read consecutive tile1 entries
    // and their TM tile0 reads collapse onto few distinct addresses.
    const ushort tn = tiitg % OUT_PROD_BN;
    const ushort tm = tiitg / OUT_PROD_BN;

    const int64_t m0 = (int64_t) tgpig.x * OUT_PROD_BM;
    const int64_t n0 = (int64_t) tgpig.y * OUT_PROD_BN;
    const int64_t i2 = (int64_t) tgpig.z % args.ne2;
    const int64_t i3 = (int64_t) tgpig.z / args.ne2;
    // Uniform across the threadgroup, so this early return cannot strand a
    // thread at the barriers below -- unlike a per-element bounds test.
    if (i3 >= args.ne3) {
        return;
    }

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;
    const int64_t off1 = i2*args.s12 + i3*args.s13;

    float acc[OUT_PROD_TM];
    for (ushort r = 0; r < OUT_PROD_TM; ++r) {
        acc[r] = 0.0f;
    }

    for (int64_t k0 = 0; k0 < args.ne01; k0 += OUT_PROD_BK) {
        loader::load(tile0, src0, args, i02, i03, m0, k0, tiitg);
        for (ushort l = tiitg; l < OUT_PROD_BK*OUT_PROD_BN; l += OUT_PROD_NTH) {
            const int64_t n = n0 + (l % OUT_PROD_BN);
            const int64_t k = k0 + (l / OUT_PROD_BN);
            tile1[l] = (n < args.ne1 && k < args.ne01)
                    ? src1[off1 + n*args.s10 + k*args.s11] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort kk = 0; kk < OUT_PROD_BK; ++kk) {
            const float bv = tile1[kk*OUT_PROD_BN + tn];
            for (ushort r = 0; r < OUT_PROD_TM; ++r) {
                acc[r] += tile0[kk*OUT_PROD_BM + tm*OUT_PROD_TM + r] * bv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int64_t n = n0 + tn;
    if (n >= args.ne1) {
        return;
    }
    for (ushort r = 0; r < OUT_PROD_TM; ++r) {
        const int64_t m = m0 + tm*OUT_PROD_TM + r;
        if (m < args.ne0) {
            dst[m + n*args.s1 + i2*args.s2 + i3*args.s3] = acc[r];
        }
    }
}

typedef decltype(kernel_out_prod_impl<out_prod_tile_f32>) out_prod_t;

template [[host_name("kernel_out_prod_f32")]]  kernel out_prod_t kernel_out_prod_impl<out_prod_tile_f32>;

// retro delta: one pipeline per decodable type, straight from the shared table.
// The pipeline name must be kernel_out_prod_<ggml_type_name(type)> -- that is
// what ggml_metal_library_get_pipeline_out_prod builds at dispatch time.
#define GGML_RETRO_OUT_PROD_PIPELINE(TYPE, BLK, NL, NAME, VKNAME)               \
    template [[host_name("kernel_out_prod_" #NAME)]]                            \
    kernel out_prod_t kernel_out_prod_impl<out_prod_tile_dq<BLK, NL, dequantize_##NAME>>;
GGML_RETRO_DEQUANT_TYPES(GGML_RETRO_OUT_PROD_PIPELINE)
#undef GGML_RETRO_OUT_PROD_PIPELINE

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

// retro delta: fused sparse cross-entropy over a (possibly quantized)
// projection head, forward and backward. Ported from the Vulkan shaders
// (fused_sparse_ce{,_back}.comp), which is the right lineage for Metal: the
// head is read *in the kernel* through the per-type dequantizers, so no F32
// scratch copy of it is ever allocated -- unlike the CUDA path, which
// dequantizes into a tiled buffer. See docs/backends/UNIFY.md 6.3 and the CPU
// oracle ggml_compute_forward_fused_sparse_ce_f32.
//
// One threadgroup per token. The [n_vocab, n_tokens] logits are never
// materialized: each logit z[v,t] = dot(w[:,v], h[:,t]) (+ bias[v]) is
// recomputed on the fly, and the log-sum-exp is accumulated online in F32.
//
// `nl` is the number of 16-element chunks per block of the head's type -- the
// same convention as kernel_mul_mm, so `dequantize_func(blk, il, reg)` hands
// back the 16 consecutive logical elements starting at 16*chunk.
//
// n_active (the divisor both the loss and the gradient use) is counted inside
// the threadgroup rather than by a separate dispatch into a scratch buffer as
// on Vulkan: it is O(n_tokens) next to O(n_vocab * n_embd) of real work, and it
// keeps this op free of any allocation the graph allocator does not already
// own -- the property that makes Metal's memory accounting exact (UNIFY.md 4).

#define FSCE_NTH 256
// Chunks of 16 embedding elements one thread may own in the backward's
// accumulator. FSCE_NTH*16*FSCE_MAXC = 16384 is the n_embd ceiling that
// supports_op enforces.
#define FSCE_MAXC 4

// z[v,t] = dot(w[:,v], h[:,t]), F32 throughout so a low-precision head never
// biases the gradient.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
static float fsce_w_dot_h(
        device const char  * w,
        device const float * h_col,
        uint64_t nb_w,
        int      v,
        int      n_chunks) {
    device const char * w_row = w + (size_t) v*nb_w;
    float z = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        float4x4 reg;
        dequantize_func((device const block_q *) w_row + c/nl, (short) (c%nl), reg);
        device const float4 * hc = (device const float4 *)(h_col + c*16);
        z += dot(reg[0], hc[0]) + dot(reg[1], hc[1]) + dot(reg[2], hc[2]) + dot(reg[3], hc[3]);
    }
    return z;
}

// Active tokens: a real target with a non-zero coefficient. Same predicate as
// the CPU reference, evaluated on device because the targets only exist there.
static int fsce_count_active(
        device const int   * targets,
        device const float * weights,
        int n_tokens,
        int n_vocab,
        threadgroup float * sh,
        ushort tpitg, ushort sgitg, ushort tiisg, ushort ntg) {
    float local = 0.0f;
    for (int i = tpitg; i < n_tokens; i += ntg) {
        const int tgt = targets[i];
        if (tgt >= 0 && tgt < n_vocab && weights[i] != 0.0f) {
            local += 1.0f;
        }
    }
    return (int) retro_tg_sum(local, sh, sgitg, tiisg, ntg);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_fused_sparse_ce(
        constant ggml_metal_kargs_fused_sparse_ce & args,
        device const char   * h,
        device const char   * w,
        device const int    * targets,
        device const float  * weights,
        device const float  * bias,
        device atomic_float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float sh[32];

    const int t = tgpig.x;

    // Uniform across the threadgroup (it depends on the token only), so the
    // reductions below can never be reached by some threads and not others.
    const int n_active = fsce_count_active(
            targets, weights, args.n_tokens, args.n_vocab, sh, tpitg.x, sgitg, tiisg, ntg.x);

    const int   tgt = targets[t];
    const float wgt = weights[t];
    if (!(tgt >= 0 && tgt < args.n_vocab && wgt != 0.0f) || n_active == 0) {
        return;
    }

    device const float * h_col = (device const float *)(h + (size_t) t*args.nb_h);
    const int n_chunks = args.n_embd/16;

    float lmax = -INFINITY;
    float lsum = 0.0f;
    float ztgt = 0.0f;
    for (int v = tpitg.x; v < args.n_vocab; v += ntg.x) {
        float z = fsce_w_dot_h<block_q, nl, dequantize_func>(w, h_col, args.nb_w, v, n_chunks);
        if (args.has_bias) {
            z += bias[v];
        }
        if (z > lmax) {
            lsum = lsum*exp(lmax - z) + 1.0f;
            lmax = z;
        } else {
            lsum += exp(z - lmax);
        }
        if (v == tgt) {
            ztgt = z;
        }
    }

    // Merge the partial online log-sum-exps: rescale every thread's sum to the
    // threadgroup max, then add. A thread that was handed no vocabulary row
    // contributes lsum = 0, and exp(-inf - max) = 0, so it stays neutral.
    const float lmax_all = retro_tg_max(lmax, sh, sgitg, tiisg, ntg.x);
    const float lsum_all = retro_tg_sum(lsum*exp(lmax - lmax_all), sh, sgitg, tiisg, ntg.x);
    // Exactly one thread saw the target row, so a sum broadcasts its value.
    const float ztgt_all = retro_tg_sum(ztgt, sh, sgitg, tiisg, ntg.x);

    if (tpitg.x == 0) {
        const float lse = lmax_all + log(lsum_all);
        atomic_fetch_add_explicit(dst, wgt*(lse - ztgt_all)/(float) n_active, memory_order_relaxed);
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_fused_sparse_ce_back(
        constant ggml_metal_kargs_fused_sparse_ce & args,
        device const float * grad,
        device const char  * h,
        device const char  * w,
        device const int   * targets,
        device const float * weights,
        device const float * bias,
        device       char  * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float sh_p[FSCE_NTH];

    const int t = tgpig.x;
    const int n_chunks = args.n_embd/16;

    const int n_active = fsce_count_active(
            targets, weights, args.n_tokens, args.n_vocab, sh, tpitg.x, sgitg, tiisg, ntg.x);

    const int   tgt = targets[t];
    const float wgt = weights[t];
    device float * dst_col = (device float *)(dst + (size_t) t*args.nb_d);

    if (!(tgt >= 0 && tgt < args.n_vocab && wgt != 0.0f) || n_active == 0) {
        for (int e = tpitg.x; e < args.n_embd; e += ntg.x) {
            dst_col[e] = 0.0f;
        }
        return;
    }

    device const float * h_col = (device const float *)(h + (size_t) t*args.nb_h);

    // Pass 1: the token's log-sum-exp, recomputed exactly as the forward did
    // rather than stored -- this op checkpoints the logits, it never keeps them.
    float lmax = -INFINITY;
    float lsum = 0.0f;
    for (int v = tpitg.x; v < args.n_vocab; v += ntg.x) {
        float z = fsce_w_dot_h<block_q, nl, dequantize_func>(w, h_col, args.nb_w, v, n_chunks);
        if (args.has_bias) {
            z += bias[v];
        }
        if (z > lmax) {
            lsum = lsum*exp(lmax - z) + 1.0f;
            lmax = z;
        } else {
            lsum += exp(z - lmax);
        }
    }
    const float lmax_all = retro_tg_max(lmax, sh, sgitg, tiisg, ntg.x);
    const float lsum_all = retro_tg_sum(lsum*exp(lmax - lmax_all), sh, sgitg, tiisg, ntg.x);
    const float lse = lmax_all + log(lsum_all);

    // Pass 2: acc[e] = sum_v softmax(z[v]) * w[e,v], streamed in threadgroup-sized
    // vocabulary tiles. Each tile's softmax weights are produced one thread per
    // vocabulary row, then consumed one thread per 16-element chunk of the
    // embedding, so every accumulation stays in registers and no atomic or
    // shared-memory accumulator is needed.
    //
    // The consumption half only has n_embd/16 threads to give work to, so it
    // runs at partial occupancy below n_embd = 4096. Deliberate: it keeps the
    // decode count identical to pass 1 (one dequantize call per 16 values),
    // which is what dominates. Splitting the tile across the idle threads would
    // need a second accumulator per thread and a reduction over it.
    float4x4 acc[FSCE_MAXC];
    for (short k = 0; k < FSCE_MAXC; ++k) {
        acc[k] = float4x4(0.0f);
    }

    for (int tile = 0; tile < args.n_vocab; tile += ntg.x) {
        const int v = tile + tpitg.x;
        float p = 0.0f;
        if (v < args.n_vocab) {
            float z = fsce_w_dot_h<block_q, nl, dequantize_func>(w, h_col, args.nb_w, v, n_chunks);
            if (args.has_bias) {
                z += bias[v];
            }
            p = exp(z - lse);
        }
        sh_p[tpitg.x] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int tile_n = min((int) ntg.x, args.n_vocab - tile);
        short k = 0;
        for (int c = tpitg.x; c < n_chunks; c += ntg.x, ++k) {
            float4x4 a(0.0f);
            for (int j = 0; j < tile_n; ++j) {
                float4x4 reg;
                dequantize_func(
                        (device const block_q *)(w + (size_t)(tile + j)*args.nb_w) + c/nl,
                        (short) (c%nl), reg);
                a += sh_p[j]*reg;
            }
            acc[k] += a;
        }
        // Also orders the last read of `h` above before the writes below, which
        // matters because the graph allocator may have placed dst on top of h
        // (the fused CE's offload_h option).
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coef = grad[0]*wgt/(float) n_active;
    short k = 0;
    for (int c = tpitg.x; c < n_chunks; c += ntg.x, ++k) {
        float4x4 wt;
        dequantize_func(
                (device const block_q *)(w + (size_t) tgt*args.nb_w) + c/nl, (short) (c%nl), wt);
        device float4 * d = (device float4 *)(dst_col + c*16);
        for (short r = 0; r < 4; ++r) {
            d[r] = coef*(acc[k][r] - wt[r]);
        }
    }
}

typedef decltype(kernel_fused_sparse_ce<float4x4, 1, dequantize_f32>) fused_sparse_ce_t;
typedef decltype(kernel_fused_sparse_ce_back<float4x4, 1, dequantize_f32>) fused_sparse_ce_back_t;

template [[host_name("kernel_fused_sparse_ce_f32")]]      kernel fused_sparse_ce_t      kernel_fused_sparse_ce     <float4x4, 1, dequantize_f32>;
template [[host_name("kernel_fused_sparse_ce_back_f32")]] kernel fused_sparse_ce_back_t kernel_fused_sparse_ce_back<float4x4, 1, dequantize_f32>;

// retro delta: the head-type family, from the same table that drives out_prod.
// It is the same (BLK, NL, dequantize_<NAME>) triple, so a type added to the
// table lands on both ops at once -- which is the point: before the table, this
// list and out_prod's had drifted apart by four types.
#define GGML_RETRO_FUSED_SPARSE_CE_PIPELINE(TYPE, BLK, NL, NAME, VKNAME)                    \
    template [[host_name("kernel_fused_sparse_ce_" #NAME)]]                                 \
    kernel fused_sparse_ce_t      kernel_fused_sparse_ce     <BLK, NL, dequantize_##NAME>;  \
    template [[host_name("kernel_fused_sparse_ce_back_" #NAME)]]                            \
    kernel fused_sparse_ce_back_t kernel_fused_sparse_ce_back<BLK, NL, dequantize_##NAME>;
GGML_RETRO_DEQUANT_TYPES(GGML_RETRO_FUSED_SPARSE_CE_PIPELINE)
#undef GGML_RETRO_FUSED_SPARSE_CE_PIPELINE

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
