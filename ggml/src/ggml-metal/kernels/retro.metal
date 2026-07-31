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


Warning: truncated output (original token count: 100321)
Total output lines: 8949

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

constant short FC_solve_tri_nsg [[function_constant(FC_SOLVE_TRI + 0)]];
constant short FC_solve_tri_n   [[function_constant(FC_SOLVE_TRI + 1)]];
constant short FC_solve_tri_k   [[function_constant(FC_SOLVE_TRI + 2)]];

kernel void kernel_solve_tri_f32(
        constant ggml_metal_kargs_solve_tri & args,
        device   const char * src0,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        ushort3 tgpig[[threadgroup_position_in_grid]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const short NSG = FC_solve_tri_nsg;
    const short N   = FC_solve_tri_n;
    const short K   = FC_solve_tri_k;
    const short NP  = PAD2(N, NW);

    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x*NSG + sgitg;

    threadgroup float * sh0 = (threadgroup float *) shmem;

    device const float * src0_ptr = (device const float *)(src0 + i02 * args.nb02 + i03 * args.nb03) + sgitg*N;
    device const float * src1_ptr = (device const float *)(src1 + i02 * args.nb12 + i03 * args.nb13) + i01;
    device       float * dst_ptr  = (device       float *)(dst  + i02 * args.nb2  + i03 * args.nb3)  + i01;

    for (short rr = 0; rr < N; rr += NSG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            threadgroup float * sh0_cur = sh0 + sgitg*NP;

            for (short t = 0; t*NW < N; ++t) {
                const short idx = t*NW + tiisg;
                sh0_cur[idx] = src0_ptr[idx];
            }

            src0_ptr += NSG*N;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i01 >= args.ne10) {
            continue;
        }

        for (short ir = 0; ir < NSG && rr + ir < N; ++ir) {
            const short r = rr + ir;

            threadgroup float * sh0_cur = sh0 + ir*NP;

            float sum = 0.0f;

            for (short t = 0; t*NW < r; ++t) {
                const short idx = t*NW + tiisg;
                sum += sh0_cur[idx] * dst_ptr[idx*K] * (idx < r);
            }

            sum = simd_sum(sum);

            if (tiisg == 0) {
                const float diag = sh0_cur[r];

                dst_ptr[r*K] = (src1_ptr[r*K] - sum) / diag;
            }
        }
    }
}

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

// F == 1 : norm (no fuse)
// F == 2 : norm + mul
// F == 3 : norm + mul + add
template <typename T, short F>
kernel void kernel_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    T sumft(0.0f);

    float sumf = 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumft += x[i00];
    }
    sumf = dot(sumft, T(1.0f));
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf/args.ne00;

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);

    sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        y[i00] = x[i00] - mean;
        sumf += dot(y[i00], y[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float variance = sumf/args.ne00;

    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (y[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (y[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (y[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_norm_fuse_impl<float4, 1>) kernel_norm_fuse_t;

template [[host_name("kernel_norm_f32")]]         kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 1>;
template [[host_name("kernel_norm_mul_f32")]]     kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 2>;
template [[host_name("kernel_norm_mul_add_f32")]] kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 3>;

template [[host_name("kernel_norm_f32_4")]]         kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_norm_mul_f32_4")]]     kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 2>;
template [[host_name("kernel_norm_mul_add_f32_4")]] kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 3>;

// F == 1 : rms_norm (no fuse)
// F == 2 : rms_norm + mul
// F == 3 : rms_norm + mul + add
template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;

template [[host_name("kernel_rms_norm_f32")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 1>;
template [[host_name("kernel_rms_norm_mul_f32")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;
template [[host_name("kernel_rms_norm_mul_add_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 3>;

template [[host_name("kernel_rms_norm_f32_4")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_rms_norm_mul_f32_4")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;
template [[host_name("kernel_rms_norm_mul_add_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 3>;

template <typename T0, typename T>
kernel void kernel_l2_norm_impl(
        constant ggml_metal_kargs_l2_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const T0 * x = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T  * y = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f/max(sqrt(sumf), args.eps);

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = x[i00] * scale;
    }
}

typedef decltype(kernel_l2_norm_impl<float, float>) kernel_l2_norm_t;

template [[host_name("kernel_l2_norm_f32_f32")]]   kernel kernel_l2_norm_t kernel_l2_norm_impl<float,  float>;
template [[host_name("kernel_l2_norm_f32_f32_4")]] kernel kernel_l2_norm_t kernel_l2_norm_impl<float4, float4>;

kernel void kernel_group_norm_f32(
        constant ggml_metal_kargs_group_norm & args,
        device const float * src0,
        device       float * dst,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = args.ne00*args.ne01*args.ne02;
    const int64_t gs = args.ne00*args.ne01*((args.ne02 + args.ngrp - 1) / args.ngrp);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// Q1_0 dot product: dot = d * (2 * Σ(yl[i] where bit=1) - sumy)
inline float block_q_n_dot_y(device const block_q1_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + il / 8;
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];

    float acc = 0.0f;

    acc += select(0.0f, yl[ 0], bool(b0 & 0x01));
    acc += select(0.0f, yl[ 1], bool(b0 & 0x02));
    acc += select(0.0f, yl[ 2], bool(b0 & 0x04));
    acc += select(0.0f, yl[ 3], bool(b0 & 0x08));
    acc += select(0.0f, yl[ 4], bool(b0 & 0x10));
    acc += select(0.0f, yl[ 5], bool(b0 & 0x20));
    acc += select(0.0f, yl[ 6], bool(b0 & 0x40));
    acc += select(0.0f, yl[ 7], bool(b0 & 0x80));

    acc += select(0.0f, yl[ 8], bool(b1 & 0x01));
    acc += select(0.0f, yl[ 9], bool(b1 & 0x02));
    acc += select(0.0f, yl[10], bool(b1 & 0x04));
    acc += select(0.0f, yl[11], bool(b1 & 0x08));
    acc += select(0.0f, yl[12], bool(b1 & 0x10));
    acc += select(0.0f, yl[13], bool(b1 & 0x20));
    acc += select(0.0f, yl[14], bool(b1 & 0x40));
    acc += select(0.0f, yl[15], bool(b1 & 0x80));

    return qb_curr->d * (2.0f * acc - sumy);
}

// Q2_0 dot: d * (sum_lo(y) + 2*sum_hi(y) - sumy) via per-bit conditional adds
inline float block_q_n_dot_y(device const block_q2_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + (il / 4);
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];
    const uint8_t b2 = qs[2];
    const uint8_t b3 = qs[3];

    // Accumulate where low bit is set (bits 0,2,4,6 of each byte)
    float acc_lo = 0.0f;
    acc_lo += select(0.0f, yl[ 0], bool(b0 & 0x01));
    acc_lo += select(0.0f, yl[ 1], bool(b0 & 0x04));
    acc_lo += select(0.0f, yl[ 2], bool(b0 & 0x10));
    acc_lo += select(0.0f, yl[ 3], bool(b0 & 0x40));
    acc_lo += select(0.0f, yl[ 4], bool(b1 & 0x01));
    acc_lo += select(0.0f, yl[ 5], bool(b1 & 0x04));
    acc_lo += select(0.0f, yl[ 6], bool(b1 & 0x10));
    acc_lo += select(0.0f, yl[ 7], bool(b1 & 0x40));
    acc_lo += select(0.0f, yl[ 8], bool(b2 & 0x01));
    acc_lo += select(0.0f, yl[ 9], bool(b2 & 0x04));
    acc_lo += select(0.0f, yl[10], bool(b2 & 0x10));
    acc_lo += select(0.0f, yl[11], bool(b2 & 0x40));
    acc_lo += select(0.0f, yl[12], bool(b3 & 0x01));
    acc_lo += select(0.0f, yl[13], bool(b3 & 0x04));
    acc_lo += select(0.0f, yl[14], bool(b3 & 0x10));
    acc_lo += select(0.0f, yl[15], bool(b3 & 0x40));

    // Accumulate where high bit is set (bits 1,3,5,7 of each byte)
    float acc_hi = 0.0f;
    acc_hi += select(0.0f, yl[ 0], bool(b0 & 0x02));
    acc_hi += select(0.0f, yl[ 1], bool(b0 & 0x08));
    acc_hi += select(0.0f, yl[ 2], bool(b0 & 0x20));
    acc_hi += select(0.0f, yl[ 3], bool(b0 & 0x80));
    acc_hi += select(0.0f, yl[ 4], bool(b1 & 0x02));
    acc_hi += select(0.0f, yl[ 5], bool(b1 & 0x08));
    acc_hi += select(0.0f, yl[ 6], bool(b1 & 0x20));
    acc_hi += select(0.0f, yl[ 7], bool(b1 & 0x80));
    acc_hi += select(0.0f, yl[ 8], bool(b2 & 0x02));
    acc_hi += select(0.0f, yl[ 9], bool(b2 & 0x08));
    acc_hi += select(0.0f, yl[10], bool(b2 & 0x20));
    acc_hi += select(0.0f, yl[11], bool(b2 & 0x80));
    acc_hi += select(0.0f, yl[12], bool(b3 & 0x02));
    acc_hi += select(0.0f, yl[13], bool(b3 & 0x08));
    acc_hi += select(0.0f, yl[14], bool(b3 & 0x20));
    acc_hi += select(0.0f, yl[15], bool(b3 & 0x80));

    return qb_curr->d * (acc_lo + 2.0f * acc_hi - sumy);
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];

template<typename block_q_type, short NR0, typename args_t>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
  //const int r0 =  tgpig.x*NR0;
    const int r1 =  tgpig.y;
    const int im =  tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    //const int ib0 = sgitg*NQ + ix;
    const int ib0 = ix;

    float yl[16]; // src1 vector cache

    //device const float * yb = y + ix*QK4_0 + il;
    device const float * yb = y + ib0*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    //for (int ib = ib0; ib < nb; ib += NSG*NQ) {
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
        //yb += NSG*NQ*QK4_0;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    //helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q1_0 *) ((device char *) src0 + offset0);
    }

    float yl[16];
    float sumf[nr0] = {0.f};

    const short ix = (tiisg/8);
    const short il = (tiisg%8)*16;

    device const float * yb = y + ix*QK1_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        float sumy = 0.f;

        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += QK1_0 * (N_SIMDWIDTH/8);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q1_0_f32")]]
kernel void kernel_mul_mv_q1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<N_R0_Q1_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK2_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q2_0 *) ((device char *) src0 + offset0);
    }

    float yl[16];
    float sumf[nr0] = {0.f};

    // group 64: 4 sub-blocks of 16 weights per Q2_0 block
    const short ix = (tiisg/4);
    const short il = (tiisg%4)*16;

    device const float * yb = y + ix*QK2_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        float sumy = 0.f;

        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += QK2_0 * (N_SIMDWIDTH/4);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q2_0_f32")]]
kernel void kernel_mul_mv_q2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// mat-vec kernel processing in chunks of float4
// chpb - chunks per quantization block
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// mat-vec kernel processing in chunks of float4x4
template<short r1ptg, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &) >
void kernel_mul_mv_ext_q4x4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 1;

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4x4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] +=
                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);

            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// dispatchers needed for compile-time nxpsg
// epb - elements per quantization block
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

template<short r1ptg, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void kernel_mul_mv_ext_q4x4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4x4_f32_impl<r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>)    mul_mv_ext_q4x4_f32_t;

template [[host_name("kernel_mul_mv_ext_f32_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, float4,       4,  dequantize_f32_t4>;

template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,        4,  dequantize_f16_t4>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, bfloat4,      4,  dequantize_bf16_t4>;
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, bfloat4,      4,  dequantize_bf16_t4>;
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, bfloat4,      4,  dequantize_bf16_t4>;
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, bfloat4,      4,  dequantize_bf16_t4>;
#endif

template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q1_0,   128, dequantize_q1_0_t4>;

template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q2_0,    64, dequantize_q2_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_0,   32, dequantize_q4_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_1,   32, dequantize_q4_1_t4>;

template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_0,   32, dequantize_q5_0_t4>;

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_1,   32, dequantize_q5_1_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_mxfp4,  32, dequantize_mxfp4_t4>;

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_K, 256, dequantize_q4_K>;

template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_K, 256, dequantize_q5_K>;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_K, 256, dequantize_q6_K>;

template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q2_K, 256, dequantize_q2_K>;

template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q3_K, 256, dequantize_q3_K>;

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const T0 * x = (device const T0 *) (src0 + offset0);
    device const T1 * y = (device const T1 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0 * ax [NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
      //case 1: kernel_mul_mv_t_t_impl<T0, T1, 1, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 3: kernel_mul_mv_t_t_impl<T0, T1, 3, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, float>;
template [[host_name("kernel_mul_mv_bf16_bf16")]] kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, bfloat>;
#endif

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
      //case 1: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 1, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 3: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 3, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

template [[host_name("kernel_mul_mv_f32_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;
template [[host_name("kernel_mul_mv_f16_f16_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  half,  half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, float,  float4>;
template [[host_name("kernel_mul_mv_bf16_bf16_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, bfloat, bfloat4>;
#endif

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ggml_metal_kargs_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

template [[host_name("kernel_mul_mv_f32_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;
template [[host_name("kernel_mul_mv_f16_f16_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, float>;
template [[host_name("kernel_mul_mv_bf16_bf16_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, bfloat>;
#endif

constant bool FC_rope_is_imrope [[function_constant(FC_ROPE + 0)]];
constant bool FC_rope_is_back   [[function_constant(FC_ROPE + 1)]];

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
    if (FC_rope_is_back) {
        *sin_theta *= -1.0f;
    }
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
            const int iw = i0 - args.n_offs; // relative idx
            const int ic = iw/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            if (args.inplace) {
                continue;
            }

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
            const int iw = i0 - args.n_offs; // relative idx
            const int ic = iw/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + (args.n_offs + ic)*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + (args.n_offs + ic)*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            if (args.inplace) {
                continue;
            }

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_multi(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
            const int iw = i0 - args.n_offs; // relative idx
            const int ic = iw/2;

            // mrope theta calculations
            // note: the rest is the same as kernel_rope_neox
            const int sect_dims = args.sect_0 + args.sect_1 + args.sect_2 + args.sect_3;
            const int sec_w01   = args.sect_0 + args.sect_1;               // end of section 1
            const int sec_w012  = args.sect_0 + args.sect_1 + args.sect_2; // end of section 2
            const int sector    = ic % sect_dims;

            float theta_base;
            if (FC_rope_is_imrope) {
                if (sector % 3 == 1 && sector < 3 * args.sect_1) { // h
                    theta_base = (float) pos[i2 + args.ne02 * 1];
                } else if (sector % 3 == 2 && sector < 3 * args.sect_2) { // w
                    theta_base = (float) pos[i2 + args.ne02 * 2];
                } else if (sector % 3 == 0 && sector < 3 * args.sect_0) { // t
                    theta_base = (float) pos[i2 + args.ne02 * 0];
                } else { // e
                    theta_base = (float) pos[i2 + args.ne02 * 3];
                }
            } else {
                if (sector < args.sect_0) {
                    theta_base = (float) pos[i2];
                } else if (sector < sec_w01) {
                    theta_base = (float) pos[i2 + args.ne02 * 1];
                } else if (sector < sec_w012) {
                    theta_base = (float) pos[i2 + args.ne02 * 2];
                } else {
                    theta_base = (float) pos[i2 + args.ne02 * 3];
                }
            }
            // end of mrope

            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + (args.n_offs + ic)*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + (args.n_offs + ic)*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            if (args.inplace) {
                continue;
            }

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_vision(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < 2*args.n_dims) { // different from kernel_rope_multi
            const int ic = i0/2;

            // mrope theta calculations (only support 2 dimensions)
            const int sect_dims = args.sect_0 + args.sect_1;
            const int sector    = ic % sect_dims;

            float p;
            float theta_base;
            if (sector < args.sect_1) {
                p = (float) sector;
                theta_base = (float) pos[i2];
            } else {
                p = (float) sector - args.sect_0;
                theta_base = (float) pos[i2 + args.ne02];
            }

            const float theta = theta_base * pow(args.freq_base, 2.0f * inv_ndims * p);
            // end of mrope

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims]; // different from kernel_rope_multi

            dst_data[0]           = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims] = x0*sin_theta + x1*cos_theta; // different from kernel_rope_multi
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;
typedef decltype(kernel_rope_multi<float>) kernel_rope_multi_t;
typedef decltype(kernel_rope_vision<float>) kernel_rope_vision_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

template [[host_name("kernel_rope_multi_f32")]] kernel kernel_rope_multi_t kernel_rope_multi<float>;
template [[host_name("kernel_rope_multi_f16")]] kernel kernel_rope_multi_t kernel_rope_multi<half>;

template [[host_name("kernel_rope_vision_f32")]] kernel kernel_rope_vision_t kernel_rope_vision<float>;
template [[host_name("kernel_rope_vision_f16")]] kernel kernel_rope_vision_t kernel_rope_vision<half>;

typedef void (im2col_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
//    const int64_t IC = tgpg[0];
    const int64_t OH = tgpg[1];
    const int64_t OW = tgpg[2];

    const int64_t KH = ntg[1];
    const int64_t KW = ntg[2];

          int64_t in  = tpitg[0];
    const int64_t ikh = tpitg[1];
    const int64_t ikw = tpitg[2];

    const int64_t iic = tgpig[0];
    const int64_t ioh = tgpig[1];
    const int64_t iow = tgpig[2];

    const int64_t iiw = iow*args.s0 + ikw*args.d0 - args.p0;
    const int64_t iih = ioh*args.s1 + ikh*args.d1 - args.p1;

    int64_t offset_dst = (in*OH*OW + ioh*OW + iow)*args.CHW + (iic*(KH*KW) + ikh*KW + ikw);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
        while (in < args.N) {
            pdst[offset_dst] = 0.0f;
            offset_dst += ntg[0]*args.CHW*OH*OW;

            in += ntg[0];
        }
    } else {
        int64_t offset_src = in*args.ofs0 + iic*args.ofs1 + iih*args.IW + iiw;

        while (in < args.N) {
            pdst[offset_dst] = x[offset_src];

            offset_dst += ntg[0]*args.CHW*OH*OW;
            offset_src += ntg[0]*args.ofs0;

            in += ntg[0];
        }
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

// TODO: optimize
typedef void (im2col_ext_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char…50321 tokens truncated…+ yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK4_NL + it*8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 16 * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;

    const short ix = tiisg/16;  // 0 or 1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster
    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_tq2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_tq2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_tq2_0 *) ((device char *) src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    // 8 threads per block, NBLOCK blocks per pass, 2 halves per block per pass
    constexpr short NBLOCK = 4;

    constexpr short NB = N_SIMDWIDTH/NBLOCK; // threads per block

    const short blk = tiisg / NB;    // 0..NBLOCK-1, block handled by this thread
    const short htg = tiisg % NB;    // 0..NB-1, thread within block (0..7)

    // byte and y base offsets within the block (32 elements per thread, 4 per byte)
    device const float4 * yb4 = (device const float4 *)(y + 4*htg + blk*QK_K);

    // hoisted per-byte coefficients (from y) and total y-sum, shared across rows
    // ref: https://github.com/ggml-org/llama.cpp/pull/26980
    float4 coef[4];

    for (int ib = blk; ib < nb; ib += NBLOCK) {
        FOR_UNROLL (short h0 = 0; h0 < 2; ++h0) {
            const float4 y0 = yb4[ 0 + 32*h0];
            const float4 y1 = yb4[ 8 + 32*h0];
            const float4 y2 = yb4[16 + 32*h0];
            const float4 y3 = yb4[24 + 32*h0];

            float sumy = 0.f;
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                coef[j] = float4(
                        y0[j],
                        y1[j] - 4.0f*y0[j],
                        y2[j] - 4.0f*y1[j],
                        y3[j] - 4.0f*y2[j]);

                sumy += (y0[j] + y1[j]) + (y2[j] + y3[j]);
            }

            FOR_UNROLL (short row = 0; row < nr0; ++row) {
                device const block_tq2_0 & xb = ax[row][ib];
                device const uchar * qs = xb.qs + 4*htg + 32*h0;

                float sum = -sumy;
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    // express the 2-bit field shifts (v>>2, v>>4, v>>6) as float floor ops
                    const float v = (float)qs[j];

                    const float f0 = v;
                    const float f1 = floor(v*0.25f);    // v>>2
                    const float f2 = floor(v*0.0625);   // v>>4
                    const float f3 = floor(v*0.015625); // v>>6

                    sum += coef[j][0]*f0 + coef[j][1]*f1 + coef[j][2]*f2 + coef[j][3]*f3;
                }

                sumf[row] += xb.d * sum;
            }
        }

        yb4 += QK_K * NBLOCK / 4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_tq2_0_f32")]]
kernel void kernel_mul_mv_tq2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<N_R0_TQ2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg  [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (device const block_q *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (device      float4x4 *) ((      device char *) dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        float4x4 temp;
        dequantize_func(psrc + ind/nl, ind%nl, temp);
        pdst[ind] = temp;

        break;
    }
}

template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (const device T0 *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (      device T  *) ((      device char *)  dst + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];

        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float, float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half,  float>;
template [[host_name("kernel_get_rows_i32")]]  kernel get_rows_f_t kernel_get_rows_f<int32_t, int32_t>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat, float>;
#endif

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q1_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q1_0,    8, dequantize_q1_0>;
template [[host_name("kernel_get_rows_q2_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_0,    4, dequantize_q2_0>;
template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_mxfp4")]]   kernel get_rows_q_t kernel_get_rows_q<block_mxfp4,   2, dequantize_mxfp4>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;
template [[host_name("kernel_get_rows_tq2_0")]]   kernel get_rows_q_t kernel_get_rows_q<block_tq2_0,   QK_NL, dequantize_tq2_0>;

template<typename TS, typename TI, short QK, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS      * src_row = (const device TS      *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + QK*ind, dst_row[ind]);
    }
}

template<typename TS, typename TI, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q32(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS      * src_row = (const device TS      *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + 32*ind, dst_row[ind]);
    }
}

template<typename TS, typename TI, typename TD>
kernel void kernel_set_rows_f(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device TD * dst_row = (      device TD *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS * src_row = (const device TS *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (TD) src_row[ind];
    }
}

typedef decltype(kernel_set_rows_f<float, int64_t, float>) set_rows_f_t;

template [[host_name("kernel_set_rows_f32_i64_f32")]]   kernel set_rows_f_t kernel_set_rows_f<float, int64_t, float>;
template [[host_name("kernel_set_rows_f32_i32_f32")]]   kernel set_rows_f_t kernel_set_rows_f<float, int32_t, float>;
template [[host_name("kernel_set_rows_f32_i64_f16")]]   kernel set_rows_f_t kernel_set_rows_f<float, int64_t, half>;
template [[host_name("kernel_set_rows_f32_i32_f16")]]   kernel set_rows_f_t kernel_set_rows_f<float, int32_t, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_f32_i64_bf16")]]  kernel set_rows_f_t kernel_set_rows_f<float, int64_t, bfloat>;
template [[host_name("kernel_set_rows_f32_i32_bf16")]]  kernel set_rows_f_t kernel_set_rows_f<float, int32_t, bfloat>;
#endif

template [[host_name("kernel_set_rows_f16_i64_f16")]]   kernel set_rows_f_t kernel_set_rows_f<half, int64_t, half>;
template [[host_name("kernel_set_rows_f16_i32_f16")]]   kernel set_rows_f_t kernel_set_rows_f<half, int32_t, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_bf16_i64_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat, int64_t, bfloat>;
template [[host_name("kernel_set_rows_bf16_i32_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat, int32_t, bfloat>;
#endif

typedef decltype(kernel_set_rows_q32<float, int64_t, block_q8_0, quantize_q8_0>) set_rows_q32_t;

template [[host_name("kernel_set_rows_f32_i64_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_f32_i32_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_f32_i64_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_f32_i32_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_f32_i64_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_f32_i32_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_f32_i64_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_f32_i32_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_f32_i64_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_f32_i32_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_f32_i64_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_iq4_nl, quantize_iq4_nl>;
template [[host_name("kernel_set_rows_f32_i32_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_iq4_nl, quantize_iq4_nl>;

typedef decltype(kernel_set_rows_q<float, int64_t, QK_K, block_tq2_0, quantize_tq2_0>) set_rows_qK_t;

template [[host_name("kernel_set_rows_f32_i64_tq2_0")]]  kernel set_rows_qK_t kernel_set_rows_q<float, int64_t, QK_K, block_tq2_0, quantize_tq2_0>;
template [[host_name("kernel_set_rows_f32_i32_tq2_0")]]  kernel set_rows_qK_t kernel_set_rows_q<float, int32_t, QK_K, block_tq2_0, quantize_tq2_0>;

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // cT.store handles bounds checking via tD's extents (M, N)
    device float * dstBatch = (device float *)dst + im * N * M;

    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    cT.store(tD.slice(ra, rb));
}

#else

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#endif // GGML_METAL_HAS_TENSOR

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
#else
    auto tA = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
#else
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mm.run(sB, sA, cT);
#endif
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef GGML_METAL_HAS_TENSOR
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
#else
    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_disp_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg);

typedef void (kernel_mul_mv2_disp_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv_disp_t disp_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, tgpig, tiisg);
}

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_t_t_disp<half, half, ggml_metal_kargs_mul_mv>>) mul_mv_disp_fn_t;

template<mul_mv_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ggml_metal_kargs_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ggml_metal_kargs_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1, // args.ne02,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02, // args.ne02 == 1
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1, // args.ne11,
        /*.ne12 =*/ 1, // args.ne12,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12, // ne12 == 1
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1, // args.ne1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>) kernel_mul_mv_id_t;

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>) kernel_mul_mv_id_4_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<half,  float>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<bfloat, float>>>;
#endif
template [[host_name("kernel_mul_mv_id_f32_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<half,  half4,  float, float4>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32_4")]]  kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<bfloat, bfloat4, float, float4>>>;
#endif

template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;

template [[host_name("kernel_mul_mv_id_q1_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q1_0_f32_impl<N_R0_Q1_0>>>;
template [[host_name("kernel_mul_mv_id_q2_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0>>>;
template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1>>>;

template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;

template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl   <N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl   <N_R0_Q3_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl   <N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl   <N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl   <N_R0_Q6_K>>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl  <N_R0_IQ1_S>>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl  <N_R0_IQ1_M>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl <N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl  <N_R0_IQ3_S>>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl  <N_R0_IQ2_S>>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl <N_R0_IQ4_NL>>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl <N_R0_IQ4_XS>>>;
template [[host_name("kernel_mul_mv_id_tq2_0_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_tq2_0_f32_impl  <N_R0_TQ2_0>>>;

kernel void kernel_pool_2d_max_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * args.IW + j]);
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}

kernel void kernel_pool_2d_avg_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    const float scale = 1. / (args.k0 * args.k1);

    float res = 0;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            float cur = i_ptr[i * args.IW + j];
            res += cur * scale;
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}


kernel void kernel_pool_1d_max_f32(
        constant        ggml_metal_kargs_pool_1d & args,
        device  const   float * src,
        device          float * dst,
        uint            gid [[thread_position_in_grid]]
) {

    if (gid >= args.np) {
        return;
    }

    const int ow  = (int)gid % args.OW;
    const int row = (int)gid / args.OW;

    const int base = ow * args.s0 - args.p0;

    float acc = -INFINITY;

    const int src_off = row * args.IW;
    const int dst_off = row * args.OW;

    for (int ki = 0; ki < args.k0; ++ki) {
        int j = base + ki;
        if (j < 0 || j >= args.IW){
            continue;
        }
        float v = src[src_off + j];
        acc = max(acc, v);
    }

    dst[dst_off + ow] = acc;
}

kernel void kernel_pool_1d_avg_f32(
        constant        ggml_metal_kargs_pool_1d & args,
        device  const   float * src,
        device          float * dst,
        uint            gid [[thread_position_in_grid]]
) {

    if (gid >= args.np) {
        return;
    }

    const int ow  = (int)gid % args.OW;
    const int row = (int)gid / args.OW;

    const int base = ow * args.s0 - args.p0;

    float acc = 0.0f;
    int   cnt = 0;

    const int src_off = row * args.IW;
    const int dst_off = row * args.OW;

    for (int ki = 0; ki < args.k0; ++ki) {
        const int j = base + ki;
        if (j < 0 || j >= args.IW) {
            continue;
        }
        acc += src[src_off + j];
        cnt += 1;
    }

    dst[dst_off + ow] = (cnt > 0) ? (acc / (float)cnt) : 0.0f;
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
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
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
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

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

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

// retro delta: L2-norm backward for LoRA training (Qwen3.5 gated delta net
// k/q normalization). src0 = dz (grad of output), src1 = x (forward input),
// same shape. Unlike RMS norm, l2_norm floors its forward scale at 1/eps
// instead of adding eps under the sqrt, so the two regimes have different
// gradients:
//   norm > eps:  dx = (dz - x * (sum_xdz / sum_xx)) / norm
//   norm <= eps: dx = dz / eps  (scale is a local constant, no cross term)
// One threadgroup per row; mirrors kernel_rms_norm_back_f32.
kernel void kernel_l2_norm_back_f32(
        constant ggml_metal_kargs_l2_norm_back & args,
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

    const float norm = sqrt(sum_xx);

    float scale_g;
    float scale_x;
    if (norm > args.eps) {
        scale_g = 1.0f / norm;
        scale_x = -scale_g * sum_xdz / sum_xx;
    } else {
        scale_g = 1.0f / args.eps;
        scale_x = 0.0f;
    }

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dx[i00] = dz[i00] * scale_g + x[i00] * scale_x;
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
