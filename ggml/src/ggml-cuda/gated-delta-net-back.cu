#include "gated-delta-net-back.cuh"

#include <cstdint>
#include <cstdlib>
#include <vector>

// retro delta: analytic backward for GGML_OP_GATED_DELTA_NET, mirroring the
// CPU reference (ggml-cpu/ops.cpp: ggml_compute_forward_gated_delta_net_back_f32).
// One *block* per (head, sequence) unit recomputes the S_prev trajectory
// forward, then reverse-scans it. The token scan is inherently sequential, but
// every step inside it is O(S_v^2) and is spread across the block: the state
// matrices are walked flat (coalesced, thread-stride), the reductions over the
// contiguous `i` axis use one warp per column `j`, and the reductions over `j`
// give each thread a whole row `i` (also coalesced, since threads then differ
// only in the contiguous index).
//
// An earlier revision ran one *thread* per (head, sequence). For a typical
// LoRA ubatch that is H_v*n_seqs threads in total -- 64 for Qwen3.5 at
// n_seqs=4 -- i.e. two warps for the whole GPU, each serially grinding
// n_tokens * S_v^2 scalar FLOPs with fully uncoalesced access. That kernel
// measured 3.7 s per launch and 99% of training wall-clock.
//
// grad_q/grad_k can be shared by several v-heads when q/k are GQA-broadcast
// (H_v % H_qk == 0), so those two outputs use atomicAdd; every other output
// (grad_v/grad_g/grad_beta/grad_state) is unique per (head, seq) and written
// directly.
//
// Memory note: this formulation keeps the full per-token state trajectory
// (n_tokens * S_v * S_v floats per unit) live at once, so the pool scratch
// scales with n_tokens * S_v^2 * H * n_seqs -- 1.00 GiB on the profiled model.
// That, and the latency chain of the token scan, is why the chunkwise
// formulation in the second half of this file is the default; this one stays as
// its reference and as what runs when it is switched off.

#define GDN_BACK_BLOCK 256

// Sum `val` across the block. `red` holds one float per warp. Safe to call
// repeatedly: the trailing barrier keeps a later call from overwriting `red`
// while earlier readers are still using it.
static __device__ __forceinline__ float gdn_block_reduce_sum(float val, float * red) {
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

// out[j] = sum_i M[i + j*S_v] * vec[i], one warp per column so the reduced
// axis is the contiguous one and every warp read is coalesced.
static __device__ __forceinline__ void gdn_col_dot(
        const float * __restrict__ M, const float * __restrict__ vec,
        float * __restrict__ out, const int64_t S_v) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

    for (int64_t j = wid; j < S_v; j += nw) {
        float acc = 0.0f;
        for (int64_t i = lane; i < S_v; i += WARP_SIZE) {
            acc += M[i + j*S_v] * vec[i];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            out[j] = acc;
        }
    }
}

// Everything the two formulations share: the geometry of one node, where a
// (head, sequence) unit's operands and gradients live inside it, and the two
// per-token steps of the recurrence. The chunkwise path below uses the steps as
// its numerical fallback, so they are written once here rather than twice.
struct gdn_geom {
    int64_t S_v, H, n_tokens, n_seqs, units, C;
    int64_t neq1, nek1, rq3, rk3;
    int64_t sq1, sq2, sq3;
    int64_t sk1, sk2, sk3;
    int64_t sv1, sv2, sv3;
    int64_t sg1, sg2, sg3;
    int64_t sb1, sb2, sb3;
    int64_t n_q, n_k, n_v, n_g, n_beta;
    int64_t state_seq_stride;
    float   scale;
    int     K;
    bool    kda;
};

// Every pointer one unit needs, resolved once: the inputs offset to the unit's
// token 0 (per-token strides live in gdn_geom) and the six gradient blocks
// packed in dst.
struct gdn_unit_ptrs {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    float * dq;
    float * dk;
    float * dv;
    float * dg;
    float * dbeta;
};

static __device__ __forceinline__ gdn_unit_ptrs gdn_unit(
        const gdn_geom & G, int64_t unit,
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        float * dst) {
    const int64_t iv1 = unit % G.H;
    const int64_t iv3 = unit / G.H;
    const int64_t iq1 = iv1 % G.neq1;
    const int64_t ik1 = iv1 % G.nek1;
    const int64_t iq3 = iv3 / G.rq3;
    const int64_t ik3 = iv3 / G.rk3;

    // Each kernel passes only the tensors it touches, so every group is
    // resolved on its own; the others stay null rather than being offset from
    // one.
    gdn_unit_ptrs p = {};
    p.q    = q    ? q    + iq3*G.sq3 + iq1*G.sq1 : nullptr;
    p.k    = k    ? k    + ik3*G.sk3 + ik1*G.sk1 : nullptr;
    p.v    = v    ? v    + iv3*G.sv3 + iv1*G.sv1 : nullptr;
    p.g    = g    ? g    + iv3*G.sg3 + iv1*G.sg1 : nullptr;
    p.beta = beta ? beta + iv3*G.sb3 + iv1*G.sb1 : nullptr;
    if (dst) {
        float * g_q    = dst;
        float * g_k    = g_q + G.n_q;
        float * g_v    = g_k + G.n_k;
        float * g_g    = g_v + G.n_v;
        float * g_beta = g_g + G.n_g;
        p.dq    = g_q + G.S_v*(iq1 + G.neq1*G.n_tokens*iq3);
        p.dk    = g_k + G.S_v*(ik1 + G.nek1*G.n_tokens*ik3);
        p.dv    = g_v + G.S_v*(iv1 + G.H*G.n_tokens*iv3);
        p.dg    = G.kda ? g_g + G.S_v*(iv1 + G.H*G.n_tokens*iv3)
                        : g_g +        (iv1 + G.H*G.n_tokens*iv3);
        p.dbeta = g_beta + (iv1 + G.H*G.n_tokens*iv3);
    }
    return p;
}

static __device__ __forceinline__ int64_t gdn_dq_stride   (const gdn_geom & G) { return G.S_v*G.neq1; }
static __device__ __forceinline__ int64_t gdn_dk_stride   (const gdn_geom & G) { return G.S_v*G.nek1; }
static __device__ __forceinline__ int64_t gdn_dv_stride   (const gdn_geom & G) { return G.S_v*G.H;    }
static __device__ __forceinline__ int64_t gdn_dg_stride   (const gdn_geom & G) { return G.kda ? G.S_v*G.H : G.H; }
static __device__ __forceinline__ int64_t gdn_dbeta_stride(const gdn_geom & G) { return G.H; }

// The unit's slice of the recurrent state tensor, which is laid out per
// sequence where every working buffer here is laid out per unit.
static __device__ __forceinline__ int64_t gdn_state_off(const gdn_geom & G, int64_t unit) {
    return (unit/G.H)*G.state_seq_stride + (unit % G.H)*G.S_v*G.S_v;
}

// One token's row of the upstream attention-score gradient.
static __device__ __forceinline__ const float * gdn_dout_row(
        const gdn_geom & G, int64_t unit, const float * grad_attn, int64_t t) {
    const int64_t iv1 = unit % G.H;
    const int64_t iv3 = unit / G.H;
    return grad_attn + (iv3*G.n_tokens*G.H + t*G.H + iv1)*G.S_v;
}

// Per-token scratch, one set per block, carved out of the dynamic shared block.
struct gdn_seq_smem {
    float * k;
    float * v;
    float * q;
    float * dout;
    float * gexp;
    float * pre;
    float * delta;
    float * dpre;
    float * ddelta;
    float * red;
};

static __device__ __forceinline__ gdn_seq_smem gdn_seq_smem_bind(float * raw, int64_t S_v) {
    gdn_seq_smem s;
    s.k      = raw;
    s.v      = s.k      + S_v;
    s.q      = s.v      + S_v;
    s.dout   = s.q      + S_v;
    s.gexp   = s.dout   + S_v;
    s.pre    = s.gexp   + S_v;
    s.delta  = s.pre    + S_v;
    s.dpre   = s.delta  + S_v;
    s.ddelta = s.dpre   + S_v;
    s.red    = s.ddelta + S_v;
    return s;
}

static size_t gdn_seq_smem_bytes(int64_t S_v, int block) {
    return (9*(size_t) S_v + block/WARP_SIZE)*sizeof(float);
}

// One token of the forward recurrence, spread over the block:
//   S1    = Diag(exp(g_t)) S_in
//   delta = beta_t (v_t - S1^T k_t)
//   S_out = S1 + outer(k_t, delta)
// `S_in` and `S_out` may alias. On exit S1, and the gexp/k/v/pre/delta entries
// of `sm`, hold what the backward step of the same token needs.
static __device__ void gdn_step_forward_dev(
        const gdn_geom & G, const gdn_unit_ptrs & p, int64_t t,
        const float * S_in, float * S1, float * S_out, const gdn_seq_smem & sm) {
    const int64_t S    = G.S_v;
    const int64_t SS   = S*S;
    const int     tid  = threadIdx.x;
    const int     nthr = blockDim.x;

    // Flat walk of an S_v*S_v matrix keeping (i, j) in step without a modulo
    // in the inner loop.
    const int64_t i0     = tid % S;
    const int64_t j0     = tid / S;
    const int64_t i_step = nthr % S;
    const int64_t j_step = nthr / S;

    const float beta_val = p.beta[t*G.sb2];

    for (int64_t i = tid; i < S; i += nthr) {
        sm.k[i]    = p.k[t*G.sk2 + i];
        sm.v[i]    = p.v[t*G.sv2 + i];
        sm.gexp[i] = expf(G.kda ? p.g[t*G.sg2 + i] : p.g[t*G.sg2]);
    }
    __syncthreads();

    for (int64_t n = tid, i = i0; n < SS; n += nthr) {
        S1[n] = S_in[n] * sm.gexp[i];
        i += i_step;
        if (i >= S) i -= S;
    }
    __syncthreads();

    gdn_col_dot(S1, sm.k, sm.pre, S);
    __syncthreads();

    for (int64_t j = tid; j < S; j += nthr) {
        sm.delta[j] = (sm.v[j] - sm.pre[j]) * beta_val;
    }
    __syncthreads();

    for (int64_t n = tid, i = i0, j = j0; n < SS; n += nthr) {
        S_out[n] = S1[n] + sm.k[i] * sm.delta[j];
        i += i_step;
        j += j_step;
        if (i >= S) { i -= S; ++j; }
    }
    __syncthreads();
}

// One token of the reverse scan. `S_prev` is the state before token t; `dS`
// holds dL/dS_new on entry and dL/dS_prev on exit; S1/Snew/dS1 are S_v*S_v of
// scratch. `d_snap` is the rollback-snapshot gradient this token's state
// receives, or null when the caller has already folded it into `dS`.
static __device__ void gdn_step_back_dev(
        const gdn_geom & G, const gdn_unit_ptrs & p, int64_t t,
        const float * grad_attn_row, const float * d_snap,
        const float * S_prev, float * S1, float * Snew, float * dS1, float * dS,
        const gdn_seq_smem & sm) {
    const int64_t S    = G.S_v;
    const int64_t SS   = S*S;
    const int     tid  = threadIdx.x;
    const int     nthr = blockDim.x;
    const int64_t i0     = tid % S;
    const int64_t j0     = tid / S;
    const int64_t i_step = nthr % S;
    const int64_t j_step = nthr / S;

    // Recompute S1, gexp, pre, delta and S_new for this token.
    gdn_step_forward_dev(G, p, t, S_prev, S1, Snew, sm);

    const float beta_val = p.beta[t*G.sb2];

    for (int64_t i = tid; i < S; i += nthr) {
        sm.q[i]    = p.q[t*G.sq2 + i];
        sm.dout[i] = grad_attn_row[i];
    }
    __syncthreads();

    // dS += scale * outer(q, d_out); grad_q = scale * S_new . d_out
    float * dq_out = p.dq + t*gdn_dq_stride(G);
    for (int64_t i = tid; i < S; i += nthr) {
        const float q_i = sm.q[i];
        float dqi = 0.0f;
        for (int64_t j = 0; j < S; ++j) {
            dS[i + j*S] += G.scale * sm.dout[j] * q_i;
            dqi         += Snew[i + j*S] * sm.dout[j];
        }
        atomicAdd(&dq_out[i], G.scale * dqi);
    }
    __syncthreads();

    if (d_snap) {
        for (int64_t n = tid; n < SS; n += nthr) {
            dS[n] += d_snap[n];
        }
        __syncthreads();
    }

    // step 3 backward: S_new = S1 + outer(k, delta)
    for (int64_t n = tid; n < SS; n += nthr) {
        dS1[n] = dS[n];
    }
    gdn_col_dot(dS, sm.k, sm.ddelta, S);
    float * dk_out = p.dk + t*gdn_dk_stride(G);
    for (int64_t i = tid; i < S; i += nthr) {
        float dki = 0.0f;
        for (int64_t j = 0; j < S; ++j) {
            dki += dS[i + j*S] * sm.delta[j];
        }
        atomicAdd(&dk_out[i], dki);
    }
    __syncthreads();

    // step 2 backward: delta[j] = beta*(v[j] - pre[j])
    float dbeta_partial = 0.0f;
    for (int64_t j = tid; j < S; j += nthr) {
        const float dd = sm.ddelta[j];
        sm.dpre[j] = -dd * beta_val;
        dbeta_partial += dd * (sm.v[j] - sm.pre[j]);
        p.dv[t*gdn_dv_stride(G) + j] += dd * beta_val;
    }
    const float dbeta_t = gdn_block_reduce_sum(dbeta_partial, sm.red);
    if (tid == 0) {
        p.dbeta[t*gdn_dbeta_stride(G)] += dbeta_t;
    }
    __syncthreads();

    for (int64_t n = tid, i = i0, j = j0; n < SS; n += nthr) {
        dS1[n] += sm.dpre[j] * sm.k[i];
        i += i_step;
        j += j_step;
        if (i >= S) { i -= S; ++j; }
    }
    __syncthreads();

    for (int64_t i = tid; i < S; i += nthr) {
        float dki2 = 0.0f;
        for (int64_t j = 0; j < S; ++j) {
            dki2 += sm.dpre[j] * S1[i + j*S];
        }
        atomicAdd(&dk_out[i], dki2);
    }
    __syncthreads();

    // step 1 backward: S1[i,j] = S_prev[i,j] * gexp[i]
    if (G.kda) {
        float * dg_out = p.dg + t*gdn_dg_stride(G);
        for (int64_t i = tid; i < S; i += nthr) {
            const float ge = sm.gexp[i];
            float dgexp_i = 0.0f;
            for (int64_t j = 0; j < S; ++j) {
                const int64_t idx = i + j*S;
                dgexp_i += dS1[idx] * S_prev[idx];
                dS[idx]  = dS1[idx] * ge;
            }
            dg_out[i] += dgexp_i * ge;
        }
    } else {
        const float ge = sm.gexp[0];
        float dgexp_partial = 0.0f;
        for (int64_t n = tid; n < SS; n += nthr) {
            dgexp_partial += dS1[n] * S_prev[n];
            dS[n] = dS1[n] * ge;
        }
        const float dgexp_sum = gdn_block_reduce_sum(dgexp_partial, sm.red);
        if (tid == 0) {
            p.dg[t*gdn_dg_stride(G)] += dgexp_sum * ge;
        }
    }
    __syncthreads();
}

// The whole-sequence sequential scan: one block per (head, sequence) unit
// recomputes the S_prev trajectory forward, then reverse-scans it. Kept as the
// reference the chunkwise path is checked against, and as what runs when the
// chunkwise path is switched off or does not apply.
static __global__ __launch_bounds__(GDN_BACK_BLOCK) void gated_delta_net_back_kernel(
        const gdn_geom G,
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        const float * __restrict__ state0, const float * __restrict__ grad,
        float * __restrict__ dst,
        float * __restrict__ scratch, const int64_t scratch_stride) {
    const int64_t unit = blockIdx.x;
    if (unit >= G.units) {
        return;
    }
    const int     tid  = threadIdx.x;
    const int     nthr = blockDim.x;
    const int64_t S    = G.S_v;
    const int64_t SS   = S*S;

    const gdn_unit_ptrs p = gdn_unit(G, unit, q, k, v, g, beta, dst);

    const float * grad_attn  = grad;
    const float * grad_state = grad + S*G.H*G.n_tokens*G.n_seqs;
    const int64_t state_size_per_snap = SS*G.units;

    float * base = scratch + unit * scratch_stride;
    float * traj = base;                        // n_tokens*S_v*S_v : S_prev per token
    float * S1   = traj + G.n_tokens * SS;      // S_v*S_v
    float * Snew = S1 + SS;                     // S_v*S_v
    float * dS   = Snew + SS;                   // S_v*S_v
    float * dS1  = dS + SS;                     // S_v*S_v

    extern __shared__ float smem[];
    const gdn_seq_smem sm = gdn_seq_smem_bind(smem, S);

    const float * s0 = state0 + gdn_state_off(G, unit);
    for (int64_t n = tid; n < SS; n += nthr) {
        traj[n] = s0[n];
    }
    __syncthreads();

    for (int64_t t = 0; t < G.n_tokens - 1; ++t) {
        gdn_step_forward_dev(G, p, t, traj + t*SS, S1, traj + (t + 1)*SS, sm);
    }

    for (int64_t n = tid; n < SS; n += nthr) {
        dS[n] = 0.0f;
    }
    __syncthreads();

    for (int64_t t = G.n_tokens - 1; t >= 0; --t) {
        const int64_t slot = G.n_tokens - 1 - t;
        const float * d_snap = slot < G.K
                ? grad_state + slot*state_size_per_snap + unit*SS
                : nullptr;
        gdn_step_back_dev(G, p, t, gdn_dout_row(G, unit, grad_attn, t), d_snap,
                          traj + t*SS, S1, Snew, dS1, dS, sm);
    }

    float * g_state = dst + G.n_q + G.n_k + G.n_v + G.n_g + G.n_beta;
    float * gs_out  = g_state + gdn_state_off(G, unit);
    for (int64_t n = tid; n < SS; n += nthr) {
        gs_out[n] += dS[n];
    }
}

// ===========================================================================
// Chunkwise backward -- docs/optims/OPTIMS_V4.md part A.
//
// The kernel above is correct and 1200x slower than its own forward: it is one
// block per (head, sequence) walking the tokens one at a time, so every one of
// n_tokens steps is a chain of block-wide reductions and __syncthreads on
// ~100 kFLOP of work, on H*n_seqs blocks. Neither compute- nor bandwidth-bound;
// a latency chain. It measured 88% of all GPU time on Qwen3.5-0.8B.
//
// This path changes the algorithm instead of the mapping. Over a chunk of C
// tokens the recurrence closes in six matrix products and one unit-triangular
// solve (derivation, notation and adjoints: see the long comment on
// ggml_compute_forward_gated_delta_net_back_chunked_f32 in ggml-cpu/ops.cpp --
// that CPU function is this code's oracle, and tests/gated_delta_net_chunked.rs
// validates the two against the sequential scan). What is left sequential is
// chunk to chunk, i.e. n_tokens/C steps, and every step is a batched GEMM over
// all H*n_seqs units at once. The per-token state trajectory disappears with
// it: only one entry state per chunk stays live, which is the 1.00 GiB of
// scratch this op used to hold divided by C.
//
// Two things are host-side decisions rather than kernel ones:
//
//   * The chunk layout is uniform across units, so every batched GEMM has one
//     shape and one stride. K > 1 injects a state gradient at the last K tokens,
//     so those become chunks of one token.
//   * The decay normalisation khat = k/A overflows F32 over a long run of
//     strongly negative gates, so the layout is also bounded by the gates:
//     k_gdn_gate_worst reduces g to one number per token -- the largest |g| over
//     every unit and channel -- and the host then walks that array greedily,
//     closing a chunk before the running sum passes GDN_CHUNK_LOG_LIMIT
//     (|sum g| <= sum |g|, so this is a valid bound on 1/A). One token whose own
//     gate exceeds even the single-token bound sends the node to the sequential
//     kernel; nothing in a trained model comes close.
//
//     This costs one n_tokens-float copy back per node -- microseconds against
//     the 101 ms it replaces -- and it matters more than it looks: an earlier
//     version halved one *global* chunk length until the worst chunk in the
//     batch fit, and on Qwen3.5-0.8B that collapsed C from 64 to ~2, because a
//     handful of tokens carry gates two orders of magnitude above the median.
//     Bounding per token isolates those tokens into short chunks and leaves the
//     rest long. The CPU reference splits per (unit, chunk) instead, which is
//     finer still but needs a per-unit layout no batched GEMM can use.
//
//     The device-to-host copy that this implies drags a cudaStreamSynchronize
//     into the middle of the graph, once per GDN node. That looked like the
//     obvious thing to remove -- Vulkan does not pay it, because its layout
//     depends only on (T, C, K) and each workgroup checks its own gates on the
//     GPU (see gated_delta_net_back_chunked.comp). Porting that here was tried
//     and reverted, and the reason is worth keeping: with a batched GEMM the
//     device-side check cannot *shorten* a chunk, only push the whole
//     (chunk, unit) pair onto a per-token fallback, and on this model the gates
//     put 1 to 43 pairs out of 256 over the bound on a typical node. A token
//     step here costs ~0.8 ms, so a fallback chunk costs seconds and the node
//     regresses by ~50x. The layout has to adapt to the gates, and adapting it
//     is what needs the gates on the host. See docs/optims/OPTIMS_V4.md A.9.
// ===========================================================================

#define GDN_CHUNK_BLOCK      256
#define GDN_CHUNK_DEFAULT     64
#define GDN_CHUNK_MAX        128
// 60*ln(2): 1/A <= 2^60 inside a chunk of two tokens or more. See the numerics
// note in ops.cpp. A chunk of one token needs no cross-token ratio, so it only
// has to keep 1/A itself and the (i)^T products representable: 80*ln(2) leaves
// ~14 decades of F32 headroom there.
#define GDN_CHUNK_LOG_LIMIT         41.5888308f
#define GDN_CHUNK_SINGLE_LOG_LIMIT  55.4518f

// The chunk layout, uniform across units. Three things bound a chunk: the
// requested length C; a state snapshot, whose gradient is injected at the
// chunk's *last* token, so with K > 1 the last K tokens are chunks of one; and
// the gates, through the running sum of `worst` (see the block comment). Returns
// empty when one token's own gate is past the single-token bound, i.e. no layout
// is safe and the caller must run the sequential kernel.
static std::vector<int64_t> gdn_chunk_layout(int64_t n_tokens, int64_t C, int64_t K,
                                             const float * worst) {
    std::vector<int64_t> starts;
    for (int64_t t = 0; t < n_tokens; ) {
        int64_t limit = MIN(C, n_tokens - t);
        if (K > 1) {
            const int64_t first_snap = MAX(t, n_tokens - K);
            if (first_snap < t + limit - 1) {
                limit = first_snap - t + 1;
            }
        }
        float   sum = 0.0f;
        int64_t L   = 0;
        while (L < limit) {
            const float w = worst[t + L];
            if (L > 0 && sum + w > GDN_CHUNK_LOG_LIMIT) {
                break;
            }
            sum += w;
            ++L;
        }
        if (L == 1 && worst[t] > GDN_CHUNK_SINGLE_LOG_LIMIT) {
            return {};
        }
        starts.push_back(t);
        t += L;
    }
    starts.push_back(n_tokens);
    return starts;
}

// g reduced to one number per token: the largest |g| over every unit and gate
// channel. That is all the host needs to bound 1/A over any candidate chunk
// (|sum g| <= sum |g|), and unlike a per-chunk scan it is fully parallel.
// Non-negative floats compare like their bit patterns, hence the integer
// atomicMax.
static __global__ void k_gdn_gate_worst(const gdn_geom G, const float * __restrict__ g,
                                        float * __restrict__ worst) {
    const int64_t unit = blockIdx.y;
    const int64_t nch  = G.kda ? G.S_v : 1;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= G.n_tokens*nch) {
        return;
    }
    const int64_t t   = n / nch;
    const int64_t ch  = n - t*nch;
    const gdn_unit_ptrs p = gdn_unit(G, unit, nullptr, nullptr, nullptr, g, nullptr, nullptr);
    atomicMax((int *) &worst[t], __float_as_int(fabsf(p.g[t*G.sg2 + ch])));
}

// A (cumulative decay relative to the chunk start), qtil = q*A, ktil = k*A,
// khat = k/A and beta, gathered into contiguous [C, S_v] tiles. One thread per
// channel: the cumulative product is sequential over tokens but independent
// across channels, and threads differ only in the contiguous index.
static __global__ void k_gdn_gather(const gdn_geom G, int64_t t0, int64_t L,
        const float * __restrict__ q, const float * __restrict__ k,
        const float * __restrict__ g, const float * __restrict__ beta,
        float * __restrict__ A, float * __restrict__ Qt, float * __restrict__ Kt,
        float * __restrict__ Kh, float * __restrict__ bet) {
    const int64_t unit = blockIdx.x;
    const int64_t S    = G.S_v;
    const gdn_unit_ptrs p = gdn_unit(G, unit, q, k, nullptr, g, beta, nullptr);

    float * Au  = A  + unit*G.C*S;
    float * Qtu = Qt + unit*G.C*S;
    float * Ktu = Kt + unit*G.C*S;
    float * Khu = Kh + unit*G.C*S;

    for (int64_t i = threadIdx.x; i < S; i += blockDim.x) {
        float acc = 1.0f;
        for (int64_t r = 0; r < L; ++r) {
            const int64_t t = t0 + r;
            acc *= expf(G.kda ? p.g[t*G.sg2 + i] : p.g[t*G.sg2]);
            const float kv = p.k[t*G.sk2 + i];
            Au [r*S + i] = acc;
            Qtu[r*S + i] = p.q[t*G.sq2 + i] * acc;
            Ktu[r*S + i] = kv * acc;
            Khu[r*S + i] = kv / acc;
        }
    }
    for (int64_t r = threadIdx.x; r < L; r += blockDim.x) {
        bet[unit*G.C + r] = p.beta[(t0 + r)*G.sb2];
    }
}

// (iii) What = beta (V - Z), keeping V - Z for beta's own gradient later.
// `Z` and `VmZ` are the same buffer -- the GEMM writes Z where V - Z will live,
// and each thread reads and writes only its own element -- so neither is
// declared __restrict__.
static __global__ void k_gdn_make_w(const gdn_geom G, int64_t t0, int64_t L,
        const float * __restrict__ v, const float * Z,
        const float * __restrict__ bet, float * VmZ, float * __restrict__ W) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t S    = G.S_v;
    if (n >= L*S) {
        return;
    }
    const int64_t r = n / S;
    const int64_t j = n - r*S;
    const gdn_unit_ptrs p = gdn_unit(G, unit, nullptr, nullptr, v, nullptr, nullptr, nullptr);
    const float d = p.v[(t0 + r)*G.sv2 + j] - Z[unit*G.C*S + n];
    VmZ[unit*G.C*S + n] = d;
    W  [unit*G.C*S + n] = bet[unit*G.C + r] * d;
}

// (iv) T = strict_lower(beta (ktil khat^T)). The masked entries are *written*,
// never multiplied by zero: the discarded upper triangle is exactly where 1/A
// piles up, and the triangular solve must not read it.
static __global__ void k_gdn_mask_t(const gdn_geom G, int64_t L,
        const float * __restrict__ M, const float * __restrict__ bet, float * __restrict__ T) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L*L) {
        return;
    }
    const int64_t r = n / L;
    const int64_t c = n - r*L;
    const int64_t o = unit*G.C*G.C + r*G.C + c;
    T[o] = c < r ? bet[unit*G.C + r] * M[o] : 0.0f;
}

// P and dP keep their inclusive lower triangle -- (4) reads the state after (3).
static __global__ void k_gdn_mask_lower_incl(const gdn_geom G, int64_t L, float * __restrict__ M) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L*L) {
        return;
    }
    const int64_t r = n / L;
    const int64_t c = n - r*L;
    if (c > r) {
        M[unit*G.C*G.C + r*G.C + c] = 0.0f;
    }
}

// (viii) S_C = Diag(A_C)(S0 + khat^T U). The GEMM leaves khat^T U in XT; this
// adds S0 in place (XT is what the dA_C reduction needs) and scales out the
// chunk's exit state.
static __global__ void k_gdn_state_out(const gdn_geom G, int64_t L,
        const float * __restrict__ A, const float * __restrict__ S0T,
        float * __restrict__ XT, float * __restrict__ SnextT) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t S    = G.S_v;
    if (n >= S*S) {
        return;
    }
    const int64_t i = n % S;
    const int64_t o = unit*S*S + n;
    const float   x = XT[o] + S0T[o];
    XT[o] = x;
    if (SnextT) {
        SnextT[o] = A[unit*G.C*S + (L - 1)*S + i] * x;
    }
}

// (viii)^T, the half that reads dS_C: G = Diag(A_C) dS_C elementwise, and
// dA_C = rowsum_j(dS_C * (S0 + khat^T U)) as a reduction over the value axis.
// Splitting it from the copy of G into the dS_0 accumulator is what keeps dS_C
// readable until both are done.
static __global__ void k_gdn_grad_state_in(const gdn_geom G, int64_t L,
        const float * __restrict__ A, const float * __restrict__ dST,
        const float * __restrict__ XT, float * __restrict__ GT, float * __restrict__ dA) {
    const int64_t unit = blockIdx.x;
    const int64_t S    = G.S_v;
    const float * Alast = A + unit*G.C*S + (L - 1)*S;

    for (int64_t i = threadIdx.x; i < S; i += blockDim.x) {
        for (int64_t r = 0; r < L - 1; ++r) {
            dA[unit*G.C*S + r*S + i] = 0.0f;
        }
        float acc = 0.0f;
        for (int64_t j = 0; j < S; ++j) {
            const int64_t o = unit*S*S + j*S + i;
            GT[o] = dST[o] * Alast[i];
            acc  += dST[o] * XT[o];
        }
        dA[unit*G.C*S + (L - 1)*S + i] = acc;
    }
}

// (iii)^T: dV, dZ, and beta's contribution from What. One block per (unit,
// token), so dbeta is a single block reduction over the value axis.
static __global__ __launch_bounds__(GDN_CHUNK_BLOCK) void k_gdn_dvz(
        const gdn_geom G, int64_t t0, int64_t L,
        const float * __restrict__ dW, const float * __restrict__ VmZ,
        const float * __restrict__ bet, float * __restrict__ dZ, float * __restrict__ dst) {
    const int64_t r    = blockIdx.x;
    const int64_t unit = blockIdx.y;
    const int64_t S    = G.S_v;
    const gdn_unit_ptrs p = gdn_unit(G, unit, nullptr, nullptr, nullptr, nullptr, nullptr, dst);
    const float b = bet[unit*G.C + r];

    float * dv_out = p.dv + (t0 + r)*gdn_dv_stride(G);
    float partial = 0.0f;
    for (int64_t j = threadIdx.x; j < S; j += blockDim.x) {
        const int64_t o  = unit*G.C*S + r*S + j;
        const float   dw = dW[o];
        dv_out[j] += b * dw;
        partial   += dw * VmZ[o];
        dZ[o]      = -b * dw;
    }
    __shared__ float red[GDN_CHUNK_BLOCK/WARP_SIZE];
    const float total = gdn_block_reduce_sum(partial, red);
    if (threadIdx.x == 0) {
        p.dbeta[(t0 + r)*gdn_dbeta_stride(G)] += total;
    }
}

// (iv)^T: dT = -strict_lower(dWhat U^T) is already in dT; read off beta's other
// contribution, then scale the strictly lower part by beta in place.
static __global__ __launch_bounds__(GDN_CHUNK_BLOCK) void k_gdn_dt(
        const gdn_geom G, int64_t t0, int64_t L,
        const float * __restrict__ M, const float * __restrict__ bet,
        float * __restrict__ dT, float * __restrict__ dst) {
    const int64_t r    = blockIdx.x;
    const int64_t unit = blockIdx.y;
    const gdn_unit_ptrs p = gdn_unit(G, unit, nullptr, nullptr, nullptr, nullptr, nullptr, dst);
    const float b = bet[unit*G.C + r];

    float partial = 0.0f;
    for (int64_t c = threadIdx.x; c < L; c += blockDim.x) {
        const int64_t o = unit*G.C*G.C + r*G.C + c;
        if (c < r) {
            const float d = dT[o];
            partial += d * M[o];
            dT[o]    = b * d;
        } else {
            dT[o] = 0.0f;
        }
    }
    __shared__ float red[GDN_CHUNK_BLOCK/WARP_SIZE];
    const float total = gdn_block_reduce_sum(partial, red);
    if (threadIdx.x == 0) {
        p.dbeta[(t0 + r)*gdn_dbeta_stride(G)] += total;
    }
}

// (i)^T: undo the three decay-scaled copies of q and k, then turn dA into dg
// with a reverse cumulative sum -- A_s depends on every gate up to s, so dg_t is
// the tail sum from t. One channel per thread across the whole token loop, which
// is why the chunked path requires S_v <= GDN_CHUNK_BLOCK.
//
// grad_q / grad_k go through atomicAdd because several v-heads can share one
// GQA-broadcast q/k head, i.e. several *blocks* target the same row.
static __global__ __launch_bounds__(GDN_CHUNK_BLOCK) void k_gdn_finalize(
        const gdn_geom G, int64_t t0, int64_t L,
        const float * __restrict__ q, const float * __restrict__ k,
        const float * __restrict__ A, const float * __restrict__ dQt,
        const float * __restrict__ dKt, const float * __restrict__ dKh,
        float * __restrict__ dA, float * __restrict__ dst) {
    const int64_t unit = blockIdx.x;
    const int64_t S    = G.S_v;
    const int64_t i    = threadIdx.x;
    const bool    live = i < S;
    const gdn_unit_ptrs p = gdn_unit(G, unit, q, k, nullptr, nullptr, nullptr, dst);
    const int64_t base = unit*G.C*S;

    for (int64_t r = 0; r < L; ++r) {
        if (!live) {
            break;
        }
        const int64_t t   = t0 + r;
        const int64_t n   = base + r*S + i;
        const float   a   = A[n];
        const float   inv = 1.0f/a;
        const float   kv  = p.k[t*G.sk2 + i];
        const float   qv  = p.q[t*G.sq2 + i];
        atomicAdd(&p.dk[t*gdn_dk_stride(G) + i], dKt[n]*a + dKh[n]*inv);
        atomicAdd(&p.dq[t*gdn_dq_stride(G) + i], dQt[n]*a);
        // ((dkhat*k)*inv)*inv, never (k/a^2): dkhat itself carries an A, so this
        // order keeps both factors inside F32 range.
        dA[n] += dKt[n]*kv + dQt[n]*qv - ((dKh[n]*kv)*inv)*inv;
    }

    __shared__ float red[GDN_CHUNK_BLOCK/WARP_SIZE];
    float run = 0.0f;
    for (int64_t r = L - 1; r >= 0; --r) {
        const int64_t n = base + r*S + i;
        run += live ? dA[n]*A[n] : 0.0f;
        if (G.kda) {
            if (live) {
                p.dg[(t0 + r)*gdn_dg_stride(G) + i] += run;
            }
        } else {
            // A scalar gate is one number per token, so the tail sums of every
            // channel collapse into it.
            const float total = gdn_block_reduce_sum(run, red);
            if (threadIdx.x == 0) {
                p.dg[(t0 + r)*gdn_dg_stride(G)] += total;
            }
        }
    }
}

// A chunk's dS_C picks up the gradient of the snapshot its last token wrote.
static __global__ void k_gdn_add_snapshot(const gdn_geom G, int64_t slot,
        const float * __restrict__ grad_state, float * __restrict__ dST) {
    const int64_t n  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t SS = G.S_v*G.S_v;
    if (n >= G.units*SS) {
        return;
    }
    dST[n] += grad_state[slot*SS*G.units + n];
}

// The chunk's slice of the upstream attention-score gradient, gathered
// contiguously so it can be a GEMM operand.
static __global__ void k_gdn_gather_dout(const gdn_geom G, int64_t t0, int64_t L,
        const float * __restrict__ grad_attn, float * __restrict__ dO) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t S    = G.S_v;
    if (n >= L*S) {
        return;
    }
    const int64_t r   = n / S;
    const int64_t j   = n - r*S;
    const int64_t iv1 = unit % G.H;
    const int64_t iv3 = unit / G.H;
    dO[unit*G.C*S + n] =
        grad_attn[(iv3*G.n_tokens*G.H + (t0 + r)*G.H + iv1)*S + j];
}

// The incoming recurrent state into the first chunk's entry slot: the source is
// laid out per sequence, the working buffers per unit.
static __global__ void k_gdn_state_in(const gdn_geom G,
        const float * __restrict__ state0, float * __restrict__ entry) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t SS   = G.S_v*G.S_v;
    if (n >= SS) {
        return;
    }
    const int64_t iv1 = unit % G.H;
    const int64_t iv3 = unit / G.H;
    entry[unit*SS + n] = state0[iv3*G.state_seq_stride + iv1*SS + n];
}

// dS_0 of the first chunk is the gradient of the incoming state.
static __global__ void k_gdn_state_grad_out(const gdn_geom G,
        const float * __restrict__ dST, float * __restrict__ g_state) {
    const int64_t unit = blockIdx.y;
    const int64_t n    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t SS   = G.S_v*G.S_v;
    if (n >= SS) {
        return;
    }
    g_state[gdn_state_off(G, unit) + n] += dST[unit*SS + n];
}

enum class gdn_gemm_precision {
    f32,
    bf16,
};

struct gdn_gemm_workspace {
    gdn_gemm_precision precision = gdn_gemm_precision::f32;
};

// Row-major front end for the chunk GEMMs: C[m,n] = alpha*op(A)*op(B) +
// beta*C, every matrix row-major with `ld` its stored row length, `s` its
// per-unit element stride. Column-major cuBLAS computes the transpose from the
// same buffers with the operands swapped. CUBLAS_COMPUTE_32F_FAST_16BF asks
// cuBLAS to round the F32 inputs to BF16 internally and use a F32 accumulator;
// the stored inputs and C therefore stay F32, with no conversion launches or
// extra scratch around these deliberately small GEMMs.
static void gdn_gemm(cublasHandle_t handle, const gdn_gemm_workspace & work,
                     bool transA, bool transB,
                     int m, int n, int k, float alpha,
                     const float * A, int lda, long long sA,
                     const float * B, int ldb, long long sB,
                     float beta, float * C, int ldc, long long sC, int batch) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11000
    if (work.precision == gdn_gemm_precision::bf16) {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                n, m, k, &alpha,
                B, CUDA_R_32F, ldb, sB,
                A, CUDA_R_32F, lda, sA,
                &beta, C, CUDA_R_32F, ldc, sC, batch,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT));
        return;
    }
#else
    GGML_ASSERT(work.precision == gdn_gemm_precision::f32);
#endif
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
            transB ? CUBLAS_OP_T : CUBLAS_OP_N,
            transA ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k, &alpha,
            B, ldb, sB,
            A, lda, sA,
            &beta, C, ldc, sC, batch));
}

static __global__ void k_gdn_fill_ptrs(const float ** a, float ** b, float ** c,
                                       const float * A, float * B, float * Cb,
                                       int64_t sA, int64_t sB, int64_t sC, int64_t batch) {
    const int64_t u = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (u >= batch) {
        return;
    }
    a[u] = A  + u*sA;
    b[u] = B  + u*sB;
    c[u] = Cb + u*sC;
}

// (ii)-(viii) for one chunk, issued for every unit at once. Called by both
// passes: forward to advance the state, backward to recompute what the adjoints
// read. `SnextT` null means "backward pass, keep the working tiles".
struct gdn_chunk_buffers {
    gdn_gemm_workspace gemm;
    float * A;
    float * Qt;
    float * Kt;
    float * Kh;
    float * VmZ;
    float * U;      // What on entry to the solve, U after it
    float * dO;
    float * dQt;
    float * dKt;
    float * dKh;
    float * dU;     // dU on entry to the solve, dWhat after it
    float * dZ;
    float * dA;
    float * bet;
    float * M;
    float * T;
    float * P;
    float * X;      // dP, then dT
    float * XT;
    float * GT;
    float * dST;
    float * entry;
    const float ** ptr_a;
    float ** ptr_b;
    float ** ptr_c;
};

static void gdn_chunk_forward(ggml_backend_cuda_context & ctx, const gdn_geom & G,
                              int64_t t0, int64_t L,
                              const float * q, const float * k, const float * v,
                              const float * g, const float * beta,
                              const float * S0T, const gdn_chunk_buffers & b,
                              float * SnextT, bool need_adjoint) {
    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    const int64_t  S      = G.S_v;
    const int      batch  = (int) G.units;
    const int      lds    = (int) S;
    const int      ldc    = (int) G.C;
    const long long scs   = (long long) G.C*S;
    const long long scc   = (long long) G.C*G.C;
    const long long sss   = (long long) S*S;

    k_gdn_gather<<<batch, GDN_CHUNK_BLOCK, 0, stream>>>(
        G, t0, L, q, k, g, beta, b.A, b.Qt, b.Kt, b.Kh, b.bet);
    CUDA_CHECK(cudaGetLastError());

    // (ii) Z = ktil S0. The state is stored transposed, so S0 on the right is a
    // B-transposed product against the raw buffer. Z lands in VmZ, which
    // k_gdn_make_w then turns into V - Z.
    gdn_gemm(handle, b.gemm, false, true, (int) L, lds, lds, 1.0f,
             b.Kt, lds, scs, S0T, lds, sss, 0.0f, b.VmZ, lds, scs, batch);

    {
        const dim3 grid((L*S + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, batch);
        k_gdn_make_w<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(G, t0, L, v, b.VmZ, b.bet, b.VmZ, b.U);
        CUDA_CHECK(cudaGetLastError());
    }

    // (iv) T, and (vi) P: two masked C x C products.
    gdn_gemm(handle, b.gemm, false, true, (int) L, (int) L, lds, 1.0f,
             b.Kt, lds, scs, b.Kh, lds, scs, 0.0f, b.M, ldc, scc, batch);
    {
        const dim3 grid((L*L + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, batch);
        k_gdn_mask_t<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(G, L, b.M, b.bet, b.T);
        CUDA_CHECK(cudaGetLastError());
    }

    // (v) U = (I+T)^-1 What, in place. In column-major terms the row-major
    // buffer of I+T is its transpose, so the strictly lower triangle becomes a
    // strictly upper one and the solve is a right-side unit-upper trsm; the unit
    // diagonal means the never-written diagonal is never read either.
    {
        const float one = 1.0f;
        CUBLAS_CHECK(cublasStrsmBatched(handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                lds, (int) L, &one,
                b.ptr_a, ldc, b.ptr_b, lds, batch));
    }

    // P is only consumed by the adjoint. The state-only first pass skips this
    // GEMM and mask; the backward recompute requests them below.
    if (need_adjoint) {
        gdn_gemm(handle, b.gemm, false, true, (int) L, (int) L, lds, 1.0f,
                 b.Qt, lds, scs, b.Kh, lds, scs, 0.0f, b.P, ldc, scc, batch);
        {
            const dim3 grid((L*L + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, batch);
            k_gdn_mask_lower_incl<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(G, L, b.P);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // (viii) khat^T U, then S0 added and the exit state scaled out.
    gdn_gemm(handle, b.gemm, true, false, lds, lds, (int) L, 1.0f,
             b.U, lds, scs, b.Kh, lds, scs, 0.0f, b.XT, lds, sss, batch);
    {
        const dim3 grid((S*S + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, batch);
        k_gdn_state_out<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(G, L, b.A, S0T, b.XT, SnextT);
        CUDA_CHECK(cudaGetLastError());
    }
}

// The adjoints of (ii)-(viii) for one chunk. dST holds dL/dS_C on entry and
// dL/dS_0 on exit, for every unit.
static void gdn_chunk_backward(ggml_backend_cuda_context & ctx, const gdn_geom & G,
                               int64_t t0, int64_t L,
                               const float * q, const float * k, const float * v,
                               const float * g, const float * beta,
                               const float * grad_attn, const float * S0T,
                               const gdn_chunk_buffers & b, float * dst) {
    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    const int64_t  S      = G.S_v;
    const int      batch  = (int) G.units;
    const int      lds    = (int) S;
    const int      ldc    = (int) G.C;
    const long long scs   = (long long) G.C*S;
    const long long scc   = (long long) G.C*G.C;
    const long long sss   = (long long) S*S;
    const float    sc     = G.scale;

    gdn_chunk_forward(ctx, G, t0, L, q, k, v, g, beta, S0T, b, nullptr, true);

    // dO for this chunk, gathered contiguously.
    {
        const dim3 grid((L*S + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, batch);
        k_gdn_gather_dout<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(G, t0, L, grad_attn, b.dO);
        CUDA_CHECK(cudaGetLastError());
    }

    // (viii)^T
    k_gdn_grad_state_in<<<batch, GDN_CHUNK_BLOCK, 0, stream>>>(G, L, b.A, b.dST, b.XT, b.GT, b.dA);
    CUDA_CHECK(cudaGetLastError());
    gdn_gemm(handle, b.gemm, false, false, (int) L, lds, lds, 1.0f,
             b.U, lds, scs, b.GT, lds, sss, 0.0f, b.dKh, lds, scs, batch);
    gdn_gemm(handle, b.gemm, false, true, (int) L, lds, lds, 1.0f,
             b.Kh, lds, scs, b.GT, lds, sss, 0.0f, b.dU, lds, scs, batch);
    CUDA_CHECK(cudaMemcpyAsync(b.dST, b.GT, (size_t) G.units*S*S*sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    // (vii)^T
    gdn_gemm(handle, b.gemm, false, false, (int) L, lds, lds, sc,
             b.dO, lds, scs, S0T, lds, sss, 0.0f, b.dQt, lds, scs, batch);
    gdn_gemm(handle, b.gemm, true, false, lds, lds, (int) L, sc,
             b.dO, lds, scs, b.Qt, lds, scs, 1.0f, b.dST, lds, sss, batch);
    gdn_gemm(handle, b.gemm, false, true, (int) L, (int) L, lds, sc,
             b.dO, lds, scs, b.U, lds, scs, 0.0f, b.X, ldc, scc, batch);
    {
        const dim3 grid((L*L + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, batch);
        k_gdn_mask_lower_incl<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(G, L, b.X);
        CUDA_CHECK(cudaGetLastError());
    }
    gdn_gemm(handle, b.gemm, true, false, (int) L, lds, (int) L, sc,
             b.P, ldc, scc, b.dO, lds, scs, 1.0f, b.dU, lds, scs, batch);

    // (vi)^T
    gdn_gemm(handle, b.gemm, false, false, (int) L, lds, (int) L, 1.0f,
             b.X, ldc, scc, b.Kh, lds, scs, 1.0f, b.dQt, lds, scs, batch);
    gdn_gemm(handle, b.gemm, true, false, (int) L, lds, (int) L, 1.0f,
             b.X, ldc, scc, b.Qt, lds, scs, 1.0f, b.dKh, lds, scs, batch);

    // (v)^T dWhat = L^T dU: the same triangular factor, transposed, in place.
    {
        const float one = 1.0f;
        CUBLAS_CHECK(cublasStrsmBatched(handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                lds, (int) L, &one,
                b.ptr_a, ldc, b.ptr_c, lds, batch));
    }

    // (iv)^T
    gdn_gemm(handle, b.gemm, false, true, (int) L, (int) L, lds, -1.0f,
             b.dU, lds, scs, b.U, lds, scs, 0.0f, b.X, ldc, scc, batch);
    k_gdn_dt<<<dim3((unsigned) L, (unsigned) batch), GDN_CHUNK_BLOCK, 0, stream>>>(
        G, t0, L, b.M, b.bet, b.X, dst);
    CUDA_CHECK(cudaGetLastError());
    gdn_gemm(handle, b.gemm, false, false, (int) L, lds, (int) L, 1.0f,
             b.X, ldc, scc, b.Kh, lds, scs, 0.0f, b.dKt, lds, scs, batch);
    gdn_gemm(handle, b.gemm, true, false, (int) L, lds, (int) L, 1.0f,
             b.X, ldc, scc, b.Kt, lds, scs, 1.0f, b.dKh, lds, scs, batch);

    // (iii)^T
    k_gdn_dvz<<<dim3((unsigned) L, (unsigned) batch), GDN_CHUNK_BLOCK, 0, stream>>>(
        G, t0, L, b.dU, b.VmZ, b.bet, b.dZ, dst);
    CUDA_CHECK(cudaGetLastError());

    // (ii)^T
    gdn_gemm(handle, b.gemm, false, false, (int) L, lds, lds, 1.0f,
             b.dZ, lds, scs, S0T, lds, sss, 1.0f, b.dKt, lds, scs, batch);
    gdn_gemm(handle, b.gemm, true, false, lds, lds, (int) L, 1.0f,
             b.dZ, lds, scs, b.Kt, lds, scs, 1.0f, b.dST, lds, sss, batch);

    // (i)^T
    k_gdn_finalize<<<batch, GDN_CHUNK_BLOCK, 0, stream>>>(
        G, t0, L, q, k, b.A, b.dQt, b.dKt, b.dKh, b.dA, dst);
    CUDA_CHECK(cudaGetLastError());
}

// Returns false when no chunk length is numerically safe, i.e. the caller must
// run the sequential kernel.
static bool ggml_cuda_gated_delta_net_back_chunked(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const gdn_geom & geom, int64_t chunk) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];
    const ggml_tensor * src_grad  = dst->src[6];

    cudaStream_t stream = ctx.stream();
    gdn_geom G = geom;

    const int64_t S        = G.S_v;
    const int64_t SS       = S*S;
    const int64_t units    = G.units;
    const int64_t n_tokens = G.n_tokens;

    const int64_t C = MIN(MIN(chunk, (int64_t) GDN_CHUNK_MAX), n_tokens);
    G.C = C;

    // G3 is deliberately opt-in: rounding every GEMM input changes the
    // numerical regime of a backward pass. Ampere is the first NVIDIA
    // generation with native BF16 tensor cores; every other backend keeps the
    // exact F32 oracle path even when the variable is set.
    bool use_bf16_gemm = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11000
    const char * mma_env = getenv("GGML_CUDA_GDN_BACK_MMA");
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    use_bf16_gemm = mma_env != nullptr && std::atoi(mma_env) != 0 &&
        GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE;
#endif

    // The gates decide the layout, so they have to be read before the first
    // GEMM shape is known: one reduction to n_tokens floats, one copy back.
    std::vector<int64_t> starts;
    {
        const int64_t nch = G.kda ? S : 1;
        std::vector<float> worst((size_t) n_tokens);
        ggml_cuda_pool_alloc<float> worst_d(ctx.pool(), (size_t) n_tokens);
        CUDA_CHECK(cudaMemsetAsync(worst_d.get(), 0, n_tokens*sizeof(float), stream));
        const dim3 grid((n_tokens*nch + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, (unsigned) units);
        k_gdn_gate_worst<<<grid, GDN_CHUNK_BLOCK, 0, stream>>>(
            G, (const float *) src_g->data, worst_d.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(worst.data(), worst_d.get(), n_tokens*sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        starts = gdn_chunk_layout(n_tokens, C, G.K, worst.data());
        if (starts.empty()) {
            return false;
        }
    }
    const int64_t n_chunks = (int64_t) starts.size() - 1;
    if (getenv("GGML_CUDA_GDN_BACK_DEBUG")) {
        fprintf(stderr, "gdn_back: n_tokens=%lld units=%lld C=%lld chunks=%lld gemm=%s\n",
                (long long) n_tokens, (long long) units, (long long) C, (long long) n_chunks,
                use_bf16_gemm ? "bf16-f32" : "f32");
    }

    // Working tiles, all per-unit strided so every GEMM is one strided-batched
    // call. The chunk tiles are sized for C, not for the current chunk length,
    // so the strides never move; a shorter tail chunk just uses fewer rows.
    // Total is O(units * C * S_v) plus the entry states, O(n_chunks * units *
    // S_v^2) -- the sequential kernel's trajectory divided by C.
    ggml_cuda_pool_alloc<float> tiles(ctx.pool(), (size_t) (13*units*C*S + units*C));
    ggml_cuda_pool_alloc<float> mats (ctx.pool(), (size_t) (4*units*C*C));
    ggml_cuda_pool_alloc<float> sq   (ctx.pool(), (size_t) (3*units*SS));
    ggml_cuda_pool_alloc<float> entry(ctx.pool(), (size_t) (n_chunks*units*SS));
    ggml_cuda_pool_alloc<const float *> ptr_a(ctx.pool(), (size_t) units);
    ggml_cuda_pool_alloc<float *>       ptr_b(ctx.pool(), (size_t) units);
    ggml_cuda_pool_alloc<float *>       ptr_c(ctx.pool(), (size_t) units);

    gdn_chunk_buffers b = {};
    {
        b.gemm.precision = use_bf16_gemm ? gdn_gemm_precision::bf16 : gdn_gemm_precision::f32;
        float * t = tiles.get();
        const int64_t n = units*C*S;
        b.A = t; t += n; b.Qt  = t; t += n; b.Kt  = t; t += n; b.Kh = t; t += n;
        b.VmZ = t; t += n; b.U = t; t += n; b.dO  = t; t += n; b.dQt = t; t += n;
        b.dKt = t; t += n; b.dKh = t; t += n; b.dU = t; t += n; b.dZ  = t; t += n;
        b.dA = t; t += n; b.bet = t;
        float * m = mats.get();
        b.M = m; m += units*C*C; b.T = m; m += units*C*C; b.P = m; m += units*C*C; b.X = m;
        float * s = sq.get();
        b.XT = s; s += units*SS; b.GT = s; s += units*SS; b.dST = s;
        b.entry = entry.get();
        b.ptr_a = ptr_a.get();
        b.ptr_b = ptr_b.get();
        b.ptr_c = ptr_c.get();
    }
    k_gdn_fill_ptrs<<<(units + 255)/256, 256, 0, stream>>>(
        b.ptr_a, b.ptr_b, b.ptr_c, b.T, b.U, b.dU, C*C, C*S, C*S, units);
    CUDA_CHECK(cudaGetLastError());

    const float * q    = (const float *) src_q->data;
    const float * k    = (const float *) src_k->data;
    const float * v    = (const float *) src_v->data;
    const float * g    = (const float *) src_g->data;
    const float * beta = (const float *) src_beta->data;
    const float * grad_attn  = (const float *) src_grad->data;
    const float * grad_state = grad_attn + S*G.H*n_tokens*G.n_seqs;
    float * dst_d = (float *) dst->data;

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    // These are gradients, so true F32 rather than the TF32 mode common.cuh puts
    // on every handle for inference matmuls -- same reasoning, and the same
    // set/restore idiom, as out-prod.cu:64.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const dim3 grid_unit((SS + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK, (unsigned) units);

    // Pass 1: forward over chunks, keeping each chunk's entry state. The
    // per-unit state is copied in from the recurrent-state tensor, which is laid
    // out per sequence rather than per unit.
    k_gdn_state_in<<<grid_unit, GDN_CHUNK_BLOCK, 0, stream>>>(
        G, (const float *) src_state->data, b.entry);
    CUDA_CHECK(cudaGetLastError());
    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t t0 = starts[c];
        const int64_t L  = starts[c + 1] - t0;
        gdn_chunk_forward(ctx, G, t0, L, q, k, v, g, beta,
                          b.entry + c*units*SS, b,
                          c + 1 < n_chunks ? b.entry + (c + 1)*units*SS : nullptr,
                          false);
    }

    // Pass 2: chunks in reverse, dS_0 of one being dS_C of the previous.
    CUDA_CHECK(cudaMemsetAsync(b.dST, 0, (size_t) units*SS*sizeof(float), stream));
    for (int64_t c = n_chunks - 1; c >= 0; --c) {
        const int64_t t0   = starts[c];
        const int64_t L    = starts[c + 1] - t0;
        const int64_t slot = n_tokens - 1 - (t0 + L - 1);
        if (slot < G.K) {
            k_gdn_add_snapshot<<<(int) ((units*SS + GDN_CHUNK_BLOCK - 1)/GDN_CHUNK_BLOCK),
                                 GDN_CHUNK_BLOCK, 0, stream>>>(G, slot, grad_state, b.dST);
            CUDA_CHECK(cudaGetLastError());
        }
        gdn_chunk_backward(ctx, G, t0, L, q, k, v, g, beta, grad_attn,
                           b.entry + c*units*SS, b, dst_d);
    }

    {
        float * g_state = dst_d + G.n_q + G.n_k + G.n_v + G.n_g + G.n_beta;
        k_gdn_state_grad_out<<<grid_unit, GDN_CHUNK_BLOCK, 0, stream>>>(G, b.dST, g_state);
        CUDA_CHECK(cudaGetLastError());
    }

    // revert to the standard mode from common.cuh
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    return true;
}

// retro delta: which formulation runs. The op param wins (tests pin it through
// ggml_gated_delta_net_back_chunked), then GGML_CUDA_GDN_BACK_CHUNK, then the
// chunked default -- the sequential kernel above stays reachable as the
// reference and as the numerical fallback.
static int64_t ggml_cuda_gdn_back_chunk_len(const ggml_tensor * dst) {
    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint != 0) {
        return hint;
    }
    const char * env = getenv("GGML_CUDA_GDN_BACK_CHUNK");
    if (env) {
        const int64_t v = (int64_t) atoll(env);
        return v > 0 ? v : -1;
    }
    return GDN_CHUNK_DEFAULT;
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
    GGML_TENSOR_LOCALS(size_t,  nbg, src_g, nb);
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

    // Geometry, shared by both formulations.
    gdn_geom G = {};
    G.S_v      = S_v;
    G.H        = H;
    G.n_tokens = n_tokens;
    G.n_seqs   = n_seqs;
    G.units    = H*n_seqs;
    G.C        = 0;                     // set by the chunkwise path
    G.neq1     = neq1;
    G.nek1     = nek1;
    G.rq3      = rq3;
    G.rk3      = rk3;
    G.sq1      = nbq1/sizeof(float); G.sq2 = nbq2/sizeof(float); G.sq3 = nbq3/sizeof(float);
    G.sk1      = nbk1/sizeof(float); G.sk2 = nbk2/sizeof(float); G.sk3 = nbk3/sizeof(float);
    G.sv1      = nbv1/sizeof(float); G.sv2 = nbv2/sizeof(float); G.sv3 = nbv3/sizeof(float);
    G.sg1      = nbg1/sizeof(float); G.sg2 = nbg2/sizeof(float); G.sg3 = nbg3/sizeof(float);
    G.sb1      = nbb1/sizeof(float); G.sb2 = nbb2/sizeof(float); G.sb3 = nbb3/sizeof(float);
    G.n_q      = n_q;
    G.n_k      = n_k;
    G.n_v      = n_v;
    G.n_g      = n_g;
    G.n_beta   = n_beta;
    G.state_seq_stride = state_seq_stride;
    G.scale    = scale;
    G.K        = K;
    G.kda      = kda;

    // The chunkwise path (see the block comment above it) if it is selected and
    // applicable, otherwise the sequential kernel below. `S_v > GDN_CHUNK_BLOCK`
    // is a hard limit rather than a heuristic: k_gdn_finalize's reverse
    // cumulative sum gives one gate channel to one thread for a whole chunk, and
    // so does the fallback's per-token step.
    const int64_t chunk = ggml_cuda_gdn_back_chunk_len(dst);
    if (chunk > 0 && S_v <= GDN_CHUNK_BLOCK &&
        ggml_cuda_gated_delta_net_back_chunked(ctx, dst, G, chunk)) {
        return;
    }

    // The trajectory dominates; S1/Snew/dS/dS1 are the four working matrices.
    // pre/delta/gexp/dpre/ddelta live in shared memory.
    const int64_t scratch_stride = n_tokens*S_v*S_v + 4*S_v*S_v;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) (G.units * scratch_stride));

    const size_t shmem = gdn_seq_smem_bytes(S_v, GDN_BACK_BLOCK);
    gated_delta_net_back_kernel<<<(unsigned) G.units, GDN_BACK_BLOCK, shmem, stream>>>(
        G,
        (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
        (const float *) src_g->data, (const float *) src_beta->data,
        (const float *) src_state->data, (const float *) src_grad->data, dst_d,
        scratch.get(), scratch_stride);
    CUDA_CHECK(cudaGetLastError());
}
