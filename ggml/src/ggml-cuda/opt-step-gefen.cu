#include "ggml-impl.h"
#include "opt-step-gefen.cuh"

#include <cstdint>

// retro delta: Gefen, both phases. One CUDA block owns one quantization block,
// so the whole block is read against the old scale before any element is rewritten.

// Nearest codebook entry for a value in [-1, 1], ties to the lower index.
// Must match ggml_gefen_nearest_code (ggml-cpu/ops.cpp),
// retro_gefen_nearest_code (ggml-metal/kernels/retro.metal) and gefen_nearest_code
// (ggml-vulkan/vulkan-shaders/gefen_head.glsl), otherwise a checkpoint written
// on one backend decodes differently on another.
static __device__ __forceinline__ uint8_t gefen_nearest_code(float value, int levels) {
    const float top = (float) (levels - 1);
    const float position = (value + 1.0f)*0.5f*top;
    if (!(position > 0.0f)) {
        // Also catches NaN.
        return 0;
    }
    if (position >= top) {
        return (uint8_t) (levels - 1);
    }
    // ceil(x - 1/2) rounds exact midpoints down to the lower index.
    return (uint8_t) ceilf(position - 0.5f);
}

static __device__ __forceinline__ float gefen_decode(
        const uint8_t * __restrict__ indices, const float * __restrict__ codebook,
        int64_t i, float scale, int levels) {
    const int code = (int) indices[i];
    return scale*codebook[code < levels ? code : levels - 1];
}

// Phase A. Reads the clipped gradient and the old state, writes [scale, second
// moment] per block. Under shared_v there is no scale; the first row stays zero.
template<bool quantized>
static __global__ void opt_step_gefen_stats_f32(
        const float   * __restrict__ g,
        const uint8_t * __restrict__ indices,
        const float   * __restrict__ scales,
        const float   * __restrict__ v,
        const float   * __restrict__ codebook,
        const float   * __restrict__ pars,
        float         * __restrict__ dst,
        const int64_t np, const int64_t bs, const int levels) {
    __shared__ float shared_vals[WARP_SIZE];

    const int64_t b = blockIdx.x;

    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float gscale = pars[7];

    const int64_t i0 = b*bs;
    const int64_t i1 = min(i0 + bs, np);

    const float scale_old = quantized ? scales[b] : 0.0f;

    float sum_sq  = 0.0f;
    float max_abs = 0.0f;
    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const float gi = g[i]*gscale;
        sum_sq += gi*gi;
        if (quantized) {
            // Recomputes the first moment the update will use, so the scale
            // matches the values that are actually quantized.
            const float mi = beta1*gefen_decode(indices, codebook, i, scale_old, levels)
                    + (1.0f - beta1)*gi;
            max_abs = fmaxf(max_abs, fabsf(mi));
        }
    }

    const float sum_total = block_reduce<block_reduce_method::SUM,
            CUDA_OPT_STEP_GEFEN_BLOCK_SIZE>(sum_sq, shared_vals);
    float max_total = 0.0f;
    if (quantized) {
        // Readers of the sum are done only at the next barrier; the max
        // reduction reuses the same scratch.
        __syncthreads();
        max_total = block_reduce<block_reduce_method::MAX,
                CUDA_OPT_STEP_GEFEN_BLOCK_SIZE>(max_abs, shared_vals);
    }

    if (threadIdx.x == 0) {
        // Mean over the block's actual elements, so a partial trailing block
        // is not diluted by padding.
        const float mean_sq = sum_total/(float) (i1 - i0);
        dst[2*b + 0] = max_total;
        dst[2*b + 1] = beta2*v[b] + (1.0f - beta2)*mean_sq;
    }
}

// Phase B. Mutates weights, first moment, scales and second moments; `stats`
// is phase A's answer.
template<bool quantized>
static __global__ void opt_step_gefen_f32(
        float   * __restrict__ w,
        const float * __restrict__ g,
        void    * __restrict__ moment,
        float   * __restrict__ scales,
        float   * __restrict__ v,
        const float * __restrict__ stats,
        const float * __restrict__ codebook,
        const float * __restrict__ pars,
        const int64_t np, const int64_t bs, const int levels, const int zero_code) {
    const int64_t b = blockIdx.x;

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const float gscale = pars[7];
    const float keep   = 1.0f - alpha*wd;

    const int64_t i0 = b*bs;
    const int64_t i1 = min(i0 + bs, np);

    const float scale_new = stats[2*b + 0];
    const float v_new     = stats[2*b + 1];
    const float denom     = sqrtf(v_new*beta2h) + eps;
    // Read before any element of this block overwrites it.
    const float scale_old = quantized ? scales[b] : 0.0f;

    uint8_t * i_state = (uint8_t *) moment;
    float   * m_state = (float   *) moment;

    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const float gi = g[i]*gscale;
        float mi;
        if (quantized) {
            mi = beta1*gefen_decode(i_state, codebook, i, scale_old, levels) + (1.0f - beta1)*gi;
        } else {
            mi = beta1*m_state[i] + (1.0f - beta1)*gi;
        }
        // Decoupled, against the old weight: decaying the updated weight adds
        // a cross term AdamW does not have.
        w[i] = w[i]*keep - alpha*(mi*beta1h)/denom;
        if (quantized) {
            // Zero block: the scale carries the zero, the index is canonical.
            i_state[i] = scale_new > 0.0f
                    ? gefen_nearest_code(mi/scale_new, levels)
                    : (uint8_t) zero_code;
        } else {
            m_state[i] = mi;
        }
    }

    // All threads have read the old scale, so the new one can be published.
    __syncthreads();

    if (threadIdx.x == 0) {
        if (quantized) {
            scales[b] = scale_new;
        }
        v[b] = v_new;
    }
}

// The variant selects the kernel instead of travelling in the arguments.
static bool ggml_cuda_gefen_is_quantized(const ggml_tensor * op) {
    return ggml_get_op_params_i32(op, 0) == GGML_OPT_GEFEN_VARIANT_QUANTIZED_M;
}

void ggml_cuda_opt_step_gefen_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad     = dst->src[0];
    const ggml_tensor * moment   = dst->src[1];
    const ggml_tensor * scales   = dst->src[2];
    const ggml_tensor * v        = dst->src[3];
    const ggml_tensor * codebook = dst->src[4];
    const ggml_tensor * pars     = dst->src[5];

    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type    == GGML_TYPE_F32);
    GGML_ASSERT(pars->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(pars) == 8);

    const int64_t bs = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(bs > 0);

    const int64_t np       = ggml_nelements(grad);
    const int64_t n_blocks = ggml_nelements(v);
    const int     levels   = codebook ? (int) ggml_nelements(codebook) : 0;

    cudaStream_t stream = ctx.stream();

    const dim3 block_dims(CUDA_OPT_STEP_GEFEN_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(n_blocks, 1, 1);

    if (ggml_cuda_gefen_is_quantized(dst)) {
        opt_step_gefen_stats_f32<true><<<block_nums, block_dims, 0, stream>>>(
            (const float   *) grad->data, (const uint8_t *) moment->data,
            (const float   *) scales->data, (const float *) v->data,
            (const float   *) codebook->data, (const float *) pars->data,
            (float *) dst->data, np, bs, levels);
    } else {
        opt_step_gefen_stats_f32<false><<<block_nums, block_dims, 0, stream>>>(
            (const float *) grad->data, nullptr, nullptr, (const float *) v->data,
            nullptr, (const float *) pars->data,
            (float *) dst->data, np, bs, levels);
    }
}

void ggml_cuda_opt_step_gefen(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0     = dst->src[0];
    const ggml_tensor * grad     = dst->src[1];
    const ggml_tensor * moment   = dst->src[2];
    const ggml_tensor * scales   = dst->src[3];
    const ggml_tensor * v        = dst->src[4];
    const ggml_tensor * stats    = dst->src[5];
    const ggml_tensor * codebook = dst->src[6];
    const ggml_tensor * pars     = dst->src[7];

    GGML_ASSERT(src0->type  == GGML_TYPE_F32);
    GGML_ASSERT(grad->type  == GGML_TYPE_F32);
    GGML_ASSERT(v->type     == GGML_TYPE_F32);
    GGML_ASSERT(stats->type == GGML_TYPE_F32);
    GGML_ASSERT(pars->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_nelements(pars) == 8);

    const int64_t bs = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(bs > 0);

    const int64_t np       = ggml_nelements(src0);
    const int64_t n_blocks = ggml_nelements(v);
    const int     levels   = codebook ? (int) ggml_nelements(codebook) : 0;

    cudaStream_t stream = ctx.stream();

    const dim3 block_dims(CUDA_OPT_STEP_GEFEN_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(n_blocks, 1, 1);

    if (ggml_cuda_gefen_is_quantized(dst)) {
        opt_step_gefen_f32<true><<<block_nums, block_dims, 0, stream>>>(
            (float *) src0->data, (const float *) grad->data, moment->data,
            (float *) scales->data, (float *) v->data, (const float *) stats->data,
            (const float *) codebook->data, (const float *) pars->data,
            np, bs, levels, GGML_GEFEN_ZERO_BLOCK_INDEX);
    } else {
        opt_step_gefen_f32<false><<<block_nums, block_dims, 0, stream>>>(
            (float *) src0->data, (const float *) grad->data, moment->data,
            nullptr, (float *) v->data, (const float *) stats->data,
            nullptr, (const float *) pars->data,
            np, bs, levels, GGML_GEFEN_ZERO_BLOCK_INDEX);
    }
}
