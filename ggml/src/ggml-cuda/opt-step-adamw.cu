#include "ggml-impl.h"
#include "opt-step-adamw.cuh"

#include <cstdint>

static __global__ void opt_step_adamw_f32(
    float * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m, float * __restrict__ g_v,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[i]*pars[8];
    const float gmi = g_m[i]*beta1 +    gi*(1.0f - beta1);
    const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    const float mh =       gmi*beta1h;
    const float vh = sqrtf(gvi*beta2h) + eps;

    x[i] = x[i]*(1.0f - alpha*wd) - alpha*mh/vh;
}

static void opt_step_adamw_f32_cuda(
    float * x, const float * g, float * g_m, float * g_v, const float * pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    opt_step_adamw_f32<<<block_nums, block_dims, 0, stream>>>(x, g, g_m, g_v, pars, k);
}

// retro delta: F16-parameter AdamW. Moments/gradient stay F32; the parameter is
// read/stored as F16 with the fork's stochastic rounding. These helpers mirror
// ggml_sr_uniform / ggml_f16_neighbour / ggml_stochastic_round_f16 (ggml-impl.h)
// bit-for-bit so the F16 store matches the CPU oracle exactly.
static __device__ __forceinline__ float adamw_sr_uniform(uint32_t seed, uint32_t index) {
    uint32_t h = seed ^ (index * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (float) (h >> 8) * (1.0f / 16777216.0f);
}

static __device__ __forceinline__ uint16_t adamw_f16_neighbour(uint16_t bits, int up) {
    const uint16_t sign = bits & 0x8000u;
    const uint16_t mag  = bits & 0x7FFFu;
    if (mag == 0) {
        return up ? (uint16_t) 0x0001u : (uint16_t) 0x8001u;
    }
    const int grow = up ? (sign == 0) : (sign != 0);
    return (uint16_t) (sign | (uint16_t) (grow ? mag + 1 : mag - 1));
}

// Returns the F16 bits of `x` after stochastic rounding with uniform `u`.
static __device__ __forceinline__ uint16_t adamw_stochastic_round_f16(float x, float u) {
    const uint16_t nearest   = __half_as_ushort(__float2half_rn(x));
    const float    nearest_f = __half2float(__ushort_as_half(nearest));
    const float    residual  = x - nearest_f;
    if (residual == 0.0f) {
        return nearest;
    }
    const uint16_t other   = adamw_f16_neighbour(nearest, residual > 0.0f);
    const float    other_f = __half2float(__ushort_as_half(other));
    const float    span    = other_f - nearest_f;
    const float    p = (span != 0.0f && isfinite(span)) ? residual / span : 0.0f;
    return u < p ? other : nearest;
}

static __global__ void opt_step_adamw_f16(
    half * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m,
    float * __restrict__ g_v, const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float keep   = 1.0f - alpha*pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const uint32_t seed = (uint32_t) pars[7];
    const float gscale = pars[8];

    const float gi = g[i]*gscale;
    const float gmi = g_m[i]*beta1 +    gi*(1.0f - beta1);
    const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);
    g_m[i] = gmi;
    g_v[i] = gvi;

    const float mh = gmi*beta1h;
    const float vh = sqrtf(gvi*beta2h) + eps;
    const float updated = __half2float(x[i])*keep - alpha*mh/vh;

    const uint16_t bits = adamw_stochastic_round_f16(updated, adamw_sr_uniform(seed, (uint32_t) i));
    x[i] = __ushort_as_half(bits);
}

static void opt_step_adamw_f16_cuda(
    half * x, const float * g, float * g_m, float * g_v, const float * pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    opt_step_adamw_f16<<<block_nums, block_dims, 0, stream>>>(x, g, g_m, g_v, pars, k);
}

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0->type         == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0_grad->type    == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_m->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_v->type  == GGML_TYPE_F32);
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
    GGML_ASSERT(ggml_is_contiguous(adamw_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 9);

    const float * src0_grad_d    = (const float *) src0_grad->data;
    float       * src0_grad_m_d  = (float       *) src0_grad_m->data;
    float       * src0_grad_v_d  = (float       *) src0_grad_v->data;
    const float * adamw_params_d = (const float *) adamw_params->data;

    cudaStream_t stream = ctx.stream();

    const int64_t ne = ggml_nelements(src0);

    if (src0->type == GGML_TYPE_F16) {
        opt_step_adamw_f16_cuda((half *) src0->data, src0_grad_d, src0_grad_m_d,
                                src0_grad_v_d, adamw_params_d, ne, stream);
    } else {
        opt_step_adamw_f32_cuda((float *) src0->data, src0_grad_d, src0_grad_m_d,
                                src0_grad_v_d, adamw_params_d, ne, stream);
    }
}
