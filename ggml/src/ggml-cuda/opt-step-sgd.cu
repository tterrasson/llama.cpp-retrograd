#include "ggml-impl.h"
#include "opt-step-sgd.cuh"
#include "retro-stochastic-round.cuh" // retro delta

#include <cstdint>

static __global__ void opt_step_sgd_f32(
    float * __restrict__ x, const float * __restrict__ g,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

static void opt_step_sgd_f32_cuda(
    float * x, const float * g, const float * __restrict__ pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_SGD_BLOCK_SIZE - 1) / CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    opt_step_sgd_f32<<<block_nums, block_dims, 0, stream>>>(x, g, pars, k);
}

// retro delta: SGD with F16 or BF16 parameters; gradient stays F32, the store
// is the shared stochastic rounding (retro-stochastic-round.cuh), seeded by pars[2].
static __global__ void opt_step_sgd_f16(
    half * __restrict__ x, const float * __restrict__ g,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float alpha = pars[0];
    const float keep  = 1.0f - alpha*pars[1];
    const uint32_t seed = (uint32_t) pars[2];

    const float updated = __half2float(x[i])*keep - alpha*g[i];
    x[i] = __ushort_as_half(retro_stochastic_round_f16(updated, retro_sr_uniform(seed, (uint32_t) i)));
}

static void opt_step_sgd_f16_cuda(
    half * x, const float * g, const float * __restrict__ pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_SGD_BLOCK_SIZE - 1) / CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    opt_step_sgd_f16<<<block_nums, block_dims, 0, stream>>>(x, g, pars, k);
}

static __global__ void opt_step_sgd_bf16(
    uint16_t * __restrict__ x, const float * __restrict__ g,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float alpha = pars[0];
    const float keep  = 1.0f - alpha*pars[1];
    const uint32_t seed = (uint32_t) pars[2];

    const float updated = retro_bf16_to_f32(x[i])*keep - alpha*g[i];
    x[i] = retro_stochastic_round_bf16(updated, retro_sr_uniform(seed, (uint32_t) i));
}

static void opt_step_sgd_bf16_cuda(
    uint16_t * x, const float * g, const float * __restrict__ pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_SGD_BLOCK_SIZE - 1) / CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    opt_step_sgd_bf16<<<block_nums, block_dims, 0, stream>>>(x, g, pars, k);
}

void ggml_cuda_opt_step_sgd(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0      = dst->src[0];
    const ggml_tensor * src0_grad = dst->src[1];
    const ggml_tensor * params    = dst->src[2];

    // retro delta: the parameter may be half precision.
    GGML_ASSERT(src0->type      == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                src0->type      == GGML_TYPE_BF16);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(params->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    // retro delta: alpha, wd and the rounding seed.
    GGML_ASSERT(ggml_nelements(params) == 3);

    const float * src0_grad_d = (const float *) src0_grad->data;
    const float * params_d    = (const float *) params->data;

    cudaStream_t stream = ctx.stream();

    const int64_t ne = ggml_nelements(src0);

    // retro delta: dispatch on the parameter's precision.
    if (src0->type == GGML_TYPE_F16) {
        opt_step_sgd_f16_cuda((half *) src0->data, src0_grad_d, params_d, ne, stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        opt_step_sgd_bf16_cuda((uint16_t *) src0->data, src0_grad_d, params_d, ne, stream);
    } else {
        opt_step_sgd_f32_cuda((float *) src0->data, src0_grad_d, params_d, ne, stream);
    }
}
