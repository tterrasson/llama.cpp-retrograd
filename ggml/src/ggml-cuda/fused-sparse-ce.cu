#include "fused-sparse-ce.cuh"

#include "convert.cuh"
#include "sum.cuh"

#include <algorithm>
#include <cstdint>

// retro delta: tiled CUDA FUSED_SPARSE_CE[_BACK].  cuBLAS materializes at most
// [min(tile, 1024), n_tokens] logits, never [n_vocab, n_tokens].  Each tile is
// folded into an online F32 log-sum-exp; backward recomputes it, converts the
// tile to scaled probabilities, and accumulates grad_h with another GEMM.

static __global__ void fused_sparse_ce_count_active(
        const int32_t * targets, const float * weights, int32_t * n_active,
        int64_t n_tokens, int64_t n_vocab) {
    const int64_t t = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (t < n_tokens && targets[t] >= 0 && targets[t] < n_vocab && weights[t] != 0.0f) {
        atomicAdd(n_active, 1);
    }
}

static __global__ void fused_sparse_ce_init(float * maxima, float * sums, float * target_logits, int64_t n_tokens) {
    const int64_t t = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (t < n_tokens) {
        maxima[t] = -INFINITY;
        sums[t] = 0.0f;
        target_logits[t] = 0.0f;
    }
}

static __global__ void fused_sparse_ce_update_lse(
        const float * logits, const int32_t * targets, float * maxima,
        float * sums, float * target_logits, int64_t tile_start,
        int64_t tile_size, int64_t tile_stride, int64_t n_tokens) {
    const int64_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) {
        return;
    }
    float running_max = maxima[t];
    float running_sum = sums[t];
    const float * column = logits + t*tile_stride;
    for (int64_t i = 0; i < tile_size; ++i) {
        const float z = column[i];
        if (z > running_max) {
            running_sum = running_sum*expf(running_max - z) + 1.0f;
            running_max = z;
        } else {
            running_sum += expf(z - running_max);
        }
    }
    const int64_t target = targets[t];
    if (target >= tile_start && target < tile_start + tile_size) {
        target_logits[t] = column[target - tile_start];
    }
    maxima[t] = running_max;
    sums[t] = running_sum;
}

static __global__ void fused_sparse_ce_finish_loss(
        const int32_t * targets, const float * weights, const int32_t * n_active,
        const float * maxima, const float * sums, const float * target_logits,
        float * losses, int64_t n_tokens, int64_t n_vocab) {
    const int64_t t = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (t >= n_tokens) {
        return;
    }
    const bool active = targets[t] >= 0 && targets[t] < n_vocab && weights[t] != 0.0f && *n_active > 0;
    losses[t] = active
        ? weights[t]*(maxima[t] + logf(sums[t]) - target_logits[t])/(float) *n_active
        : 0.0f;
}

static __global__ void fused_sparse_ce_make_probs(
        float * logits, const float * grad, const int32_t * targets,
        const float * weights, const int32_t * n_active, const float * maxima,
        const float * sums, int64_t tile_size, int64_t tile_stride,
        int64_t n_tokens, int64_t n_vocab) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = tile_size*n_tokens;
    if (i >= n) {
        return;
    }
    const int64_t t = i/tile_size;
    const int64_t v = i - t*tile_size;
    const bool active = targets[t] >= 0 && targets[t] < n_vocab && weights[t] != 0.0f && *n_active > 0;
    const float coef = active ? *grad*weights[t]/(float) *n_active : 0.0f;
    const int64_t offset = t*tile_stride + v;
    logits[offset] = active ? coef*expf(logits[offset] - maxima[t])/sums[t] : 0.0f;
}

static __global__ void fused_sparse_ce_subtract_target(
        const float * grad, const float * w, const int32_t * targets,
        const float * weights, const int32_t * n_active, float * dst,
        int64_t n_embd, int64_t n_tokens, int64_t n_vocab) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n_embd*n_tokens) {
        return;
    }
    const int64_t t = i/n_embd;
    const int64_t e = i - t*n_embd;
    const int32_t target = targets[t];
    if (target >= 0 && target < n_vocab && weights[t] != 0.0f && *n_active > 0) {
        const float coef = *grad*weights[t]/(float) *n_active;
        dst[i] -= coef*w[int64_t(target)*n_embd + e];
    }
}

static const float * fused_sparse_ce_head_f32(
        const ggml_tensor * w, ggml_cuda_pool_alloc<float> & scratch, cudaStream_t stream) {
    if (w->type == GGML_TYPE_F32) {
        return (const float *) w->data;
    }
    const int64_t n = ggml_nelements(w);
    scratch.alloc(n);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(w->type);
    GGML_ASSERT(to_fp32 != nullptr);
    to_fp32(w->data, scratch.get(), n, stream);
    return scratch.get();
}

struct fused_sparse_ce_work {
    ggml_cuda_pool_alloc<int32_t> n_active;
    ggml_cuda_pool_alloc<float> maxima;
    ggml_cuda_pool_alloc<float> sums;
    ggml_cuda_pool_alloc<float> target_logits;

    fused_sparse_ce_work(ggml_cuda_pool & pool, int64_t n_tokens) :
        n_active(pool, 1), maxima(pool, n_tokens), sums(pool, n_tokens), target_logits(pool, n_tokens) {}
};

static void fused_sparse_ce_prepare(
        cudaStream_t stream, const ggml_tensor * targets, const ggml_tensor * weights,
        int64_t n_vocab, fused_sparse_ce_work & work) {
    const int64_t n_tokens = ggml_nelements(targets);
    const int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(work.n_active.get(), 0, sizeof(int32_t), stream));
    fused_sparse_ce_count_active<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        (const int32_t *) targets->data, (const float *) weights->data,
        work.n_active.get(), n_tokens, n_vocab);
    fused_sparse_ce_init<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        work.maxima.get(), work.sums.get(), work.target_logits.get(), n_tokens);
    CUDA_CHECK(cudaGetLastError());
}

static int64_t fused_sparse_ce_tile_size(const ggml_tensor * dst, int64_t n_vocab) {
    const int32_t requested_tiles = std::max(1, ggml_get_op_params_i32(dst, 0));
    return std::min<int64_t>(1024, (n_vocab + requested_tiles - 1)/requested_tiles);
}

static void fused_sparse_ce_lse(
        ggml_backend_cuda_context & ctx, const float * h, const float * w,
        const ggml_tensor * targets, int64_t n_embd, int64_t n_tokens,
        int64_t n_vocab, int64_t tile_capacity, float * logits,
        fused_sparse_ce_work & work) {
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
    for (int64_t v0 = 0; v0 < n_vocab; v0 += tile_capacity) {
        const int64_t nv = std::min(tile_capacity, n_vocab - v0);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            nv, n_tokens, n_embd, &alpha, w + v0*n_embd, n_embd,
            h, n_embd, &beta, logits, tile_capacity));
        fused_sparse_ce_update_lse<<<n_tokens, 1, 0, stream>>>(
            logits, (const int32_t *) targets->data, work.maxima.get(), work.sums.get(),
            work.target_logits.get(), v0, nv, tile_capacity, n_tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_fused_sparse_ce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * h = dst->src[0], * w = dst->src[1];
    const ggml_tensor * targets = dst->src[2], * weights = dst->src[3];
    const int64_t n_embd = h->ne[0], n_tokens = h->ne[1], n_vocab = w->ne[1];
    GGML_ASSERT(h->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(targets->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(h) && ggml_is_contiguous(w));

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> w_scratch(pool);
    const float * w_f32 = fused_sparse_ce_head_f32(w, w_scratch, stream);
    fused_sparse_ce_work work(pool, n_tokens);
    fused_sparse_ce_prepare(stream, targets, weights, n_vocab, work);
    const int64_t tile = fused_sparse_ce_tile_size(dst, n_vocab);
    ggml_cuda_pool_alloc<float> logits(pool, tile*n_tokens);
    fused_sparse_ce_lse(ctx, (const float *) h->data, w_f32, targets,
        n_embd, n_tokens, n_vocab, tile, logits.get(), work);

    ggml_cuda_pool_alloc<float> losses(pool, n_tokens);
    const int threads = 256;
    fused_sparse_ce_finish_loss<<<(n_tokens + threads - 1)/threads, threads, 0, stream>>>(
        (const int32_t *) targets->data, (const float *) weights->data, work.n_active.get(),
        work.maxima.get(), work.sums.get(), work.target_logits.get(), losses.get(), n_tokens, n_vocab);
    CUDA_CHECK(cudaGetLastError());
    sum_f32_cuda(pool, losses.get(), (float *) dst->data, n_tokens, stream);
}

void ggml_cuda_fused_sparse_ce_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad = dst->src[0], * h = dst->src[1], * w = dst->src[2];
    const ggml_tensor * targets = dst->src[3], * weights = dst->src[4];
    const int64_t n_embd = h->ne[0], n_tokens = h->ne[1], n_vocab = w->ne[1];
    GGML_ASSERT(grad->type == GGML_TYPE_F32 && h->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(h) && ggml_is_contiguous(w) && ggml_is_contiguous(dst));

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> w_scratch(pool);
    const float * w_f32 = fused_sparse_ce_head_f32(w, w_scratch, stream);
    fused_sparse_ce_work work(pool, n_tokens);
    fused_sparse_ce_prepare(stream, targets, weights, n_vocab, work);
    const int64_t tile = fused_sparse_ce_tile_size(dst, n_vocab);
    ggml_cuda_pool_alloc<float> logits(pool, tile*n_tokens);
    fused_sparse_ce_lse(ctx, (const float *) h->data, w_f32, targets,
        n_embd, n_tokens, n_vocab, tile, logits.get(), work);

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;
    const int threads = 256;
    bool first = true;
    for (int64_t v0 = 0; v0 < n_vocab; v0 += tile) {
        const int64_t nv = std::min(tile, n_vocab - v0);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            nv, n_tokens, n_embd, &alpha, w_f32 + v0*n_embd, n_embd,
            (const float *) h->data, n_embd, &beta0, logits.get(), tile));
        const int64_t n_probs = nv*n_tokens;
        fused_sparse_ce_make_probs<<<(n_probs + threads - 1)/threads, threads, 0, stream>>>(
            logits.get(), (const float *) grad->data, (const int32_t *) targets->data,
            (const float *) weights->data, work.n_active.get(), work.maxima.get(), work.sums.get(),
            nv, tile, n_tokens, n_vocab);
        CUDA_CHECK(cudaGetLastError());
        const float * beta = first ? &beta0 : &beta1;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n_embd, n_tokens, nv, &alpha, w_f32 + v0*n_embd, n_embd,
            logits.get(), tile, beta, (float *) dst->data, n_embd));
        first = false;
    }
    const int64_t n_out = n_embd*n_tokens;
    fused_sparse_ce_subtract_target<<<(n_out + threads - 1)/threads, threads, 0, stream>>>(
        (const float *) grad->data, w_f32, (const int32_t *) targets->data,
        (const float *) weights->data, work.n_active.get(), (float *) dst->data,
        n_embd, n_tokens, n_vocab);
    CUDA_CHECK(cudaGetLastError());
}
