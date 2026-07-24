#include "conv_rs_gather.cuh"

// retro delta: single-launch replacement for the K-iteration host loop in
// build_conv_state's rollback branch (delta-net-base.cpp,
// [TAG_RECURRENT_ROLLBACK_SPLITS]). Only reached when a caller sets
// n_rs_seq > 0; Retroback pins it to 0 (see retro_backend.cpp), so this is
// dead weight there but kept correct and tested for llama.cpp callers that do
// enable rollback. One thread per (channel, seq); each thread
// gathers its K overlapping causal-conv windows in a small inner loop. Slot 0
// is the window ending at the last token, slot s reads s tokens further back,
// clamped to the start of conv_input's new-token region when n_seq_tokens < K
// -- reproducing the original per-slot `std::max<int64_t>(0, ...)` exactly.
static __global__ void conv_rs_gather_cuda(
        const float * __restrict__ src,
        float *       __restrict__ dst,
        const int64_t kernel_m1,
        const int64_t n_channels,
        const int64_t n_seqs,
        const int64_t K,
        const int64_t base,
        const int64_t nb0_f,
        const int64_t nb1_f,
        const int64_t nb2_f,
        const int64_t row_count) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_channels * n_seqs) {
        return;
    }
    const int64_t c = idx % n_channels;
    const int64_t s = idx / n_channels;

    const float * src_seq  = src + s * nb2_f + c * nb1_f;
    float *       dst_base = dst + s * row_count;

    for (int64_t slot = 0; slot < K; ++slot) {
        int64_t s_idx = base - slot;
        if (s_idx < 0) {
            s_idx = 0;
        }
        const float * src_win = src_seq + s_idx * nb0_f;
        float *       dst_col = dst_base + slot * row_count * n_seqs + c * kernel_m1;
        for (int64_t k = 0; k < kernel_m1; ++k) {
            dst_col[k] = src_win[k * nb0_f];
        }
    }
}

void ggml_cuda_op_conv_rs_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * conv_input = dst->src[0];

    GGML_ASSERT(conv_input->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(conv_input));

    const int64_t kernel_m1 = ggml_get_op_params_i32(dst, 0);
    const int64_t K         = ggml_get_op_params_i32(dst, 1);

    const int64_t n_channels = conv_input->ne[1];
    const int64_t n_seqs     = conv_input->ne[2];
    const int64_t base       = conv_input->ne[0] - kernel_m1; // n_seq_tokens
    const int64_t row_count  = kernel_m1 * n_channels;

    const int64_t nb0_f = conv_input->nb[0] / sizeof(float);
    const int64_t nb1_f = conv_input->nb[1] / sizeof(float);
    const int64_t nb2_f = conv_input->nb[2] / sizeof(float);

    const float * src_d = (const float *) conv_input->data;
    float *       dst_d = (float *) dst->data;

    const int64_t total = n_channels * n_seqs;
    const int     block = 256;
    const int64_t grid  = (total + block - 1) / block;

    conv_rs_gather_cuda<<<grid, block, 0, ctx.stream()>>>(
            src_d, dst_d, kernel_m1, n_channels, n_seqs, K, base,
            nb0_f, nb1_f, nb2_f, row_count);
}
