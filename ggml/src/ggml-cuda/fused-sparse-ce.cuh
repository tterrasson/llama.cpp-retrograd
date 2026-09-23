#pragma once

#include "common.cuh"

// retro delta: vocabulary cross-entropy without materializing dense logits.
void ggml_cuda_fused_sparse_ce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_fused_sparse_ce_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
