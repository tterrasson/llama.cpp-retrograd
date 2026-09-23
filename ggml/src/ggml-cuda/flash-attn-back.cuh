#include "common.cuh"

// retro delta: differentiable Flash Attention backward (GGML_OP_FLASH_ATTN_BACK)
// for LoRA training. Mirrors the fork's streaming CPU reference and the Vulkan
// kernel: gradients dQ/dK/dV in F32 packed into the result tensor.
void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
