#include "common.cuh"

// retro delta: differentiable Flash Attention backward (GGML_OP_FLASH_ATTN_BACK)
// for LoRA training. Mirrors the fork's streaming CPU reference and the Vulkan
// kernel: gradients dQ/dK/dV in F32 packed into the result tensor.

// Largest head dimension any kernel instantiation covers; the supports check in
// ggml-cuda.cu must use this rather than a hardcoded literal. Only bump it
// together with a new bucket *and* a gradient-parity test at that head
// dimension: the op reports support to the scheduler, so an untested bucket
// would silently produce wrong gradients rather than fail. Bounded by
// GGML_FLASH_ATTN_BACK_MAX_HEAD_DIM (static_assert in flash-attn-back.cu).
#define FA_BACK_MAX_D 512

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
