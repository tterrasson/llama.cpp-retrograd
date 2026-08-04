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

// Kept in the header because nvcc's host stub needs the complete by-value
// kernel-argument type before it includes the generated launch wrappers.
struct fa_back_mma_args {
    int64_t d, nq, nkv, nhead, nheadk, ratio;
    size_t nbq0, nbq1, nbq2, nbq3;
    size_t nbk1, nbk2, nbk3;
    size_t nbv1, nbv2, nbv3;
    size_t nbm0, nbm1, nbm3;
    size_t nbo0, nbo1, nbo2, nbo3;
    size_t nbd0, nbd1, nbd2, nbd3;
    size_t off_k, off_v, off_s;
    const int32_t * window_idxs;
    int64_t window_nwin, window_stride, window_stream0;
    float scale, softcap;
    bool has_mask, want_dq, want_dk, want_dv;
};

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
