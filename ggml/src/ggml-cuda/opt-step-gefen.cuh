#include "common.cuh"

// retro delta: fixed-block Gefen. One CUDA block owns one quantization block in
// both phases, so this is a reduction width, not a partition of the parameter.
#define CUDA_OPT_STEP_GEFEN_BLOCK_SIZE 256

void ggml_cuda_opt_step_gefen_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_opt_step_gefen(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
