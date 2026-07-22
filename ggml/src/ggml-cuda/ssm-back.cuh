#include "common.cuh"

// retro delta: SSM backward for recurrent (Mamba/Falcon-H1) LoRA training.
void ggml_cuda_ssm_conv_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_ssm_scan_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
