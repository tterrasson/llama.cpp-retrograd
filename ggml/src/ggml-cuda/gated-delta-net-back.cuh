#include "common.cuh"

// retro delta: analytic backward for GGML_OP_GATED_DELTA_NET (Qwen3-Next /
// KDA-style gated delta net), mirroring the CPU reference
// (ggml-cpu/ops.cpp: ggml_compute_forward_gated_delta_net_back_f32).
void ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
