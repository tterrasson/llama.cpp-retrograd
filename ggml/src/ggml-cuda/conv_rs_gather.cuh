#include "common.cuh"
#include "ggml.h"

// retro delta: see ggml_conv_rs_gather (ggml.h) / GGML_OP_CONV_RS_GATHER.
void ggml_cuda_op_conv_rs_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
