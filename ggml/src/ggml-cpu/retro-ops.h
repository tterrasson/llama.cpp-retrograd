#pragma once

// retro delta: the few symbols retro-ops.cpp has to hand back to ops.cpp.
//
// Everything the fork adds to the CPU backend lives in retro-ops.cpp and is
// reached through ops.h like any other compute function. This header exists
// only for the cases where an *upstream* dispatcher in ops.cpp has to call a
// kernel the fork added — one `#include` in ops.cpp instead of a declaration
// wedged into upstream's ops.h.

#include "ggml.h"
#include "ggml-impl.h"

// AdamW with stochastic rounding, F16 and BF16 parameters.
void ggml_compute_forward_opt_step_adamw_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst);

void ggml_compute_forward_opt_step_adamw_bf16(
        const ggml_compute_params * params,
        ggml_tensor * dst);

// SGD with stochastic rounding, F16 and BF16 parameters.
void ggml_compute_forward_opt_step_sgd_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst);

void ggml_compute_forward_opt_step_sgd_bf16(
        const ggml_compute_params * params,
        ggml_tensor * dst);
