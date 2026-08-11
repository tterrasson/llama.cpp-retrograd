#pragma once

// retro delta: the few symbols retro-ops.cpp has to hand back to ops.cpp.
//
// Everything the fork adds to the CPU backend lives in retro-ops.cpp and is
// reached through ops.h like any other compute function. This header exists
// only for the cases where an *upstream* dispatcher in ops.cpp has to call a
// kernel the fork added — one `#include` in ops.cpp instead of a declaration
// wedged into upstream's ops.h (docs/RETRO_FORK.md).

#include "ggml.h"
#include "ggml-impl.h"

// GGML_OP_OPT_STEP_ADAMW, F16 parameters with stochastic rounding. Called from
// ggml_compute_forward_opt_step_adamw's type switch in ops.cpp.
void ggml_compute_forward_opt_step_adamw_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst);
