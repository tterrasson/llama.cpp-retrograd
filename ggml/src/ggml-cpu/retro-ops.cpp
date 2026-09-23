// retro delta: the CPU kernels this fork adds to ggml, kept out of ops.cpp so
// upstream's file stays upstream's. Same headers as ops.cpp, same dispatch:
// ops.h declares these, ggml-cpu.c calls them, nothing else changes.
//
// Only whole added functions live here. Where the fork rewrites an upstream
// function in place -- gated delta net backward, flash attention backward --
// the code stays in ops.cpp, because there is no seam to cut along.
//
// Adding a CPU kernel to the fork belongs here.

#include "ops.h"
#include "retro-ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "binary-ops.h"
#include "simd-gemm.h"
#include "ggml.h"
#include "unary-ops.h"
#include "vec.h"

#include <algorithm>
#include <numeric>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <vector>

// ---- l2_norm_back ----


// ---- ssm_conv_back / ssm_scan_back ----


// ---- fused sparse cross-entropy ----


// ---- F16 AdamW with stochastic rounding ----

