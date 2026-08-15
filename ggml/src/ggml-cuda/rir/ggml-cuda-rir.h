// retro delta: the RIR launch stubs, addressed by artifact name
// (docs/CUDA_v1.md §C2, §3.1).
//
// This header is hand-written and stable: it names no kernel. Adding a
// generated kernel changes only `rir_cuda_launchers.h` — which the generator
// writes and which exactly one small translation unit expands — plus one
// `rir/rir_<artifact>.cu` the backend's `file(GLOB "rir/*.cu")` picks up.
//
// Why a pointer and not a name: a CUDA `__global__` is not launched by its
// name. `<<<>>>` needs the typed symbol in a translation unit `nvcc` sees, so
// what crosses the boundary is the generated `extern "C"` stub — and an adapter
// that holds a function pointer cannot name a kernel that does not exist.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// The signature every generated stub has, published once. `bufs` is one device
// pointer per binding in registry order, `params` the bytes
// `ggml_rir_fill_params` laid out for this variant's `rir_<kernel>_params`, and
// `grid` the three workgroup counts `ggml_rir_grid` computed. The block is not
// passed: it is the schedule's, and the stub was generated with it.
typedef void (*ggml_cuda_rir_launch_fn)(void * const * bufs, const void * params,
                                        const uint32_t grid[3], cudaStream_t stream);

// Returns the stub the fork carries for `artifact` — the name the registry
// publishes in `rir_variant_desc.artifact` — or nullptr when this build has no
// such kernel.
ggml_cuda_rir_launch_fn ggml_cuda_rir_launcher(const char * artifact);
