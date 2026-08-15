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

// retro delta: the CUDA adapter (docs/CUDA_v1.md §C4).
//
// Three entry points, and everything else about the dispatch lives in
// `ggml-rir.cpp`: the mode, the policy, the site attribution, the counters, the
// `require` enforcement and the whole portable half of the contract. What
// remains below is what needs *this* device to answer.
struct ggml_backend_cuda_context;
struct ggml_cgraph;
struct ggml_tensor;

// The one line an integrated op adds at its dispatch site. Returns true when a
// generated variant was launched and the native kernel must not run.
//
// It is called from inside `ggml_cuda_compute_forward`, which is reached only
// when `ggml_cuda_try_fuse` found nothing — so a RIR site is always **after**
// the fusion test, never before (docs/CUDA_v1.md §3.6). That ordering is a
// constraint the `ggml-vulkan.cpp` template does not teach, Vulkan fusing far
// less.
bool ggml_cuda_rir_try(ggml_backend_cuda_context * ctx, const ggml_tensor * node);

// The whole encoder of a pair whose **native kernel has been retired**
// (docs/CUDA_v1.md §C7). `L2_NORM_BACK` routes here and adds no line: the
// generated variant is the only implementation, so there is no second branch to
// write.
//
// It cannot fail silently. `ggml_rir_supports_op` is what admitted this node and
// it evaluated the same portable contract; the only ways to arrive and be
// refused are a device-half rejection — this build cannot run on this machine —
// or a mode lowered after the graph split, and both abort inside
// `ggml_rir_dispatch_begin` naming the op and the reason.
void ggml_cuda_rir_only(ggml_backend_cuda_context * ctx, const ggml_tensor * node);

// The device half of the contract, in the shape `ggml_rir_evaluate` and
// `ggml_rir_preflight_graph` consume. Pure and counter-free: the same answer
// serves the preflight, which asks about a node it will not launch, and the
// dispatch site, which is the only one entitled to count.
int32_t ggml_cuda_rir_device_check(void * device_ctx, const ggml_tensor * node);

// `require`'s graph preflight plus the census, at the top of graph_compute.
// Returns false after recording the first violation, and the caller must then
// fail the graph rather than run a native kernel.
bool ggml_cuda_rir_graph_begin(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph);
