// retro delta: the one translation unit that sees the RIR launch stubs
// (docs/CUDA_v1.md §C2).
//
// It exists so that nothing else does, which is the same reason
// `ggml-vulkan-rir.cpp` exists — and one CUDA adds: a generated `.cu` includes
// `rir_kernel_params.h`, so a build that named the stubs from `ggml-cuda.cu`
// would recompile that unit for every kernel added to the registry.
//
// The table is not written by hand either: `RIR_CUDA_LAUNCHERS(X)` is emitted
// by rir-gen from the production variants it actually generated, so a kernel
// added to the registry cannot be missing from it, and one removed cannot
// linger. This file expands it twice — once to declare, once to tabulate — and
// that is its whole content.

#include "ggml-cuda-rir.h"

#include "rir_cuda_launchers.h"

#include <cstring>

// The declarations. The signature is `ggml_cuda_rir_launch_fn`'s, written out
// because a declaration cannot be spelled through a function-pointer typedef.
#define RIR_CUDA_DECL(name)                                                     \
    extern "C" void rir_launch_##name(void * const * bufs, const void * params, \
                                      const uint32_t grid[3], cudaStream_t stream);
RIR_CUDA_LAUNCHERS(RIR_CUDA_DECL)
#undef RIR_CUDA_DECL

ggml_cuda_rir_launch_fn ggml_cuda_rir_launcher(const char * artifact) {
    if (artifact == nullptr) {
        return nullptr;
    }
    struct entry {
        const char *            name;
        ggml_cuda_rir_launch_fn fn;
    };
    static const entry entries[] = {
#define RIR_CUDA_ENTRY(name) { #name, rir_launch_##name },
        RIR_CUDA_LAUNCHERS(RIR_CUDA_ENTRY)
#undef RIR_CUDA_ENTRY
        // Terminator: a build with no RIR kernel must still declare an array.
        { nullptr, nullptr },
    };
    for (const entry & e : entries) {
        if (e.name != nullptr && std::strcmp(e.name, artifact) == 0) {
            return e.fn;
        }
    }
    return nullptr;
}
