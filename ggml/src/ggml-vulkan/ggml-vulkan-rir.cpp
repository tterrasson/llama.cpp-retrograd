// retro delta: the one translation unit that sees the RIR shader blobs
// (docs/INT_RIR_V3.md §R0).
//
// It exists so that nothing else does. The generated header below gains two
// `extern` lines per new kernel; keeping that inside ggml-vulkan.cpp meant
// recompiling a 20 000-line unit for wiring it never reads. Here the cost is
// this file, which is a table and a string comparison.
//
// The table is not written by hand either: `RIR_VK_SHADERS(X)` is emitted by
// vulkan-shaders-gen from the shaders it actually compiled, so a kernel added
// to the registry cannot be missing from it, and one removed cannot linger.

#include "ggml-vulkan-rir.h"

#include "ggml-vulkan-rir-shaders.hpp"

#include <cstring>

const unsigned char * ggml_vk_rir_spirv(const char * artifact, uint64_t * len) {
    if (artifact == nullptr) {
        return nullptr;
    }
    struct entry {
        const char *          name;
        const unsigned char * data;
        uint64_t              len;
    };
    // The shader symbol is `rir_<artifact>`; the registry publishes the
    // artifact, so the prefix is the only thing this file knows about naming.
    static const entry entries[] = {
#define RIR_VK_ENTRY(name) { #name, name##_data, name##_len },
        RIR_VK_SHADERS(RIR_VK_ENTRY)
#undef RIR_VK_ENTRY
        // Terminator: a build with no RIR shader must still declare an array.
        { nullptr, nullptr, 0 },
    };
    for (const entry & e : entries) {
        if (e.name != nullptr && std::strncmp(e.name, "rir_", 4) == 0
                && std::strcmp(e.name + 4, artifact) == 0) {
            *len = e.len;
            return e.data;
        }
    }
    return nullptr;
}
