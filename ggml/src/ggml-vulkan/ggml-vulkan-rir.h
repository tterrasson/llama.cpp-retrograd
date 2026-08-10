// retro delta: SPIR-V of the RIR-generated shaders, addressed by artifact name
// (docs/INT_RIR_V3.md §R0).
//
// This header is hand-written and stable: it names no kernel. Adding a
// generated kernel changes only `ggml-vulkan-rir-shaders.hpp` — which the
// generator writes and which exactly one small translation unit includes —
// so the 20 000 lines of ggml-vulkan.cpp are not recompiled for two lines of
// wiring. That was 6 of the 9 min 30 a `--build` of the RIR lane used to cost.
#pragma once

#include <cstddef>
#include <cstdint>

// Returns the SPIR-V module the fork carries for `artifact` — the name the
// registry publishes in `rir_variant_desc.artifact` — or nullptr when this
// build has no such shader. `len` is only written on success.
const unsigned char * ggml_vk_rir_spirv(const char * artifact, uint64_t * len);
