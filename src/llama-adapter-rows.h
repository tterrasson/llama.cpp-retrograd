#pragma once

// retro delta: per-row LoRA selection. Several LoRA adapters of one base model
// are stacked per target weight so a single decode can apply a different
// adapter to each sequence: y = W x + B[a] (A[a] x) s[a], with `a` the slot the
// token's sequence routes to. The two products are ggml_mul_mat_id calls over
// the stacks (shrink, then expand), the same gather MoE layers use for their
// experts.

#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_adapter_lora;

struct llama_adapter_lora_rows {
    // One stacked target weight. Slot 0 is the null adapter (zero factors), so
    // a token routed to no adapter gathers an exact zero delta; slot k >= 1 is
    // adapters[k - 1]. A slot whose adapter does not target this weight is
    // zero as well.
    struct target {
        ggml_tensor * a     = nullptr; // F32 [n_in,  r_max, n_slots]
        ggml_tensor * b     = nullptr; // F32 [r_max, n_out, n_slots]
        ggml_tensor * scale = nullptr; // F32 [1, n_slots]: alpha/rank * user scale, 0 for slot 0
    };

    // keyed by the base weight's name
    std::unordered_map<std::string, target> targets;

    // the routed adapters, slot k >= 1 is adapters[k - 1]; not owned
    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> scales;

    // the largest rank over the routed adapters: every slot is zero-padded to it,
    // which is exact (the padded rows of A and columns of B contribute nothing)
    // and keeps each adapter's own alpha/rank scale
    int64_t r_max = 0;

    // bytes held by the stacks, the padding included
    size_t n_bytes = 0;
    // bytes the same adapters hold unpadded, so a caller can report the padding cost
    size_t n_bytes_unpadded = 0;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    int32_t n_slots() const { return (int32_t) adapters.size() + 1; }

    const target * get(const ggml_tensor * w) const;

    // nodes the routed delta adds per target to a forward graph
    uint32_t get_n_nodes() const {
        return (uint32_t) targets.size() * 12u;
    }

    // Builds the stacks. Refuses (false, `err` set) what this path cannot
    // apply per row: activated LoRA (aLoRA), token-embedding targets and
    // expert (3D) weights.
    bool init(
            const llama_model & model,
            llama_adapter_lora * const * adapters,
            size_t n_adapters,
            const float * scales,
            std::string & err);
};

struct llama_adapter_lora_rows_info {
    int32_t  n_adapters;
    int32_t  r_max;
    uint64_t n_bytes;
    uint64_t n_bytes_unpadded;
};

//
// context API
//
// A sequence's route picks what its tokens apply on top of the base weights:
//   0  the context's LoRA set (llama_set_adapters_lora), the default
//  -1  no adapter at all, even when the context has a LoRA set
//   k  the k-th routed adapter (1-based) and not the context's LoRA set
// A token shared by several sequences takes the route of its first one.
// Without a routed set every route is 0 and the graph is the upstream one.

// Stacks `n_adapters` adapters as the routed set, replacing any previous one
// and resetting every route to 0. n_adapters == 0 installs an empty set, which
// is enough for route -1. On failure the context is left without a routed set
// and `err` (when not null) says why.
LLAMA_API int32_t llama_set_adapters_lora_rows(
        struct llama_context * ctx,
        struct llama_adapter_lora * const * adapters,
        size_t n_adapters,
        const float * scales,
        std::string * err);

// Drops the routed set and every route.
LLAMA_API void llama_clear_adapters_lora_rows(struct llama_context * ctx);

// Sets the route of `seq_id`. Refused (-1) without a routed set, for a
// sequence outside the context or a route outside [-1, n_adapters].
LLAMA_API int32_t llama_set_seq_lora_route(struct llama_context * ctx, llama_seq_id seq_id, int32_t route);

// Resets every route to 0.
LLAMA_API void llama_clear_seq_lora_routes(struct llama_context * ctx);

// false when the context has no routed set.
LLAMA_API bool llama_get_adapters_lora_rows_info(const struct llama_context * ctx, llama_adapter_lora_rows_info * out);
