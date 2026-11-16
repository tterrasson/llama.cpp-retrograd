#pragma once

#include "llama.h"

#include "ggml-backend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct llama_model;

// marks the ids a layer's expert banks consume so they stay readable after the graph
void llama_moe_route_retain(std::vector<ggml_tensor *> & routes, int layer, ggml_tensor * ids);

// the per-ubatch state behind llama_moe_route_set_observer: it delivers each
// layer's ids once, from the first path that read them on the host, and reads
// back the layers no path saw
class llama_moe_route_observer {
public:
    llama_moe_route_observer(const llama_model & model, llama_moe_route_cb cb, void * user_data);

    void begin();
    void observe_slots(int32_t layer, const ggml_tensor * ids, const int32_t * data);
    void observe_copy(const ggml_tensor * bank, const ggml_tensor * ids, const int32_t * data);
    // success: the ubatch computed, read back the remaining layers and close it
    void end(ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & routes, bool success);

private:
    void emit(int32_t layer, llama_moe_route_event event, const int32_t * ids,
              int64_t n_expert_used, int64_t n_tokens, int64_t row_stride);
    void emit(int32_t layer, llama_moe_route_event event, const ggml_tensor * ids, const int32_t * data);

    llama_moe_route_cb cb;
    void * user_data;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of;
    std::vector<uint8_t> seen;
    std::vector<std::vector<int32_t>> readback;
    bool observing = false;
};
