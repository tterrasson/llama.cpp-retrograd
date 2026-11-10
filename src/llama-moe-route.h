#pragma once

#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct llama_model;

void llama_moe_route_retain(std::vector<ggml_tensor *> & routes, int layer, ggml_tensor * ids);

// The validated graph uses one selection for all routed expert banks in a layer.
class llama_moe_route {
public:
    llama_moe_route(const llama_model & model, uint32_t sample_every);

    void begin();
    void observe_slots(int layer, const ggml_tensor * ids, const int32_t * data);
    void observe_copy(const ggml_tensor * bank, const ggml_tensor * ids, const int32_t * data);
    void finish(ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & routes);

    size_t counts(uint32_t layer, uint32_t * out, size_t n_max) const;
    void paths(uint64_t out[3]) const;
    void reset();

private:
    enum source { none, slots, copy, readback };
    void observe(size_t layer, source path, const ggml_tensor * ids, const int32_t * data);

    struct row {
        source path = none;
        std::vector<uint32_t> counts;
        std::vector<int32_t> pending;
        std::vector<int32_t> readback_data;
    };

    std::vector<row> rows;
    std::unordered_map<const ggml_tensor *, size_t> layer_of;
    std::array<uint64_t, 3> path_counts = {};
    uint32_t period;
    uint32_t until_sample = 0;
    bool active = false;
};
