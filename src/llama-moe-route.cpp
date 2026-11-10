#include "llama-moe-route.h"

#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>

void llama_moe_route_retain(std::vector<ggml_tensor *> & routes, int layer, ggml_tensor * ids) {
    GGML_ASSERT(layer >= 0);
    if (routes.size() <= size_t(layer)) {
        routes.resize(size_t(layer) + 1);
    }
    GGML_ASSERT(routes[layer] == nullptr);
    routes[layer] = ids;
    // Retain the backing allocation of the selected IDs (often an argsort view).
    // A graph output flag would prevent some backends from fusing the router.
    (ids->view_src ? ids->view_src : ids)->flags |= GGML_TENSOR_FLAG_RETAIN;
}

llama_moe_route::llama_moe_route(const llama_model & model, uint32_t sample_every) :
    rows(model.layers.size()), period(sample_every) {
    for (size_t il = 0; il < rows.size(); ++il) {
        const auto & l = model.layers[il];
        for (const auto * bank : {l.ffn_gate_up_exps, l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps}) {
            if (bank != nullptr) {
                GGML_ASSERT(bank->ne[2] > 0 && bank->ne[2] == model.hparams.n_expert);
                rows[il].counts.resize(model.hparams.n_expert);
                layer_of.emplace(bank, il);
            }
        }
    }
}

void llama_moe_route::begin() {
    active = until_sample == 0;
    if (active) {
        until_sample = period - 1;
        for (auto & row : rows) {
            row.path = none;
        }
    } else {
        --until_sample;
    }
}

void llama_moe_route::observe(size_t layer, source path, const ggml_tensor * ids, const int32_t * data) {
    auto & row = rows.at(layer);
    if (row.path != none || row.counts.empty() || ggml_nelements(ids) == 0) {
        return;
    }
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    row.path = path;
    row.pending.clear();
    for (int64_t j = 0; j < ids->ne[1]; ++j) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t id = data[(j*ids->nb[1] + i*ids->nb[0])/sizeof(int32_t)];
            GGML_ASSERT(id >= 0 && size_t(id) < row.counts.size());
            row.pending.push_back(id);
        }
    }
}

void llama_moe_route::observe_slots(int layer, const ggml_tensor * ids, const int32_t * data) {
    if (active) {
        observe(layer, slots, ids, data);
    }
}

void llama_moe_route::observe_copy(const ggml_tensor * bank, const ggml_tensor * ids, const int32_t * data) {
    if (active) {
        const auto it = layer_of.find(bank);
        if (it != layer_of.end()) {
            observe(it->second, copy, ids, data);
        }
    }
}

void llama_moe_route::finish(ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & routes) {
    if (!active) {
        return;
    }
    bool reads = false;
    for (size_t il = 0; il < routes.size(); ++il) {
        auto * ids = routes[il];
        auto & row = rows.at(il);
        if (ids != nullptr && row.path == none && ggml_nelements(ids) != 0) {
            const size_t bytes = ggml_nbytes(ids);
            row.readback_data.resize(bytes/sizeof(int32_t));
            ggml_backend_tensor_get_async(ggml_backend_sched_get_tensor_backend(sched, ids),
                    ids, row.readback_data.data(), 0, bytes);
            reads = true;
        }
    }
    if (reads) {
        ggml_backend_sched_synchronize(sched);
    }
    for (size_t il = 0; il < rows.size(); ++il) {
        auto & row = rows[il];
        if (row.path == none && il < routes.size() && routes[il] != nullptr) {
            observe(il, readback, routes[il], row.readback_data.data());
        }
        if (row.path != none) {
            for (const auto id : row.pending) {
                auto & count = row.counts[id];
                count += count != std::numeric_limits<uint32_t>::max();
            }
            auto & count = path_counts[row.path - 1];
            count += count != std::numeric_limits<uint64_t>::max();
        }
    }
    active = false;
}

size_t llama_moe_route::counts(uint32_t layer, uint32_t * out, size_t n_max) const {
    if (layer >= rows.size()) {
        return 0;
    }
    const auto & counts = rows[layer].counts;
    if (out != nullptr) {
        std::copy_n(counts.begin(), std::min(n_max, counts.size()), out);
    }
    return counts.size();
}

void llama_moe_route::paths(uint64_t out[3]) const {
    std::copy(path_counts.begin(), path_counts.end(), out);
}

void llama_moe_route::reset() {
    for (auto & row : rows) {
        std::fill(row.counts.begin(), row.counts.end(), 0);
    }
    path_counts.fill(0);
}

bool llama_context::moe_route_enable(uint32_t sample_every) {
    if (sample_every != 0 && (model.arch != LLM_ARCH_OLMOE || model.hparams.n_expert == 0)) {
        return false;
    }
    moe_route = sample_every != 0 ? std::make_unique<llama_moe_route>(model, sample_every) : nullptr;
    cparams.moe_route = moe_route != nullptr;
    sched_need_reserve = true;
    return true;
}

bool llama_moe_route_enable(llama_context * ctx, uint32_t sample_every) {
    return ctx->moe_route_enable(sample_every);
}

size_t llama_moe_route_counts(const llama_context * ctx, uint32_t layer, uint32_t * out, size_t n_max) {
    const auto * route = ctx->get_moe_route();
    return route ? route->counts(layer, out, n_max) : 0;
}

void llama_moe_route_paths(const llama_context * ctx, uint64_t out[3]) {
    const auto * route = ctx->get_moe_route();
    if (route) {
        route->paths(out);
    } else {
        std::fill_n(out, 3, 0);
    }
}

void llama_moe_route_reset(llama_context * ctx) {
    ctx->moe_route_reset();
}
