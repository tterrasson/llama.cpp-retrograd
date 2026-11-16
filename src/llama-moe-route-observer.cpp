#include "llama-moe-route-observer.h"

#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>

void llama_moe_route_retain(std::vector<ggml_tensor *> & routes, int layer, ggml_tensor * ids) {
    GGML_ASSERT(layer >= 0);
    if (routes.size() <= size_t(layer)) {
        routes.resize(size_t(layer) + 1);
    }
    GGML_ASSERT(routes[layer] == nullptr);
    routes[layer] = ids;
    // Retain the backing allocation of the selected ids (often an argsort view).
    // A graph output flag would prevent some backends from fusing the router.
    (ids->view_src ? ids->view_src : ids)->flags |= GGML_TENSOR_FLAG_RETAIN;
}

llama_moe_route_observer::llama_moe_route_observer(const llama_model & model, llama_moe_route_cb cb, void * user_data) :
    cb(cb), user_data(user_data), seen(model.layers.size()), readback(model.layers.size()) {
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        for (const auto * bank : {l.ffn_gate_up_exps, l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps}) {
            if (bank != nullptr) {
                layer_of.emplace(bank, int32_t(il));
            }
        }
    }
}

void llama_moe_route_observer::begin() {
    observing = cb(user_data, LLAMA_MOE_ROUTE_BEGIN, -1, nullptr, 0, 0, 0);
    if (observing) {
        std::fill(seen.begin(), seen.end(), 0);
    }
}

void llama_moe_route_observer::emit(int32_t layer, llama_moe_route_event event, const int32_t * ids,
        int64_t n_expert_used, int64_t n_tokens, int64_t row_stride) {
    if (!observing || layer < 0 || size_t(layer) >= seen.size() || seen[layer] || n_expert_used*n_tokens == 0) {
        return;
    }
    seen[layer] = 1;
    cb(user_data, event, layer, ids, n_expert_used, n_tokens, row_stride);
}

void llama_moe_route_observer::emit(int32_t layer, llama_moe_route_event event, const ggml_tensor * ids, const int32_t * data) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[2] == 1 && ids->ne[3] == 1);
    emit(layer, event, data, ids->ne[0], ids->ne[1], int64_t(ids->nb[1]/sizeof(int32_t)));
}

void llama_moe_route_observer::observe_slots(int32_t layer, const ggml_tensor * ids, const int32_t * data) {
    // the slot planner reads the flattened [n_expert_used, n_tokens] ids, see llama_moe_cache::build_lookup
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ids->op == GGML_OP_RESHAPE);
    const int64_t n_used = ids->src[0]->ne[0];
    emit(layer, LLAMA_MOE_ROUTE_SLOTS, data, n_used, ggml_nelements(ids)/n_used, n_used);
}

void llama_moe_route_observer::observe_copy(const ggml_tensor * bank, const ggml_tensor * ids, const int32_t * data) {
    if (!observing) {
        return;
    }
    const auto it = layer_of.find(bank);
    if (it != layer_of.end()) {
        emit(it->second, LLAMA_MOE_ROUTE_COPY, ids, data);
    }
}

void llama_moe_route_observer::end(ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & routes, bool success) {
    if (!observing) {
        return;
    }
    observing = success;
    if (!success) {
        return;
    }
    bool reads = false;
    for (size_t il = 0; il < routes.size() && il < seen.size(); ++il) {
        ggml_tensor * ids = routes[il];
        if (ids != nullptr && !seen[il] && ggml_nelements(ids) != 0) {
            const size_t bytes = ggml_nbytes(ids);
            readback[il].resize(bytes/sizeof(int32_t));
            ggml_backend_tensor_get_async(ggml_backend_sched_get_tensor_backend(sched, ids),
                    ids, readback[il].data(), 0, bytes);
            reads = true;
        }
    }
    if (reads) {
        ggml_backend_sched_synchronize(sched);
    }
    for (size_t il = 0; il < routes.size() && il < seen.size(); ++il) {
        if (routes[il] != nullptr && !seen[il] && ggml_nelements(routes[il]) != 0) {
            emit(int32_t(il), LLAMA_MOE_ROUTE_READBACK, routes[il], readback[il].data());
        }
    }
    cb(user_data, LLAMA_MOE_ROUTE_END, -1, nullptr, 0, 0, 0);
    observing = false;
}

bool llama_context::moe_route_set_observer(llama_moe_route_cb cb, void * user_data) {
    if (cb != nullptr && model.hparams.n_expert == 0) {
        return false;
    }
    moe_route = cb != nullptr ? std::make_unique<llama_moe_route_observer>(model, cb, user_data) : nullptr;
    if (cparams.moe_route != (moe_route != nullptr)) {
        cparams.moe_route = moe_route != nullptr;
        sched_need_reserve = true;
    }
    return true;
}

bool llama_moe_route_set_observer(llama_context * ctx, llama_moe_route_cb cb, void * user_data) {
    return ctx != nullptr && ctx->moe_route_set_observer(cb, user_data);
}
