// retro delta: per-row LoRA selection, see llama-adapter-rows.h.

#include "llama-adapter-rows.h"

#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <map>

const llama_adapter_lora_rows::target * llama_adapter_lora_rows::get(const ggml_tensor * w) const {
    const auto it = targets.find(w->name);
    return it == targets.end() ? nullptr : &it->second;
}

// Reads a 2D adapter factor into `dst` as F32, whatever its stored type.
static bool llama_adapter_rows_read_f32(const ggml_tensor * t, std::vector<float> & dst, std::string & err) {
    const int64_t n = ggml_nelements(t);
    dst.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
        return true;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        err = std::string("LoRA tensor '") + t->name + "' has a type that cannot be read as F32";
        return false;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    traits->to_float(raw.data(), dst.data(), n);
    return true;
}

bool llama_adapter_lora_rows::init(
        const llama_model & model,
        llama_adapter_lora * const * adapters_in,
        size_t n_adapters,
        const float * scales_in,
        std::string & err) {
    // Empty sets accept null pointers; do not form a range from nullptr.
    if (n_adapters > 0) {
        adapters.assign(adapters_in, adapters_in + n_adapters);
        scales.assign(scales_in, scales_in + n_adapters);
    }

    // the union of the targets, with the buffer type of the first adapter that
    // holds each one (the adapter loader already chose it next to the base weight)
    std::map<std::string, ggml_backend_buffer_type_t> names;
    std::map<std::string, std::pair<int64_t, int64_t>> shapes; // n_in, n_out
    for (size_t i = 0; i < n_adapters; ++i) {
        const llama_adapter_lora * adapter = adapters[i];
        if (adapter == nullptr) {
            err = "a routed LoRA adapter is null";
            return false;
        }
        if (!adapter->alora_invocation_tokens.empty()) {
            err = "an activated LoRA (aLoRA) cannot be selected per row";
            return false;
        }
        for (const auto & [name, w] : adapter->ab_map) {
            const ggml_tensor * base = model.get_tensor(name.c_str());
            if (base == nullptr) {
                err = "LoRA target '" + name + "' is not a tensor of the base model";
                return false;
            }
            if (name.size() >= 17 && name.compare(name.size() - 17, 17, "token_embd.weight") == 0) {
                err = "LoRA target '" + name + "' is a token embedding, which cannot be selected per row";
                return false;
            }
            if (ggml_n_dims(w.a) > 2 || ggml_n_dims(w.b) > 2 || base->ne[2] > 1) {
                err = "LoRA target '" + name + "' is an expert (3D) weight, which cannot be selected per row";
                return false;
            }
            r_max = std::max<int64_t>(r_max, w.a->ne[1]);
            n_bytes_unpadded += ggml_nelements(w.a) * sizeof(float) + ggml_nelements(w.b) * sizeof(float);
            const auto shape = std::make_pair(w.a->ne[0], w.b->ne[1]);
            const auto [it, inserted] = shapes.emplace(name, shape);
            if (!inserted && it->second != shape) {
                err = "LoRA target '" + name + "' has different shapes across the routed adapters";
                return false;
            }
            names.emplace(name, ggml_backend_buffer_get_type(w.a->buffer));
        }
    }
    if (names.empty()) {
        return true;
    }

    const int64_t n_slots_v = n_slots();

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (const auto & [name, buft] : names) {
        ggml_context * ctx = nullptr;
        if (auto it = ctx_map.find(buft); it != ctx_map.end()) {
            ctx = it->second;
        } else {
            ggml_init_params params = {
                /*.mem_size   =*/ 3 * names.size() * ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ctx = ggml_init(params);
            if (ctx == nullptr) {
                err = "failed to create a context for the routed LoRA stacks";
                return false;
            }
            ctxs.emplace_back(ctx);
            ctx_map[buft] = ctx;
        }
        const auto [n_in, n_out] = shapes.at(name);
        target t;
        t.a     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in,  r_max, n_slots_v);
        t.b     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, r_max, n_out, n_slots_v);
        t.scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_slots_v);
        ggml_format_name(t.a,     "%s.lora_rows_a",     name.c_str());
        ggml_format_name(t.b,     "%s.lora_rows_b",     name.c_str());
        ggml_format_name(t.scale, "%s.lora_rows_scale", name.c_str());
        targets.emplace(name, t);
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft) };
        if (!buf) {
            err = "failed to allocate the routed LoRA stacks";
            return false;
        }
        ggml_backend_buffer_clear(buf.get(), 0);
        n_bytes += ggml_backend_buffer_get_size(buf.get());
        bufs.emplace_back(std::move(buf));
    }

    // fill each slot through host staging: the zero padding is left by the clear
    std::vector<float> src;
    std::vector<float> padded;
    for (auto & [name, t] : targets) {
        const int64_t n_in  = t.a->ne[0];
        const int64_t n_out = t.b->ne[1];
        std::vector<float> slot_scales(n_slots_v, 0.0f);
        for (size_t i = 0; i < n_adapters; ++i) {
            llama_adapter_lora * adapter = adapters[i];
            const auto it = adapter->ab_map.find(name);
            if (it == adapter->ab_map.end()) {
                continue;
            }
            const llama_adapter_lora_weight & w = it->second;
            const int64_t r = w.a->ne[1];
            const int64_t slot = (int64_t) i + 1;
            slot_scales[slot] = w.get_scale(adapter->alpha, scales[i]);

            // A [n_in, r] is the prefix of the slot's [n_in, r_max] block
            if (!llama_adapter_rows_read_f32(w.a, src, err)) {
                return false;
            }
            ggml_backend_tensor_set(t.a, src.data(), slot * n_in * r_max * sizeof(float), src.size() * sizeof(float));

            // B [r, n_out]: each of the n_out rows is padded from r to r_max
            if (!llama_adapter_rows_read_f32(w.b, src, err)) {
                return false;
            }
            padded.assign(r_max * n_out, 0.0f);
            for (int64_t j = 0; j < n_out; ++j) {
                std::memcpy(padded.data() + j * r_max, src.data() + j * r, r * sizeof(float));
            }
            ggml_backend_tensor_set(t.b, padded.data(), slot * r_max * n_out * sizeof(float), padded.size() * sizeof(float));
        }
        ggml_backend_tensor_set(t.scale, slot_scales.data(), 0, slot_scales.size() * sizeof(float));
    }

    LLAMA_LOG_INFO("%s: %zu routed LoRA adapters, %zu targets, max rank %lld, %.2f MiB stacked (%.2f MiB unpadded)\n",
            __func__, n_adapters, targets.size(), (long long) r_max,
            n_bytes / 1024.0 / 1024.0, n_bytes_unpadded / 1024.0 / 1024.0);

    return true;
}

//
// graph
//

#include "llama-batch.h"
#include "llama-context.h"
#include "llama-graph.h"

// The routes of the ubatch rows, read from the context's table at each decode:
// the topology depends on the routed set only, so a reused graph just refills these.
class llm_graph_input_lora_rows : public llm_graph_input_i {
public:
    llm_graph_input_lora_rows(
            const llama_adapter_lora_rows * rows,
            const std::vector<int32_t> * routes,
            int64_t n_tokens,
            int64_t n_outputs) : rows(rows), routes(routes), n_tokens(n_tokens), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_lora_rows() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        return params.lora_rows == rows && params.lora_routes == routes &&
            (int64_t) params.ubatch.n_tokens == n_tokens && (int64_t) params.n_outputs == n_outputs;
    }

    ggml_tensor * ids_tok  = nullptr; // I32 [1, n_tokens]
    ggml_tensor * mask_tok = nullptr; // F32 [1, n_tokens]
    ggml_tensor * ids_out  = nullptr; // I32 [1, n_outputs]
    ggml_tensor * mask_out = nullptr; // F32 [1, n_outputs]

    const llama_adapter_lora_rows * rows;
    const std::vector<int32_t> * routes;
    const int64_t n_tokens;
    const int64_t n_outputs;
};

void llm_graph_input_lora_rows::set_input(const llama_ubatch * ubatch) {
    const int64_t n = ubatch->n_tokens;
    std::vector<int32_t> route(n, 0);
    for (int64_t i = 0; i < n; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];
        if (seq_id >= 0 && (size_t) seq_id < routes->size()) {
            route[i] = (*routes)[seq_id];
        }
    }

    // the output rows, in the order build_inp_out_ids gathers them
    std::vector<int64_t> out_rows;
    if (ids_out == nullptr && mask_out == nullptr) {
        // no output-row operand in this graph
    } else if (n_outputs == n) {
        out_rows.resize(n);
        for (int64_t i = 0; i < n; ++i) {
            out_rows[i] = i;
        }
    } else {
        GGML_ASSERT(ubatch->output);
        for (int64_t i = 0; i < n; ++i) {
            if (ubatch->output[i]) {
                out_rows.push_back(i);
            }
        }
    }

    auto fill = [&](ggml_tensor * ids, ggml_tensor * mask, const std::vector<int64_t> * select) {
        const int64_t count = select ? (int64_t) select->size() : n;
        std::vector<int32_t> id(count);
        std::vector<float>   on(count);
        for (int64_t j = 0; j < count; ++j) {
            const int32_t r = route[select ? (*select)[j] : j];
            id[j] = r > 0 ? r : 0;
            on[j] = r == 0 ? 1.0f : 0.0f;
        }
        if (ids) {
            GGML_ASSERT(ggml_nelements(ids) == count);
            ggml_backend_tensor_set(ids, id.data(), 0, ggml_nbytes(ids));
        }
        if (mask) {
            GGML_ASSERT(ggml_nelements(mask) == count);
            ggml_backend_tensor_set(mask, on.data(), 0, ggml_nbytes(mask));
        }
    };
    fill(ids_tok, mask_tok, nullptr);
    fill(ids_out, mask_out, &out_rows);
}

static llm_graph_input_lora_rows * llama_lora_rows_input(const llm_graph_context & g) {
    if (g.inp_lora_rows == nullptr) {
        auto inp = std::make_unique<llm_graph_input_lora_rows>(g.lora_rows, g.lora_routes, g.n_tokens, g.n_outputs);
        g.inp_lora_rows = inp.get();
        g.res->add_input(std::move(inp));
    }
    return g.inp_lora_rows;
}

// whether `n_rows` rows are the ubatch tokens (true) or its outputs (false)
static bool llama_lora_rows_are_tokens(const llm_graph_context & g, int64_t n_rows) {
    if (n_rows == g.n_tokens) {
        return true;
    }
    if (n_rows == g.n_outputs) {
        return false;
    }
    GGML_ABORT("per-row LoRA: a LoRA target is multiplied over %lld rows, neither the %lld ubatch tokens nor its %lld outputs",
            (long long) n_rows, (long long) g.n_tokens, (long long) g.n_outputs);
}

ggml_tensor * llm_graph_context::build_lora_rows_ids(int64_t n_rows) const {
    llm_graph_input_lora_rows * inp = llama_lora_rows_input(*this);
    ggml_tensor ** slot = llama_lora_rows_are_tokens(*this, n_rows) ? &inp->ids_tok : &inp->ids_out;
    if (*slot == nullptr) {
        *slot = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_rows);
        ggml_set_input(*slot);
        ggml_set_name(*slot, "lora_rows_ids");
    }
    return *slot;
}

ggml_tensor * llm_graph_context::build_lora_rows_mask(int64_t n_rows) const {
    llm_graph_input_lora_rows * inp = llama_lora_rows_input(*this);
    ggml_tensor ** slot = llama_lora_rows_are_tokens(*this, n_rows) ? &inp->mask_tok : &inp->mask_out;
    if (*slot == nullptr) {
        *slot = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_rows);
        ggml_set_input(*slot);
        ggml_set_name(*slot, "lora_rows_mask");
    }
    return *slot;
}

ggml_tensor * llm_graph_context::build_lora_rows_delta(
        ggml_tensor * w,
        ggml_tensor * cur,
        ggml_tensor * like) const {
    const llama_adapter_lora_rows::target * t = lora_rows->get(w);
    if (t == nullptr) {
        return nullptr;
    }
    const int64_t n_rows = ggml_nrows(cur);

    ggml_tensor * x = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx0, cur);
    if (x->type != GGML_TYPE_F32) {
        x = ggml_cast(ctx0, x, GGML_TYPE_F32);
    }
    x = ggml_reshape_3d(ctx0, x, x->ne[0], 1, n_rows);

    ggml_tensor * ids = build_lora_rows_ids(n_rows);

    // shrink, scaled per row by its adapter's alpha/rank and user scale, then expand
    ggml_tensor * ax = ggml_mul_mat_id(ctx0, t->a, x, ids);                      // [r_max, 1, n_rows]
    ggml_tensor * s  = ggml_get_rows(ctx0, t->scale, ggml_reshape_1d(ctx0, ids, n_rows)); // [1, n_rows]
    ax = ggml_mul(ctx0, ax, ggml_reshape_3d(ctx0, s, 1, 1, n_rows));
    ggml_tensor * bax = ggml_mul_mat_id(ctx0, t->b, ax, ids);                    // [n_out, 1, n_rows]

    return ggml_reshape_4d(ctx0, bax, like->ne[0], like->ne[1], like->ne[2], like->ne[3]);
}

//
// context
//

bool llama_context::set_adapters_lora_rows(
        llama_adapter_lora * const * adapters,
        size_t n_adapters,
        const float * scales,
        std::string & err) {
    lora_rows.reset();
    lora_routes.clear();
    sched_need_reserve = true;

    auto rows = std::make_unique<llama_adapter_lora_rows>();
    if (!rows->init(model, adapters, n_adapters, scales, err)) {
        return false;
    }
    lora_rows = std::move(rows);
    lora_routes.assign(cparams.n_seq_max, 0);
    return true;
}

void llama_context::clear_adapters_lora_rows() {
    if (lora_rows) {
        lora_rows.reset();
        lora_routes.clear();
        sched_need_reserve = true;
    }
}

bool llama_context::set_seq_lora_route(llama_seq_id seq_id, int32_t route) {
    if (!lora_rows || seq_id < 0 || (size_t) seq_id >= lora_routes.size()
            || route < -1 || route >= lora_rows->n_slots()) {
        return false;
    }
    lora_routes[seq_id] = route;
    return true;
}

void llama_context::clear_seq_lora_routes() {
    std::fill(lora_routes.begin(), lora_routes.end(), 0);
}

int32_t llama_set_adapters_lora_rows(
        llama_context * ctx,
        llama_adapter_lora * const * adapters,
        size_t n_adapters,
        const float * scales,
        std::string * err) {
    std::string message;
    if (n_adapters > 0 && (adapters == nullptr || scales == nullptr)) {
        message = "routed LoRA adapters and scales are required";
    } else if (ctx->set_adapters_lora_rows(adapters, n_adapters, scales, message)) {
        return 0;
    }
    LLAMA_LOG_ERROR("%s: %s\n", __func__, message.c_str());
    if (err) {
        *err = message;
    }
    return -1;
}

void llama_clear_adapters_lora_rows(llama_context * ctx) {
    ctx->clear_adapters_lora_rows();
}

int32_t llama_set_seq_lora_route(llama_context * ctx, llama_seq_id seq_id, int32_t route) {
    return ctx->set_seq_lora_route(seq_id, route) ? 0 : -1;
}

void llama_clear_seq_lora_routes(llama_context * ctx) {
    ctx->clear_seq_lora_routes();
}

bool llama_get_adapters_lora_rows_info(const llama_context * ctx, llama_adapter_lora_rows_info * out) {
    const llama_adapter_lora_rows * rows = ctx->get_adapters_lora_rows();
    if (rows == nullptr || out == nullptr) {
        return false;
    }
    out->n_adapters       = (int32_t) rows->adapters.size();
    out->r_max            = (int32_t) rows->r_max;
    out->n_bytes          = rows->n_bytes;
    out->n_bytes_unpadded = rows->n_bytes_unpadded;
    return true;
}
