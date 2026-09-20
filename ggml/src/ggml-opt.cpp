#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct ggml_opt_dataset {
    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    struct ggml_tensor    * data   = nullptr;
    struct ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    // Optional second caller-owned segment. `data`/`labels` describe the
    // logical combined shape while shard reads switch backing pointers at
    // split_shard. Internal datasets and single external views leave these
    // fields unset.
    void *  data_second   = nullptr;
    void *  labels_second = nullptr;
    int64_t split_shard   = -1;

    std::vector<int64_t> permutation;
};

// retro delta: one live persistent slot. The owner is a parameter's name in
// the per-parameter scope and an optimizer's name in the shared one, so both
// tables are read through one shape and a checkpoint that round-trips one
// round-trips the other.
struct ggml_opt_slot {
    std::string             owner;
    std::string             name;
    struct ggml_tensor    * tensor = nullptr;
    enum ggml_opt_slot_init init   = GGML_OPT_SLOT_INIT_ZERO;
    uint8_t                 code   = 0;
};

// AdamW's two F32 moments, zero-initialized. Written as a table rather than
// as two fields so that the allocator, the initializer and the update graph
// read one declaration; nothing below knows the number two.
static const struct ggml_opt_slot_def ggml_opt_slots_adamw[] = {
    { "m", GGML_TYPE_F32, GGML_OPT_SLOT_SHAPE_PARAMETER, 0, GGML_OPT_SLOT_INIT_ZERO, 0 },
    { "v", GGML_TYPE_F32, GGML_OPT_SLOT_SHAPE_PARAMETER, 0, GGML_OPT_SLOT_INIT_ZERO, 0 },
};

const struct ggml_opt_slot_def * ggml_opt_optimizer_slots(
        enum ggml_opt_optimizer_type optimizer, int64_t * n_slots) {
    switch (optimizer) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            if (n_slots) {
                *n_slots = (int64_t) (sizeof(ggml_opt_slots_adamw)/sizeof(ggml_opt_slots_adamw[0]));
            }
            return ggml_opt_slots_adamw;
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            // Not "not initialized": SGD keeps no per-parameter state and still
            // has an iteration counter and a schedule.
            if (n_slots) {
                *n_slots = 0;
            }
            return nullptr;
        default:
            GGML_ABORT("unknown optimizer");
    }
}

const struct ggml_opt_slot_def * ggml_opt_optimizer_shared_slots(
        enum ggml_opt_optimizer_type optimizer, int64_t * n_slots) {
    switch (optimizer) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            if (n_slots) {
                *n_slots = 0;
            }
            return nullptr;
        default:
            GGML_ABORT("unknown optimizer");
    }
}

int64_t ggml_opt_optimizer_n_params(enum ggml_opt_optimizer_type optimizer) {
    switch (optimizer) {
        // alpha, beta1, beta2, eps, wd, beta1h, beta2h, and the per-step seed
        // the stochastic rounding of an F16 parameter needs.
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW: return 8;
        case GGML_OPT_OPTIMIZER_TYPE_SGD:   return 2;
        default: GGML_ABORT("unknown optimizer");
    }
}

// The widest shared table any optimizer declares, so the static context can be
// sized before the run's owners are known.
#define GGML_OPT_MAX_SHARED_SLOTS 4

struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    enum ggml_opt_loss_type    loss_type;
    enum ggml_opt_build_type   build_type;
    enum ggml_opt_build_type   build_type_alloc;

    struct ggml_tensor * inputs  = nullptr;
    struct ggml_tensor * outputs = nullptr;
    struct ggml_tensor * labels  = nullptr;

    struct ggml_tensor * loss     = nullptr;
    struct ggml_tensor * pred     = nullptr;
    struct ggml_tensor * ncorrect = nullptr;

    struct ggml_cgraph * gf      = nullptr;
    struct ggml_cgraph * gb_grad = nullptr;
    struct ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool cache_dynamic_graph     = false;
    bool eval_ready              = false;
    std::vector<struct ggml_tensor *> grad_accs;
    // Persistent accumulators must follow parameter identity, because dynamic
    // graphs can assign different node indices to the same parameter.
    std::map<std::string, struct ggml_tensor *> grad_acc_by_name;
    struct ggml_tensor * loss_acc = nullptr;
    // Forward tensors retained across the backward pass. Empty preserves the
    // ordinary graph exactly; dynamic callers refresh this list per build.
    std::vector<struct ggml_tensor *> gradient_checkpoints;
    // retro delta: type the checkpoints are held in across the backward.
    // GGML_TYPE_COUNT keeps them as built, which is the bit-exact default.
    enum ggml_type                    gradient_checkpoint_type = GGML_TYPE_COUNT;
    // retro delta: the profile of the last backward graph that actually retained
    // checkpoints. Stamped at build time rather than computed on demand, because
    // the list above is cleared by every ggml_opt_prepare_alloc — a forward-only
    // build between a training step and the read would otherwise erase the
    // measurement rather than report the last one.
    struct ggml_opt_checkpoint_profile checkpoint_profile = {};
    // retro delta: the live persistent-state table, allocated from the
    // optimizer descriptor. Keyed by parameter name, because a dynamic graph
    // is rebuilt on every ggml_opt_alloc and a node position is only
    // incidentally stable across builds. Checkpointing hands these same
    // tensors to the caller under the same key, so the update step and the
    // restore agree by construction rather than by coincidence.
    std::vector<ggml_opt_slot>                          slots;
    // parameter name -> [begin, end) into `slots`, its owner's whole table.
    std::map<std::string, std::pair<size_t, size_t>>    slots_by_param;
    // parameter name -> the optimizer that owns it. A run with one optimizer
    // records the same value for every parameter; a mixed run is what makes
    // this a table rather than a field.
    std::map<std::string, enum ggml_opt_optimizer_type> param_optimizer;
    // State one optimizer keeps once rather than once per parameter.
    std::vector<ggml_opt_slot>                          shared_slots;
    // Which optimizers this run actually builds a step for, in enumeration
    // order. Filling the hyperparameters of an optimizer nothing uses would
    // validate values no update ever reads.
    bool optimizer_used[GGML_OPT_OPTIMIZER_TYPE_COUNT] = {};
    // CE nodes are stamped once per ubatch. Cache their addresses across
    // repeated calls and rebuild the cache only when ggml_opt_build replaces
    // a dynamic graph.
    std::vector<struct ggml_tensor *> loss_active_row_nodes;
    uint64_t graph_generation = 0;
    uint64_t loss_nodes_generation = UINT64_MAX;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    bool    loss_per_datapoint = false;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    void *                        get_opt_pars_ud = nullptr;
    // retro delta: one hyperparameter tensor per optimizer, because a mixed
    // run reads two update kernels' worth of coefficients in one step, and
    // the shared gradient-clipping scale beside them: the norm is one norm
    // over every trainable gradient whatever owns each parameter.
    struct ggml_tensor * opt_step_params[GGML_OPT_OPTIMIZER_TYPE_COUNT] = {};
    struct ggml_tensor * opt_max_grad_norm = nullptr;

    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    ggml_opt_get_param_optimizer get_param_optimizer    = nullptr;
    void *                       get_param_optimizer_ud = nullptr;
};

// The optimizer that owns one parameter: the run's, unless the caller answers
// otherwise. Asked before anything is allocated, because a parameter's slot
// table is its owner's and not the run's.
static enum ggml_opt_optimizer_type ggml_opt_owner_of(
        ggml_opt_context_t opt_ctx, const struct ggml_tensor * param) {
    if (!opt_ctx->get_param_optimizer) {
        return opt_ctx->optimizer;
    }
    const enum ggml_opt_optimizer_type owner =
            opt_ctx->get_param_optimizer(param, opt_ctx->get_param_optimizer_ud);
    GGML_ASSERT(owner >= 0 && owner < GGML_OPT_OPTIMIZER_TYPE_COUNT &&
            "parameter assignment named an optimizer this build does not have");
    return owner;
}

// One slot of one owner, allocated in the static context from its definition.
static ggml_opt_slot ggml_opt_slot_alloc(
        struct ggml_context            * ctx,
        const struct ggml_opt_slot_def & def,
        const struct ggml_tensor       * param,
        const char                     * owner,
        const char                     * optimizer_name) {
    struct ggml_tensor * tensor = nullptr;
    switch (def.shape) {
        case GGML_OPT_SLOT_SHAPE_PARAMETER:
            GGML_ASSERT(param);
            tensor = ggml_new_tensor(ctx, def.type, GGML_MAX_DIMS, param->ne);
            break;
        case GGML_OPT_SLOT_SHAPE_BLOCKS:
            GGML_ASSERT(param);
            tensor = ggml_new_tensor_1d(
                    ctx, def.type, ggml_opt_slot_n_elements(&def, ggml_nelements(param)));
            break;
        case GGML_OPT_SLOT_SHAPE_FIXED:
            GGML_ASSERT(def.n_block > 0);
            tensor = ggml_new_tensor_1d(ctx, def.type, ggml_opt_slot_n_elements(&def, 0));
            break;
        default:
            GGML_ABORT("unknown optimizer slot shape");
    }
    ggml_format_name(tensor, "%s %s for %s", optimizer_name, def.name, owner);

    ggml_opt_slot slot;
    slot.owner  = owner;
    slot.name   = def.name;
    slot.tensor = tensor;
    slot.init   = def.init;
    slot.code   = def.code;
    return slot;
}

int64_t ggml_opt_slot_n_elements(const struct ggml_opt_slot_def * def, int64_t n_param_elements) {
    if (!def) {
        return 0;
    }
    switch (def->shape) {
        case GGML_OPT_SLOT_SHAPE_PARAMETER:
            return n_param_elements;
        case GGML_OPT_SLOT_SHAPE_BLOCKS: {
            const int64_t block = def->n_block > 0 ? def->n_block : 1;
            // A partial trailing block still costs an element.
            // retro delta: avoid overflowing the rounded-up numerator.
            return n_param_elements / block + (n_param_elements % block != 0);
        }
        case GGML_OPT_SLOT_SHAPE_FIXED:
            return def->n_block;
        default:
            GGML_ABORT("unknown optimizer slot shape");
    }
}

bool ggml_opt_slot_initial_bytes(
        const struct ggml_opt_slot_def * def, int64_t n_elements, void * out, size_t n_bytes) {
    if (!def || !out || n_elements < 0) {
        return false;
    }
    const size_t element = ggml_type_size(def->type);
    // retro delta: validate before multiplying; wrapped sizes must not admit
    // a small buffer for a large codebook.
    if ((uint64_t) n_elements > SIZE_MAX / element ||
            n_bytes != element * (size_t) n_elements) {
        return false;
    }
    switch (def->init) {
        case GGML_OPT_SLOT_INIT_ZERO:
            std::memset(out, 0, n_bytes);
            return true;
        case GGML_OPT_SLOT_INIT_CODE:
            // A byte pattern, not a value: the slot's dtype decides what the
            // code means and the initializer only has to be canonical.
            std::memset(out, def->code, n_bytes);
            return true;
        case GGML_OPT_SLOT_INIT_UNIFORM_CODEBOOK: {
            if (def->type != GGML_TYPE_F32) {
                return false;
            }
            // retro delta: callers provide bytes, with no float alignment guarantee.
            uint8_t * values = (uint8_t *) out;
            const float denominator = n_elements > 1 ? (float) (n_elements - 1) : 1.0f;
            for (int64_t k = 0; k < n_elements; ++k) {
                const float value = -1.0f + 2.0f*((float) k)/denominator;
                std::memcpy(values + (size_t) k * sizeof(float), &value, sizeof(value));
            }
            return true;
        }
        default:
            return false;
    }
}

// retro delta: the initializer. A slot allocated from a definition is filled
// according to that definition before the first update, whatever its shape and
// whatever its dtype -- zeroing a byte-index slot would be a code the codebook
// does not reserve, and a codebook is generated rather than filled.
//
// The bytes come from ggml_opt_slot_initial_bytes so that what a live slot
// holds and what the declaration says it holds cannot drift apart: there is one
// function and this is its only other caller.
static void ggml_opt_fill_slot(const ggml_opt_slot & slot) {
    struct ggml_tensor * tensor = slot.tensor;
    if (!tensor) {
        return;
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (slot.init == GGML_OPT_SLOT_INIT_ZERO) {
        // The common case, and the one a backend can do without a host buffer.
        ggml_set_zero(tensor);
        return;
    }
    const struct ggml_opt_slot_def def = {
        slot.name.c_str(), tensor->type, GGML_OPT_SLOT_SHAPE_PARAMETER, 0, slot.init, slot.code,
    };
    std::vector<uint8_t> bytes(nbytes);
    GGML_ASSERT(ggml_opt_slot_initial_bytes(&def, ggml_nelements(tensor), bytes.data(), nbytes) &&
            "a slot declares an initializer its dtype cannot hold");
    ggml_backend_tensor_set(tensor, bytes.data(), 0, nbytes);
}

static void ggml_opt_fill_slots(ggml_opt_context_t opt_ctx) {
    for (const ggml_opt_slot & slot : opt_ctx->slots) {
        ggml_opt_fill_slot(slot);
    }
    for (const ggml_opt_slot & slot : opt_ctx->shared_slots) {
        ggml_opt_fill_slot(slot);
    }
}

// retro delta: one optimizer's update, as a graph.
//
// The one place an optimizer's arithmetic is written. Its inputs are the
// parameter, its gradient, the slots its own table declared for it and the
// arguments its own kernel reads; the caller roots the result in the
// executable graph, so a state update this function builds but does not return
// would be constructed and never run.
static struct ggml_tensor * ggml_opt_build_step(
        ggml_opt_context_t             opt_ctx,
        enum ggml_opt_optimizer_type   optimizer,
        struct ggml_tensor           * param,
        struct ggml_tensor           * grad,
        const ggml_opt_slot          * slots,
        size_t                         n_slots,
        struct ggml_tensor           * step_args,
        struct ggml_tensor           * grad_scale) {
    struct ggml_context * ctx = opt_ctx->ctx_compute;
    switch (optimizer) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
            GGML_ASSERT(n_slots == 2 && "adamw declares two slots per parameter");
            struct ggml_tensor * m = slots[0].tensor;
            struct ggml_tensor * v = slots[1].tensor;
            GGML_ASSERT(ggml_are_same_shape(m, param) && ggml_are_same_shape(v, param) &&
                    "optimizer parameter changed shape between graph builds");
            return ggml_opt_step_adamw(ctx, param, grad, m, v, step_args);
        }
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            GGML_ASSERT(n_slots == 0 && "sgd keeps no per-parameter state");
            GGML_UNUSED(slots);
            return ggml_opt_step_sgd(ctx, param, ggml_mul(ctx, grad, grad_scale), step_args);
        default:
            GGML_ABORT("unknown optimizer");
    }
}

// retro delta: defined below, next to the rest of the checkpoint accounting;
// declared here because ggml_opt_build (above it) stamps the profile.
static bool ggml_opt_measure_checkpoint_profile(
        ggml_opt_context_t   opt_ctx,
        struct ggml_cgraph * graph,
        struct ggml_opt_checkpoint_profile * out_profile);

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

ggml_opt_dataset_t ggml_opt_dataset_init_external(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard,
        void *         data,
        void *         labels) {
    GGML_ASSERT(ne_datapoint > 0 && ndata > 0 && ndata_shard > 0);
    GGML_ASSERT(data);
    GGML_ASSERT((ne_label == 0) == (labels == nullptr));
    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    result->ctx = ggml_init(params);
    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->data->data = data;
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;
    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->labels->data = labels;
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    }
    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

ggml_opt_dataset_t ggml_opt_dataset_init_external_split(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata_first,
        int64_t        ndata_second,
        int64_t        ndata_shard,
        void *         data_first,
        void *         labels_first,
        void *         data_second,
        void *         labels_second) {
    GGML_ASSERT(ne_datapoint > 0 && ndata_first > 0 && ndata_second > 0 && ndata_shard > 0);
    GGML_ASSERT(ndata_first <= INT64_MAX - ndata_second);
    GGML_ASSERT(ndata_first % ndata_shard == 0);
    GGML_ASSERT(ndata_second % ndata_shard == 0);
    GGML_ASSERT(data_first && data_second);
    GGML_ASSERT((ne_label == 0) == (labels_first == nullptr));
    GGML_ASSERT((ne_label == 0) == (labels_second == nullptr));

    ggml_opt_dataset_t result = ggml_opt_dataset_init_external(
            type_data, type_label, ne_datapoint, ne_label,
            ndata_first + ndata_second, ndata_shard,
            data_first, labels_first);
    result->data_second   = data_second;
    result->labels_second = labels_second;
    result->split_shard   = ndata_first / ndata_shard;
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    if (dataset->buf) {
        ggml_backend_buffer_free(dataset->buf);
    }
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

static const char * ggml_opt_dataset_shard_ptr(
        ggml_opt_dataset_t dataset,
        bool               labels,
        int64_t            ishard) {
    const size_t nbs = labels ? dataset->nbs_labels : dataset->nbs_data;
    const void * first = labels ? dataset->labels->data : dataset->data->data;
    const void * second = labels ? dataset->labels_second : dataset->data_second;
    if (second && ishard >= dataset->split_shard) {
        return (const char *) second + (ishard - dataset->split_shard)*nbs;
    }
    return (const char *) first + ishard*nbs;
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = ggml_opt_dataset_shard_ptr(dataset, false, ishard);
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = ggml_opt_dataset_shard_ptr(dataset, true, ishard);
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data       = ggml_opt_dataset_shard_ptr(dataset, false, ishard);
        char       * ptr_data_batch = (char       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels       = ggml_opt_dataset_shard_ptr(dataset, true, ishard);
        char       * ptr_labels_batch = (char       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    // Generic ggml callers keep the historical unclipped behavior unless they
    // opt in. Retrograd overrides this with its validated default of 1.0.
    result.max_grad_norm = INFINITY;
    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    return result;
}


struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct ggml_opt_optimizer_params *) userdata);
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        enum ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        /*get_param_optimizer    =*/ nullptr,
        /*get_param_optimizer_ud =*/ nullptr,
    };
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static void ggml_opt_copy_recompute_backend(
        ggml_tensor * original,
        ggml_tensor * clone,
        void * userdata) {
    ggml_backend_sched_t sched = static_cast<ggml_backend_sched_t>(userdata);
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, original);
    if (backend) {
        ggml_backend_sched_set_tensor_backend(sched, clone, backend);
    }
}

// Whether the backward pass sums into persistent accumulators rather than
// writing the gradients of a single evaluation. True whenever more than one
// evaluation contributes to a step, and true for every dynamic graph: the
// gradient tensors of a graph that is rebuilt per evaluation do not survive it,
// so the sum has to live in ctx_static either way.
//
// Read by ggml_opt_alloc as well as by the build, because "these accumulators
// hold a partial sum" is exactly the condition under which a new accumulation
// window has to start by clearing them.
static bool ggml_opt_grads_accumulate(const struct ggml_opt_context * opt_ctx) {
    return opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);
}

static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");
    ++opt_ctx->graph_generation;

    const bool accumulate = ggml_opt_grads_accumulate(opt_ctx);

    // retro delta: persistent state is allocated from the descriptor, so the
    // question is no longer "is this AdamW" but "is this the optimizer build".
    const bool need_slots = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT;

    ggml_set_input(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    int n_param = 0;
    // Counted from the tables of the optimizers that actually own parameters,
    // not from one optimizer's shape: a mixed run allocates a different number
    // of slots per parameter.
    int n_slot_tensors = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor * node = opt_ctx->gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
            if (need_slots) {
                int64_t n_defs = 0;
                ggml_opt_optimizer_slots(ggml_opt_owner_of(opt_ctx, node), &n_defs);
                n_slot_tensors += (int) n_defs;
            }
        }
        GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - the persistent slots every owning optimizer declares
        //   - the shared slots every owning optimizer declares
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = accumulate ? 1 : 0;
        const size_t tensors_shared = need_slots
                ? GGML_OPT_OPTIMIZER_TYPE_COUNT*GGML_OPT_MAX_SHARED_SLOTS : 0;
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + (size_t) n_slot_tensors
                + tensors_shared + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    struct ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    switch (opt_ctx->loss_type) {
        case GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = false;
            break;
        }
        case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            if (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_EXTERNAL: {
            // retro delta (plan 03): `outputs` is already the scalar loss (the
            // fused sparse cross-entropy averages over the active tokens itself).
            // Use it verbatim; autodiff still flows through it into the graph.
            GGML_ASSERT(ggml_is_scalar(opt_ctx->outputs) &&
                    "external loss must be a scalar output node");
            opt_ctx->labels = nullptr;
            opt_ctx->loss = opt_ctx->outputs;
            ggml_set_name(opt_ctx->loss, "loss_external");
            if (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_external_scaled");
            }
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_acc_by_name.empty() && !opt_ctx->loss_acc) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_acc_by_name[node->name] = opt_ctx->grad_accs[i];
                } else {
                    GGML_ASSERT(!opt_ctx->loss_acc);
                    opt_ctx->loss_acc = opt_ctx->grad_accs[i];
                }
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        // retro delta: allocate every owner's declared slots, in parameter
        // order and within a parameter in declaration order. Nothing here
        // knows what AdamW keeps: the tables do.
        if (need_slots) {
            std::vector<enum ggml_opt_optimizer_type> owners;
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (!(node->flags & GGML_TENSOR_FLAG_PARAM)) {
                    continue;
                }
                const enum ggml_opt_optimizer_type owner = ggml_opt_owner_of(opt_ctx, node);
                opt_ctx->param_optimizer[node->name] = owner;
                if (std::find(owners.begin(), owners.end(), owner) == owners.end()) {
                    owners.push_back(owner);
                }
                int64_t n_defs = 0;
                const struct ggml_opt_slot_def * defs = ggml_opt_optimizer_slots(owner, &n_defs);
                const size_t begin = opt_ctx->slots.size();
                for (int64_t slot = 0; slot < n_defs; ++slot) {
                    opt_ctx->slots.push_back(ggml_opt_slot_alloc(
                            opt_ctx->ctx_static, defs[slot], node, node->name,
                            ggml_opt_optimizer_name(owner)));
                }
                opt_ctx->slots_by_param[node->name] = { begin, opt_ctx->slots.size() };
            }
            // One shared row per owner that owns something, in first-seen
            // order. An owner with an empty shared table contributes none,
            // which is every optimizer this build can run.
            for (const enum ggml_opt_optimizer_type owner : owners) {
                int64_t n_defs = 0;
                const struct ggml_opt_slot_def * defs = ggml_opt_optimizer_shared_slots(owner, &n_defs);
                GGML_ASSERT(n_defs <= GGML_OPT_MAX_SHARED_SLOTS &&
                        "an optimizer declares more shared slots than the static context was sized for");
                for (int64_t slot = 0; slot < n_defs; ++slot) {
                    opt_ctx->shared_slots.push_back(ggml_opt_slot_alloc(
                            opt_ctx->ctx_static, defs[slot], nullptr,
                            ggml_opt_optimizer_name(owner), ggml_opt_optimizer_name(owner)));
                }
            }
        }
    } else {
        // Rebuild the graph-indexed view after a topology change (for example,
        // standard preflight followed by shared-prefix packed training).
        opt_ctx->grad_accs.assign(opt_ctx->gf->n_nodes, nullptr);
        for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if (accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
                const auto slot = opt_ctx->grad_acc_by_name.find(node->name);
                GGML_ASSERT(slot != opt_ctx->grad_acc_by_name.end() &&
                        "optimizer parameter has no gradient accumulator");
                GGML_ASSERT(ggml_are_same_shape(slot->second, node) &&
                        "optimizer parameter changed shape between graph builds");
                opt_ctx->grad_accs[i] = slot->second;
            } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(opt_ctx->loss_acc && ggml_are_same_shape(opt_ctx->loss_acc, node));
                opt_ctx->grad_accs[i] = opt_ctx->loss_acc;
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    if (opt_ctx->gradient_checkpoints.empty()) {
        opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
        ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());
    } else {
        opt_ctx->gb_grad = ggml_build_backward_gradient_checkpointing(
                opt_ctx->ctx_compute,
                opt_ctx->gf,
                opt_ctx->grad_accs.data(),
                opt_ctx->gradient_checkpoints.data(),
                (int) opt_ctx->gradient_checkpoints.size(),
                opt_ctx->gradient_checkpoint_type,
                ggml_opt_copy_recompute_backend,
                opt_ctx->backend_sched);
        // retro delta: measure what this build retains while the graph and its
        // checkpoint list are both still in hand. One
        // walk per backward build, which is the same cadence as the build itself.
        //
        // Measured on gb_grad rather than gb_opt: the optimizer step appended on
        // top reads gradients and momenta, never a checkpoint, so it would only
        // lengthen the graph the spans are fractions of and make every lifetime
        // look shorter than it is.
        struct ggml_opt_checkpoint_profile profile = {};
        if (ggml_opt_measure_checkpoint_profile(opt_ctx, opt_ctx->gb_grad, &profile)) {
            opt_ctx->checkpoint_profile = profile;
        }
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    if (!opt_ctx->ctx_cpu) {
        // retro delta: one hyperparameter tensor per optimizer, plus the
        // clipping ceiling they share. Every optimizer gets one whether or not
        // this run uses it: the tensors are a handful of floats, and a run that
        // allocated only the optimizers of its first build could not be
        // rebuilt with another one without reallocating the CPU context.
        const size_t size_meta = (GGML_OPT_OPTIMIZER_TYPE_COUNT + 1) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_cpu = ggml_init(params);
        for (int type = 0; type < GGML_OPT_OPTIMIZER_TYPE_COUNT; ++type) {
            const enum ggml_opt_optimizer_type optimizer_type = (enum ggml_opt_optimizer_type) type;
            struct ggml_tensor * step_params = ggml_new_tensor_1d(
                    opt_ctx->ctx_cpu, GGML_TYPE_F32, ggml_opt_optimizer_n_params(optimizer_type));
            ggml_set_input(step_params);
            ggml_format_name(step_params, "%s_params", ggml_opt_optimizer_name(optimizer_type));
            opt_ctx->opt_step_params[type] = step_params;
        }
        // The gradient norm is one norm over every trainable gradient, so the
        // ceiling it is compared against is one value whatever owns what.
        opt_ctx->opt_max_grad_norm = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, 1);
        ggml_set_input(opt_ctx->opt_max_grad_norm);
        ggml_set_name(opt_ctx->opt_max_grad_norm, "max_grad_norm");
        opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
    }
    ggml_tensor * max_grad_norm = opt_ctx->opt_max_grad_norm;

    // Compute one norm over every trainable parameter gradient, then use the
    // same scale for all tensors so clipping preserves the gradient direction.
    ggml_tensor * global_grad_norm_sq = nullptr;
    for (int i = opt_ctx->gf->n_nodes - 1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);
        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            ggml_tensor * tensor_norm_sq = ggml_sum(opt_ctx->ctx_compute,
                    ggml_sqr(opt_ctx->ctx_compute, grad));
            global_grad_norm_sq = global_grad_norm_sq
                    ? ggml_add(opt_ctx->ctx_compute, global_grad_norm_sq, tensor_norm_sq)
                    : tensor_norm_sq;
        }
    }
    GGML_ASSERT(global_grad_norm_sq != nullptr);
    ggml_tensor * global_grad_norm = ggml_clamp(
            opt_ctx->ctx_compute,
            ggml_sqrt(opt_ctx->ctx_compute, global_grad_norm_sq),
            1e-12f,
            INFINITY);
    ggml_tensor * grad_scale = ggml_clamp(
            opt_ctx->ctx_compute,
            ggml_div(opt_ctx->ctx_compute, max_grad_norm, global_grad_norm),
            0.0f,
            1.0f);
    // retro delta: the per-optimizer argument each update step reads. AdamW
    // folds the clipping scale into its own kernel, so the scale travels with
    // the hyperparameters instead of being multiplied into a full-size copy of
    // every gradient; the concat runs once per optimizer per graph and its
    // result is shared by every parameter that optimizer owns.
    struct ggml_tensor * step_args[GGML_OPT_OPTIMIZER_TYPE_COUNT] = {};
    for (int type = 0; type < GGML_OPT_OPTIMIZER_TYPE_COUNT; ++type) {
        struct ggml_tensor * declared = opt_ctx->opt_step_params[type];
        switch ((enum ggml_opt_optimizer_type) type) {
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                step_args[type] = ggml_concat(opt_ctx->ctx_compute, declared, grad_scale, 0);
                break;
            case GGML_OPT_OPTIMIZER_TYPE_SGD:
                // SGD keeps the explicit multiply: its kernel is untouched by
                // the F16 work and takes a scaled gradient instead.
                step_args[type] = declared;
                break;
            default:
                GGML_ABORT("unknown optimizer");
        }
    }

    for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            // The owner and its slots, both keyed by the parameter's name. The
            // assert turns "a parameter appeared after the state was
            // allocated" into a failure rather than a step against another
            // parameter's slots.
            const auto owner_row = opt_ctx->param_optimizer.find(node->name);
            GGML_ASSERT(owner_row != opt_ctx->param_optimizer.end() &&
                    "optimizer parameter has no slot table; it appeared after the state was allocated");
            const enum ggml_opt_optimizer_type owner = owner_row->second;
            const auto range = opt_ctx->slots_by_param.at(node->name);
            const size_t n_slots = range.second - range.first;
            // retro delta: an all-SGD run has no backing slot array.
            const ggml_opt_slot * slots = n_slots ? opt_ctx->slots.data() + range.first : nullptr;
            opt_ctx->optimizer_used[owner] = true;

            struct ggml_tensor * opt_step = ggml_opt_build_step(
                    opt_ctx, owner, node, grad, slots, n_slots, step_args[owner], grad_scale);
            ggml_format_name(opt_step, "%s step for %s", ggml_opt_optimizer_name(owner), node->name);
            // Every write the step performs is rooted here: a state update that
            // is constructed but not reachable from the graph's outputs is
            // constructed, not executed.
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
        // The declared initializer, after the buffers exist and before the
        // first step. Separate from ggml_graph_reset, which knows one
        // optimizer's node shape and cannot reach a slot that is neither of
        // AdamW's two.
        ggml_opt_fill_slots(opt_ctx);
    }
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->loss_type        = params.loss_type;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs          = params.outputs;
    result->opt_period       = params.opt_period;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;
    result->optimizer        = params.optimizer;
    result->get_param_optimizer    = params.get_param_optimizer;
    result->get_param_optimizer_ud = params.get_param_optimizer_ud;

    GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    if (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs);
        return result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    ggml_opt_build(result);

    return result;
}

void ggml_opt_set_backend_sched(ggml_opt_context_t opt_ctx, ggml_backend_sched_t backend_sched) {
    // The optimizer keeps its persistent tensors (e.g. AdamW moments) in buffers
    // tied to the backends, not to the scheduler object, so swapping in a new
    // scheduler that uses the same backends is safe. The dynamic-graph path
    // rebuilds and re-allocates the graph against this scheduler on the next
    // ggml_opt_alloc, so no cached allocation is reused across the swap.
    opt_ctx->backend_sched = backend_sched;
    opt_ctx->allocated_graph = nullptr;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        // retro delta: back to what the declaration says an empty slot holds.
        // ggml_graph_reset zeroes the two operands of an AdamW step node and
        // reaches no other slot, so the initializer is what makes "reset" mean
        // the same thing for every optimizer.
        ggml_opt_fill_slots(opt_ctx);
        opt_ctx->iter = 1;
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_set_period(ggml_opt_context_t opt_ctx, int32_t opt_period) {
    if (!opt_ctx || opt_period < 1 || opt_ctx->eval_ready || opt_ctx->opt_i != 0) {
        return false;
    }
    opt_ctx->opt_period = opt_period;
    return true;
}

bool ggml_opt_abort_accumulation(ggml_opt_context_t opt_ctx) {
    if (!opt_ctx || opt_ctx->eval_ready) {
        return false;
    }
    for (const auto & item : opt_ctx->grad_acc_by_name) {
        ggml_set_zero(item.second);
    }
    if (opt_ctx->loss_acc) {
        ggml_set_zero(opt_ctx->loss_acc);
    }
    opt_ctx->opt_i = 0;
    opt_ctx->opt_period = 1;
    return true;
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

void ggml_opt_set_loss_active_rows(ggml_opt_context_t opt_ctx, int32_t n_active_rows) {
    // Only stamp actual cross-entropy nodes. With opt_period > 1 the loss
    // tensor is a SCALE node whose op_params[0] holds the (float) scale
    // factor; overwriting it would corrupt both the reported loss and the
    // gradient seed. The graph loops below reach the underlying CE nodes.
    if (opt_ctx->loss_nodes_generation != opt_ctx->graph_generation) {
        opt_ctx->loss_active_row_nodes.clear();
        ggml_cgraph * graphs[3] = { opt_ctx->gf, opt_ctx->gb_grad, opt_ctx->gb_opt };
        for (ggml_cgraph * graph : graphs) {
            if (!graph) {
                continue;
            }
            for (int i = 0; i < graph->n_nodes; ++i) {
                ggml_tensor * node = graph->nodes[i];
                if ((node->op == GGML_OP_CROSS_ENTROPY_LOSS ||
                            node->op == GGML_OP_CROSS_ENTROPY_LOSS_BACK) &&
                        std::find(opt_ctx->loss_active_row_nodes.begin(),
                            opt_ctx->loss_active_row_nodes.end(), node) ==
                                opt_ctx->loss_active_row_nodes.end()) {
                    opt_ctx->loss_active_row_nodes.push_back(node);
                }
            }
        }
        opt_ctx->loss_nodes_generation = opt_ctx->graph_generation;
    }
    for (ggml_tensor * node : opt_ctx->loss_active_row_nodes) {
        node->op_params[0] = n_active_rows;
        node->op_params[1] = 1;
    }
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    return ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

struct ggml_tensor * ggml_opt_grad_acc_by_name(ggml_opt_context_t opt_ctx, const char * name) {
    if (!opt_ctx || !name) {
        return nullptr;
    }
    const auto slot = opt_ctx->grad_acc_by_name.find(name);
    return slot == opt_ctx->grad_acc_by_name.end() ? nullptr : slot->second;
}

// ====== Optimizer state access (retro delta) ======

int64_t ggml_opt_iter(ggml_opt_context_t opt_ctx) {
    return opt_ctx->iter;
}

void ggml_opt_set_iter(ggml_opt_context_t opt_ctx, int64_t iter) {
    opt_ctx->iter = iter;
}

int64_t ggml_opt_slot_count(ggml_opt_context_t opt_ctx) {
    return (int64_t) opt_ctx->slots.size();
}

const char * ggml_opt_slot_owner(ggml_opt_context_t opt_ctx, int64_t index) {
    if (index < 0 || index >= (int64_t) opt_ctx->slots.size()) {
        return nullptr;
    }
    return opt_ctx->slots[(size_t) index].owner.c_str();
}

const char * ggml_opt_slot_name(ggml_opt_context_t opt_ctx, int64_t index) {
    if (index < 0 || index >= (int64_t) opt_ctx->slots.size()) {
        return nullptr;
    }
    return opt_ctx->slots[(size_t) index].name.c_str();
}

struct ggml_tensor * ggml_opt_slot_tensor(ggml_opt_context_t opt_ctx, int64_t index) {
    if (index < 0 || index >= (int64_t) opt_ctx->slots.size()) {
        return nullptr;
    }
    return opt_ctx->slots[(size_t) index].tensor;
}

int64_t ggml_opt_shared_slot_count(ggml_opt_context_t opt_ctx) {
    return (int64_t) opt_ctx->shared_slots.size();
}

const char * ggml_opt_shared_slot_owner(ggml_opt_context_t opt_ctx, int64_t index) {
    if (index < 0 || index >= (int64_t) opt_ctx->shared_slots.size()) {
        return nullptr;
    }
    return opt_ctx->shared_slots[(size_t) index].owner.c_str();
}

const char * ggml_opt_shared_slot_name(ggml_opt_context_t opt_ctx, int64_t index) {
    if (index < 0 || index >= (int64_t) opt_ctx->shared_slots.size()) {
        return nullptr;
    }
    return opt_ctx->shared_slots[(size_t) index].name.c_str();
}

struct ggml_tensor * ggml_opt_shared_slot_tensor(ggml_opt_context_t opt_ctx, int64_t index) {
    if (index < 0 || index >= (int64_t) opt_ctx->shared_slots.size()) {
        return nullptr;
    }
    return opt_ctx->shared_slots[(size_t) index].tensor;
}

enum ggml_opt_optimizer_type ggml_opt_param_optimizer(
        ggml_opt_context_t opt_ctx, const char * name, bool * found) {
    const auto row = name ? opt_ctx->param_optimizer.find(name) : opt_ctx->param_optimizer.end();
    if (row == opt_ctx->param_optimizer.end()) {
        if (found) {
            *found = false;
        }
        return opt_ctx->optimizer;
    }
    if (found) {
        *found = true;
    }
    return row->second;
}

size_t ggml_opt_rng_state(ggml_opt_context_t opt_ctx, char * buffer, size_t n_buffer) {
    std::ostringstream stream;
    stream << opt_ctx->rng;
    const std::string state = stream.str();
    if (buffer && n_buffer > state.size()) {
        std::memcpy(buffer, state.c_str(), state.size() + 1);
    }
    return state.size();
}

bool ggml_opt_set_rng_state(ggml_opt_context_t opt_ctx, const char * state) {
    if (!state) {
        return false;
    }
    std::istringstream stream(state);
    std::mt19937 rng;
    stream >> rng;
    if (stream.fail()) {
        return false;
    }
    opt_ctx->rng = rng;
    return true;
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
    opt_ctx->gradient_checkpoints.clear();
}

void ggml_opt_set_gradient_checkpoints(
        ggml_opt_context_t    opt_ctx,
        struct ggml_tensor ** checkpoints,
        int                   n_checkpoints) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    GGML_ASSERT(n_checkpoints >= 0);
    GGML_ASSERT(n_checkpoints == 0 || checkpoints != nullptr);
    opt_ctx->gradient_checkpoints.clear();
    if (n_checkpoints > 0) {
        opt_ctx->gradient_checkpoints.assign(checkpoints, checkpoints + n_checkpoints);
    }
}

// retro delta
void ggml_opt_set_gradient_checkpoint_type(
        ggml_opt_context_t opt_ctx,
        enum ggml_type     type) {
    // Only a plain float narrowing is meaningful: the casts are ggml_cast, and a
    // quantized checkpoint would round far more coarsely than the activations it
    // stands in for.
    GGML_ASSERT(type == GGML_TYPE_COUNT || type == GGML_TYPE_F32 ||
                type == GGML_TYPE_F16   || type == GGML_TYPE_BF16);
    opt_ctx->gradient_checkpoint_type = type;
}

// retro delta: see ggml_opt_checkpoint_profile.
//
// A checkpoint is not one tensor once it is narrowed: the forward builds it in
// F32, a `store` cast holds it 16-bit across the backward, and a `fetch` cast
// widens it again for the first recompute that reads it. Accounting only the
// F32 tensor would report double the bytes actually retained, and accounting
// only the store would lose the forward-side hold. So the walk follows the
// chain: a node is part of a checkpoint's chain if it *is* the checkpoint, or
// it is a copy whose source is already in the chain. That is exactly the two
// casts and nothing else — a recompute clone reads the fetch, it is not a copy
// of it.
//
// Each chain member contributes its own interval [produced, last read] and its
// own byte count, so the concurrency sweep below is over what is really held at
// each point rather than over an idealized "the checkpoint exists".
static bool ggml_opt_checkpoint_chain_member(
        const struct ggml_tensor * node,
        const std::set<const struct ggml_tensor *> & chain) {
    if (node->op != GGML_OP_CPY && node->op != GGML_OP_CONT && node->op != GGML_OP_DUP) {
        return false;
    }
    return node->src[0] && chain.count(node->src[0]) > 0;
}

static bool ggml_opt_measure_checkpoint_profile(
        ggml_opt_context_t   opt_ctx,
        struct ggml_cgraph * graph,
        struct ggml_opt_checkpoint_profile * out_profile) {
    GGML_ASSERT(out_profile);
    memset(out_profile, 0, sizeof(*out_profile));
    if (opt_ctx->gradient_checkpoints.empty() || !graph) {
        return false;
    }
    const int n_nodes = ggml_graph_n_nodes(graph);
    if (n_nodes <= 0) {
        return false;
    }

    // Position of every node, so a source can be located without rescanning.
    std::map<const struct ggml_tensor *, int> position;
    for (int i = 0; i < n_nodes; ++i) {
        position.emplace(ggml_graph_node(graph, i), i);
    }

    // One interval per retained tensor: (first node index, last node index, bytes).
    struct held_interval { int first; int last; size_t bytes; };
    std::vector<held_interval> intervals;
    // Per checkpoint, the span of its whole chain and the bytes of the member
    // held longest — the copy an offload would actually move.
    std::vector<int>    chain_first(opt_ctx->gradient_checkpoints.size(), n_nodes);
    std::vector<int>    chain_last(opt_ctx->gradient_checkpoints.size(), 0);
    std::vector<size_t> chain_bytes(opt_ctx->gradient_checkpoints.size(), 0);

    for (size_t c = 0; c < opt_ctx->gradient_checkpoints.size(); ++c) {
        struct ggml_tensor * checkpoint = opt_ctx->gradient_checkpoints[c];
        const auto found = position.find(checkpoint);
        if (found == position.end()) {
            // A checkpoint absent from this graph retains nothing in it. Skipped
            // rather than counted as zero, so n_checkpoints below stays the count
            // of what was measured.
            continue;
        }
        std::set<const struct ggml_tensor *> chain;
        chain.insert(checkpoint);
        std::map<const struct ggml_tensor *, held_interval> members;
        members.emplace(checkpoint, held_interval{ found->second, found->second, ggml_nbytes(checkpoint) });
        // One forward sweep: the chain only grows forward (a copy comes after
        // what it copies), so members are discovered before they are read.
        for (int i = found->second + 1; i < n_nodes; ++i) {
            struct ggml_tensor * node = ggml_graph_node(graph, i);
            if (ggml_opt_checkpoint_chain_member(node, chain)) {
                chain.insert(node);
                members.emplace(node, held_interval{ i, i, ggml_nbytes(node) });
                // The copy is the last reader of what it copies.
                members[node->src[0]].last = i;
                continue;
            }
            for (int src = 0; src < GGML_MAX_SRC; ++src) {
                if (node->src[src] && chain.count(node->src[src])) {
                    members[node->src[src]].last = i;
                }
            }
        }
        size_t longest_bytes = 0;
        int    longest_span  = -1;
        for (const auto & member : members) {
            intervals.push_back(member.second);
            chain_first[c] = std::min(chain_first[c], member.second.first);
            chain_last[c]  = std::max(chain_last[c],  member.second.last);
            const int span = member.second.last - member.second.first;
            if (span > longest_span) {
                longest_span  = span;
                longest_bytes = member.second.bytes;
            }
        }
        chain_bytes[c] = longest_bytes;
        out_profile->n_checkpoints++;
        out_profile->retained_bytes += longest_bytes;
        const int64_t span = chain_last[c] - chain_first[c];
        out_profile->total_span_nodes += span;
        out_profile->max_span_nodes = std::max(out_profile->max_span_nodes, span);
        if (2*span >= n_nodes) {
            out_profile->n_long_lived++;
            out_profile->long_lived_bytes += longest_bytes;
        }
    }
    if (out_profile->n_checkpoints == 0) {
        return false;
    }
    out_profile->n_nodes = n_nodes;

    // Concurrency by sweeping the interval endpoints rather than the nodes: the
    // curve only changes where an interval opens or closes, and there are two
    // endpoints per retained tensor against tens of thousands of nodes.
    // Signed deltas, so an interval's open and close cancel exactly. The cast of
    // a tensor's byte count to int64_t cannot lose anything: ggml_nbytes is
    // bounded by the tensor's allocation, and no allocation approaches 2^63
    // (a narrowing with a proof written beside it).
    struct sweep_event { int node; int64_t d_count; int64_t d_bytes; };
    std::vector<sweep_event> events;
    events.reserve(2*intervals.size());
    for (const held_interval & interval : intervals) {
        const int64_t bytes = (int64_t) interval.bytes;
        events.push_back({ interval.first,     +1, +bytes });
        events.push_back({ interval.last + 1,  -1, -bytes });
    }
    std::sort(events.begin(), events.end(),
            [](const sweep_event & a, const sweep_event & b) { return a.node < b.node; });
    int64_t live_count = 0;
    int64_t live_bytes = 0;
    for (size_t i = 0; i < events.size(); ++i) {
        live_count += events[i].d_count;
        live_bytes += events[i].d_bytes;
        // Only compare once every event at this node has been applied, otherwise
        // a close-then-open pair at the same index reports a phantom peak.
        if (i + 1 < events.size() && events[i + 1].node == events[i].node) {
            continue;
        }
        // live_bytes is a partial sum of matched +bytes/-bytes pairs and every
        // open precedes its close, so it is never negative here.
        if ((size_t) live_bytes > out_profile->live_peak_bytes) {
            out_profile->live_peak_bytes = (size_t) live_bytes;
            out_profile->live_peak_count = live_count;
            out_profile->live_peak_node  = events[i].node;
        }
    }
    return true;
}

bool ggml_opt_get_checkpoint_profile(
        ggml_opt_context_t opt_ctx,
        struct ggml_opt_checkpoint_profile * out_profile) {
    GGML_ASSERT(out_profile);
    *out_profile = opt_ctx->checkpoint_profile;
    return out_profile->n_checkpoints > 0;
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);

    // retro delta: a backward pass at opt_i == 0 opens an accumulation window,
    // and a window starts empty. Upstream clears the accumulators here through
    // opt_ctx->gb_grad, which is the graph of the *previous* evaluation - with
    // dynamic graphs ggml_opt_eval has already dropped it, so the clear ran on
    // a null graph and every window after the first carried the gradients of
    // all the windows before it. That is invisible inside one uninterrupted
    // run (the sum is deterministic) and shows up the moment a run is resumed
    // from a checkpoint: the accumulators are live state no checkpoint holds,
    // so a restored trainer starts its next window from zero while the
    // uninterrupted one starts from the carried sum.
    //
    // The window is identified from opt_i alone rather than from the previous
    // build type, which an intervening evaluation-only pass would have changed
    // to FORWARD.
    const bool window_start =
        backward && opt_ctx->opt_i == 0 && ggml_opt_grads_accumulate(opt_ctx);
    if (window_start && opt_ctx->gb_grad) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs && !opt_ctx->cache_dynamic_graph) {
        ggml_opt_build(opt_ctx);
        // The graph this window accumulates into only exists now. The
        // accumulators it names are the persistent ones of ctx_static, so
        // clearing them through a freshly built gb_grad clears the same
        // tensors the cleared-above branch would have.
        if (window_start) {
            ggml_graph_reset(opt_ctx->gb_grad);
        }
    }

    struct ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } break;
    }
    GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

struct ggml_cgraph * ggml_opt_graph(ggml_opt_context_t opt_ctx) {
    return opt_ctx->allocated_graph_copy;
}

void ggml_opt_set_graph_cache(ggml_opt_context_t opt_ctx, bool enabled) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    GGML_ASSERT(!enabled || opt_ctx->eval_ready);
    if (opt_ctx->cache_dynamic_graph == enabled) {
        return;
    }
    opt_ctx->cache_dynamic_graph = enabled;
    if (!enabled) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
        opt_ctx->ctx_compute          = nullptr;
    }
}

void ggml_opt_invalidate_graph_allocation(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    opt_ctx->allocated_graph      = nullptr;
    opt_ctx->allocated_graph_copy = nullptr;
}

void ggml_opt_cancel(ggml_opt_context_t opt_ctx) {
    if (!opt_ctx->static_graphs && !opt_ctx->cache_dynamic_graph) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }
    opt_ctx->eval_ready = false;
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        const ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);
        GGML_ASSERT(opt_pars.max_grad_norm > 0.0f);
        // One ceiling for one norm, whatever owns each parameter.
        GGML_ASSERT(opt_ctx->opt_max_grad_norm);
        ggml_get_data_f32(opt_ctx->opt_max_grad_norm)[0] = opt_pars.max_grad_norm;

        // retro delta: one fill per optimizer this run actually builds a step
        // for. Validating the coefficients of an optimizer nothing uses would
        // refuse a value no update ever reads.
        for (int type = 0; type < GGML_OPT_OPTIMIZER_TYPE_COUNT; ++type) {
            if (!opt_ctx->optimizer_used[type]) {
                continue;
            }
            float * values = ggml_get_data_f32(opt_ctx->opt_step_params[type]);
            switch ((enum ggml_opt_optimizer_type) type) {
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                    GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                    GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                    GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                    GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                    GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                    GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                    GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                    GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                    // beta1, beta2 after applying warmup
                    const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                    const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                    values[0] = opt_pars.adamw.alpha;
                    values[1] = opt_pars.adamw.beta1;
                    values[2] = opt_pars.adamw.beta2;
                    values[3] = opt_pars.adamw.eps;
                    values[4] = opt_pars.adamw.wd;
                    values[5] = beta1h;
                    values[6] = beta2h;
                    // retro delta: per-step seed for the stochastic rounding an
                    // F16 parameter needs. Exact as a float below 2^24
                    // iterations; past that the stream repeats a step, which
                    // costs nothing but decorrelation.
                    values[7] = (float) opt_ctx->iter;
                } break;
                case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                    GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                    GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                    GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                    values[0] = opt_pars.sgd.alpha;
                    values[1] = opt_pars.sgd.wd;
                } break;
                default:
                    GGML_ABORT("unknown optimizer");
            }
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    if (!opt_ctx->static_graphs && !opt_ctx->cache_dynamic_graph) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    if (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    struct ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_loss_type         loss_type,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    return c->optimizer;
}

GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o) {
    switch (o) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            return "adamw";
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            return "sgd";
        default:
            return "undefined";
    };
}
