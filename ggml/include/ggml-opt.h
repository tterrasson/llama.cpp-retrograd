// This file contains functionality for training models using GGML.
// It is not strictly needed vs. just vanilla GGML but it provides a more high-level interface for common needs such as datasets.
// At the bottom of this file especially there are relatively high-level functions that are suitable use or adaptation in user code.
//
// Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_opt_dataset;
    struct ggml_opt_context;
    struct ggml_opt_result;

    typedef struct ggml_opt_dataset * ggml_opt_dataset_t;
    typedef struct ggml_opt_context * ggml_opt_context_t;
    typedef struct ggml_opt_result  * ggml_opt_result_t;

    // ====== Loss ======

    // built-in loss types, i.e. the built-in quantities minimized by the optimizer
    // custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
    enum ggml_opt_loss_type {
        GGML_OPT_LOSS_TYPE_MEAN,
        GGML_OPT_LOSS_TYPE_SUM,
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
        GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
        // retro delta (plan 03): the caller supplies a scalar loss node directly
        // as `outputs`; the optimizer uses it verbatim (loss = outputs) and runs
        // autodiff through it. Used by the fused sparse cross-entropy path, whose
        // loss node already averages over the active tokens.
        GGML_OPT_LOSS_TYPE_EXTERNAL,
    };

    // ====== Dataset ======

    GGML_API ggml_opt_dataset_t ggml_opt_dataset_init(
            enum ggml_type type_data,    // the type for the internal data tensor
            enum ggml_type type_label,   // the type for the internal labels tensor
            int64_t        ne_datapoint, // number of elements per datapoint
            int64_t        ne_label,     // number of elements per label
            int64_t        ndata,        // total number of datapoints/labels
            int64_t        ndata_shard); // number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
    // View over caller-owned CPU memory. Buffers remain owned by the caller
    // and must outlive the returned dataset.
    GGML_API ggml_opt_dataset_t ggml_opt_dataset_init_external(
            enum ggml_type type_data,
            enum ggml_type type_label,
            int64_t        ne_datapoint,
            int64_t        ne_label,
            int64_t        ndata,
            int64_t        ndata_shard,
            void *         data,
            void *         labels);
    // Logical concatenation of two caller-owned CPU-memory datasets. The
    // split must fall on a shard boundary; batches and shuffling continue to
    // address one global permutation without copying either segment.
    GGML_API ggml_opt_dataset_t ggml_opt_dataset_init_external_split(
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
            void *         labels_second);
    GGML_API void ggml_opt_dataset_free(ggml_opt_dataset_t dataset);

    // get underlying tensors that store the data
    GGML_API int64_t              ggml_opt_dataset_ndata (ggml_opt_dataset_t dataset);
    GGML_API struct ggml_tensor * ggml_opt_dataset_data  (ggml_opt_dataset_t dataset); // shape = [ne_datapoint, ndata]
    GGML_API struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset); // shape = [nd_label,     ndata]

    // shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
    GGML_API void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata);

    // get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
    GGML_API void ggml_opt_dataset_get_batch(
            ggml_opt_dataset_t   dataset,
            struct ggml_tensor * data_batch,   // shape = [ne_datapoint, ndata_batch]
            struct ggml_tensor * labels_batch, // shape = [ne_label,     ndata_batch]
            int64_t              ibatch);
    GGML_API void ggml_opt_dataset_get_batch_host(
            ggml_opt_dataset_t   dataset,
            void               * data_batch,
            size_t               nb_data_batch,
            void               * labels_batch,
            int64_t              ibatch);

    // ====== Model / Context ======

    enum ggml_opt_build_type {
        GGML_OPT_BUILD_TYPE_FORWARD = 10,
        GGML_OPT_BUILD_TYPE_GRAD    = 20,
        GGML_OPT_BUILD_TYPE_OPT     = 30,
    };

    enum ggml_opt_optimizer_type {
        GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        GGML_OPT_OPTIMIZER_TYPE_SGD,
        // retro delta: orthogonalized momentum on eligible matrices. Built out
        // of ordinary graph ops rather than a kernel of its own.
        GGML_OPT_OPTIMIZER_TYPE_MUON,
        // retro delta: fixed-block second moments, optionally with a quantized
        // first moment. The variant is part of the layout, not of the name.
        GGML_OPT_OPTIMIZER_TYPE_GEFEN,

        GGML_OPT_OPTIMIZER_TYPE_COUNT
    };


    // The frozen v1 constants: five Newton-Schulz iterations, 1024-element
    // quantization blocks and a 256-entry uniform codebook. Declared here
    // because the allocator, the update graph and the default layout are three
    // readers of one number.
    #define GGML_OPT_MUON_NS_STEPS          5
    #define GGML_OPT_GEFEN_BLOCK_SIZE       1024
    #define GGML_OPT_GEFEN_CODEBOOK_LEVELS  256

    // retro delta: an optimizer's *structural* parameters, the ones that decide
    // a slot's shape or a graph's topology instead of scaling an update. They
    // are fixed when the context is created, because a schedule cannot move the
    // number of elements a block holds or the number of nodes a step builds.
    struct ggml_opt_optimizer_layout {
        struct {
            int32_t ns_steps; // Newton-Schulz iterations; <= 0 selects five
            bool    nesterov;
        } muon;
        struct {
            enum ggml_opt_gefen_variant variant;
            int64_t                     block_size; // <= 0 selects 1024
        } gefen;
        // An F32 master copy per half-precision parameter, updated by the F32
        // step and cast back into the parameter's own store. Structural rather
        // than a hyperparameter: it decides whether a slot exists and whether
        // the step is one node or two, which is what this struct carries.
        // Ignored by an F32 parameter, which is already its own master.
        bool master_weights;
    };

    // The layout every optimizer's frozen v1 declares.
    GGML_API struct ggml_opt_optimizer_layout ggml_opt_default_optimizer_layout(void);

    // retro delta: the executable half of an optimizer's descriptor.
    //
    // An optimizer declares what persistent state it keeps; the allocator, the
    // initializer and the update graph all read that declaration instead of
    // spelling one optimizer's layout out in three places. AdamW's pair is one
    // table among others, and an optimizer keeping three slots of two shapes
    // needs no new branch in the allocator.

    // How large one slot is, relative to the parameter it belongs to.
    enum ggml_opt_slot_shape {
        GGML_OPT_SLOT_SHAPE_PARAMETER = 0, // the parameter's own shape
        GGML_OPT_SLOT_SHAPE_BLOCKS    = 1, // ceil(nelements / n_block) elements
        GGML_OPT_SLOT_SHAPE_FIXED     = 2, // n_block elements, parameter-independent
    };

    // What a slot holds before the first update. An enumeration rather than a
    // scalar fill because a codebook is generated, not filled.
    enum ggml_opt_slot_init {
        GGML_OPT_SLOT_INIT_ZERO             = 0, // every element zero
        GGML_OPT_SLOT_INIT_CODE             = 1, // every byte the same code
        GGML_OPT_SLOT_INIT_UNIFORM_CODEBOOK = 2, // c[k] = -1 + 2k/(n-1), F32
        // The parameter's own values, widened. The one initializer whose bytes
        // do not come from the definition: the allocator widens the parameter
        // instead, and ggml_opt_slot_initial_bytes answers false for it so the
        // "one definition, one reader" contract of that function still holds.
        GGML_OPT_SLOT_INIT_PARAMETER        = 3,
    };

    struct ggml_opt_slot_def {
        const char *             name;
        enum ggml_type           type;
        enum ggml_opt_slot_shape shape;
        // Block size for BLOCKS, element count for FIXED, ignored otherwise.
        int64_t                  n_block;
        enum ggml_opt_slot_init  init;
        uint8_t                  code; // GGML_OPT_SLOT_INIT_CODE only
    };

    // The per-parameter and per-owner slot tables of one optimizer. An empty
    // table is a valid answer and is not "no optimizer": SGD keeps no
    // per-parameter state and still has an iteration counter and an RNG.
    // A NULL layout reads the defaults; an optimizer whose slot shapes depend
    // on its layout (Gefen) answers for the layout it is given and not for the
    // one the last context happened to use.
    GGML_API const struct ggml_opt_slot_def * ggml_opt_optimizer_slots(
            enum ggml_opt_optimizer_type             optimizer,
            const struct ggml_opt_optimizer_layout * layout,
            int64_t                                * n_slots);
    GGML_API const struct ggml_opt_slot_def * ggml_opt_optimizer_shared_slots(
            enum ggml_opt_optimizer_type             optimizer,
            const struct ggml_opt_optimizer_layout * layout,
            int64_t                                * n_slots);

    // retro delta: the F32 master copy of one half-precision parameter, when
    // the layout asks for one.
    //
    // The one slot the tables above cannot declare. A slot table belongs to an
    // optimizer, and this slot's existence depends on the *parameter's* dtype,
    // so it is answered per parameter here rather than folded into a table
    // that has no parameter to look at. Returns false, leaving `out` untouched,
    // for an F32 parameter and for a layout that did not ask.
    GGML_API bool ggml_opt_master_slot(
            const struct ggml_opt_optimizer_layout * layout,
            enum ggml_type                           param_type,
            struct ggml_opt_slot_def               * out);

    // How many hyperparameters one optimizer's update step reads, excluding
    // the shared gradient-clipping scale appended to them.
    GGML_API int64_t ggml_opt_optimizer_n_params(enum ggml_opt_optimizer_type optimizer);

    // How many elements one slot of this definition holds for a parameter of
    // n_param_elements. A partial trailing block still costs an element, and a
    // FIXED slot ignores the parameter entirely.
    GGML_API int64_t ggml_opt_slot_n_elements(
            const struct ggml_opt_slot_def * def, int64_t n_param_elements);

    // The bytes a slot holds before the first update, from its definition
    // alone. This is what the allocator writes into a live slot, exposed so the
    // initializer can be checked without allocating a graph: one definition,
    // one reader. Returns false for a buffer that is not exactly the slot's
    // size, which is the only way a caller can get a partial answer, and false
    // for GGML_OPT_SLOT_INIT_PARAMETER, whose bytes are the parameter's and
    // not the definition's.
    GGML_API bool ggml_opt_slot_initial_bytes(
            const struct ggml_opt_slot_def * def, int64_t n_elements, void * out, size_t n_bytes);

    // Which optimizer owns one parameter. A null callback puts every parameter
    // on the context's own optimizer, which is the single-optimizer run.
    typedef enum ggml_opt_optimizer_type (*ggml_opt_get_param_optimizer)(
            const struct ggml_tensor * param, void * userdata);

    // parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
    struct ggml_opt_optimizer_params {
        // Maximum global L2 norm across all parameter gradients. Gradients are
        // scaled together before the optimizer step when this is exceeded.
        float max_grad_norm;
        struct {
            float alpha; // learning rate
            float beta1; // first AdamW momentum
            float beta2; // second AdamW momentum
            float eps;   // epsilon for numerical stability
            float wd;    // weight decay - 0.0f to disable
        } adamw;
        struct {
            float alpha; // learning rate
            float wd;    // weight decay
        } sgd;
        // retro delta: an orthogonalized update and an AdamW one are not in the
        // same units, so the fallback AdamW rate a Muon run uses is declared
        // rather than borrowed - the caller scales both by one schedule.
        struct {
            float alpha;      // learning rate
            float momentum;   // EMA coefficient of the first moment
            float wd;         // decoupled weight decay
            float ns_epsilon; // added to the Frobenius norm before normalizing
        } muon;
        struct {
            float alpha; // learning rate
            float beta1; // first moment, quantized under the quantized_m variant
            float beta2; // per-block second moment
            float eps;   // epsilon for numerical stability
            float wd;    // weight decay
        } gefen;
    };

    // callback to calculate optimizer parameters prior to a backward pass
    // userdata can be used to pass arbitrary data
    typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);

    // returns the default optimizer params (constant, hard-coded values)
    // userdata is not used
    GGML_API struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata);

    // casts userdata to ggml_opt_optimizer_params and returns it
    GGML_API struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata);

    // parameters for initializing a new optimization context
    struct ggml_opt_params {
        ggml_backend_sched_t backend_sched; // defines which backends are used to construct the compute graphs

        // by default the forward graph needs to be reconstructed for each eval
        // if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
        struct ggml_context * ctx_compute;
        struct ggml_tensor  * inputs;
        struct ggml_tensor  * outputs;

        enum ggml_opt_loss_type  loss_type;
        enum ggml_opt_build_type build_type;

        int32_t opt_period; // after how many gradient accumulation steps an optimizer step should be done

        ggml_opt_get_optimizer_params get_opt_pars;    // callback for calculating optimizer parameters
        void *                        get_opt_pars_ud; // userdata for calculating optimizer parameters

        // the optimizer every parameter uses, unless get_param_optimizer answers
        // otherwise; its slot table is what the state allocator reads
        enum ggml_opt_optimizer_type optimizer;

        // retro delta: per-parameter optimizer assignment. NULL keeps the whole
        // run on `optimizer` above. A mixed run allocates each parameter's
        // slots from the table of the optimizer that owns it, and builds each
        // update step from that optimizer's own graph.
        ggml_opt_get_param_optimizer get_param_optimizer;
        void *                       get_param_optimizer_ud;

        // retro delta: the structural half of the optimizers this run may
        // build a step for. Read when the slots are allocated and when the
        // update graph is built, never afterwards.
        struct ggml_opt_optimizer_layout layout;
    };

    // get parameters for an optimization context with defaults set where possible
    // parameters for which no sensible defaults exist are supplied as arguments to this function
    GGML_API struct ggml_opt_params ggml_opt_default_params(
            ggml_backend_sched_t    backend_sched,
            enum ggml_opt_loss_type loss_type);

    GGML_API ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params);
    GGML_API void ggml_opt_free(ggml_opt_context_t opt_ctx);

    // rebind the optimizer to a (re)created backend scheduler; the new scheduler
    // must use the same backends as the one passed to ggml_opt_init. Needed when
    // the owner recreates its scheduler (e.g. on a LoRA adapter change) while the
    // optimizer context lives on, so ggml_opt_alloc does not use a freed scheduler.
    GGML_API void ggml_opt_set_backend_sched(ggml_opt_context_t opt_ctx, ggml_backend_sched_t backend_sched);

    // set gradients to zero, initialize loss, and optionally reset the optimizer
    GGML_API void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer);
    // Change the number of physical graphs in the next logical update. This
    // is legal only at an update boundary; dynamic graphs are rebuilt with the
    // corresponding GRAD/OPT topology on each evaluation.
    GGML_API bool ggml_opt_set_period(ggml_opt_context_t opt_ctx, int32_t opt_period);
    // Abandons an incomplete accumulation period without applying an optimizer
    // step. Persistent momenta and the published iteration are unchanged.
    GGML_API bool ggml_opt_abort_accumulation(ggml_opt_context_t opt_ctx);

    GGML_API bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx); // whether the graphs are allocated_statically

    // get underlying tensors that store data
    // if not using static graphs these pointers become invalid with the next call to ggml_opt_alloc
    GGML_API struct ggml_tensor * ggml_opt_inputs(  ggml_opt_context_t opt_ctx); // forward graph input tensor
    GGML_API struct ggml_tensor * ggml_opt_outputs( ggml_opt_context_t opt_ctx); // forward graph output tensor
    GGML_API struct ggml_tensor * ggml_opt_labels(  ggml_opt_context_t opt_ctx); // labels to compare outputs against
    GGML_API struct ggml_tensor * ggml_opt_loss(    ggml_opt_context_t opt_ctx); // scalar tensor that contains the loss
    // Set the number of active (non-masked) cross-entropy rows for the next
    // evaluation. Zero keeps the operation's normal all-row behaviour.
    GGML_API void ggml_opt_set_loss_active_rows(ggml_opt_context_t opt_ctx, int32_t n_active_rows);
    GGML_API struct ggml_tensor * ggml_opt_pred(    ggml_opt_context_t opt_ctx); // predictions made by outputs
    GGML_API struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx); // number of matching predictions between outputs and labels

    // get the gradient accumulator for a node from the forward graph
    GGML_API struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node);

    // retro delta: the same accumulator, keyed by parameter name.
    //
    // ggml_opt_grad_acc() reads the graph, and a dynamic-graph context drops
    // its graphs at the end of every ggml_opt_eval() - so the gradient the
    // update step just consumed is unreachable the moment the step returns.
    // The accumulators themselves live in ctx_static and outlive the graph,
    // exactly like the momenta below, and the name is what indexes them.
    // Returns NULL before the first optimizer graph is built and for a name
    // that is not a parameter of it.
    GGML_API struct ggml_tensor * ggml_opt_grad_acc_by_name(ggml_opt_context_t opt_ctx, const char * name);

    // retro delta: optimizer-state access for training checkpoints.
    //
    // The persistent slots are allocated lazily, by the first ggml_opt_alloc
    // that builds the optimizer graph; before that ggml_opt_slot_count() is 0
    // and a checkpoint must record that it carries no state. Once allocated the
    // tensors outlive individual graph allocations (they live in ctx_static),
    // so they stay readable and writable between epochs, and they are indexed
    // by a stable (owner, slot) pair rather than by graph node order.
    //
    // Two scopes: per-parameter slots, whose owner is the parameter's name, and
    // shared slots, allocated once per owning optimizer and named by it. The
    // shared table is empty for an optimizer that keeps nothing once.
    GGML_API int64_t ggml_opt_iter(    ggml_opt_context_t opt_ctx);
    GGML_API void    ggml_opt_set_iter(ggml_opt_context_t opt_ctx, int64_t iter);

    GGML_API int64_t               ggml_opt_slot_count( ggml_opt_context_t opt_ctx);
    GGML_API const char *          ggml_opt_slot_owner( ggml_opt_context_t opt_ctx, int64_t index);
    GGML_API const char *          ggml_opt_slot_name(  ggml_opt_context_t opt_ctx, int64_t index);
    GGML_API struct ggml_tensor  * ggml_opt_slot_tensor(ggml_opt_context_t opt_ctx, int64_t index);

    GGML_API int64_t               ggml_opt_shared_slot_count( ggml_opt_context_t opt_ctx);
    GGML_API const char *          ggml_opt_shared_slot_owner( ggml_opt_context_t opt_ctx, int64_t index);
    GGML_API const char *          ggml_opt_shared_slot_name(  ggml_opt_context_t opt_ctx, int64_t index);
    GGML_API struct ggml_tensor  * ggml_opt_shared_slot_tensor(ggml_opt_context_t opt_ctx, int64_t index);

    // The optimizer that owns one marked parameter. `found` distinguishes "no
    // such parameter" from the first optimizer of the enumeration, and may be
    // NULL when the caller already knows the name is marked.
    GGML_API enum ggml_opt_optimizer_type ggml_opt_param_optimizer(
            ggml_opt_context_t opt_ctx, const char * name, bool * found);

    // mt19937 state, serialized as the whitespace-separated decimal word list
    // produced by operator<<. Returns the number of bytes required excluding
    // the terminating NUL; writes only when n_buffer is large enough.
    GGML_API size_t ggml_opt_rng_state(    ggml_opt_context_t opt_ctx, char * buffer, size_t n_buffer);
    GGML_API bool   ggml_opt_set_rng_state(ggml_opt_context_t opt_ctx, const char * state);

    GGML_API enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t); //TODO consistent naming scheme

    GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type);

    // retro delta: upper bound on the graph nodes one parameter's update
    // appends, including its share of the clipping norm. Muon's update is
    // tens of nodes per matrix, not one, so callers must size graphs with
    // this number rather than assuming one node per parameter.
    GGML_API int64_t ggml_opt_step_graph_nodes(
            enum ggml_opt_optimizer_type             optimizer,
            const struct ggml_opt_optimizer_layout * layout);

    // ====== Optimization Result ======

    GGML_API ggml_opt_result_t ggml_opt_result_init(void);
    GGML_API void ggml_opt_result_free(ggml_opt_result_t result);
    GGML_API void ggml_opt_result_reset(ggml_opt_result_t result);

    // get data from result, uncertainties are optional and can be ignored by passing NULL
    GGML_API void ggml_opt_result_ndata(   ggml_opt_result_t result, int64_t * ndata);                  // writes 1 value, number of datapoints
    GGML_API void ggml_opt_result_loss(    ggml_opt_result_t result, double  * loss,     double * unc); // writes 1 value
    GGML_API void ggml_opt_result_pred(    ggml_opt_result_t result, int32_t * pred);                   // writes ndata values
    GGML_API void ggml_opt_result_accuracy(ggml_opt_result_t result, double  * accuracy, double * unc); // writes 1 value

    // ====== Computation ======

    // if not using static graphs, this function must be called prior to ggml_opt_alloc
    GGML_API void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs);

    // Select forward tensors retained as activation checkpoints by the next
    // dynamic backward build. Passing NULL/zero restores the ordinary path.
    GGML_API void ggml_opt_set_gradient_checkpoints(
        ggml_opt_context_t    opt_ctx,
        struct ggml_tensor ** checkpoints,
        int                   n_checkpoints);

    // retro delta: type the retained checkpoints are held in across the backward.
    // GGML_TYPE_F16 halves that term at the cost of recompute bit-parity, since
    // the recomputed activations then start from a rounded checkpoint.
    // GGML_TYPE_COUNT (the default) or GGML_TYPE_F32 keeps them unchanged.
    GGML_API void ggml_opt_set_gradient_checkpoint_type(
        ggml_opt_context_t opt_ctx,
        enum ggml_type     type);

    // retro delta: what the retained checkpoints actually cost, and for how long.
    //
    // The two questions activation offloading has to answer before a single copy
    // is written: how many bytes the
    // checkpoints hold against the device peak, and whether any of them are held
    // long enough for a host round-trip to hide behind compute. Neither is
    // answerable from the checkpoint list alone — the list has shapes, not
    // lifetimes — so this walks the built backward graph.
    //
    // Spans are measured in node indices of that graph, from the node that
    // produces a checkpoint to the last node that reads it. A node index is not a
    // duration: nodes differ by orders of magnitude in cost, and turning a span
    // into seconds would need a per-node timing whose fixed cost exceeds most
    // nodes. It is a position, and that is all it is read as — the ordering of
    // "which checkpoints are held longest" is what plan step 4 needs, and it
    // survives the units it is not measured in.
    //
    // A narrowed checkpoint (see above) is held as its 16-bit copy, and the bytes
    // reported are that copy's: the whole chain — the F32 tensor, its store and
    // its fetch — is followed, so the figure is what is retained, not what the
    // forward built.
    struct ggml_opt_checkpoint_profile {
        int64_t n_checkpoints;
        int64_t n_nodes;           // graph length the spans below are positions in
        size_t  retained_bytes;    // summed over checkpoints, as held
        size_t  live_peak_bytes;   // most bytes held simultaneously
        int64_t live_peak_count;   // checkpoints alive at that point
        int64_t live_peak_node;    // where it occurs
        size_t  long_lived_bytes;  // bytes of checkpoints spanning >= half the graph
        int64_t n_long_lived;
        int64_t max_span_nodes;
        int64_t total_span_nodes;  // divide by n_checkpoints for the mean span
    };

    // Fills `out_profile` from the last backward graph built by ggml_opt_alloc.
    // False (and a zeroed profile) when checkpointing is off or no backward graph
    // has been built yet — the two cases where there is nothing to report, as
    // opposed to a measured nothing.
    GGML_API bool ggml_opt_get_checkpoint_profile(
        ggml_opt_context_t opt_ctx,
        struct ggml_opt_checkpoint_profile * out_profile);

    // allocate the next graph for evaluation, either forward or forward + backward
    // must be called exactly once prior to calling ggml_opt_eval
    GGML_API void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward);

    // retro delta: the graph selected and allocated by the last ggml_opt_alloc
    // (forward, grad, or opt depending on the build type), NULL when none is
    // allocated. Used by the training preflight to inspect the concrete graph
    // without evaluating it.
    GGML_API struct ggml_cgraph * ggml_opt_graph(ggml_opt_context_t opt_ctx);

    // retro delta: retain a graph built through the dynamic API and reuse it
    // on later evaluations. The caller must keep ctx_compute and the forward
    // graph alive. Invalidating only the allocation is sufficient when another
    // graph used the shared backend scheduler between evaluations.
    GGML_API void ggml_opt_set_graph_cache(ggml_opt_context_t opt_ctx, bool enabled);
    GGML_API void ggml_opt_invalidate_graph_allocation(ggml_opt_context_t opt_ctx);

    // retro delta: abandon a graph prepared with ggml_opt_alloc without
    // evaluating it, so the next ggml_opt_alloc starts from a clean state.
    // With dynamic graphs the caller frees the compute context afterwards.
    GGML_API void ggml_opt_cancel(ggml_opt_context_t opt_ctx);

    // do forward pass, increment result if not NULL, do backward pass if allocated
    GGML_API void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result);

    // ############################################################################
    // ## The high-level functions start here. They do not depend on any private ##
    // ## functions or structs and can be copied to and adapted for user code.   ##
    // ############################################################################

    // ====== Intended Usage ======
    //
    // 1. Select the appropriate loss for your problem.
    // 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
    //    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).
    // 3. Create a GGML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
    //    The first context should contain the model parameters and inputs and be allocated statically in user code.
    //    The second context should contain all other tensors and will be (re)allocated automatically.
    //    Due to this automated allocation the data of the second context is not defined when accessed in user code.
    //    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
    // 4. Call ggml_opt_fit. If you need more control you can use ggml_opt_epoch instead.

    // signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
    typedef void (*ggml_opt_epoch_callback)(
            bool               train,       // true after training evaluation, false after validation evaluation
            ggml_opt_context_t opt_ctx,
            ggml_opt_dataset_t dataset,
            ggml_opt_result_t  result,      // result associated with the dataset subsection
            int64_t            ibatch,      // number of batches that have been evaluated so far
            int64_t            ibatch_max,  // total number of batches in this dataset subsection
            int64_t            t_start_us); // time at which the evaluation on the dataset subsection was started

    // do training on front of dataset, do evaluation only on back of dataset
    GGML_API void ggml_opt_epoch(
            ggml_opt_context_t      opt_ctx,
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,   // result to increment during training, ignored if NULL
            ggml_opt_result_t       result_eval,    // result to increment during evaluation, ignored if NULL
            int64_t                 idata_split,    // data index at which to split training and evaluation
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    // callback that prints a progress bar on stderr
    GGML_API void ggml_opt_epoch_callback_progress_bar(
            bool               train,
            ggml_opt_context_t opt_ctx,
            ggml_opt_dataset_t dataset,
            ggml_opt_result_t  result,
            int64_t            ibatch,
            int64_t            ibatch_max,
            int64_t            t_start_us);

    // fit model defined by inputs and outputs to dataset
    GGML_API void ggml_opt_fit(
            ggml_backend_sched_t            backend_sched,  // backend scheduler for constructing the compute graphs
            struct ggml_context           * ctx_compute,    // context with temporarily allocated tensors to calculate the outputs
            struct ggml_tensor            * inputs,         // input tensor with shape [ne_datapoint, ndata_batch]
            struct ggml_tensor            * outputs,        // output tensor, must have shape [ne_label, ndata_batch] if labels are used
            ggml_opt_dataset_t              dataset,        // dataset with data and optionally also labels
            enum ggml_opt_loss_type         loss_type,      // loss to minimize
            enum ggml_opt_optimizer_type    optimizer,      // sgd or adamw
            ggml_opt_get_optimizer_params   get_opt_pars,   // callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
            int64_t                         nepoch,         // how many times the dataset should be iterated over
            int64_t                         nbatch_logical, // datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
            float                           val_split,      // fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
            bool                            silent);        // whether or not info prints to stderr should be suppressed


#ifdef  __cplusplus
}
#endif
