// retro delta: RIR integration support (docs/INT_RIR.md in the parent repo).
//
// The RIR policy, the per-op counters and the AOT registry lookups shared by
// the backend adapters. The dispatch itself lives in each backend, next to
// the buffers and the queue it needs; this header only carries what every
// adapter needs to agree on: the activation mode, the rejection taxonomy and
// the observability counters.
#pragma once

#include "rir_registry.h"
#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// RETRO_RIR_MODE=off|observe|prefer|require, read once before any backend
// context is created. `observe` computes eligibility and counts it but always
// runs the native kernel; `prefer` dispatches the RIR variant when its
// contract matches and falls back to native *before* launch otherwise.
// `require` is `prefer` plus a graph preflight: every node whose (op, backend)
// is integrated must be eligible before anything is encoded, and a node that is
// not turns the whole graph_compute into GGML_STATUS_FAILED instead of a silent
// native fallback (docs/INT_RIR_V2.md §P0).
typedef enum ggml_rir_mode {
    GGML_RIR_MODE_OFF     = 0,
    GGML_RIR_MODE_OBSERVE = 1,
    GGML_RIR_MODE_PREFER  = 2,
    GGML_RIR_MODE_REQUIRE = 3,
} ggml_rir_mode;

// The configured mode. Pipeline creation reads this one, at device init.
ggml_rir_mode ggml_rir_get_mode(void);

// Set the policy programmatically, so it can be a versioned runtime option
// rather than only an environment variable (docs/INT_RIR.md §9.4). Takes
// precedence over RETRO_RIR_MODE.
//
// Returns 0 on success. Returns non-zero, and changes nothing, once the mode has
// been *read* — from that point a backend context may already have decided which
// pipelines to build, and a mode that no longer matches what was compiled is the
// one way `executed_impl` could start lying. Setting the value it already has is
// always accepted, so a second trainer asking for the same policy is not an
// error while a second trainer asking for a different one is.
int ggml_rir_set_mode(ggml_rir_mode mode);

// Whether the mode has been read, i.e. whether ggml_rir_set_mode can still
// succeed with a new value.
bool ggml_rir_mode_is_latched(void);

// The mode a dispatch site must honour: the configured mode, lowered to OFF
// while the probe forces the native path. Only the dispatch sites use it, so
// forcing native can never un-create a pipeline that was already built.
ggml_rir_mode ggml_rir_get_dispatch_mode(void);

// Probe-only knob: force every subsequent dispatch onto the native kernel.
// It can only *lower* the effective mode, never raise it — raising it would
// claim a pipeline that was never created at device init. That asymmetry is
// what keeps `retro_probe_op_run_ex(..., NATIVE, ...)` honest without
// reintroducing a mid-run policy change (docs/INT_RIR.md §9.4).
void ggml_rir_set_force_native(bool force);
bool ggml_rir_get_force_native(void);

// Stable rejection taxonomy (docs/INT_RIR.md §5.1). Order is ABI.
typedef enum ggml_rir_reject {
    GGML_RIR_MATCHED              = 0,
    GGML_RIR_REJECT_WRONG_OP      = 1,
    GGML_RIR_REJECT_DTYPE         = 2,
    GGML_RIR_REJECT_RANK          = 3,
    GGML_RIR_REJECT_SHAPE         = 4,
    GGML_RIR_REJECT_STRIDE        = 5,
    GGML_RIR_REJECT_QUANT_BLOCK   = 6,
    GGML_RIR_REJECT_INTEGER_RANGE = 7,
    GGML_RIR_REJECT_MISSING_FEATURE = 8,
    GGML_RIR_REJECT_PIPELINE      = 9,
    GGML_RIR_REJECT_POLICY_NATIVE = 10,
    GGML_RIR_REJECT_DEVICE_GRID   = 11,
    GGML_RIR_REJECT_DEVICE_ALIGNMENT = 12,
} ggml_rir_reject;

#define GGML_RIR_REJECT_COUNT 13

// Monotonic process-wide counters; one struct, every backend accumulates.
typedef struct ggml_rir_counters {
    uint64_t ops_seen;           // ops with a registered RIR variant reaching a dispatch site
    uint64_t rir_eligible;       // contract matched (observe or prefer)
    uint64_t rir_dispatched;     // RIR pipeline actually encoded
    uint64_t native_dispatched;  // native kernel ran for an op with a variant
    uint64_t fallback_contract;  // reject: dtype/rank/shape/stride/range
    uint64_t fallback_feature;   // reject: missing feature or device limit
    uint64_t fallback_pipeline;  // reject: pipeline absent/not built
    // Same rejections, one bucket per ggml_rir_reject. The three aggregates
    // above answer "how much fell back"; this answers "widen what first",
    // which is the question a coverage run on a real graph has to settle
    // (docs/INT_RIR.md §11 phase D).
    uint64_t reject_by_reason[GGML_RIR_REJECT_COUNT];
} ggml_rir_counters;

// Returns a snapshot; accumulation functions below are thread-safe.
ggml_rir_counters ggml_rir_counters_snapshot(void);
void ggml_rir_count_seen(void);
void ggml_rir_count_eligible(void);
// `variant_id` must be the static registry string of the variant that was
// encoded, so a probe can report *which* variant ran and not merely "some RIR".
void ggml_rir_count_dispatched(const char * variant_id);
void ggml_rir_count_native(void);
void ggml_rir_count_reject(ggml_rir_reject why);

// Per-site breakdown (docs/INT_RIR.md §9.2). The aggregate above answers "how
// much of everything fell back"; with more than one integrated op that is no
// longer actionable, because two ops on two backends land in the same bucket.
//
// A *site* is one (ggml_op, backend) selection chain inside a backend
// dispatcher. `ggml_rir_site_begin` binds every count_* call made afterwards on
// this thread — including the ones the eligibility helpers make several frames
// down — to that site, until `ggml_rir_site_end`. Sites do not nest: a backend
// dispatcher handles one node at a time. Counting outside any site still feeds
// the aggregate, it is just not attributable.
//
// `ggml_op` and `backend` must be static storage (the "GGML_OP_*" spelling from
// the registry and a rir_backend), because rows keep the pointer.
void ggml_rir_site_begin(const char * ggml_op, uint8_t backend);
void ggml_rir_site_end(void);

#define GGML_RIR_MAX_SITES 32
// Variants dispatched *at one site*. One per priority level in practice; the
// cap only bounds the row, and an overflow is reported rather than dropped.
#define GGML_RIR_MAX_SITE_VARIANTS 4

typedef struct ggml_rir_site_counters {
    const char *      ggml_op;
    uint8_t           backend;   // rir_backend
    ggml_rir_counters counters;  // same shape as the aggregate, this site only
    // Which variant actually ran here, and how often. `rir_dispatched` above is
    // the sum; this is the axis that tells two variants of one op apart.
    uint32_t          n_variants;
    const char *      variant_id[GGML_RIR_MAX_SITE_VARIANTS];
    uint64_t          variant_dispatched[GGML_RIR_MAX_SITE_VARIANTS];
    // Dispatches whose variant did not fit the array above. Non-zero means the
    // report is lossy on the variant axis and the cap needs raising — it must
    // never be silently folded into another variant's count.
    uint64_t          variants_overflow;
} ggml_rir_site_counters;

// Rows are append-only and never reordered, so an index stays valid for the
// process lifetime. `ggml_rir_site_count` is monotonic.
uint32_t ggml_rir_site_count(void);
// Zeroed row (ggml_op == NULL) when `index` is out of range.
ggml_rir_site_counters ggml_rir_site_snapshot(uint32_t index);

// Which implementation actually ran, for the op-at-a-time probe path. The
// counters answer "how often"; this answers "what ran, and why not the other
// one", which is the only thing that can prove a test is not silently green
// on the native kernel (docs/INT_RIR.md §9.1).
typedef enum ggml_rir_impl {
    GGML_RIR_IMPL_UNKNOWN = 0,
    GGML_RIR_IMPL_NATIVE  = 1,
    GGML_RIR_IMPL_RIR     = 2,
} ggml_rir_impl;

typedef struct ggml_rir_decision {
    int32_t      impl;     // ggml_rir_impl
    int32_t      reject;   // ggml_rir_reject; MATCHED when impl == RIR
    const char * variant;  // static registry string, or NULL
} ggml_rir_decision;

// Process-wide, last writer wins. Meaningful for a single-op graph; a full
// training graph must be read through the counters instead.
void ggml_rir_decision_reset(void);
ggml_rir_decision ggml_rir_decision_last(void);

// Stable lowercase spelling of a rir_backend ("cuda"/"vulkan"/"metal"/"cpu",
// "unknown" otherwise). One spelling for every report, so a test can match on
// it: the FFI report, the capability report and the stats line must not each
// invent their own.
const char * ggml_rir_backend_name(uint8_t backend);

// Stable lowercase spelling of a ggml_rir_reject ("matched", "stride", ...),
// "unknown" out of range. Same reason: the reject taxonomy is what a coverage
// run is read through, so its names must be one list, not four.
const char * ggml_rir_reject_name(int32_t reject);

// Registry lookup: the **fallback** variant of `kernel` for `backend`, or NULL.
// Called at context init to resolve pipelines, never in the hot path.
//
// A (kernel, backend) pair may publish several variants arbitrated per shape;
// this returns the one with no shape rule, i.e. the one the pair is named
// after. Ask `ggml_rir_find_variant_named` for a specialization.
const rir_variant_desc * ggml_rir_find_variant(const char * kernel, rir_backend backend);

// Registry lookup by full artifact identity: `variant` is the name the schedule
// table gave this lowering ("blocked"), or NULL for the pair's fallback.
//
// This is the lookup a backend needs once a kernel has more than one lowering:
// a Vulkan pipeline is built from a per-variant SPIR-V blob, so the object and
// the registry row must be matched on the variant and not on the kernel
// (docs/INT_RIR_V3.md §R1).
const rir_variant_desc * ggml_rir_find_variant_named(const char * kernel, const char * variant,
                                                     rir_backend backend);

// Registry lookup by the thing a dispatcher actually holds: a ggml op. Returns
// the highest-priority variant registered for `(op, backend)` whose policy is
// not NATIVE_ONLY, or NULL when the pair is native-only or unregistered.
//
// A variant returned here is *registered*, not necessarily *dispatchable*: an
// OBSERVE_GENERATED pair has a variant whose contract is evaluated and counted
// while the native kernel keeps running. Ask `ggml_rir_op_policy` before
// encoding anything (docs/INT_RIR_V2.md §P2).
//
// This is what makes the registry's `priority` and `rir_op_policies` load-bearing
// instead of descriptive: a backend must not name a kernel string of its own
// (docs/INT_RIR_V2.md §P1). `ggml_op` is an `enum ggml_op`, taken as int32_t so
// the header stays usable from C and Objective-C.
const rir_variant_desc * ggml_rir_find_variant_for_op(int32_t ggml_op, rir_backend backend);

// The same question asked about a *node*, which is the only form that can
// answer it once a pair publishes more than one variant: among the rows
// matching `(op, backend)` whose shape rules hold for this node, the highest
// priority wins. The pair's fallback carries no rule, so this never returns
// NULL where `ggml_rir_find_variant_for_op` would return a row.
//
// Every dispatch site and the require-preflight go through this one, because a
// contract evaluated against one variant and encoded with another is exactly
// the drift the registry exists to prevent (docs/INT_RIR_V3.md §R1).
const rir_variant_desc * ggml_rir_find_variant_for_node(int32_t ggml_op, rir_backend backend,
                                                        const struct ggml_tensor * node);

// Whether every shape rule `v` publishes holds for `node`. Exposed because a
// backend that keeps one object per variant (a Vulkan pipeline) needs to say
// which variant a node would use before it can check that object exists.
bool ggml_rir_variant_fits_shape(const rir_variant_desc * v, const struct ggml_tensor * node);

// Whether the *layout* a variant needs holds for `node`. Today that is the
// vector width: a variant reading four elements at a time only accepts a
// tensor whose contiguous stride is one element. It is a claim in the same
// sense as a shape rule — evaluated before a variant is selected, so a
// permuted view falls to the pair's scalar fallback instead of falling out of
// RIR entirely (docs/INT_RIR_V4.md §P1).
bool ggml_rir_variant_fits_layout(const rir_variant_desc * v, const struct ggml_tensor * node);

// The registry's policy for `(op, backend)`: NATIVE_ONLY when the pair is
// pinned or simply absent. Resolved once, like the selection above.
//
// Benchmark/test-only override: RETRO_RIR_TEST_PREFER=<OP>[,<OP>…] raises an
// OBSERVE_GENERATED pair to PREFER_GENERATED for the process. It exists so the
// differential matrix can run a registered-but-not-promoted variant against the
// native kernel — the measurement a promotion decision needs. It can only raise
// observe to prefer; a NATIVE_ONLY pair has no pipeline and stays unreachable.
rir_policy ggml_rir_op_policy(int32_t ggml_op, rir_backend backend);

// The parts of this op's ggml domain the RIR kernel does **not** claim, as a
// bitmask of `ggml_rir_reject` values — `rir_op_policy.assumed_domain`
// (docs/INT_RIR_V4.md §P4).
//
// This is what makes a native fallback *checkable* rather than merely counted.
// A pair's `reject_by_reason` must be a subset of this mask: a bit that is set
// is a restriction the registry published and §11 lets the native kernel be
// kept for, a bit that is clear is a node the kernel said it would serve and
// did not. The lane and the real-graph test both assert exactly that, per site.
//
// The granularity is the category, not the node — "some dtypes are out of
// scope", never which — so a narrowing *inside* a declared category does not
// show up here. `scripts/rir-domain-baseline.tsv` records the claimed-node
// count per pair for that; the two checks are complementary.
//
// Keyed by the registry's `"GGML_OP_*"` spelling rather than by the enum,
// because that is what a *site row* carries: the two consumers that need this
// are both reporting on a site they have already attributed.
uint32_t ggml_rir_op_assumed_domain(const char * ggml_op_spelling, rir_backend backend);

// Whether `(op, backend)` is *targeted*: a variant exists **and** its policy is
// PREFER_GENERATED, i.e. this pair is one whose native kernel RIR intends to
// replace. Only targeted nodes are what `require` requires — requiring a pair
// that is registered for observation only would fail graphs over a variant no
// mode is ever allowed to encode.
bool ggml_rir_op_is_targeted(int32_t ggml_op, rir_backend backend);

// The selection rule itself, over an explicit table: among the rows matching
// `(ggml_op, backend)` whose policy is not NATIVE_ONLY **and whose shape rules
// hold for `node`**, the highest priority wins, and the first row wins a tie.
// The shipped registry is resolved through this same function, so the rule has
// exactly one implementation.
//
// `node` may be NULL, which skips the shape filter and answers the weaker
// question "is this pair registered at all" — what the policy cache and the
// pipeline-creation path ask, neither of which holds a node.
//
// Taking the tables as arguments is what makes the rule testable: the shipped
// registry cannot exhibit a tie, a veto and a shape miss in the same run.
const rir_variant_desc * ggml_rir_select_variant(
        const rir_variant_desc * rows, uint32_t n_rows,
        const rir_op_policy * policies, uint32_t n_policies,
        int32_t ggml_op, uint8_t backend, const struct ggml_tensor * node);

// Runs the selection rule on synthetic tables built for the purpose. Returns 0
// when every case held, otherwise a bitmask of the cases that failed — see
// ggml-rir.cpp for what each bit means. Exposed so a lane with no GPU can still
// assert that the registry's `priority` and `rir_op_policies` are load-bearing
// (docs/INT_RIR_V2.md §P1).
uint32_t ggml_rir_selftest_selection(void);

// The tensor a binding of `v` reads or writes on `node`: `src[source]`, or
// `node` itself for the dst binding. NULL when the node has no such src.
const struct ggml_tensor * ggml_rir_binding_tensor(const rir_variant_desc * v, uint32_t i,
                                                   const struct ggml_tensor * node);

// The portable half of the contract, decided from the registry row and the
// tensors alone: element types, effective rank, agreement of every axis extent
// across the bindings that share it, stride alignment, and representability of
// everything the constant buffer carries in `index_bits`.
//
// No backend restates any of this. A constraint that needs a pipeline, a device
// limit or a descriptor offset to answer is the device half, and only that half
// belongs in `ggml_rir_device_check_fn` (docs/INT_RIR_V2.md §P1).
int32_t ggml_rir_evaluate_portable(const rir_variant_desc * v, const struct ggml_tensor * node);

// The device half of the contract, supplied by the backend: everything that
// needs a pipeline, a device limit or a descriptor offset to answer. Returns
// GGML_RIR_MATCHED or the reject reason. Must be pure — it is called once per
// node during the preflight and again at the dispatch site.
typedef int32_t (*ggml_rir_device_check_fn)(void * device_ctx, const struct ggml_tensor * node);

// The single entry both the preflight and the dispatch site go through, so no
// contract can be visible to one and not the other (docs/INT_RIR_V2.md §P1).
// Evaluates the portable part first, then delegates to `device_check`.
// Counts nothing: the caller decides whether this evaluation is a dispatch
// decision or a preflight probe.
int32_t ggml_rir_evaluate(uint8_t backend, const struct ggml_tensor * node,
                          ggml_rir_device_check_fn device_check, void * device_ctx);

// --- descriptor-driven dispatch ---------------------------------------------
//
// What a backend still owns is its queue, its buffers and its pipeline object.
// The constant buffer and the grid are pure functions of the registry row, so
// they are computed here once instead of being retyped in every adapter.

// Capacity of the buffer a backend stages a constant buffer in before a
// dispatch. Fixed here rather than read from the generated
// `RIR_MAX_PUSH_CONSTANT_BYTES`, so that adding a kernel — which may raise that
// maximum — does not recompile the backend translation units for a number they
// only use as a bound (docs/INT_RIR_V3.md §R0). `ggml-rir.cpp` includes the
// generated header and static_asserts that the real maximum still fits, so the
// two cannot drift apart silently.
#define RIR_PUSH_CONSTANT_CAPACITY 256

// Fills `out` — exactly `v->push_constant_bytes` — in the layout the generated
// shader declares: kernel params read from `node->op_params`, then one axis
// extent per axis, then each binding's `nb[d]`. Returns false, writing nothing
// meaningful, when the buffer size does not match or a binding is missing;
// callers must have passed `ggml_rir_evaluate` first.
bool ggml_rir_fill_params(const rir_variant_desc * v, const struct ggml_tensor * node,
                          void * out, size_t n_out);

// Workgroup counts for `node`, per the variant's dispatch geometry:
// `grid[d] = ceil(extent(axis[d]) / per_workgroup[d])`, and 1 for an unused
// dimension. Values are the *counts*, not the total thread extent.
void ggml_rir_grid(const rir_variant_desc * v, const struct ggml_tensor * node, uint32_t grid[3]);

// The whole selection chain for one node, at a dispatch site. It opens the
// per-site attribution, counts the node as seen, evaluates the contract through
// `ggml_rir_evaluate`, counts eligibility or the rejection, and enforces
// `require`.
//
// Returns the variant the caller must encode, or NULL when the native kernel
// has to run — in which case the native dispatch is already counted and the
// site is already closed. After encoding, the caller closes the site with
// `ggml_rir_dispatch_end`.
//
// Everything a dispatch site used to spell out — the mode test, the site name,
// the counters, the require abort — lives here, so integrating a second op is a
// call and not a copy of forty lines (docs/INT_RIR_V2.md §P2).
const rir_variant_desc * ggml_rir_dispatch_begin(uint8_t backend, const struct ggml_tensor * node,
                                                 ggml_rir_device_check_fn device_check,
                                                 void * device_ctx);

// Counts the dispatch of the variant `ggml_rir_dispatch_begin` returned and
// closes the site. Call it exactly once, after the encoding.
void ggml_rir_dispatch_end(const rir_variant_desc * variant);

// Graph preflight for `require`, called by a backend at the top of its
// graph_compute, before any encoding. Returns true when the mode is not
// `require`, or when every targeted node of `cgraph` is eligible. Returns false
// after recording the first violation, and the backend must then fail the graph
// rather than run a native kernel.
bool ggml_rir_preflight_graph(uint8_t backend, struct ggml_cgraph * cgraph,
                              ggml_rir_device_check_fn device_check, void * device_ctx);

// --- graph census -----------------------------------------------------------
//
// The site counters answer "how did the ops RIR already covers behave". They
// cannot answer the question R2 opens (docs/INT_RIR_V3.md §5): *which op should
// be written next*. That one is about the ops RIR does **not** cover, so it
// needs a hook that sees every node of the graph, not only the registered ones.
//
// The census is that hook. It is called from the same place as the preflight —
// the top of a backend's graph_compute, where the whole graph is in hand — and
// records, per (ggml_op, backend): how many nodes, how many output elements,
// how many bytes the node reads and writes, and the distinct shapes behind it.
//
// It measures **work**, not time. A per-node timing would need a
// synchronization per node, whose fixed submission cost on macOS is larger than
// most nodes (docs/INT_RIR_V3.md §10, R0 niveau 1) and would flatten exactly
// the ranking it is meant to produce. So the census publishes the shapes, and
// the time of a shape is what `scripts/test-rir.sh` already measures in
// isolation. Two measurements, each honest about what it observed.
//
// Off unless RETRO_RIR_CENSUS=1: it walks every node of every graph.
bool ggml_rir_census_enabled(void);

#define GGML_RIR_MAX_CENSUS_ROWS   96
// Distinct destination shapes kept per row. A ranking needs the shapes to feed
// the isolated bench, and an op that runs at forty different shapes is one
// whose cost no single shape represents — which the overflow counter says.
#define GGML_RIR_MAX_CENSUS_SHAPES 6

typedef struct ggml_rir_census_shape {
    int64_t  ne[4];
    int32_t  type;   // ggml_type of the destination
    uint64_t n_nodes;
} ggml_rir_census_shape;

typedef struct ggml_rir_census_row {
    const char * ggml_op;      // ggml_op_name, NULL for an out-of-range index
    uint8_t      backend;      // rir_backend
    uint64_t     n_nodes;
    uint64_t     n_elements;   // Σ ggml_nelements(node)
    uint64_t     n_bytes;      // Σ nbytes(node) + Σ nbytes(src[i])
    bool         registered;   // the registry knows this (op, backend)
    uint32_t              n_shapes;
    ggml_rir_census_shape shapes[GGML_RIR_MAX_CENSUS_SHAPES];
    uint64_t              shapes_overflow;  // nodes whose shape did not fit
} ggml_rir_census_row;

// Walks `cgraph` and accumulates into the process-wide table. A no-op when the
// census is disabled. Accumulates across calls: "cumulative" is over the whole
// run, so one training step is one call per graph the scheduler computes.
void ggml_rir_census_graph(uint8_t backend, const struct ggml_cgraph * cgraph);

// Rows are append-only, like the site rows: an index stays valid for the
// process lifetime and the count is monotonic.
uint32_t            ggml_rir_census_count(void);
ggml_rir_census_row ggml_rir_census_snapshot(uint32_t index);

// The last violation recorded by a preflight (or by a dispatch site that found
// one under `require`). Process-wide, last writer wins — the same contract the
// decision snapshot has.
typedef struct ggml_rir_violation {
    const char * ggml_op;    // registry spelling, NULL when nothing was recorded
    uint8_t      backend;    // rir_backend
    int32_t      reject;     // ggml_rir_reject
    const char * node_name;  // static buffer, valid until the next violation
} ggml_rir_violation;

void               ggml_rir_violation_reset(void);
ggml_rir_violation ggml_rir_violation_last(void);
// One-line rendering of the last violation into `buf` ("" when none). The same
// sentence for the backend log and the FFI error, so a test can match one text.
void               ggml_rir_violation_format(char * buf, size_t n_buf);

// Largest byte offset a fully strided kernel will compute for `t`, i.e.
// `Σ_d (ne[d]-1)*nb[d] + type_size` over the four ggml dimensions. The caller
// rejects with GGML_RIR_REJECT_INTEGER_RANGE when it does not fit the u32 the
// manifests publish; a silent cast would address the wrong bytes.
//
// Every dimension counts because the variants now carry one `nb[d]` per
// argument per dimension: a view whose planes sit further apart than the rows
// they contain (Qwen3.5's packed QKV, docs/INT_RIR.md §11 phase D) addresses
// well past what a row-folded bound would predict.
uint64_t ggml_rir_max_byte_offset(const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif
