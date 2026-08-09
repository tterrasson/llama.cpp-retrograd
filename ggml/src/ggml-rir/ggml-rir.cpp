// retro delta: RIR integration support — see ggml-rir.h.
#include "ggml-rir.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {
std::atomic<bool> g_force_native{false};
} // namespace

void ggml_rir_set_force_native(bool force) {
    g_force_native.store(force, std::memory_order_relaxed);
}

bool ggml_rir_get_force_native(void) {
    return g_force_native.load(std::memory_order_relaxed);
}

ggml_rir_mode ggml_rir_get_dispatch_mode(void) {
    return g_force_native.load(std::memory_order_relaxed) ? GGML_RIR_MODE_OFF : ggml_rir_get_mode();
}

namespace {
// -1 until something resolves it: an explicit ggml_rir_set_mode, or the first
// read falling back to the environment.
std::atomic<int32_t> g_mode{-1};
// Set by the first read. After that the mode is what the pipelines were built
// against, so it can no longer change.
std::atomic<bool> g_mode_latched{false};

ggml_rir_mode mode_from_env() {
    const char * v = std::getenv("RETRO_RIR_MODE");
    if (v == nullptr || std::strcmp(v, "off") == 0) {
        return GGML_RIR_MODE_OFF;
    }
    if (std::strcmp(v, "observe") == 0) {
        return GGML_RIR_MODE_OBSERVE;
    }
    if (std::strcmp(v, "prefer") == 0) {
        return GGML_RIR_MODE_PREFER;
    }
    if (std::strcmp(v, "require") == 0) {
        return GGML_RIR_MODE_REQUIRE;
    }
    // An unknown value must not silently enable anything.
    return GGML_RIR_MODE_OFF;
}
} // namespace

int ggml_rir_set_mode(ggml_rir_mode mode) {
    const int32_t current = g_mode.load(std::memory_order_relaxed);
    if (current == (int32_t) mode) {
        return 0;  // idempotent, latched or not
    }
    if (g_mode_latched.load(std::memory_order_relaxed)) {
        return -1;
    }
    g_mode.store((int32_t) mode, std::memory_order_relaxed);
    return 0;
}

bool ggml_rir_mode_is_latched(void) {
    return g_mode_latched.load(std::memory_order_relaxed);
}

ggml_rir_mode ggml_rir_get_mode(void) {
    // The mode must be fixed before any backend context exists and must not
    // appear to change mid-run, so the first read is also what freezes it.
    int32_t mode = g_mode.load(std::memory_order_relaxed);
    if (mode < 0) {
        mode = (int32_t) mode_from_env();
        int32_t expected = -1;
        g_mode.compare_exchange_strong(expected, mode, std::memory_order_relaxed);
        mode = g_mode.load(std::memory_order_relaxed);
    }
    g_mode_latched.store(true, std::memory_order_relaxed);
    return (ggml_rir_mode) mode;
}

namespace {
struct counters {
    std::atomic<uint64_t> ops_seen{0};
    std::atomic<uint64_t> rir_eligible{0};
    std::atomic<uint64_t> rir_dispatched{0};
    std::atomic<uint64_t> native_dispatched{0};
    std::atomic<uint64_t> fallback_contract{0};
    std::atomic<uint64_t> fallback_feature{0};
    std::atomic<uint64_t> fallback_pipeline{0};
    std::atomic<uint64_t> reject_by_reason[GGML_RIR_REJECT_COUNT] = {};
};
counters & g() {
    static counters c;
    return c;
}

ggml_rir_counters snapshot_of(const counters & c) {
    ggml_rir_counters out;
    out.ops_seen          = c.ops_seen.load(std::memory_order_relaxed);
    out.rir_eligible      = c.rir_eligible.load(std::memory_order_relaxed);
    out.rir_dispatched    = c.rir_dispatched.load(std::memory_order_relaxed);
    out.native_dispatched = c.native_dispatched.load(std::memory_order_relaxed);
    out.fallback_contract = c.fallback_contract.load(std::memory_order_relaxed);
    out.fallback_feature  = c.fallback_feature.load(std::memory_order_relaxed);
    out.fallback_pipeline = c.fallback_pipeline.load(std::memory_order_relaxed);
    for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
        out.reject_by_reason[i] = c.reject_by_reason[i].load(std::memory_order_relaxed);
    }
    return out;
}

// --- per-site breakdown -----------------------------------------------------
//
// Rows are append-only: `n_rows` only ever grows, a published row is never
// moved or rewritten, so a reader can scan `[0, n_rows)` without a lock while a
// writer appends. Appends take the mutex; increments are relaxed atomics, the
// same cost the aggregate already paid.
struct site_row {
    const char *          ggml_op = nullptr;
    uint8_t               backend = 0;
    counters              c;
    std::atomic<uint32_t> n_variants{0};
    std::atomic<const char *> variant_id[GGML_RIR_MAX_SITE_VARIANTS] = {};
    std::atomic<uint64_t> variant_dispatched[GGML_RIR_MAX_SITE_VARIANTS] = {};
    std::atomic<uint64_t> variants_overflow{0};
};

site_row               g_sites[GGML_RIR_MAX_SITES];
std::atomic<uint32_t>  g_n_sites{0};
std::mutex             g_sites_mutex;

// The site the current thread is counting into, or null outside a site. A
// pointer rather than a key so the count_* calls stay a single relaxed add:
// the lookup happens once per node, in site_begin.
thread_local site_row * t_site = nullptr;

// A site is one (op, backend), whatever variant served the node. The op is
// compared by **content**, not by pointer: with several variants per pair the
// spelling now reaches this function from several registry rows, and C++ does
// not guarantee that two identical string literals share an address. Comparing
// addresses would split one op's counters — and its per-variant breakdown, the
// very thing those rows exist to show — across two site lines.
bool same_site(const site_row & row, const char * ggml_op, uint8_t backend) {
    if (row.backend != backend || row.ggml_op == nullptr) {
        return false;
    }
    return row.ggml_op == ggml_op || std::strcmp(row.ggml_op, ggml_op) == 0;
}

site_row * find_or_append_site(const char * ggml_op, uint8_t backend) {
    const uint32_t n = g_n_sites.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i) {
        if (same_site(g_sites[i], ggml_op, backend)) {
            return &g_sites[i];
        }
    }
    std::lock_guard<std::mutex> lock(g_sites_mutex);
    // Re-scan under the lock: another thread may have appended the same row.
    const uint32_t n2 = g_n_sites.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < n2; ++i) {
        if (same_site(g_sites[i], ggml_op, backend)) {
            return &g_sites[i];
        }
    }
    if (n2 >= GGML_RIR_MAX_SITES) {
        // Out of rows: the aggregate stays exact, this site is just not broken
        // out. Silently reusing another row would be worse than losing the axis.
        return nullptr;
    }
    g_sites[n2].ggml_op = ggml_op;
    g_sites[n2].backend = backend;
    g_n_sites.store(n2 + 1, std::memory_order_release);
    return &g_sites[n2];
}

// Both the aggregate and — when inside a site — that site's row.
template <typename Pick> void bump(Pick pick) {
    pick(g()).fetch_add(1, std::memory_order_relaxed);
    if (t_site) {
        pick(t_site->c).fetch_add(1, std::memory_order_relaxed);
    }
}

// Defined with the census below, in this same anonymous namespace: the printer
// is one place, and the census table is the last thing it prints.
void census_print(std::FILE * out);

// RETRO_RIR_STATS=1 prints the counters at process exit — the cheap
// observability path while the FFI report is not wired yet. Never a
// substitute for asserting executed_impl in tests, but enough to see that
// "prefer" really dispatched RIR and not a silent native fallback.
struct stats_printer {
    ~stats_printer() {
        const char * v = std::getenv("RETRO_RIR_STATS");
        if (v == nullptr || v[0] == '0') {
            // The census is asked for separately and answers a different
            // question, so it must not require the counters to be on too.
            census_print(stderr);
            return;
        }
        // The selection rule, exercised on synthetic tables. It lives here
        // because this is the one path a lane reliably reaches: the rule is
        // pure, the shipped registry cannot exhibit a tie, a policy veto and a
        // shape miss in one run, and a variant table nothing checks is a table
        // that silently stops arbitrating (docs/INT_RIR_V3.md §R1).
        std::fprintf(stderr, "ggml-rir: selftest selection=0x%x\n",
            ggml_rir_selftest_selection());
        const ggml_rir_counters c = ggml_rir_counters_snapshot();
        std::fprintf(stderr,
            "ggml-rir: seen=%llu eligible=%llu rir_dispatched=%llu native=%llu "
            "fb_contract=%llu fb_feature=%llu fb_pipeline=%llu\n",
            (unsigned long long) c.ops_seen,
            (unsigned long long) c.rir_eligible,
            (unsigned long long) c.rir_dispatched,
            (unsigned long long) c.native_dispatched,
            (unsigned long long) c.fallback_contract,
            (unsigned long long) c.fallback_feature,
            (unsigned long long) c.fallback_pipeline);
        // The per-reason breakdown: what to widen first, in one line.
        std::fprintf(stderr, "ggml-rir: rejects");
        for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
            if (c.reject_by_reason[i] != 0) {
                std::fprintf(stderr, " %s=%llu", ggml_rir_reject_name(i),
                    (unsigned long long) c.reject_by_reason[i]);
            }
        }
        std::fprintf(stderr, "\n");
        // The same numbers split by (op, backend, variant). With one op these
        // lines restate the aggregate; with two they are the only way to tell
        // which op is not being covered (docs/INT_RIR.md §9.2).
        const uint32_t n_sites = ggml_rir_site_count();
        for (uint32_t s = 0; s < n_sites; ++s) {
            const ggml_rir_site_counters row = ggml_rir_site_snapshot(s);
            std::fprintf(stderr,
                "ggml-rir: site %s/%s seen=%llu eligible=%llu rir=%llu native=%llu",
                row.ggml_op ? row.ggml_op : "?",
                ggml_rir_backend_name(row.backend),
                (unsigned long long) row.counters.ops_seen,
                (unsigned long long) row.counters.rir_eligible,
                (unsigned long long) row.counters.rir_dispatched,
                (unsigned long long) row.counters.native_dispatched);
            for (uint32_t v = 0; v < row.n_variants; ++v) {
                std::fprintf(stderr, " [%s]=%llu",
                    row.variant_id[v] ? row.variant_id[v] : "?",
                    (unsigned long long) row.variant_dispatched[v]);
            }
            if (row.variants_overflow != 0) {
                std::fprintf(stderr, " [overflow]=%llu",
                    (unsigned long long) row.variants_overflow);
            }
            for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
                if (row.counters.reject_by_reason[i] != 0) {
                    std::fprintf(stderr, " %s=%llu", ggml_rir_reject_name(i),
                        (unsigned long long) row.counters.reject_by_reason[i]);
                }
            }
            std::fprintf(stderr, "\n");
        }
        census_print(stderr);
    }
};
stats_printer g_stats_printer;
} // namespace

ggml_rir_counters ggml_rir_counters_snapshot(void) {
    return snapshot_of(g());
}

namespace {
// Last decision, process-wide. Three independent atomics rather than a lock:
// the probe writes and reads them from one op at a time, and a torn read on a
// concurrent training graph is harmless because that case is meant to be read
// through the counters instead.
std::atomic<int32_t>      g_last_impl{GGML_RIR_IMPL_UNKNOWN};
std::atomic<int32_t>      g_last_reject{GGML_RIR_MATCHED};
std::atomic<const char *> g_last_variant{nullptr};
} // namespace

void ggml_rir_decision_reset(void) {
    g_last_impl.store(GGML_RIR_IMPL_UNKNOWN, std::memory_order_relaxed);
    g_last_reject.store(GGML_RIR_MATCHED, std::memory_order_relaxed);
    g_last_variant.store(nullptr, std::memory_order_relaxed);
}

ggml_rir_decision ggml_rir_decision_last(void) {
    ggml_rir_decision d;
    d.impl    = g_last_impl.load(std::memory_order_relaxed);
    d.reject  = g_last_reject.load(std::memory_order_relaxed);
    d.variant = g_last_variant.load(std::memory_order_relaxed);
    return d;
}

void ggml_rir_site_begin(const char * ggml_op, uint8_t backend) {
    t_site = ggml_op ? find_or_append_site(ggml_op, backend) : nullptr;
}

void ggml_rir_site_end(void) {
    t_site = nullptr;
}

uint32_t ggml_rir_site_count(void) {
    return g_n_sites.load(std::memory_order_acquire);
}

ggml_rir_site_counters ggml_rir_site_snapshot(uint32_t index) {
    ggml_rir_site_counters out = {};
    if (index >= g_n_sites.load(std::memory_order_acquire)) {
        return out;
    }
    const site_row & row = g_sites[index];
    out.ggml_op  = row.ggml_op;
    out.backend  = row.backend;
    out.counters = snapshot_of(row.c);
    const uint32_t n = row.n_variants.load(std::memory_order_acquire);
    out.n_variants = n < GGML_RIR_MAX_SITE_VARIANTS ? n : GGML_RIR_MAX_SITE_VARIANTS;
    for (uint32_t i = 0; i < out.n_variants; ++i) {
        out.variant_id[i]         = row.variant_id[i].load(std::memory_order_relaxed);
        out.variant_dispatched[i] = row.variant_dispatched[i].load(std::memory_order_relaxed);
    }
    out.variants_overflow = row.variants_overflow.load(std::memory_order_relaxed);
    return out;
}

void ggml_rir_count_seen(void)     { bump([](counters & c) -> std::atomic<uint64_t> & { return c.ops_seen;     }); }
void ggml_rir_count_eligible(void) { bump([](counters & c) -> std::atomic<uint64_t> & { return c.rir_eligible; }); }

void ggml_rir_count_dispatched(const char * variant_id) {
    bump([](counters & c) -> std::atomic<uint64_t> & { return c.rir_dispatched; });
    if (t_site) {
        // Keyed by **content**, with pointer identity as the fast path. Registry
        // strings are static, but two rows spelling the same variant_id are two
        // literals, and C++ does not promise they share an address — keying on
        // the address alone would show one variant twice under the same name.
        site_row & row = *t_site;
        auto same_variant = [&](uint32_t i) {
            const char * stored = row.variant_id[i].load(std::memory_order_relaxed);
            if (stored == variant_id) {
                return true;
            }
            return stored != nullptr && variant_id != nullptr
                && std::strcmp(stored, variant_id) == 0;
        };
        const uint32_t n = row.n_variants.load(std::memory_order_acquire);
        uint32_t slot = GGML_RIR_MAX_SITE_VARIANTS;
        for (uint32_t i = 0; i < n && i < GGML_RIR_MAX_SITE_VARIANTS; ++i) {
            if (same_variant(i)) {
                slot = i;
                break;
            }
        }
        if (slot == GGML_RIR_MAX_SITE_VARIANTS) {
            std::lock_guard<std::mutex> lock(g_sites_mutex);
            const uint32_t n2 = row.n_variants.load(std::memory_order_relaxed);
            for (uint32_t i = 0; i < n2 && i < GGML_RIR_MAX_SITE_VARIANTS; ++i) {
                if (same_variant(i)) {
                    slot = i;
                    break;
                }
            }
            if (slot == GGML_RIR_MAX_SITE_VARIANTS && n2 < GGML_RIR_MAX_SITE_VARIANTS) {
                row.variant_id[n2].store(variant_id, std::memory_order_relaxed);
                row.n_variants.store(n2 + 1, std::memory_order_release);
                slot = n2;
            }
        }
        if (slot < GGML_RIR_MAX_SITE_VARIANTS) {
            row.variant_dispatched[slot].fetch_add(1, std::memory_order_relaxed);
        } else {
            row.variants_overflow.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_last_variant.store(variant_id, std::memory_order_relaxed);
    g_last_reject.store(GGML_RIR_MATCHED, std::memory_order_relaxed);
    g_last_impl.store(GGML_RIR_IMPL_RIR, std::memory_order_relaxed);
}

void ggml_rir_count_native(void) {
    bump([](counters & c) -> std::atomic<uint64_t> & { return c.native_dispatched; });
    // Keeps whatever reject reason the eligibility check just recorded: that
    // pair (NATIVE, why) is the answer to "why did RIR not run here".
    g_last_impl.store(GGML_RIR_IMPL_NATIVE, std::memory_order_relaxed);
}

void ggml_rir_count_reject(ggml_rir_reject why) {
    g_last_reject.store((int32_t) why, std::memory_order_relaxed);
    if (why >= 0 && why < GGML_RIR_REJECT_COUNT) {
        const int slot = (int) why;
        bump([slot](counters & c) -> std::atomic<uint64_t> & { return c.reject_by_reason[slot]; });
    }
    switch (why) {
        case GGML_RIR_REJECT_MISSING_FEATURE:
            bump([](counters & c) -> std::atomic<uint64_t> & { return c.fallback_feature; });
            break;
        case GGML_RIR_REJECT_PIPELINE:
            bump([](counters & c) -> std::atomic<uint64_t> & { return c.fallback_pipeline; });
            break;
        case GGML_RIR_MATCHED:
        case GGML_RIR_REJECT_POLICY_NATIVE:
            break;
        default:
            bump([](counters & c) -> std::atomic<uint64_t> & { return c.fallback_contract; });
            break;
    }
}

const char * ggml_rir_reject_name(int32_t reject) {
    static const char * names[GGML_RIR_REJECT_COUNT] = {
        "matched", "wrong_op", "dtype", "rank", "shape", "stride",
        "quant_block", "integer_range", "missing_feature", "pipeline",
        "policy_native",
    };
    if (reject < 0 || reject >= GGML_RIR_REJECT_COUNT) {
        return "unknown";
    }
    return names[reject];
}

const char * ggml_rir_backend_name(uint8_t backend) {
    switch ((rir_backend) backend) {
        case RIR_BACKEND_CUDA:   return "cuda";
        case RIR_BACKEND_VULKAN: return "vulkan";
        case RIR_BACKEND_METAL:  return "metal";
        case RIR_BACKEND_CPU:    return "cpu";
    }
    return "unknown";
}

uint64_t ggml_rir_max_byte_offset(const struct ggml_tensor * t) {
    uint64_t last = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const uint64_t n = t->ne[d] > 0 ? (uint64_t) (t->ne[d] - 1) : 0;
        last += n*(uint64_t) t->nb[d];
    }
    return last + ggml_type_size(t->type);
}

const rir_variant_desc * ggml_rir_find_variant_named(const char * kernel, const char * variant,
                                                     rir_backend backend) {
    if (kernel == nullptr) {
        return nullptr;
    }
    for (uint32_t i = 0; i < rir_variant_count; ++i) {
        const rir_variant_desc & v = rir_variants[i];
        if (v.backend != (uint8_t) backend || std::strcmp(v.kernel, kernel) != 0) {
            continue;
        }
        // NULL means "the pair's fallback", which is the row that carries no
        // variant name — not "any row of this kernel". Matching loosely is how
        // a second variant would silently steal the fallback's pipeline.
        const bool same = variant == nullptr
                ? v.variant == nullptr
                : (v.variant != nullptr && std::strcmp(v.variant, variant) == 0);
        if (same) {
            return &v;
        }
    }
    return nullptr;
}

const rir_variant_desc * ggml_rir_find_variant(const char * kernel, rir_backend backend) {
    return ggml_rir_find_variant_named(kernel, nullptr, backend);
}

// --- selection: (ggml_op, backend) -> variant -------------------------------
//
// Resolved once into a table, for two reasons. The registry spells ops as
// "GGML_OP_L2_NORM_BACK" while ggml spells them "L2_NORM_BACK", so matching
// costs a strcmp per row; and the preflight asks this question for every node
// of the graph. The table also settles priority and policy in one place, so a
// dispatcher can no longer pick "the first row carrying this kernel name".

namespace {

constexpr int RIR_N_BACKENDS = 4;  // rir_backend is dense: cuda/vulkan/metal/cpu

// The variants a (op, backend) pair may resolve to, resolved once. `variant` is
// the shape-blind answer — "is this pair registered, and under what policy" —
// which is what the mode, the policy and the pipeline-creation paths ask.
//
// `candidates` is what a *node* is resolved against, in descending priority so
// the per-node choice is a scan that stops at the first fit. It is small by
// construction: a pair publishes one lowering per shape regime, not one per
// shape.
constexpr uint32_t RIR_MAX_PAIR_VARIANTS = 4;

struct op_selection {
    const rir_variant_desc * variant = nullptr;  // null when unregistered
    rir_policy               policy  = RIR_POLICY_NATIVE_ONLY;
    const rir_variant_desc * candidates[RIR_MAX_PAIR_VARIANTS] = {};
    uint32_t                 n_candidates = 0;
};

op_selection      g_selection[GGML_OP_COUNT][RIR_N_BACKENDS];
std::once_flag    g_selection_once;

// "GGML_OP_L2_NORM_BACK" -> "L2_NORM_BACK", the spelling ggml_op_name returns.
const char * strip_ggml_op_prefix(const char * spelling) {
    static const char prefix[] = "GGML_OP_";
    const size_t n = sizeof(prefix) - 1;
    return std::strncmp(spelling, prefix, n) == 0 ? spelling + n : spelling;
}

// Whether the policy table *explicitly* pins this pair to the native kernel.
// Distinct from "absent", which only means no variant was ever registered: only
// the first is a policy decision worth reporting as one.
bool policy_pins_native(int32_t ggml_op, uint8_t backend);

// Declared before use by ggml_rir_select_variant, which is defined below.
rir_policy policy_of(const rir_op_policy * policies, uint32_t n_policies,
                     const char * ggml_op_spelling, uint8_t backend) {
    for (uint32_t i = 0; i < n_policies; ++i) {
        const rir_op_policy & p = policies[i];
        if (p.backend == backend && std::strcmp(p.ggml_op, ggml_op_spelling) == 0) {
            return (rir_policy) p.policy;
        }
    }
    // A pair absent from the table is native-only by definition (rir_registry.h).
    return RIR_POLICY_NATIVE_ONLY;
}

// Whether `name` appears in a comma-separated list ("CUMSUM,L2_NORM_BACK").
bool name_in_list(const char * spec, const char * name) {
    if (spec == nullptr || spec[0] == '\0' || name == nullptr) {
        return false;
    }
    const size_t n = std::strlen(name);
    for (const char * p = spec; *p != '\0';) {
        const char * end = std::strchr(p, ',');
        const size_t len = end ? (size_t) (end - p) : std::strlen(p);
        if (len == n && std::strncmp(p, name, n) == 0) {
            return true;
        }
        p = end ? end + 1 : p + len;
    }
    return false;
}

// Whether `ggml_op` is named in an env var holding such a list.
bool op_listed_in_env(const char * var, int32_t ggml_op) {
    if (ggml_op < 0 || ggml_op >= GGML_OP_COUNT) {
        return false;
    }
    return name_in_list(std::getenv(var), ggml_op_name((enum ggml_op) ggml_op));
}

void build_selection() {
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        for (int b = 0; b < RIR_N_BACKENDS; ++b) {
            const rir_variant_desc * v = ggml_rir_select_variant(
                    rir_variants, rir_variant_count,
                    rir_op_policies, rir_op_policy_count, op, (uint8_t) b, nullptr);
            rir_policy policy =
                v ? policy_of(rir_op_policies, rir_op_policy_count, v->ggml_op, (uint8_t) b)
                  : RIR_POLICY_NATIVE_ONLY;
            // Benchmark/test-only promotion: RETRO_RIR_TEST_PREFER=<OP>[,<OP>…]
            // raises an OBSERVE_GENERATED pair to PREFER_GENERATED for this
            // process. It is what lets the differential matrix run RIR against
            // the native kernel *before* the registry promotes the pair — the
            // measurement the promotion decision depends on. It can only raise
            // observe to prefer: a NATIVE_ONLY pair has no pipeline built and
            // must stay unreachable (docs/INT_RIR_V2.md §P2).
            if (policy == RIR_POLICY_OBSERVE_GENERATED &&
                    op_listed_in_env("RETRO_RIR_TEST_PREFER", op)) {
                policy = RIR_POLICY_PREFER_GENERATED;
            }
            g_selection[op][b].variant = v;
            g_selection[op][b].policy  = policy;

            // The candidate list, highest priority first. Insertion sort over a
            // handful of rows, once per process: a node then picks by scanning
            // until the shape rules hold, and the first hit is the highest
            // priority that fits — the selection rule, without re-running it.
            op_selection & sel = g_selection[op][b];
            if (v == nullptr) {
                continue;  // native-only or unregistered: nothing to arbitrate
            }
            for (uint32_t i = 0; i < rir_variant_count; ++i) {
                const rir_variant_desc & row = rir_variants[i];
                if (row.backend != (uint8_t) b ||
                        std::strcmp(strip_ggml_op_prefix(row.ggml_op), ggml_op_name((enum ggml_op) op)) != 0) {
                    continue;
                }
                if (sel.n_candidates >= RIR_MAX_PAIR_VARIANTS) {
                    // Never silently dropped: a variant the table cannot hold is
                    // a variant that would never be selected, which is worse
                    // than a build that says so.
                    GGML_ABORT("ggml-rir: %s/%s publishes more than %u variants",
                            row.ggml_op, ggml_rir_backend_name((uint8_t) b),
                            RIR_MAX_PAIR_VARIANTS);
                }
                uint32_t at = sel.n_candidates;
                while (at > 0 && sel.candidates[at - 1]->priority < row.priority) {
                    sel.candidates[at] = sel.candidates[at - 1];
                    --at;
                }
                sel.candidates[at] = &row;
                ++sel.n_candidates;
            }
        }
    }
}

bool policy_pins_native(int32_t ggml_op, uint8_t backend) {
    if (ggml_op < 0 || ggml_op >= GGML_OP_COUNT) {
        return false;
    }
    const char * name = ggml_op_name((enum ggml_op) ggml_op);
    for (uint32_t i = 0; i < rir_op_policy_count; ++i) {
        const rir_op_policy & p = rir_op_policies[i];
        if (p.backend == backend && std::strcmp(strip_ggml_op_prefix(p.ggml_op), name) == 0) {
            return p.policy == RIR_POLICY_NATIVE_ONLY;
        }
    }
    return false;
}

// Test-only contract injection: RETRO_RIR_TEST_REJECT=<OP>[,<OP>...] makes the
// evaluator answer `shape` for those ops on every backend. There is no natural
// way to build an ineligible L2_NORM_BACK from the probe API — every rejection
// left in the contract needs a tensor too large to allocate or a device offset
// the caller cannot choose — and `require` is exactly the semantics that must
// be tested on the rejecting path. Read once, like the mode.
bool op_reject_injected(int32_t ggml_op) {
    static const char * spec = std::getenv("RETRO_RIR_TEST_REJECT");
    if (ggml_op < 0 || ggml_op >= GGML_OP_COUNT) {
        return false;
    }
    return name_in_list(spec, ggml_op_name((enum ggml_op) ggml_op));
}

} // namespace

const rir_variant_desc * ggml_rir_select_variant(
        const rir_variant_desc * rows, uint32_t n_rows,
        const rir_op_policy * policies, uint32_t n_policies,
        int32_t ggml_op, uint8_t backend, const struct ggml_tensor * node) {
    if (rows == nullptr || ggml_op < 0 || ggml_op >= GGML_OP_COUNT) {
        return nullptr;
    }
    const char * want = ggml_op_name((enum ggml_op) ggml_op);
    const rir_variant_desc * best = nullptr;
    for (uint32_t i = 0; i < n_rows; ++i) {
        const rir_variant_desc & v = rows[i];
        if (v.backend != backend) {
            continue;
        }
        if (std::strcmp(strip_ggml_op_prefix(v.ggml_op), want) != 0) {
            continue;
        }
        if (policy_of(policies, n_policies, v.ggml_op, backend) == RIR_POLICY_NATIVE_ONLY) {
            continue;  // the policy table vetoes, whatever the row says
        }
        // The shape filter, when a node is in hand. A specialization that does
        // not fit this node is not a worse candidate, it is not a candidate:
        // its schedule was measured on the shapes it claims and loses — often
        // by a factor of several — everywhere else (docs/INT_RIR_V3.md §R1).
        if (node != nullptr && !ggml_rir_variant_fits_shape(&v, node)) {
            continue;
        }
        // Strictly greater: the first row wins a tie, so the table order is
        // only ever a tie-breaker and never overrides a priority.
        if (best == nullptr || v.priority > best->priority) {
            best = &v;
        }
    }
    return best;
}

uint32_t ggml_rir_selftest_selection(void) {
    // A real op spelling, because the rule resolves through `ggml_op_name`; a
    // backend that carries no production row, because the point is to exercise
    // the rule and not the shipped table.
    const char * op_spelling = "GGML_OP_L2_NORM_BACK";
    const int32_t op         = GGML_OP_L2_NORM_BACK;
    const uint8_t backend    = RIR_BACKEND_CPU;

    auto row = [&](const char * id, uint8_t priority, uint8_t b) {
        rir_variant_desc v = {};
        v.kernel     = "selftest";
        v.ggml_op    = op_spelling;
        v.variant_id = id;
        v.entrypoint = id;
        v.backend    = b;
        v.priority   = priority;
        return v;
    };
    const rir_op_policy prefer[]  = {{op_spelling, backend, RIR_POLICY_PREFER_GENERATED}};
    const rir_op_policy native[]  = {{op_spelling, backend, RIR_POLICY_NATIVE_ONLY}};
    const rir_op_policy observe[] = {{op_spelling, backend, RIR_POLICY_OBSERVE_GENERATED}};

    auto picked = [&](const rir_variant_desc * rows, uint32_t n,
                      const rir_op_policy * pol, uint32_t n_pol) {
        const rir_variant_desc * v =
            ggml_rir_select_variant(rows, n, pol, n_pol, op, backend, nullptr);
        return v ? v->variant_id : nullptr;
    };
    auto is = [](const char * got, const char * want) {
        return got != nullptr && std::strcmp(got, want) == 0;
    };

    uint32_t failures = 0;

    // bit 0/1: the higher priority wins from either position in the table, so
    // what decides is the number and not the row order.
    const rir_variant_desc hi_first[] = {row("hi", 90, backend), row("lo", 10, backend)};
    const rir_variant_desc lo_first[] = {row("lo", 10, backend), row("hi", 90, backend)};
    if (!is(picked(hi_first, 2, prefer, 1), "hi")) { failures |= 1u << 0; }
    if (!is(picked(lo_first, 2, prefer, 1), "hi")) { failures |= 1u << 1; }

    // bit 2/3: the policy table vetoes, whether it pins the native kernel or
    // simply says nothing about the pair.
    if (picked(lo_first, 2, native, 1) != nullptr) { failures |= 1u << 2; }
    if (picked(lo_first, 2, nullptr, 0) != nullptr) { failures |= 1u << 3; }

    // bit 4: equal priority falls back to table order, and to nothing else.
    const rir_variant_desc tied[] = {row("first", 50, backend), row("second", 50, backend)};
    if (!is(picked(tied, 2, prefer, 1), "first")) { failures |= 1u << 4; }

    // bit 5: a row for another backend is never selected here.
    const rir_variant_desc elsewhere[] = {row("other", 99, RIR_BACKEND_CUDA)};
    if (picked(elsewhere, 1, prefer, 1) != nullptr) { failures |= 1u << 5; }

    // bit 6: OBSERVE_GENERATED still *selects* — the variant must be found, so
    // its contract can be evaluated and counted — and priority still decides.
    if (!is(picked(lo_first, 2, observe, 1), "hi")) { failures |= 1u << 6; }

    // --- per-shape arbitration ---------------------------------------------
    //
    // The rule the second cumsum variant needs, exercised on synthetic rows:
    // the shipped registry has exactly one specialization, so nothing in it
    // could tell "the rule was evaluated" apart from "the rule happened to
    // hold" (docs/INT_RIR_V3.md §R1).
    //
    // One axis, `row`, read from dimension 1 of the dst binding. The
    // specialization claims rows ≤ 4 and outranks the fallback.
    auto shaped = [&](const char * id, uint8_t priority, uint8_t n_rules,
                      uint32_t lo, uint32_t hi) {
        rir_variant_desc v = row(id, priority, backend);
        v.n_bindings = 1;
        v.bindings[0].source = -1;  // the dst tensor
        v.n_axes  = 1;
        v.axes[0] = {"row", {1, -1, -1, -1}};
        v.n_shape_rules = n_rules;
        v.shape_rules[0] = {{0, -1, -1, -1}, 1, lo, hi};
        return v;
    };
    const rir_variant_desc arbitrated[] = {
        shaped("fallback", 80, 0, 0, 0),      // no rule: every shape
        shaped("special",  90, 1, 1, 4),      // rows ≤ 4, and it outranks
    };

    ggml_tensor probe = {};
    probe.op = (enum ggml_op) op;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        probe.ne[d] = 1;
    }
    auto picked_for = [&](int64_t n_rows) {
        probe.ne[1] = n_rows;
        const rir_variant_desc * v = ggml_rir_select_variant(
                arbitrated, 2, prefer, 1, op, backend, &probe);
        return v ? v->variant_id : nullptr;
    };

    // bit 7: inside the claimed range, the specialization wins on priority.
    if (!is(picked_for(4), "special")) { failures |= 1u << 7; }
    // bit 8: outside it, the rule removes it from the running entirely — the
    // higher priority must not survive a shape it did not claim.
    if (!is(picked_for(5), "fallback")) { failures |= 1u << 8; }
    // bit 9: below the lower bound too. A rule is an interval, not a ceiling.
    probe.ne[1] = 0;
    if (!is(picked_for(0), "fallback")) { failures |= 1u << 9; }
    // bit 10: with no node, the shape filter is skipped and the shape-blind
    // answer is the highest priority — the question pipeline creation asks.
    if (!is(picked(arbitrated, 2, prefer, 1), "special")) { failures |= 1u << 10; }

    return failures;
}

const rir_variant_desc * ggml_rir_find_variant_for_op(int32_t ggml_op, rir_backend backend) {
    if (ggml_op < 0 || ggml_op >= GGML_OP_COUNT || (int) backend >= RIR_N_BACKENDS) {
        return nullptr;
    }
    std::call_once(g_selection_once, build_selection);
    return g_selection[ggml_op][(int) backend].variant;
}

const rir_variant_desc * ggml_rir_find_variant_for_node(int32_t ggml_op, rir_backend backend,
                                                        const struct ggml_tensor * node) {
    if (ggml_op < 0 || ggml_op >= GGML_OP_COUNT || (int) backend >= RIR_N_BACKENDS) {
        return nullptr;
    }
    std::call_once(g_selection_once, build_selection);
    const op_selection & sel = g_selection[ggml_op][(int) backend];
    if (node == nullptr) {
        return sel.variant;
    }
    // Descending priority, so the first fit is the winner. The pair's fallback
    // carries no shape rule and therefore always fits, which is what makes this
    // loop total: it cannot answer "no variant" where the shape-blind lookup
    // answers "a variant". That property is checked at generation
    // (`check_schedule_table`), not hoped for here.
    for (uint32_t i = 0; i < sel.n_candidates; ++i) {
        if (ggml_rir_variant_fits_shape(sel.candidates[i], node)) {
            return sel.candidates[i];
        }
    }
    return sel.variant;
}

rir_policy ggml_rir_op_policy(int32_t ggml_op, rir_backend backend) {
    if (ggml_op < 0 || ggml_op >= GGML_OP_COUNT || (int) backend >= RIR_N_BACKENDS) {
        return RIR_POLICY_NATIVE_ONLY;
    }
    std::call_once(g_selection_once, build_selection);
    return g_selection[ggml_op][(int) backend].policy;
}

bool ggml_rir_op_is_targeted(int32_t ggml_op, rir_backend backend) {
    return ggml_rir_find_variant_for_op(ggml_op, backend) != nullptr
        && ggml_rir_op_policy(ggml_op, backend) == RIR_POLICY_PREFER_GENERATED;
}

// --- the portable half of the contract --------------------------------------

const struct ggml_tensor * ggml_rir_binding_tensor(const rir_variant_desc * v, uint32_t i,
                                                   const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr || i >= v->n_bindings) {
        return nullptr;
    }
    const int8_t source = v->bindings[i].source;
    if (source < 0) {
        return node;  // the dst binding
    }
    return source < GGML_MAX_SRC ? node->src[source] : nullptr;
}

namespace {

// The ggml type a registry dtype spelling denotes, or GGML_TYPE_COUNT. The
// registry stays free of ggml enum values — it is generated by a crate that
// does not include ggml.h — so the mapping lives on this side, in one place.
ggml_type ggml_type_of_dtype(const char * dtype) {
    if (dtype == nullptr)                    { return GGML_TYPE_COUNT; }
    if (std::strcmp(dtype, "f32")  == 0)     { return GGML_TYPE_F32;  }
    if (std::strcmp(dtype, "f16")  == 0)     { return GGML_TYPE_F16;  }
    if (std::strcmp(dtype, "bf16") == 0)     { return GGML_TYPE_BF16; }
    if (std::strcmp(dtype, "i32")  == 0)     { return GGML_TYPE_I32;  }
    return GGML_TYPE_COUNT;
}

// The extent of axis `a`, taken from the first binding that indexes it. Only
// meaningful once `ggml_rir_evaluate_portable` has checked that every binding
// sharing the axis agrees, since a single extent is passed for all of them.
int64_t axis_extent(const rir_variant_desc * v, uint32_t a, const struct ggml_tensor * node) {
    for (uint32_t b = 0; b < v->n_bindings; ++b) {
        const int8_t dim = v->axes[a].dim[b];
        if (dim < 0) {
            continue;
        }
        const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
        if (t != nullptr) {
            return t->ne[dim];
        }
    }
    return 0;
}

} // namespace

bool ggml_rir_variant_fits_shape(const rir_variant_desc * v, const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr) {
        return false;
    }
    for (uint32_t r = 0; r < v->n_shape_rules; ++r) {
        const rir_shape_rule & rule = v->shape_rules[r];
        // The product of the named extents, in 64 bits: four ggml dimensions
        // multiplied together overflow a u32 long before any of them is
        // individually suspicious, and an overflowed product would read as a
        // small shape and select the wrong variant.
        uint64_t product = 1;
        for (uint32_t i = 0; i < rule.n_axis; ++i) {
            const int8_t a = rule.axis[i];
            if (a < 0 || (uint32_t) a >= v->n_axes) {
                return false;  // a rule the registry could not resolve
            }
            const int64_t extent = axis_extent(v, (uint32_t) a, node);
            if (extent <= 0) {
                return false;
            }
            product *= (uint64_t) extent;
            if (product > 0xffffffffull) {
                product = 0xffffffffull;  // saturate: only comparisons follow
            }
        }
        if (product < rule.min_extent || product > rule.max_extent) {
            return false;
        }
    }
    return true;
}

int32_t ggml_rir_evaluate_portable(const rir_variant_desc * v, const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr) {
        return GGML_RIR_REJECT_WRONG_OP;
    }
    // `index_bits` is what the generated shader computes its byte offsets in.
    // A wider variant would need this bound widened with it, not ignored.
    const uint64_t index_max =
        v->index_bits >= 64 ? UINT64_MAX : ((uint64_t) 1 << v->index_bits) - 1;

    for (uint32_t b = 0; b < v->n_bindings; ++b) {
        const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
        if (t == nullptr) {
            return GGML_RIR_REJECT_WRONG_OP;  // the node has no such operand
        }
        if (t->type != ggml_type_of_dtype(v->bindings[b].dtype)) {
            return GGML_RIR_REJECT_DTYPE;
        }
        if (ggml_n_dims(t) > (int) v->max_rank) {
            return GGML_RIR_REJECT_RANK;
        }
        // Dimensions the binding publishes no stride for are not addressed, so
        // they must be degenerate; otherwise the kernel would read one plane
        // and call it the whole tensor.
        for (int d = v->bindings[b].rank; d < GGML_MAX_DIMS; ++d) {
            if (t->ne[d] != 1) {
                return GGML_RIR_REJECT_RANK;
            }
        }
        for (int d = 0; d < v->bindings[b].rank; ++d) {
            // Generated code addresses elements, not bytes: a stride that is
            // not a whole number of elements would truncate to another one.
            if (v->bindings[b].elem_bytes != 0 && t->nb[d] % v->bindings[b].elem_bytes != 0) {
                return GGML_RIR_REJECT_STRIDE;
            }
            if ((uint64_t) t->ne[d] > index_max || (uint64_t) t->nb[d] > index_max) {
                return GGML_RIR_REJECT_INTEGER_RANGE;
            }
        }
        if (ggml_rir_max_byte_offset(t) > index_max) {
            return GGML_RIR_REJECT_INTEGER_RANGE;
        }
    }

    // One extent is passed per axis, for every binding that indexes it: the
    // shape half of the contract is exactly that agreement. It subsumes the
    // hand-written "same shape as dst" checks, and says *why* they were needed.
    for (uint32_t a = 0; a < v->n_axes; ++a) {
        int64_t extent = -1;
        for (uint32_t b = 0; b < v->n_bindings; ++b) {
            const int8_t dim = v->axes[a].dim[b];
            if (dim < 0) {
                continue;
            }
            const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
            if (extent < 0) {
                extent = t->ne[dim];
            } else if (t->ne[dim] != extent) {
                return GGML_RIR_REJECT_SHAPE;
            }
        }
    }

    // The grid the dispatcher will ask for, in the same integers.
    uint32_t grid[3];
    ggml_rir_grid(v, node, grid);
    for (uint32_t d = 0; d < 3; ++d) {
        const int8_t a = v->dispatch[d].axis;
        if (a < 0) {
            continue;
        }
        const uint32_t per = v->dispatch[d].per_workgroup ? v->dispatch[d].per_workgroup : 1;
        const uint64_t want = ((uint64_t) axis_extent(v, (uint32_t) a, node) + per - 1) / per;
        if (want > index_max || want != grid[d]) {
            return GGML_RIR_REJECT_INTEGER_RANGE;
        }
    }
    return GGML_RIR_MATCHED;
}

int32_t ggml_rir_evaluate(uint8_t backend, const struct ggml_tensor * node,
                          ggml_rir_device_check_fn device_check, void * device_ctx) {
    if (node == nullptr) {
        return GGML_RIR_REJECT_WRONG_OP;
    }
    const rir_variant_desc * variant =
        ggml_rir_find_variant_for_node((int32_t) node->op, (rir_backend) backend, node);
    if (variant == nullptr) {
        // Either no variant, or a policy that pins the native kernel. The two
        // are distinguishable through the registry; for a dispatch decision they
        // mean the same thing, and only the second can happen at a site that
        // asked about its own op.
        return policy_pins_native((int32_t) node->op, backend)
                ? GGML_RIR_REJECT_POLICY_NATIVE
                : GGML_RIR_REJECT_WRONG_OP;
    }
    if (op_reject_injected((int32_t) node->op)) {
        return GGML_RIR_REJECT_SHAPE;
    }
    // Portable first: a contract the registry publishes must not depend on
    // which backend asked, and a device check must never be reached with a
    // node the manifest already refuses.
    const int32_t portable = ggml_rir_evaluate_portable(variant, node);
    if (portable != GGML_RIR_MATCHED) {
        return portable;
    }
    return device_check ? device_check(device_ctx, node) : GGML_RIR_MATCHED;
}

// --- descriptor-driven dispatch ---------------------------------------------

bool ggml_rir_fill_params(const rir_variant_desc * v, const struct ggml_tensor * node,
                          void * out, size_t n_out) {
    if (v == nullptr || node == nullptr || out == nullptr || n_out != v->push_constant_bytes) {
        return false;
    }
    // Every slot is four bytes, in the order the generated shader declares:
    // params, extents, then one nb[] block per binding.
    uint32_t * slot = (uint32_t *) out;
    const uint32_t n_slots = v->push_constant_bytes / 4;
    uint32_t i = 0;

    for (uint32_t p = 0; p < v->n_params && i < n_slots; ++p, ++i) {
        // op_params is int32_t[]; the bytes are copied, never converted — the
        // shader declares the scalar's type, not this code.
        const uint32_t off = v->params[p].op_params_offset;
        if (off + sizeof(uint32_t) > sizeof(node->op_params)) {
            return false;
        }
        std::memcpy(&slot[i], (const char *) node->op_params + off, sizeof(uint32_t));
    }
    for (uint32_t a = 0; a < v->n_axes && i < n_slots; ++a, ++i) {
        slot[i] = (uint32_t) axis_extent(v, a, node);
    }
    for (uint32_t b = 0; b < v->n_bindings; ++b) {
        const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
        if (t == nullptr) {
            return false;
        }
        for (int d = 0; d < v->bindings[b].rank && i < n_slots; ++d, ++i) {
            slot[i] = (uint32_t) t->nb[d];
        }
    }
    // A layout that does not fill the buffer exactly is a generator/consumer
    // mismatch, and the tail would be whatever the caller's stack held.
    return i == n_slots;
}

void ggml_rir_grid(const rir_variant_desc * v, const struct ggml_tensor * node, uint32_t grid[3]) {
    for (uint32_t d = 0; d < 3; ++d) {
        grid[d] = 1;
        if (v == nullptr || node == nullptr) {
            continue;
        }
        const int8_t a = v->dispatch[d].axis;
        if (a < 0 || (uint32_t) a >= v->n_axes) {
            continue;
        }
        const uint32_t per = v->dispatch[d].per_workgroup ? v->dispatch[d].per_workgroup : 1;
        const int64_t extent = axis_extent(v, (uint32_t) a, node);
        grid[d] = extent > 0 ? (uint32_t) (((uint64_t) extent + per - 1) / per) : 0;
    }
}

// --- the dispatch-site selection chain --------------------------------------

const rir_variant_desc * ggml_rir_dispatch_begin(uint8_t backend, const struct ggml_tensor * node,
                                                 ggml_rir_device_check_fn device_check,
                                                 void * device_ctx) {
    const ggml_rir_mode mode = ggml_rir_get_dispatch_mode();
    if (mode == GGML_RIR_MODE_OFF || node == nullptr) {
        // Nothing is measured under `off`: the site would report a variant the
        // build may not even have created a pipeline for.
        return nullptr;
    }
    const rir_variant_desc * variant =
        ggml_rir_find_variant_for_node((int32_t) node->op, (rir_backend) backend, node);
    const rir_policy policy = ggml_rir_op_policy((int32_t) node->op, (rir_backend) backend);

    // Attribute every count below — including the rejects the evaluator records
    // — to (this op, this backend), so the report can name what is not covered
    // instead of one process-wide total (docs/INT_RIR.md §9.2). The name is the
    // registry's own spelling; a node with no variant has no site, and only
    // feeds the aggregate.
    ggml_rir_site_begin(variant ? variant->ggml_op : nullptr, backend);
    ggml_rir_count_seen();

    const int32_t why = ggml_rir_evaluate(backend, node, device_check, device_ctx);
    if (why == GGML_RIR_MATCHED) {
        ggml_rir_count_eligible();
        // Two independent permissions: the mode says how far *any* integrated
        // op may go, the policy how far *this* pair may go. Encoding needs both.
        if (mode >= GGML_RIR_MODE_PREFER && policy == RIR_POLICY_PREFER_GENERATED) {
            return variant;  // the caller encodes, then calls dispatch_end
        }
    } else {
        ggml_rir_count_reject((ggml_rir_reject) why);
        if (mode == GGML_RIR_MODE_REQUIRE && policy == RIR_POLICY_PREFER_GENERATED) {
            // Unreachable: the graph preflight evaluates the same contract
            // through the same function before anything is encoded. If it is
            // reached, the two answers disagree, and falling back to the native
            // kernel here is precisely what `require` forbids.
            char msg[512];
            ggml_rir_violation_format(msg, sizeof(msg));
            GGML_ABORT("ggml-rir: require violated at dispatch: %s reject=%s (%s)",
                    ggml_op_name(node->op), ggml_rir_reject_name(why),
                    msg[0] ? msg : "no preflight violation recorded");
        }
    }
    ggml_rir_count_native();
    ggml_rir_site_end();
    return nullptr;
}

void ggml_rir_dispatch_end(const rir_variant_desc * variant) {
    ggml_rir_count_dispatched(variant ? variant->variant_id : nullptr);
    ggml_rir_site_end();
}

// --- require preflight ------------------------------------------------------

namespace {
std::mutex   g_violation_mutex;
const char * g_violation_op      = nullptr;  // static registry string
uint8_t      g_violation_backend = 0;
int32_t      g_violation_reject  = GGML_RIR_MATCHED;
char         g_violation_node[GGML_MAX_NAME] = {};

void record_violation(const char * ggml_op, uint8_t backend, int32_t reject, const char * node_name) {
    std::lock_guard<std::mutex> lock(g_violation_mutex);
    g_violation_op      = ggml_op;
    g_violation_backend = backend;
    g_violation_reject  = reject;
    std::snprintf(g_violation_node, sizeof(g_violation_node), "%s", node_name ? node_name : "");
}
} // namespace

void ggml_rir_violation_reset(void) {
    std::lock_guard<std::mutex> lock(g_violation_mutex);
    g_violation_op      = nullptr;
    g_violation_backend = 0;
    g_violation_reject  = GGML_RIR_MATCHED;
    g_violation_node[0] = '\0';
}

ggml_rir_violation ggml_rir_violation_last(void) {
    std::lock_guard<std::mutex> lock(g_violation_mutex);
    ggml_rir_violation out;
    out.ggml_op   = g_violation_op;
    out.backend   = g_violation_backend;
    out.reject    = g_violation_reject;
    out.node_name = g_violation_node;
    return out;
}

void ggml_rir_violation_format(char * buf, size_t n_buf) {
    if (buf == nullptr || n_buf == 0) {
        return;
    }
    const ggml_rir_violation v = ggml_rir_violation_last();
    if (v.ggml_op == nullptr) {
        buf[0] = '\0';
        return;
    }
    std::snprintf(buf, n_buf,
        "RIR mode 'require': %s has no eligible variant on %s (reject=%s, node='%s')",
        v.ggml_op, ggml_rir_backend_name(v.backend), ggml_rir_reject_name(v.reject),
        v.node_name ? v.node_name : "");
}

bool ggml_rir_preflight_graph(uint8_t backend, struct ggml_cgraph * cgraph,
                              ggml_rir_device_check_fn device_check, void * device_ctx) {
    // Only `require` promises anything about a node that was never encoded, and
    // the *dispatch* mode is the one that matters: a probe forcing native must
    // not be failed by a preflight it deliberately bypassed.
    if (ggml_rir_get_dispatch_mode() != GGML_RIR_MODE_REQUIRE || cgraph == nullptr) {
        return true;
    }
    const int n_nodes = ggml_graph_n_nodes(cgraph);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(cgraph, i);
        if (node == nullptr || ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
            continue;
        }
        if (!ggml_rir_op_is_targeted((int32_t) node->op, (rir_backend) backend)) {
            // Not targeted: either no variant, or one registered for observation
            // only. In both cases the native kernel *is* the contract here, and
            // `require` promises nothing about a variant it will never encode.
            continue;
        }
        const rir_variant_desc * variant =
            ggml_rir_find_variant_for_node((int32_t) node->op, (rir_backend) backend, node);
        const int32_t why = ggml_rir_evaluate(backend, node, device_check, device_ctx);
        if (why != GGML_RIR_MATCHED) {
            record_violation(variant->ggml_op, backend, why, node->name);
            return false;
        }
    }
    return true;
}

// --- graph census -----------------------------------------------------------

namespace {
// Same append-only discipline as the site rows: `n_rows` only grows, a
// published row is never moved, so a reader scans `[0, n_rows)` without a lock.
// Unlike the site rows this one is written from a graph walk rather than a
// dispatch site, so the whole update takes the mutex — it runs once per node
// per graph, not once per encoded kernel, and correctness of the shape table is
// worth more here than the last microsecond.
struct census_row {
    const char *          ggml_op = nullptr;
    uint8_t               backend = 0;
    uint64_t              n_nodes = 0;
    uint64_t              n_elements = 0;
    uint64_t              n_bytes = 0;
    bool                  registered = false;
    uint32_t              n_shapes = 0;
    ggml_rir_census_shape shapes[GGML_RIR_MAX_CENSUS_SHAPES] = {};
    uint64_t              shapes_overflow = 0;
};

census_row            g_census[GGML_RIR_MAX_CENSUS_ROWS];
std::atomic<uint32_t> g_n_census{0};
std::mutex            g_census_mutex;

// Resolved once. The census is a diagnostic, so it must cost nothing at all
// when nobody asked for it — not even a getenv per node.
bool census_enabled_uncached() {
    const char * v = std::getenv("RETRO_RIR_CENSUS");
    return v != nullptr && v[0] != '0' && v[0] != '\0';
}

// Requires g_census_mutex. `ggml_op` is `ggml_op_name`, i.e. static storage.
census_row * find_or_append_census(const char * ggml_op, uint8_t backend) {
    const uint32_t n = g_n_census.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
        if (g_census[i].backend == backend && g_census[i].ggml_op != nullptr &&
            (g_census[i].ggml_op == ggml_op || std::strcmp(g_census[i].ggml_op, ggml_op) == 0)) {
            return &g_census[i];
        }
    }
    if (n >= GGML_RIR_MAX_CENSUS_ROWS) {
        // ggml has fewer ops than this cap, so reaching it means the cap is
        // wrong rather than the graph exotic. Dropping the row keeps every
        // other row exact; folding it into a neighbour would not.
        return nullptr;
    }
    g_census[n].ggml_op = ggml_op;
    g_census[n].backend = backend;
    g_n_census.store(n + 1, std::memory_order_release);
    return &g_census[n];
}

// Requires g_census_mutex.
void census_note_shape(census_row & row, const ggml_tensor * node) {
    for (uint32_t i = 0; i < row.n_shapes; ++i) {
        ggml_rir_census_shape & s = row.shapes[i];
        if (s.type == (int32_t) node->type && s.ne[0] == node->ne[0] && s.ne[1] == node->ne[1] &&
            s.ne[2] == node->ne[2] && s.ne[3] == node->ne[3]) {
            s.n_nodes++;
            return;
        }
    }
    if (row.n_shapes >= GGML_RIR_MAX_CENSUS_SHAPES) {
        row.shapes_overflow++;
        return;
    }
    ggml_rir_census_shape & s = row.shapes[row.n_shapes++];
    for (int d = 0; d < 4; ++d) {
        s.ne[d] = node->ne[d];
    }
    s.type    = (int32_t) node->type;
    s.n_nodes = 1;
}

// The ranking, printed at exit. Sorted by bytes moved, because that is the
// first-order cost model of every op in this graph that is not a matmul, and
// the column that says which shapes to hand to the isolated bench next. It is
// deliberately not called a time: nothing here executed anything.
void census_print(std::FILE * out) {
    const uint32_t n = ggml_rir_census_count();
    if (n == 0) {
        return;
    }
    // Index sort, so the append-only table is never reordered.
    uint32_t order[GGML_RIR_MAX_CENSUS_ROWS];
    for (uint32_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    std::sort(order, order + n, [](uint32_t a, uint32_t b) {
        if (g_census[a].n_bytes != g_census[b].n_bytes) {
            return g_census[a].n_bytes > g_census[b].n_bytes;
        }
        return g_census[a].n_nodes > g_census[b].n_nodes;
    });
    uint64_t total_bytes = 0;
    uint64_t total_nodes = 0;
    for (uint32_t i = 0; i < n; ++i) {
        total_bytes += g_census[i].n_bytes;
        total_nodes += g_census[i].n_nodes;
    }
    std::fprintf(out, "ggml-rir: census nodes=%llu bytes=%llu rows=%u\n",
        (unsigned long long) total_nodes, (unsigned long long) total_bytes, n);
    for (uint32_t k = 0; k < n; ++k) {
        const census_row & row = g_census[order[k]];
        const double share = total_bytes ? 100.0 * (double) row.n_bytes / (double) total_bytes : 0.0;
        std::fprintf(out, "ggml-rir: census %s/%s nodes=%llu bytes=%llu (%.2f%%) elems=%llu %s",
            row.ggml_op ? row.ggml_op : "?",
            ggml_rir_backend_name(row.backend),
            (unsigned long long) row.n_nodes,
            (unsigned long long) row.n_bytes,
            share,
            (unsigned long long) row.n_elements,
            row.registered ? "registered" : "uncovered");
        for (uint32_t s = 0; s < row.n_shapes; ++s) {
            const ggml_rir_census_shape & sh = row.shapes[s];
            std::fprintf(out, " [%s,%lldx%lldx%lldx%lld]=%llu",
                ggml_type_name((ggml_type) sh.type),
                (long long) sh.ne[0], (long long) sh.ne[1],
                (long long) sh.ne[2], (long long) sh.ne[3],
                (unsigned long long) sh.n_nodes);
        }
        if (row.shapes_overflow != 0) {
            // Not cosmetic: it says no single shape represents this op, so a
            // per-shape measurement of it would be a sample and not a total.
            std::fprintf(out, " [other]=%llu", (unsigned long long) row.shapes_overflow);
        }
        std::fprintf(out, "\n");
    }
}
} // namespace

bool ggml_rir_census_enabled(void) {
    static const bool enabled = census_enabled_uncached();
    return enabled;
}

void ggml_rir_census_graph(uint8_t backend, const struct ggml_cgraph * cgraph) {
    if (!ggml_rir_census_enabled() || cgraph == nullptr) {
        return;
    }
    // `ggml_graph_n_nodes`/`ggml_graph_node` take a mutable graph; the census
    // only reads, and taking a const pointer is what says so at the call site.
    ggml_cgraph * g = const_cast<ggml_cgraph *>(cgraph);
    const int n_nodes = ggml_graph_n_nodes(g);
    std::lock_guard<std::mutex> lock(g_census_mutex);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(g, i);
        if (node == nullptr || ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
            // The same filter the preflight applies: a view or a reshape is a
            // relabelling, not a kernel, and counting it would rank ops by how
            // often the graph builder renamed a tensor.
            continue;
        }
        census_row * row = find_or_append_census(ggml_op_name(node->op), backend);
        if (row == nullptr) {
            continue;
        }
        row->n_nodes++;
        row->n_elements += (uint64_t) ggml_nelements(node);
        row->n_bytes    += (uint64_t) ggml_nbytes(node);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] != nullptr) {
                row->n_bytes += (uint64_t) ggml_nbytes(node->src[s]);
            }
        }
        // Whether the pair is registered at all, not whether it is promoted:
        // the ranking exists to pick what to write next, so "already has a
        // variant" is the column that removes a row from the candidate list.
        row->registered =
            ggml_rir_find_variant_for_op((int32_t) node->op, (rir_backend) backend) != nullptr;
        census_note_shape(*row, node);
    }
}

uint32_t ggml_rir_census_count(void) {
    return g_n_census.load(std::memory_order_acquire);
}

ggml_rir_census_row ggml_rir_census_snapshot(uint32_t index) {
    ggml_rir_census_row out = {};
    if (index >= ggml_rir_census_count()) {
        return out;
    }
    std::lock_guard<std::mutex> lock(g_census_mutex);
    const census_row & row = g_census[index];
    out.ggml_op         = row.ggml_op;
    out.backend         = row.backend;
    out.n_nodes         = row.n_nodes;
    out.n_elements      = row.n_elements;
    out.n_bytes         = row.n_bytes;
    out.registered      = row.registered;
    out.n_shapes        = row.n_shapes;
    out.shapes_overflow = row.shapes_overflow;
    for (uint32_t i = 0; i < row.n_shapes && i < GGML_RIR_MAX_CENSUS_SHAPES; ++i) {
        out.shapes[i] = row.shapes[i];
    }
    return out;
}
