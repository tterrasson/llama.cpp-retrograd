// retro delta: RIR integration support — see ggml-rir.h.
#include "ggml-rir.h"

// The generated constant-buffer layouts, included *here* and nowhere in a
// backend: this is the unit that owns the check that the capacity the
// backends stage into still holds every kernel (docs/INT_RIR_V3.md §R0).
#include "rir_kernel_params.h"

#include "ggml-impl.h"

// The canonical quantized-format table, for the one thing this unit needs from
// it: the ggml_type a registry dtype spelling denotes.
#include "ggml-retro-quant.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

static_assert(RIR_MAX_PUSH_CONSTANT_BYTES <= RIR_PUSH_CONSTANT_CAPACITY,
              "un kernel demande plus de constantes que la capacité publiée par ggml-rir.h");

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
    // Absent is `prefer` since docs/INT_RIR_V4.md §P6: the promoted pairs are
    // the build's own dispatch path, not an opt-in experiment. `off` stays a
    // spelling, so the bench and the diagnostic keep their native run — for the
    // pairs that still have a native kernel to run.
    if (v == nullptr) {
        return GGML_RIR_MODE_PREFER;
    }
    if (std::strcmp(v, "off") == 0) {
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

// Declared here, defined with the rest of the registry lookups below: the site
// lines publish the declared domain next to the rejects it explains, and that
// printing happens before the lookup is defined.
uint32_t assumed_domain_of(const rir_op_policy * policies, uint32_t n_policies,
                           const char * ggml_op_spelling, uint8_t backend);

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
            // The coverage rate and the domain that explains it, on the same
            // line as the counters that produced them (docs/INT_RIR_V4.md §P4).
            //
            // Printed per site and never derived from the aggregate, which
            // cannot answer this question: a shared encoder — Metal's
            // `ggml_metal_op_unary` serves a dozen ops, `ggml_metal_op_bin`
            // takes SUB and DIV too — pushes nodes of *unregistered* ops
            // through `ops_seen` with no site to attribute them to. A rate
            // computed from the total would count those against a kernel that
            // was never asked about them.
            const uint64_t seen = row.counters.ops_seen;
            std::fprintf(stderr, " coverage=%llu/%llu",
                (unsigned long long) row.counters.rir_dispatched,
                (unsigned long long) seen);
            const uint32_t domain =
                assumed_domain_of(rir_op_policies, rir_op_policy_count,
                                  row.ggml_op ? row.ggml_op : "", row.backend);
            std::fprintf(stderr, " domain=");
            if (domain == 0) {
                // "none" rather than an empty field: a consumer must be able to
                // tell "claims the whole op" from "this build prints no domain".
                std::fprintf(stderr, "none");
            } else {
                const char * sep = "";
                for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
                    if (domain & (1u << i)) {
                        std::fprintf(stderr, "%s%s", sep, ggml_rir_reject_name(i));
                        sep = "|";
                    }
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
        case GGML_RIR_REJECT_DEVICE_GRID:
        case GGML_RIR_REJECT_DEVICE_ALIGNMENT:
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
        "policy_native", "device_grid", "device_alignment", "op_variant",
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
    // Dimension 0 of a quantized tensor advances by **block**, not by element:
    // `ne[0]` counts logical elements while `nb[0]` is the block size. The
    // largest index that multiplies `nb[0]` is therefore `ne[0]/blck - 1`, and
    // using `ne[0] - 1` overstated the span by a factor of `block_elements` —
    // enough to refuse a perfectly addressable q4_K row with `integer_range`.
    // Unreachable until §P5 gave a quantized binding to the portable contract,
    // which is why it stood.
    const int64_t blck = ggml_blck_size(t->type);
    uint64_t last = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        int64_t units = t->ne[d];
        if (d == 0 && blck > 1) {
            units = (units + blck - 1) / blck;
        }
        const uint64_t n = units > 0 ? (uint64_t) (units - 1) : 0;
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
// the per-node choice is a scan that stops at the first fit. It is bounded by
// construction: a pair publishes one lowering per shape regime and one per
// `src0` dtype it can read (docs/INT_RIR_V4.md §P5), not one per shape.
//
// Sixteen and not eight since FUTURE V1 §5: `OUT_PROD` reads the ten standard
// quantized formats plus two, and its F32 fallback, which is thirteen rows for
// one pair. The bound is still a bound and still aborts rather than dropping a
// row — a variant the table cannot hold is one that would never be selected,
// which is worse than a loud failure at startup.
//
// Forty-eight since docs/CUDA_v1.md §C6, for the same pair that raised it to
// thirty-two: `GGML_OP_UNARY` on CUDA carries a third lowering — the flattened
// dispatch — over fourteen members, so forty-two rows resolve one node.
constexpr uint32_t RIR_MAX_PAIR_VARIANTS = 48;

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

// The declared domain of the same row. Absent from the table means no claim was
// made at all, and an empty mask is then the honest answer: nothing is
// published, so nothing is excused.
uint32_t assumed_domain_of(const rir_op_policy * policies, uint32_t n_policies,
                           const char * ggml_op_spelling, uint8_t backend) {
    for (uint32_t i = 0; i < n_policies; ++i) {
        const rir_op_policy & p = policies[i];
        if (p.backend == backend && std::strcmp(p.ggml_op, ggml_op_spelling) == 0) {
            return p.assumed_domain;
        }
    }
    return 0;
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
        // The claim filters, when a node is in hand. A specialization that does
        // not fit this node is not a worse candidate, it is not a candidate:
        // its schedule was measured on the shapes it claims and loses — often
        // by a factor of several — everywhere else (docs/INT_RIR_V3.md §R1),
        // and a row whose dtype does not match would reinterpret the bytes.
        if (node != nullptr
                && (!ggml_rir_variant_fits_shape(&v, node) || !ggml_rir_variant_fits_layout(&v, node)
                    || !ggml_rir_variant_fits_dtype(&v, node))) {
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
        v.bindings[0].dtype  = "f32";
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
    probe.op   = (enum ggml_op) op;
    probe.type = GGML_TYPE_F32;
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

    // --- per-dtype arbitration (docs/INT_RIR_V4.md §P5) ---------------------
    //
    // The same mechanism, on the third claim: `OUT_PROD` publishes one row per
    // `src0` dtype it can read, and what picks between them is the node's own
    // type. Exercised synthetically for the reason the shape rules were — with
    // the shipped table alone, "the dtype was checked" and "the F32 row
    // happened to come first" are indistinguishable.
    auto typed = [&](const char * id, const char * dtype) {
        rir_variant_desc v = row(id, 80, backend);
        v.n_bindings = 1;
        v.bindings[0].source = -1;
        v.bindings[0].dtype  = dtype;
        return v;
    };
    // Deliberately in this order and at equal priority: the quantized row comes
    // first, so anything picking it for an F32 node is the dtype filter missing
    // and not the table order saving us.
    const rir_variant_desc by_dtype[] = {typed("quant", "q4_0"), typed("plain", "f32")};
    auto picked_type = [&](ggml_type t) {
        probe.type  = t;
        probe.ne[1] = 1;
        const rir_variant_desc * v = ggml_rir_select_variant(
                by_dtype, 2, prefer, 1, op, backend, &probe);
        return v ? v->variant_id : nullptr;
    };
    // bit 11: an F32 node takes the F32 row, past a higher-placed quantized one.
    if (!is(picked_type(GGML_TYPE_F32), "plain")) { failures |= 1u << 11; }
    // bit 12: a quantized node takes the row that claims its format.
    if (!is(picked_type(GGML_TYPE_Q4_0), "quant")) { failures |= 1u << 12; }
    // bit 13: a format no row claims selects nothing here — the caller's own
    // fallback then takes it and the contract rejects it with `dtype`, which is
    // the one outcome that must never be a reinterpreted read.
    if (picked_type(GGML_TYPE_Q6_K) != nullptr) { failures |= 1u << 13; }
    probe.type = GGML_TYPE_F32;

    // --- the vectorized fold claim (docs/FUTURE_V1.md §6) -------------------
    //
    // The fourth claim, and the only one that is a *relation* between two axes:
    // a vec4 variant reading an operand through a fold needs that fold to be
    // the identity **and** that operand's own contiguous stride to be one
    // element. Both halves are exercised here rather than on the shipped table,
    // for the reason the shape rules were: with `add`/`add_repeat` alone,
    // "the claim was evaluated" and "the claim happened to hold" look alike.
    //
    // Two axes: `col` (vectorized, read from dim 0 of src0) and `col_b`
    // (dim 0 of src1, folded into `col`).
    auto folded = [&](const char * id, uint8_t width) {
        rir_variant_desc v = row(id, width > 1 ? 90 : 80, backend);
        v.n_bindings = 2;
        v.bindings[0].source = 0;
        v.bindings[0].dtype  = "f32";
        v.bindings[0].rank   = 1;
        v.bindings[0].elem_bytes = 4;
        v.bindings[1].source = 1;
        v.bindings[1].dtype  = "f32";
        v.bindings[1].rank   = 1;
        v.bindings[1].elem_bytes = 4;
        v.n_axes  = 2;
        v.axes[0] = {"col",   {0, -1, -1, -1}, -1};
        v.axes[1] = {"col_b", {-1, 0, -1, -1},  0};
        v.dispatch[0] = {0, 1};
        v.vector_width = width;
        return v;
    };
    const rir_variant_desc by_fold[] = {folded("vec4", 4), folded("scalar", 1)};

    ggml_tensor src0 = {};
    ggml_tensor src1 = {};
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        src0.ne[d] = 1;
        src1.ne[d] = 1;
        src0.nb[d] = 4;
        src1.nb[d] = 4;
    }
    src0.type = GGML_TYPE_F32;
    src1.type = GGML_TYPE_F32;
    src0.ne[0] = 8;
    ggml_tensor folded_node = {};
    folded_node.op   = (enum ggml_op) op;
    folded_node.type = GGML_TYPE_F32;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        folded_node.ne[d] = 1;
    }
    folded_node.ne[0] = 8;
    folded_node.src[0] = &src0;
    folded_node.src[1] = &src1;
    auto picked_fold = [&]() {
        const rir_variant_desc * v = ggml_rir_select_variant(
                by_fold, 2, prefer, 1, op, backend, &folded_node);
        return v ? v->variant_id : nullptr;
    };

    // bit 14: the fold is the identity and src1 is contiguous — the vectorized
    // row is selected, which is what makes the claim worth publishing at all.
    src1.ne[0] = 8;
    src1.nb[0] = 4;
    if (!is(picked_fold(), "vec4")) { failures |= 1u << 14; }
    // bit 15: a real repeat on the contiguous axis. Four consecutive indices
    // are no longer four consecutive addresses, so the scalar row must take it.
    src1.ne[0] = 2;
    if (!is(picked_fold(), "scalar")) { failures |= 1u << 15; }
    // bit 16: the same extent, but a **permuted** src1. This is the half a
    // fold-only check misses: the operand is indexed by `col_b`, so the stride
    // test on `col` never looks at it, and the vec4 row would read four
    // consecutive addresses against a stride that is not one element.
    src1.ne[0] = 8;
    src1.nb[0] = 8;
    if (!is(picked_fold(), "scalar")) { failures |= 1u << 16; }

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
        // `fits_axes` is here since FUTURE V1 §6, and it is what makes two
        // kernels of one op selectable by *shape*. Until then no two variants
        // of a pair disagreed about which extents must match, so the shape half
        // of the contract could wait for `evaluate_portable`. Now `mul` claims
        // three equal shapes and `mul_repeat` claims a divisor: a repeated node
        // that reached `mul` would be refused with `shape` and leave for the
        // native kernel, never trying the variant written for it.
        if (ggml_rir_variant_fits_shape(sel.candidates[i], node)
                && ggml_rir_variant_fits_axes(sel.candidates[i], node)
                && ggml_rir_variant_fits_layout(sel.candidates[i], node)
                && ggml_rir_variant_fits_dtype(sel.candidates[i], node)
                && ggml_rir_variant_fits_op_variant(sel.candidates[i], node)) {
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

uint32_t ggml_rir_op_assumed_domain(const char * ggml_op_spelling, rir_backend backend) {
    if (ggml_op_spelling == nullptr || (int) backend >= RIR_N_BACKENDS) {
        return 0;
    }
    return assumed_domain_of(rir_op_policies, rir_op_policy_count, ggml_op_spelling,
                             (uint8_t) backend);
}

bool ggml_rir_op_is_targeted(int32_t ggml_op, rir_backend backend) {
    return ggml_rir_find_variant_for_op(ggml_op, backend) != nullptr
        && ggml_rir_op_policy(ggml_op, backend) == RIR_POLICY_PREFER_GENERATED;
}

bool ggml_rir_op_native_retired(const char * ggml_op_spelling, rir_backend backend) {
    if (ggml_op_spelling == nullptr || (int) backend >= RIR_N_BACKENDS) {
        return false;
    }
    // Both spellings are accepted, and deliberately: the registry writes
    // "GGML_OP_ADD", a site row carries the same string, and a dispatch site
    // holds `ggml_op_name`, which is "ADD". One lookup for all three beats
    // three call sites remembering which one they have.
    const char * want = strip_ggml_op_prefix(ggml_op_spelling);
    for (uint32_t i = 0; i < rir_op_policy_count; ++i) {
        const rir_op_policy & p = rir_op_policies[i];
        if (p.backend == (uint8_t) backend &&
                std::strcmp(strip_ggml_op_prefix(p.ggml_op), want) == 0) {
            return p.native_retired != 0;
        }
    }
    return false;
}

bool ggml_rir_supports_op(uint8_t backend, const struct ggml_tensor * node) {
    if (node == nullptr || backend >= RIR_N_BACKENDS) {
        return false;
    }
    // The configured mode, not the dispatch mode: `ggml_rir_set_force_native`
    // is a probe knob that lowers what a *site* may encode, and letting it move
    // `supports_op` would move the graph split under a running probe — the node
    // would leave the backend instead of taking its native path. A pair whose
    // native is retired has no native path to force, and that is refused where
    // the forcing is asked for, not silently here.
    if (ggml_rir_get_mode() < GGML_RIR_MODE_PREFER) {
        return false;
    }
    if (ggml_rir_op_policy((int32_t) node->op, (rir_backend) backend)
            != RIR_POLICY_PREFER_GENERATED) {
        return false;
    }
    const rir_variant_desc * v =
        ggml_rir_find_variant_for_node((int32_t) node->op, (rir_backend) backend, node);
    if (v == nullptr) {
        return false;
    }
    return ggml_rir_evaluate_portable(v, node) == GGML_RIR_MATCHED;
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
    if (std::strcmp(dtype, "bf16") == 0)     { return GGML_TYPE_BF16; }
    if (std::strcmp(dtype, "i32")  == 0)     { return GGML_TYPE_I32;  }
    // The quantized spellings are **expanded from the canonical table**, not
    // restated: `ggml_type_name(q4_K)` is what the registry prints, and the
    // same header generates both sides (docs/INT_RIR.md §7.2). Writing the
    // list here is how a format ends up known to one side and not the other.
    // F16 is a row of that table, so it needs no line of its own.
#define RIR_DTYPE_ROW(TYPE, BLK, NL, NAME, VKNAME) \
    if (std::strcmp(dtype, #NAME) == 0) { return TYPE; }
    GGML_RETRO_OUT_PROD_TYPES(RIR_DTYPE_ROW)
#undef RIR_DTYPE_ROW
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

// The number of points a flattened dispatch covers: the product of
// `ceil(extent / per_index)` over the axes the variant decomposes
// (docs/CUDA_v1.md §C6). Zero when the variant is not flattened, and
// **saturating** rather than wrapping: the caller compares it against a bound,
// and a wrapped product would compare small.
uint64_t flat_total(const rir_variant_desc * v, const struct ggml_tensor * node) {
    uint64_t total = 1;
    for (uint32_t i = 0; i < v->n_flat; ++i) {
        const int8_t a = v->flat[i].axis;
        if (a < 0 || (uint32_t) a >= v->n_axes) {
            return 0;
        }
        const uint32_t per = v->flat[i].per_workgroup ? v->flat[i].per_workgroup : 1;
        const int64_t extent = axis_extent(v, (uint32_t) a, node);
        if (extent <= 0) {
            return 0;
        }
        total *= ((uint64_t) extent + per - 1) / per;
        if (total > UINT64_MAX / 2) {
            return UINT64_MAX / 2;  // saturate: only comparisons follow
        }
    }
    return v->n_flat ? total : 0;
}

// The multiplier and the shift of an unsigned division by `d`, as
// `init_fastdiv_values` computes them in ggml-cuda/common.cuh and as
// `rir_lower::fastdiv_magic` computes them for the oracle
// (docs/CUDA_v1.md §C1.5). One formula, three implementations, and the parity
// harness is what keeps them equal.
void fastdiv_magic(uint32_t d, uint32_t * mp, uint32_t * sh) {
    uint32_t l = 0;
    while (l < 32 && ((uint64_t) 1 << l) < (uint64_t) d) {
        ++l;
    }
    *mp = (uint32_t) ((((uint64_t) 1 << 32) * (((uint64_t) 1 << l) - d)) / d + 1);
    *sh = l;
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

bool ggml_rir_variant_fits_layout(const rir_variant_desc * v, const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr) {
        return false;
    }
    if (v->vector_width <= 1) {
        return true;  // the scalar lowering accepts any stride the contract does
    }
    // The vectorized axis is the one walking grid dimension x — the contiguous
    // one, by construction of the lowering. Only the bindings this axis indexes
    // *at dimension 0* are read in vectors; a binding it indexes elsewhere (or
    // not at all) is read scalar and broadcast, and its stride is free.
    //
    // A flattened variant has no grid axis, and the contiguous one is the first
    // it decomposes into — `flat[0]`, fastest first (docs/CUDA_v1.md §C6). The
    // width means the same thing there and claims the same stride; what changes
    // is only where the axis is published.
    const int8_t a = v->n_flat > 0 ? v->flat[0].axis : v->dispatch[0].axis;
    if (a < 0 || (uint32_t) a >= v->n_axes) {
        return false;
    }
    // `w` consecutive elements have to be `w` consecutive addresses, which is
    // exactly what a unit contiguous stride says and nothing else does.
    auto unit_stride_at_dim0 = [&](uint32_t axis) {
        for (uint32_t b = 0; b < v->n_bindings; ++b) {
            if (v->axes[axis].dim[b] != 0) {
                continue;
            }
            const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
            if (t == nullptr) {
                return false;
            }
            if (v->bindings[b].elem_bytes == 0 || t->nb[0] != v->bindings[b].elem_bytes) {
                return false;
            }
        }
        return true;
    };
    if (!unit_stride_at_dim0(a)) {
        return false;
    }
    // The operand read through a **fold** owes both halves of that sentence
    // (docs/FUTURE_V1.md §6). An axis folded into the vectorized one replays
    // index `i` as `i % extent`, so it is read in vectors too — which needs the
    // fold to be the identity *and* its own binding to have a unit contiguous
    // stride. The first alone is not enough: a `src1` with the same `ne0` but a
    // permuted `nb[0]` is indexed by the folded axis, not by `a`, so the loop
    // above never looks at it and four consecutive indices would be read as
    // four consecutive addresses against its stride.
    for (uint32_t f = 0; f < v->n_axes; ++f) {
        if (v->axes[f].folded_of != (int8_t) a) {
            continue;
        }
        if (axis_extent(v, f, node) != axis_extent(v, a, node)) {
            return false;
        }
        if (!unit_stride_at_dim0(f)) {
            return false;
        }
    }
    return true;
}

bool ggml_rir_variant_fits_axes(const rir_variant_desc * v, const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr) {
        return false;
    }
    for (uint32_t a = 0; a < v->n_axes; ++a) {
        // A **folded** axis is the divisor of a repeat (docs/FUTURE_V1.md §6),
        // not a shared extent: what it owes is divisibility, which is
        // `ggml_can_repeat` and is checked just below. Requiring agreement here
        // would refuse the very shapes the repeating kernel exists to serve.
        const int8_t folded = v->axes[a].folded_of;
        if (folded >= 0) {
            if ((uint32_t) folded >= v->n_axes) {
                return false;  // a relation the registry could not resolve
            }
            const int64_t under = axis_extent(v, (uint32_t) folded, node);
            const int64_t over  = axis_extent(v, a, node);
            if (over <= 0 || under % over != 0) {
                return false;
            }
            continue;
        }
        int64_t extent = -1;
        for (uint32_t b = 0; b < v->n_bindings; ++b) {
            const int8_t dim = v->axes[a].dim[b];
            if (dim < 0) {
                continue;
            }
            const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
            if (t == nullptr) {
                return false;
            }
            if (extent < 0) {
                extent = t->ne[dim];
            } else if (t->ne[dim] != extent) {
                return false;
            }
        }
    }
    return true;
}

bool ggml_rir_variant_fits_dtype(const rir_variant_desc * v, const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr) {
        return false;
    }
    for (uint32_t b = 0; b < v->n_bindings; ++b) {
        const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
        if (t == nullptr || t->type != ggml_type_of_dtype(v->bindings[b].dtype)) {
            return false;
        }
    }
    return true;
}

bool ggml_rir_variant_fits_op_variant(const rir_variant_desc * v, const struct ggml_tensor * node) {
    if (v == nullptr || node == nullptr) {
        return false;
    }
    // A variant that claims no member serves every node of its op — which is
    // every op that is not a family, and is what keeps this check free for them.
    if (v->ggml_op_variant == nullptr) {
        return true;
    }
    if (node->op != GGML_OP_UNARY) {
        // The only family today. A registry row claiming a member on any other
        // op is a generator bug, and answering "no" is what turns it into a
        // published refusal instead of a wrong function silently computed.
        return false;
    }
    static const char prefix[] = "GGML_UNARY_OP_";
    const size_t n = sizeof(prefix) - 1;
    const char * want = std::strncmp(v->ggml_op_variant, prefix, n) == 0
                      ? v->ggml_op_variant + n : v->ggml_op_variant;
    // Compared against ggml's own spelling rather than a numeric value copied
    // into the registry: the enum belongs to ggml.h, and a second copy of it
    // would compile whichever way it drifted.
    return std::strcmp(want, ggml_unary_op_name(ggml_get_unary_op(node))) == 0;
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
        // A quantized binding is addressed by block: the generated code reads
        // `i / block_elements` and `i % block_elements`, so a row that is not a
        // whole number of blocks would send its tail into the next block. The
        // portable oracle already refuses this (`RejectReason::QuantBlock`);
        // saying it here is what makes the two sides agree on the same node
        // (docs/INT_RIR_V4.md §P5).
        if (v->bindings[b].block_elements > 1
                && t->ne[0] % (int64_t) v->bindings[b].block_elements != 0) {
            return GGML_RIR_REJECT_QUANT_BLOCK;
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

    // The layout half of a variant's claim. Reached only when selection handed
    // this row a node it does not fit — a shape-blind lookup, or a caller
    // naming the variant — and it must say `stride` rather than run.
    // **After** the bindings, and the order is the reason attributed to a node
    // rather than an accident (docs/FUTURE_V1.md §7). When no variant of the
    // family fits, selection hands back the pair's fallback — some *other*
    // member — so an F16 node would be refused as "wrong member" when what is
    // actually out of scope is its dtype. Checking the bindings first makes each
    // node carry the restriction that really excludes it, which is what the
    // declared-domain check compares against.
    if (!ggml_rir_variant_fits_op_variant(v, node)) {
        return GGML_RIR_REJECT_OP_VARIANT;
    }
    if (!ggml_rir_variant_fits_layout(v, node)) {
        return GGML_RIR_REJECT_STRIDE;
    }

    // One extent is passed per axis, for every binding that indexes it: the
    // shape half of the contract is exactly that agreement. It subsumes the
    // hand-written "same shape as dst" checks, and says *why* they were needed.
    if (!ggml_rir_variant_fits_axes(v, node)) {
        return GGML_RIR_REJECT_SHAPE;
    }

    // The flattened index has to be representable, and the bound is tighter
    // than `index_bits` says (docs/CUDA_v1.md §C6). The decomposition adds
    // `umulhi(n, mp)` to `n` in 32 bits, which is exact while `n < 2^31` and
    // wraps above it — so the claim this variant makes is not "the index fits
    // in 32 bits" but "it fits in 31". It is checked here, with the other
    // representability conditions, and it is what makes the flattened variant a
    // specialization with a fallback under it rather than a lowering that is
    // right for most shapes.
    if (v->n_flat > 0) {
        const uint64_t total = flat_total(v, node);
        if (total == 0 || total > 0x7fffffffull || total > index_max) {
            return GGML_RIR_REJECT_INTEGER_RANGE;
        }
    }

    // The grid the dispatcher will ask for, in the same integers.
    uint32_t grid[3];
    ggml_rir_grid(v, node, grid);
    if (v->n_flat > 0) {
        const uint32_t per = v->dispatch[0].per_workgroup ? v->dispatch[0].per_workgroup : 1;
        const uint64_t want = (flat_total(v, node) + per - 1) / per;
        if (want > index_max || want != grid[0] || grid[1] != 1 || grid[2] != 1) {
            return GGML_RIR_REJECT_INTEGER_RANGE;
        }
        return GGML_RIR_MATCHED;
    }
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
    // The flattened dispatch's own block, last, exactly as the generated shader
    // declares it (docs/CUDA_v1.md §C6): the total, then one (divisor, magic
    // multiplier, shift) triple per divisor — one fewer than there are axes,
    // the last index being the quotient itself.
    //
    // This is the only place a push constant is *computed* rather than copied,
    // and it is why the flattening is not free of the host: the reciprocal of a
    // runtime divisor cannot be a compile-time constant, and dividing on the
    // device is exactly the cost §C1.5 says the native does not pay.
    if (v->n_flat > 0) {
        const uint64_t total = flat_total(v, node);
        if (total == 0 || total > 0x7fffffffull) {
            return false;  // refused by `evaluate_portable` long before here
        }
        if (i < n_slots) {
            slot[i++] = (uint32_t) total;
        }
        for (uint32_t f = 0; f + 1 < v->n_flat && i + 2 < n_slots; ++f) {
            const uint32_t per = v->flat[f].per_workgroup ? v->flat[f].per_workgroup : 1;
            const int64_t extent = axis_extent(v, (uint32_t) v->flat[f].axis, node);
            if (extent <= 0) {
                return false;
            }
            const uint32_t d = (uint32_t) (((uint64_t) extent + per - 1) / per);
            uint32_t mp = 0;
            uint32_t sh = 0;
            fastdiv_magic(d, &mp, &sh);
            slot[i++] = d;
            slot[i++] = mp;
            slot[i++] = sh;
        }
    }
    // A layout that does not fill the buffer exactly is a generator/consumer
    // mismatch, and the tail would be whatever the caller's stack held.
    return i == n_slots;
}

void ggml_rir_grid(const rir_variant_desc * v, const struct ggml_tensor * node, uint32_t grid[3]) {
    // The flattened dispatch first (docs/CUDA_v1.md §C6): one dimension over
    // the whole parallel space. It is read before `dispatch[]` and not beside
    // it because such a variant has **no** grid axis — `dispatch[0].axis` is -1
    // — and the loop below would answer a single workgroup for the whole
    // tensor.
    if (v != nullptr && node != nullptr && v->n_flat > 0) {
        const uint32_t per = v->dispatch[0].per_workgroup ? v->dispatch[0].per_workgroup : 1;
        const uint64_t total = flat_total(v, node);
        grid[0] = (uint32_t) ((total + per - 1) / per);
        grid[1] = 1;
        grid[2] = 1;
        return;
    }
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
        //
        // A pair whose native is retired cannot legitimately be here under
        // `off`: `ggml_rir_supports_op` answers on the RIR contract alone for
        // it, so the scheduler placed the node on another backend long before a
        // site was reached. Arriving anyway means something lowered the mode
        // *after* the split — the force-native probe knob is the only thing
        // that can — and the honest answer is to say so, not to return nullptr
        // and let the caller invoke a kernel that no longer exists.
        if (node != nullptr && ggml_rir_op_native_retired(ggml_op_name(node->op), (rir_backend) backend)) {
            GGML_ABORT("ggml-rir: %s/%s a été dispatché sous mode=off alors que son natif "
                       "est retiré (docs/INT_RIR_V4.md §P6)",
                    ggml_op_name(node->op), ggml_rir_backend_name(backend));
        }
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
    // Same reasoning as the `off` branch above, on the other exit. There is no
    // native kernel to count here, so counting one would be a lie the coverage
    // report would then publish. What produced this is either a device-half
    // rejection — a build that cannot run on this machine, never a domain (§P4)
    // — or a portable rejection on a pair that declared no restriction at all,
    // which is the defect `assumed_domain` exists to make visible.
    if (ggml_rir_op_native_retired(ggml_op_name(node->op), (rir_backend) backend)) {
        GGML_ABORT("ggml-rir: %s/%s reject=%s et le natif est retiré : aucun repli "
                   "(docs/INT_RIR_V4.md §P6)",
                ggml_op_name(node->op), ggml_rir_backend_name(backend),
                ggml_rir_reject_name(why));
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
