// retro delta: the CUDA half of the RIR dispatch contract (docs/CUDA_v1.md §C4).
//
// A fork-owned translation unit reached from `ggml-cuda.cu` by one line per
// site, per RETRO_FORK.md: nothing here is a hunk in an upstream file, and the
// upstream dispatcher gains an `if` and not forty lines.
//
// What the contract leaves to a backend is its queue, its buffers and its
// pipeline object. On CUDA that is the stream, the tensor pointers, and a
// function pointer — there is no descriptor set, no pipeline cache and no
// extension to negotiate (§3.1, §3.3). So this file is shorter than its Vulkan
// counterpart, and everything it does *not* do — the mode, the policy, the site
// attribution, the counters, the `require` abort, and the whole portable half of
// the contract — is `ggml-rir.cpp`'s, once, for three backends.

#include "ggml-cuda-rir.h"

#include "../common.cuh"

#include "../../ggml-rir/ggml-rir.h"

#include <cstring>
#include <mutex>

// The two device limits §3.3 leaves as *real* constraints on CUDA, read once
// per device instead of per node.
//
// `smpb` comes from `ggml_cuda_info()`, which already holds it. The grid bounds
// and the block ceiling are queried rather than written down: 65 535 on y and z
// has been true since sm_30 and 1 024 threads per block since sm_20, but a
// constant this file believes is a constant nothing checks — and §C3 settled
// that an unevaluated requirement is a rejection, not a shrug.
struct rir_cuda_limits {
    int max_grid[3];
    int max_threads_per_block;
};

static const rir_cuda_limits & rir_cuda_limits_of(int device) {
    static rir_cuda_limits cache[GGML_CUDA_MAX_DEVICES] = {};
    static std::once_flag   initialized[GGML_CUDA_MAX_DEVICES];
    // Separate contexts may evaluate graphs concurrently on the same device.
    // `call_once` both prevents concurrent writes and publishes the completely
    // initialized limits before another thread returns this slot.
    std::call_once(initialized[device], [device] {
        rir_cuda_limits l = {};
        CUDA_CHECK(cudaDeviceGetAttribute(&l.max_grid[0], cudaDevAttrMaxGridDimX, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l.max_grid[1], cudaDevAttrMaxGridDimY, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l.max_grid[2], cudaDevAttrMaxGridDimZ, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l.max_threads_per_block,
                                          cudaDevAttrMaxThreadsPerBlock, device));
        cache[device] = l;
    });
    return cache[device];
}

// Every scalar a generated kernel dereferences is at most four bytes wide — an
// F32 element, or the `uint32_t` a packed header is read through; a vector
// lowering reads its `w` components one at a time from a byte address, so it
// asks for no wider alignment than a scalar one (docs/CUDA_v1.md §5.1).
//
// The kernel has no misalignment parameter, so the tensor pointer must be exact.
// In practice a CUDA buffer is 256-byte aligned and a view offset is a multiple
// of the type size; this is the check that says so rather than assuming it.
static bool ggml_cuda_rir_pointer_ok(const ggml_tensor * t) {
    return ((uintptr_t) t->data % 4u) == 0;
}

int32_t ggml_cuda_rir_device_check(void * device_ctx, const ggml_tensor * node) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) device_ctx;

    // Per node, not per op: the variant whose launcher and grid are checked here
    // must be the one the dispatch site will launch, and with more than one
    // lowering per pair that is a function of this node's shape.
    const rir_variant_desc * v = ggml_rir_find_variant_for_node(node->op, RIR_BACKEND_CUDA, node);
    if (v == nullptr) {
        return GGML_RIR_REJECT_WRONG_OP;
    }
    // The stub the fork carries, by artifact. A build whose `rir/*.cu` glob
    // picked up nothing has no launcher, and that is a pipeline rejection in
    // exactly the sense Vulkan gives the word: the object this node needs was
    // never created.
    if (ggml_cuda_rir_launcher(v->artifact) == nullptr) {
        return GGML_RIR_REJECT_PIPELINE;
    }

    const rir_cuda_limits & lim  = rir_cuda_limits_of(ctx->device);
    const auto &            dev = ggml_cuda_info().devices[ctx->device];

    uint32_t grid[3];
    ggml_rir_grid(v, node, grid);
    for (int d = 0; d < 3; ++d) {
        if (grid[d] > (uint32_t) lim.max_grid[d]) {
            return GGML_RIR_REJECT_DEVICE_GRID;
        }
    }
    // The block the launch stub hard-codes, and the shared storage the lowering
    // declared — the two budgets `KernelNeeds` publishes (§5.4). Both are
    // properties of the compiled kernel, so a device that cannot run them is a
    // build that cannot run here, which the site says rather than answering
    // elsewhere.
    const uint32_t threads = v->workgroup[0] * v->workgroup[1] * v->workgroup[2];
    if (threads > (uint32_t) lim.max_threads_per_block) {
        return GGML_RIR_REJECT_DEVICE_GRID;
    }
    if (v->shared_bytes > dev.smpb) {
        return GGML_RIR_REJECT_MISSING_FEATURE;
    }

    for (uint32_t b = 0; b < v->n_bindings; ++b) {
        const ggml_tensor * t = ggml_rir_binding_tensor(v, b, node);
        if (t == nullptr) {
            return GGML_RIR_REJECT_WRONG_OP;
        }
        if (!ggml_cuda_rir_pointer_ok(t)) {
            return GGML_RIR_REJECT_DEVICE_ALIGNMENT;
        }
    }
    return GGML_RIR_MATCHED;
}

static void ggml_cuda_rir_launch(ggml_backend_cuda_context * ctx, const rir_variant_desc * v,
                                 const ggml_tensor * node) {
    ggml_cuda_rir_launch_fn launch = ggml_cuda_rir_launcher(v->artifact);
    if (launch == nullptr) {
        GGML_ABORT("ggml-rir: %s has no launch stub in this build", v->variant_id);
    }

    // Same buffers and same stream as any native op — the adapter passes ggml's
    // own allocations, never a host copy. Which tensor sits at which binding is
    // the registry's answer, not this function's.
    void * bufs[RIR_MAX_BINDINGS] = {};
    for (uint32_t b = 0; b < v->n_bindings; ++b) {
        bufs[b] = ggml_rir_binding_tensor(v, b, node)->data;
    }

    // The bytes the generated `rir_<kernel>_params` expects, laid out by the
    // portable half. Its size is checked at *compile* time of the fork by the
    // stub's `static_assert` (§5.3), so this runtime check is the one Vulkan
    // needs and CUDA keeps only against a registry that disagrees with itself.
    uint32_t params[RIR_PUSH_CONSTANT_CAPACITY / sizeof(uint32_t)] = {};
    if (!ggml_rir_fill_params(v, node, params, v->push_constant_bytes)) {
        GGML_ABORT("ggml-rir: %s params layout mismatch (registry says %u bytes)", v->variant_id,
                   v->push_constant_bytes);
    }

    uint32_t grid[3];
    ggml_rir_grid(v, node, grid);
    launch(bufs, params, grid, ctx->stream());
}

bool ggml_cuda_rir_try(ggml_backend_cuda_context * ctx, const ggml_tensor * node) {
    const rir_variant_desc * v =
        ggml_rir_dispatch_begin(RIR_BACKEND_CUDA, node, ggml_cuda_rir_device_check, ctx);
    if (v == nullptr) {
        return false;  // native, already counted
    }
    ggml_cuda_rir_launch(ctx, v, node);
    ggml_rir_dispatch_end(v);
    return true;
}

void ggml_cuda_rir_only(ggml_backend_cuda_context * ctx, const ggml_tensor * node) {
    if (!ggml_cuda_rir_try(ctx, node)) {
        GGML_ABORT("ggml-rir: %s sans variante dispatchée et sans natif (cuda)",
                   ggml_op_name(node->op));
    }
}

bool ggml_cuda_rir_graph_begin(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph) {
    // Under RETRO_RIR_CENSUS, rank the ops of the *real* graph by node count and
    // traffic — including the ones RIR does not cover, which is the only place
    // they are visible (docs/CUDA_v1.md §C1). Off by default, and it costs
    // nothing when nobody asked for it.
    ggml_rir_census_graph(RIR_BACKEND_CUDA, cgraph);
    // And under `require`, refuse the whole graph before anything is launched
    // rather than falling back to a native kernel node by node.
    return ggml_rir_preflight_graph(RIR_BACKEND_CUDA, (ggml_cgraph *) cgraph,
                                    ggml_cuda_rir_device_check, ctx);
}
