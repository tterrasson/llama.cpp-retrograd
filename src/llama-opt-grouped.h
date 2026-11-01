#pragma once

// retro delta: the grouped softmax cross-entropy, the loss of a model whose
// answer is read from its embeddings output - a decision head reading one
// score per option, a reranker reading one score per document. Fork-owned:
// the graph and its inputs; the training step that drives them is
// llama_context::opt_step_grouped.

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

// The masked score of a padded option. Large enough to vanish in the softmax,
// finite so that a zero target times it is zero and not NaN.
constexpr float LLAMA_OPT_GROUPED_MASK = -1.0e4f;

struct llama_opt_grouped_loss {
    // I32 [n_opt_max*n_groups]: flat row*n_embd_out + column, or the output row
    // of each option for a pointer loss
    ggml_tensor * index   = nullptr;
    ggml_tensor * pointer = nullptr; // I32 [n_groups], output row of each query; pointer loss only
    ggml_tensor * mask    = nullptr; // F32 [n_opt_max, n_groups], 0 or LLAMA_OPT_GROUPED_MASK
    ggml_tensor * targets = nullptr; // F32 [n_opt_max, n_groups]
    ggml_tensor * loss    = nullptr; // F32 scalar, the mean over groups
};

// embd is the embeddings output [n_embd_out, n_outputs]. The graph reads one
// score per option with a single get_rows over the flattened output, so every
// op on the path has a backward rule:
//   reshape -> get_rows(index) -> reshape [n_opt_max, n_groups]
//           -> scale(inv_temperature) -> add(mask) -> cross_entropy_loss(targets)
// With `pointer`, the score of an option is the product of the first half of
// its query's output row with the second half of its own (llama_grouped_target
// pointer_rows), from ops that all have a backward rule:
//   get_rows(index) -> second half [d, n_opt_max, n_groups]
//   get_rows(pointer) -> first half [d, 1, n_groups]
//   mul (broadcast over the options) -> sum_rows -> reshape [n_opt_max, n_groups]
//           -> scale(inv_temperature/sqrt(d)) -> add(mask) -> cross_entropy_loss(targets)
llama_opt_grouped_loss llama_opt_build_grouped_loss(
        ggml_context * ctx,
        ggml_tensor  * embd,
        uint32_t       n_opt_max,
        uint32_t       n_groups,
        float          inv_temperature,
        bool           pointer = false);

// The host values of index, mask and (for a pointer target) pointer for one
// target. output_row[i] is the output row of batch token i, -1 if the token
// has no output. Returns false with a reason on a target that does not fit the
// batch.
bool llama_opt_grouped_inputs(
        const llama_grouped_target & target,
        const std::vector<int32_t> & output_row,
        int64_t                      n_embd_out,
        std::vector<int32_t>       & index,
        std::vector<float>         & mask,
        std::vector<int32_t>       & pointer,
        std::string                & error);

// Nodes the grouped loss adds on top of the model graph (for the pointer loss:
// two get_rows, two views and conts, the reshapes, mul, sum_rows, scale, add,
// cross_entropy_loss and their inputs), with room to spare.
constexpr size_t LLAMA_OPT_GROUPED_EXTRA_NODES = 32;
