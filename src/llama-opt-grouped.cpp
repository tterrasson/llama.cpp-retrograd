// retro delta: see llama-opt-grouped.h

#include "llama-opt-grouped.h"

#include "ggml.h"

#include <cmath>
#include <climits>
#include <limits>

llama_opt_grouped_loss llama_opt_build_grouped_loss(
        ggml_context * ctx,
        ggml_tensor  * embd,
        uint32_t       n_opt_max,
        uint32_t       n_groups,
        float          inv_temperature,
        bool           pointer) {
    llama_opt_grouped_loss out;

    out.index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n_opt_max*n_groups);
    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_opt_max, n_groups);
    out.targets = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_opt_max, n_groups);
    ggml_set_input(out.index);
    ggml_set_input(out.mask);
    ggml_set_input(out.targets);
    ggml_set_name(out.index, "grouped_index");
    ggml_set_name(out.mask, "grouped_mask");
    ggml_set_name(out.targets, "grouped_targets");

    ggml_tensor * scores = ggml_is_contiguous(embd) ? embd : ggml_cont(ctx, embd);
    if (pointer) {
        out.pointer = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_groups);
        ggml_set_input(out.pointer);
        ggml_set_name(out.pointer, "grouped_pointer");

        const int64_t d = scores->ne[0]/2;
        ggml_tensor * keys = ggml_get_rows(ctx, scores, out.index);
        keys = ggml_view_2d(ctx, keys, d, keys->ne[1], keys->nb[1], d*ggml_element_size(keys));
        keys = ggml_reshape_3d(ctx, ggml_cont(ctx, keys), d, n_opt_max, n_groups);
        ggml_tensor * queries = ggml_get_rows(ctx, scores, out.pointer);
        queries = ggml_view_2d(ctx, queries, d, queries->ne[1], queries->nb[1], 0);
        queries = ggml_reshape_3d(ctx, ggml_cont(ctx, queries), d, 1, n_groups);
        scores = ggml_sum_rows(ctx, ggml_mul(ctx, keys, queries));
        scores = ggml_reshape_2d(ctx, scores, n_opt_max, n_groups);
        scores = ggml_scale(ctx, scores, inv_temperature/std::sqrt((float) d));
    } else {
        scores = ggml_reshape_2d(ctx, scores, 1, ggml_nelements(scores));
        scores = ggml_get_rows(ctx, scores, out.index);
        scores = ggml_reshape_2d(ctx, scores, n_opt_max, n_groups);
        if (inv_temperature != 1.0f) {
            scores = ggml_scale(ctx, scores, inv_temperature);
        }
    }
    scores = ggml_add(ctx, scores, out.mask);
    ggml_set_name(scores, "grouped_scores");

    out.loss = ggml_cross_entropy_loss(ctx, scores, out.targets);
    ggml_set_name(out.loss, "grouped_loss");
    // ggml-opt marks it as the loss: it refuses a graph that carries another
    ggml_set_output(out.loss);
    return out;
}

bool llama_opt_grouped_inputs(
        const llama_grouped_target & target,
        const std::vector<int32_t> & output_row,
        int64_t                      n_embd_out,
        std::vector<int32_t>       & index,
        std::vector<float>         & mask,
        std::vector<int32_t>       & pointer,
        std::string                & error) {
    const bool is_pointer = target.pointer_rows != nullptr;
    if (target.n_groups == 0 || target.n_opt_max == 0 || !target.rows || (!target.columns && !is_pointer) || !target.targets ||
            target.n_groups > std::numeric_limits<size_t>::max() / target.n_opt_max ||
            n_embd_out <= 0 || n_embd_out > INT32_MAX ||
            output_row.size() > (size_t) INT32_MAX / (size_t) n_embd_out) {
        error = "invalid grouped dimensions or flattened output exceeds I32 indices";
        return false;
    }
    if (is_pointer && n_embd_out % 2 != 0) {
        error = "a pointer target needs an even embeddings output, [q|k]";
        return false;
    }
    const size_t n = (size_t) target.n_opt_max*target.n_groups;
    index.assign(n, 0);
    mask.assign(n, 0.0f);
    pointer.assign(is_pointer ? target.n_groups : 0, 0);

    for (uint32_t g = 0; g < target.n_groups; ++g) {
        // a pointer target reads whole output rows, a column target one score
        const int32_t column = is_pointer ? 0 : target.columns[g];
        if (column < 0 || column >= n_embd_out) {
            error = "group " + std::to_string(g) + ": column " + std::to_string(column) + " is outside the output";
            return false;
        }
        if (is_pointer) {
            const int32_t row = target.pointer_rows[g];
            if (row < 0 || (size_t) row >= output_row.size() || output_row[row] < 0) {
                error = "group " + std::to_string(g) + ": query token " + std::to_string(row) + " is not an output of the batch";
                return false;
            }
            pointer[g] = output_row[row];
        }
        uint32_t n_options = 0;
        double   sum       = 0.0;
        for (uint32_t j = 0; j < target.n_opt_max; ++j) {
            const size_t  k   = (size_t) g*target.n_opt_max + j;
            const int32_t row = target.rows[k];
            const float   p   = target.targets[k];
            if (!std::isfinite(p) || p < 0.0f) {
                error = "group " + std::to_string(g) + ": a target is negative or not finite";
                return false;
            }
            if (row < -1) {
                error = "group " + std::to_string(g) + ": invalid padding row";
                return false;
            }
            if (row == -1) {
                if (p != 0.0f) {
                    error = "group " + std::to_string(g) + ": a padded option has a non-zero target";
                    return false;
                }
                // any valid index: the mask removes it from the softmax
                mask[k] = LLAMA_OPT_GROUPED_MASK;
                continue;
            }
            if ((size_t) row >= output_row.size() || output_row[row] < 0) {
                error = "group " + std::to_string(g) + ": token " + std::to_string(row) + " is not an output of the batch";
                return false;
            }
            index[k] = is_pointer ? output_row[row] : (int32_t) ((int64_t) output_row[row]*n_embd_out + column);
            sum += p;
            ++n_options;
        }
        if (n_options == 0) {
            error = "group " + std::to_string(g) + " has no option";
            return false;
        }
        // Anchor padding to a score from this group. Index zero can belong
        // to another group with a much larger score and defeat the mask.
        int32_t anchor = 0;
        for (uint32_t j = 0; j < target.n_opt_max; ++j) {
            const size_t k = (size_t) g*target.n_opt_max + j;
            if (target.rows[k] >= 0) {
                anchor = index[k];
                break;
            }
        }
        for (uint32_t j = 0; j < target.n_opt_max; ++j) {
            const size_t k = (size_t) g*target.n_opt_max + j;
            if (target.rows[k] == -1) {
                index[k] = anchor;
            }
        }
        if (std::fabs(sum - 1.0) > 1.0e-3) {
            error = "group " + std::to_string(g) + ": the targets sum to " + std::to_string(sum) + ", not 1";
            return false;
        }
    }
    return true;
}
