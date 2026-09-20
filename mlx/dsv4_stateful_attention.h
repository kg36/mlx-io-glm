// Copyright © 2026 Apple Inc.

#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core {

/** Exact fixed-width DeepSeek V4 attention over a full local ring.
 *
 * The operation consumes already projected/rotated BF16 query and key rows.
 * It applies the verifier rows sequentially to the 128-row local ring, then
 * attends over the chronological local window followed by either the causal
 * dense pool prefix or caller-supplied sparse pool indices.  The second
 * output contains one functional ring snapshot per verifier position so a
 * speculative transaction can commit any accepted prefix without mutating
 * the input cache.
 */
MLX_API std::vector<array> dsv4_stateful_attention(
    const array& queries,
    const array& new_keys,
    const array& local_ring,
    const array& pooled,
    const array& indices,
    const array& pool_counts,
    const array& sinks,
    int cache_index,
    bool sparse,
    float scale,
    StreamOrDevice s = {});

} // namespace mlx::core
