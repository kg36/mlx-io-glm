// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core {

// CPU-published Metal event used to order ExpertSSD in-place slot writes
// before the GPU graph that consumes those rows.
struct ExpertSSDIoEventState {
  std::shared_ptr<void> event; // MTL::SharedEvent*, opaque outside Metal.
  std::mutex mutex;
  std::set<uint64_t> completed;
  uint64_t published{0};
};

MLX_API std::shared_ptr<ExpertSSDIoEventState>
expert_ssd_io_event_state_new();

MLX_API void expert_ssd_io_event_signal(
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value);

// Block the calling CPU thread until a GPU signal reaches ``value``.
MLX_API void expert_ssd_io_event_wait(
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value);

// Identity node with a GPU-side wait. Operations encoded after this node
// cannot read refilled slot rows until the IO workers publish value.
MLX_API array expert_ssd_io_gate(
    const array& x,
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value,
    StreamOrDevice s = {});

// Identity node that signals a shared Metal event after its input completes.
MLX_API array expert_ssd_gpu_event_signal(
    const array& x,
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value,
    StreamOrDevice s = {});

// Non-spinning companion for a group whose rows are already resident. It
// fuses the two launches while retaining stock fp_qmv_fast_impl arithmetic
// and two explicit BF16 projection outputs.
MLX_API std::vector<array> expert_ssd_mxfp4_pair_qmv(
    const array& x,
    const array& up_weight,
    const array& up_scales,
    const array& gate_weight,
    const array& gate_scales,
    const array& routes,
    StreamOrDevice s = {});

// One dense MXFP4 weight applied to exactly two BF16 rows. The two QMV
// reductions remain independent and width-one-ordered inside one dispatch.
MLX_API array expert_ssd_mxfp4_two_row_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    StreamOrDevice s = {});

// Batched-weight form used by attention MultiLinear output groups.
MLX_API array expert_ssd_mxfp4_grouped_two_row_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    StreamOrDevice s = {});

// Apply one row-major matrix to exactly two BF16 or FP32 vectors using the
// same GEMV specialization and reduction order selected by width-one matmul.
// The vectors share one Metal dispatch but remain independent GEMV batches.
MLX_API array expert_ssd_two_row_gemv(
    const array& x,
    const array& weight,
    StreamOrDevice s = {});

// Exact stock-QMV down projection with 0xffffffff route lanes masked to zero.
MLX_API array expert_ssd_mxfp4_masked_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    const array& routes,
    StreamOrDevice s = {});

// Mode B consumes each complete lossless ScaleX record directly from its
// resident row. The three logical scale tensors are never hydrated. This
// single-projection primitive preserves the stock gather_qmm graph shape.
// projection follows physical ScaleX order: gate=0, down=1, up=2.
MLX_API array expert_ssd_scalex_mxfp4_qmv(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& routes,
    uint32_t projection,
    StreamOrDevice s = {});

// Coupled projection-cache variant. Down weights and their shared ScaleX
// records may use different physical row ids while retaining one resident
// expert set.
MLX_API array expert_ssd_scalex_mxfp4_qmv_split_routes(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& weight_routes,
    const array& scale_routes,
    uint32_t projection,
    StreamOrDevice s = {});

// Fixed width-two/top-six ScaleX Down QMV with exact BF16 route-score
// reduction and shared-expert addition in the same Metal dispatch.
MLX_API array expert_ssd_scalex_mxfp4_width2_down_reduce(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& weight_routes,
    const array& scale_routes,
    const array& scores,
    const array& shared,
    StreamOrDevice s = {});

// Exact optimistic all-hit ScaleX MoE. Router IDs and slot lookup stay on the
// GPU; misses disable the resident continuation through indirect dispatches.
// The existing host cache transaction remains authoritative for selection.
MLX_API array expert_ssd_scalex_conditional_m0(
    const array& indices,
    const array& x,
    const array& scores,
    const array& shared,
    const array& scale_records,
    const array& gate_weight,
    const array& down_weight,
    const array& up_weight,
    const array& gate_directory,
    const array& down_directory,
    const array& gate_routes_scratch,
    const array& down_routes_scratch,
    const array& all_hit_scratch,
    const array& up_scratch,
    const array& gate_scratch,
    const array& activated_scratch,
    const array& routed_scratch,
    const array& indirect_scratch,
    float swiglu_limit = 10.0f,
    StreamOrDevice s = {});

} // namespace mlx::core
