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

} // namespace mlx::core
