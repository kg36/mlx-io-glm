// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

/** Optional low-overhead decode profiler, enabled before process startup with
 * MLX_PROFILE_METAL_COUNTERS=1. The counters are cumulative so callers can
 * take exact deltas around lazy graph evaluation without changing scheduling.
 */
struct MetalProfileCounters {
  bool enabled;
  uint64_t dispatch_threads;
  uint64_t dispatch_threadgroups;
  uint64_t primitive_evals;
  uint64_t astype_ops;
  uint64_t gather_qmm_ops;
  uint64_t quantized_matmul_ops;
  uint64_t custom_kernel_ops;
  uint64_t compiled_ops;
  uint64_t rms_norm_ops;
  uint64_t hc_sinkhorn_collapse_kernels;
};

MLX_API MetalProfileCounters profile_counters();
MLX_API void profile_record_primitive(const char* name);
MLX_API void profile_record_custom_kernel(const std::string& name);

} // namespace mlx::core::metal
