// Copyright © 2026 LivMLX contributors.

#include "mlx/perfetto_trace.h"

namespace mlx::core::perfetto_trace {

State& State::instance() {
  // Deliberately leaky: Metal completion handlers and detached SSD workers can
  // outlive Python shutdown. One out-of-line instance is required because MLX
  // builds with hidden inline visibility; a header-local singleton would give
  // the Python binding and Metal backend separate trace sessions.
  static auto* state = new State();
  return *state;
}

} // namespace mlx::core::perfetto_trace
