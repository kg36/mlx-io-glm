// Copyright © 2026 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/expert_ssd_io_event.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

MTL::SharedEvent* shared_event(const ExpertSSDIoEventState& state) {
  return static_cast<MTL::SharedEvent*>(state.event.get());
}

class ExpertSSDIoGate : public UnaryPrimitive {
 public:
  ExpertSSDIoGate(
      Stream stream,
      std::shared_ptr<ExpertSSDIoEventState> state,
      uint64_t value)
      : UnaryPrimitive(stream), state_(std::move(state)), value_(value) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("[ExpertSSDIoGate] CPU evaluation not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    out.copy_shared_buffer(inputs[0]);
    auto& encoder = metal::get_command_encoder(stream());
    encoder.end_encoding();
    encoder.get_command_buffer()->encodeWait(
        static_cast<MTL::Event*>(state_->event.get()), value_);
  }

  DEFINE_NAME(ExpertSSDIoGate)

 private:
  std::shared_ptr<ExpertSSDIoEventState> state_;
  uint64_t value_;
};

class ExpertSSDGpuEventSignal : public UnaryPrimitive {
 public:
  ExpertSSDGpuEventSignal(
      Stream stream,
      std::shared_ptr<ExpertSSDIoEventState> state,
      uint64_t value)
      : UnaryPrimitive(stream), state_(std::move(state)), value_(value) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error(
        "[ExpertSSDGpuEventSignal] CPU evaluation not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    out.copy_shared_buffer(inputs[0]);
    auto& encoder = metal::get_command_encoder(stream());
    encoder.end_encoding();
    encoder.get_command_buffer()->encodeSignalEvent(
        static_cast<MTL::Event*>(state_->event.get()), value_);
  }

  DEFINE_NAME(ExpertSSDGpuEventSignal)

 private:
  std::shared_ptr<ExpertSSDIoEventState> state_;
  uint64_t value_;
};

class ExpertSSDMXFP4PairQMV : public Primitive {
 public:
  explicit ExpertSSDMXFP4PairQMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDMXFP4PairQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 6 || outputs.size() != 2) {
      throw std::runtime_error(
          "[ExpertSSDMXFP4PairQMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDMXFP4PairQMV] inputs must be row-contiguous");
      }
    }
    for (auto& output : outputs) {
      output.set_data(allocator::malloc(output.nbytes()));
    }
    const auto& x = inputs[0];
    const auto& up_weight = inputs[1];
    const auto& up_scales = inputs[2];
    const auto& gate_weight = inputs[3];
    const auto& gate_scales = inputs[4];
    const auto& routes = inputs[5];
    const int K = x.shape(-1);
    const int N = up_weight.shape(-2);

    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel("dsv4_mxfp4_pair_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    int binding = 0;
    encoder.set_input_array(up_weight, binding++);
    encoder.set_input_array(up_scales, binding++);
    encoder.set_input_array(gate_weight, binding++);
    encoder.set_input_array(gate_scales, binding++);
    encoder.set_input_array(x, binding++);
    encoder.set_input_array(routes, binding++);
    encoder.set_output_array(outputs[0], binding++);
    encoder.set_output_array(outputs[1], binding++);
    encoder.set_bytes(K, binding++);
    encoder.set_bytes(N, binding++);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDMXFP4PairQMV)
};

class ExpertSSDMXFP4MaskedQMV : public Primitive {
 public:
  explicit ExpertSSDMXFP4MaskedQMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>&,
      std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDMXFP4MaskedQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 4 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDMXFP4MaskedQMV] invalid input arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDMXFP4MaskedQMV] inputs must be row-contiguous");
      }
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& routes = inputs[3];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);

    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel("dsv4_mxfp4_masked_down_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    int binding = 0;
    encoder.set_input_array(weight, binding++);
    encoder.set_input_array(scales, binding++);
    encoder.set_input_array(x, binding++);
    encoder.set_input_array(routes, binding++);
    encoder.set_output_array(output, binding++);
    encoder.set_bytes(K, binding++);
    encoder.set_bytes(N, binding++);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDMXFP4MaskedQMV)
};

} // namespace

std::shared_ptr<ExpertSSDIoEventState> expert_ssd_io_event_state_new() {
  auto state = std::make_shared<ExpertSSDIoEventState>();
  auto dtor = [](void* ptr) {
    auto pool = metal::new_scoped_memory_pool();
    static_cast<MTL::SharedEvent*>(ptr)->release();
  };
  auto pool = metal::new_scoped_memory_pool();
  state->event = std::shared_ptr<void>(
      metal::device(Device::gpu).mtl_device()->newSharedEvent(), dtor);
  if (state->event == nullptr) {
    throw std::runtime_error(
        "[expert_ssd_io_event_state_new] Failed to create Metal shared event");
  }
  return state;
}

void expert_ssd_io_event_signal(
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value) {
  if (!state || value == 0) {
    throw std::invalid_argument(
        "[expert_ssd_io_event_signal] state and positive value required");
  }

  uint64_t publish = 0;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (value <= state->published) {
      return;
    }
    state->completed.insert(value);
    while (state->completed.erase(state->published + 1) != 0) {
      ++state->published;
    }
    publish = state->published;
  }
  if (publish > shared_event(*state)->signaledValue()) {
    shared_event(*state)->setSignaledValue(publish);
  }
}

void expert_ssd_io_event_wait(
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value) {
  if (!state || value == 0) {
    throw std::invalid_argument(
        "[expert_ssd_io_event_wait] state and positive value required");
  }
  if (!shared_event(*state)->waitUntilSignaledValue(value, -1)) {
    throw std::runtime_error(
        "[expert_ssd_io_event_wait] Metal shared-event wait failed");
  }
}

array expert_ssd_io_gate(
    const array& x,
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value,
    StreamOrDevice s) {
  if (!state || value == 0) {
    throw std::invalid_argument(
        "[expert_ssd_io_gate] state and positive value required");
  }
  auto stream = to_stream(s, Device::gpu);
  if (stream.device != Device::gpu) {
    throw std::invalid_argument("[expert_ssd_io_gate] requires a GPU stream");
  }
  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<ExpertSSDIoGate>(stream, state, value),
      {x});
}

array expert_ssd_gpu_event_signal(
    const array& x,
    const std::shared_ptr<ExpertSSDIoEventState>& state,
    uint64_t value,
    StreamOrDevice s) {
  if (!state || value == 0) {
    throw std::invalid_argument(
        "[expert_ssd_gpu_event_signal] state and positive value required");
  }
  auto stream = to_stream(s, Device::gpu);
  if (stream.device != Device::gpu) {
    throw std::invalid_argument(
        "[expert_ssd_gpu_event_signal] requires a GPU stream");
  }
  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<ExpertSSDGpuEventSignal>(stream, state, value),
      {x});
}

std::vector<array> expert_ssd_mxfp4_pair_qmv(
    const array& x,
    const array& up_weight,
    const array& up_scales,
    const array& gate_weight,
    const array& gate_scales,
    const array& routes,
    StreamOrDevice s) {
  if (x.dtype() != bfloat16 || up_weight.dtype() != uint32 ||
      gate_weight.dtype() != uint32 || up_scales.dtype() != uint8 ||
      gate_scales.dtype() != uint8 || routes.dtype() != uint32) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_pair_qmv] requires BF16 MXFP4 inputs and uint32 routes");
  }
  if (x.size() != x.shape(-1) || routes.ndim() != 1 || routes.size() == 0 ||
      up_weight.ndim() != 3 || gate_weight.shape() != up_weight.shape() ||
      up_scales.ndim() != 3 || gate_scales.shape() != up_scales.shape() ||
      up_weight.shape(0) != up_scales.shape(0) ||
      up_weight.shape(1) != up_scales.shape(1)) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_pair_qmv] invalid fixed-decode geometry");
  }
  const int K = x.shape(-1);
  const int N = up_weight.shape(-2);
  if (K % 512 != 0 || N % 8 != 0 || up_weight.shape(-1) * 8 != K ||
      up_scales.shape(-1) * 32 != K) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_pair_qmv] geometry is not mxfp4_gather_qmv_fast compatible");
  }
  Shape output_shape{static_cast<ShapeElem>(routes.size()), 1, N};
  auto primitive =
      std::make_shared<ExpertSSDMXFP4PairQMV>(to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape, output_shape},
      {bfloat16, bfloat16},
      primitive,
      {x, up_weight, up_scales, gate_weight, gate_scales, routes});
}

array expert_ssd_mxfp4_masked_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    const array& routes,
    StreamOrDevice s) {
  if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
      scales.dtype() != uint8 || routes.dtype() != uint32) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_masked_qmv] requires BF16 MXFP4 inputs and uint32 routes");
  }
  if (x.ndim() < 2 || x.shape(-2) != 1 ||
      x.size() != routes.size() * x.shape(-1) || routes.ndim() != 1 ||
      routes.size() == 0 || weight.ndim() != 3 || scales.ndim() != 3 ||
      weight.shape(0) != scales.shape(0) ||
      weight.shape(1) != scales.shape(1)) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_masked_qmv] invalid fixed-decode geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  if (K % 512 != 0 || N % 8 != 0 || weight.shape(-1) * 8 != K ||
      scales.shape(-1) * 32 != K) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_masked_qmv] geometry is not mxfp4_gather_qmv_fast compatible");
  }
  Shape output_shape{static_cast<ShapeElem>(routes.size()), 1, N};
  auto primitive =
      std::make_shared<ExpertSSDMXFP4MaskedQMV>(to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x, weight, scales, routes})[0];
}

} // namespace mlx::core
