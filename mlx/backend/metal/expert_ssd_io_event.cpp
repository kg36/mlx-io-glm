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

MTL::Buffer* expert_ssd_buffer(const array& value) {
  return static_cast<MTL::Buffer*>(const_cast<void*>(value.buffer().ptr()));
}

NS::UInteger expert_ssd_offset(const array& value, size_t extra = 0) {
  return static_cast<NS::UInteger>(value.offset() + extra);
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

class ExpertSSDScaleXMXFP4QMV : public Primitive {
 public:
  ExpertSSDScaleXMXFP4QMV(Stream stream, uint32_t projection)
      : Primitive(stream), projection_(projection) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4QMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 4 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4QMV] invalid input/output arity");
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scale_records = inputs[2];
    const auto& routes = inputs[3];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);
    const int record_stride = scale_records.shape(-1);
    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel("dsv4_scalex_mxfp4_qmv_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scale_records, 1);
    encoder.set_input_array(x, 2);
    encoder.set_input_array(routes, 3);
    encoder.set_output_array(output, 4);
    encoder.set_bytes(K, 5);
    encoder.set_bytes(N, 6);
    encoder.set_bytes(record_stride, 7);
    encoder.set_bytes(projection_, 8);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4QMV)

 private:
  uint32_t projection_;
};

class ExpertSSDScaleXMXFP4QMVSplitRoutes : public Primitive {
 public:
  ExpertSSDScaleXMXFP4QMVSplitRoutes(Stream stream, uint32_t projection)
      : Primitive(stream), projection_(projection) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4QMVSplitRoutes] CPU evaluation not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    if (inputs.size() != 5 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4QMVSplitRoutes] invalid input/output arity");
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scale_records = inputs[2];
    const auto& weight_routes = inputs[3];
    const auto& scale_routes = inputs[4];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);
    const int record_stride = scale_records.shape(-1);
    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel("dsv4_scalex_mxfp4_qmv_split_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scale_records, 1);
    encoder.set_input_array(x, 2);
    encoder.set_input_array(weight_routes, 3);
    encoder.set_input_array(scale_routes, 4);
    encoder.set_output_array(output, 5);
    encoder.set_bytes(K, 6);
    encoder.set_bytes(N, 7);
    encoder.set_bytes(record_stride, 8);
    encoder.set_bytes(projection_, 9);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, weight_routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4QMVSplitRoutes)

 private:
  uint32_t projection_;
};

class ExpertSSDScaleXMXFP4Width2DownReduce : public Primitive {
 public:
  explicit ExpertSSDScaleXMXFP4Width2DownReduce(Stream stream)
      : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4Width2DownReduce] CPU evaluation not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    if (inputs.size() != 7 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4Width2DownReduce] invalid input/output arity");
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scale_records = inputs[2];
    const auto& weight_routes = inputs[3];
    const auto& scale_routes = inputs[4];
    const auto& scores = inputs[5];
    const auto& shared = inputs[6];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);
    const int record_stride = scale_records.shape(-1);
    auto& d = metal::device(stream().device);
    auto* kernel =
        d.get_kernel("dsv4_scalex_mxfp4_width2_down_reduce_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scale_records, 1);
    encoder.set_input_array(x, 2);
    encoder.set_input_array(weight_routes, 3);
    encoder.set_input_array(scale_routes, 4);
    encoder.set_input_array(scores, 5);
    encoder.set_input_array(shared, 6);
    encoder.set_output_array(output, 7);
    encoder.set_bytes(K, 8);
    encoder.set_bytes(N, 9);
    encoder.set_bytes(record_stride, 10);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, 2), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4Width2DownReduce)
};

class ExpertSSDScaleXConditionalM0 : public Primitive {
 public:
  ExpertSSDScaleXConditionalM0(Stream stream, int width, float swiglu_limit)
      : Primitive(stream), width_(width), swiglu_limit_(swiglu_limit) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXConditionalM0] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 18 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXConditionalM0] invalid input/output arity");
    }
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

    constexpr int hidden = 4096;
    constexpr int intermediate = 2048;
    constexpr uint32_t top_k = 6;
    const uint32_t route_count = static_cast<uint32_t>(width_ * top_k);
    const uint32_t expert_count = static_cast<uint32_t>(inputs[8].size());
    const int record_stride = inputs[4].shape(-1);
    const size_t bf16_bytes = 2;
    const size_t route_bytes = sizeof(uint32_t);
    auto& d = metal::device(stream().device);
    auto& command_encoder = metal::get_command_encoder(stream());
    for (const auto& input : inputs) {
      command_encoder.register_input_array(input);
    }
    command_encoder.register_output_array(outputs[0]);
    for (size_t index = 10; index < inputs.size(); ++index) {
      command_encoder.register_output_array(inputs[index]);
    }
    auto* encoder = command_encoder.expert_ssd_raw_compute_encoder();

    encoder->setComputePipelineState(
        d.get_kernel("dsv4_scalex_m0_map_indirect"));
    encoder->setBuffer(expert_ssd_buffer(inputs[0]), expert_ssd_offset(inputs[0]), 0);
    encoder->setBuffer(expert_ssd_buffer(inputs[8]), expert_ssd_offset(inputs[8]), 1);
    encoder->setBuffer(expert_ssd_buffer(inputs[9]), expert_ssd_offset(inputs[9]), 2);
    encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10]), 3);
    encoder->setBuffer(expert_ssd_buffer(inputs[11]), expert_ssd_offset(inputs[11]), 4);
    encoder->setBuffer(expert_ssd_buffer(inputs[12]), expert_ssd_offset(inputs[12]), 5);
    encoder->setBuffer(expert_ssd_buffer(inputs[17]), expert_ssd_offset(inputs[17]), 6);
    encoder->setBytes(&route_count, sizeof(route_count), 7);
    encoder->setBytes(&expert_count, sizeof(expert_count), 8);
    const uint32_t width_u32 = static_cast<uint32_t>(width_);
    encoder->setBytes(&width_u32, sizeof(width_u32), 9);
    encoder->dispatchThreads(
        MTL::Size(route_count, 1, 1), MTL::Size(32, 1, 1));
    // The following dispatches consume both mapped rows and GPU-published
    // indirect dimensions. This is the one mandatory visibility boundary in
    // the fixed island; later stages remain ordered in the same encoder.
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    auto* pair_kernel =
        d.get_kernel("dsv4_scalex_m0_pair_qmv_sparse_bf16");
    for (int position = 0; position < width_; ++position) {
      const size_t route_offset = position * top_k * route_bytes;
      const size_t x_offset = position * hidden * bf16_bytes;
      const size_t intermediate_offset =
          position * top_k * intermediate * bf16_bytes;
      encoder->setComputePipelineState(pair_kernel);
      encoder->setBuffer(expert_ssd_buffer(inputs[7]), expert_ssd_offset(inputs[7]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[5]), expert_ssd_offset(inputs[5]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[4]), expert_ssd_offset(inputs[4]), 2);
      encoder->setBuffer(expert_ssd_buffer(inputs[1]), expert_ssd_offset(inputs[1], x_offset), 3);
      encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10], route_offset), 4);
      encoder->setBuffer(expert_ssd_buffer(inputs[13]), expert_ssd_offset(inputs[13], intermediate_offset), 5);
      encoder->setBuffer(expert_ssd_buffer(inputs[14]), expert_ssd_offset(inputs[14], intermediate_offset), 6);
      encoder->setBytes(&hidden, sizeof(hidden), 7);
      encoder->setBytes(&intermediate, sizeof(intermediate), 8);
      encoder->setBytes(&record_stride, sizeof(record_stride), 9);
      encoder->setBytes(&top_k, sizeof(top_k), 10);
      encoder->dispatchThreadgroups(
          expert_ssd_buffer(inputs[17]),
          expert_ssd_offset(inputs[17], position * 3 * sizeof(uint32_t)),
          MTL::Size(32, 2, 1));
    }
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    const uint32_t activated_count = route_count * intermediate;
    encoder->setComputePipelineState(
        d.get_kernel("dsv4_scalex_m0_limited_swiglu_bf16"));
    encoder->setBuffer(expert_ssd_buffer(inputs[13]), expert_ssd_offset(inputs[13]), 0);
    encoder->setBuffer(expert_ssd_buffer(inputs[14]), expert_ssd_offset(inputs[14]), 1);
    encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10]), 2);
    encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 3);
    const uint32_t intermediate_u32 = static_cast<uint32_t>(intermediate);
    encoder->setBytes(&intermediate_u32, sizeof(intermediate_u32), 4);
    encoder->setBytes(&activated_count, sizeof(activated_count), 5);
    encoder->setBytes(&swiglu_limit_, sizeof(swiglu_limit_), 6);
    encoder->dispatchThreadgroups(
        expert_ssd_buffer(inputs[17]),
        expert_ssd_offset(inputs[17], width_ * 3 * sizeof(uint32_t)),
        MTL::Size(256, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    if (width_ == 2) {
      encoder->setComputePipelineState(
          d.get_kernel("dsv4_scalex_mxfp4_width2_down_reduce_bf16"));
      encoder->setBuffer(expert_ssd_buffer(inputs[6]), expert_ssd_offset(inputs[6]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[4]), expert_ssd_offset(inputs[4]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 2);
      encoder->setBuffer(expert_ssd_buffer(inputs[11]), expert_ssd_offset(inputs[11]), 3);
      encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10]), 4);
      encoder->setBuffer(expert_ssd_buffer(inputs[2]), expert_ssd_offset(inputs[2]), 5);
      encoder->setBuffer(expert_ssd_buffer(inputs[3]), expert_ssd_offset(inputs[3]), 6);
      encoder->setBuffer(expert_ssd_buffer(outputs[0]), expert_ssd_offset(outputs[0]), 7);
      encoder->setBytes(&intermediate, sizeof(intermediate), 8);
      encoder->setBytes(&hidden, sizeof(hidden), 9);
      encoder->setBytes(&record_stride, sizeof(record_stride), 10);
    } else {
      encoder->setComputePipelineState(
          d.get_kernel("dsv4_scalex_mxfp4_qmv_split_bf16"));
      encoder->setBuffer(expert_ssd_buffer(inputs[6]), expert_ssd_offset(inputs[6]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[4]), expert_ssd_offset(inputs[4]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 2);
      encoder->setBuffer(expert_ssd_buffer(inputs[11]), expert_ssd_offset(inputs[11]), 3);
      encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10]), 4);
      encoder->setBuffer(expert_ssd_buffer(inputs[16]), expert_ssd_offset(inputs[16]), 5);
      encoder->setBytes(&intermediate, sizeof(intermediate), 6);
      encoder->setBytes(&hidden, sizeof(hidden), 7);
      encoder->setBytes(&record_stride, sizeof(record_stride), 8);
      const uint32_t split_projection = 1;
      encoder->setBytes(&split_projection, sizeof(split_projection), 9);
    }
    encoder->dispatchThreadgroups(
        expert_ssd_buffer(inputs[17]),
        expert_ssd_offset(inputs[17], (width_ + 1) * 3 * sizeof(uint32_t)),
        MTL::Size(32, 2, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    if (width_ == 1) {
      encoder->setComputePipelineState(
          d.get_kernel("dsv4_scalex_m0_score_reduce_shared_bf16"));
      encoder->setBuffer(expert_ssd_buffer(inputs[16]), expert_ssd_offset(inputs[16]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[2]), expert_ssd_offset(inputs[2]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[3]), expert_ssd_offset(inputs[3]), 2);
      encoder->setBuffer(expert_ssd_buffer(outputs[0]), expert_ssd_offset(outputs[0]), 3);
      const uint32_t hidden_u32 = static_cast<uint32_t>(hidden);
      encoder->setBytes(&hidden_u32, sizeof(hidden_u32), 4);
      encoder->setBytes(&top_k, sizeof(top_k), 5);
      encoder->dispatchThreadgroups(
          expert_ssd_buffer(inputs[17]),
          expert_ssd_offset(inputs[17], (width_ + 2) * 3 * sizeof(uint32_t)),
          MTL::Size(256, 1, 1));
    }
  }

  DEFINE_NAME(ExpertSSDScaleXConditionalM0)

 private:
  int width_;
  float swiglu_limit_;
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

array expert_ssd_scalex_mxfp4_qmv(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& routes,
    uint32_t projection,
    StreamOrDevice s) {
  if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
      scale_records.dtype() != uint8 ||
      (routes.dtype() != uint32 && routes.dtype() != int32) ||
      projection > 2) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv] incompatible dtype or projection");
  }
  const bool input_rows_match =
      projection == 1 ? x.size() == routes.size() * x.shape(-1)
                      : x.size() == x.shape(-1);
  if (x.ndim() < 2 || x.shape(-2) != 1 || !input_rows_match ||
      routes.ndim() != 1 ||
      routes.size() == 0 || weight.ndim() != 3 ||
      scale_records.ndim() != 2 ||
      weight.shape(0) != scale_records.shape(0)) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv] invalid fixed-decode geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  const bool expected_geometry =
      (projection == 1 && K == 2048 && N == 4096) ||
      (projection != 1 && K == 4096 && N == 2048);
  if (!expected_geometry || weight.shape(-1) * 8 != K) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv] unsupported target geometry");
  }
  Shape output_shape{static_cast<ShapeElem>(routes.size()), 1, N};
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4QMV>(
      to_stream(s, Device::gpu), projection);
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x, weight, scale_records, routes})[0];
}

array expert_ssd_scalex_mxfp4_qmv_split_routes(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& weight_routes,
    const array& scale_routes,
    uint32_t projection,
    StreamOrDevice s) {
  if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
      scale_records.dtype() != uint8 ||
      (weight_routes.dtype() != uint32 && weight_routes.dtype() != int32) ||
      scale_routes.dtype() != weight_routes.dtype() || projection > 2) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_split_routes] incompatible dtype or projection");
  }
  const bool input_rows_match = projection == 1
      ? x.size() == weight_routes.size() * x.shape(-1)
      : x.size() == x.shape(-1);
  if (x.ndim() < 2 || x.shape(-2) != 1 || !input_rows_match ||
      weight_routes.ndim() != 1 || scale_routes.ndim() != 1 ||
      weight_routes.size() == 0 ||
      weight_routes.size() != scale_routes.size() || weight.ndim() != 3 ||
      scale_records.ndim() != 2) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_split_routes] invalid fixed-decode geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  const bool expected_geometry = (projection == 1 && K == 2048 && N == 4096) ||
      (projection != 1 && K == 4096 && N == 2048);
  if (!expected_geometry || weight.shape(-1) * 8 != K) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_split_routes] unsupported target geometry");
  }
  Shape output_shape{static_cast<ShapeElem>(weight_routes.size()), 1, N};
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4QMVSplitRoutes>(
      to_stream(s, Device::gpu), projection);
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x, weight, scale_records, weight_routes, scale_routes})[0];
}

array expert_ssd_scalex_mxfp4_width2_down_reduce(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& weight_routes,
    const array& scale_routes,
    const array& scores,
    const array& shared,
    StreamOrDevice s) {
  const bool route_dtype =
      weight_routes.dtype() == uint32 || weight_routes.dtype() == int32;
  if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
      scale_records.dtype() != uint8 || !route_dtype ||
      scale_routes.dtype() != weight_routes.dtype() ||
      scores.dtype() != float32 || shared.dtype() != bfloat16) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_width2_down_reduce] incompatible dtype");
  }
  if (x.ndim() < 2 || x.shape(-2) != 1 || x.shape(-1) != 2048 ||
      x.size() != 12 * 2048 || weight.ndim() != 3 ||
      weight.shape(-2) != 4096 || weight.shape(-1) * 8 != 2048 ||
      scale_records.ndim() != 2 ||
      weight_routes.ndim() != 1 || weight_routes.size() != 12 ||
      scale_routes.ndim() != 1 || scale_routes.size() != 12 ||
      scores.size() != 12 || shared.size() != 2 * 4096) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_width2_down_reduce] invalid width-two geometry");
  }
  Shape output_shape{2, 1, 4096};
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4Width2DownReduce>(
      to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x,
       weight,
       scale_records,
       weight_routes,
       scale_routes,
       scores,
       shared})[0];
}

array expert_ssd_scalex_conditional_m0(
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
    float swiglu_limit,
    StreamOrDevice s) {
  const int width = x.ndim() == 3 ? x.shape(1) : 0;
  if ((width != 1 && width != 2) || x.dtype() != bfloat16 ||
      x.shape(0) != 1 || x.shape(2) != 4096 || indices.dtype() != int32 ||
      indices.shape() != Shape{1, width, 6} || scores.dtype() != float32 ||
      scores.shape() != Shape{1, width, 6} || shared.dtype() != bfloat16 ||
      shared.shape() != x.shape() || scale_records.dtype() != uint8 ||
      scale_records.ndim() != 2 || gate_weight.dtype() != uint32 ||
      up_weight.dtype() != uint32 || down_weight.dtype() != uint32 ||
      gate_weight.shape() != up_weight.shape() || gate_weight.ndim() != 3 ||
      gate_weight.shape(1) != 2048 || gate_weight.shape(2) != 512 ||
      down_weight.ndim() != 3 || down_weight.shape(0) != gate_weight.shape(0) ||
      down_weight.shape(1) != 4096 || down_weight.shape(2) != 256 ||
      scale_records.shape(0) != gate_weight.shape(0) ||
      gate_directory.dtype() != int32 || down_directory.dtype() != int32 ||
      gate_directory.ndim() != 1 || down_directory.ndim() != 1 ||
      gate_directory.shape() != down_directory.shape() ||
      gate_routes_scratch.dtype() != uint32 ||
      gate_routes_scratch.shape() != Shape{12} ||
      down_routes_scratch.dtype() != uint32 ||
      down_routes_scratch.shape() != Shape{12} ||
      all_hit_scratch.dtype() != int32 ||
      all_hit_scratch.shape() != Shape{1} ||
      up_scratch.dtype() != bfloat16 ||
      up_scratch.shape() != Shape{12, 1, 2048} ||
      gate_scratch.dtype() != bfloat16 ||
      gate_scratch.shape() != Shape{12, 1, 2048} ||
      activated_scratch.dtype() != bfloat16 ||
      activated_scratch.shape() != Shape{12, 1, 2048} ||
      routed_scratch.dtype() != bfloat16 ||
      routed_scratch.shape() != Shape{12, 1, 4096} ||
      indirect_scratch.dtype() != uint32 ||
      indirect_scratch.shape() != Shape{5, 3} ||
      !(swiglu_limit > 0.0f)) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_conditional_m0] incompatible inputs");
  }
  auto primitive = std::make_shared<ExpertSSDScaleXConditionalM0>(
      to_stream(s, Device::gpu), width, swiglu_limit);
  return array(
      x.shape(),
      bfloat16,
      primitive,
      {indices,
       x,
       scores,
       shared,
       scale_records,
       gate_weight,
       down_weight,
       up_weight,
       gate_directory,
       down_directory,
       gate_routes_scratch,
       down_routes_scratch,
       all_hit_scratch,
       up_scratch,
       gate_scratch,
       activated_scratch,
       routed_scratch,
       indirect_scratch});
}

} // namespace mlx::core
