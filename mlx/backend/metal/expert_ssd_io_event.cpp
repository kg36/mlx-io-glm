// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
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

class LivSeekLegacyMXFP4Quantize : public Primitive {
 public:
  explicit LivSeekLegacyMXFP4Quantize(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[LivSeekLegacyMXFP4Quantize] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 1 || outputs.size() != 2) {
      throw std::runtime_error(
          "[LivSeekLegacyMXFP4Quantize] invalid input/output arity");
    }
    const auto& weight = inputs[0];
    if (!weight.flags().row_contiguous) {
      throw std::runtime_error(
          "[LivSeekLegacyMXFP4Quantize] input must be row-contiguous");
    }
    for (auto& output : outputs) {
      output.set_data(allocator::malloc(output.nbytes()));
    }

    auto& device = metal::device(stream().device);
    auto* kernel = device.get_kernel(
        weight.dtype() == bfloat16 ? "dsv4_legacy_mxfp4_quantize_bf16"
                                   : "dsv4_legacy_mxfp4_quantize_f32");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_output_array(outputs[0], 1);
    encoder.set_output_array(outputs[1], 2);

    size_t group_size = std::min<size_t>(
        256, kernel->maxTotalThreadsPerThreadgroup());
    group_size -= group_size % 32;
    encoder.dispatch_threads(
        MTL::Size(weight.size(), 1, 1), MTL::Size(group_size, 1, 1));
  }

  DEFINE_NAME(LivSeekLegacyMXFP4Quantize)
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

class ExpertSSDMXFP4TwoRowQMV : public Primitive {
 public:
  explicit ExpertSSDMXFP4TwoRowQMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDMXFP4TwoRowQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDMXFP4TwoRowQMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDMXFP4TwoRowQMV] inputs must be row-contiguous");
      }
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);

    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel(
        x.dtype() == bfloat16 ? "dsv4_mxfp4_two_row_bf16"
                              : "dsv4_mxfp4_two_row_f32");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    int binding = 0;
    encoder.set_input_array(weight, binding++);
    encoder.set_input_array(scales, binding++);
    encoder.set_input_array(x, binding++);
    encoder.set_output_array(output, binding++);
    encoder.set_bytes(K, binding++);
    encoder.set_bytes(N, binding++);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, 1), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDMXFP4TwoRowQMV)
};

class ExpertSSDMXFP4GroupedTwoRowQMV : public Primitive {
 public:
  explicit ExpertSSDMXFP4GroupedTwoRowQMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDMXFP4GroupedTwoRowQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDMXFP4GroupedTwoRowQMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDMXFP4GroupedTwoRowQMV] inputs must be row-contiguous");
      }
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);
    const int groups = weight.shape(0);

    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel(
        x.dtype() == bfloat16 ? "dsv4_mxfp4_grouped_two_row_bf16"
                              : "dsv4_mxfp4_grouped_two_row_f32");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    int binding = 0;
    encoder.set_input_array(weight, binding++);
    encoder.set_input_array(scales, binding++);
    encoder.set_input_array(x, binding++);
    encoder.set_output_array(output, binding++);
    encoder.set_bytes(K, binding++);
    encoder.set_bytes(N, binding++);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, groups), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDMXFP4GroupedTwoRowQMV)
};

class ExpertSSDMXFP4ThreeRowQMV : public Primitive {
 public:
  explicit ExpertSSDMXFP4ThreeRowQMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDMXFP4ThreeRowQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDMXFP4ThreeRowQMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDMXFP4ThreeRowQMV] inputs must be row-contiguous");
      }
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);

    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel(
        x.dtype() == bfloat16 ? "dsv4_mxfp4_three_row_bf16"
                              : "dsv4_mxfp4_three_row_f32");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    int binding = 0;
    encoder.set_input_array(weight, binding++);
    encoder.set_input_array(scales, binding++);
    encoder.set_input_array(x, binding++);
    encoder.set_output_array(output, binding++);
    encoder.set_bytes(K, binding++);
    encoder.set_bytes(N, binding++);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, 1), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDMXFP4ThreeRowQMV)
};

class ExpertSSDMXFP4GroupedThreeRowQMV : public Primitive {
 public:
  explicit ExpertSSDMXFP4GroupedThreeRowQMV(Stream stream)
      : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDMXFP4GroupedThreeRowQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDMXFP4GroupedThreeRowQMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDMXFP4GroupedThreeRowQMV] inputs must be row-contiguous");
      }
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const int K = x.shape(-1);
    const int N = weight.shape(-2);
    const int groups = weight.shape(0);

    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel(
        x.dtype() == bfloat16 ? "dsv4_mxfp4_grouped_three_row_bf16"
                              : "dsv4_mxfp4_grouped_three_row_f32");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    int binding = 0;
    encoder.set_input_array(weight, binding++);
    encoder.set_input_array(scales, binding++);
    encoder.set_input_array(x, binding++);
    encoder.set_output_array(output, binding++);
    encoder.set_bytes(K, binding++);
    encoder.set_bytes(N, binding++);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, groups), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDMXFP4GroupedThreeRowQMV)
};

class ExpertSSDTwoRowGEMV : public Primitive {
 public:
  explicit ExpertSSDTwoRowGEMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDTwoRowGEMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 2 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDTwoRowGEMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDTwoRowGEMV] inputs must be row-contiguous");
      }
    }

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    const int K = x.shape(-1);
    const int N = weight.shape(0);
    constexpr int bm = 1;
    constexpr int bn = 8;
    constexpr int sm = 1;
    constexpr int sn = 32;
    constexpr int tm = 4;
    constexpr int tn = 4;
    constexpr bool noncontiguous_batch = false;
    constexpr bool axpby = false;

    auto& d = metal::device(stream().device);
    auto* kernel = get_gemv_kernel(
        d,
        x.dtype() == bfloat16
            ? "gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0"
            : "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
        output,
        false,
        bm,
        bn,
        sm,
        sn,
        tm,
        tn,
        noncontiguous_batch,
        axpby);
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(x, 1);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(K, 4);
    encoder.set_bytes(N, 5);
    encoder.set_bytes(K, 6);

    const int batch_ndim = 1;
    const Shape batch_shape{2};
    const Strides vector_batch_stride{K};
    const Strides matrix_batch_stride{0};
    encoder.set_bytes(batch_ndim, 9);
    encoder.set_vector_bytes(batch_shape, 10);
    encoder.set_vector_bytes(vector_batch_stride, 11);
    encoder.set_vector_bytes(matrix_batch_stride, 12);

    const int outputs_per_group = bm * sm * tm;
    const int groups = (N + outputs_per_group - 1) / outputs_per_group;
    encoder.dispatch_threadgroups(
        MTL::Size(groups, 1, 2), MTL::Size(32, bn, bm));
  }

  DEFINE_NAME(ExpertSSDTwoRowGEMV)
};

class ExpertSSDThreeRowGEMV : public Primitive {
 public:
  explicit ExpertSSDThreeRowGEMV(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDThreeRowGEMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 2 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDThreeRowGEMV] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[ExpertSSDThreeRowGEMV] inputs must be row-contiguous");
      }
    }

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    const int K = x.shape(-1);
    const int N = weight.shape(0);
    constexpr int bm = 1;
    constexpr int bn = 8;
    constexpr int sm = 1;
    constexpr int sn = 32;
    constexpr int tm = 4;
    constexpr int tn = 4;
    constexpr bool noncontiguous_batch = false;
    constexpr bool axpby = false;

    auto& d = metal::device(stream().device);
    auto* kernel = get_gemv_kernel(
        d,
        x.dtype() == bfloat16
            ? "gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0"
            : "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
        output,
        false,
        bm,
        bn,
        sm,
        sn,
        tm,
        tn,
        noncontiguous_batch,
        axpby);
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(x, 1);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(K, 4);
    encoder.set_bytes(N, 5);
    encoder.set_bytes(K, 6);

    const int batch_ndim = 1;
    const Shape batch_shape{3};
    const Strides vector_batch_stride{K};
    const Strides matrix_batch_stride{0};
    encoder.set_bytes(batch_ndim, 9);
    encoder.set_vector_bytes(batch_shape, 10);
    encoder.set_vector_bytes(vector_batch_stride, 11);
    encoder.set_vector_bytes(matrix_batch_stride, 12);

    const int outputs_per_group = bm * sm * tm;
    const int groups = (N + outputs_per_group - 1) / outputs_per_group;
    encoder.dispatch_threadgroups(
        MTL::Size(groups, 1, 3), MTL::Size(32, bn, bm));
  }

  DEFINE_NAME(ExpertSSDThreeRowGEMV)
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

class ExpertSSDScaleXMXFP4GroupedQMV : public Primitive {
 public:
  ExpertSSDScaleXMXFP4GroupedQMV(
      Stream stream,
      uint32_t projection,
      uint32_t top_k)
      : Primitive(stream), projection_(projection), top_k_(top_k) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4GroupedQMV] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 4 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4GroupedQMV] invalid input/output arity");
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
    auto* kernel = d.get_kernel("dsv4_scalex_mxfp4_grouped_qmv_bf16");
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
    encoder.set_bytes(top_k_, 9);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4GroupedQMV)

 private:
  uint32_t projection_;
  uint32_t top_k_;
};

class ExpertSSDScaleXMXFP4QMVTwoBank : public Primitive {
 public:
  ExpertSSDScaleXMXFP4QMVTwoBank(Stream stream, uint32_t projection)
      : Primitive(stream), projection_(projection) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4QMVTwoBank] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 7 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4QMVTwoBank] invalid input/output arity");
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& private_weight = inputs[1];
    const auto& private_records = inputs[2];
    const auto& shared_weight = inputs[3];
    const auto& shared_records = inputs[4];
    const auto& routes = inputs[5];
    const auto& bank_routes = inputs[6];
    const int K = x.shape(-1);
    const int N = private_weight.shape(-2);
    const int private_record_stride = private_records.shape(-1);
    const int shared_record_stride = shared_records.shape(-1);
    auto& d = metal::device(stream().device);
    auto* kernel = d.get_kernel("dsv4_scalex_mxfp4_qmv_two_bank_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(private_weight, 0);
    encoder.set_input_array(private_records, 1);
    encoder.set_input_array(shared_weight, 2);
    encoder.set_input_array(shared_records, 3);
    encoder.set_input_array(x, 4);
    encoder.set_input_array(routes, 5);
    encoder.set_input_array(bank_routes, 6);
    encoder.set_output_array(output, 7);
    encoder.set_bytes(K, 8);
    encoder.set_bytes(N, 9);
    encoder.set_bytes(private_record_stride, 10);
    encoder.set_bytes(shared_record_stride, 11);
    encoder.set_bytes(projection_, 12);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4QMVTwoBank)

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

class ExpertSSDScaleXMXFP4QMVSplitRoutesTwoBank : public Primitive {
 public:
  ExpertSSDScaleXMXFP4QMVSplitRoutesTwoBank(
      Stream stream,
      uint32_t projection)
      : Primitive(stream), projection_(projection) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4QMVSplitRoutesTwoBank] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 8 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4QMVSplitRoutesTwoBank] invalid input/output arity");
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& private_weight = inputs[1];
    const auto& private_records = inputs[2];
    const auto& shared_weight = inputs[3];
    const auto& shared_records = inputs[4];
    const auto& weight_routes = inputs[5];
    const auto& scale_routes = inputs[6];
    const auto& bank_routes = inputs[7];
    const int K = x.shape(-1);
    const int N = private_weight.shape(-2);
    const int private_record_stride = private_records.shape(-1);
    const int shared_record_stride = shared_records.shape(-1);
    auto& d = metal::device(stream().device);
    auto* kernel =
        d.get_kernel("dsv4_scalex_mxfp4_qmv_split_two_bank_bf16");
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(private_weight, 0);
    encoder.set_input_array(private_records, 1);
    encoder.set_input_array(shared_weight, 2);
    encoder.set_input_array(shared_records, 3);
    encoder.set_input_array(x, 4);
    encoder.set_input_array(weight_routes, 5);
    encoder.set_input_array(scale_routes, 6);
    encoder.set_input_array(bank_routes, 7);
    encoder.set_output_array(output, 8);
    encoder.set_bytes(K, 9);
    encoder.set_bytes(N, 10);
    encoder.set_bytes(private_record_stride, 11);
    encoder.set_bytes(shared_record_stride, 12);
    encoder.set_bytes(projection_, 13);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, weight_routes.size()), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4QMVSplitRoutesTwoBank)

 private:
  uint32_t projection_;
};

class ExpertSSDScaleXMXFP4FixedDownReduce : public Primitive {
 public:
  ExpertSSDScaleXMXFP4FixedDownReduce(Stream stream, int width)
      : Primitive(stream), width_(width) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4FixedDownReduce] CPU evaluation not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    if (inputs.size() != 7 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4FixedDownReduce] invalid input/output arity");
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
    const char* kernel_name = width_ == 2
        ? "dsv4_scalex_mxfp4_width2_down_reduce_bf16"
        : "dsv4_scalex_mxfp4_width3_down_reduce_bf16";
    auto* kernel = d.get_kernel(kernel_name);
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
        MTL::Size(1, N / 8, width_), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4FixedDownReduce)

 private:
  int width_;
};

class ExpertSSDScaleXMXFP4FixedDownReduceTwoBank : public Primitive {
 public:
  ExpertSSDScaleXMXFP4FixedDownReduceTwoBank(Stream stream, int width)
      : Primitive(stream), width_(width) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXMXFP4FixedDownReduceTwoBank] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 10 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXMXFP4FixedDownReduceTwoBank] invalid input/output arity");
    }
    auto& output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));
    const auto& x = inputs[0];
    const auto& private_weight = inputs[1];
    const auto& private_records = inputs[2];
    const auto& shared_weight = inputs[3];
    const auto& shared_records = inputs[4];
    const auto& weight_routes = inputs[5];
    const auto& scale_routes = inputs[6];
    const auto& bank_routes = inputs[7];
    const auto& scores = inputs[8];
    const auto& shared = inputs[9];
    const int K = x.shape(-1);
    const int N = private_weight.shape(-2);
    const int private_record_stride = private_records.shape(-1);
    const int shared_record_stride = shared_records.shape(-1);
    auto& d = metal::device(stream().device);
    const char* kernel_name = width_ == 2
        ? "dsv4_scalex_mxfp4_width2_down_reduce_two_bank_bf16"
        : "dsv4_scalex_mxfp4_width3_down_reduce_two_bank_bf16";
    auto* kernel = d.get_kernel(kernel_name);
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(private_weight, 0);
    encoder.set_input_array(private_records, 1);
    encoder.set_input_array(shared_weight, 2);
    encoder.set_input_array(shared_records, 3);
    encoder.set_input_array(x, 4);
    encoder.set_input_array(weight_routes, 5);
    encoder.set_input_array(scale_routes, 6);
    encoder.set_input_array(bank_routes, 7);
    encoder.set_input_array(scores, 8);
    encoder.set_input_array(shared, 9);
    encoder.set_output_array(output, 10);
    encoder.set_bytes(K, 11);
    encoder.set_bytes(N, 12);
    encoder.set_bytes(private_record_stride, 13);
    encoder.set_bytes(shared_record_stride, 14);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, width_), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(ExpertSSDScaleXMXFP4FixedDownReduceTwoBank)

 private:
  int width_;
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

class ExpertSSDScaleXConditionalM0TwoBank : public Primitive {
 public:
  ExpertSSDScaleXConditionalM0TwoBank(
      Stream stream,
      int width,
      float swiglu_limit)
      : Primitive(stream), width_(width), swiglu_limit_(swiglu_limit) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[ExpertSSDScaleXConditionalM0TwoBank] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 24 || outputs.size() != 1) {
      throw std::runtime_error(
          "[ExpertSSDScaleXConditionalM0TwoBank] invalid input/output arity");
    }
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

    constexpr int hidden = 4096;
    constexpr int intermediate = 2048;
    constexpr uint32_t top_k = 6;
    const uint32_t route_count = static_cast<uint32_t>(width_ * top_k);
    const uint32_t expert_count = static_cast<uint32_t>(inputs[12].size());
    const int private_record_stride = inputs[4].shape(-1);
    const int shared_record_stride = inputs[8].shape(-1);
    const size_t bf16_bytes = 2;
    const size_t route_bytes = sizeof(uint32_t);
    auto& d = metal::device(stream().device);
    auto& command_encoder = metal::get_command_encoder(stream());
    for (const auto& input : inputs) {
      command_encoder.register_input_array(input);
    }
    command_encoder.register_output_array(outputs[0]);
    for (size_t index = 15; index < inputs.size(); ++index) {
      command_encoder.register_output_array(inputs[index]);
    }
    auto* encoder = command_encoder.expert_ssd_raw_compute_encoder();

    encoder->setComputePipelineState(
        d.get_kernel("dsv4_scalex_m0_map_two_bank_indirect"));
    encoder->setBuffer(expert_ssd_buffer(inputs[0]), expert_ssd_offset(inputs[0]), 0);
    encoder->setBuffer(expert_ssd_buffer(inputs[12]), expert_ssd_offset(inputs[12]), 1);
    encoder->setBuffer(expert_ssd_buffer(inputs[13]), expert_ssd_offset(inputs[13]), 2);
    encoder->setBuffer(expert_ssd_buffer(inputs[14]), expert_ssd_offset(inputs[14]), 3);
    encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 4);
    encoder->setBuffer(expert_ssd_buffer(inputs[16]), expert_ssd_offset(inputs[16]), 5);
    encoder->setBuffer(expert_ssd_buffer(inputs[17]), expert_ssd_offset(inputs[17]), 6);
    encoder->setBuffer(expert_ssd_buffer(inputs[18]), expert_ssd_offset(inputs[18]), 7);
    encoder->setBuffer(expert_ssd_buffer(inputs[23]), expert_ssd_offset(inputs[23]), 8);
    encoder->setBytes(&route_count, sizeof(route_count), 9);
    encoder->setBytes(&expert_count, sizeof(expert_count), 10);
    const uint32_t width_u32 = static_cast<uint32_t>(width_);
    encoder->setBytes(&width_u32, sizeof(width_u32), 11);
    encoder->dispatchThreads(
        MTL::Size(route_count, 1, 1), MTL::Size(32, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    auto* pair_kernel =
        d.get_kernel("dsv4_scalex_m0_pair_qmv_sparse_two_bank_bf16");
    for (int position = 0; position < width_; ++position) {
      const size_t route_offset = position * top_k * route_bytes;
      const size_t x_offset = position * hidden * bf16_bytes;
      const size_t intermediate_offset =
          position * top_k * intermediate * bf16_bytes;
      encoder->setComputePipelineState(pair_kernel);
      encoder->setBuffer(expert_ssd_buffer(inputs[7]), expert_ssd_offset(inputs[7]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[5]), expert_ssd_offset(inputs[5]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[4]), expert_ssd_offset(inputs[4]), 2);
      encoder->setBuffer(expert_ssd_buffer(inputs[11]), expert_ssd_offset(inputs[11]), 3);
      encoder->setBuffer(expert_ssd_buffer(inputs[9]), expert_ssd_offset(inputs[9]), 4);
      encoder->setBuffer(expert_ssd_buffer(inputs[8]), expert_ssd_offset(inputs[8]), 5);
      encoder->setBuffer(expert_ssd_buffer(inputs[1]), expert_ssd_offset(inputs[1], x_offset), 6);
      encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15], route_offset), 7);
      encoder->setBuffer(expert_ssd_buffer(inputs[17]), expert_ssd_offset(inputs[17], route_offset), 8);
      encoder->setBuffer(expert_ssd_buffer(inputs[19]), expert_ssd_offset(inputs[19], intermediate_offset), 9);
      encoder->setBuffer(expert_ssd_buffer(inputs[20]), expert_ssd_offset(inputs[20], intermediate_offset), 10);
      encoder->setBytes(&hidden, sizeof(hidden), 11);
      encoder->setBytes(&intermediate, sizeof(intermediate), 12);
      encoder->setBytes(&private_record_stride, sizeof(private_record_stride), 13);
      encoder->setBytes(&shared_record_stride, sizeof(shared_record_stride), 14);
      encoder->setBytes(&top_k, sizeof(top_k), 15);
      encoder->dispatchThreadgroups(
          expert_ssd_buffer(inputs[23]),
          expert_ssd_offset(inputs[23], position * 3 * sizeof(uint32_t)),
          MTL::Size(32, 2, 1));
    }
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    const uint32_t activated_count = route_count * intermediate;
    encoder->setComputePipelineState(
        d.get_kernel("dsv4_scalex_m0_limited_swiglu_bf16"));
    encoder->setBuffer(expert_ssd_buffer(inputs[19]), expert_ssd_offset(inputs[19]), 0);
    encoder->setBuffer(expert_ssd_buffer(inputs[20]), expert_ssd_offset(inputs[20]), 1);
    encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 2);
    encoder->setBuffer(expert_ssd_buffer(inputs[21]), expert_ssd_offset(inputs[21]), 3);
    const uint32_t intermediate_u32 = static_cast<uint32_t>(intermediate);
    encoder->setBytes(&intermediate_u32, sizeof(intermediate_u32), 4);
    encoder->setBytes(&activated_count, sizeof(activated_count), 5);
    encoder->setBytes(&swiglu_limit_, sizeof(swiglu_limit_), 6);
    encoder->dispatchThreadgroups(
        expert_ssd_buffer(inputs[23]),
        expert_ssd_offset(inputs[23], width_ * 3 * sizeof(uint32_t)),
        MTL::Size(256, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    if (width_ == 2) {
      encoder->setComputePipelineState(
          d.get_kernel("dsv4_scalex_mxfp4_width2_down_reduce_two_bank_bf16"));
      encoder->setBuffer(expert_ssd_buffer(inputs[6]), expert_ssd_offset(inputs[6]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[4]), expert_ssd_offset(inputs[4]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10]), 2);
      encoder->setBuffer(expert_ssd_buffer(inputs[8]), expert_ssd_offset(inputs[8]), 3);
      encoder->setBuffer(expert_ssd_buffer(inputs[21]), expert_ssd_offset(inputs[21]), 4);
      encoder->setBuffer(expert_ssd_buffer(inputs[16]), expert_ssd_offset(inputs[16]), 5);
      encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 6);
      encoder->setBuffer(expert_ssd_buffer(inputs[17]), expert_ssd_offset(inputs[17]), 7);
      encoder->setBuffer(expert_ssd_buffer(inputs[2]), expert_ssd_offset(inputs[2]), 8);
      encoder->setBuffer(expert_ssd_buffer(inputs[3]), expert_ssd_offset(inputs[3]), 9);
      encoder->setBuffer(expert_ssd_buffer(outputs[0]), expert_ssd_offset(outputs[0]), 10);
      encoder->setBytes(&intermediate, sizeof(intermediate), 11);
      encoder->setBytes(&hidden, sizeof(hidden), 12);
      encoder->setBytes(&private_record_stride, sizeof(private_record_stride), 13);
      encoder->setBytes(&shared_record_stride, sizeof(shared_record_stride), 14);
    } else {
      encoder->setComputePipelineState(
          d.get_kernel("dsv4_scalex_mxfp4_qmv_split_two_bank_bf16"));
      encoder->setBuffer(expert_ssd_buffer(inputs[6]), expert_ssd_offset(inputs[6]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[4]), expert_ssd_offset(inputs[4]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[10]), expert_ssd_offset(inputs[10]), 2);
      encoder->setBuffer(expert_ssd_buffer(inputs[8]), expert_ssd_offset(inputs[8]), 3);
      encoder->setBuffer(expert_ssd_buffer(inputs[21]), expert_ssd_offset(inputs[21]), 4);
      encoder->setBuffer(expert_ssd_buffer(inputs[16]), expert_ssd_offset(inputs[16]), 5);
      encoder->setBuffer(expert_ssd_buffer(inputs[15]), expert_ssd_offset(inputs[15]), 6);
      encoder->setBuffer(expert_ssd_buffer(inputs[17]), expert_ssd_offset(inputs[17]), 7);
      encoder->setBuffer(expert_ssd_buffer(inputs[22]), expert_ssd_offset(inputs[22]), 8);
      encoder->setBytes(&intermediate, sizeof(intermediate), 9);
      encoder->setBytes(&hidden, sizeof(hidden), 10);
      encoder->setBytes(&private_record_stride, sizeof(private_record_stride), 11);
      encoder->setBytes(&shared_record_stride, sizeof(shared_record_stride), 12);
      const uint32_t split_projection = 1;
      encoder->setBytes(&split_projection, sizeof(split_projection), 13);
    }
    encoder->dispatchThreadgroups(
        expert_ssd_buffer(inputs[23]),
        expert_ssd_offset(inputs[23], (width_ + 1) * 3 * sizeof(uint32_t)),
        MTL::Size(32, 2, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    if (width_ == 1) {
      encoder->setComputePipelineState(
          d.get_kernel("dsv4_scalex_m0_score_reduce_shared_bf16"));
      encoder->setBuffer(expert_ssd_buffer(inputs[22]), expert_ssd_offset(inputs[22]), 0);
      encoder->setBuffer(expert_ssd_buffer(inputs[2]), expert_ssd_offset(inputs[2]), 1);
      encoder->setBuffer(expert_ssd_buffer(inputs[3]), expert_ssd_offset(inputs[3]), 2);
      encoder->setBuffer(expert_ssd_buffer(outputs[0]), expert_ssd_offset(outputs[0]), 3);
      const uint32_t hidden_u32 = static_cast<uint32_t>(hidden);
      encoder->setBytes(&hidden_u32, sizeof(hidden_u32), 4);
      encoder->setBytes(&top_k, sizeof(top_k), 5);
      encoder->dispatchThreadgroups(
          expert_ssd_buffer(inputs[23]),
          expert_ssd_offset(inputs[23], (width_ + 2) * 3 * sizeof(uint32_t)),
          MTL::Size(256, 1, 1));
    }
  }

  DEFINE_NAME(ExpertSSDScaleXConditionalM0TwoBank)

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

std::vector<array> livseek_legacy_mxfp4_quantize(
    const array& weight,
    StreamOrDevice s) {
  if ((weight.dtype() != bfloat16 && weight.dtype() != float32) ||
      weight.ndim() < 2 ||
      weight.shape(-1) % 32 != 0) {
    throw std::invalid_argument(
        "[livseek_legacy_mxfp4_quantize] requires BF16/FP32 rows divisible by 32");
  }
  auto stream = to_stream(s, Device::gpu);
  if (stream.device != Device::gpu) {
    throw std::invalid_argument(
        "[livseek_legacy_mxfp4_quantize] requires a GPU stream");
  }

  auto packed_shape = weight.shape();
  packed_shape.back() /= 8;
  auto scales_shape = weight.shape();
  scales_shape.back() /= 32;
  return array::make_arrays(
      {std::move(packed_shape), std::move(scales_shape)},
      {uint32, uint8},
      std::make_shared<LivSeekLegacyMXFP4Quantize>(stream),
      {weight});
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

array expert_ssd_mxfp4_two_row_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    StreamOrDevice s) {
  if ((x.dtype() != bfloat16 && x.dtype() != float32) ||
      weight.dtype() != uint32 ||
      scales.dtype() != uint8) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_two_row_qmv] requires BF16/FP32 input and MXFP4 weights");
  }
  if (x.ndim() < 2 || x.shape(-2) != 2 || x.size() != 2 * x.shape(-1) ||
      weight.ndim() != 2 || scales.ndim() != 2 ||
      weight.shape(0) != scales.shape(0)) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_two_row_qmv] invalid two-row geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  if (K % 512 != 0 || N % 8 != 0 || weight.shape(-1) * 8 != K ||
      scales.shape(-1) * 32 != K) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_two_row_qmv] geometry is not MXFP4 QMV compatible");
  }
  Shape output_shape = x.shape();
  output_shape.back() = N;
  auto primitive =
      std::make_shared<ExpertSSDMXFP4TwoRowQMV>(to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape}, {x.dtype()}, primitive, {x, weight, scales})[0];
}

array expert_ssd_mxfp4_grouped_two_row_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    StreamOrDevice s) {
  if ((x.dtype() != bfloat16 && x.dtype() != float32) ||
      weight.dtype() != uint32 ||
      scales.dtype() != uint8) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_grouped_two_row_qmv] requires BF16/FP32 input and MXFP4 weights");
  }
  if (x.ndim() != 3 || x.shape(-2) != 2 || weight.ndim() != 3 ||
      scales.ndim() != 3 || x.shape(0) != weight.shape(0) ||
      weight.shape(0) != scales.shape(0) ||
      weight.shape(1) != scales.shape(1)) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_grouped_two_row_qmv] invalid grouped geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  if (K % 512 != 0 || N % 8 != 0 || weight.shape(-1) * 8 != K ||
      scales.shape(-1) * 32 != K) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_grouped_two_row_qmv] geometry is not MXFP4 QMV compatible");
  }
  Shape output_shape = x.shape();
  output_shape.back() = N;
  auto primitive = std::make_shared<ExpertSSDMXFP4GroupedTwoRowQMV>(
      to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape}, {x.dtype()}, primitive, {x, weight, scales})[0];
}

array expert_ssd_mxfp4_three_row_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    StreamOrDevice s) {
  if ((x.dtype() != bfloat16 && x.dtype() != float32) ||
      weight.dtype() != uint32 || scales.dtype() != uint8) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_three_row_qmv] requires BF16/FP32 input and MXFP4 weights");
  }
  if (x.ndim() < 2 || x.shape(-2) != 3 || x.size() != 3 * x.shape(-1) ||
      weight.ndim() != 2 || scales.ndim() != 2 ||
      weight.shape(0) != scales.shape(0)) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_three_row_qmv] invalid three-row geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  if (K % 512 != 0 || N % 8 != 0 || weight.shape(-1) * 8 != K ||
      scales.shape(-1) * 32 != K) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_three_row_qmv] geometry is not MXFP4 QMV compatible");
  }
  Shape output_shape = x.shape();
  output_shape.back() = N;
  auto primitive =
      std::make_shared<ExpertSSDMXFP4ThreeRowQMV>(to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape}, {x.dtype()}, primitive, {x, weight, scales})[0];
}

array expert_ssd_mxfp4_grouped_three_row_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    StreamOrDevice s) {
  if ((x.dtype() != bfloat16 && x.dtype() != float32) ||
      weight.dtype() != uint32 || scales.dtype() != uint8) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_grouped_three_row_qmv] requires BF16/FP32 input and MXFP4 weights");
  }
  if (x.ndim() != 3 || x.shape(-2) != 3 || weight.ndim() != 3 ||
      scales.ndim() != 3 || x.shape(0) != weight.shape(0) ||
      weight.shape(0) != scales.shape(0) ||
      weight.shape(1) != scales.shape(1)) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_grouped_three_row_qmv] invalid grouped geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  if (K % 512 != 0 || N % 8 != 0 || weight.shape(-1) * 8 != K ||
      scales.shape(-1) * 32 != K) {
    throw std::invalid_argument(
        "[expert_ssd_mxfp4_grouped_three_row_qmv] geometry is not MXFP4 QMV compatible");
  }
  Shape output_shape = x.shape();
  output_shape.back() = N;
  auto primitive = std::make_shared<ExpertSSDMXFP4GroupedThreeRowQMV>(
      to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape}, {x.dtype()}, primitive, {x, weight, scales})[0];
}

array expert_ssd_two_row_gemv(
    const array& x,
    const array& weight,
    StreamOrDevice s) {
  if ((x.dtype() != bfloat16 && x.dtype() != float32) ||
      weight.dtype() != x.dtype()) {
    throw std::invalid_argument(
        "[expert_ssd_two_row_gemv] requires matching BF16/FP32 inputs");
  }
  if (x.ndim() < 2 || x.shape(-2) != 2 ||
      x.size() != 2 * x.shape(-1) || weight.ndim() != 2 ||
      weight.shape(1) != x.shape(-1)) {
    throw std::invalid_argument(
        "[expert_ssd_two_row_gemv] invalid two-row geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(0);
  if (K < 16 * N || N < 4 || N % 4 != 0) {
    throw std::invalid_argument(
        "[expert_ssd_two_row_gemv] geometry does not select the canonical width-one GEMV specialization");
  }
  Shape output_shape = x.shape();
  output_shape.back() = N;
  auto primitive =
      std::make_shared<ExpertSSDTwoRowGEMV>(to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape}, {x.dtype()}, primitive, {x, weight})[0];
}

array expert_ssd_three_row_gemv(
    const array& x,
    const array& weight,
    StreamOrDevice s) {
  if ((x.dtype() != bfloat16 && x.dtype() != float32) ||
      weight.dtype() != x.dtype()) {
    throw std::invalid_argument(
        "[expert_ssd_three_row_gemv] requires matching BF16/FP32 inputs");
  }
  if (x.ndim() < 2 || x.shape(-2) != 3 ||
      x.size() != 3 * x.shape(-1) || weight.ndim() != 2 ||
      weight.shape(1) != x.shape(-1)) {
    throw std::invalid_argument(
        "[expert_ssd_three_row_gemv] invalid three-row geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(0);
  if (K < 16 * N || N < 4 || N % 4 != 0) {
    throw std::invalid_argument(
        "[expert_ssd_three_row_gemv] geometry does not select the canonical width-one GEMV specialization");
  }
  Shape output_shape = x.shape();
  output_shape.back() = N;
  auto primitive =
      std::make_shared<ExpertSSDThreeRowGEMV>(to_stream(s, Device::gpu));
  return array::make_arrays(
      {output_shape}, {x.dtype()}, primitive, {x, weight})[0];
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

array expert_ssd_scalex_mxfp4_grouped_qmv(
    const array& x,
    const array& weight,
    const array& scale_records,
    const array& routes,
    uint32_t projection,
    uint32_t top_k,
    StreamOrDevice s) {
  if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
      scale_records.dtype() != uint8 ||
      (routes.dtype() != uint32 && routes.dtype() != int32) ||
      (projection != 0 && projection != 2) || top_k == 0) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_grouped_qmv] incompatible dtype or projection");
  }
  const auto input_rows = x.size() / x.shape(-1);
  if (x.ndim() < 2 || x.size() % x.shape(-1) != 0 ||
      routes.ndim() != 1 || routes.size() == 0 ||
      routes.size() != input_rows * top_k || weight.ndim() != 3 ||
      scale_records.ndim() != 2 ||
      weight.shape(0) != scale_records.shape(0)) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_grouped_qmv] invalid wide-prompt geometry");
  }
  const int K = x.shape(-1);
  const int N = weight.shape(-2);
  if (K != 4096 || N != 2048 || weight.shape(-1) * 8 != K) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_grouped_qmv] unsupported target geometry");
  }
  Shape output_shape{static_cast<ShapeElem>(routes.size()), 1, N};
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4GroupedQMV>(
      to_stream(s, Device::gpu), projection, top_k);
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x, weight, scale_records, routes})[0];
}

array expert_ssd_scalex_mxfp4_qmv_two_bank(
    const array& x,
    const array& private_weight,
    const array& private_scale_records,
    const array& shared_weight,
    const array& shared_scale_records,
    const array& routes,
    const array& bank_routes,
    uint32_t projection,
    StreamOrDevice s) {
  const bool route_dtype =
      routes.dtype() == uint32 || routes.dtype() == int32;
  if (x.dtype() != bfloat16 || private_weight.dtype() != uint32 ||
      shared_weight.dtype() != uint32 ||
      private_scale_records.dtype() != uint8 ||
      shared_scale_records.dtype() != uint8 || !route_dtype ||
      bank_routes.dtype() != routes.dtype() || projection > 2) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_two_bank] incompatible dtype or projection");
  }
  const bool input_rows_match =
      projection == 1 ? x.size() == routes.size() * x.shape(-1)
                      : x.size() == x.shape(-1);
  if (x.ndim() < 2 || x.shape(-2) != 1 || !input_rows_match ||
      routes.ndim() != 1 || routes.size() == 0 ||
      bank_routes.ndim() != 1 || bank_routes.size() != routes.size() ||
      private_weight.ndim() != 3 || shared_weight.ndim() != 3 ||
      private_scale_records.ndim() != 2 ||
      shared_scale_records.ndim() != 2 ||
      private_weight.shape(0) != private_scale_records.shape(0) ||
      shared_weight.shape(0) != shared_scale_records.shape(0) ||
      private_weight.shape(-2) != shared_weight.shape(-2) ||
      private_weight.shape(-1) != shared_weight.shape(-1)) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_two_bank] invalid fixed-decode geometry");
  }
  const int K = x.shape(-1);
  const int N = private_weight.shape(-2);
  const bool expected_geometry =
      (projection == 1 && K == 2048 && N == 4096) ||
      (projection != 1 && K == 4096 && N == 2048);
  if (!expected_geometry || private_weight.shape(-1) * 8 != K) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_two_bank] unsupported target geometry");
  }
  Shape output_shape{static_cast<ShapeElem>(routes.size()), 1, N};
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4QMVTwoBank>(
      to_stream(s, Device::gpu), projection);
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x,
       private_weight,
       private_scale_records,
       shared_weight,
       shared_scale_records,
       routes,
       bank_routes})[0];
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

array expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank(
    const array& x,
    const array& private_weight,
    const array& private_scale_records,
    const array& shared_weight,
    const array& shared_scale_records,
    const array& weight_routes,
    const array& scale_routes,
    const array& bank_routes,
    uint32_t projection,
    StreamOrDevice s) {
  const bool route_dtype =
      weight_routes.dtype() == uint32 || weight_routes.dtype() == int32;
  if (x.dtype() != bfloat16 || private_weight.dtype() != uint32 ||
      shared_weight.dtype() != uint32 ||
      private_scale_records.dtype() != uint8 ||
      shared_scale_records.dtype() != uint8 || !route_dtype ||
      scale_routes.dtype() != weight_routes.dtype() ||
      bank_routes.dtype() != weight_routes.dtype() || projection > 2) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank] incompatible dtype or projection");
  }
  const bool input_rows_match = projection == 1
      ? x.size() == weight_routes.size() * x.shape(-1)
      : x.size() == x.shape(-1);
  if (x.ndim() < 2 || x.shape(-2) != 1 || !input_rows_match ||
      weight_routes.ndim() != 1 || weight_routes.size() == 0 ||
      scale_routes.ndim() != 1 ||
      scale_routes.size() != weight_routes.size() ||
      bank_routes.ndim() != 1 ||
      bank_routes.size() != weight_routes.size() ||
      private_weight.ndim() != 3 || shared_weight.ndim() != 3 ||
      private_scale_records.ndim() != 2 ||
      shared_scale_records.ndim() != 2 ||
      private_weight.shape(0) != private_scale_records.shape(0) ||
      shared_weight.shape(0) != shared_scale_records.shape(0) ||
      private_weight.shape(-2) != shared_weight.shape(-2) ||
      private_weight.shape(-1) != shared_weight.shape(-1)) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank] invalid fixed-decode geometry");
  }
  const int K = x.shape(-1);
  const int N = private_weight.shape(-2);
  const bool expected_geometry = (projection == 1 && K == 2048 && N == 4096) ||
      (projection != 1 && K == 4096 && N == 2048);
  if (!expected_geometry || private_weight.shape(-1) * 8 != K) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank] unsupported target geometry");
  }
  Shape output_shape{static_cast<ShapeElem>(weight_routes.size()), 1, N};
  auto primitive =
      std::make_shared<ExpertSSDScaleXMXFP4QMVSplitRoutesTwoBank>(
          to_stream(s, Device::gpu), projection);
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x,
       private_weight,
       private_scale_records,
       shared_weight,
       shared_scale_records,
       weight_routes,
       scale_routes,
       bank_routes})[0];
}

static array expert_ssd_scalex_mxfp4_fixed_down_reduce_two_bank(
    const array& x,
    const array& private_weight,
    const array& private_scale_records,
    const array& shared_weight,
    const array& shared_scale_records,
    const array& weight_routes,
    const array& scale_routes,
    const array& bank_routes,
    const array& scores,
    const array& shared,
    int width,
    StreamOrDevice s) {
  const bool route_dtype =
      weight_routes.dtype() == uint32 || weight_routes.dtype() == int32;
  const int route_count = width * 6;
  if ((width != 2 && width != 3) || x.dtype() != bfloat16 ||
      private_weight.dtype() != uint32 || shared_weight.dtype() != uint32 ||
      private_scale_records.dtype() != uint8 ||
      shared_scale_records.dtype() != uint8 || !route_dtype ||
      scale_routes.dtype() != weight_routes.dtype() ||
      bank_routes.dtype() != weight_routes.dtype() ||
      scores.dtype() != float32 || shared.dtype() != bfloat16) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_fixed_down_reduce_two_bank] incompatible dtype");
  }
  if (x.ndim() < 2 || x.shape(-2) != 1 || x.shape(-1) != 2048 ||
      x.size() != route_count * 2048 || private_weight.ndim() != 3 ||
      shared_weight.ndim() != 3 || private_weight.shape(-2) != 4096 ||
      shared_weight.shape(-2) != 4096 ||
      private_weight.shape(-1) * 8 != 2048 ||
      shared_weight.shape(-1) != private_weight.shape(-1) ||
      private_scale_records.ndim() != 2 ||
      shared_scale_records.ndim() != 2 ||
      private_weight.shape(0) != private_scale_records.shape(0) ||
      shared_weight.shape(0) != shared_scale_records.shape(0) ||
      weight_routes.ndim() != 1 || weight_routes.size() != route_count ||
      scale_routes.ndim() != 1 || scale_routes.size() != route_count ||
      bank_routes.ndim() != 1 || bank_routes.size() != route_count ||
      scores.size() != route_count || shared.size() != width * 4096) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_fixed_down_reduce_two_bank] invalid fixed-width geometry");
  }
  Shape output_shape{width, 1, 4096};
  auto primitive =
      std::make_shared<ExpertSSDScaleXMXFP4FixedDownReduceTwoBank>(
          to_stream(s, Device::gpu), width);
  return array::make_arrays(
      {output_shape},
      {bfloat16},
      primitive,
      {x,
       private_weight,
       private_scale_records,
       shared_weight,
       shared_scale_records,
       weight_routes,
       scale_routes,
       bank_routes,
       scores,
       shared})[0];
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
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4FixedDownReduce>(
      to_stream(s, Device::gpu), 2);
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

array expert_ssd_scalex_mxfp4_width2_down_reduce_two_bank(
    const array& x,
    const array& private_weight,
    const array& private_scale_records,
    const array& shared_weight,
    const array& shared_scale_records,
    const array& weight_routes,
    const array& scale_routes,
    const array& bank_routes,
    const array& scores,
    const array& shared,
    StreamOrDevice s) {
  return expert_ssd_scalex_mxfp4_fixed_down_reduce_two_bank(
      x,
      private_weight,
      private_scale_records,
      shared_weight,
      shared_scale_records,
      weight_routes,
      scale_routes,
      bank_routes,
      scores,
      shared,
      2,
      s);
}

array expert_ssd_scalex_mxfp4_width3_down_reduce(
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
        "[expert_ssd_scalex_mxfp4_width3_down_reduce] incompatible dtype");
  }
  if (x.ndim() < 2 || x.shape(-2) != 1 || x.shape(-1) != 2048 ||
      x.size() != 18 * 2048 || weight.ndim() != 3 ||
      weight.shape(-2) != 4096 || weight.shape(-1) * 8 != 2048 ||
      scale_records.ndim() != 2 ||
      weight_routes.ndim() != 1 || weight_routes.size() != 18 ||
      scale_routes.ndim() != 1 || scale_routes.size() != 18 ||
      scores.size() != 18 || shared.size() != 3 * 4096) {
    throw std::invalid_argument(
        "[expert_ssd_scalex_mxfp4_width3_down_reduce] invalid width-three geometry");
  }
  Shape output_shape{3, 1, 4096};
  auto primitive = std::make_shared<ExpertSSDScaleXMXFP4FixedDownReduce>(
      to_stream(s, Device::gpu), 3);
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

array expert_ssd_scalex_mxfp4_width3_down_reduce_two_bank(
    const array& x,
    const array& private_weight,
    const array& private_scale_records,
    const array& shared_weight,
    const array& shared_scale_records,
    const array& weight_routes,
    const array& scale_routes,
    const array& bank_routes,
    const array& scores,
    const array& shared,
    StreamOrDevice s) {
  return expert_ssd_scalex_mxfp4_fixed_down_reduce_two_bank(
      x,
      private_weight,
      private_scale_records,
      shared_weight,
      shared_scale_records,
      weight_routes,
      scale_routes,
      bank_routes,
      scores,
      shared,
      3,
      s);
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

array expert_ssd_scalex_conditional_m0_two_bank(
    const array& indices,
    const array& x,
    const array& scores,
    const array& shared,
    const array& private_scale_records,
    const array& private_gate_weight,
    const array& private_down_weight,
    const array& private_up_weight,
    const array& shared_scale_records,
    const array& shared_gate_weight,
    const array& shared_down_weight,
    const array& shared_up_weight,
    const array& gate_directory,
    const array& down_directory,
    const array& bank_directory,
    const array& gate_routes_scratch,
    const array& down_routes_scratch,
    const array& bank_routes_scratch,
    const array& all_hit_scratch,
    const array& up_scratch,
    const array& gate_scratch,
    const array& activated_scratch,
    const array& routed_scratch,
    const array& indirect_scratch,
    float swiglu_limit,
    StreamOrDevice s) {
  const int width = x.ndim() == 3 ? x.shape(1) : 0;
  const auto private_rows = private_gate_weight.shape(0);
  const auto shared_rows = shared_gate_weight.shape(0);
  if ((width != 1 && width != 2) || x.dtype() != bfloat16 ||
      x.shape(0) != 1 || x.shape(2) != 4096 || indices.dtype() != int32 ||
      indices.shape() != Shape{1, width, 6} || scores.dtype() != float32 ||
      scores.shape() != Shape{1, width, 6} || shared.dtype() != bfloat16 ||
      shared.shape() != x.shape() ||
      private_scale_records.dtype() != uint8 ||
      shared_scale_records.dtype() != uint8 ||
      private_scale_records.ndim() != 2 || shared_scale_records.ndim() != 2 ||
      private_gate_weight.dtype() != uint32 ||
      private_down_weight.dtype() != uint32 ||
      private_up_weight.dtype() != uint32 ||
      shared_gate_weight.dtype() != uint32 ||
      shared_down_weight.dtype() != uint32 ||
      shared_up_weight.dtype() != uint32 ||
      private_gate_weight.shape() != private_up_weight.shape() ||
      shared_gate_weight.shape() != shared_up_weight.shape() ||
      private_gate_weight.ndim() != 3 || shared_gate_weight.ndim() != 3 ||
      private_gate_weight.shape(1) != 2048 ||
      private_gate_weight.shape(2) != 512 ||
      shared_gate_weight.shape(1) != 2048 ||
      shared_gate_weight.shape(2) != 512 ||
      private_down_weight.ndim() != 3 || shared_down_weight.ndim() != 3 ||
      private_down_weight.shape(0) != private_rows ||
      shared_down_weight.shape(0) != shared_rows ||
      private_down_weight.shape(1) != 4096 ||
      private_down_weight.shape(2) != 256 ||
      shared_down_weight.shape(1) != 4096 ||
      shared_down_weight.shape(2) != 256 ||
      private_scale_records.shape(0) != private_rows ||
      shared_scale_records.shape(0) != shared_rows ||
      gate_directory.dtype() != int32 || down_directory.dtype() != int32 ||
      bank_directory.dtype() != int32 || gate_directory.ndim() != 1 ||
      down_directory.ndim() != 1 || bank_directory.ndim() != 1 ||
      gate_directory.shape() != down_directory.shape() ||
      gate_directory.shape() != bank_directory.shape() ||
      gate_routes_scratch.dtype() != uint32 ||
      gate_routes_scratch.shape() != Shape{12} ||
      down_routes_scratch.dtype() != uint32 ||
      down_routes_scratch.shape() != Shape{12} ||
      bank_routes_scratch.dtype() != uint32 ||
      bank_routes_scratch.shape() != Shape{12} ||
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
        "[expert_ssd_scalex_conditional_m0_two_bank] incompatible inputs");
  }
  auto primitive = std::make_shared<ExpertSSDScaleXConditionalM0TwoBank>(
      to_stream(s, Device::gpu), width, swiglu_limit);
  return array(
      x.shape(),
      bfloat16,
      primitive,
      {indices,
       x,
       scores,
       shared,
       private_scale_records,
       private_gate_weight,
       private_down_weight,
       private_up_weight,
       shared_scale_records,
       shared_gate_weight,
       shared_down_weight,
       shared_up_weight,
       gate_directory,
       down_directory,
       bank_directory,
       gate_routes_scratch,
       down_routes_scratch,
       bank_routes_scratch,
       all_hit_scratch,
       up_scratch,
       gate_scratch,
       activated_scratch,
       routed_scratch,
       indirect_scratch});
}

} // namespace mlx::core
