// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/dsv4_stateful_attention.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

class Dsv4StatefulAttention : public Primitive {
 public:
  Dsv4StatefulAttention(
      Stream stream,
      int cache_index,
      bool sparse,
      float scale)
      : Primitive(stream),
        cache_index_(cache_index),
        sparse_(sparse),
        scale_(scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[Dsv4StatefulAttention] CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    if (inputs.size() != 7 || outputs.size() != 2) {
      throw std::runtime_error(
          "[Dsv4StatefulAttention] invalid input/output arity");
    }
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::runtime_error(
            "[Dsv4StatefulAttention] inputs must be row-contiguous");
      }
    }
    for (auto& output : outputs) {
      output.set_data(allocator::malloc(output.nbytes()));
    }

    const auto& queries = inputs[0];
    const auto& pooled = inputs[3];
    const auto& indices = inputs[4];
    const int width = queries.shape(0);
    const int heads = queries.shape(1);
    const int dim = queries.shape(2);
    const int pool_rows = pooled.shape(0);
    const int index_width = indices.shape(1);

    auto& device = metal::device(stream().device);
    auto& encoder = metal::get_command_encoder(stream());

    auto* attention = device.get_kernel("dsv4_stateful_attention_bf16");
    encoder.set_compute_pipeline_state(attention);
    int binding = 0;
    for (const auto& input : inputs) {
      encoder.set_input_array(input, binding++);
    }
    encoder.set_output_array(outputs[0], binding++);
    encoder.set_bytes(width, binding++);
    encoder.set_bytes(heads, binding++);
    encoder.set_bytes(dim, binding++);
    encoder.set_bytes(pool_rows, binding++);
    encoder.set_bytes(index_width, binding++);
    encoder.set_bytes(cache_index_, binding++);
    const int sparse_value = sparse_ ? 1 : 0;
    encoder.set_bytes(sparse_value, binding++);
    encoder.set_bytes(scale_, binding++);
    const size_t group_size = std::min<size_t>(
        256, attention->maxTotalThreadsPerThreadgroup());
    if (group_size != 256) {
      throw std::runtime_error(
          "[Dsv4StatefulAttention] device cannot launch 256-thread group");
    }
    encoder.dispatch_threadgroups(
        MTL::Size(heads, width, 1), MTL::Size(256, 1, 1));

    auto* update = device.get_kernel("dsv4_stateful_ring_prefixes_bf16");
    encoder.set_compute_pipeline_state(update);
    binding = 0;
    encoder.set_input_array(inputs[1], binding++);
    encoder.set_input_array(inputs[2], binding++);
    encoder.set_output_array(outputs[1], binding++);
    encoder.set_bytes(width, binding++);
    encoder.set_bytes(dim, binding++);
    encoder.set_bytes(cache_index_, binding++);
    const size_t elements = static_cast<size_t>(width) * 128 * dim;
    const size_t copy_group = std::min<size_t>(
        256, update->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(
        MTL::Size(elements, 1, 1), MTL::Size(copy_group, 1, 1));
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto* candidate = dynamic_cast<const Dsv4StatefulAttention*>(&other);
    return candidate != nullptr && candidate->cache_index_ == cache_index_ &&
        candidate->sparse_ == sparse_ && candidate->scale_ == scale_;
  }

  DEFINE_NAME(Dsv4StatefulAttention)

 private:
  int cache_index_;
  bool sparse_;
  float scale_;
};

} // namespace

std::vector<array> dsv4_stateful_attention(
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
    StreamOrDevice s) {
  if (queries.dtype() != bfloat16 || new_keys.dtype() != bfloat16 ||
      local_ring.dtype() != bfloat16 || pooled.dtype() != bfloat16 ||
      indices.dtype() != int32 || pool_counts.dtype() != int32 ||
      sinks.dtype() != float32) {
    throw std::invalid_argument(
        "[dsv4_stateful_attention] requires BF16 state, int32 indices/counts and FP32 sinks");
  }
  if (queries.ndim() != 3 || new_keys.ndim() != 2 ||
      local_ring.ndim() != 2 || pooled.ndim() != 2 ||
      indices.ndim() != 2 || pool_counts.ndim() != 1 ||
      sinks.ndim() != 1) {
    throw std::invalid_argument(
        "[dsv4_stateful_attention] invalid tensor ranks");
  }
  const int width = queries.shape(0);
  const int heads = queries.shape(1);
  const int dim = queries.shape(2);
  if ((width != 2 && width != 3) || dim != 512 || heads != 64 ||
      new_keys.shape(0) != width || new_keys.shape(1) != dim ||
      local_ring.shape(0) != 128 || local_ring.shape(1) != dim ||
      pooled.shape(1) != dim || indices.shape(0) != width ||
      pool_counts.shape(0) != width || sinks.shape(0) != heads ||
      cache_index < 0 || cache_index > 128 ||
      indices.shape(1) < 1) {
    throw std::invalid_argument(
        "[dsv4_stateful_attention] unsupported fixed-width geometry");
  }
  if (sparse && indices.shape(1) > 512) {
    throw std::invalid_argument(
        "[dsv4_stateful_attention] sparse selection exceeds 512 rows");
  }

  Shape attended_shape{width, heads, dim};
  Shape rings_shape{width, 128, dim};
  auto primitive = std::make_shared<Dsv4StatefulAttention>(
      to_stream(s, Device::gpu), cache_index, sparse, scale);
  return array::make_arrays(
      {std::move(attended_shape), std::move(rings_shape)},
      {bfloat16, bfloat16},
      std::move(primitive),
      {queries, new_keys, local_ring, pooled, indices, pool_counts, sinks});
}

} // namespace mlx::core
