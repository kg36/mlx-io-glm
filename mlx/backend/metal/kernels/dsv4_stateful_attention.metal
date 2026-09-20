// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

METAL_FUNC bfloat16_t dsv4_stateful_key(
    const device bfloat16_t* new_keys,
    const device bfloat16_t* local_ring,
    const device bfloat16_t* pooled,
    const device int* indices,
    int position,
    int logical_row,
    int column,
    int dim,
    int pool_rows,
    int pool_limit,
    int index_width,
    int cache_index,
    bool sparse,
    thread bool& valid) {
  if (logical_row < 128) {
    const int historical = 127 - position;
    valid = true;
    if (logical_row < historical) {
      const int physical =
          (cache_index + position + 1 + logical_row) & 127;
      return local_ring[physical * dim + column];
    }
    const int appended = logical_row - historical;
    return new_keys[appended * dim + column];
  }

  const int selected_row = logical_row - 128;
  const int pool_index = sparse
      ? indices[position * index_width + selected_row]
      : selected_row;
  valid = pool_index >= 0 && pool_index < pool_rows && pool_index < pool_limit;
  return valid ? pooled[pool_index * dim + column] : bfloat16_t(0.0f);
}

[[kernel]] void dsv4_stateful_attention_bf16(
    const device bfloat16_t* queries [[buffer(0)]],
    const device bfloat16_t* new_keys [[buffer(1)]],
    const device bfloat16_t* local_ring [[buffer(2)]],
    const device bfloat16_t* pooled [[buffer(3)]],
    const device int* indices [[buffer(4)]],
    const device int* pool_counts [[buffer(5)]],
    const device float* sinks [[buffer(6)]],
    device bfloat16_t* output [[buffer(7)]],
    const constant int& width [[buffer(8)]],
    const constant int& heads [[buffer(9)]],
    const constant int& dim [[buffer(10)]],
    const constant int& pool_rows [[buffer(11)]],
    const constant int& index_width [[buffer(12)]],
    const constant int& raw_cache_index [[buffer(13)]],
    const constant int& sparse_value [[buffer(14)]],
    const constant float& scale [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int head = int(group.x);
  const int position = int(group.y);
  if (position >= width || head >= heads) {
    return;
  }
  const bool sparse = sparse_value != 0;
  const int cache_index = raw_cache_index == 128 ? 0 : raw_cache_index;
  const int requested_pools = sparse ? index_width : pool_counts[position];
  const int pool_limit = pool_counts[position];
  const int total_rows = 128 + requested_pools;
  const device bfloat16_t* query =
      queries + (position * heads + head) * dim;

  threadgroup float scores[64];
  threadgroup bfloat16_t probabilities[64];
  threadgroup float numerator[512];
  threadgroup float shared_maximum;
  threadgroup float shared_factor;
  threadgroup float shared_block_sum;
  threadgroup float shared_denominator;

  for (int column = int(tid); column < dim; column += 256) {
    numerator[column] = 0.0f;
  }
  if (tid == 0) {
    shared_maximum = -INFINITY;
    shared_denominator = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int block = 0; block < total_rows; block += 64) {
    const int block_width = min(64, total_rows - block);
    for (int base = 0; base < block_width; base += 8) {
      const int row = base + int(simd_gid);
      if (row < block_width) {
        const int logical_row = block + row;
        bool row_valid = true;
        float sum = 0.0f;
        float correction = 0.0f;
        for (int column = int(lane); column < dim; column += 32) {
          bool element_valid = true;
          const float key = float(dsv4_stateful_key(
              new_keys,
              local_ring,
              pooled,
              indices,
              position,
              logical_row,
              column,
              dim,
              pool_rows,
              pool_limit,
              index_width,
              cache_index,
              sparse,
              element_valid));
          row_valid = row_valid && element_valid;
          const float product = float(query[column]) * key;
          const float adjusted = product - correction;
          const float next = sum + adjusted;
          correction = (next - sum) - adjusted;
          sum = next;
        }
        const float reduced = simd_sum(sum);
        if (lane == 0) {
          scores[row] = row_valid ? reduced * scale : -INFINITY;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      float block_max = -INFINITY;
      for (int row = 0; row < block_width; ++row) {
        block_max = max(block_max, scores[row]);
      }
      const float next_maximum = max(shared_maximum, block_max);
      const float safe_maximum = isfinite(next_maximum) ? next_maximum : 0.0f;
      shared_factor = metal::precise::exp(shared_maximum - safe_maximum);
      shared_maximum = next_maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float safe_maximum = isfinite(shared_maximum) ? shared_maximum : 0.0f;
    if (tid < uint(block_width)) {
      scores[tid] = metal::precise::exp(scores[tid] - safe_maximum);
      probabilities[tid] = bfloat16_t(scores[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      // For this shape MLX's reduction plan has one reduction row and uses
      // thread_reduce: one thread consumes the contiguous row from left to
      // right.  Do not replace this with a simd tree; its FP32 rounding is a
      // different arithmetic contract.
      float block_sum = 0.0f;
      for (int row = 0; row < block_width; ++row) {
        block_sum += scores[row];
      }
      // The canonical graph materializes the multiply before adding the
      // independently materialized reduction.  Publish the product through
      // threadgroup memory so Metal cannot contract these into one fma.
      shared_block_sum = block_sum;
      shared_denominator = shared_denominator * shared_factor;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      shared_denominator = shared_denominator + shared_block_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int column = int(tid); column < dim; column += 256) {
      float value = numerator[column] * shared_factor;
      for (int row = 0; row < block_width; ++row) {
        bool valid = true;
        const int logical_row = block + row;
        const float key_value = float(dsv4_stateful_key(
            new_keys,
            local_ring,
            pooled,
            indices,
            position,
            logical_row,
            column,
            dim,
            pool_rows,
            pool_limit,
            index_width,
            cache_index,
            sparse,
            valid));
        value = fma(float(probabilities[row]), key_value, value);
      }
      numerator[column] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    shared_denominator +=
        metal::precise::exp(sinks[head] - shared_maximum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat16_t* out = output + (position * heads + head) * dim;
  for (int column = int(tid); column < dim; column += 256) {
    out[column] = bfloat16_t(numerator[column] / shared_denominator);
  }
}

[[kernel]] void dsv4_stateful_ring_prefixes_bf16(
    const device bfloat16_t* new_keys [[buffer(0)]],
    const device bfloat16_t* local_ring [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    const constant int& width [[buffer(3)]],
    const constant int& dim [[buffer(4)]],
    const constant int& raw_cache_index [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint ring_elements = uint(128 * dim);
  const uint position = index / ring_elements;
  if (position >= uint(width)) {
    return;
  }
  const uint within = index % ring_elements;
  const int row = int(within / uint(dim));
  const int column = int(within % uint(dim));
  const int cache_index = raw_cache_index == 128 ? 0 : raw_cache_index;
  bfloat16_t value = local_ring[row * dim + column];
  for (int appended = 0; appended <= int(position); ++appended) {
    if (((cache_index + appended) & 127) == row) {
      value = new_keys[appended * dim + column];
    }
  }
  output[index] = value;
}
