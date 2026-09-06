// Copyright © 2025-2026 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

// LivSeek ScaleX Mode B keeps the complete lossless Mode-A record resident.
// The record layout is:
//   24-byte header, packed palette codes, sorted uint32 exception positions,
//   uint8 exception values, even-byte padding, uint16 tile prefixes.
// No hydrated scale tensor or model-wide scale sidecar is required.
METAL_FUNC uint dsv4_scalex_load_u32(const device uint8_t* p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
      (uint(p[3]) << 24);
}

METAL_FUNC uint8_t dsv4_scalex_primary_value(
    const device uint8_t* record,
    const device uint8_t* primary,
    uint codec,
    uint raw_index) {
  if (codec == 0u) {
    return primary[raw_index];
  } else if (codec == 1u) {
    const uint code =
        (uint(primary[raw_index >> 3]) >> (raw_index & 7u)) & 1u;
    return record[5u + code];
  } else {
    const uint code =
        (uint(primary[raw_index >> 2]) >> ((raw_index & 3u) << 1)) & 3u;
    return record[5u + code];
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void dsv4_scalex_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* record,
    threadgroup uint8_t* local_scales,
    uint scale_projection_offset,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const uint codec = uint(record[4]);
  const uint primary_size = dsv4_scalex_load_u32(record + 16);
  const uint exception_count = dsv4_scalex_load_u32(record + 20);
  const device uint8_t* primary = record + 24;
  const device uint8_t* positions = primary + primary_size;
  const device uint8_t* exception_values =
      positions + ulong(exception_count) * 4ul;
  const uint encoded_size = 24u + primary_size + 5u * exception_count;
  const device uint8_t* tile_prefix =
      record + ((encoded_size + 1u) & ~1u);

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  // Up/gate SIMD groups own one 512-scale tile; down groups own one aligned
  // half. The transient uint16 prefix appended by the I/O worker resolves the
  // containing tile's sparse interval without two binary searches. The patch
  // loop below rejects entries outside a down group's half.
  uint exception_range = 0;
  if (simd_lid == 0u) {
    const uint group_start =
        scale_projection_offset + uint(out_row) * uint(in_vec_size_g);
    const uint tile = group_start >> 9;
    exception_range = dsv4_scalex_load_u32(tile_prefix + 2u * tile);
  }
  exception_range = simd_shuffle(exception_range, ushort(0));
  const uint exception_begin = exception_range & 0xffffu;
  const uint exception_end = exception_range >> 16;

  // Decode only this SIMD group's four output rows into a 512-byte local
  // tile. Sparse exceptions are applied once before QMV instead of being
  // rescanned for every dot-product scale lookup.
  const uint group_start =
      scale_projection_offset + uint(out_row) * uint(in_vec_size_g);
  const uint group_scale_count =
      uint(results_per_simdgroup * in_vec_size_g);
  if (codec == 1u) {
    const uint8_t palette0 = record[5];
    const uint8_t palette1 = record[6];
    for (uint local = simd_lid;
         local < group_scale_count;
         local += SIMD_SIZE) {
      const uint raw_index = group_start + local;
      const uint code =
          (uint(primary[raw_index >> 3]) >> (raw_index & 7u)) & 1u;
      local_scales[local] = code == 0u ? palette0 : palette1;
    }
  } else {
    for (uint local = simd_lid;
         local < group_scale_count;
         local += SIMD_SIZE) {
      local_scales[local] = dsv4_scalex_primary_value(
          record, primary, codec, group_start + local);
    }
  }
  for (uint index = exception_begin + simd_lid;
       index < exception_end;
       index += SIMD_SIZE) {
    const uint position =
        dsv4_scalex_load_u32(positions + ulong(index) * 4ul);
    const uint local = position - group_start;
    if (local < group_scale_count) {
      local_scales[local] = exception_values[index];
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);
    const uint scale_group =
        uint(simd_lid / scale_step_per_thread + k / group_size);

    for (int row = 0; row < results_per_simdgroup; ++row) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const uint8_t encoded_scale =
          local_scales[uint(row) * uint(in_vec_size_g) + scale_group];
      const U scale = dequantize_scale<U, group_size>(encoded_scale);
      result[row] +=
          qdot<U, values_per_thread, bits>(wl, x_thread, scale);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0u) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

[[kernel]] void dsv4_mxfp4_pair_bf16(
    const device uint32_t* up_weight [[buffer(0)]],
    const device uint8_t* up_scales [[buffer(1)]],
    const device uint32_t* gate_weight [[buffer(2)]],
    const device uint8_t* gate_scales [[buffer(3)]],
    const device bfloat16_t* x [[buffer(4)]],
    const device uint32_t* routes [[buffer(5)]],
    device bfloat16_t* up_output [[buffer(6)]],
    device bfloat16_t* gate_output [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route_position = tid.z;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) {
    if (simd_lid < 4u) {
      const uint row = tid.y * 8u + simd_gid * 4u + simd_lid;
      if (row < uint(out_vec_size)) {
        up_output[ulong(route_position) * ulong(out_vec_size) + row] =
            bfloat16_t(0.0f);
        gate_output[ulong(route_position) * ulong(out_vec_size) + row] =
            bfloat16_t(0.0f);
      }
    }
    return;
  }
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const ulong scale_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 32);
  const uint3 qmv_tid(0u, tid.y, 0u);
  fp_qmv_fast_impl<bfloat16_t, 32, 4>(
      up_weight + ulong(slot) * weight_stride,
      up_scales + ulong(slot) * scale_stride,
      nullptr,
      x,
      up_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
  fp_qmv_fast_impl<bfloat16_t, 32, 4>(
      gate_weight + ulong(slot) * weight_stride,
      gate_scales + ulong(slot) * scale_stride,
      nullptr,
      x,
      gate_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

// Two independent width-one projections sharing one dispatch. Each row
// enters fp_qmv_fast_impl with tid.x == 0, preserving the exact reduction
// order of two canonical QMV calls while avoiding a width-two GEMM path.
[[kernel]] void dsv4_mxfp4_two_row_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 2u; ++row) {
    fp_qmv_fast_impl<bfloat16_t, 32, 4>(
        weight,
        scales,
        nullptr,
        x + ulong(row) * ulong(in_vec_size),
        output + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

// Batched-weight companion for MultiLinear output groups. Grid z selects one
// independent matrix and its corresponding pair of input rows.
[[kernel]] void dsv4_mxfp4_grouped_two_row_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const ulong group = ulong(tid.z);
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const ulong scale_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 32);
  const ulong input_stride = 2ul * ulong(in_vec_size);
  const ulong output_stride = 2ul * ulong(out_vec_size);
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 2u; ++row) {
    fp_qmv_fast_impl<bfloat16_t, 32, 4>(
        weight + group * weight_stride,
        scales + group * scale_stride,
        nullptr,
        x + group * input_stride + ulong(row) * ulong(in_vec_size),
        output + group * output_stride + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

[[kernel]] void dsv4_mxfp4_two_row_f32(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 2u; ++row) {
    fp_qmv_fast_impl<float, 32, 4>(
        weight,
        scales,
        nullptr,
        x + ulong(row) * ulong(in_vec_size),
        output + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

[[kernel]] void dsv4_mxfp4_grouped_two_row_f32(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const ulong group = ulong(tid.z);
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const ulong scale_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 32);
  const ulong input_stride = 2ul * ulong(in_vec_size);
  const ulong output_stride = 2ul * ulong(out_vec_size);
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 2u; ++row) {
    fp_qmv_fast_impl<float, 32, 4>(
        weight + group * weight_stride,
        scales + group * scale_stride,
        nullptr,
        x + group * input_stride + ulong(row) * ulong(in_vec_size),
        output + group * output_stride + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

// Three independent width-one projections sharing one dispatch. Keep each
// row on the canonical QMV reduction path; the verifier positions are a
// dispatch batch, not a width-three GEMM.
[[kernel]] void dsv4_mxfp4_three_row_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 3u; ++row) {
    fp_qmv_fast_impl<bfloat16_t, 32, 4>(
        weight,
        scales,
        nullptr,
        x + ulong(row) * ulong(in_vec_size),
        output + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

[[kernel]] void dsv4_mxfp4_grouped_three_row_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const ulong group = ulong(tid.z);
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const ulong scale_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 32);
  const ulong input_stride = 3ul * ulong(in_vec_size);
  const ulong output_stride = 3ul * ulong(out_vec_size);
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 3u; ++row) {
    fp_qmv_fast_impl<bfloat16_t, 32, 4>(
        weight + group * weight_stride,
        scales + group * scale_stride,
        nullptr,
        x + group * input_stride + ulong(row) * ulong(in_vec_size),
        output + group * output_stride + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

[[kernel]] void dsv4_mxfp4_three_row_f32(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 3u; ++row) {
    fp_qmv_fast_impl<float, 32, 4>(
        weight,
        scales,
        nullptr,
        x + ulong(row) * ulong(in_vec_size),
        output + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

[[kernel]] void dsv4_mxfp4_grouped_three_row_f32(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* output [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const ulong group = ulong(tid.z);
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const ulong scale_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 32);
  const ulong input_stride = 3ul * ulong(in_vec_size);
  const ulong output_stride = 3ul * ulong(out_vec_size);
  const uint3 qmv_tid(0u, tid.y, 0u);
  for (uint row = 0u; row < 3u; ++row) {
    fp_qmv_fast_impl<float, 32, 4>(
        weight + group * weight_stride,
        scales + group * scale_stride,
        nullptr,
        x + group * input_stride + ulong(row) * ulong(in_vec_size),
        output + group * output_stride + ulong(row) * ulong(out_vec_size),
        in_vec_size,
        out_vec_size,
        qmv_tid,
        simd_gid,
        simd_lid);
  }
}

[[kernel]] void dsv4_mxfp4_masked_down_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    const device uint32_t* routes [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route_position = tid.z;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) {
    if (simd_lid < 4u) {
      const uint row = tid.y * 8u + simd_gid * 4u + simd_lid;
      if (row < uint(out_vec_size)) {
        output[ulong(route_position) * ulong(out_vec_size) + row] =
            bfloat16_t(0.0f);
      }
    }
    return;
  }
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const ulong scale_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 32);
  const uint3 qmv_tid(0u, tid.y, 0u);
  fp_qmv_fast_impl<bfloat16_t, 32, 4>(
      weight + ulong(slot) * weight_stride,
      scales + ulong(slot) * scale_stride,
      nullptr,
      x + ulong(route_position) * ulong(in_vec_size),
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_qmv_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scale_records [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    const device uint32_t* routes [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& record_stride [[buffer(7)]],
    const constant uint& projection [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) {
    if (simd_lid < 4u) {
      const uint row = tid.y * 8u + simd_gid * 4u + simd_lid;
      if (row < uint(out_vec_size)) {
        output[ulong(route_position) * ulong(out_vec_size) + row] =
            bfloat16_t(0.0f);
      }
    }
    return;
  }
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(slot) * ulong(record_stride);
  const device bfloat16_t* route_x =
      projection == 1u
      ? x + ulong(route_position) * ulong(in_vec_size)
      : x;
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      weight + ulong(slot) * weight_stride,
      record,
      scale_tile + simd_gid * 512u,
      projection * scale_count,
      route_x,
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_grouped_qmv_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scale_records [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    const device uint32_t* routes [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& record_stride [[buffer(7)]],
    const constant uint& projection [[buffer(8)]],
    const constant uint& top_k [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) {
    if (simd_lid < 4u) {
      const uint row = tid.y * 8u + simd_gid * 4u + simd_lid;
      if (row < uint(out_vec_size)) {
        output[ulong(route_position) * ulong(out_vec_size) + row] =
            bfloat16_t(0.0f);
      }
    }
    return;
  }
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(slot) * ulong(record_stride);
  const uint input_position = route_position / top_k;
  const device bfloat16_t* route_x =
      x + ulong(input_position) * ulong(in_vec_size);
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      weight + ulong(slot) * weight_stride,
      record,
      scale_tile + simd_gid * 512u,
      projection * scale_count,
      route_x,
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

// One logical QMV over two physical slot banks. Bank selection is uniform for
// every threadgroup because tid.z names one routed expert, so this preserves
// the single-bank dispatch topology without route partitioning.
[[kernel]] void dsv4_scalex_mxfp4_qmv_two_bank_bf16(
    const device uint32_t* private_weight [[buffer(0)]],
    const device uint8_t* private_scale_records [[buffer(1)]],
    const device uint32_t* shared_weight [[buffer(2)]],
    const device uint8_t* shared_scale_records [[buffer(3)]],
    const device bfloat16_t* x [[buffer(4)]],
    const device uint32_t* routes [[buffer(5)]],
    const device uint32_t* bank_routes [[buffer(6)]],
    device bfloat16_t* output [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    const constant int& private_record_stride [[buffer(10)]],
    const constant int& shared_record_stride [[buffer(11)]],
    const constant uint& projection [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) {
    if (simd_lid < 4u) {
      const uint row = tid.y * 8u + simd_gid * 4u + simd_lid;
      if (row < uint(out_vec_size)) {
        output[ulong(route_position) * ulong(out_vec_size) + row] =
            bfloat16_t(0.0f);
      }
    }
    return;
  }
  const bool use_shared = bank_routes[route_position] != 0u;
  const device uint32_t* weight =
      use_shared ? shared_weight : private_weight;
  const device uint8_t* scale_records =
      use_shared ? shared_scale_records : private_scale_records;
  const int record_stride =
      use_shared ? shared_record_stride : private_record_stride;
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(slot) * ulong(record_stride);
  const device bfloat16_t* route_x =
      projection == 1u
      ? x + ulong(route_position) * ulong(in_vec_size)
      : x;
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      weight + ulong(slot) * weight_stride,
      record,
      scale_tile + simd_gid * 512u,
      projection * scale_count,
      route_x,
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_qmv_split_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scale_records [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    const device uint32_t* weight_routes [[buffer(3)]],
    const device uint32_t* scale_routes [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant int& record_stride [[buffer(8)]],
    const constant uint& projection [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  const uint weight_slot = weight_routes[route_position];
  const uint scale_slot = scale_routes[route_position];
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(scale_slot) * ulong(record_stride);
  const device bfloat16_t* route_x =
      projection == 1u
      ? x + ulong(route_position) * ulong(in_vec_size)
      : x;
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      weight + ulong(weight_slot) * weight_stride,
      record,
      scale_tile + simd_gid * 512u,
      projection * scale_count,
      route_x,
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_qmv_split_two_bank_bf16(
    const device uint32_t* private_weight [[buffer(0)]],
    const device uint8_t* private_scale_records [[buffer(1)]],
    const device uint32_t* shared_weight [[buffer(2)]],
    const device uint8_t* shared_scale_records [[buffer(3)]],
    const device bfloat16_t* x [[buffer(4)]],
    const device uint32_t* weight_routes [[buffer(5)]],
    const device uint32_t* scale_routes [[buffer(6)]],
    const device uint32_t* bank_routes [[buffer(7)]],
    device bfloat16_t* output [[buffer(8)]],
    const constant int& in_vec_size [[buffer(9)]],
    const constant int& out_vec_size [[buffer(10)]],
    const constant int& private_record_stride [[buffer(11)]],
    const constant int& shared_record_stride [[buffer(12)]],
    const constant uint& projection [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  const uint weight_slot = weight_routes[route_position];
  const uint scale_slot = scale_routes[route_position];
  const bool use_shared = bank_routes[route_position] != 0u;
  const device uint32_t* weight =
      use_shared ? shared_weight : private_weight;
  const device uint8_t* scale_records =
      use_shared ? shared_scale_records : private_scale_records;
  const int record_stride =
      use_shared ? shared_record_stride : private_record_stride;
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(scale_slot) * ulong(record_stride);
  const device bfloat16_t* route_x =
      projection == 1u
      ? x + ulong(route_position) * ulong(in_vec_size)
      : x;
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      weight + ulong(weight_slot) * weight_stride,
      record,
      scale_tile + simd_gid * 512u,
      projection * scale_count,
      route_x,
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

// Fixed-width/top-six Down projection with the exact BF16 score-reduction
// order and shared-expert add folded into the same dispatch. Width two and
// width three have separate Metal entry points but deliberately share this
// arithmetic body so both preserve canonical width-one route accumulation.
METAL_FUNC void dsv4_scalex_mxfp4_fixed_down_reduce_bf16_impl(
    const device uint32_t* private_weight,
    const device uint8_t* private_scale_records,
    const device uint32_t* shared_weight,
    const device uint8_t* shared_scale_records,
    const device bfloat16_t* x,
    const device uint32_t* weight_routes,
    const device uint32_t* scale_routes,
    const device uint32_t* bank_routes,
    const device float* scores,
    const device bfloat16_t* shared,
    device bfloat16_t* output,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& private_record_stride,
    const constant int& shared_record_stride,
    const bool two_bank,
    threadgroup uint8_t* scale_tile,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<32, 4>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = 32 / values_per_thread;
  constexpr uint topk = 6u;
  const uint token = tid.z;
  const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / 32;
  const ulong weight_stride = ulong(out_vec_size) * ulong(in_vec_size_w);
  const uint scale_count = uint(out_vec_size * in_vec_size_g);
  thread bfloat16_t total[results_per_simdgroup] = {
      bfloat16_t(0.0f), bfloat16_t(0.0f),
      bfloat16_t(0.0f), bfloat16_t(0.0f)};

  for (uint expert = 0; expert < topk; ++expert) {
    const uint route_position = token * topk + expert;
    const uint weight_slot = weight_routes[route_position];
    const uint scale_slot = scale_routes[route_position];
    const bool use_shared =
        two_bank && bank_routes[route_position] != 0u;
    const device uint32_t* weight =
        use_shared ? shared_weight : private_weight;
    const device uint8_t* scale_records =
        use_shared ? shared_scale_records : private_scale_records;
    const int record_stride =
        use_shared ? shared_record_stride : private_record_stride;
    const device uint8_t* record =
        scale_records + ulong(scale_slot) * ulong(record_stride);
    const uint codec = uint(record[4]);
    const uint primary_size = dsv4_scalex_load_u32(record + 16);
    const uint exception_count = dsv4_scalex_load_u32(record + 20);
    const device uint8_t* primary = record + 24;
    const device uint8_t* positions = primary + primary_size;
    const device uint8_t* exception_values =
        positions + ulong(exception_count) * 4ul;
    const uint encoded_size = 24u + primary_size + 5u * exception_count;
    const device uint8_t* tile_prefix =
        record + ((encoded_size + 1u) & ~1u);
    threadgroup uint8_t* local_scales =
        scale_tile + simd_gid * 512u;
    const uint group_start = scale_count + uint(out_row * in_vec_size_g);
    const uint group_scale_count =
        uint(results_per_simdgroup * in_vec_size_g);

    uint exception_range = 0;
    if (simd_lid == 0u) {
      exception_range =
          dsv4_scalex_load_u32(tile_prefix + 2u * (group_start >> 9));
    }
    exception_range = simd_shuffle(exception_range, ushort(0));
    const uint exception_begin = exception_range & 0xffffu;
    const uint exception_end = exception_range >> 16;
    if (codec == 1u) {
      const uint8_t palette0 = record[5];
      const uint8_t palette1 = record[6];
      for (uint local = simd_lid; local < group_scale_count;
           local += SIMD_SIZE) {
        const uint raw_index = group_start + local;
        const uint code =
            (uint(primary[raw_index >> 3]) >> (raw_index & 7u)) & 1u;
        local_scales[local] = code == 0u ? palette0 : palette1;
      }
    } else {
      for (uint local = simd_lid; local < group_scale_count;
           local += SIMD_SIZE) {
        local_scales[local] = dsv4_scalex_primary_value(
            record, primary, codec, group_start + local);
      }
    }
    for (uint index = exception_begin + simd_lid;
         index < exception_end; index += SIMD_SIZE) {
      const uint position =
          dsv4_scalex_load_u32(positions + ulong(index) * 4ul);
      const uint local = position - group_start;
      if (local < group_scale_count) {
        local_scales[local] = exception_values[index];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const device uint8_t* ws =
        reinterpret_cast<const device uint8_t*>(weight) +
        ulong(weight_slot) * weight_stride +
        ulong(out_row * in_vec_size_w +
              simd_lid * packs_per_thread * bytes_per_pack);
    const device bfloat16_t* route_x =
        x + ulong(route_position) * ulong(in_vec_size) +
        ulong(simd_lid * values_per_thread);
    thread float result[results_per_simdgroup] = {0};
    thread float x_thread[values_per_thread];
    for (int k = 0; k < in_vec_size; k += block_size) {
      load_vector<bfloat16_t, float, values_per_thread>(route_x, x_thread);
      const uint scale_group =
          uint(simd_lid / scale_step_per_thread + k / 32);
      for (int row = 0; row < results_per_simdgroup; ++row) {
        result[row] += qdot<float, values_per_thread, 4>(
            ws + row * in_vec_size_w,
            x_thread,
            dequantize_scale<float, 32>(
                local_scales[uint(row) * uint(in_vec_size_g) + scale_group]));
      }
      ws += block_size * bytes_per_pack / pack_factor;
      route_x += block_size;
    }
    const bfloat16_t score = bfloat16_t(scores[route_position]);
    for (int row = 0; row < results_per_simdgroup; ++row) {
      const bfloat16_t down = bfloat16_t(simd_sum(result[row]));
      const bfloat16_t product =
          bfloat16_t(float(down) * float(score));
      total[row] = bfloat16_t(float(total[row]) + float(product));
    }
  }

  if (simd_lid == 0u) {
    for (int row = 0; row < results_per_simdgroup; ++row) {
      const ulong offset =
          ulong(token) * ulong(out_vec_size) + ulong(out_row + row);
      output[offset] =
          bfloat16_t(float(total[row]) + float(shared[offset]));
    }
  }
}

[[kernel]] void dsv4_scalex_mxfp4_width2_down_reduce_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scale_records [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    const device uint32_t* weight_routes [[buffer(3)]],
    const device uint32_t* scale_routes [[buffer(4)]],
    const device float* scores [[buffer(5)]],
    const device bfloat16_t* shared [[buffer(6)]],
    device bfloat16_t* output [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    const constant int& record_stride [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  dsv4_scalex_mxfp4_fixed_down_reduce_bf16_impl(
      weight,
      scale_records,
      weight,
      scale_records,
      x,
      weight_routes,
      scale_routes,
      weight_routes,
      scores,
      shared,
      output,
      in_vec_size,
      out_vec_size,
      record_stride,
      record_stride,
      false,
      scale_tile,
      tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_width3_down_reduce_bf16(
    const device uint32_t* weight [[buffer(0)]],
    const device uint8_t* scale_records [[buffer(1)]],
    const device bfloat16_t* x [[buffer(2)]],
    const device uint32_t* weight_routes [[buffer(3)]],
    const device uint32_t* scale_routes [[buffer(4)]],
    const device float* scores [[buffer(5)]],
    const device bfloat16_t* shared [[buffer(6)]],
    device bfloat16_t* output [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    const constant int& record_stride [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  dsv4_scalex_mxfp4_fixed_down_reduce_bf16_impl(
      weight,
      scale_records,
      weight,
      scale_records,
      x,
      weight_routes,
      scale_routes,
      weight_routes,
      scores,
      shared,
      output,
      in_vec_size,
      out_vec_size,
      record_stride,
      record_stride,
      false,
      scale_tile,
      tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_width2_down_reduce_two_bank_bf16(
    const device uint32_t* private_weight [[buffer(0)]],
    const device uint8_t* private_scale_records [[buffer(1)]],
    const device uint32_t* shared_weight [[buffer(2)]],
    const device uint8_t* shared_scale_records [[buffer(3)]],
    const device bfloat16_t* x [[buffer(4)]],
    const device uint32_t* weight_routes [[buffer(5)]],
    const device uint32_t* scale_routes [[buffer(6)]],
    const device uint32_t* bank_routes [[buffer(7)]],
    const device float* scores [[buffer(8)]],
    const device bfloat16_t* shared [[buffer(9)]],
    device bfloat16_t* output [[buffer(10)]],
    const constant int& in_vec_size [[buffer(11)]],
    const constant int& out_vec_size [[buffer(12)]],
    const constant int& private_record_stride [[buffer(13)]],
    const constant int& shared_record_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  dsv4_scalex_mxfp4_fixed_down_reduce_bf16_impl(
      private_weight,
      private_scale_records,
      shared_weight,
      shared_scale_records,
      x,
      weight_routes,
      scale_routes,
      bank_routes,
      scores,
      shared,
      output,
      in_vec_size,
      out_vec_size,
      private_record_stride,
      shared_record_stride,
      true,
      scale_tile,
      tid,
      simd_gid,
      simd_lid);
}

[[kernel]] void dsv4_scalex_mxfp4_width3_down_reduce_two_bank_bf16(
    const device uint32_t* private_weight [[buffer(0)]],
    const device uint8_t* private_scale_records [[buffer(1)]],
    const device uint32_t* shared_weight [[buffer(2)]],
    const device uint8_t* shared_scale_records [[buffer(3)]],
    const device bfloat16_t* x [[buffer(4)]],
    const device uint32_t* weight_routes [[buffer(5)]],
    const device uint32_t* scale_routes [[buffer(6)]],
    const device uint32_t* bank_routes [[buffer(7)]],
    const device float* scores [[buffer(8)]],
    const device bfloat16_t* shared [[buffer(9)]],
    device bfloat16_t* output [[buffer(10)]],
    const constant int& in_vec_size [[buffer(11)]],
    const constant int& out_vec_size [[buffer(12)]],
    const constant int& private_record_stride [[buffer(13)]],
    const constant int& shared_record_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  dsv4_scalex_mxfp4_fixed_down_reduce_bf16_impl(
      private_weight,
      private_scale_records,
      shared_weight,
      shared_scale_records,
      x,
      weight_routes,
      scale_routes,
      bank_routes,
      scores,
      shared,
      output,
      in_vec_size,
      out_vec_size,
      private_record_stride,
      shared_record_stride,
      true,
      scale_tile,
      tid,
      simd_gid,
      simd_lid);
}

// GPU-only M0 selector. The current authoritative slot directories are read
// without materializing router IDs on the host. A miss writes zero-sized
// indirect dispatches; an all-hit route enables the fixed resident graph.
[[kernel]] void dsv4_scalex_m0_map_indirect(
    const device int32_t* expert_ids [[buffer(0)]],
    const device int32_t* gate_directory [[buffer(1)]],
    const device int32_t* down_directory [[buffer(2)]],
    device uint32_t* gate_routes [[buffer(3)]],
    device uint32_t* down_routes [[buffer(4)]],
    device int32_t* all_hit_status [[buffer(5)]],
    device uint* indirect_threadgroups [[buffer(6)]],
    const constant uint& route_count [[buffer(7)]],
    const constant uint& expert_count [[buffer(8)]],
    const constant uint& width [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
  if (index < route_count) {
    const int32_t expert = expert_ids[index];
    const bool valid = expert >= 0 && uint(expert) < expert_count;
    const int32_t gate_slot = valid ? gate_directory[expert] : -1;
    const int32_t down_slot = valid ? down_directory[expert] : -1;
    gate_routes[index] = gate_slot >= 0 ? uint(gate_slot) : 0xffffffffu;
    down_routes[index] = down_slot >= 0 ? uint(down_slot) : 0xffffffffu;
  }
  if (index == 0u) {
    bool all_hit = true;
    for (uint route = 0u; route < route_count; ++route) {
      const int32_t expert = expert_ids[route];
      const bool valid = expert >= 0 && uint(expert) < expert_count;
      all_hit = all_hit && valid && gate_directory[expert] >= 0 &&
          down_directory[expert] >= 0;
    }
    all_hit_status[0] = all_hit ? 1 : 0;
    const uint enabled = all_hit ? 1u : 0u;
    for (uint position = 0u; position < width; ++position) {
      const uint base = position * 3u;
      indirect_threadgroups[base] = enabled;
      indirect_threadgroups[base + 1u] = 256u;
      indirect_threadgroups[base + 2u] = 6u;
    }
    uint base = width * 3u;
    indirect_threadgroups[base] =
        enabled * ((route_count * 2048u + 255u) / 256u);
    indirect_threadgroups[base + 1u] = 1u;
    indirect_threadgroups[base + 2u] = 1u;
    base += 3u;
    indirect_threadgroups[base] = enabled;
    indirect_threadgroups[base + 1u] = 512u;
    indirect_threadgroups[base + 2u] = width == 2u ? 2u : route_count;
    base += 3u;
    indirect_threadgroups[base] = width == 1u ? enabled * 16u : 0u;
    indirect_threadgroups[base + 1u] = 1u;
    indirect_threadgroups[base + 2u] = 1u;
  }
}

// Two-bank variant of the optimistic M0 selector. Row directories remain
// separate from the bank directory so a shared row can use the same compact
// physical index as a private row without ambiguity.
[[kernel]] void dsv4_scalex_m0_map_two_bank_indirect(
    const device int32_t* expert_ids [[buffer(0)]],
    const device int32_t* gate_directory [[buffer(1)]],
    const device int32_t* down_directory [[buffer(2)]],
    const device int32_t* bank_directory [[buffer(3)]],
    device uint32_t* gate_routes [[buffer(4)]],
    device uint32_t* down_routes [[buffer(5)]],
    device uint32_t* bank_routes [[buffer(6)]],
    device int32_t* all_hit_status [[buffer(7)]],
    device uint* indirect_threadgroups [[buffer(8)]],
    const constant uint& route_count [[buffer(9)]],
    const constant uint& expert_count [[buffer(10)]],
    const constant uint& width [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
  if (index < route_count) {
    const int32_t expert = expert_ids[index];
    const bool valid = expert >= 0 && uint(expert) < expert_count;
    const int32_t gate_slot = valid ? gate_directory[expert] : -1;
    const int32_t down_slot = valid ? down_directory[expert] : -1;
    const int32_t bank = valid ? bank_directory[expert] : -1;
    gate_routes[index] = gate_slot >= 0 ? uint(gate_slot) : 0xffffffffu;
    down_routes[index] = down_slot >= 0 ? uint(down_slot) : 0xffffffffu;
    bank_routes[index] = bank >= 0 ? uint(bank) : 0xffffffffu;
  }
  if (index == 0u) {
    bool all_hit = true;
    for (uint route = 0u; route < route_count; ++route) {
      const int32_t expert = expert_ids[route];
      const bool valid = expert >= 0 && uint(expert) < expert_count;
      all_hit = all_hit && valid && gate_directory[expert] >= 0 &&
          down_directory[expert] >= 0 && bank_directory[expert] >= 0;
    }
    all_hit_status[0] = all_hit ? 1 : 0;
    const uint enabled = all_hit ? 1u : 0u;
    for (uint position = 0u; position < width; ++position) {
      const uint base = position * 3u;
      indirect_threadgroups[base] = enabled;
      indirect_threadgroups[base + 1u] = 256u;
      indirect_threadgroups[base + 2u] = 6u;
    }
    uint base = width * 3u;
    indirect_threadgroups[base] =
        enabled * ((route_count * 2048u + 255u) / 256u);
    indirect_threadgroups[base + 1u] = 1u;
    indirect_threadgroups[base + 2u] = 1u;
    base += 3u;
    indirect_threadgroups[base] = enabled;
    indirect_threadgroups[base + 1u] = 512u;
    indirect_threadgroups[base + 2u] = width == 2u ? 2u : route_count;
    base += 3u;
    indirect_threadgroups[base] = width == 1u ? enabled * 16u : 0u;
    indirect_threadgroups[base + 1u] = 1u;
    indirect_threadgroups[base + 2u] = 1u;
  }
}

[[kernel]] void dsv4_scalex_m0_pair_qmv_sparse_bf16(
    const device uint32_t* up_weight [[buffer(0)]],
    const device uint32_t* gate_weight [[buffer(1)]],
    const device uint8_t* scale_records [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device uint32_t* routes [[buffer(4)]],
    device bfloat16_t* up_output [[buffer(5)]],
    device bfloat16_t* gate_output [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& record_stride [[buffer(9)]],
    const constant uint& route_count [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  if (route_position >= route_count) return;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) return;
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(slot) * ulong(record_stride);
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      up_weight + ulong(slot) * weight_stride, record,
      scale_tile + simd_gid * 512u, 2u * scale_count, x,
      up_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size, out_vec_size, qmv_tid, simd_gid, simd_lid);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      gate_weight + ulong(slot) * weight_stride, record,
      scale_tile + simd_gid * 512u, 0u, x,
      gate_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size, out_vec_size, qmv_tid, simd_gid, simd_lid);
}

[[kernel]] void dsv4_scalex_m0_pair_qmv_sparse_two_bank_bf16(
    const device uint32_t* private_up_weight [[buffer(0)]],
    const device uint32_t* private_gate_weight [[buffer(1)]],
    const device uint8_t* private_scale_records [[buffer(2)]],
    const device uint32_t* shared_up_weight [[buffer(3)]],
    const device uint32_t* shared_gate_weight [[buffer(4)]],
    const device uint8_t* shared_scale_records [[buffer(5)]],
    const device bfloat16_t* x [[buffer(6)]],
    const device uint32_t* routes [[buffer(7)]],
    const device uint32_t* bank_routes [[buffer(8)]],
    device bfloat16_t* up_output [[buffer(9)]],
    device bfloat16_t* gate_output [[buffer(10)]],
    const constant int& in_vec_size [[buffer(11)]],
    const constant int& out_vec_size [[buffer(12)]],
    const constant int& private_record_stride [[buffer(13)]],
    const constant int& shared_record_stride [[buffer(14)]],
    const constant uint& route_count [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup uint8_t scale_tile[1024];
  const uint route_position = tid.z;
  if (route_position >= route_count) return;
  const uint slot = routes[route_position];
  if (slot == 0xffffffffu) return;
  const bool use_shared = bank_routes[route_position] != 0u;
  const device uint32_t* up_weight =
      use_shared ? shared_up_weight : private_up_weight;
  const device uint32_t* gate_weight =
      use_shared ? shared_gate_weight : private_gate_weight;
  const device uint8_t* scale_records =
      use_shared ? shared_scale_records : private_scale_records;
  const int record_stride =
      use_shared ? shared_record_stride : private_record_stride;
  const ulong weight_stride =
      ulong(out_vec_size) * ulong(in_vec_size / 8);
  const uint scale_count = uint(out_vec_size * (in_vec_size / 32));
  const device uint8_t* record =
      scale_records + ulong(slot) * ulong(record_stride);
  const uint3 qmv_tid(0u, tid.y, 0u);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      up_weight + ulong(slot) * weight_stride, record,
      scale_tile + simd_gid * 512u, 2u * scale_count, x,
      up_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size, out_vec_size, qmv_tid, simd_gid, simd_lid);
  dsv4_scalex_qmv_fast_impl<bfloat16_t, 32, 4>(
      gate_weight + ulong(slot) * weight_stride, record,
      scale_tile + simd_gid * 512u, 0u, x,
      gate_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size, out_vec_size, qmv_tid, simd_gid, simd_lid);
}

[[kernel]] void dsv4_scalex_m0_limited_swiglu_bf16(
    const device bfloat16_t* up [[buffer(0)]],
    const device bfloat16_t* gate [[buffer(1)]],
    const device uint32_t* routes [[buffer(2)]],
    device bfloat16_t* activated [[buffer(3)]],
    const constant uint& intermediate [[buffer(4)]],
    const constant uint& count [[buffer(5)]],
    const constant float& limit [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count || routes[index / intermediate] == 0xffffffffu) return;
  bfloat16_t gate_value = bfloat16_t(min(float(gate[index]), limit));
  bfloat16_t up_value = bfloat16_t(clamp(float(up[index]), -limit, limit));
  bfloat16_t y = bfloat16_t(1) /
      (bfloat16_t(1) + metal::fast::exp(metal::abs(gate_value)));
  bfloat16_t sigmoid = gate_value < bfloat16_t(0)
      ? y : bfloat16_t(1) - y;
  // nn.silu is a separately compiled BF16 operation in the product graph.
  // Preserve its storage/rounding boundary before the outer multiply by Up.
  bfloat16_t silu = bfloat16_t(gate_value * sigmoid);
  activated[index] = bfloat16_t(silu * up_value);
}

[[kernel]] void dsv4_scalex_m0_score_reduce_shared_bf16(
    const device bfloat16_t* routed [[buffer(0)]],
    const device float* scores [[buffer(1)]],
    const device bfloat16_t* shared [[buffer(2)]],
    device bfloat16_t* combined [[buffer(3)]],
    const constant uint& hidden [[buffer(4)]],
    const constant uint& top_k [[buffer(5)]],
    uint dimension [[thread_position_in_grid]]) {
  if (dimension >= hidden) return;
  bfloat16_t total = bfloat16_t(0.0f);
  for (uint expert = 0u; expert < top_k; ++expert) {
    const bfloat16_t score = bfloat16_t(scores[expert]);
    const bfloat16_t product = bfloat16_t(
        routed[ulong(expert) * ulong(hidden) + dimension] * score);
    total = bfloat16_t(total + product);
  }
  combined[dimension] = bfloat16_t(total + shared[dimension]);
}

#define instantiate_quantized(mode, name, type, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits, \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits)           \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs", \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      true)

#define instantiate_quantized_batched(mode, name, type, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      batched)        \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched "_hgs", \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      batched,        \
      true)

#define instantiate_quantized_aligned(mode, name, type, aligned, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      aligned)

#define instantiate_quantized_aligned_batched(mode, name, type, aligned, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      aligned, \
      batched)

#define instantiate_quantized_qmv_fast(mode, type, results, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_r_" #results "_batch_" #batched, \
      fp_qmv_fast, \
      type, \
      group_size, \
      bits, \
      batched, \
      false, \
      results) \
  instantiate_kernel( \
      #mode "_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_r_" #results "_batch_" #batched "_hgs", \
      fp_qmv_fast, \
      type, \
      group_size, \
      bits, \
      batched, \
      true, \
      results)

#define instantiate_quantized_qmv_fast_r2(mode, type, group_size, bits) \
  instantiate_quantized_qmv_fast(mode, type, 2, 1, group_size, bits) \
  instantiate_quantized_qmv_fast(mode, type, 2, 0, group_size, bits)

#define instantiate_quantized_quad(mode, name, type, D, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      D,       \
      batched)

#define instantiate_quantized_wide(mode, name, type, vecs_per_tg, k_lanes, group_size, bits, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      vecs_per_tg,       \
      k_lanes,       \
      batched)

#define instantiate_quantized_split_k(mode, name, type, split_k, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, bm, bn, bk, wm, wn, transpose, mode, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,    \
      type,    \
      group_size,      \
      bits,       \
      bm,      \
      bn,      \
      bk,      \
      wm,      \
      wn,      \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, mode, group_size, bits) \
  instantiate_quantized_batched(mode, name, type, 1, group_size, bits)         \
  instantiate_quantized_batched(mode, name, type, 0, group_size, bits)

#define instantiate_quantized_all_batched(type, mode, group_size, bits) \
  instantiate_quantized_batched_wrap(qmv_fast, type, mode, group_size, bits) \
  instantiate_quantized_batched_wrap(qmv, type, mode, group_size, bits)      \
  instantiate_quantized_batched_wrap(qvm, type, mode, group_size, bits) \
  instantiate_quantized_batched_wrap(qmm_n, type, mode, group_size, bits)

#define instantiate_quantized_all_single(type, mode, group_size, bits) \
  instantiate_quantized(mode, gather_qmv_fast, type, group_size, bits) \
  instantiate_quantized(mode, gather_qmv, type, group_size, bits)      \
  instantiate_quantized(mode, gather_qvm, type, group_size, bits) \
  instantiate_quantized(mode, gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_aligned(mode, gather_qmm_t, type, true, group_size, bits)      \
  instantiate_quantized_aligned(mode, gather_qmm_t, type, false, group_size, bits)     \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, true, 1, group_size, bits)  \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, true, 0, group_size, bits)  \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, false, 1, group_size, bits) \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, false, 0, group_size, bits)

#define instantiate_quantized_all_quad(type, mode, group_size, bits) \
  instantiate_quantized_quad(mode, qmv_quad, type, 64, 1, group_size, bits)  \
  instantiate_quantized_quad(mode, qmv_quad, type, 64, 0, group_size, bits)  \
  instantiate_quantized_quad(mode, qmv_quad, type, 128, 1, group_size, bits) \
  instantiate_quantized_quad(mode, qmv_quad, type, 128, 0, group_size, bits)

// vecs_per_tg (input-vector tile) 2..5; the fp path uses k_lanes=16.
#define instantiate_quantized_wide_wrap(mode, name, type, vecs_per_tg, k_lanes, group_size, bits) \
  instantiate_quantized_wide(mode, name, type, vecs_per_tg, k_lanes, group_size, bits, 0)         \
  instantiate_quantized_wide(mode, name, type, vecs_per_tg, k_lanes, group_size, bits, 1)

#define instantiate_quantized_all_wide(type, mode, group_size, bits) \
  instantiate_quantized_wide_wrap(mode, qmv_wide, type, 2, 16, group_size, bits) \
  instantiate_quantized_wide_wrap(mode, qmv_wide, type, 3, 16, group_size, bits) \
  instantiate_quantized_wide_wrap(mode, qmv_wide, type, 4, 16, group_size, bits) \
  instantiate_quantized_wide_wrap(mode, qmv_wide, type, 5, 16, group_size, bits)

#define instantiate_quantized_all_splitk(type, mode, group_size, bits) \
  instantiate_quantized_split_k(mode, qvm_split_k, type, 8, group_size, bits) \
  instantiate_quantized_split_k(mode, qvm_split_k, type, 32, group_size, bits) \
  instantiate_quantized_aligned(mode, qmm_t_splitk, type, true, group_size, bits) \
  instantiate_quantized_aligned(mode, qmm_t_splitk, type, false, group_size, bits)

#define instantiate_quantized_all_rhs(type, mode, group_size, bits) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs, gather_qmm_rhs_nt, type, 16, 32, 32, 1, 2, true, mode, group_size, bits) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs, gather_qmm_rhs_nn, type, 16, 32, 32, 1, 2, false, mode, group_size, bits)

#define instantiate_quantize_dequantize(type, mode, group_size, bits, has_global_scale) \
  instantiate_kernel(       \
    #mode "_quantize_dequantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale, \
    fp_quantize_dequantize, \
    type,                   \
    group_size,             \
    bits,                   \
    has_global_scale)       \
  instantiate_kernel(       \
    #mode "_quantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale, \
    fp_quantize,            \
    type,                   \
    group_size,             \
    bits,                   \
    has_global_scale)       \
  instantiate_kernel(       \
    #mode "_dequantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale, \
    fp_dequantize,          \
    type,                   \
    group_size,             \
    bits,                   \
    has_global_scale)

#define instantiate_quantized_modes(type, mode, group_size, bits) \
  instantiate_quantized_all_batched(type, mode, group_size, bits) \
  instantiate_quantized_all_single(type, mode, group_size, bits)  \
  instantiate_quantized_all_quad(type, mode, group_size, bits)    \
  instantiate_quantized_all_wide(type, mode, group_size, bits)    \
  instantiate_quantized_all_splitk(type, mode, group_size, bits)  \
  instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_all_rhs(type, mode, group_size, bits)

#define instantiate_quantized_types(type) \
 instantiate_quantized_modes(type, nvfp4, 16, 4) \
 instantiate_quantized_modes(type, mxfp8, 32, 8) \
 instantiate_quantized_modes(type, mxfp4, 32, 4) \
 instantiate_quantize_dequantize(type, nvfp4, 16, 4, false) \
 instantiate_quantize_dequantize(type, nvfp4, 16, 4, true)  \
 instantiate_quantize_dequantize(type, mxfp8, 32, 8, false) \
 instantiate_quantize_dequantize(type, mxfp4, 32, 4, false) \
 instantiate_quantized_qmv_fast_r2(nvfp4, type, 16, 4)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)
    // clang-format on
