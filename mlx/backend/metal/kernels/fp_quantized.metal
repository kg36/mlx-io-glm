// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

// LivSeek ScaleX Mode B keeps the complete lossless Mode-A record resident.
// The record layout is:
//   24-byte header, packed palette codes, sorted uint32 exception positions,
//   uint8 exception values.
// No hydrated scale tensor or model-wide scale sidecar is required.
METAL_FUNC uint dsv4_scalex_load_u32(const device uint8_t* p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
      (uint(p[3]) << 24);
}

METAL_FUNC uint dsv4_scalex_lower_bound(
    const device uint8_t* positions,
    uint count,
    uint target) {
  uint lo = 0;
  uint hi = count;
  while (lo < hi) {
    const uint mid = lo + ((hi - lo) >> 1);
    if (dsv4_scalex_load_u32(positions + ulong(mid) * 4ul) < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

METAL_FUNC uint8_t dsv4_scalex_value(
    const device uint8_t* record,
    const device uint8_t* primary,
    const device uint8_t* positions,
    const device uint8_t* exception_values,
    uint codec,
    uint raw_index,
    uint exception_begin,
    uint exception_end) {
  uint8_t value;
  if (codec == 0u) {
    value = primary[raw_index];
  } else if (codec == 1u) {
    const uint code =
        (uint(primary[raw_index >> 3]) >> (raw_index & 7u)) & 1u;
    value = record[5u + code];
  } else {
    const uint code =
        (uint(primary[raw_index >> 2]) >> ((raw_index & 3u) << 1)) & 3u;
    value = record[5u + code];
  }

  // Exceptions are sparse (roughly hundreds over 786,432 scale bytes). The
  // caller narrows this loop to one output row, so most rows execute zero
  // iterations and rows with an exception normally execute one.
  for (uint i = exception_begin; i < exception_end; ++i) {
    if (dsv4_scalex_load_u32(positions + ulong(i) * 4ul) == raw_index) {
      return exception_values[i];
    }
  }
  return value;
}

template <typename T, int group_size, int bits>
METAL_FUNC void dsv4_scalex_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* record,
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

  // One SIMD group owns four consecutive output rows. Search the sparse
  // exception list once for the whole four-row interval; almost every such
  // interval is empty, and this removes six of the eight per-row searches.
  uint exception_begin = 0;
  uint exception_end = 0;
  if (simd_lid == 0u) {
    const uint group_start =
        scale_projection_offset + uint(out_row) * uint(in_vec_size_g);
    exception_begin =
        dsv4_scalex_lower_bound(positions, exception_count, group_start);
    exception_end = dsv4_scalex_lower_bound(
        positions,
        exception_count,
        group_start + uint(results_per_simdgroup * in_vec_size_g));
  }
  exception_begin = simd_shuffle(exception_begin, ushort(0));
  exception_end = simd_shuffle(exception_end, ushort(0));

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);
    const uint scale_group =
        uint(simd_lid / scale_step_per_thread + k / group_size);

    for (int row = 0; row < results_per_simdgroup; ++row) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const uint raw_scale_index = scale_projection_offset +
          uint(out_row + row) * uint(in_vec_size_g) + scale_group;
      const uint8_t encoded_scale = dsv4_scalex_value(
          record,
          primary,
          positions,
          exception_values,
          codec,
          raw_scale_index,
          exception_begin,
          exception_end);
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
      x,
      gate_output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
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
      projection * scale_count,
      route_x,
      output + ulong(route_position) * ulong(out_vec_size),
      in_vec_size,
      out_vec_size,
      qmv_tid,
      simd_gid,
      simd_lid);
}

#define instantiate_quantized(mode, name, type, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits, \
      fp_ ## name, \
      type, \
      group_size,   \
      bits)

#define instantiate_quantized_batched(mode, name, type, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      batched)

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

#define instantiate_quantize_dequantize(type, mode, group_size, bits) \
  instantiate_kernel( \
    #mode "_quantize_dequantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_quantize_dequantize, \
    type, \
    group_size,  \
    bits) \
  instantiate_kernel( \
    #mode "_quantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_quantize, \
    type, \
    group_size,  \
    bits) \
  instantiate_kernel( \
    #mode "_dequantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_dequantize, \
    type, \
    group_size,  \
    bits)

#define instantiate_quantized_modes(type, mode, group_size, bits) \
  instantiate_quantized_all_batched(type, mode, group_size, bits) \
  instantiate_quantized_all_single(type, mode, group_size, bits)  \
  instantiate_quantized_all_quad(type, mode, group_size, bits)    \
  instantiate_quantized_all_wide(type, mode, group_size, bits)    \
  instantiate_quantized_all_splitk(type, mode, group_size, bits)  \
  instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_all_rhs(type, mode, group_size, bits)     \
  instantiate_quantize_dequantize(type, mode, group_size, bits)

#define instantiate_quantized_types(type) \
  instantiate_quantized_modes(type, nvfp4, 16, 4) \
  instantiate_quantized_modes(type, mxfp8, 32, 8) \
  instantiate_quantized_modes(type, mxfp4, 32, 4)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)
    // clang-format on
