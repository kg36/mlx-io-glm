// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

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
