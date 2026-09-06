// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

// LivSeek checkpoints were qualified with the pre-v0.32.2 E8M0 rounding
// behavior. Keep that conversion private so the public quantizer can retain
// upstream's saturation fix. This is deliberately a separate Metal translation
// unit so adding the compatibility kernel cannot perturb qmm code generation.
template <typename T>
METAL_FUNC void dsv4_legacy_mxfp4_quantize_impl(
    const device T* w,
    device uint8_t* packed,
    device uint8_t* scales,
    uint index,
    uint simd_lid) {
  const float value = float(w[index]);
  const float block_max = simd_max(abs(value));
  fp8_e8m0 scale(block_max / F4E2M1_MAX);

  if (simd_lid == 0u) {
    scales[index / 32u] = scale.bits;
  }

  const float decoded_scale = float(scale);
  const float scale_recip = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
  uint8_t output = Quantize<4>{}(value * scale_recip);
  output |= simd_shuffle_down(output, 1) << 4;
  if ((index & 1u) == 0u) {
    packed[index / 2u] = output;
  }
}

[[kernel]] void dsv4_legacy_mxfp4_quantize_bf16(
    const device bfloat16_t* w [[buffer(0)]],
    device uint8_t* packed [[buffer(1)]],
    device uint8_t* scales [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  dsv4_legacy_mxfp4_quantize_impl(w, packed, scales, index, simd_lid);
}

[[kernel]] void dsv4_legacy_mxfp4_quantize_f32(
    const device float* w [[buffer(0)]],
    device uint8_t* packed [[buffer(1)]],
    device uint8_t* scales [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  dsv4_legacy_mxfp4_quantize_impl(w, packed, scales, index, simd_lid);
}
