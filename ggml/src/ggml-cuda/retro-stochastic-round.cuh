#pragma once

// retro delta: stochastic rounding for a half-precision parameter store.
//
// Mirrors the ggml-impl.h helpers bit for bit, so a CUDA update step and the
// CPU oracle land on the same word; an exact equality test covers the pair.
// Shared by every optimizer whose step writes such a parameter.

#include <cstdint>

static __device__ __forceinline__ float retro_sr_uniform(uint32_t seed, uint32_t index) {
    uint32_t h = seed ^ (index * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (float) (h >> 8) * (1.0f / 16777216.0f);
}

// One function for both grids: F16 and BF16 are both sign-magnitude in 16
// bits, so the neighbour is a magnitude increment either way.
static __device__ __forceinline__ uint16_t retro_half_neighbour(uint16_t bits, int up) {
    const uint16_t sign = bits & 0x8000u;
    const uint16_t mag  = bits & 0x7FFFu;
    if (mag == 0) {
        return up ? (uint16_t) 0x0001u : (uint16_t) 0x8001u;
    }
    const int grow = up ? (sign == 0) : (sign != 0);
    return (uint16_t) (sign | (uint16_t) (grow ? mag + 1 : mag - 1));
}

// Returns the F16 bits of `x` after stochastic rounding with uniform `u`.
static __device__ __forceinline__ uint16_t retro_stochastic_round_f16(float x, float u) {
    const uint16_t nearest   = __half_as_ushort(__float2half_rn(x));
    const float    nearest_f = __half2float(__ushort_as_half(nearest));
    const float    residual  = x - nearest_f;
    if (residual == 0.0f) {
        return nearest;
    }
    const uint16_t other   = retro_half_neighbour(nearest, residual > 0.0f);
    const float    other_f = __half2float(__ushort_as_half(other));
    const float    span    = other_f - nearest_f;
    const float    p = (span != 0.0f && isfinite(span)) ? residual / span : 0.0f;
    return u < p ? other : nearest;
}

// BF16 is the top 16 bits of the F32 with the same value, so both
// conversions are shifts, round-half-to-even on the way down. Written on the
// raw bits rather than through __nv_bfloat16.
static __device__ __forceinline__ uint16_t retro_f32_to_bf16(float x) {
    const uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (uint16_t) ((bits >> 16) | 64);  // NaN, forced quiet
    }
    return (uint16_t) ((bits + (0x7FFFu + ((bits >> 16) & 1u))) >> 16);
}

static __device__ __forceinline__ float retro_bf16_to_f32(uint16_t bits) {
    return __uint_as_float((uint32_t) bits << 16);
}

// Returns the BF16 bits of `x` after stochastic rounding with uniform `u`.
static __device__ __forceinline__ uint16_t retro_stochastic_round_bf16(float x, float u) {
    const uint16_t nearest   = retro_f32_to_bf16(x);
    const float    nearest_f = retro_bf16_to_f32(nearest);
    const float    residual  = x - nearest_f;
    if (residual == 0.0f) {
        return nearest;
    }
    const uint16_t other   = retro_half_neighbour(nearest, residual > 0.0f);
    const float    other_f = retro_bf16_to_f32(other);
    const float    span    = other_f - nearest_f;
    const float    p = (span != 0.0f && isfinite(span)) ? residual / span : 0.0f;
    return u < p ? other : nearest;
}
