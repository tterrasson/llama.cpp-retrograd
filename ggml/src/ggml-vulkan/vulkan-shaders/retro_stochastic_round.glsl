// retro delta: stochastic rounding for a half-precision parameter store,
// shared by every optimizer. Guarded by the same two defines the includer
// sets, so an F32 variant compiles none of it.
#if defined(STOCHASTIC_ROUNDING) || defined(STOCHASTIC_ROUNDING_BF16)
// retro delta: mirrors the ggml-impl.h helpers bit for bit; an exact
// CPU-vs-GPU equality test covers them.
float retro_sr_uniform(uint seed, uint index) {
    uint h = seed ^ (index * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return float(h >> 8) * (1.0f / 16777216.0f);
}

// One function for both grids: F16 and BF16 are both sign-magnitude in 16
// bits, so the neighbour is a magnitude increment either way.
uint retro_half_neighbour(uint bits, bool up) {
    const uint sign = bits & 0x8000u;
    const uint mag  = bits & 0x7FFFu;
    if (mag == 0u) {
        return up ? 0x0001u : 0x8001u;
    }
    const bool grow = up ? (sign == 0u) : (sign != 0u);
    return sign | (grow ? mag + 1u : mag - 1u);
}
#endif

#ifdef STOCHASTIC_ROUNDING
// packHalf2x16/unpackHalf2x16 are the round-to-nearest F32<->F16 pair, and give
// access to the raw F16 bits without needing 16-bit arithmetic support.
float retro_stochastic_round_f16(float x, float u) {
    const uint  nearest   = packHalf2x16(vec2(x, 0.0f)) & 0xFFFFu;
    const float nearest_f = unpackHalf2x16(nearest).x;
    const float residual  = x - nearest_f;
    if (residual == 0.0f) {
        return nearest_f;
    }
    const uint  other   = retro_half_neighbour(nearest, residual > 0.0f);
    const float other_f = unpackHalf2x16(other).x;
    const float span    = other_f - nearest_f;
    const float p = (span != 0.0f && !isinf(span)) ? residual / span : 0.0f;
    return u < p ? other_f : nearest_f;
}
#endif

#ifdef STOCHASTIC_ROUNDING_BF16
// BF16 is the top 16 bits of the F32 with the same value,
// round-half-to-even on the way down (ggml_compute_fp32_to_bf16).
uint retro_f32_to_bf16(float x) {
    const uint bits = floatBitsToUint(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (bits >> 16) | 64u;  // NaN, forced quiet
    }
    return (bits + (0x7FFFu + ((bits >> 16) & 1u))) >> 16;
}

float retro_bf16_to_f32(uint bits) {
    return uintBitsToFloat(bits << 16);
}

// Returns the BF16 bits of `x` after stochastic rounding with uniform `u`.
uint retro_stochastic_round_bf16(float x, float u) {
    const uint  nearest   = retro_f32_to_bf16(x);
    const float nearest_f = retro_bf16_to_f32(nearest);
    const float residual  = x - nearest_f;
    if (residual == 0.0f) {
        return nearest;
    }
    const uint  other   = retro_half_neighbour(nearest, residual > 0.0f);
    const float other_f = retro_bf16_to_f32(other);
    const float span    = other_f - nearest_f;
    const float p = (span != 0.0f && !isinf(span)) ? residual / span : 0.0f;
    return u < p ? other : nearest;
}
#endif
