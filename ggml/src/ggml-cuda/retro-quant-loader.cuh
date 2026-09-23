#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "../ggml-retro-quant.h"

// retro delta: the in-kernel decoders a training op uses to read a frozen
// quantized tensor without ever materializing an F32 copy of it in the CUDA
// pool. Two ops need exactly that -- OUT_PROD's src0 and the
// output head of FUSED_SPARSE_CE[_BACK] -- and they must read one table of
// formats, not two, so the loaders live here rather than
// inside either kernel's translation unit.
//
// Contract, and it is a block-wide one: *every* thread of a block of
// RETRO_QUANT_THREADS threads calls
//
//     loader::load(row, m0, ne0, tile);
//
// and on return tile[0 .. RETRO_QUANT_TILE) holds the logical elements
// [m0, m0 + RETRO_QUANT_TILE) of `row`. `row` points at the start of a row that
// is a tightly packed run of quantization blocks (the super-block decoders take
// a block index, the pair decoders derive one from the element offset), so only
// the within-row packing has to hold; rows themselves are reached by the caller
// through the row stride. `m0` must be a multiple of `traits::alignment`.
//
// Elements past ne0 are *not* written: a caller that can see a partial tile has
// to zero it first (k_out_prod_quant does), and a caller that requires whole
// tiles has to check the extent (fused sparse CE does).
static constexpr int RETRO_QUANT_TILE    = 256;
// table from ggml-retro-quant.h, so adding a CPU/Vulkan/CUDA type without a CUDA
// loader is a compile error rather than a silent trip through a fallback.
template<ggml_type type> struct retro_quant_traits;

#define RETRO_QUANT_F16_TRAITS(type) \
    template<> struct retro_quant_traits<type> { \
        using loader = retro_quant_f16_loader; \
        static constexpr int64_t alignment = 1; \
    }
#define RETRO_QUANT_PAIR_TRAITS(type, qk, qr, decoder) \
    template<> struct retro_quant_traits<type> { \
        using loader = retro_quant_pair_loader<qk, qr, decoder>; \
        static constexpr int64_t alignment = qk; \
    }
#define RETRO_QUANT_SUPER_TRAITS(type, threads, decoder) \
    template<> struct retro_quant_traits<type> { \
        using loader = retro_quant_superblock_loader<threads, decoder>; \
        static constexpr int64_t alignment = QK_K; \
    }

RETRO_QUANT_F16_TRAITS(GGML_TYPE_F16);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q1_0, QK1_0, QR1_0, dequantize_q1_0);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q2_0, QK2_0, QR2_0, dequantize_q2_0);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q4_0, QK4_0, QR4_0, dequantize_q4_0);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q4_1, QK4_1, QR4_1, dequantize_q4_1);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q5_0, QK5_0, QR5_0, dequantize_q5_0);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q5_1, QK5_1, QR5_1, dequantize_q5_1);
RETRO_QUANT_PAIR_TRAITS(GGML_TYPE_Q8_0, QK8_0, QR8_0, dequantize_q8_0);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_MXFP4,   32, dequantize_mxfp4<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_Q2_K,    64, dequantize_q2_K<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_Q3_K,    64, dequantize_q3_K<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_Q4_K,    32, dequantize_q4_K<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_Q5_K,    64, dequantize_q5_K<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_Q6_K,    64, dequantize_q6_K<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ2_XXS, 32, dequantize_iq2_xxs<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ2_XS,  32, dequantize_iq2_xs<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ2_S,   32, dequantize_iq2_s<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ3_XXS, 32, dequantize_iq3_xxs<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ3_S,   32, dequantize_iq3_s<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ1_S,   32, dequantize_iq1_s<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ1_M,   32, dequantize_iq1_m<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ4_NL,  32, dequantize_iq4_nl<float>);
RETRO_QUANT_SUPER_TRAITS(GGML_TYPE_IQ4_XS,  32, dequantize_iq4_xs<float>);
template<> struct retro_quant_traits<GGML_TYPE_NVFP4> {
    using loader = retro_quant_nvfp4_loader;
    static constexpr int64_t alignment = QK_NVFP4;
};

#undef RETRO_QUANT_F16_TRAITS
#undef RETRO_QUANT_PAIR_TRAITS
#undef RETRO_QUANT_SUPER_TRAITS
