#pragma once

// retro delta: the list of types a training kernel can decode in place.
//
// Two training ops read a frozen, quantized tensor directly: OUT_PROD (src0 is
// the frozen projection whose input gradient we need, dx = out_prod(W, dy)) and
// FUSED_SPARSE_CE[_BACK] (the frozen output head). Which types each could decode
// used to be written out by hand ten times - two supports_op cases and two
// template blocks on Metal, two CREATE_* blocks plus the shader generator on
// Vulkan, one case on CUDA, two more in the retroback probe ABI. Nothing tied
// them together, so they drifted: Metal had Q5_0 but not Q4_0, Vulkan had Q4_0
// but no IQ type, CUDA had every quant but not F16.
//
// One row here, one expansion per consumer. Macro-only by design: ggml-metal.metal
// is a consumer, so this header must stay legal MSL - no #include, no types, no
// prototypes.
//
// Columns:
//   TYPE    ggml_type enum value
//   BLK     Metal block struct for one quantization block
//   NL      16-value chunks per block, i.e. QK/16
//   NAME    ggml_type_name() spelling; suffixes dequantize_<NAME> and the
//           kernel_<op>_<NAME> pipeline names
//   VKNAME  lowercased NAME; suffixes Vulkan shader/pipeline names and feeds
//           DATA_A_<upper(VKNAME)>
//
// (BLK, NL, dequantize_<NAME>) is the triple kernel_mul_mm already instantiates
// with for every quant type: dequantize_<NAME>(blk, il, reg) fills the 16
// consecutive elements at il*16 within a block of NL*16. That uniform contract is
// what lets one template cover every row.
//
// A type belongs here only if the CPU has a path for it in both ops (so a parity
// test has an oracle), dequantize_<NAME> exists in ggml-metal.metal and
// dequantize()/get_dm() exist for DATA_A_<upper(VKNAME)> in dequant_funcs.glsl,
// and real GGUF files use it.
//
// Deliberately excluded:
//   IQ1_S, IQ1_M   1.5-1.75 bpw. The templates would cover them for free, but a
//                  LoRA over a base that coarse has a questionable gradient
//                  signal-to-noise ratio, and it is not claimed as supported
//                  without a parity test that argues otherwise. Two lines to add.
//   BF16           ggml_compute_forward_out_prod hits GGML_ABORT on it, so there
//                  is no CPU oracle. Needs a CPU path first.
//   NVFP4          Blackwell-only, essentially no public GGUF.
//   Q1_0, Q2_0     Very recent, no models in circulation.
//   TQ1_0, TQ2_0   Same signal-to-noise objection as IQ1, and no dequantize_tq*.
//
// F32 is not a row: it needs no decoding and both ops give it a cheaper dedicated
// path. Sites that want it list it explicitly alongside the expansion.

// Spelled QK_NL in ggml-metal.metal; not reused so this header stays include-free.
#define GGML_RETRO_NL_256 16

//        TYPE                 BLK             NL                  NAME      VKNAME
#define GGML_RETRO_DEQUANT_TYPES(X)                                                     \
    X(GGML_TYPE_F16,      half4x4,             1,                  f16,      f16)       \
    X(GGML_TYPE_Q4_0,     block_q4_0,          2,                  q4_0,     q4_0)      \
    X(GGML_TYPE_Q4_1,     block_q4_1,          2,                  q4_1,     q4_1)      \
    X(GGML_TYPE_Q5_0,     block_q5_0,          2,                  q5_0,     q5_0)      \
    X(GGML_TYPE_Q5_1,     block_q5_1,          2,                  q5_1,     q5_1)      \
    X(GGML_TYPE_Q8_0,     block_q8_0,          2,                  q8_0,     q8_0)      \
    X(GGML_TYPE_MXFP4,    block_mxfp4,         2,                  mxfp4,    mxfp4)     \
    X(GGML_TYPE_Q2_K,     block_q2_K,          GGML_RETRO_NL_256,  q2_K,     q2_k)      \
    X(GGML_TYPE_Q3_K,     block_q3_K,          GGML_RETRO_NL_256,  q3_K,     q3_k)      \
    X(GGML_TYPE_Q4_K,     block_q4_K,          GGML_RETRO_NL_256,  q4_K,     q4_k)      \
    X(GGML_TYPE_Q5_K,     block_q5_K,          GGML_RETRO_NL_256,  q5_K,     q5_k)      \
    X(GGML_TYPE_Q6_K,     block_q6_K,          GGML_RETRO_NL_256,  q6_K,     q6_k)      \
    X(GGML_TYPE_IQ2_XXS,  block_iq2_xxs,       GGML_RETRO_NL_256,  iq2_xxs,  iq2_xxs)   \
    X(GGML_TYPE_IQ2_XS,   block_iq2_xs,        GGML_RETRO_NL_256,  iq2_xs,   iq2_xs)    \
    X(GGML_TYPE_IQ2_S,    block_iq2_s,         GGML_RETRO_NL_256,  iq2_s,    iq2_s)     \
    X(GGML_TYPE_IQ3_XXS,  block_iq3_xxs,       GGML_RETRO_NL_256,  iq3_xxs,  iq3_xxs)   \
    X(GGML_TYPE_IQ3_S,    block_iq3_s,         GGML_RETRO_NL_256,  iq3_s,    iq3_s)     \
    X(GGML_TYPE_IQ4_NL,   block_iq4_nl,        2,                  iq4_nl,   iq4_nl)    \
    X(GGML_TYPE_IQ4_XS,   block_iq4_xs,        GGML_RETRO_NL_256,  iq4_xs,   iq4_xs)

// Row count, for the probe ABI and for tests asserting they enumerate the whole
// table rather than a stale copy of it.
#define GGML_RETRO_DEQUANT_TYPE_COUNT 19
