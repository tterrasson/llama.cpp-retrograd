// retro delta: Gefen, the part both phases share.
//
// One workgroup owns one quantization block, so the whole block is read
// against the old scale before any element is rewritten.
//
// The quantized_m variant addresses byte indices as packed 32-bit words: GLSL
// has no 8-bit store, and two invocations read-modify-writing one word would
// lose an index. The block size is a multiple of four so a block boundary
// never falls inside a word.

#extension GL_EXT_control_flow_attributes : enable

#define GEFEN_WORKGROUP_SIZE 256

layout (push_constant) uniform parameter
{
    uint np;        // elements of the parameter
    uint bs;        // elements per quantization block
    uint levels;    // codebook entries, 0 under shared_v
    uint zero_code; // the index a zero block stores
} p;

// Nearest codebook entry for a value in [-1, 1], ties to the lower index.
// Must match gefen_nearest_code in ggml-cpu/ops.cpp, ggml-metal/kernels/retro.metal
// and ggml-cuda/opt-step-gefen.cu, otherwise a checkpoint written on one
// backend decodes differently on another.
uint gefen_nearest_code(float value, uint levels) {
    const float top = float(levels - 1u);
    const float position = (value + 1.0f)*0.5f*top;
    if (!(position > 0.0f)) {
        // Also catches NaN.
        return 0u;
    }
    if (position >= top) {
        return levels - 1u;
    }
    // ceil(x - 1/2) rounds exact midpoints down to the lower index.
    return uint(ceil(position - 0.5f));
}
