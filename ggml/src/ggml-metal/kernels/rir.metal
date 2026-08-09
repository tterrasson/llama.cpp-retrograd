#include "common.h" // retro delta: fork-owned RIR kernels

// retro delta: RIR-GENERATED-BEGIN (docs/INT_RIR.md)
// Committed copy of generated/rir/<kernel>/kernel[.<variant>].metal bodies
// (header lines stripped). A parent-repo test compares this block to the
// generator output — edit only via rir-gen. One entry per production
// *variant*: a kernel arbitrated per shape publishes several
// (docs/INT_RIR_V3.md §R1).

struct rir_l2_norm_back_params {
    float eps;
    uint n_row;
    uint n_plane;
    uint n_batch;
    uint n_col;
    uint dz_nb0;
    uint dz_nb1;
    uint dz_nb2;
    uint dz_nb3;
    uint x_nb0;
    uint x_nb1;
    uint x_nb2;
    uint x_nb3;
    uint dx_nb0;
    uint dx_nb1;
    uint dx_nb2;
    uint dx_nb3;
};

kernel void rir_l2_norm_back(
    device const uchar* dz [[buffer(0)]],
    device const uchar* x [[buffer(1)]],
    device uchar* dx [[buffer(2)]],
    constant rir_l2_norm_back_params& pc [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 global_id [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint row_v0 = group_id.x;
    if (row_v0 >= pc.n_row) {
        return;
    }
    const uint plane_v1 = group_id.y;
    if (plane_v1 >= pc.n_plane) {
        return;
    }
    const uint batch_v2 = group_id.z;
    if (batch_v2 >= pc.n_batch) {
        return;
    }
    const uint lane_v3 = simd_lane_id;
    float acc9_v4 = 0.0f;
    float acc10_v5 = 0.0f;
    for (uint col_v6 = lane_v3; col_v6 < pc.n_col; col_v6 += 32u) {
        const float t_v7 = *(device const float *)(x + (col_v6 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        const float t_v8 = t_v7 * t_v7;
        acc9_v4 += t_v8;
        const float t_v9 = *(device const float *)(dz + (col_v6 * pc.dz_nb0 + row_v0 * pc.dz_nb1 + plane_v1 * pc.dz_nb2 + batch_v2 * pc.dz_nb3));
        const float t_v10 = t_v7 * t_v9;
        acc10_v5 += t_v10;
    }
    const float red9_v11 = simd_sum(acc9_v4);
    const float red10_v12 = simd_sum(acc10_v5);
    for (uint col_v13 = lane_v3; col_v13 < pc.n_col; col_v13 += 32u) {
        const float t_v14 = sqrt(red9_v11);
        const float p_v15 = pc.eps;
        const bool b_v16 = t_v14 > p_v15;
        const float t_v17 = *(device const float *)(dz + (col_v13 * pc.dz_nb0 + row_v0 * pc.dz_nb1 + plane_v1 * pc.dz_nb2 + batch_v2 * pc.dz_nb3));
        const float t_v18 = *(device const float *)(x + (col_v13 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        const float t_v19 = red10_v12 / red9_v11;
        const float t_v20 = t_v18 * t_v19;
        const float t_v21 = t_v17 - t_v20;
        const float t_v22 = t_v21 / t_v14;
        const float t_v23 = t_v17 / p_v15;
        const float t_v24 = b_v16 ? t_v22 : t_v23;
        *(device float *)(dx + (col_v13 * pc.dx_nb0 + row_v0 * pc.dx_nb1 + plane_v1 * pc.dx_nb2 + batch_v2 * pc.dx_nb3)) = t_v24;
    }
}

struct rir_cumsum_params {
    uint n_row;
    uint n_plane;
    uint n_batch;
    uint n_col;
    uint x_nb0;
    uint x_nb1;
    uint x_nb2;
    uint x_nb3;
    uint y_nb0;
    uint y_nb1;
    uint y_nb2;
    uint y_nb3;
};

kernel void rir_cumsum(
    device const uchar* x [[buffer(0)]],
    device uchar* y [[buffer(1)]],
    constant rir_cumsum_params& pc [[buffer(2)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 global_id [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint row_v0 = global_id.x;
    if (row_v0 >= pc.n_row) {
        return;
    }
    const uint plane_v1 = global_id.y;
    if (plane_v1 >= pc.n_plane) {
        return;
    }
    const uint batch_v2 = global_id.z;
    if (batch_v2 >= pc.n_batch) {
        return;
    }
    float scan5_v3 = 0.0f;
    for (uint col_v4 = 0u; col_v4 < pc.n_col; ++col_v4) {
        const float t_v5 = *(device const float *)(x + (col_v4 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        scan5_v3 += t_v5;
        const float s5_v6 = scan5_v3;
        *(device float *)(y + (col_v4 * pc.y_nb0 + row_v0 * pc.y_nb1 + plane_v1 * pc.y_nb2 + batch_v2 * pc.y_nb3)) = s5_v6;
    }
}

struct rir_cumsum_blocked_params {
    uint n_row;
    uint n_plane;
    uint n_batch;
    uint n_col;
    uint x_nb0;
    uint x_nb1;
    uint x_nb2;
    uint x_nb3;
    uint y_nb0;
    uint y_nb1;
    uint y_nb2;
    uint y_nb3;
};

kernel void rir_cumsum_blocked(
    device const uchar* x [[buffer(0)]],
    device uchar* y [[buffer(1)]],
    constant rir_cumsum_blocked_params& pc [[buffer(2)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 global_id [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint row_v0 = group_id.x;
    if (row_v0 >= pc.n_row) {
        return;
    }
    const uint plane_v1 = group_id.y;
    if (plane_v1 >= pc.n_plane) {
        return;
    }
    const uint batch_v2 = group_id.z;
    if (batch_v2 >= pc.n_batch) {
        return;
    }
    const uint lane_v3 = simd_lane_id;
    float part5_v4 = 0.0f;
    const uint chunk_col_v5 = (pc.n_col + 32u - 1u) / 32u;
    const uint beg_col_v5 = min(lane_v3 * chunk_col_v5, pc.n_col);
    const uint end_col_v5 = min((lane_v3 + 1u) * chunk_col_v5, pc.n_col);
    for (uint col_v5 = beg_col_v5; col_v5 < end_col_v5; ++col_v5) {
        const float t_v6 = *(device const float *)(x + (col_v5 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        part5_v4 += t_v6;
    }
    const float off5_v7 = simd_prefix_exclusive_sum(part5_v4);
    float scan5_v9 = 0.0f;
    scan5_v9 += off5_v7;
    const uint chunk_col_v8 = (pc.n_col + 32u - 1u) / 32u;
    const uint beg_col_v8 = min(lane_v3 * chunk_col_v8, pc.n_col);
    const uint end_col_v8 = min((lane_v3 + 1u) * chunk_col_v8, pc.n_col);
    for (uint col_v8 = beg_col_v8; col_v8 < end_col_v8; ++col_v8) {
        const float t_v10 = *(device const float *)(x + (col_v8 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        scan5_v9 += t_v10;
        const float s5_v11 = scan5_v9;
        *(device float *)(y + (col_v8 * pc.y_nb0 + row_v0 * pc.y_nb1 + plane_v1 * pc.y_nb2 + batch_v2 * pc.y_nb3)) = s5_v11;
    }
}
struct rir_cumsum_shared_params {
    uint n_row;
    uint n_plane;
    uint n_batch;
    uint n_col;
    uint x_nb0;
    uint x_nb1;
    uint x_nb2;
    uint x_nb3;
    uint y_nb0;
    uint y_nb1;
    uint y_nb2;
    uint y_nb3;
};

kernel void rir_cumsum_shared(
    device const uchar* x [[buffer(0)]],
    device uchar* y [[buffer(1)]],
    constant rir_cumsum_shared_params& pc [[buffer(2)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 global_id [[thread_position_in_grid]],
    uint threadgroup_lane_id [[thread_index_in_threadgroup]])
{
    threadgroup float rir_shared_off5_v7[256];
    const uint row_v0 = group_id.x;
    if (row_v0 >= pc.n_row) {
        return;
    }
    const uint plane_v1 = group_id.y;
    if (plane_v1 >= pc.n_plane) {
        return;
    }
    const uint batch_v2 = group_id.z;
    if (batch_v2 >= pc.n_batch) {
        return;
    }
    const uint lane_v3 = threadgroup_lane_id;
    float part5_v4 = 0.0f;
    const uint chunk_col_v5 = (pc.n_col + 256u - 1u) / 256u;
    const uint beg_col_v5 = min(lane_v3 * chunk_col_v5, pc.n_col);
    const uint end_col_v5 = min((lane_v3 + 1u) * chunk_col_v5, pc.n_col);
    for (uint col_v5 = beg_col_v5; col_v5 < end_col_v5; ++col_v5) {
        const float t_v6 = *(device const float *)(x + (col_v5 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        part5_v4 += t_v6;
    }
    rir_shared_off5_v7[threadgroup_lane_id] = part5_v4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint up_off5_v7 = 1u; up_off5_v7 < 256u; up_off5_v7 <<= 1u) {
        const uint up_idx_off5_v7 = (threadgroup_lane_id + 1u) * 2u * up_off5_v7 - 1u;
        if (up_idx_off5_v7 < 256u) {
            rir_shared_off5_v7[up_idx_off5_v7] = rir_shared_off5_v7[up_idx_off5_v7 - up_off5_v7] + rir_shared_off5_v7[up_idx_off5_v7];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadgroup_lane_id == 0u) {
        rir_shared_off5_v7[255u] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint down_off5_v7 = 128u; down_off5_v7 > 0u; down_off5_v7 >>= 1u) {
        const uint down_idx_off5_v7 = (threadgroup_lane_id + 1u) * 2u * down_off5_v7 - 1u;
        if (down_idx_off5_v7 < 256u) {
            const float swap_off5_v7 = rir_shared_off5_v7[down_idx_off5_v7 - down_off5_v7];
            rir_shared_off5_v7[down_idx_off5_v7 - down_off5_v7] = rir_shared_off5_v7[down_idx_off5_v7];
            rir_shared_off5_v7[down_idx_off5_v7] = rir_shared_off5_v7[down_idx_off5_v7] + swap_off5_v7;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float off5_v7 = rir_shared_off5_v7[threadgroup_lane_id];
    float scan5_v9 = 0.0f;
    scan5_v9 += off5_v7;
    const uint chunk_col_v8 = (pc.n_col + 256u - 1u) / 256u;
    const uint beg_col_v8 = min(lane_v3 * chunk_col_v8, pc.n_col);
    const uint end_col_v8 = min((lane_v3 + 1u) * chunk_col_v8, pc.n_col);
    for (uint col_v8 = beg_col_v8; col_v8 < end_col_v8; ++col_v8) {
        const float t_v10 = *(device const float *)(x + (col_v8 * pc.x_nb0 + row_v0 * pc.x_nb1 + plane_v1 * pc.x_nb2 + batch_v2 * pc.x_nb3));
        scan5_v9 += t_v10;
        const float s5_v11 = scan5_v9;
        *(device float *)(y + (col_v8 * pc.y_nb0 + row_v0 * pc.y_nb1 + plane_v1 * pc.y_nb2 + batch_v2 * pc.y_nb3)) = s5_v11;
    }
}
// retro delta: RIR-GENERATED-END

