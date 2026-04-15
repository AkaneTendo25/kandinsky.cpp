#ifndef KD_ROPE_HPP
#define KD_ROPE_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

// ── RoPE for Kandinsky 5 ─────────────────────────────────────────────
// Text: 1D rotary embeddings
// Visual: 3D rotary embeddings (temporal, height, width)
//
// The RoPE is precomputed and stored as rotation matrices:
//   cos/sin pairs stacked into [seq, head_dim/2, 2, 2]
// Applied via matrix multiply with pairs of features.

namespace rope {

// Get frequencies: exp(-log(max_period) * arange(0, dim) / dim)
static inline std::vector<float> get_freqs(int dim, float max_period = 10000.0f) {
    std::vector<float> freqs(dim);
    for (int i = 0; i < dim; i++) {
        freqs[i] = expf(-logf(max_period) * (float)i / (float)dim);
    }
    return freqs;
}

// ── 1D RoPE for text tokens ─────────────────────────────────────────

// Compute rope data as a flat vector [seq_len * head_dim]
// Layout (pairwise, matching PyTorch apply_rotary):
//   [cos_0, sin_0, cos_1, sin_1, ...]
static inline std::vector<float> compute_text_rope(int seq_len, int head_dim,
                                                     float theta = 10000.0f) {
    int half_dim = head_dim / 2;
    auto freqs = get_freqs(half_dim, theta);
    std::vector<float> data(seq_len * head_dim);
    for (int pos = 0; pos < seq_len; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float angle = (float)pos * freqs[d];
            data[pos * head_dim + 2 * d + 0] = cosf(angle);
            data[pos * head_dim + 2 * d + 1] = sinf(angle);
        }
    }
    return data;
}

// ── 3D RoPE for visual tokens ────────────────────────────────────────

// Compute 3D rope data as a flat vector [total_tokens * total_dim]
// axes_dims: e.g. {16, 24, 24} for video, {32, 48, 48} for image
// scale_factor: e.g. {1, 1, 1} or {1, 2, 2}
static inline std::vector<float> compute_visual_rope(
        int n_frames, int height, int width,
        std::array<int, 3> axes_dims,
        std::array<float, 3> scale_factor = {1.0f, 1.0f, 1.0f},
        float theta = 10000.0f) {
    int total_dim = axes_dims[0] + axes_dims[1] + axes_dims[2];
    int half_dims[3] = { axes_dims[0] / 2, axes_dims[1] / 2, axes_dims[2] / 2 };

    auto freqs_t = get_freqs(half_dims[0], theta);
    auto freqs_h = get_freqs(half_dims[1], theta);
    auto freqs_w = get_freqs(half_dims[2], theta);

    int total_tokens = n_frames * height * width;
    std::vector<float> data(total_tokens * total_dim);

    for (int t = 0; t < n_frames; t++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = (t * height + h) * width + w;
                float* row = data.data() + idx * total_dim;

                int offset = 0;
                for (int d = 0; d < half_dims[0]; d++) {
                    float angle = (float)t * freqs_t[d] / scale_factor[0];
                    row[offset + 2 * d + 0] = cosf(angle);
                    row[offset + 2 * d + 1] = sinf(angle);
                }
                offset += axes_dims[0];

                for (int d = 0; d < half_dims[1]; d++) {
                    float angle = (float)h * freqs_h[d] / scale_factor[1];
                    row[offset + 2 * d + 0] = cosf(angle);
                    row[offset + 2 * d + 1] = sinf(angle);
                }
                offset += axes_dims[1];

                for (int d = 0; d < half_dims[2]; d++) {
                    float angle = (float)w * freqs_w[d] / scale_factor[2];
                    row[offset + 2 * d + 0] = cosf(angle);
                    row[offset + 2 * d + 1] = sinf(angle);
                }
            }
        }
    }

    return data;
}

// ── Apply RoPE rotation ─────────────────────────────────────────────

// Apply precomputed RoPE to query/key tensor.
// x layout in ggml: [head_dim, seq_len, n_head, batch]
// rope_cos_sin layout: [head_dim, seq_len] with pairwise [cos_i, sin_i].
static inline ggml_tensor* apply(ggml_context* ctx, ggml_tensor* x,
                                  ggml_tensor* rope_cos_sin) {
    const int64_t head_dim = x->ne[0];
    const int64_t n_head = x->ne[1];
    const int64_t seq_len = x->ne[2];
    const int64_t batch = x->ne[3];
    const int64_t half_dim = head_dim / 2;

    // Pair-adjacent representation of x: [2, half_dim, n_head*seq, batch]
    auto* x_pairs = ggml_reshape_4d(ctx, x, 2, half_dim, n_head * seq_len, batch);
    auto* x0_v = ggml_view_4d(ctx, x_pairs, 1, half_dim, n_head * seq_len, batch,
                              x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    auto* x1_v = ggml_view_4d(ctx, x_pairs, 1, half_dim, n_head * seq_len, batch,
                              x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3],
                              x_pairs->nb[0]);
    auto* x0 = ggml_reshape_4d(ctx, ggml_cont(ctx, x0_v), half_dim, n_head, seq_len, batch);
    auto* x1 = ggml_reshape_4d(ctx, ggml_cont(ctx, x1_v), half_dim, n_head, seq_len, batch);

    // Pair-adjacent rope layout: [2, half_dim, seq_len, 1] -> cos/sin
    auto* rope_pairs = ggml_reshape_4d(ctx, rope_cos_sin, 2, half_dim, seq_len, 1);
    auto* cos_v = ggml_view_4d(ctx, rope_pairs, 1, half_dim, seq_len, 1,
                               rope_pairs->nb[1], rope_pairs->nb[2], rope_pairs->nb[3], 0);
    auto* sin_v = ggml_view_4d(ctx, rope_pairs, 1, half_dim, seq_len, 1,
                               rope_pairs->nb[1], rope_pairs->nb[2], rope_pairs->nb[3],
                               rope_pairs->nb[0]);
    auto* cos_vals = ggml_reshape_4d(ctx, ggml_cont(ctx, cos_v), half_dim, 1, seq_len, 1);
    auto* sin_vals = ggml_reshape_4d(ctx, ggml_cont(ctx, sin_v), half_dim, 1, seq_len, 1);

    auto* out0 = ggml_sub(ctx, ggml_mul(ctx, x0, cos_vals), ggml_mul(ctx, x1, sin_vals));
    auto* out1 = ggml_add(ctx, ggml_mul(ctx, x0, sin_vals), ggml_mul(ctx, x1, cos_vals));

    // Re-pack pairs back to [head_dim, seq_len, n_head, batch]
    auto* out0_v = ggml_reshape_4d(ctx, out0, 1, half_dim, n_head * seq_len, batch);
    auto* out1_v = ggml_reshape_4d(ctx, out1, 1, half_dim, n_head * seq_len, batch);

    ggml_tensor* out_pairs = nullptr;
    if (batch == 1) {
        // CUDA concat along dim0 can exceed grid.z for large seq; concat along dim3
        // takes the memcpy path in ggml-cuda and avoids that limit.
        auto* stacked = ggml_concat(ctx, out0_v, out1_v, 3); // [1, half, n_head*seq, 2]
        auto* perm = ggml_permute(ctx, stacked, 3, 1, 2, 0); // [2, half, n_head*seq, 1]
        out_pairs = ggml_cont(ctx, perm);
    } else {
        out_pairs = ggml_concat(ctx, out0_v, out1_v, 0); // [2, half_dim, n_head*seq, batch]
    }
    return ggml_reshape_4d(ctx, out_pairs, head_dim, n_head, seq_len, batch);
}

} // namespace rope

#endif // KD_ROPE_HPP
