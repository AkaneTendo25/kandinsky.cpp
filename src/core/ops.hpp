#ifndef KD_OPS_HPP
#define KD_OPS_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

// Forward-declare RNG for fill_randn
class RNG;

#define KD_EPS 1e-05f

namespace ops {

// ── Utility ──────────────────────────────────────────────────────────

static inline ggml_tensor* cont(ggml_context* ctx, ggml_tensor* x) {
    return ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
}

// ── Linear ───────────────────────────────────────────────────────────

static inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                                   ggml_tensor* w, ggml_tensor* b = nullptr) {
    // x: [..., in_features]
    // w: [in_features, out_features]
    // Flatten high dims if needed for large batches
    if (x->ne[2] * x->ne[3] > 1024) {
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * ne2 * ne3);
        x = ggml_mul_mat(ctx, w, x);
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
    }
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// ── Conv2d ───────────────────────────────────────────────────────────

// w: [OC, IC, KH, KW], x: [N, IC, IH, IW], b: [OC]
static inline ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* x,
                                   ggml_tensor* w, ggml_tensor* b,
                                   int s0 = 1, int s1 = 1,
                                   int p0 = 0, int p1 = 0,
                                   int d0 = 1, int d1 = 1) {
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    if (b) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;
}

// ── LayerNorm ────────────────────────────────────────────────────────

static inline ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* w = nullptr, ggml_tensor* b = nullptr,
                                       float eps = KD_EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w) {
        x = ggml_mul(ctx, x, w);
        if (b) x = ggml_add(ctx, x, b);
    }
    return x;
}

// ── RMS Norm ─────────────────────────────────────────────────────────

static inline ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x,
                                     ggml_tensor* w, float eps = 1e-6f) {
    x = ggml_rms_norm(ctx, x, eps);
    if (w) x = ggml_mul(ctx, x, w);
    return x;
}

// ── Group Norm ───────────────────────────────────────────────────────

static inline ggml_tensor* group_norm(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* w, ggml_tensor* b,
                                       int num_groups = 32, float eps = 1e-6f) {
    if (ggml_n_dims(x) >= 3 && w && b) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }
    x = ggml_group_norm(ctx, x, num_groups, eps);
    if (w && b) {
        x = ggml_mul(ctx, x, w);
        x = ggml_add(ctx, x, b);
    }
    return x;
}

// ── Scaled Dot-Product Attention ─────────────────────────────────────

// q: [N, L_q, C]  where C = n_head * d_head
// k: [N, L_k, C_kv]
// v: [N, L_k, C_kv]
// mask: [N, L_q, L_k] or nullptr
// Returns: [N, L_q, C]
static inline ggml_tensor* attention(ggml_context* ctx,
                                      ggml_backend_t backend,
                                      ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                      int64_t n_head,
                                      ggml_tensor* mask = nullptr,
                                      bool flash = false,
                                      bool causal = false) {
    int64_t L_q    = q->ne[1];
    int64_t L_k    = k->ne[1];
    int64_t C      = q->ne[0];
    int64_t N      = q->ne[2];
    int64_t d_head = C / n_head;
    int64_t n_kv_head = k->ne[0] / d_head;

    // Reshape to multi-head: [N * n_head, L, d_head]
    q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);
    q = cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);

    // Keep K/V in [d_head, L_k, n_kv_head, N] layout for explicit GQA handling.
    auto* k4 = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);
    k4 = cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3));

    auto* v4 = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);
    v4 = cont(ctx, ggml_permute(ctx, v4, 0, 2, 1, 3));

    // Expand grouped KV heads to full query head count when needed (GQA).
    // Qwen uses grouped-query attention (e.g. n_head=28, n_kv_head=4).
    if (n_kv_head != n_head) {
        if (n_head % n_kv_head != 0) {
            LOG_ERROR("attention: invalid GQA ratio n_head=%lld n_kv_head=%lld",
                      (long long)n_head, (long long)n_kv_head);
            return nullptr;
        }

        const int64_t n_rep = n_head / n_kv_head;

        // Build [d, L, n_rep, n_kv*N] so that reshaping to [d, L, n_head, N]
        // yields contiguous groups per KV head: [kv0 x n_rep, kv1 x n_rep, ...].
        auto* k_fold = ggml_reshape_4d(ctx, k4, d_head, L_k, 1, n_kv_head * N);
        auto* k_rep_shape = ggml_new_tensor_4d(ctx, k4->type, d_head, L_k, n_rep, n_kv_head * N);
        k_fold = ggml_repeat(ctx, k_fold, k_rep_shape);
        k4 = ggml_reshape_4d(ctx, k_fold, d_head, L_k, n_head, N);

        auto* v_fold = ggml_reshape_4d(ctx, v4, d_head, L_k, 1, n_kv_head * N);
        auto* v_rep_shape = ggml_new_tensor_4d(ctx, v4->type, d_head, L_k, n_rep, n_kv_head * N);
        v_fold = ggml_repeat(ctx, v_fold, v_rep_shape);
        v4 = ggml_reshape_4d(ctx, v_fold, d_head, L_k, n_head, N);

        n_kv_head = n_head;
    }

    k = ggml_reshape_3d(ctx, k4, d_head, L_k, n_kv_head * N);

    float scale = 1.0f / sqrtf((float)d_head);
    ggml_tensor* kqv = nullptr;

    bool use_flash = flash && !causal;
    if (use_flash) {
        // Try flash attention path
        k = ggml_cast(ctx, k, GGML_TYPE_F16);
        auto v_perm = ggml_reshape_3d(ctx, v4, d_head, L_k, n_kv_head * N);
        v_perm = ggml_cast(ctx, v_perm, GGML_TYPE_F16);

        ggml_tensor* mask_f16 = nullptr;
        if (mask) {
            mask_f16 = ggml_cast(ctx, ggml_transpose(ctx, mask), GGML_TYPE_F16);
        }

        kqv = ggml_flash_attn_ext(ctx, q, k, v_perm, mask_f16, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);

        if (backend && !ggml_backend_supports_op(backend, kqv)) {
            use_flash = false;
            kqv = nullptr;
        } else {
            kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
        }
    }

    if (!use_flash) {
        // Standard attention path
        auto v2 = cont(ctx, ggml_permute(ctx, v4, 1, 0, 2, 3));
        v2 = ggml_reshape_3d(ctx, v2, L_k, d_head, n_kv_head * N);

        auto kq = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale(ctx, kq, scale);
        if (mask) kq = ggml_add(ctx, kq, mask);
        if (causal) kq = ggml_diag_mask_inf(ctx, kq, 0);
        kq = ggml_soft_max(ctx, kq);

        kqv = ggml_mul_mat(ctx, v2, kq);
        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    }

    kqv = cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);
    return kqv;
}

// ── GELU activation ──────────────────────────────────────────────────

static inline ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu(ctx, cont(ctx, x));
}

// ── SiLU activation ──────────────────────────────────────────────────

static inline ggml_tensor* silu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_silu(ctx, cont(ctx, x));
}

// ── Timestep embedding (sinusoidal) ──────────────────────────────────

static inline ggml_tensor* timestep_embedding(ggml_context* ctx, ggml_tensor* t,
                                               int dim, int max_period = 10000) {
    return ggml_timestep_embedding(ctx, t, dim, max_period);
}

// ── Helper: vector to tensor ─────────────────────────────────────────

static inline ggml_tensor* vec_to_tensor(ggml_context* ctx, const std::vector<float>& vec) {
    auto t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, vec.data(), ggml_nbytes(t));
    return t;
}

// ── Helper: set tensor from float data ───────────────────────────────

static inline void tensor_set_f32(ggml_tensor* t, float val,
                                   int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    assert(t->nb[0] == sizeof(float));
    *(float*)((char*)t->data + i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0]) = val;
}

static inline float tensor_get_f32(const ggml_tensor* t,
                                    int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (t->buffer) {
        float val;
        ggml_backend_tensor_get(t, &val,
            i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0], sizeof(float));
        return val;
    }
    assert(t->nb[0] == sizeof(float));
    return *(float*)((char*)t->data + i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0]);
}

// ── Fill tensor with random normal ───────────────────────────────────

static inline void fill_randn(ggml_tensor* t, std::shared_ptr<RNG> rng) {
    assert(t->type == GGML_TYPE_F32 && "fill_randn requires F32 tensor");
    assert(t->data && "fill_randn requires CPU tensor with data");
    uint32_t n = (uint32_t)ggml_nelements(t);
    auto vals = rng->randn(n);
    memcpy(t->data, vals.data(), n * sizeof(float));
}

} // namespace ops

#endif // KD_OPS_HPP
