#ifndef KD_CROSSDIT_HPP
#define KD_CROSSDIT_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/module.hpp"
#include "core/executor.hpp"
#include "rope.hpp"

// ── Kandinsky 5 CrossDiT (DiffusionTransformer3D) ────────────────────
//
// Three variants detected from tensor shapes:
//   Lite Video: model_dim=1792, ff_dim=7168,  2t + 32v blocks, axes_dims=[16,24,24]
//   Lite Image: model_dim=2560, ff_dim=10240, 2t + 50v blocks, axes_dims=[32,48,48]
//   Pro Video:  model_dim=4096, ff_dim=16384, 4t + 60v blocks, axes_dims=[32,48,48]
//
// Pipeline: time_embed + text_proj(qwen) + pool_proj(clip) → text blocks → visual blocks → output

// ── Configuration ────────────────────────────────────────────────────

struct CrossDiTConfig {
    int64_t model_dim        = 1792;
    int64_t ff_dim           = 7168;
    int64_t time_dim         = 512;
    int64_t in_visual_dim    = 132;   // actual linear input: (2*16+1) * prod([1,2,2]) for visual_cond
    int64_t out_visual_dim   = 64;    // actual linear output: 16 * prod([1,2,2])
    int64_t in_text_dim      = 3584;  // Qwen hidden size
    int64_t in_text_dim2     = 768;   // CLIP pooled size
    int     num_text_blocks  = 2;
    int     num_visual_blocks = 32;
    std::array<int, 3> axes_dims = {16, 24, 24};  // head_dim = sum = 64 → num_heads = 1792/64 = 28
    std::array<int, 3> patch_size = {1, 2, 2};
    // Kandinsky-5 video checkpoints use anisotropic RoPE scaling: [1, 2, 2].
    std::array<float, 3> scale_factor = {1.0f, 2.0f, 2.0f};
    // Number of DiT transformer blocks to keep on GPU (from the tail).
    // -1 means all on GPU (default behavior).
    int gpu_layers = -1;

    int64_t head_dim() const { return axes_dims[0] + axes_dims[1] + axes_dims[2]; }
    int64_t num_heads() const { return model_dim / head_dim(); }

    // Pre-configured variants
    static CrossDiTConfig lite_video() {
        CrossDiTConfig c;
        c.model_dim = 1792; c.ff_dim = 7168; c.time_dim = 512;
        c.in_visual_dim = 132; c.out_visual_dim = 64;
        c.num_text_blocks = 2; c.num_visual_blocks = 32;
        c.axes_dims = {16, 24, 24};
        return c;
    }

    static CrossDiTConfig lite_image() {
        CrossDiTConfig c;
        c.model_dim = 2560; c.ff_dim = 10240; c.time_dim = 512;
        c.in_visual_dim = 64; c.out_visual_dim = 64;  // no visual_cond for image
        c.num_text_blocks = 2; c.num_visual_blocks = 50;
        c.axes_dims = {32, 48, 48};
        c.scale_factor = {1.0f, 2.0f, 2.0f};
        return c;
    }

    static CrossDiTConfig pro_video() {
        CrossDiTConfig c;
        c.model_dim = 4096; c.ff_dim = 16384; c.time_dim = 1024;
        c.in_visual_dim = 132; c.out_visual_dim = 64;
        c.num_text_blocks = 4; c.num_visual_blocks = 60;
        c.axes_dims = {32, 48, 48};  // head_dim=128, num_heads=32
        c.scale_factor = {1.0f, 2.0f, 2.0f};
        return c;
    }
};

// ── AdaLN Modulation ─────────────────────────────────────────────────
// SiLU(time_embed) → Linear → N*model_dim params, chunked into groups of 3

class Modulation : public Module {
public:
    int64_t time_dim, model_dim;
    int     num_params;  // 6 for text (2 sublayers × 3), 9 for visual (3 sublayers × 3)

    Modulation() : time_dim(512), model_dim(2560), num_params(6) {}
    Modulation(int64_t td, int64_t md, int np)
        : time_dim(td), model_dim(md), num_params(np) {
        submodules["out_layer"] = std::make_shared<Linear>(td, np * md, true);
    }

    // Returns modulation parameters tensor [batch, num_params * model_dim]
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* time_embed) {
        auto* ctx = fctx->ctx;
        auto* x = ggml_silu(ctx, time_embed);
        return std::static_pointer_cast<Linear>(submodules["out_layer"])->forward(fctx, x);
    }
};

// ── Self-Attention with QK-norm ──────────────────────────────────────

class DiTSelfAttention : public Module {
public:
    int64_t model_dim, head_dim;

    DiTSelfAttention() : model_dim(2560), head_dim(128) {}
    DiTSelfAttention(int64_t md, int64_t hd)
        : model_dim(md), head_dim(hd) {
        submodules["to_query"]   = std::make_shared<Linear>(md, md, true);
        submodules["to_key"]     = std::make_shared<Linear>(md, md, true);
        submodules["to_value"]   = std::make_shared<Linear>(md, md, true);
        submodules["query_norm"] = std::make_shared<RMSNorm>(hd);
        submodules["key_norm"]   = std::make_shared<RMSNorm>(hd);
        submodules["out_layer"]  = std::make_shared<Linear>(md, md, true);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x,
                          ggml_tensor* rope_data = nullptr,
                          ggml_tensor* mask = nullptr) {
        auto* ctx = fctx->ctx;
        int64_t n_head = model_dim / head_dim;

        auto* q = std::static_pointer_cast<Linear>(submodules["to_query"])->forward(fctx, x);
        auto* k = std::static_pointer_cast<Linear>(submodules["to_key"])->forward(fctx, x);
        auto* v = std::static_pointer_cast<Linear>(submodules["to_value"])->forward(fctx, x);

        // Per-head QK norm
        // q: [batch, seq, model_dim] → reshape → [batch, seq, n_head, head_dim] → norm → reshape back
        int64_t seq = q->ne[1];
        int64_t batch = q->ne[2];

        q = ggml_reshape_4d(ctx, q, head_dim, n_head, seq, batch);
        k = ggml_reshape_4d(ctx, k, head_dim, n_head, seq, batch);

        q = std::static_pointer_cast<RMSNorm>(submodules["query_norm"])->forward(fctx, q);
        k = std::static_pointer_cast<RMSNorm>(submodules["key_norm"])->forward(fctx, k);

        // Apply RoPE if provided
        if (rope_data) {
            q = rope::apply(ctx, q, rope_data);
            k = rope::apply(ctx, k, rope_data);
        }

        // Reshape back to [batch, seq, model_dim]
        q = ggml_reshape_3d(ctx, q, model_dim, seq, batch);
        k = ggml_reshape_3d(ctx, k, model_dim, seq, batch);
        v = ggml_reshape_3d(ctx, v, model_dim, seq, batch);

        // Attention
        auto* attn = ops::attention(ctx, fctx->backend, q, k, v, n_head, mask, fctx->flash_attn);

        // Output projection
        return std::static_pointer_cast<Linear>(submodules["out_layer"])->forward(fctx, attn);
    }
};

// ── Cross-Attention ──────────────────────────────────────────────────

class DiTCrossAttention : public Module {
public:
    int64_t model_dim, head_dim;

    DiTCrossAttention() : model_dim(2560), head_dim(128) {}
    DiTCrossAttention(int64_t md, int64_t hd)
        : model_dim(md), head_dim(hd) {
        submodules["to_query"]   = std::make_shared<Linear>(md, md, true);
        submodules["to_key"]     = std::make_shared<Linear>(md, md, true);
        submodules["to_value"]   = std::make_shared<Linear>(md, md, true);
        submodules["query_norm"] = std::make_shared<RMSNorm>(hd);
        submodules["key_norm"]   = std::make_shared<RMSNorm>(hd);
        submodules["out_layer"]  = std::make_shared<Linear>(md, md, true);
    }

    ggml_tensor* forward(ForwardContext* fctx,
                          ggml_tensor* x_query,    // visual
                          ggml_tensor* x_kv,       // text
                          ggml_tensor* mask = nullptr) {
        auto* ctx = fctx->ctx;
        int64_t n_head = model_dim / head_dim;

        auto* q = std::static_pointer_cast<Linear>(submodules["to_query"])->forward(fctx, x_query);
        auto* k = std::static_pointer_cast<Linear>(submodules["to_key"])->forward(fctx, x_kv);
        auto* v = std::static_pointer_cast<Linear>(submodules["to_value"])->forward(fctx, x_kv);

        // Per-head QK norm
        int64_t seq_q = q->ne[1], seq_kv = k->ne[1];
        int64_t batch = q->ne[2];

        q = ggml_reshape_4d(ctx, q, head_dim, n_head, seq_q, batch);
        k = ggml_reshape_4d(ctx, k, head_dim, n_head, seq_kv, batch);

        q = std::static_pointer_cast<RMSNorm>(submodules["query_norm"])->forward(fctx, q);
        k = std::static_pointer_cast<RMSNorm>(submodules["key_norm"])->forward(fctx, k);

        q = ggml_reshape_3d(ctx, q, model_dim, seq_q, batch);
        k = ggml_reshape_3d(ctx, k, model_dim, seq_kv, batch);

        auto* attn = ops::attention(ctx, fctx->backend, q, k, v, n_head, mask, fctx->flash_attn);
        return std::static_pointer_cast<Linear>(submodules["out_layer"])->forward(fctx, attn);
    }
};

// ── Feed-Forward (GELU, no bias) ─────────────────────────────────────

class DiTFeedForward : public Module {
public:
    DiTFeedForward() = default;
    DiTFeedForward(int64_t model_dim, int64_t ff_dim) {
        submodules["in_layer"]  = std::make_shared<Linear>(model_dim, ff_dim, false);
        submodules["out_layer"] = std::make_shared<Linear>(ff_dim, model_dim, false);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        x = std::static_pointer_cast<Linear>(submodules["in_layer"])->forward(fctx, x);
        x = ops::gelu(fctx->ctx, x);
        x = std::static_pointer_cast<Linear>(submodules["out_layer"])->forward(fctx, x);
        return x;
    }
};

// ── AdaLN helper: apply scale+shift modulation ───────────────────────

static inline ggml_tensor* ada_modulate(ggml_context* ctx, ggml_tensor* x,
                                         ggml_tensor* shift, ggml_tensor* scale) {
    // x = x * (1 + scale) + shift
    auto* ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    x = ggml_mul(ctx, x, ggml_add(ctx, scale, ggml_repeat(ctx, ones, scale)));
    x = ggml_add(ctx, x, shift);
    return x;
}

// ── Text Encoder Block ───────────────────────────────────────────────

class TextBlock : public Module {
public:
    int64_t model_dim, ff_dim, time_dim, head_dim;

    TextBlock() = default;
    TextBlock(int64_t md, int64_t fd, int64_t td, int64_t hd)
        : model_dim(md), ff_dim(fd), time_dim(td), head_dim(hd) {
        submodules["text_modulation"] = std::make_shared<Modulation>(td, md, 6);
        submodules["self_attention"]  = std::make_shared<DiTSelfAttention>(md, hd);
        submodules["feed_forward"]   = std::make_shared<DiTFeedForward>(md, fd);
        // LayerNorm with affine=false (no learned params — modulation provides shift/scale)
        sa_norm_ = LayerNorm(md, 1e-5f, false);
        ff_norm_ = LayerNorm(md, 1e-5f, false);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x,
                          ggml_tensor* time_embed,
                          ggml_tensor* rope_data = nullptr,
                          ggml_tensor* mask = nullptr) {
        auto* ctx = fctx->ctx;

        // Get 6 modulation params: [shift1, scale1, gate1, shift2, scale2, gate2]
        auto* mod_params = std::static_pointer_cast<Modulation>(
            submodules["text_modulation"])->forward(fctx, time_embed);

        // Chunk into 6 parts along last dim
        int64_t chunk_size = model_dim;

        // Self-attention sublayer
        auto* shift1 = ggml_view_2d(ctx, mod_params, chunk_size, mod_params->ne[1],
                                     mod_params->nb[1], 0 * chunk_size * sizeof(float));
        auto* scale1 = ggml_view_2d(ctx, mod_params, chunk_size, mod_params->ne[1],
                                     mod_params->nb[1], 1 * chunk_size * sizeof(float));
        auto* gate1  = ggml_view_2d(ctx, mod_params, chunk_size, mod_params->ne[1],
                                     mod_params->nb[1], 2 * chunk_size * sizeof(float));

        auto* residual = x;
        x = sa_norm_.forward(fctx, x);
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, scale1,
            ggml_repeat(ctx, fctx->one, scale1))), shift1);
        x = std::static_pointer_cast<DiTSelfAttention>(
            submodules["self_attention"])->forward(fctx, x, rope_data, mask);
        x = ggml_mul(ctx, x, gate1);
        x = ggml_add(ctx, x, residual);

        // Feed-forward sublayer
        auto* shift2 = ggml_view_2d(ctx, mod_params, chunk_size, mod_params->ne[1],
                                     mod_params->nb[1], 3 * chunk_size * sizeof(float));
        auto* scale2 = ggml_view_2d(ctx, mod_params, chunk_size, mod_params->ne[1],
                                     mod_params->nb[1], 4 * chunk_size * sizeof(float));
        auto* gate2  = ggml_view_2d(ctx, mod_params, chunk_size, mod_params->ne[1],
                                     mod_params->nb[1], 5 * chunk_size * sizeof(float));

        residual = x;
        x = ff_norm_.forward(fctx, x);
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, scale2,
            ggml_repeat(ctx, fctx->one, scale2))), shift2);
        x = std::static_pointer_cast<DiTFeedForward>(
            submodules["feed_forward"])->forward(fctx, x);
        x = ggml_mul(ctx, x, gate2);
        x = ggml_add(ctx, x, residual);

        return x;
    }

private:
    LayerNorm sa_norm_, ff_norm_;
};

// ── Visual Decoder Block ─────────────────────────────────────────────

class VisualBlock : public Module {
public:
    int64_t model_dim, ff_dim, time_dim, head_dim;

    VisualBlock() = default;
    VisualBlock(int64_t md, int64_t fd, int64_t td, int64_t hd)
        : model_dim(md), ff_dim(fd), time_dim(td), head_dim(hd) {
        submodules["visual_modulation"] = std::make_shared<Modulation>(td, md, 9);
        submodules["self_attention"]    = std::make_shared<DiTSelfAttention>(md, hd);
        submodules["cross_attention"]   = std::make_shared<DiTCrossAttention>(md, hd);
        submodules["feed_forward"]     = std::make_shared<DiTFeedForward>(md, fd);
        // LayerNorm with affine=false (modulation provides shift/scale)
        sa_norm_ = LayerNorm(md, 1e-5f, false);
        ca_norm_ = LayerNorm(md, 1e-5f, false);
        ff_norm_ = LayerNorm(md, 1e-5f, false);
    }

    ggml_tensor* forward(ForwardContext* fctx,
                          ggml_tensor* visual,
                          ggml_tensor* text,
                          ggml_tensor* time_embed,
                          ggml_tensor* visual_rope = nullptr,
                          ggml_tensor* text_mask = nullptr) {
        auto* ctx = fctx->ctx;

        // Get 9 modulation params
        auto* mod_params = std::static_pointer_cast<Modulation>(
            submodules["visual_modulation"])->forward(fctx, time_embed);

        int64_t cs = model_dim;
        auto view_mod = [&](int idx) {
            return ggml_view_2d(ctx, mod_params, cs, mod_params->ne[1],
                                mod_params->nb[1], idx * cs * sizeof(float));
        };

        // Self-attention sublayer (indices 0,1,2)
        auto* residual = visual;
        auto* x = sa_norm_.forward(fctx, visual);
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, view_mod(1),
            ggml_repeat(ctx, fctx->one, view_mod(1)))), view_mod(0));
        x = std::static_pointer_cast<DiTSelfAttention>(
            submodules["self_attention"])->forward(fctx, x, visual_rope);
        x = ggml_mul(ctx, x, view_mod(2));
        visual = ggml_add(ctx, x, residual);

        // Cross-attention sublayer (indices 3,4,5)
        residual = visual;
        x = ca_norm_.forward(fctx, visual);
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, view_mod(4),
            ggml_repeat(ctx, fctx->one, view_mod(4)))), view_mod(3));
        x = std::static_pointer_cast<DiTCrossAttention>(
            submodules["cross_attention"])->forward(fctx, x, text, text_mask);
        x = ggml_mul(ctx, x, view_mod(5));
        visual = ggml_add(ctx, x, residual);

        // Feed-forward sublayer (indices 6,7,8)
        residual = visual;
        x = ff_norm_.forward(fctx, visual);
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, view_mod(7),
            ggml_repeat(ctx, fctx->one, view_mod(7)))), view_mod(6));
        x = std::static_pointer_cast<DiTFeedForward>(
            submodules["feed_forward"])->forward(fctx, x);
        x = ggml_mul(ctx, x, view_mod(8));
        visual = ggml_add(ctx, x, residual);

        return visual;
    }

private:
    LayerNorm sa_norm_, ca_norm_, ff_norm_;
};

// ── Output Layer ─────────────────────────────────────────────────────

class DiTOutLayer : public Module {
public:
    int64_t model_dim, time_dim, out_dim;

    DiTOutLayer() = default;
    DiTOutLayer(int64_t md, int64_t td, int64_t od) : model_dim(md), time_dim(td), out_dim(od) {
        submodules["modulation"] = std::make_shared<Modulation>(td, md, 2);
        submodules["out_layer"]  = std::make_shared<Linear>(md, od, true);
        norm_ = LayerNorm(md, 1e-5f, false);  // affine=false, no learned params
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x, ggml_tensor* time_embed) {
        auto* ctx = fctx->ctx;

        auto* mod_params = std::static_pointer_cast<Modulation>(
            submodules["modulation"])->forward(fctx, time_embed);

        int64_t cs = model_dim;
        auto* shift = ggml_view_2d(ctx, mod_params, cs, mod_params->ne[1],
                                    mod_params->nb[1], 0);
        auto* scale = ggml_view_2d(ctx, mod_params, cs, mod_params->ne[1],
                                    mod_params->nb[1], cs * sizeof(float));

        x = norm_.forward(fctx, x);
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, scale,
            ggml_repeat(ctx, fctx->one, scale))), shift);
        x = std::static_pointer_cast<Linear>(submodules["out_layer"])->forward(fctx, x);

        return x;
    }

private:
    LayerNorm norm_;
};

// ── CrossDiT: Top-level DiffusionTransformer3D ───────────────────────

class CrossDiT : public Executor {
public:
    CrossDiTConfig cfg;

    CrossDiT(ggml_backend_t backend, CrossDiTConfig config = {})
        : Executor(backend), cfg(config) {}

    std::string get_desc() override { return "CrossDiT"; }

    bool init(ggml_type wtype = GGML_TYPE_F16) {
        auto* ctx = params_ctx();
        if (!ctx) {
            LOG_ERROR("CrossDiT: params context is not initialized");
            return false;
        }

        // Re-create embedding modules with correct dimensions
        time_in_   = Linear(cfg.model_dim, cfg.time_dim, true);
        time_out_  = Linear(cfg.time_dim, cfg.time_dim, true);
        text_proj_ = Linear(cfg.in_text_dim, cfg.model_dim, true);
        text_norm_ = LayerNorm(cfg.model_dim);
        pool_proj_ = Linear(cfg.in_text_dim2, cfg.time_dim, true);
        pool_norm_ = LayerNorm(cfg.time_dim);
        visual_in_ = Linear(cfg.in_visual_dim, cfg.model_dim, true);
        out_layer_ = DiTOutLayer(cfg.model_dim, cfg.time_dim, cfg.out_visual_dim);

        // Time embeddings: sinusoidal → time_in → SiLU → time_out
        time_in_.init(ctx, wtype, "time_embeddings.in_layer");
        time_out_.init(ctx, wtype, "time_embeddings.out_layer");

        // Text projection: qwen hidden → model_dim
        text_proj_.init(ctx, wtype, "text_embeddings.in_layer");
        text_norm_.init(ctx, GGML_TYPE_F32, "text_embeddings.norm");

        // CLIP pooled projection: 768 → time_dim
        pool_proj_.init(ctx, wtype, "pooled_text_embeddings.in_layer");
        pool_norm_.init(ctx, GGML_TYPE_F32, "pooled_text_embeddings.norm");

        // Visual input projection: in_visual_dim → model_dim
        visual_in_.init(ctx, wtype, "visual_embeddings.in_layer");

        // Text blocks
        for (int i = 0; i < cfg.num_text_blocks; i++) {
            text_blocks_.push_back(TextBlock(cfg.model_dim, cfg.ff_dim, cfg.time_dim, cfg.head_dim()));
            text_blocks_.back().init(ctx, wtype, "text_transformer_blocks." + std::to_string(i));
        }

        // Visual blocks
        for (int i = 0; i < cfg.num_visual_blocks; i++) {
            visual_blocks_.push_back(VisualBlock(cfg.model_dim, cfg.ff_dim, cfg.time_dim, cfg.head_dim()));
            visual_blocks_.back().init(ctx, wtype, "visual_transformer_blocks." + std::to_string(i));
        }

        // Output layer
        out_layer_.init(ctx, wtype, "out_layer");

        // Optional llama.cpp-style layer split: keep the last N DiT blocks on GPU,
        // offload earlier blocks to CPU RAM.
        if (cfg.gpu_layers >= 0 && !ggml_backend_is_cpu(runtime_backend())) {
            const int total_layers = cfg.num_text_blocks + cfg.num_visual_blocks;
            const int gpu_layers = std::max(0, std::min(cfg.gpu_layers, total_layers));
            const int cpu_layers = total_layers - gpu_layers;

            if (cpu_layers > 0) {
                ggml_backend_t cpu_backend = ensure_cpu_split_backend();
                std::map<std::string, ggml_tensor*> tensors;
                collect_all_params(tensors);

                auto parse_idx = [](const std::string& name, const std::string& prefix) -> int {
                    if (name.rfind(prefix, 0) != 0) return -1;
                    size_t p = prefix.size();
                    size_t q = p;
                    while (q < name.size() && name[q] >= '0' && name[q] <= '9') q++;
                    if (q == p) return -1;
                    return std::atoi(name.substr(p, q - p).c_str());
                };

                int offloaded_tensors = 0;
                for (auto& kv : tensors) {
                    const std::string& name = kv.first;
                    ggml_tensor* t = kv.second;

                    int global_layer = -1;
                    int ti = parse_idx(name, "text_transformer_blocks.");
                    if (ti >= 0) {
                        global_layer = ti;
                    } else {
                        int vi = parse_idx(name, "visual_transformer_blocks.");
                        if (vi >= 0) {
                            global_layer = cfg.num_text_blocks + vi;
                        }
                    }

                    if (global_layer >= 0 && global_layer < cpu_layers) {
                        set_param_tensor_backend(t, cpu_backend);
                        offloaded_tensors++;
                    }
                }

                LOG_INFO("CrossDiT layer split: gpu_layers=%d/%d, cpu_layers=%d, offloaded_tensors=%d",
                         gpu_layers, total_layers, cpu_layers, offloaded_tensors);
            } else {
                LOG_INFO("CrossDiT layer split: gpu_layers=%d/%d, cpu_layers=0", gpu_layers, total_layers);
            }
        }

        return alloc_params_buffer();
    }

    void collect_all_params(std::map<std::string, ggml_tensor*>& tensors) {
        time_in_.collect_params(tensors, "time_embeddings.in_layer");
        time_out_.collect_params(tensors, "time_embeddings.out_layer");
        text_proj_.collect_params(tensors, "text_embeddings.in_layer");
        text_norm_.collect_params(tensors, "text_embeddings.norm");
        pool_proj_.collect_params(tensors, "pooled_text_embeddings.in_layer");
        pool_norm_.collect_params(tensors, "pooled_text_embeddings.norm");
        visual_in_.collect_params(tensors, "visual_embeddings.in_layer");

        for (int i = 0; i < cfg.num_text_blocks; i++) {
            text_blocks_[i].collect_params(tensors, "text_transformer_blocks." + std::to_string(i));
        }
        for (int i = 0; i < cfg.num_visual_blocks; i++) {
            visual_blocks_[i].collect_params(tensors, "visual_transformer_blocks." + std::to_string(i));
        }
        out_layer_.collect_params(tensors, "out_layer");
    }

    // Forward pass: latent + text embeddings + pooled clip → predicted velocity
    // Forward pass: visual_patched [in_visual_dim, n_visual_tokens] (pre-patchified)
    //   + text embeddings + pooled clip → predicted velocity [out_visual_dim, n_visual_tokens]
    bool forward(ggml_tensor* latent,          // [in_visual_dim, n_visual_tokens] pre-patchified
                 ggml_tensor* text_embed,       // [3584, seq_len, 1]
                 ggml_tensor* pooled_embed,     // [768]
                 float timestep,                // scalar timestep (flow-matching, t*1000 so range [0, 1000])
                 int n_frames, int lat_h, int lat_w,
                 int n_threads,
                 ggml_tensor** output,
                 ggml_context* out_ctx) {
        // Pre-compute RoPE data on CPU (must outlive the lambda for set_backend_tensor_data)
        int ph = cfg.patch_size[1], pw = cfg.patch_size[2];
        int n_patches_h = lat_h / ph;
        int n_patches_w = lat_w / pw;
        int n_visual_tokens = (n_frames / cfg.patch_size[0]) * n_patches_h * n_patches_w;
        int text_seq_len = (int)text_embed->ne[1];
        int head_dim = (int)cfg.head_dim();

        auto text_rope_data = rope::compute_text_rope(text_seq_len, head_dim);
        auto vis_rope_data  = rope::compute_visual_rope(
            n_frames / cfg.patch_size[0], n_patches_h, n_patches_w,
            cfg.axes_dims, cfg.scale_factor);

        auto build_graph = [&]() -> ggml_cgraph* {
            auto* ctx = compute_ctx_;
            auto* gf = ggml_new_graph_custom(ctx, MAX_GRAPH_SIZE, false);
            ForwardContext fctx = get_forward_ctx();

            // 1. Time embedding: sinusoidal(t) → Linear → SiLU → Linear
            auto* t_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            set_backend_tensor_data(t_scalar, &timestep);
            auto* t_emb = ops::timestep_embedding(ctx, t_scalar, (int)cfg.model_dim);
            t_emb = time_in_.forward(&fctx, t_emb);
            t_emb = ggml_silu(ctx, t_emb);
            t_emb = time_out_.forward(&fctx, t_emb);

            // 2. Add CLIP pooled embedding
            auto* p_emb = to_backend(pooled_embed);
            p_emb = pool_proj_.forward(&fctx, p_emb);
            p_emb = pool_norm_.forward(&fctx, p_emb);
            t_emb = ggml_add(ctx, t_emb, p_emb);

            // 3. Text embedding: project Qwen hidden states
            auto* text = to_backend(text_embed);
            text = text_proj_.forward(&fctx, text);
            text = text_norm_.forward(&fctx, text);

            // 4. Visual input (already patchified by caller)
            auto* vis = to_backend(latent);
            vis = ggml_reshape_2d(ctx, vis, cfg.in_visual_dim, n_visual_tokens);
            vis = visual_in_.forward(&fctx, vis);
            vis = ggml_reshape_3d(ctx, vis, cfg.model_dim, n_visual_tokens, 1);

            // 5. RoPE tensors (data pre-computed, passed via set_backend_tensor_data)
            auto* text_rope = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, text_seq_len);
            set_backend_tensor_data(text_rope, text_rope_data.data());

            auto* vis_rope = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_visual_tokens);
            set_backend_tensor_data(vis_rope, vis_rope_data.data());

            // 6. Text transformer blocks
            for (int i = 0; i < cfg.num_text_blocks; i++) {
                text = text_blocks_[i].forward(&fctx, text, t_emb, text_rope);
            }

            // 7. Visual transformer blocks
            for (int i = 0; i < cfg.num_visual_blocks; i++) {
                vis = visual_blocks_[i].forward(&fctx, vis, text, t_emb, vis_rope);
            }

            // 8. Output layer
            vis = ggml_reshape_2d(ctx, vis, cfg.model_dim, n_visual_tokens);
            vis = out_layer_.forward(&fctx, vis, t_emb);

            // Output: [out_visual_dim, n_visual_tokens]
            ggml_build_forward_expand(gf, vis);
            return gf;
        };

        return compute(build_graph, n_threads, true, output, out_ctx);
    }

private:
    // Embeddings
    Linear    time_in_{(int64_t)0, (int64_t)0};
    Linear    time_out_{(int64_t)0, (int64_t)0};
    Linear    text_proj_{(int64_t)0, (int64_t)0};
    LayerNorm text_norm_;
    Linear    pool_proj_{(int64_t)0, (int64_t)0};
    LayerNorm pool_norm_;
    Linear    visual_in_{(int64_t)0, (int64_t)0};

    // Blocks
    std::vector<TextBlock>   text_blocks_;
    std::vector<VisualBlock> visual_blocks_;

    // Output
    DiTOutLayer out_layer_;

};

#endif // KD_CROSSDIT_HPP
