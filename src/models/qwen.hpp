#ifndef KD_QWEN_HPP
#define KD_QWEN_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "core/module.hpp"
#include "core/executor.hpp"
#include "rope.hpp"

// ── Qwen2.5 Text Encoder for K5 ─────────────────────────────────────
// Architecture: 28 layers, hidden_dim=3584, heads=28, kv_heads=4,
//               head_dim=128, intermediate=18944, RoPE theta=1e6
// Output: last hidden state [seq_len, 3584], cropped system tokens

// ── Qwen Attention ───────────────────────────────────────────────────

class QwenAttention : public Module {
public:
    int64_t hidden_dim, num_heads, num_kv_heads, head_dim;

    QwenAttention() : hidden_dim(3584), num_heads(28), num_kv_heads(4), head_dim(128) {}
    QwenAttention(int64_t dim, int64_t heads, int64_t kv_heads, int64_t hd)
        : hidden_dim(dim), num_heads(heads), num_kv_heads(kv_heads), head_dim(hd) {
        submodules["q_proj"] = std::make_shared<Linear>(dim, heads * hd, true);
        submodules["k_proj"] = std::make_shared<Linear>(dim, kv_heads * hd, true);
        submodules["v_proj"] = std::make_shared<Linear>(dim, kv_heads * hd, true);
        submodules["o_proj"] = std::make_shared<Linear>(heads * hd, dim, false);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x, ggml_tensor* rope_freqs = nullptr) {
        auto* ctx = fctx->ctx;
        int64_t seq_len = x->ne[1];
        int64_t batch = x->ne[2];

        auto* q = std::static_pointer_cast<Linear>(submodules["q_proj"])->forward(fctx, x);
        auto* k = std::static_pointer_cast<Linear>(submodules["k_proj"])->forward(fctx, x);
        auto* v = std::static_pointer_cast<Linear>(submodules["v_proj"])->forward(fctx, x);

        // Reshape for multi-head: [batch, seq, heads*head_dim] -> [batch, seq, heads, head_dim]
        q = ggml_reshape_4d(ctx, q, head_dim, num_heads, seq_len, batch);
        k = ggml_reshape_4d(ctx, k, head_dim, num_kv_heads, seq_len, batch);
        v = ggml_reshape_4d(ctx, v, head_dim, num_kv_heads, seq_len, batch);

        // Apply RoPE to Q and K (precomputed cos/sin, same as DiT)
        if (rope_freqs) {
            q = rope::apply(ctx, q, rope_freqs);
            k = rope::apply(ctx, k, rope_freqs);
        }

        // Reshape back for attention: [batch, seq, heads*head_dim]
        q = ggml_reshape_3d(ctx, q, num_heads * head_dim, seq_len, batch);
        k = ggml_reshape_3d(ctx, k, num_kv_heads * head_dim, seq_len, batch);
        v = ggml_reshape_3d(ctx, v, num_kv_heads * head_dim, seq_len, batch);

        // Scaled dot-product attention with GQA
        auto* attn_out = ops::attention(ctx, fctx->backend, q, k, v, num_heads,
                                        nullptr, fctx->flash_attn, true);

        // Output projection
        auto* out = std::static_pointer_cast<Linear>(submodules["o_proj"])->forward(fctx, attn_out);
        return out;
    }
};

// ── Qwen MLP (SwiGLU) ───────────────────────────────────────────────

class QwenMLP : public Module {
public:
    QwenMLP() = default;
    QwenMLP(int64_t dim, int64_t intermediate) {
        submodules["gate_proj"] = std::make_shared<Linear>(dim, intermediate, false);
        submodules["up_proj"]   = std::make_shared<Linear>(dim, intermediate, false);
        submodules["down_proj"] = std::make_shared<Linear>(intermediate, dim, false);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;

        auto* gate = std::static_pointer_cast<Linear>(submodules["gate_proj"])->forward(fctx, x);
        auto* up   = std::static_pointer_cast<Linear>(submodules["up_proj"])->forward(fctx, x);

        // SwiGLU: silu(gate) * up
        gate = ggml_silu(ctx, gate);
        auto* hidden = ggml_mul(ctx, gate, up);

        return std::static_pointer_cast<Linear>(submodules["down_proj"])->forward(fctx, hidden);
    }
};

// ── Qwen Layer ───────────────────────────────────────────────────────

class QwenLayer : public Module {
public:
    QwenLayer() = default;
    QwenLayer(int64_t dim, int64_t heads, int64_t kv_heads, int64_t head_dim,
              int64_t intermediate, float rms_eps = 1e-6f) {
        submodules["input_layernorm"]     = std::make_shared<RMSNorm>(dim, rms_eps);
        submodules["self_attn"]           = std::make_shared<QwenAttention>(dim, heads, kv_heads, head_dim);
        submodules["post_attention_layernorm"] = std::make_shared<RMSNorm>(dim, rms_eps);
        submodules["mlp"]                 = std::make_shared<QwenMLP>(dim, intermediate);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x, ggml_tensor* rope_freqs = nullptr) {
        auto* ctx = fctx->ctx;

        // Self-attention with pre-norm
        auto* residual = x;
        x = std::static_pointer_cast<RMSNorm>(submodules["input_layernorm"])->forward(fctx, x);
        x = std::static_pointer_cast<QwenAttention>(submodules["self_attn"])->forward(fctx, x, rope_freqs);
        x = ggml_add(ctx, x, residual);

        // MLP with pre-norm
        residual = x;
        x = std::static_pointer_cast<RMSNorm>(submodules["post_attention_layernorm"])->forward(fctx, x);
        x = std::static_pointer_cast<QwenMLP>(submodules["mlp"])->forward(fctx, x);
        x = ggml_add(ctx, x, residual);

        return x;
    }
};

// ── Qwen Encoder (full model, no lm_head) ───────────────────────────

class QwenEncoder : public Executor {
public:
    int64_t hidden_dim;
    int     num_layers;

    QwenEncoder(ggml_backend_t backend,
                int64_t dim = 3584, int heads = 28, int kv_heads = 4,
                int64_t head_dim = 128, int64_t intermediate = 18944,
                int layers = 28, float rms_eps = 1e-6f)
        : Executor(backend), hidden_dim(dim), num_layers(layers) {
        // Token embeddings
        embed_ = Embedding(152064, dim);  // Qwen2.5-VL vocab size

        // Transformer layers
        for (int i = 0; i < layers; i++) {
            layers_.push_back(QwenLayer(dim, heads, kv_heads, head_dim, intermediate, rms_eps));
        }

        // Output norm (no lm_head for encoder-only use)
        output_norm_ = RMSNorm(dim, rms_eps);
    }

    std::string get_desc() override { return "QwenEncoder"; }

    bool init(ggml_type wtype = GGML_TYPE_F16) {
        auto* ctx = params_ctx();
        if (!ctx) {
            LOG_ERROR("QwenEncoder: params context is not initialized");
            return false;
        }
        embed_.init(ctx, wtype, "model.embed_tokens");
        for (int i = 0; i < num_layers; i++) {
            layers_[i].init(ctx, wtype, "model.layers." + std::to_string(i));
        }
        output_norm_.init(ctx, GGML_TYPE_F32, "model.norm");
        return alloc_params_buffer();
    }

    void collect_all_params(std::map<std::string, ggml_tensor*>& tensors) {
        embed_.collect_params(tensors, "model.embed_tokens");
        for (int i = 0; i < num_layers; i++) {
            layers_[i].collect_params(tensors, "model.layers." + std::to_string(i));
        }
        output_norm_.collect_params(tensors, "model.norm");
    }

    // Forward: token_ids → hidden_states [batch, seq_len, 3584]
    bool forward(ggml_tensor* input_ids, int n_threads,
                 ggml_tensor** hidden_states, ggml_context* out_ctx) {
        // Pre-compute RoPE data on CPU (must outlive the lambda)
        int seq_len = (int)input_ids->ne[0];
        int head_dim_val = 128;
        auto qwen_rope_data = rope::compute_text_rope(seq_len, head_dim_val, 1000000.0f);

        auto build_graph = [&]() -> ggml_cgraph* {
            auto* ctx = compute_ctx_;
            auto* gf = ggml_new_graph_custom(ctx, MAX_GRAPH_SIZE, false);

            ForwardContext fctx = get_forward_ctx();

            auto* ids = to_backend(input_ids);
            auto* x = embed_.forward(&fctx, ids);

            // Create RoPE tensor and upload data
            auto* rope_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim_val, seq_len);
            set_backend_tensor_data(rope_tensor, qwen_rope_data.data());

            // Run through all layers with RoPE
            for (int i = 0; i < num_layers; i++) {
                x = layers_[i].forward(&fctx, x, rope_tensor);
            }

            // Final norm
            x = output_norm_.forward(&fctx, x);

            ggml_build_forward_expand(gf, x);
            return gf;
        };

        return compute(build_graph, n_threads, true, hidden_states, out_ctx);
    }

private:
    Embedding              embed_;
    std::vector<QwenLayer> layers_;
    RMSNorm                output_norm_;
};

#endif // KD_QWEN_HPP
