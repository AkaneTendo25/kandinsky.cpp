#ifndef KD_VAE_HPP
#define KD_VAE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "core/module.hpp"
#include "core/executor.hpp"

// ── AutoencoderKL (FLUX-style 2D VAE) ───────────────────────────────
// 16 latent channels, 8x spatial downsample
// Scaling factor: 0.476986
// Used for K5 image mode

static constexpr float VAE_SCALING_FACTOR = 0.476986f;

// ── VAE ResNet Block ─────────────────────────────────────────────────

class VAEResnetBlock : public Module {
public:
    int64_t in_ch, out_ch;

    VAEResnetBlock() : in_ch(0), out_ch(0) {}
    VAEResnetBlock(int64_t in_channels, int64_t out_channels)
        : in_ch(in_channels), out_ch(out_channels) {
        submodules["norm1"] = std::make_shared<GroupNorm>(32, in_ch);
        submodules["conv1"] = std::make_shared<Conv2d>(in_ch, out_ch, std::make_pair(3,3),
                                                       std::make_pair(1,1), std::make_pair(1,1));
        submodules["norm2"] = std::make_shared<GroupNorm>(32, out_ch);
        submodules["conv2"] = std::make_shared<Conv2d>(out_ch, out_ch, std::make_pair(3,3),
                                                       std::make_pair(1,1), std::make_pair(1,1));
        if (in_ch != out_ch) {
            submodules["nin_shortcut"] = std::make_shared<Conv2d>(in_ch, out_ch, std::make_pair(1,1));
        }
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;
        auto* residual = x;

        x = std::static_pointer_cast<GroupNorm>(submodules["norm1"])->forward(fctx, x);
        x = ggml_silu(ctx, x);
        x = std::static_pointer_cast<Conv2d>(submodules["conv1"])->forward(fctx, x);

        x = std::static_pointer_cast<GroupNorm>(submodules["norm2"])->forward(fctx, x);
        x = ggml_silu(ctx, x);
        x = std::static_pointer_cast<Conv2d>(submodules["conv2"])->forward(fctx, x);

        if (in_ch != out_ch) {
            residual = std::static_pointer_cast<Conv2d>(submodules["nin_shortcut"])->forward(fctx, residual);
        }

        return ggml_add(ctx, x, residual);
    }
};

// ── VAE Attention Block ──────────────────────────────────────────────

class VAEAttnBlock : public Module {
public:
    int64_t channels;

    VAEAttnBlock() : channels(0) {}
    VAEAttnBlock(int64_t ch) : channels(ch) {
        submodules["norm"]     = std::make_shared<GroupNorm>(32, ch);
        submodules["q"]        = std::make_shared<Conv2d>(ch, ch, std::make_pair(1,1));
        submodules["k"]        = std::make_shared<Conv2d>(ch, ch, std::make_pair(1,1));
        submodules["v"]        = std::make_shared<Conv2d>(ch, ch, std::make_pair(1,1));
        submodules["proj_out"] = std::make_shared<Conv2d>(ch, ch, std::make_pair(1,1));
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;
        auto* residual = x;

        x = std::static_pointer_cast<GroupNorm>(submodules["norm"])->forward(fctx, x);

        auto* q = std::static_pointer_cast<Conv2d>(submodules["q"])->forward(fctx, x);
        auto* k = std::static_pointer_cast<Conv2d>(submodules["k"])->forward(fctx, x);
        auto* v = std::static_pointer_cast<Conv2d>(submodules["v"])->forward(fctx, x);

        // Reshape: [N, C, H, W] -> [N, H*W, C] for attention
        int64_t N = q->ne[3], C = q->ne[2], H = q->ne[1], W = q->ne[0];
        q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)), C, H * W, N);
        k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3)), C, H * W, N);
        v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)), C, H * W, N);

        auto* attn = ops::attention(ctx, fctx->backend, q, k, v, 1, nullptr, fctx->flash_attn);

        // Reshape back: [N, H*W, C] -> [N, C, H, W]
        attn = ggml_reshape_4d(ctx, attn, C, W, H, N);
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 2, 0, 1, 3));

        attn = std::static_pointer_cast<Conv2d>(submodules["proj_out"])->forward(fctx, attn);
        return ggml_add(ctx, attn, residual);
    }
};

// ── VAE Downsample ───────────────────────────────────────────────────

class VAEDownsample : public Module {
public:
    VAEDownsample() = default;
    VAEDownsample(int64_t ch) {
        submodules["conv"] = std::make_shared<Conv2d>(ch, ch, std::make_pair(3,3),
                                                      std::make_pair(2,2), std::make_pair(1,1));
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        return std::static_pointer_cast<Conv2d>(submodules["conv"])->forward(fctx, x);
    }
};

// ── VAE Upsample ─────────────────────────────────────────────────────

class VAEUpsample : public Module {
public:
    VAEUpsample() = default;
    VAEUpsample(int64_t ch) {
        submodules["conv"] = std::make_shared<Conv2d>(ch, ch, std::make_pair(3,3),
                                                      std::make_pair(1,1), std::make_pair(1,1));
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;
        // Nearest neighbor upsample 2x
        x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
        return std::static_pointer_cast<Conv2d>(submodules["conv"])->forward(fctx, x);
    }
};

// ── VAE Encoder ──────────────────────────────────────────────────────

class VAEEncoder : public Module {
public:
    // Channel multipliers: [1, 2, 4, 4] (relative to base=128)
    static constexpr int BASE_CH = 128;
    static constexpr int LATENT_CH = 16;

    VAEEncoder() {
        int ch_mults[] = {1, 2, 4, 4};
        int ch_in = 3;

        submodules["conv_in"] = std::make_shared<Conv2d>(ch_in, BASE_CH,
            std::make_pair(3,3), std::make_pair(1,1), std::make_pair(1,1));

        int ch = BASE_CH;
        for (int level = 0; level < 4; level++) {
            int ch_out = BASE_CH * ch_mults[level];
            std::string prefix = "down." + std::to_string(level);
            submodules[prefix + ".block.0"] = std::make_shared<VAEResnetBlock>(ch, ch_out);
            submodules[prefix + ".block.1"] = std::make_shared<VAEResnetBlock>(ch_out, ch_out);
            if (level < 3) {
                submodules[prefix + ".downsample"] = std::make_shared<VAEDownsample>(ch_out);
            }
            ch = ch_out;
        }

        // Mid block
        submodules["mid.block_1"] = std::make_shared<VAEResnetBlock>(ch, ch);
        submodules["mid.attn_1"]  = std::make_shared<VAEAttnBlock>(ch);
        submodules["mid.block_2"] = std::make_shared<VAEResnetBlock>(ch, ch);

        // Output
        submodules["norm_out"] = std::make_shared<GroupNorm>(32, ch);
        submodules["conv_out"] = std::make_shared<Conv2d>(ch, LATENT_CH * 2,
            std::make_pair(3,3), std::make_pair(1,1), std::make_pair(1,1));
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;

        x = std::static_pointer_cast<Conv2d>(submodules["conv_in"])->forward(fctx, x);

        for (int level = 0; level < 4; level++) {
            std::string prefix = "down." + std::to_string(level);
            x = std::static_pointer_cast<VAEResnetBlock>(submodules[prefix + ".block.0"])->forward(fctx, x);
            x = std::static_pointer_cast<VAEResnetBlock>(submodules[prefix + ".block.1"])->forward(fctx, x);
            if (level < 3) {
                x = std::static_pointer_cast<VAEDownsample>(submodules[prefix + ".downsample"])->forward(fctx, x);
            }
        }

        x = std::static_pointer_cast<VAEResnetBlock>(submodules["mid.block_1"])->forward(fctx, x);
        x = std::static_pointer_cast<VAEAttnBlock>(submodules["mid.attn_1"])->forward(fctx, x);
        x = std::static_pointer_cast<VAEResnetBlock>(submodules["mid.block_2"])->forward(fctx, x);

        x = std::static_pointer_cast<GroupNorm>(submodules["norm_out"])->forward(fctx, x);
        x = ggml_silu(ctx, x);
        x = std::static_pointer_cast<Conv2d>(submodules["conv_out"])->forward(fctx, x);

        // Take mean of posterior (first LATENT_CH channels)
        x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], LATENT_CH, x->ne[3],
                          x->nb[1], x->nb[2], x->nb[3], 0);
        return x;
    }
};

// ── VAE Decoder ──────────────────────────────────────────────────────

class VAEDecoder : public Module {
public:
    static constexpr int BASE_CH = 128;
    static constexpr int LATENT_CH = 16;

    VAEDecoder() {
        int ch_mults[] = {1, 2, 4, 4};
        int ch = BASE_CH * ch_mults[3]; // Start from deepest

        submodules["conv_in"] = std::make_shared<Conv2d>(LATENT_CH, ch,
            std::make_pair(3,3), std::make_pair(1,1), std::make_pair(1,1));

        // Mid block
        submodules["mid.block_1"] = std::make_shared<VAEResnetBlock>(ch, ch);
        submodules["mid.attn_1"]  = std::make_shared<VAEAttnBlock>(ch);
        submodules["mid.block_2"] = std::make_shared<VAEResnetBlock>(ch, ch);

        // Up blocks (reverse order)
        for (int level = 3; level >= 0; level--) {
            int ch_out = BASE_CH * ch_mults[level];
            std::string prefix = "up." + std::to_string(level);
            submodules[prefix + ".block.0"] = std::make_shared<VAEResnetBlock>(ch, ch_out);
            submodules[prefix + ".block.1"] = std::make_shared<VAEResnetBlock>(ch_out, ch_out);
            submodules[prefix + ".block.2"] = std::make_shared<VAEResnetBlock>(ch_out, ch_out);
            if (level > 0) {
                submodules[prefix + ".upsample"] = std::make_shared<VAEUpsample>(ch_out);
            }
            ch = ch_out;
        }

        submodules["norm_out"] = std::make_shared<GroupNorm>(32, ch);
        submodules["conv_out"] = std::make_shared<Conv2d>(ch, 3,
            std::make_pair(3,3), std::make_pair(1,1), std::make_pair(1,1));
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;

        x = std::static_pointer_cast<Conv2d>(submodules["conv_in"])->forward(fctx, x);

        x = std::static_pointer_cast<VAEResnetBlock>(submodules["mid.block_1"])->forward(fctx, x);
        x = std::static_pointer_cast<VAEAttnBlock>(submodules["mid.attn_1"])->forward(fctx, x);
        x = std::static_pointer_cast<VAEResnetBlock>(submodules["mid.block_2"])->forward(fctx, x);

        for (int level = 3; level >= 0; level--) {
            std::string prefix = "up." + std::to_string(level);
            x = std::static_pointer_cast<VAEResnetBlock>(submodules[prefix + ".block.0"])->forward(fctx, x);
            x = std::static_pointer_cast<VAEResnetBlock>(submodules[prefix + ".block.1"])->forward(fctx, x);
            x = std::static_pointer_cast<VAEResnetBlock>(submodules[prefix + ".block.2"])->forward(fctx, x);
            if (level > 0) {
                x = std::static_pointer_cast<VAEUpsample>(submodules[prefix + ".upsample"])->forward(fctx, x);
            }
        }

        x = std::static_pointer_cast<GroupNorm>(submodules["norm_out"])->forward(fctx, x);
        x = ggml_silu(ctx, x);
        x = std::static_pointer_cast<Conv2d>(submodules["conv_out"])->forward(fctx, x);
        return x;
    }
};

// ── AutoencoderKL (full VAE model) ───────────────────────────────────

class AutoEncoderKL : public Executor {
public:
    AutoEncoderKL(ggml_backend_t backend) : Executor(backend) {}

    std::string get_desc() override { return "AutoEncoderKL"; }

    bool init(ggml_type wtype = GGML_TYPE_F16) {
        auto* ctx = params_ctx();
        if (!ctx) {
            LOG_ERROR("AutoEncoderKL: params context is not initialized");
            return false;
        }
        encoder_.init(ctx, wtype, "encoder");
        decoder_.init(ctx, wtype, "decoder");
        return alloc_params_buffer();
    }

    void collect_all_params(std::map<std::string, ggml_tensor*>& tensors) {
        encoder_.collect_params(tensors, "encoder");
        decoder_.collect_params(tensors, "decoder");
    }

    // Encode image to latent
    bool encode(ggml_tensor* image, int n_threads,
                ggml_tensor** latent, ggml_context* out_ctx) {
        auto build_graph = [&]() -> ggml_cgraph* {
            auto* ctx = compute_ctx_;
            auto* gf = ggml_new_graph_custom(ctx, 20480, false);
            ForwardContext fctx = get_forward_ctx();

            auto* x = to_backend(image);
            x = encoder_.forward(&fctx, x);
            x = ggml_scale(ctx, x, VAE_SCALING_FACTOR);

            ggml_build_forward_expand(gf, x);
            return gf;
        };
        return compute(build_graph, n_threads, true, latent, out_ctx);
    }

    // Decode latent to image
    bool decode(ggml_tensor* latent, int n_threads,
                ggml_tensor** image, ggml_context* out_ctx) {
        auto build_graph = [&]() -> ggml_cgraph* {
            auto* ctx = compute_ctx_;
            auto* gf = ggml_new_graph_custom(ctx, 20480, false);
            ForwardContext fctx = get_forward_ctx();

            auto* x = to_backend(latent);
            x = ggml_scale(ctx, x, 1.0f / VAE_SCALING_FACTOR);
            x = decoder_.forward(&fctx, x);

            ggml_build_forward_expand(gf, x);
            return gf;
        };
        return compute(build_graph, n_threads, true, image, out_ctx);
    }

private:
    VAEEncoder encoder_;
    VAEDecoder decoder_;
};

#endif // KD_VAE_HPP
