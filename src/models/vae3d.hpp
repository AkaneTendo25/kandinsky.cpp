#ifndef KD_VAE3D_HPP
#define KD_VAE3D_HPP

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "core/module.hpp"
#include "core/executor.hpp"

// ── HunyuanVideo CausalVideoVAE (3D VAE decoder) ────────────────────
//
// Decoder path only (for generation).
// All intermediate data flows as std::vector<ggml_tensor*> (one per frame)
// where each tensor is [W, H, C, 1] — avoids 5D tensors entirely.
//
// Conv3d weights stored as 4D [OC*kD, IC, kH, kW] with kD suffix in name.
// At forward time, decomposed into kD separate 2D convolutions per output frame.
//
// Compression: 8x spatial, 4x temporal. scaling_factor = 0.476986

static constexpr float VAE3D_SCALING_FACTOR = 0.476986f;

// ── Frame vector helpers ─────────────────────────────────────────────

using FrameVec = std::vector<ggml_tensor*>;

static inline bool kd_can_repeat(ggml_tensor* dst, ggml_tensor* src) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        int64_t a = dst->ne[i] > 0 ? dst->ne[i] : 1;
        int64_t b = src->ne[i] > 0 ? src->ne[i] : 1;
        if (b == 0 || (a % b) != 0) {
            return false;
        }
    }
    return true;
}

static inline void kd_log_shape_mismatch(const char* op, const char* where,
                                         ggml_tensor* a, ggml_tensor* b) {
    LOG_ERROR("%s: %s shape mismatch a=[%lld,%lld,%lld,%lld] "
              "b=[%lld,%lld,%lld,%lld]",
              where, op,
              (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
              (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
}

static inline ggml_tensor* kd_checked_repeat(ggml_context* ctx, ggml_tensor* src,
                                             ggml_tensor* dst_shape, const char* where) {
    if (!kd_can_repeat(dst_shape, src)) {
        kd_log_shape_mismatch("repeat", where, dst_shape, src);
        std::abort();
    }
    return ggml_repeat(ctx, src, dst_shape);
}

static inline ggml_tensor* kd_checked_add(ggml_context* ctx, ggml_tensor* a,
                                          ggml_tensor* b, const char* where) {
    if (!kd_can_repeat(a, b)) {
        kd_log_shape_mismatch("add", where, a, b);
        std::abort();
    }
    return ggml_add(ctx, a, b);
}

static inline ggml_tensor* kd_checked_mul(ggml_context* ctx, ggml_tensor* a,
                                          ggml_tensor* b, const char* where) {
    if (!kd_can_repeat(a, b)) {
        kd_log_shape_mismatch("mul", where, a, b);
        std::abort();
    }
    return ggml_mul(ctx, a, b);
}

// ── CausalConv3d ─────────────────────────────────────────────────────
// Stores weight as 4D [kW, kH, IC, OC*kT] (ggml order).
// Forward: frame-by-frame temporal convolution with causal padding.

class CausalConv3d : public Module {
public:
    int64_t in_ch, out_ch;
    int kT, kH, kW;   // kernel sizes
    int sT, sH, sW;   // strides

    CausalConv3d() : in_ch(0), out_ch(0), kT(1), kH(1), kW(1), sT(1), sH(1), sW(1) {}

    CausalConv3d(int64_t in_channels, int64_t out_channels,
                 int kt, int kh, int kw,
                 int st = 1, int sh = 1, int sw = 1)
        : in_ch(in_channels), out_ch(out_channels),
          kT(kt), kH(kh), kW(kw), sT(st), sH(sh), sW(sw) {}

protected:
    void init_params(ggml_context* ctx, ggml_type wtype, const std::string& prefix) override {
        // Weight stored as 4D: [kW, kH, IC, OC*kT] in ggml order
        // Name always has .kd{kT} suffix to match GGUF naming
        std::string w_name = "weight.kd" + std::to_string(kT);

        params[w_name] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kW, kH, in_ch, out_ch * kT);
        params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);
    }

public:
    ggml_tensor* get_weight() {
        // Find the weight param (may have kd suffix)
        for (auto& [name, t] : params) {
            if (name.find("weight") != std::string::npos) return t;
        }
        return nullptr;
    }

    static ggml_tensor* replicate_pad_2d(ggml_context* ctx, ggml_tensor* x, int pW, int pH) {
        if (pW <= 0 && pH <= 0) return x;

        auto* out = x;

        if (pW > 0) {
            int64_t W = out->ne[0], H = out->ne[1], C = out->ne[2];
            auto* left = ggml_view_4d(ctx, out, 1, H, C, 1,
                                      out->nb[1], out->nb[2], out->nb[3], 0);
            auto* right = ggml_view_4d(ctx, out, 1, H, C, 1,
                                       out->nb[1], out->nb[2], out->nb[3],
                                       (W - 1) * out->nb[0]);
            auto* lshape = ggml_new_tensor_4d(ctx, out->type, pW, H, C, 1);
            auto* rshape = ggml_new_tensor_4d(ctx, out->type, pW, H, C, 1);
            auto* lpad = kd_checked_repeat(ctx, left, lshape, "CausalConv3d::replicate_pad_2d(left)");
            auto* rpad = kd_checked_repeat(ctx, right, rshape, "CausalConv3d::replicate_pad_2d(right)");
            out = ggml_concat(ctx, lpad, out, 0);
            out = ggml_concat(ctx, out, rpad, 0);
        }

        if (pH > 0) {
            int64_t W = out->ne[0], H = out->ne[1], C = out->ne[2];
            auto* top = ggml_view_4d(ctx, out, W, 1, C, 1,
                                     out->nb[1], out->nb[2], out->nb[3], 0);
            auto* bot = ggml_view_4d(ctx, out, W, 1, C, 1,
                                     out->nb[1], out->nb[2], out->nb[3],
                                     (H - 1) * out->nb[1]);
            auto* tshape = ggml_new_tensor_4d(ctx, out->type, W, pH, C, 1);
            auto* bshape = ggml_new_tensor_4d(ctx, out->type, W, pH, C, 1);
            auto* tpad = kd_checked_repeat(ctx, top, tshape, "CausalConv3d::replicate_pad_2d(top)");
            auto* bpad = kd_checked_repeat(ctx, bot, bshape, "CausalConv3d::replicate_pad_2d(bottom)");
            out = ggml_concat(ctx, tpad, out, 1);
            out = ggml_concat(ctx, out, bpad, 1);
        }

        return out;
    }

    // Forward: takes frames, returns frames
    FrameVec forward(ForwardContext* fctx, const FrameVec& frames) {
        auto* ctx = fctx->ctx;
        int n_frames = (int)frames.size();
        int pH = kH / 2;
        int pW = kW / 2;

        auto* weight = get_weight();
        auto* bias = params["bias"];
        auto* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, out_ch, 1);

        // Causal padding: prepend (kT-1) copies of the first frame
        std::vector<ggml_tensor*> padded;
        padded.reserve(n_frames + kT - 1);
        for (int i = 0; i < kT - 1; i++) {
            padded.push_back(frames[0]);
        }
        for (int i = 0; i < n_frames; i++) {
            padded.push_back(frames[i]);
        }

        // Compute output frames
        // Number of output frames depends on temporal stride
        int n_padded = (int)padded.size();
        int n_out = (n_padded - kT) / sT + 1;

        // Match HunyuanVideoCausalConv3d: spatial replicate padding before conv.
        FrameVec padded_spatial;
        padded_spatial.reserve(n_padded);
        for (int i = 0; i < n_padded; i++) {
            padded_spatial.push_back(replicate_pad_2d(ctx, padded[i], pW, pH));
        }

        FrameVec output;
        output.reserve(n_out);

        for (int t = 0; t < n_out; t++) {
            int t_start = t * sT;
            ggml_tensor* acc = nullptr;

            for (int k = 0; k < kT; k++) {
                // Extract weight slice for this temporal position: [kW, kH, IC, OC]
                // Stored layout is OC*kT (from converter flattening [OC, kT] as oc*kT + k),
                // so we need strided selection along dim3 with stride kT.
                auto* w_slice = ggml_view_4d(ctx, weight,
                    kW, kH, in_ch, out_ch,
                    weight->nb[1], weight->nb[2], weight->nb[3] * kT,
                    (int64_t)k * weight->nb[3]);
                w_slice = ggml_cont(ctx, w_slice);

                auto* frame_k = padded_spatial[t_start + k];
                auto* conv_out = ggml_conv_2d(ctx, w_slice, frame_k, sW, sH, 0, 0, 1, 1);

                if (acc == nullptr) {
                    acc = conv_out;
                } else {
                    acc = ggml_add(ctx, acc, conv_out);
                }
            }

            // Add bias
            acc = kd_checked_add(ctx, acc, bias_4d, "CausalConv3d::forward(bias)");
            output.push_back(acc);
        }

        return output;
    }
};

// ── GroupNorm3d ──────────────────────────────────────────────────────
// Applies GroupNorm across all frames by stacking them along the H dimension.
// Input: frames [W, H, C, 1] each → concatenate to [W, H*n_frames, C, 1] →
// apply GroupNorm → split back.

class GroupNorm3d : public Module {
public:
    int64_t num_channels;
    int num_groups;
    float eps;

    GroupNorm3d() : num_channels(0), num_groups(32), eps(1e-6f) {}
    GroupNorm3d(int groups, int64_t channels, float eps = 1e-6f)
        : num_channels(channels), num_groups(groups), eps(eps) {}

protected:
    void init_params(ggml_context* ctx, ggml_type, const std::string& prefix) override {
        params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

public:
    FrameVec forward(ForwardContext* fctx, const FrameVec& frames) {
        auto* ctx = fctx->ctx;
        int n_frames = (int)frames.size();

        if (n_frames == 1) {
            // Single frame — just apply regular group norm
            auto* w = ggml_reshape_4d(ctx, params["weight"], 1, 1, params["weight"]->ne[0], 1);
            auto* b = ggml_reshape_4d(ctx, params["bias"], 1, 1, params["bias"]->ne[0], 1);
            auto* x = ggml_group_norm(ctx, frames[0], num_groups, eps);
            x = kd_checked_mul(ctx, x, w, "GroupNorm3d::forward(single,weight)");
            x = kd_checked_add(ctx, x, b, "GroupNorm3d::forward(single,bias)");
            return {x};
        }

        // Stack frames along height: [W, H*n_frames, C, 1]
        int64_t W = frames[0]->ne[0];
        int64_t H = frames[0]->ne[1];
        int64_t C = frames[0]->ne[2];

        auto* stacked = ggml_concat(ctx, frames[0], frames[1], 1);
        for (int i = 2; i < n_frames; i++) {
            stacked = ggml_concat(ctx, stacked, frames[i], 1);
        }

        // Apply group norm
        auto* w = ggml_reshape_4d(ctx, params["weight"], 1, 1, num_channels, 1);
        auto* b = ggml_reshape_4d(ctx, params["bias"], 1, 1, num_channels, 1);
        auto* normed = ggml_group_norm(ctx, stacked, num_groups, eps);
        normed = kd_checked_mul(ctx, normed, w, "GroupNorm3d::forward(multi,weight)");
        normed = kd_checked_add(ctx, normed, b, "GroupNorm3d::forward(multi,bias)");

        // Split back into frames
        FrameVec output;
        output.reserve(n_frames);
        for (int i = 0; i < n_frames; i++) {
            auto* slice = ggml_view_4d(ctx, normed,
                W, H, C, 1,
                normed->nb[1], normed->nb[2], normed->nb[3],
                (int64_t)i * H * normed->nb[1]);
            output.push_back(ggml_cont(ctx, slice));
        }

        return output;
    }
};

// ── SiLU3d (per-frame activation) ───────────────────────────────────

inline FrameVec silu3d(ggml_context* ctx, const FrameVec& frames) {
    FrameVec out;
    out.reserve(frames.size());
    for (auto* f : frames) {
        out.push_back(ggml_silu(ctx, f));
    }
    return out;
}

// ── Add3d (per-frame element-wise add) ──────────────────────────────

inline FrameVec add3d(ggml_context* ctx, const FrameVec& a, const FrameVec& b) {
    FrameVec out;
    out.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        out.push_back(kd_checked_add(ctx, a[i], b[i], "add3d"));
    }
    return out;
}

// ── VAE3DResnetBlock ────────────────────────────────────────────────

class VAE3DResnetBlock : public Module {
public:
    int64_t in_ch, out_ch;

    VAE3DResnetBlock() : in_ch(0), out_ch(0) {}
    VAE3DResnetBlock(int64_t in_channels, int64_t out_channels)
        : in_ch(in_channels), out_ch(out_channels) {
        submodules["norm1"] = std::make_shared<GroupNorm3d>(32, in_ch);
        submodules["conv1"] = std::make_shared<CausalConv3d>(in_ch, out_ch, 3, 3, 3);
        submodules["norm2"] = std::make_shared<GroupNorm3d>(32, out_ch);
        submodules["conv2"] = std::make_shared<CausalConv3d>(out_ch, out_ch, 3, 3, 3);
        if (in_ch != out_ch) {
            submodules["nin_shortcut"] = std::make_shared<CausalConv3d>(in_ch, out_ch, 1, 1, 1);
        }
    }

    FrameVec forward(ForwardContext* fctx, const FrameVec& x) {
        auto* ctx = fctx->ctx;
        auto residual = x;

        auto h = std::static_pointer_cast<GroupNorm3d>(submodules["norm1"])->forward(fctx, x);
        h = silu3d(ctx, h);
        h = std::static_pointer_cast<CausalConv3d>(submodules["conv1"])->forward(fctx, h);

        h = std::static_pointer_cast<GroupNorm3d>(submodules["norm2"])->forward(fctx, h);
        h = silu3d(ctx, h);
        h = std::static_pointer_cast<CausalConv3d>(submodules["conv2"])->forward(fctx, h);

        if (in_ch != out_ch) {
            residual = std::static_pointer_cast<CausalConv3d>(submodules["nin_shortcut"])->forward(fctx, residual);
        }

        return add3d(ctx, h, residual);
    }
};

// ── VAE3DAttnBlock ──────────────────────────────────────────────────
// Self-attention across all frames with causal masking.
// All frames' spatial tokens are concatenated: [N=1, F*H*W, C]

class VAE3DAttnBlock : public Module {
public:
    int64_t channels;

    VAE3DAttnBlock() : channels(0) {}
    VAE3DAttnBlock(int64_t ch) : channels(ch) {
        submodules["norm"]     = std::make_shared<GroupNorm3d>(32, ch);
        submodules["q"]        = std::make_shared<Linear>(ch, ch);
        submodules["k"]        = std::make_shared<Linear>(ch, ch);
        submodules["v"]        = std::make_shared<Linear>(ch, ch);
        submodules["proj_out"] = std::make_shared<Linear>(ch, ch);
    }

    FrameVec forward(ForwardContext* fctx, const FrameVec& frames) {
        auto* ctx = fctx->ctx;
        int n_frames = (int)frames.size();

        auto residual = frames;

        // Apply group norm across all frames
        auto normed = std::static_pointer_cast<GroupNorm3d>(submodules["norm"])->forward(fctx, frames);

        // Compute Q, K, V per frame, then concatenate
        int64_t W = normed[0]->ne[0];
        int64_t H = normed[0]->ne[1];
        int64_t C = normed[0]->ne[2];
        int64_t HW = H * W;
        int64_t total_tokens = n_frames * HW;

        // Apply linear projections per frame
        auto q_lin = std::static_pointer_cast<Linear>(submodules["q"]);
        auto k_lin = std::static_pointer_cast<Linear>(submodules["k"]);
        auto v_lin = std::static_pointer_cast<Linear>(submodules["v"]);

        // Build concatenated Q, K, V: [C, total_tokens, 1]
        std::vector<ggml_tensor*> q_frames, k_frames, v_frames;
        for (int i = 0; i < n_frames; i++) {
            // Reshape frame [W, H, C, 1] -> [C, H*W] for linear.
            // ggml_permute() takes destination axes for each source dim, so
            // (1,2,0,3) maps source [W,H,C,1] -> [C,W,H,1].
            // After cont+reshape, each of the HW tokens holds its C-channel vector.
            auto* flat = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, normed[i], 1, 2, 0, 3)), C, HW);

            auto* qf = q_lin->forward(fctx, flat);  // [C, H*W]
            auto* kf = k_lin->forward(fctx, flat);
            auto* vf = v_lin->forward(fctx, flat);

            // Reshape to [C, HW, 1] for attention
            qf = ggml_reshape_3d(ctx, qf, C, HW, 1);
            kf = ggml_reshape_3d(ctx, kf, C, HW, 1);
            vf = ggml_reshape_3d(ctx, vf, C, HW, 1);

            q_frames.push_back(qf);
            k_frames.push_back(kf);
            v_frames.push_back(vf);
        }

        // Concatenate along sequence dim (ne[1])
        auto* q_all = q_frames[0];
        auto* k_all = k_frames[0];
        auto* v_all = v_frames[0];
        for (int i = 1; i < n_frames; i++) {
            q_all = ggml_concat(ctx, q_all, q_frames[i], 1);
            k_all = ggml_concat(ctx, k_all, k_frames[i], 1);
            v_all = ggml_concat(ctx, v_all, v_frames[i], 1);
        }

        // Build causal mask: frame i can attend to frames 0..i
        // mask shape: [total_tokens, total_tokens, 1] (additive mask)
        auto* mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, total_tokens, total_tokens, 1);
        ggml_set_name(mask, "vae3d_causal_mask");

        // Fill mask data: -inf where attending to future frames
        std::vector<float> mask_data(total_tokens * total_tokens, 0.0f);
        for (int qi = 0; qi < n_frames; qi++) {
            for (int ki = 0; ki < n_frames; ki++) {
                if (ki > qi) {
                    // Future frame — mask out
                    for (int64_t qp = qi * HW; qp < (qi + 1) * HW; qp++) {
                        for (int64_t kp = ki * HW; kp < (ki + 1) * HW; kp++) {
                            mask_data[qp * total_tokens + kp] = -1e9f;
                        }
                    }
                }
            }
        }
        fctx->backend; // ensure we use the backend tensor data mechanism
        // We need to pass mask data via set_backend_tensor_data on the executor
        // Since we don't have direct access here, we'll allocate it differently
        // Actually, for CPU tensors in the compute context with no_alloc=true,
        // we need to use the backend_tensor_data mechanism.
        // Store the mask data as a member so it lives long enough.
        mask_data_ = std::move(mask_data);
        if (fctx->set_tensor_data) {
            fctx->set_tensor_data(mask, mask_data_.data());
        }

        // Apply attention: q, k, v all [C, total_tokens, 1]
        auto* attn = ops::attention(ctx, fctx->backend, q_all, k_all, v_all,
                                     1, mask, false);  // 1 head for VAE, no flash

        // attn: [C, total_tokens, 1]
        // Split back into per-frame and reshape
        auto proj = std::static_pointer_cast<Linear>(submodules["proj_out"]);

        FrameVec output;
        output.reserve(n_frames);
        for (int i = 0; i < n_frames; i++) {
            // Extract frame i: [C, HW, 1]
            auto* frame_attn = ggml_view_3d(ctx, attn,
                C, HW, 1,
                attn->nb[1], attn->nb[2],
                (int64_t)i * HW * attn->nb[1]);
            frame_attn = ggml_cont(ctx, frame_attn);

            // Apply proj_out: [C, HW] -> [C, HW]
            auto* proj_in = ggml_reshape_2d(ctx, frame_attn, C, HW);
            auto* proj_out = proj->forward(fctx, proj_in);

            // Reshape back to [W, H, C, 1]
            // Data is [C, HW] with spatial order h*W+w (H outer, W inner).
            // reshape to [C, W, H, 1] then permute(2,0,1,3) → [W, H, C, 1].
            proj_out = ggml_reshape_4d(ctx, proj_out, C, W, H, 1);
            proj_out = ggml_cont(ctx, ggml_permute(ctx, proj_out, 2, 0, 1, 3));

            // Residual connection
            output.push_back(kd_checked_add(ctx, proj_out, residual[i], "VAE3DAttnBlock::forward(residual)"));
        }

        return output;
    }

    // Provide access to mask data for set_backend_tensor_data
    const float* get_mask_data() const { return mask_data_.data(); }
    size_t get_mask_size() const { return mask_data_.size() * sizeof(float); }

private:
    std::vector<float> mask_data_;
};

// ── VAE3DMidBlock ───────────────────────────────────────────────────

class VAE3DMidBlock : public Module {
public:
    VAE3DMidBlock() = default;
    VAE3DMidBlock(int64_t ch) {
        submodules["block_1"] = std::make_shared<VAE3DResnetBlock>(ch, ch);
        submodules["attn_1"]  = std::make_shared<VAE3DAttnBlock>(ch);
        submodules["block_2"] = std::make_shared<VAE3DResnetBlock>(ch, ch);
    }

    FrameVec forward(ForwardContext* fctx, const FrameVec& x) {
        auto h = std::static_pointer_cast<VAE3DResnetBlock>(submodules["block_1"])->forward(fctx, x);
        h = std::static_pointer_cast<VAE3DAttnBlock>(submodules["attn_1"])->forward(fctx, h);
        h = std::static_pointer_cast<VAE3DResnetBlock>(submodules["block_2"])->forward(fctx, h);
        return h;
    }
};

// ── VAE3DUpsample ───────────────────────────────────────────────────
// Spatial upsample: 2x nearest neighbor + CausalConv3d(3,3,3)
// Temporal upsample (if enabled): duplicate each frame (except first)

class VAE3DUpsample : public Module {
public:
    bool temporal;  // Whether to also upsample temporally (2x)

    VAE3DUpsample() : temporal(false) {}
    VAE3DUpsample(int64_t ch, bool temporal_upsample = true)
        : temporal(temporal_upsample) {
        submodules["conv"] = std::make_shared<CausalConv3d>(ch, ch, 3, 3, 3);
    }

    FrameVec forward(ForwardContext* fctx, const FrameVec& frames) {
        auto* ctx = fctx->ctx;

        FrameVec upsampled;

        if (temporal) {
            // Temporal upsample: first frame stays single, rest are duplicated
            // This gives 2*(n-1)+1 = 2n-1 output frames from n input frames
            upsampled.reserve(frames.size() * 2 - 1);
            upsampled.push_back(ggml_upscale(ctx, frames[0], 2, GGML_SCALE_MODE_NEAREST));
            for (size_t i = 1; i < frames.size(); i++) {
                auto* up = ggml_upscale(ctx, frames[i], 2, GGML_SCALE_MODE_NEAREST);
                upsampled.push_back(up);  // original position
                upsampled.push_back(up);  // duplicated for temporal upsample
            }
        } else {
            // Spatial only — no temporal change
            upsampled.reserve(frames.size());
            for (auto* f : frames) {
                upsampled.push_back(ggml_upscale(ctx, f, 2, GGML_SCALE_MODE_NEAREST));
            }
        }

        // Apply CausalConv3d
        return std::static_pointer_cast<CausalConv3d>(submodules["conv"])->forward(fctx, upsampled);
    }
};

// ── VAE3DUpBlock ────────────────────────────────────────────────────

class VAE3DUpBlock : public Module {
public:
    bool has_upsample;

    VAE3DUpBlock() : has_upsample(false) {}
    VAE3DUpBlock(int64_t in_ch, int64_t out_ch, bool upsample = false, bool temporal_upsample = true)
        : has_upsample(upsample) {
        submodules["block.0"] = std::make_shared<VAE3DResnetBlock>(in_ch, out_ch);
        submodules["block.1"] = std::make_shared<VAE3DResnetBlock>(out_ch, out_ch);
        submodules["block.2"] = std::make_shared<VAE3DResnetBlock>(out_ch, out_ch);
        if (upsample) {
            submodules["upsample"] = std::make_shared<VAE3DUpsample>(out_ch, temporal_upsample);
        }
    }

    FrameVec forward(ForwardContext* fctx, const FrameVec& x) {
        auto h = std::static_pointer_cast<VAE3DResnetBlock>(submodules["block.0"])->forward(fctx, x);
        h = std::static_pointer_cast<VAE3DResnetBlock>(submodules["block.1"])->forward(fctx, h);
        h = std::static_pointer_cast<VAE3DResnetBlock>(submodules["block.2"])->forward(fctx, h);
        if (has_upsample) {
            h = std::static_pointer_cast<VAE3DUpsample>(submodules["upsample"])->forward(fctx, h);
        }
        return h;
    }
};

// ── VAE3DDecoder ────────────────────────────────────────────────────
// Full decoder: conv_in → mid → up_blocks → norm_out → silu → conv_out

class VAE3DDecoder : public Module {
public:
    static constexpr int BASE_CH = 128;
    static constexpr int LATENT_CH = 16;

    VAE3DDecoder() {
        int ch = BASE_CH * 4;  // 512, starting from deepest

        submodules["conv_in"] = std::make_shared<CausalConv3d>(LATENT_CH, ch, 3, 3, 3);

        // Mid block
        submodules["mid"] = std::make_shared<VAE3DMidBlock>(ch);

        // Up blocks (in decoder order: 0→3)
        // Temporal upsample placement (time_compression_ratio=4 → 2 temporal upsamples):
        //   condition: i >= len(blocks)-1 - num_time_upsample_layers && !is_final
        //   i.e. i >= 4-1-2 = 1, so blocks 1 and 2 get temporal upsample.
        // Block 0: 512→512, upsample(spatial only, scale=(1,2,2))
        submodules["up.0"] = std::make_shared<VAE3DUpBlock>(ch, ch, true, false);
        // Block 1: 512→512, upsample(temporal+spatial, scale=(2,2,2))
        submodules["up.1"] = std::make_shared<VAE3DUpBlock>(ch, ch, true, true);
        // Block 2: 512→256, upsample(temporal+spatial, scale=(2,2,2))
        submodules["up.2"] = std::make_shared<VAE3DUpBlock>(ch, BASE_CH * 2, true, true);
        ch = BASE_CH * 2;  // 256
        // Block 3: 256→128, no upsample
        submodules["up.3"] = std::make_shared<VAE3DUpBlock>(ch, BASE_CH, false, false);
        ch = BASE_CH;  // 128

        submodules["norm_out"] = std::make_shared<GroupNorm3d>(32, ch);
        submodules["conv_out"] = std::make_shared<CausalConv3d>(ch, 3, 3, 3, 3);
    }

    FrameVec forward(ForwardContext* fctx, const FrameVec& x) {
        auto* ctx = fctx->ctx;

        auto h = std::static_pointer_cast<CausalConv3d>(submodules["conv_in"])->forward(fctx, x);

        h = std::static_pointer_cast<VAE3DMidBlock>(submodules["mid"])->forward(fctx, h);

        h = std::static_pointer_cast<VAE3DUpBlock>(submodules["up.0"])->forward(fctx, h);
        h = std::static_pointer_cast<VAE3DUpBlock>(submodules["up.1"])->forward(fctx, h);
        h = std::static_pointer_cast<VAE3DUpBlock>(submodules["up.2"])->forward(fctx, h);
        h = std::static_pointer_cast<VAE3DUpBlock>(submodules["up.3"])->forward(fctx, h);

        h = std::static_pointer_cast<GroupNorm3d>(submodules["norm_out"])->forward(fctx, h);
        h = silu3d(ctx, h);
        h = std::static_pointer_cast<CausalConv3d>(submodules["conv_out"])->forward(fctx, h);

        return h;
    }
};

// ── HunyuanVideoVAE (Executor wrapper) ──────────────────────────────

class HunyuanVideoVAE : public Executor {
public:
    HunyuanVideoVAE(ggml_backend_t backend) : Executor(backend) {}

    std::string get_desc() override { return "HunyuanVideoVAE"; }

    static constexpr int TEMPORAL_COMPRESSION_RATIO     = 4;
    static constexpr int TILE_SAMPLE_MIN_NUM_FRAMES     = 16;
    static constexpr int TILE_SAMPLE_STRIDE_NUM_FRAMES  = 12;

    bool init(ggml_type wtype = GGML_TYPE_F16) {
        auto* ctx = params_ctx();
        if (!ctx) {
            LOG_ERROR("HunyuanVideoVAE: params context is not initialized");
            return false;
        }
        decoder_.init(ctx, wtype, "decoder");
        post_quant_conv_.init(ctx, wtype, "post_quant_conv");
        return alloc_params_buffer();
    }

    void collect_all_params(std::map<std::string, ggml_tensor*>& tensors) {
        decoder_.collect_params(tensors, "decoder");
        post_quant_conv_.collect_params(tensors, "post_quant_conv");
    }

    // Decode latent to video frames
    // latent: [lat_w, lat_h, 16, n_lat_frames] (ggml order)
    // Returns: vector of RGB frame tensors, each [W, H, 3, 1]
    bool decode(ggml_tensor* latent, int n_lat_frames, int n_threads,
                std::vector<ggml_tensor*>& output_frames, ggml_context* out_ctx) {
        const int tile_latent_min_num_frames =
            TILE_SAMPLE_MIN_NUM_FRAMES / TEMPORAL_COMPRESSION_RATIO;

        // Match upstream HunyuanVideo behavior: long videos are decoded in
        // overlapping temporal tiles instead of a single monolithic pass.
        if (n_lat_frames > tile_latent_min_num_frames + 1) {
            return decode_temporal_tiled(latent, n_lat_frames, n_threads, output_frames, out_ctx);
        }

        std::vector<std::vector<float>> host_frames;
        int64_t out_w = 0, out_h = 0, out_ch = 0;
        if (!decode_full_to_host_buffers(latent, n_lat_frames, n_threads,
                                         host_frames, out_w, out_h, out_ch)) {
            return false;
        }

        copy_host_frames_to_output(host_frames, out_w, out_h, out_ch, output_frames, out_ctx);
        return !output_frames.empty();
    }

private:
    bool decode_full_to_host_buffers(ggml_tensor* latent, int n_lat_frames, int n_threads,
                                     std::vector<std::vector<float>>& host_frames,
                                     int64_t& out_w, int64_t& out_h, int64_t& out_ch) {
        // First, unscale and split latent into per-frame tensors on CPU.
        const int64_t lat_w = latent->ne[0];
        const int64_t lat_h = latent->ne[1];
        const int64_t lat_ch = latent->ne[2];

        frame_data_.clear();
        frame_data_.resize(n_lat_frames);

        float* lat_data = (float*)latent->data;
        const size_t frame_size = (size_t)lat_w * (size_t)lat_h * (size_t)lat_ch;
        for (int f = 0; f < n_lat_frames; f++) {
            frame_data_[f].resize(frame_size);
            float* src = lat_data + (size_t)f * frame_size;
            for (size_t i = 0; i < frame_size; i++) {
                frame_data_[f][i] = src[i] / VAE3D_SCALING_FACTOR;
            }
        }

        lat_w_ = lat_w;
        lat_h_ = lat_h;
        lat_ch_ = lat_ch;
        n_lat_frames_ = n_lat_frames;

        auto build_graph = [&]() -> ggml_cgraph* {
            auto* ctx = compute_ctx_;
            auto* gf = ggml_new_graph_custom(ctx, MAX_GRAPH_SIZE, false);
            ForwardContext fctx = get_forward_ctx();

            FrameVec frames;
            frames.reserve(n_lat_frames_);
            for (int f = 0; f < n_lat_frames_; f++) {
                auto* ft = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, lat_w_, lat_h_, lat_ch_, 1);
                set_backend_tensor_data(ft, frame_data_[f].data());
                frames.push_back(ft);
            }

            frames = post_quant_conv_.forward(&fctx, frames);
            frames = decoder_.forward(&fctx, frames);

            if (frames.size() == 1) {
                ggml_build_forward_expand(gf, frames[0]);
            } else {
                auto* result = frames[0];
                for (size_t i = 1; i < frames.size(); i++) {
                    result = ggml_concat(ctx, result, frames[i], 3);
                }
                ggml_build_forward_expand(gf, result);
            }

            for (auto* f : frames) {
                ggml_build_forward_expand(gf, f);
            }

            return gf;
        };

        const int64_t expected_out_w = lat_w * 8;
        const int64_t expected_out_h = lat_h * 8;
        const int64_t expected_out_ch = 3;
        const int expected_out_frames = (n_lat_frames - 1) * TEMPORAL_COMPRESSION_RATIO + 1;
        const size_t out_bytes =
            (size_t)expected_out_w * (size_t)expected_out_h * (size_t)expected_out_ch *
            (size_t)expected_out_frames * sizeof(float);
        ggml_init_params op = {
            out_bytes + ggml_tensor_overhead() * 8,
            nullptr,
            false
        };
        ggml_context* out_tmp_ctx = ggml_init(op);
        if (!out_tmp_ctx) {
            LOG_ERROR("VAE3D decode: failed to allocate temporary output context");
            return false;
        }

        ggml_tensor* packed_output = nullptr;
        bool ok = compute(build_graph, n_threads, true, &packed_output, out_tmp_ctx);
        if (!ok) {
            ggml_free(out_tmp_ctx);
            LOG_ERROR("VAE3D decode compute failed");
            return false;
        }

        if (!packed_output) {
            ggml_free(out_tmp_ctx);
            LOG_ERROR("VAE3D decode produced no output");
            return false;
        }

        out_w = packed_output->ne[0];
        out_h = packed_output->ne[1];
        out_ch = packed_output->ne[2];
        const int n_out_frames = (int)packed_output->ne[3];
        const size_t frame_floats = (size_t)out_w * (size_t)out_h * (size_t)out_ch;
        float* out_data = (float*)packed_output->data;

        host_frames.clear();
        host_frames.reserve(n_out_frames);
        for (int f = 0; f < n_out_frames; f++) {
            const float* src = out_data + (size_t)f * frame_floats;
            host_frames.emplace_back(src, src + frame_floats);
        }

        ggml_free(out_tmp_ctx);
        return true;
    }

    static void blend_temporal(const std::vector<std::vector<float>>& prev,
                               std::vector<std::vector<float>>& curr,
                               int blend_extent) {
        int be = std::min(blend_extent, (int)prev.size());
        be = std::min(be, (int)curr.size());
        if (be <= 0) return;

        for (int t = 0; t < be; t++) {
            const std::vector<float>& a = prev[prev.size() - be + t];
            std::vector<float>& b = curr[t];
            const float wa = 1.0f - (float)t / (float)be;
            const float wb = (float)t / (float)be;
            const size_t n = std::min(a.size(), b.size());
            for (size_t i = 0; i < n; i++) {
                b[i] = a[i] * wa + b[i] * wb;
            }
        }
    }

    static void copy_host_frames_to_output(const std::vector<std::vector<float>>& host_frames,
                                           int64_t out_w, int64_t out_h, int64_t out_ch,
                                           std::vector<ggml_tensor*>& output_frames,
                                           ggml_context* out_ctx) {
        output_frames.clear();
        output_frames.reserve(host_frames.size());
        const size_t frame_bytes =
            (size_t)out_w * (size_t)out_h * (size_t)out_ch * sizeof(float);

        for (const auto& frame_data : host_frames) {
            auto* frame = ggml_new_tensor_4d(out_ctx, GGML_TYPE_F32, out_w, out_h, out_ch, 1);
            memcpy(frame->data, frame_data.data(), std::min(frame_bytes, frame_data.size() * sizeof(float)));
            output_frames.push_back(frame);
        }
    }

    bool decode_temporal_tiled(ggml_tensor* latent, int n_lat_frames, int n_threads,
                               std::vector<ggml_tensor*>& output_frames, ggml_context* out_ctx) {
        const int tile_latent_min_num_frames =
            TILE_SAMPLE_MIN_NUM_FRAMES / TEMPORAL_COMPRESSION_RATIO;
        const int tile_latent_stride_num_frames =
            TILE_SAMPLE_STRIDE_NUM_FRAMES / TEMPORAL_COMPRESSION_RATIO;
        const int expected_out_frames =
            (n_lat_frames - 1) * TEMPORAL_COMPRESSION_RATIO + 1;
        const int tile_frames = tile_latent_min_num_frames + 1;
        const int max_tile_start = std::max(0, n_lat_frames - tile_frames);

        const int64_t lat_w = latent->ne[0];
        const int64_t lat_h = latent->ne[1];
        const int64_t lat_ch = latent->ne[2];
        const size_t latent_frame_floats = (size_t)lat_w * (size_t)lat_h * (size_t)lat_ch;
        float* latent_data = (float*)latent->data;

        LOG_INFO("VAE3D: using temporal tiled decode for %d latent frames", n_lat_frames);

        struct DecodedTile {
            int sample_start = 0;
            std::vector<std::vector<float>> frames;
        };

        std::vector<int> tile_starts;
        for (int start = 0; start <= max_tile_start; start += tile_latent_stride_num_frames) {
            tile_starts.push_back(start);
        }
        if (tile_starts.empty() || tile_starts.back() != max_tile_start) {
            tile_starts.push_back(max_tile_start);
        }

        std::vector<DecodedTile> decoded_tiles;
        int64_t out_w = 0, out_h = 0, out_ch = 0;

        for (int start : tile_starts) {
            ggml_init_params tp = {
                latent_frame_floats * (size_t)tile_frames * sizeof(float) +
                    ggml_tensor_overhead() * 4,
                nullptr,
                false
            };
            ggml_context* tile_ctx = ggml_init(tp);
            if (!tile_ctx) {
                LOG_ERROR("VAE3D tiled decode: failed to allocate latent tile context");
                return false;
            }

            auto* tile_latent = ggml_new_tensor_4d(tile_ctx, GGML_TYPE_F32,
                                                   lat_w, lat_h, lat_ch, tile_frames);
            memcpy(tile_latent->data,
                   latent_data + (size_t)start * latent_frame_floats,
                   latent_frame_floats * (size_t)tile_frames * sizeof(float));

            std::vector<std::vector<float>> decoded_tile;
            int64_t tile_out_w = 0, tile_out_h = 0, tile_out_ch = 0;
            bool ok = decode_full_to_host_buffers(tile_latent, tile_frames, n_threads,
                                                  decoded_tile, tile_out_w, tile_out_h, tile_out_ch);
            ggml_free(tile_ctx);
            if (!ok) return false;

            out_w = tile_out_w;
            out_h = tile_out_h;
            out_ch = tile_out_ch;
            decoded_tiles.push_back(DecodedTile{
                start * TEMPORAL_COMPRESSION_RATIO,
                std::move(decoded_tile)
            });
        }

        std::vector<std::vector<float>> result_frames;
        for (size_t tile_idx = 0; tile_idx < decoded_tiles.size(); ++tile_idx) {
            auto& tile = decoded_tiles[tile_idx];
            if (result_frames.empty()) {
                result_frames = tile.frames;
                continue;
            }

            const int overlap = std::max(0, (int)result_frames.size() - tile.sample_start);
            const int blend_extent = std::min(overlap, (int)tile.frames.size());
            for (int t = 0; t < blend_extent; ++t) {
                std::vector<float>& dst = result_frames[tile.sample_start + t];
                std::vector<float>& src = tile.frames[t];
                const float wb = (float)(t + 1) / (float)(blend_extent + 1);
                const float wa = 1.0f - wb;
                const size_t n = std::min(dst.size(), src.size());
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = dst[i] * wa + src[i] * wb;
                }
            }

            if (blend_extent < (int)tile.frames.size()) {
                result_frames.insert(result_frames.end(),
                                     tile.frames.begin() + blend_extent,
                                     tile.frames.end());
            }
        }

        if ((int)result_frames.size() > expected_out_frames) {
            result_frames.resize(expected_out_frames);
        }

        copy_host_frames_to_output(result_frames, out_w, out_h, out_ch, output_frames, out_ctx);
        return !output_frames.empty();
    }

    VAE3DDecoder decoder_;
    CausalConv3d post_quant_conv_{16, 16, 1, 1, 1};

    // Temporary storage for frame data during decode
    std::vector<std::vector<float>> frame_data_;
    int64_t lat_w_ = 0, lat_h_ = 0, lat_ch_ = 0;
    int n_lat_frames_ = 0;
};

#endif // KD_VAE3D_HPP
