#ifndef KD_PIPELINE_HPP
#define KD_PIPELINE_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "ggml.h"
#include "ggml-backend.h"

#include "kandinsky.h"
#include "util.hpp"
#include "rng.hpp"
#include "models/clip.hpp"
#include "models/qwen.hpp"
#include "models/qwen_tokenizer.hpp"
#include "models/crossdit.hpp"
#include "models/vae.hpp"
#include "models/vae3d.hpp"
#include "sampling/sampler.hpp"
#include "name_conversion.hpp"
#include "core/model_loader.hpp"

// ── Pipeline: orchestrates text encoding → denoising → decoding ──────

class KandinskyPipeline {
public:
    struct Config {
        int     width            = 1024;
        int     height           = 1024;
        int     num_frames       = 1;
        int     num_steps        = 50;
        float   guidance_scale   = 3.5f;
        float   scheduler_scale  = 3.0f;
        int64_t seed             = -1;
        int     n_threads        = 4;
        bool    flash_attn       = true;
        ggml_type wtype          = GGML_TYPE_F16;
        bool    text_cpu         = false; // Run Qwen+CLIP on CPU to save VRAM
        bool    vae3d_cpu        = false; // Run video decoder on CPU to save VRAM
        int     dit_gpu_layers   = -1;    // Keep last N DiT blocks on GPU (-1 = all)
        std::string vocab_dir;      // Optional: directory with tokenizer files
        std::string dit_filename = "dit.gguf";  // DiT GGUF filename
    };

    KandinskyPipeline() = default;
    ~KandinskyPipeline() { unload(); }

    bool load(const std::string& model_dir, Config cfg);
    void unload();

    // Generate image from text prompt
    bool txt2img(const std::string& prompt, const std::string& neg_prompt,
                 Config cfg, uint8_t** output_rgb, int* out_w, int* out_h);

    // Generate video frames from text prompt
    bool txt2vid(const std::string& prompt, const std::string& neg_prompt,
                 Config cfg, uint8_t** output_frames,
                 int* out_w, int* out_h, int* out_n_frames);

private:
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t text_backend_ = nullptr;
    ggml_backend_t vae_backend_ = nullptr;

    std::unique_ptr<QwenEncoder>   qwen_;
    std::unique_ptr<CLIPTextModel> clip_;
    std::unique_ptr<CrossDiT>      dit_;
    std::unique_ptr<AutoEncoderKL>  vae_;
    std::unique_ptr<HunyuanVideoVAE> vae3d_;

    QwenTokenizer  qwen_tokenizer_;
    CLIPTokenizer  clip_tokenizer_;

    std::shared_ptr<RNG> rng_;

    bool loaded_ = false;

    // Initialize backend (GPU > CPU fallback)
    ggml_backend_t init_backend();

    // Load model weights from directory
    bool load_dit(const std::string& dir, ggml_type wtype,
                  const std::string& dit_filename = "dit.gguf",
                  int dit_gpu_layers = -1);
    bool load_qwen(const std::string& dir, ggml_type wtype);
    bool load_clip(const std::string& dir, ggml_type wtype);
    bool load_vae(const std::string& dir, ggml_type wtype);
    bool load_vae3d(const std::string& dir, ggml_type wtype);
};

#endif // KD_PIPELINE_HPP
