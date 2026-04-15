#include "pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

#include "gguf.h"

namespace {

constexpr const char* kTokenizerVocabJsonKey = "tokenizer.vocab_json";
constexpr const char* kTokenizerMergesTxtKey = "tokenizer.merges_txt";

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

bool path_has_any(const std::string& text, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (text.find(needle) != std::string::npos) return true;
    }
    return false;
}

bool is_temp_model_artifact(const std::string& path_lc) {
    return ends_with(path_lc, ".tmp") || path_has_any(path_lc, {".data.tmp", ".part", ".partial"});
}

ggml_type infer_wtype_from_path(const std::string& path) {
    const std::string path_lc = to_lower_copy(path);
    if (path_has_any(path_lc, {".q4_0.gguf", ".q4_0.safetensors", ".safetensors.q4_0.gguf"})) return GGML_TYPE_Q4_0;
    if (path_has_any(path_lc, {".q4_1.gguf", ".q4_1.safetensors", ".safetensors.q4_1.gguf"})) return GGML_TYPE_Q4_1;
    if (path_has_any(path_lc, {".q5_0.gguf", ".q5_0.safetensors", ".safetensors.q5_0.gguf"})) return GGML_TYPE_Q5_0;
    if (path_has_any(path_lc, {".q5_1.gguf", ".q5_1.safetensors", ".safetensors.q5_1.gguf"})) return GGML_TYPE_Q5_1;
    if (path_has_any(path_lc, {".q8_0.gguf", ".q8_0.safetensors", ".safetensors.q8_0.gguf"})) return GGML_TYPE_Q8_0;
    if (path_has_any(path_lc, {".f16.gguf", ".f16.safetensors", ".safetensors.f16.gguf"})) return GGML_TYPE_F16;
    if (path_has_any(path_lc, {".f32.gguf", ".f32.safetensors", ".safetensors.f32.gguf"})) return GGML_TYPE_F32;
    return GGML_TYPE_COUNT;
}

int score_type_match(ggml_type desired, ggml_type candidate) {
    if (candidate == GGML_TYPE_COUNT) return 0;
    if (candidate == desired) return 500;

    const bool desired_quantized =
        desired == GGML_TYPE_Q8_0 || desired == GGML_TYPE_Q4_0 ||
        desired == GGML_TYPE_Q4_1 || desired == GGML_TYPE_Q5_0 ||
        desired == GGML_TYPE_Q5_1;
    const bool candidate_quantized =
        candidate == GGML_TYPE_Q8_0 || candidate == GGML_TYPE_Q4_0 ||
        candidate == GGML_TYPE_Q4_1 || candidate == GGML_TYPE_Q5_0 ||
        candidate == GGML_TYPE_Q5_1;

    if (!desired_quantized && candidate_quantized) return 200;
    if (desired_quantized && !candidate_quantized) return 100;
    return 50;
}

std::string pick_model_candidate(const std::string& dir,
                                 ggml_type desired_wtype,
                                 std::initializer_list<const char*> required_terms,
                                 std::initializer_list<const char*> optional_terms,
                                 std::initializer_list<const char*> exact_names = {}) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(dir)) return {};

    std::vector<std::string> required_lc;
    for (const char* term : required_terms) required_lc.push_back(to_lower_copy(term));

    std::vector<std::string> optional_lc;
    for (const char* term : optional_terms) optional_lc.push_back(to_lower_copy(term));

    std::vector<std::string> exact_lc;
    for (const char* name : exact_names) exact_lc.push_back(to_lower_copy(name));

    int best_score = std::numeric_limits<int>::min();
    std::string best_path;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        const std::string fname = entry.path().filename().string();
        const std::string fname_lc = to_lower_copy(fname);
        if (!(ends_with(fname_lc, ".gguf") || ends_with(fname_lc, ".safetensors"))) continue;
        if (is_temp_model_artifact(fname_lc)) continue;

        bool matches_required = required_lc.empty();
        if (!required_lc.empty()) {
            matches_required = true;
            for (const auto& term : required_lc) {
                if (fname_lc.find(term) == std::string::npos) {
                    matches_required = false;
                    break;
                }
            }
        }
        if (!matches_required) continue;

        int score = 0;
        if (std::find(exact_lc.begin(), exact_lc.end(), fname_lc) != exact_lc.end()) score += 1000;
        for (const auto& term : optional_lc) {
            if (fname_lc.find(term) != std::string::npos) score += 100;
        }

        score += score_type_match(desired_wtype, infer_wtype_from_path(fname_lc));
        if (ends_with(fname_lc, ".gguf")) score += 25;
        if (!path_has_any(fname_lc, {"debug"})) score += 10;
        score -= (int)fname_lc.size();

        if (score > best_score) {
            best_score = score;
            best_path = entry.path().string();
        }
    }

    return best_path;
}

std::string find_qwen_model_path(const std::string& dir, ggml_type wtype) {
    std::string qwen_path;

    for (auto& name : {"qwen.gguf", "text_encoder.gguf", "text_encoder.q8_0.gguf", "text_encoder.q4_0.gguf"}) {
        auto p = path_join(dir, name);
        if (file_exists(p)) { qwen_path = p; break; }
    }

    if (qwen_path.empty()) {
        qwen_path = pick_model_candidate(dir, wtype, {"qwen"}, {"text_encoder"});
    }

    if (qwen_path.empty()) {
        for (auto& subdir : {"text_encoder", "k5_modules/text_encoder"}) {
            auto d = path_join(dir, subdir);
            for (auto& name : {"model-00001-of-00005.safetensors",
                               "model-00001-of-00004.safetensors",
                               "model.safetensors"}) {
                auto pp = path_join(d, name);
                if (file_exists(pp)) { qwen_path = d; break; }
            }
            if (!qwen_path.empty()) break;
        }
    }

    return qwen_path;
}

std::string find_clip_model_path(const std::string& dir, ggml_type wtype) {
    std::string clip_path;

    for (auto& name : {"clip.gguf", "text_encoder2.gguf", "text_encoder2.q8_0.gguf", "text_encoder2.q4_0.gguf"}) {
        auto p = path_join(dir, name);
        if (file_exists(p)) { clip_path = p; break; }
    }

    if (clip_path.empty()) {
        clip_path = pick_model_candidate(dir, wtype, {"clip"}, {"text_encoder2"});
    }

    if (clip_path.empty()) {
        for (auto& subdir : {"text_encoder_2", "text_encoder2",
                              "k5_modules/text_encoder2"}) {
            auto d = path_join(dir, subdir);
            auto pp = path_join(d, "model.safetensors");
            if (file_exists(pp)) { clip_path = pp; break; }
        }
    }

    return clip_path;
}

bool read_embedded_tokenizer(const std::string& gguf_path,
                             std::string* vocab_json,
                             std::string* merges_txt) {
    if (gguf_path.empty() || !ends_with(to_lower_copy(gguf_path), ".gguf")) {
        return false;
    }

    ggml_context* meta_ctx = nullptr;
    struct gguf_init_params gguf_params = { true, &meta_ctx };
    auto* gguf = gguf_init_from_file(gguf_path.c_str(), gguf_params);
    if (!gguf) {
        LOG_WARN("Failed to open %s for tokenizer metadata", gguf_path.c_str());
        return false;
    }

    const int64_t vocab_id = gguf_find_key(gguf, kTokenizerVocabJsonKey);
    const int64_t merges_id = gguf_find_key(gguf, kTokenizerMergesTxtKey);
    const bool found = vocab_id >= 0 && merges_id >= 0;
    if (found) {
        *vocab_json = gguf_get_val_str(gguf, vocab_id);
        *merges_txt = gguf_get_val_str(gguf, merges_id);
    }

    if (meta_ctx) ggml_free(meta_ctx);
    gguf_free(gguf);
    return found;
}

} // namespace

// ── Backend initialization ───────────────────────────────────────────

ggml_backend_t KandinskyPipeline::init_backend() {
    ggml_backend_t backend = nullptr;

#ifdef KD_USE_CUDA
    LOG_INFO("Attempting CUDA backend...");
    backend = ggml_backend_cuda_init(0);
    if (backend) { LOG_INFO("Using CUDA backend"); return backend; }
#endif

#ifdef KD_USE_VULKAN
    LOG_INFO("Attempting Vulkan backend...");
    backend = ggml_backend_vk_init(0);
    if (backend) { LOG_INFO("Using Vulkan backend"); return backend; }
#endif

#ifdef KD_USE_METAL
    LOG_INFO("Attempting Metal backend...");
    backend = ggml_backend_metal_init();
    if (backend) { LOG_INFO("Using Metal backend"); return backend; }
#endif

    LOG_INFO("Using CPU backend");
    backend = ggml_backend_cpu_init();
    return backend;
}

// ── Load individual models ───────────────────────────────────────────

bool KandinskyPipeline::load_dit(const std::string& dir, ggml_type wtype,
                                  const std::string& dit_filename,
                                  int dit_gpu_layers) {
    // Look for DiT GGUF or safetensors
    std::string dit_path;
    bool auto_selected = false;
    namespace fs = std::filesystem;

    // 1. If caller passed an absolute path, use it directly.
    if (!dit_filename.empty() && fs::path(dit_filename).is_absolute() && file_exists(dit_filename)) {
        dit_path = dit_filename;
    }

    // 2. Check for configured filename under model dir.
    if (dit_path.empty()) {
        auto p = path_join(dir, dit_filename);
        if (file_exists(p)) { dit_path = p; }
    }

    // 3. Auto-discover K5 DiT weights. This repo commonly ships files like
    // kandinsky5pro_t2v_sft_5s.q4_0.gguf rather than a literal dit.gguf.
    if (dit_path.empty()) {
        dit_path = pick_model_candidate(
            dir,
            wtype,
            {},
            {"kandinsky5", "diffusion", "dit", "transformer"},
            {"dit.gguf", "dit.safetensors", "diffusion_pytorch_model.safetensors"});
        auto_selected = !dit_path.empty();
    }

    // 4. Check subdirectories
    if (dit_path.empty()) {
        for (auto& subdir : {"dit", "transformer", "model"}) {
            auto d = path_join(dir, subdir);
            for (auto& ext : {"model.safetensors", "diffusion_pytorch_model.safetensors"}) {
                auto p = path_join(d, ext);
                if (file_exists(p)) { dit_path = p; break; }
            }
            if (!dit_path.empty()) break;
        }
        auto_selected = !dit_path.empty();
    }

    if (dit_path.empty()) {
        LOG_ERROR("DiT model not found in %s", dir.c_str());
        return false;
    }

    LOG_INFO("Loading DiT from %s", dit_path.c_str());

    ggml_type effective_wtype = wtype;
    ggml_type inferred_wtype = infer_wtype_from_path(dit_path);
    if (auto_selected && inferred_wtype != GGML_TYPE_COUNT && inferred_wtype != wtype) {
        LOG_WARN("Auto-selected DiT %s is %s; using matching tensor type instead of requested %s",
                 dit_path.c_str(), ggml_type_name(inferred_wtype), ggml_type_name(wtype));
        effective_wtype = inferred_wtype;
    }

    // Detect config from GGUF metadata or from filename
    CrossDiTConfig dit_cfg;

    if (ends_with(dit_path, ".gguf")) {
        // Read config from GGUF metadata
        ggml_context* meta_ctx = nullptr;
        struct gguf_init_params gguf_params = { true, &meta_ctx };
        auto* gguf = gguf_init_from_file(dit_path.c_str(), gguf_params);
        if (gguf) {
            auto read_u32 = [&](const char* key, uint32_t def) -> uint32_t {
                int64_t id = gguf_find_key(gguf, key);
                return id >= 0 ? gguf_get_val_u32(gguf, id) : def;
            };
            dit_cfg.model_dim = read_u32("dit.model_dim", (uint32_t)dit_cfg.model_dim);
            dit_cfg.ff_dim = read_u32("dit.ff_dim", (uint32_t)dit_cfg.ff_dim);
            dit_cfg.time_dim = read_u32("dit.time_dim", (uint32_t)dit_cfg.time_dim);
            dit_cfg.in_visual_dim = read_u32("dit.in_visual_dim", (uint32_t)dit_cfg.in_visual_dim);
            dit_cfg.out_visual_dim = read_u32("dit.out_visual_dim", (uint32_t)dit_cfg.out_visual_dim);
            dit_cfg.num_text_blocks = read_u32("dit.num_text_blocks", (uint32_t)dit_cfg.num_text_blocks);
            dit_cfg.num_visual_blocks = read_u32("dit.num_visual_blocks", (uint32_t)dit_cfg.num_visual_blocks);
            if (meta_ctx) ggml_free(meta_ctx);
            gguf_free(gguf);
        }
        // Restore known variant-specific rope settings when GGUF lacks explicit metadata.
        if (dit_cfg.model_dim == 1792 && dit_cfg.num_visual_blocks == 32) {
            dit_cfg.axes_dims = {16, 24, 24};
            dit_cfg.scale_factor = {1.0f, 2.0f, 2.0f};
        } else if (dit_cfg.model_dim == 2560 && dit_cfg.num_visual_blocks == 50) {
            dit_cfg.axes_dims = {32, 48, 48};
            dit_cfg.scale_factor = {1.0f, 1.0f, 1.0f};
        } else if (dit_cfg.model_dim == 4096 && dit_cfg.num_visual_blocks == 60) {
            dit_cfg.axes_dims = {32, 48, 48};
            dit_cfg.scale_factor = {1.0f, 2.0f, 2.0f};
        }
    } else {
        // Guess from filename
        if (dit_path.find("pro") != std::string::npos) {
            dit_cfg = CrossDiTConfig::pro_video();
        } else if (dit_path.find("lite") != std::string::npos) {
            if (dit_path.find("t2i") != std::string::npos || dit_path.find("i2i") != std::string::npos) {
                dit_cfg = CrossDiTConfig::lite_image();
            } else {
                dit_cfg = CrossDiTConfig::lite_video();
            }
        }
    }

    LOG_INFO("DiT config: model_dim=%lld, ff=%lld, time=%lld, blocks=%dt+%dv, vis_in=%lld, vis_out=%lld, axes=[%d,%d,%d], scale=[%.1f,%.1f,%.1f]",
             (long long)dit_cfg.model_dim, (long long)dit_cfg.ff_dim, (long long)dit_cfg.time_dim,
             dit_cfg.num_text_blocks, dit_cfg.num_visual_blocks,
             (long long)dit_cfg.in_visual_dim, (long long)dit_cfg.out_visual_dim,
             dit_cfg.axes_dims[0], dit_cfg.axes_dims[1], dit_cfg.axes_dims[2],
             dit_cfg.scale_factor[0], dit_cfg.scale_factor[1], dit_cfg.scale_factor[2]);

    dit_cfg.gpu_layers = dit_gpu_layers;
    if (dit_gpu_layers >= 0) {
        LOG_INFO("DiT layer split requested: gpu_layers=%d", dit_gpu_layers);
    }

    dit_ = std::make_unique<CrossDiT>(backend_, dit_cfg);
    if (!dit_->init(effective_wtype)) {
        LOG_ERROR("Failed to initialize DiT parameter storage");
        return false;
    }

    ModelLoader loader;
    if (!loader.init_from_file(dit_path)) return false;

    std::map<std::string, ggml_tensor*> tensors;
    dit_->collect_all_params(tensors);

    // For safetensors, names match directly (no conversion needed)
    // For GGUF, names also stored as-is by our converter
    if (!loader.load_tensors(tensors)) {
        LOG_ERROR("DiT tensor loading failed — model weights may be corrupt");
        return false;
    }

    // Diagnostic: count matched vs total expected tensors
    int expected = (int)tensors.size();
    int in_file = 0;
    for (auto& [name, _] : loader.tensor_storage()) in_file++;
    int matched = 0;
    for (auto& [name, t] : tensors) {
        if (loader.tensor_storage().find(name)) matched++;
    }
    LOG_INFO("DiT tensors: %d in model, %d in file, %d matched",
             expected, in_file, matched);
    if (matched < expected) {
        LOG_WARN("DiT: %d tensors NOT found in file (missing weights!)", expected - matched);
        for (auto& [name, t] : tensors) {
            if (!loader.tensor_storage().find(name)) {
                LOG_WARN("  MISSING: %s", name.c_str());
            }
        }
    }

    return true;
}

bool KandinskyPipeline::load_qwen(const std::string& dir, ggml_type wtype) {
    std::string qwen_path = find_qwen_model_path(dir, wtype);

    if (qwen_path.empty()) {
        LOG_ERROR("Qwen model not found in %s", dir.c_str());
        return false;
    }

    LOG_INFO("Loading Qwen from %s", qwen_path.c_str());
    qwen_ = std::make_unique<QwenEncoder>(text_backend_ ? text_backend_ : backend_);
    if (!qwen_->init(wtype)) {
        LOG_ERROR("Failed to initialize Qwen parameter storage");
        return false;
    }

    // Load all shards from directory, or single file
    ModelLoader loader;
    namespace fs = std::filesystem;
    if (fs::is_directory(qwen_path)) {
        bool found_shard = false;
        for (auto& entry : fs::directory_iterator(qwen_path)) {
            auto fname = entry.path().filename().string();
            if (ends_with(fname, ".safetensors") && fname.find("model") != std::string::npos) {
                if (!loader.init_from_file(entry.path().string())) return false;
                found_shard = true;
            }
        }
        if (!found_shard) {
            LOG_ERROR("Qwen shard directory contains no model*.safetensors files: %s", qwen_path.c_str());
            return false;
        }
    } else {
        if (!loader.init_from_file(qwen_path)) return false;
    }

    std::map<std::string, ggml_tensor*> tensors;
    qwen_->collect_all_params(tensors);
    return loader.load_tensors(tensors);
}

bool KandinskyPipeline::load_clip(const std::string& dir, ggml_type wtype) {
    std::string clip_path = find_clip_model_path(dir, wtype);

    if (clip_path.empty()) {
        LOG_ERROR("CLIP model not found in %s", dir.c_str());
        return false;
    }

    LOG_INFO("Loading CLIP from %s", clip_path.c_str());
    clip_ = std::make_unique<CLIPTextModel>(text_backend_ ? text_backend_ : backend_);
    if (!clip_->init(wtype)) {
        LOG_ERROR("Failed to initialize CLIP parameter storage");
        return false;
    }

    ModelLoader loader;
    if (!loader.init_from_file(clip_path)) return false;

    std::map<std::string, ggml_tensor*> tensors;
    clip_->collect_all_params(tensors);
    return loader.load_tensors(tensors);
}

bool KandinskyPipeline::load_vae(const std::string& dir, ggml_type wtype) {
    std::string vae_path;

    auto p = path_join(dir, "vae.gguf");
    if (file_exists(p)) { vae_path = p; }

    if (vae_path.empty()) {
        for (auto& subdir : {"vae", "k5_modules/hunyuan/vae",
                              "flux/vae"}) {
            auto d = path_join(dir, subdir);
            for (auto& name : {"diffusion_pytorch_model.safetensors", "model.safetensors"}) {
                auto pp = path_join(d, name);
                if (file_exists(pp)) { vae_path = pp; break; }
            }
            if (!vae_path.empty()) break;
        }
    }

    if (vae_path.empty()) {
        LOG_ERROR("VAE model not found in %s", dir.c_str());
        return false;
    }

    LOG_INFO("Loading VAE from %s", vae_path.c_str());

    // Probe GGUF first — ggml crashes on 5D tensors (e.g. HunyuanVideo 3D VAE)
    if (ends_with(vae_path, ".gguf")) {
        FILE* f = fopen(vae_path.c_str(), "rb");
        if (!f) { LOG_ERROR("Cannot open %s", vae_path.c_str()); return false; }
        // Quick check: just try opening — the actual validation happens in init_from_file
        fclose(f);
    }

    ModelLoader loader;
    if (!loader.init_from_file(vae_path)) {
        LOG_WARN("VAE GGUF loading failed (may contain unsupported 5D tensors)");
        return false;
    }

    vae_ = std::make_unique<AutoEncoderKL>(backend_);
    if (!vae_->init(wtype)) {
        LOG_ERROR("Failed to initialize VAE parameter storage");
        return false;
    }

    std::map<std::string, ggml_tensor*> tensors;
    vae_->collect_all_params(tensors);
    return loader.load_tensors(tensors);
}

bool KandinskyPipeline::load_vae3d(const std::string& dir, ggml_type wtype) {
    std::string vae3d_path;

    for (auto& name : {"vae3d.gguf", "hunyuan_vae3d.gguf", "hunyuan_vae3d.q8_0.gguf", "hunyuan_vae3d.q4_0.gguf"}) {
        auto p = path_join(dir, name);
        if (file_exists(p)) { vae3d_path = p; break; }
    }

    if (vae3d_path.empty()) {
        vae3d_path = pick_model_candidate(dir, wtype, {"vae3d"}, {"hunyuan", "vae"});
    }

    if (vae3d_path.empty()) {
        LOG_INFO("VAE3D model not found in %s (looking for vae3d.gguf)", dir.c_str());
        return false;
    }

    LOG_INFO("Loading VAE3D from %s", vae3d_path.c_str());

    ModelLoader loader;
    if (!loader.init_from_file(vae3d_path)) {
        LOG_WARN("VAE3D GGUF loading failed");
        return false;
    }

    vae3d_ = std::make_unique<HunyuanVideoVAE>(vae_backend_ ? vae_backend_ : backend_);
    if (!vae3d_->init(wtype)) {
        LOG_ERROR("Failed to initialize VAE3D parameter storage");
        return false;
    }

    std::map<std::string, ggml_tensor*> tensors;
    vae3d_->collect_all_params(tensors);
    return loader.load_tensors(tensors);
}

// ── Load all models ──────────────────────────────────────────────────

bool KandinskyPipeline::load(const std::string& model_dir, Config cfg) {
    LOG_INFO("Loading Kandinsky 5 pipeline from %s", model_dir.c_str());

    backend_ = init_backend();
    if (!backend_) {
        LOG_ERROR("Failed to initialize backend");
        return false;
    }

    text_backend_ = backend_;
    vae_backend_ = backend_;
    if (cfg.text_cpu && !ggml_backend_is_cpu(backend_)) {
        text_backend_ = ggml_backend_cpu_init();
        if (text_backend_) {
            LOG_INFO("Text encoder backend: CPU (Qwen+CLIP offloaded from VRAM)");
        } else {
            LOG_WARN("Failed to init CPU backend for text encoder; using runtime backend");
            text_backend_ = backend_;
        }
    }
    if (cfg.vae3d_cpu && !ggml_backend_is_cpu(backend_)) {
        vae_backend_ = ggml_backend_cpu_init();
        if (vae_backend_) {
            LOG_INFO("VAE3D backend: CPU (decoder offloaded from VRAM)");
        } else {
            LOG_WARN("Failed to init CPU backend for VAE3D; using runtime backend");
            vae_backend_ = backend_;
        }
    }

    // Quantizing DiT too aggressively is sometimes acceptable, but quantizing
    // text encoders / decoder often collapses conditioning and produces noise.
    auto is_quantized = [](ggml_type t) {
        return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
               t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1;
    };
    ggml_type dit_wtype = cfg.wtype;
    ggml_type shared_wtype = is_quantized(cfg.wtype) ? GGML_TYPE_F16 : cfg.wtype;

    // Load tokenizers — search in model_dir, vocab_dir, and common K5 paths
    std::vector<std::string> search_dirs = { model_dir };
    if (!cfg.vocab_dir.empty()) search_dirs.push_back(cfg.vocab_dir);

    // CLIP tokenizer (prefer vocab.json + merges.txt from text_encoder2)
    bool clip_tok_ok = false;
    for (auto& dir : search_dirs) {
        for (auto& sub : {"", "text_encoder_2", "text_encoder2",
                           "k5_modules/text_encoder2"}) {
            auto d = path_join(dir, sub);
            auto vocab = path_join(d, "vocab.json");
            auto merges = path_join(d, "merges.txt");
            if (file_exists(vocab) && file_exists(merges)) {
                clip_tok_ok = clip_tokenizer_.load(vocab, merges);
                if (clip_tok_ok) {
                    LOG_INFO("CLIP tokenizer loaded from %s", d.c_str());
                    break;
                }
            }
        }
        if (clip_tok_ok) break;
    }
    if (!clip_tok_ok) {
        std::string clip_vocab_json;
        std::string clip_merges_txt;
        const std::string clip_path = find_clip_model_path(model_dir, shared_wtype);
        if (read_embedded_tokenizer(clip_path, &clip_vocab_json, &clip_merges_txt)) {
            clip_tok_ok = clip_tokenizer_.load_from_memory(clip_vocab_json, clip_merges_txt);
            if (clip_tok_ok) {
                LOG_INFO("CLIP tokenizer loaded from GGUF metadata in %s", clip_path.c_str());
            }
        }
    }
    if (!clip_tok_ok) {
        LOG_ERROR("CLIP tokenizer not found in model_dir/vocab_dir or embedded GGUF metadata");
        return false;
    }

    // Qwen tokenizer (vocab.json + merges.txt from text_encoder)
    bool qwen_tok_ok = false;
    for (auto& dir : search_dirs) {
        for (auto& sub : {"", "text_encoder",
                           "k5_modules/text_encoder"}) {
            auto d = path_join(dir, sub);
            auto vocab = path_join(d, "vocab.json");
            auto merges = path_join(d, "merges.txt");
            if (file_exists(vocab) && file_exists(merges)) {
                qwen_tok_ok = qwen_tokenizer_.load(vocab, merges);
                if (qwen_tok_ok) {
                    LOG_INFO("Qwen tokenizer loaded from %s", d.c_str());
                    break;
                }
            }
        }
        if (qwen_tok_ok) break;
    }
    if (!qwen_tok_ok) {
        std::string qwen_vocab_json;
        std::string qwen_merges_txt;
        const std::string qwen_path = find_qwen_model_path(model_dir, shared_wtype);
        if (read_embedded_tokenizer(qwen_path, &qwen_vocab_json, &qwen_merges_txt)) {
            qwen_tok_ok = qwen_tokenizer_.load_from_memory(qwen_vocab_json, qwen_merges_txt);
            if (qwen_tok_ok) {
                LOG_INFO("Qwen tokenizer loaded from GGUF metadata in %s", qwen_path.c_str());
            }
        }
    }
    if (!qwen_tok_ok) {
        LOG_ERROR("Qwen tokenizer not found in model_dir/vocab_dir or embedded GGUF metadata");
        return false;
    }

    // Load models.
    if (shared_wtype != cfg.wtype) {
        LOG_INFO("Using mixed precision: DiT=%s, shared modules=%s",
                 ggml_type_name(dit_wtype), ggml_type_name(shared_wtype));
    }
    LOG_INFO("Flash attention: %s", cfg.flash_attn ? "enabled" : "disabled");

    if (!load_dit(model_dir, dit_wtype, cfg.dit_filename, cfg.dit_gpu_layers)) return false;
    if (!load_qwen(model_dir, shared_wtype)) return false;
    if (!load_clip(model_dir, shared_wtype)) return false;
    if (dit_) dit_->set_flash_attn(cfg.flash_attn);
    if (qwen_) qwen_->set_flash_attn(cfg.flash_attn);
    if (clip_) clip_->set_flash_attn(cfg.flash_attn);

    // VAE is optional (3D VAE has 5D tensors that ggml can't handle yet)
    if (!load_vae(model_dir, shared_wtype)) {
        LOG_WARN("2D VAE not loaded — raw latent output only for images");
    }
    if (vae_) vae_->set_flash_attn(cfg.flash_attn);

    // 3D VAE for video
    if (!load_vae3d(model_dir, shared_wtype)) {
        LOG_WARN("3D VAE not loaded — raw latent output only for video");
    }
    if (vae3d_) vae3d_->set_flash_attn(cfg.flash_attn);

    // Init RNG
    rng_ = std::make_shared<PhiloxRNG>(cfg.seed >= 0 ? cfg.seed : 42);

    loaded_ = true;
    LOG_INFO("Pipeline loaded successfully");
    return true;
}

void KandinskyPipeline::unload() {
    ggml_backend_t runtime_backend = backend_;
    ggml_backend_t text_backend = text_backend_;
    ggml_backend_t vae_backend = vae_backend_;

    dit_.reset();
    qwen_.reset();
    clip_.reset();
    vae_.reset();
    vae3d_.reset();

    if (text_backend && text_backend != runtime_backend) {
        ggml_backend_free(text_backend);
    }
    if (vae_backend && vae_backend != runtime_backend && vae_backend != text_backend) {
        ggml_backend_free(vae_backend);
    }
    if (runtime_backend) {
        ggml_backend_free(runtime_backend);
    }

    backend_ = nullptr;
    text_backend_ = nullptr;
    vae_backend_ = nullptr;
    loaded_ = false;
}

// ── Unpatchify helper ─────────────────────────────────────────────────

static void unpatchify_velocity(
        const float* vel_data, float* vel_spatial, int64_t noise_n,
        int n_frames, int lat_h, int lat_w, int out_channels,
        int patch_area, int pd, int pph, int ppw,
        int n_patches_h, int n_patches_w, int64_t out_visual_dim) {
    std::fill(vel_spatial, vel_spatial + noise_n, 0.0f);
    for (int d = 0; d < n_frames; d++) {
        int di = d / pd, dd = d % pd;
        for (int h = 0; h < lat_h; h++) {
            int hi = h / pph, dh = h % pph;
            for (int w = 0; w < lat_w; w++) {
                int wi = w / ppw, dw = w % ppw;
                int token = (di * n_patches_h + hi) * n_patches_w + wi;
                for (int c = 0; c < out_channels; c++) {
                    int patch_idx = dd * pph * ppw + dh * ppw + dw;
                    // Channel-major flatten order: channel groups of patch positions.
                    int s = token * (int)out_visual_dim
                          + c * patch_area + patch_idx;
                    int d_idx = ((d * lat_h + h) * lat_w + w) * out_channels + c;
                    vel_spatial[d_idx] = vel_data[s];
                }
            }
        }
    }
}

// Log tensor statistics (mean, std, min, max) for diagnostics
static void log_tensor_stats(const char* label, ggml_tensor* t) {
    if (!t) { LOG_INFO("%s: (null)", label); return; }
    int64_t n = ggml_nelements(t);
    if (n == 0) { LOG_INFO("%s: (empty)", label); return; }

    // Read data — handle both CPU tensors and backend tensors
    std::vector<float> buf;
    float* data = nullptr;
    if (t->type == GGML_TYPE_F32 && t->data && !t->buffer) {
        data = (float*)t->data;
    } else {
        buf.resize(n);
        if (t->buffer) {
            // Backend tensor — need to convert to F32
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
            } else {
                LOG_INFO("%s: type=%s ne=[%lld,%lld,%lld] (non-F32 backend, skipped)",
                         label, ggml_type_name(t->type),
                         (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
                return;
            }
        } else if (t->data) {
            if (t->type == GGML_TYPE_F32) {
                memcpy(buf.data(), t->data, n * sizeof(float));
            } else {
                LOG_INFO("%s: type=%s ne=[%lld,%lld,%lld] (non-F32, skipped)",
                         label, ggml_type_name(t->type),
                         (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
                return;
            }
        } else {
            LOG_INFO("%s: (no data)", label);
            return;
        }
        data = buf.data();
    }

    float mn = data[0], mx = data[0];
    double sum = 0, sum2 = 0;
    int nan_count = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = data[i];
        if (std::isnan(v)) { nan_count++; continue; }
        sum += v; sum2 += (double)v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    int64_t valid = n - nan_count;
    double mean = valid > 0 ? sum / valid : 0;
    double var = valid > 0 ? sum2 / valid - mean * mean : 0;
    if (var < 0) var = 0;
    LOG_INFO("%s: ne=[%lld,%lld,%lld] mean=%.6f std=%.6f min=%.4f max=%.4f%s",
             label, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
             (float)mean, (float)sqrt(var), mn, mx,
             nan_count > 0 ? " (HAS NaN!)" : "");
}

// Compute L2 distance between two F32 tensors
static float tensor_l2_dist(ggml_tensor* a, ggml_tensor* b) {
    if (!a || !b) return -1.0f;
    int64_t n = ggml_nelements(a);
    if (n != ggml_nelements(b)) return -1.0f;

    std::vector<float> da(n), db(n);
    if (a->buffer) ggml_backend_tensor_get(a, da.data(), 0, n * sizeof(float));
    else if (a->data) memcpy(da.data(), a->data, n * sizeof(float));
    if (b->buffer) ggml_backend_tensor_get(b, db.data(), 0, n * sizeof(float));
    else if (b->data) memcpy(db.data(), b->data, n * sizeof(float));

    double dist = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)da[i] - (double)db[i];
        dist += d * d;
    }
    return (float)sqrt(dist / n);
}

static inline uint8_t float_to_u8(float x) {
    float v = (x + 1.0f) * 127.5f;
    v = std::max(0.0f, std::min(255.0f, v));
    return (uint8_t)v;
}

static std::pair<float, float> mean_std(const std::vector<float>& v) {
    if (v.empty()) return {0.0f, 0.0f};
    double sum = 0.0;
    double sum2 = 0.0;
    for (float x : v) {
        sum += x;
        sum2 += (double)x * (double)x;
    }
    double n = (double)v.size();
    double mean = sum / n;
    double var = sum2 / n - mean * mean;
    if (var < 0.0) var = 0.0;
    return {(float)mean, (float)sqrt(var)};
}

static bool validate_generate_config(const KandinskyPipeline::Config& cfg,
                                     bool video_mode,
                                     const char* caller) {
    if (cfg.width <= 0 || cfg.height <= 0) {
        LOG_ERROR("%s: width and height must be positive", caller);
        return false;
    }
    if (cfg.width % 16 != 0 || cfg.height % 16 != 0) {
        LOG_ERROR("%s: width and height must be multiples of 16", caller);
        return false;
    }
    if (cfg.num_steps <= 0) {
        LOG_ERROR("%s: num_steps must be > 0", caller);
        return false;
    }
    if (cfg.n_threads <= 0) {
        LOG_ERROR("%s: n_threads must be > 0", caller);
        return false;
    }
    if (cfg.scheduler_scale <= 0.0f) {
        LOG_ERROR("%s: scheduler_scale must be > 0", caller);
        return false;
    }
    if (cfg.guidance_scale < 0.0f) {
        LOG_ERROR("%s: guidance_scale must be >= 0", caller);
        return false;
    }
    if (video_mode && cfg.num_frames <= 0) {
        LOG_ERROR("%s: num_frames must be > 0 for video generation", caller);
        return false;
    }
    return true;
}

// Convert ggml image tensor [W, H, C, 1] (planar) to interleaved RGB uint8.
static void tensor_to_rgb_u8(const ggml_tensor* image, uint8_t* dst) {
    const int64_t w = image->ne[0];
    const int64_t h = image->ne[1];
    const int64_t c = image->ne[2];
    if (c < 3) return;

    const char* base = (const char*)image->data;
    for (int64_t y = 0; y < h; y++) {
        for (int64_t x = 0; x < w; x++) {
            int64_t pix = (y * w + x) * 3;
            for (int64_t ch = 0; ch < 3; ch++) {
                const float* src = (const float*)(base +
                    x * image->nb[0] +
                    y * image->nb[1] +
                    ch * image->nb[2]);
                dst[pix + ch] = float_to_u8(*src);
            }
        }
    }
}

// ── Text-to-Image generation ─────────────────────────────────────────

bool KandinskyPipeline::txt2img(const std::string& prompt, const std::string& neg_prompt,
                                 Config cfg, uint8_t** output_rgb, int* out_w, int* out_h) {
    if (!loaded_) {
        LOG_ERROR("Pipeline not loaded");
        return false;
    }
    if (!output_rgb || !out_w || !out_h) {
        LOG_ERROR("txt2img: output pointers must not be null");
        return false;
    }
    *output_rgb = nullptr;
    *out_w = 0;
    *out_h = 0;
    if (!validate_generate_config(cfg, false, "txt2img")) {
        return false;
    }
    if (!qwen_tokenizer_.is_loaded()) {
        LOG_ERROR("Qwen tokenizer not loaded. Provide --vocab-dir with text_encoder/vocab.json and merges.txt");
        return false;
    }
    if (!clip_tokenizer_.is_loaded()) {
        LOG_ERROR("CLIP tokenizer not loaded. Provide --vocab-dir with text_encoder_2/vocab.json + merges.txt");
        return false;
    }

    LOG_INFO("Generating %dx%d image with %d steps", cfg.width, cfg.height, cfg.num_steps);

    bool use_cfg = cfg.guidance_scale > 1.0f;

    // 1. Tokenize + encode with Qwen
    LOG_INFO("Encoding text with Qwen...");
    auto qwen_tokens = qwen_tokenizer_.encode_for_k5(prompt, "image");
    int crop_start = qwen_tokenizer_.crop_start("image");

    std::vector<int32_t> neg_qwen_tokens;
    if (use_cfg) {
        neg_qwen_tokens = qwen_tokenizer_.encode_for_k5(neg_prompt, "image");
    }

    // Buffer for positive + negative encodings
    size_t max_seq = std::max(qwen_tokens.size(), (size_t)128);
    if (use_cfg && neg_qwen_tokens.size() > max_seq)
        max_seq = neg_qwen_tokens.size();

    size_t tok_buf_size = max_seq * 3584 * sizeof(float) * (use_cfg ? 2 : 1)
                        + 768 * sizeof(float) * (use_cfg ? 2 : 1)
                        + max_seq * sizeof(int32_t) * (use_cfg ? 4 : 2)
                        + ggml_tensor_overhead() * (use_cfg ? 32 : 16);
    ggml_init_params tp = { tok_buf_size, nullptr, false };
    auto* tok_ctx = ggml_init(tp);
    if (!tok_ctx) {
        LOG_ERROR("Failed to allocate tokenizer context");
        return false;
    }
    auto* qwen_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, qwen_tokens.size());
    memcpy(qwen_ids->data, qwen_tokens.data(), qwen_tokens.size() * sizeof(int32_t));

    ggml_tensor* qwen_hidden = nullptr;
    if (!qwen_->forward(qwen_ids, cfg.n_threads, &qwen_hidden, tok_ctx)) {
        LOG_ERROR("Qwen forward failed");
        ggml_free(tok_ctx);
        return false;
    }

    // Crop system tokens from Qwen output
    if (qwen_hidden && crop_start > 0 && crop_start < qwen_hidden->ne[1]) {
        int64_t new_seq = qwen_hidden->ne[1] - crop_start;
        qwen_hidden = ggml_view_3d(tok_ctx, qwen_hidden,
            qwen_hidden->ne[0], new_seq, qwen_hidden->ne[2],
            qwen_hidden->nb[1], qwen_hidden->nb[2],
            crop_start * qwen_hidden->nb[1]);
    }
    auto tensor_hash64 = [](ggml_tensor* t) -> uint64_t {
        if (!t) return 0;
        std::vector<uint8_t> raw(ggml_nbytes(t));
        if (t->buffer) {
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        } else if (t->data) {
            memcpy(raw.data(), t->data, raw.size());
        } else {
            return 0;
        }
        uint64_t h = 1469598103934665603ull; // FNV-1a
        for (uint8_t b : raw) {
            h ^= (uint64_t)b;
            h *= 1099511628211ull;
        }
        return h;
    };
    if (qwen_hidden) {
        LOG_INFO("Qwen hidden: type=%s ne=[%lld,%lld,%lld] hash=%llu",
                 ggml_type_name(qwen_hidden->type),
                 (long long)qwen_hidden->ne[0], (long long)qwen_hidden->ne[1], (long long)qwen_hidden->ne[2],
                 (unsigned long long)tensor_hash64(qwen_hidden));
    }

    // 2. Tokenize + encode with CLIP
    LOG_INFO("Encoding text with CLIP...");
    auto clip_tokens = clip_tokenizer_.tokenize(prompt);
    auto* clip_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, clip_tokens.size());
    memcpy(clip_ids->data, clip_tokens.data(), clip_tokens.size() * sizeof(int32_t));

    ggml_tensor* clip_pooled = nullptr;
    if (!clip_->forward(clip_ids, cfg.n_threads, &clip_pooled, tok_ctx)) {
        LOG_ERROR("CLIP forward failed");
        ggml_free(tok_ctx);
        return false;
    }
    if (clip_pooled) {
        LOG_INFO("CLIP pooled: type=%s ne=[%lld,%lld,%lld] hash=%llu",
                 ggml_type_name(clip_pooled->type),
                 (long long)clip_pooled->ne[0], (long long)clip_pooled->ne[1], (long long)clip_pooled->ne[2],
                 (unsigned long long)tensor_hash64(clip_pooled));
    }

    // 3. Encode negative prompt (for CFG)
    ggml_tensor* neg_qwen_hidden = nullptr;
    ggml_tensor* neg_clip_pooled = nullptr;
    if (use_cfg) {
        LOG_INFO("Encoding negative prompt...");

        auto* neg_qwen_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, neg_qwen_tokens.size());
        memcpy(neg_qwen_ids->data, neg_qwen_tokens.data(),
               neg_qwen_tokens.size() * sizeof(int32_t));

        if (!qwen_->forward(neg_qwen_ids, cfg.n_threads, &neg_qwen_hidden, tok_ctx)) {
            LOG_ERROR("Qwen forward failed (negative)");
            ggml_free(tok_ctx);
            return false;
        }

        if (neg_qwen_hidden && crop_start > 0 && crop_start < neg_qwen_hidden->ne[1]) {
            int64_t new_seq = neg_qwen_hidden->ne[1] - crop_start;
            neg_qwen_hidden = ggml_view_3d(tok_ctx, neg_qwen_hidden,
                neg_qwen_hidden->ne[0], new_seq, neg_qwen_hidden->ne[2],
                neg_qwen_hidden->nb[1], neg_qwen_hidden->nb[2],
                crop_start * neg_qwen_hidden->nb[1]);
        }
        if (neg_qwen_hidden) {
            LOG_INFO("Qwen neg hidden: type=%s ne=[%lld,%lld,%lld] hash=%llu",
                     ggml_type_name(neg_qwen_hidden->type),
                     (long long)neg_qwen_hidden->ne[0], (long long)neg_qwen_hidden->ne[1], (long long)neg_qwen_hidden->ne[2],
                     (unsigned long long)tensor_hash64(neg_qwen_hidden));
        }

        auto neg_clip_tokens = clip_tokenizer_.tokenize(neg_prompt);
        auto* neg_clip_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, neg_clip_tokens.size());
        memcpy(neg_clip_ids->data, neg_clip_tokens.data(),
               neg_clip_tokens.size() * sizeof(int32_t));

        if (!clip_->forward(neg_clip_ids, cfg.n_threads, &neg_clip_pooled, tok_ctx)) {
            LOG_ERROR("CLIP forward failed (negative)");
            ggml_free(tok_ctx);
            return false;
        }
        if (neg_clip_pooled) {
            LOG_INFO("CLIP neg pooled: type=%s ne=[%lld,%lld,%lld] hash=%llu",
                     ggml_type_name(neg_clip_pooled->type),
                     (long long)neg_clip_pooled->ne[0], (long long)neg_clip_pooled->ne[1], (long long)neg_clip_pooled->ne[2],
                     (unsigned long long)tensor_hash64(neg_clip_pooled));
        }
    }

    // 4. Generate noise latent + set up patchification
    LOG_INFO("Generating noise latent...");
    int lat_h = cfg.height / 8;  // VAE 8x downsample
    int lat_w = cfg.width / 8;
    int n_frames = 1;  // Image mode

    // Derive dimensions from DiT config
    auto& dit_cfg = dit_->cfg;
    int pd  = dit_cfg.patch_size[0];   // temporal patch (1 for images)
    int pph = dit_cfg.patch_size[1];   // height patch (2)
    int ppw = dit_cfg.patch_size[2];   // width patch (2)
    int patch_area   = pd * pph * ppw;
    int out_channels = (int)(dit_cfg.out_visual_dim / patch_area);
    int in_channels  = (int)(dit_cfg.in_visual_dim / patch_area);

    int n_patches_h     = lat_h / pph;
    int n_patches_w     = lat_w / ppw;
    int n_visual_tokens = (n_frames / pd) * n_patches_h * n_patches_w;

    LOG_INFO("Latent: %dx%d, patches: %dx%d = %d tokens, out_ch=%d",
             lat_w, lat_h, n_patches_w, n_patches_h, n_visual_tokens, out_channels);

    // Generate noise (out_channels only — extra input channels are zeros for txt2img)
    int64_t noise_n = (int64_t)n_frames * lat_h * lat_w * out_channels;
    auto noise = rng_->randn((uint32_t)noise_n);

    // Pre-allocate CPU buffers for patchified I/O
    int64_t patched_in_size  = (int64_t)n_visual_tokens * dit_cfg.in_visual_dim;
    int64_t patched_out_size = (int64_t)n_visual_tokens * dit_cfg.out_visual_dim;
    std::vector<float> patched_input(patched_in_size, 0.0f);
    std::vector<float> vel_spatial(noise_n);
    std::vector<float> vel_uncond_spatial;
    if (use_cfg) vel_uncond_spatial.resize(noise_n);

    // Allocate lat_ctx for dit_input + velocity + final latent tensors
    size_t lat_buf_size =
        patched_in_size * sizeof(float) +    // dit input
        patched_out_size * sizeof(float) +   // velocity output
        noise_n * sizeof(float) +            // final latent for VAE
        ggml_tensor_overhead() * 16;
    if (use_cfg) {
        lat_buf_size += patched_out_size * sizeof(float) + ggml_tensor_overhead();
    }
    ggml_init_params lp = { lat_buf_size, nullptr, false };
    auto* lat_ctx = ggml_init(lp);
    if (!lat_ctx) {
        LOG_ERROR("Failed to allocate latent context");
        ggml_free(tok_ctx);
        return false;
    }

    // DiT input tensor (2D pre-patchified, reused across steps)
    auto* dit_input = ggml_new_tensor_2d(lat_ctx, GGML_TYPE_F32,
                                          dit_cfg.in_visual_dim, n_visual_tokens);
    // Pre-allocate velocity output tensor (reused across steps)
    auto* velocity = ggml_new_tensor_2d(lat_ctx, GGML_TYPE_F32,
                                         dit_cfg.out_visual_dim, n_visual_tokens);
    ggml_tensor* velocity_uncond = nullptr;
    if (use_cfg) {
        velocity_uncond = ggml_new_tensor_2d(lat_ctx, GGML_TYPE_F32,
                                              dit_cfg.out_visual_dim, n_visual_tokens);
    }

    // 5. Denoise loop
    LOG_INFO("Denoising with %d steps (guidance=%.1f, scheduler_scale=%.1f)...",
             cfg.num_steps, cfg.guidance_scale, cfg.scheduler_scale);

    auto timesteps = scheduler::flow_schedule(cfg.num_steps, cfg.scheduler_scale);

    for (int step = 0; step < cfg.num_steps; step++) {
        float t = timesteps[step];
        float dt = timesteps[step + 1] - timesteps[step];

        // Patchify: noise (out_channels) into in_channels-ch patchified format
        // Extra channels (cond_image + mask) are zero for txt2img
        std::fill(patched_input.begin(), patched_input.end(), 0.0f);
        for (int d = 0; d < n_frames; d++) {
            int di = d / pd, dd = d % pd;
            for (int h = 0; h < lat_h; h++) {
                int hi = h / pph, dh = h % pph;
                for (int w = 0; w < lat_w; w++) {
                    int wi = w / ppw, dw = w % ppw;
                    int token = (di * n_patches_h + hi) * n_patches_w + wi;
                    for (int c = 0; c < out_channels; c++) {
                        int src = ((d * lat_h + h) * lat_w + w) * out_channels + c;
                        int patch_idx = dd * pph * ppw + dh * ppw + dw;
                        int dst = token * (int)dit_cfg.in_visual_dim
                                + patch_idx * in_channels + c;
                        patched_input[dst] = noise[src];
                    }
                }
            }
        }
        memcpy(dit_input->data, patched_input.data(), patched_in_size * sizeof(float));

        // Conditional velocity
        ggml_tensor* vel_ptr = velocity;
        if (!dit_->forward(dit_input, qwen_hidden, clip_pooled,
                           t * 1000.0f, n_frames, lat_h, lat_w,
                           cfg.n_threads, &vel_ptr, lat_ctx)) {
            LOG_ERROR("DiT forward failed at step %d", step);
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }

        // Unconditional velocity (CFG)
        if (use_cfg) {
            ggml_tensor* vel_unc_ptr = velocity_uncond;
            if (!dit_->forward(dit_input, neg_qwen_hidden, neg_clip_pooled,
                               t * 1000.0f, n_frames, lat_h, lat_w,
                               cfg.n_threads, &vel_unc_ptr, lat_ctx)) {
                LOG_ERROR("DiT unconditional forward failed at step %d", step);
                ggml_free(tok_ctx);
                ggml_free(lat_ctx);
                return false;
            }
        }

        // Unpatchify conditional velocity
        unpatchify_velocity((float*)velocity->data, vel_spatial.data(), noise_n,
                            n_frames, lat_h, lat_w, out_channels, patch_area,
                            pd, pph, ppw, n_patches_h, n_patches_w,
                            dit_cfg.out_visual_dim);

        // CFG blending: vel = vel_uncond + scale * (vel_cond - vel_uncond)
        if (use_cfg) {
            unpatchify_velocity((float*)velocity_uncond->data, vel_uncond_spatial.data(),
                                noise_n, n_frames, lat_h, lat_w, out_channels, patch_area,
                                pd, pph, ppw, n_patches_h, n_patches_w,
                                dit_cfg.out_visual_dim);
            for (int64_t i = 0; i < noise_n; i++) {
                vel_spatial[i] = vel_uncond_spatial[i] +
                    cfg.guidance_scale * (vel_spatial[i] - vel_uncond_spatial[i]);
            }
        }

        // Log velocity stats before Euler step
        {
            float v_min = vel_spatial[0], v_max = vel_spatial[0];
            double v_sum = 0, v_sum2 = 0;
            for (int64_t i = 0; i < noise_n; i++) {
                float v = vel_spatial[i];
                v_sum += v; v_sum2 += (double)v * v;
                if (v < v_min) v_min = v;
                if (v > v_max) v_max = v;
            }
            double v_mean = v_sum / noise_n;
            double v_std = sqrt(v_sum2 / noise_n - v_mean * v_mean);
            LOG_INFO("Step %d/%d: t=%.4f dt=%.4f vel: mean=%.6f std=%.6f min=%.4f max=%.4f",
                     step+1, cfg.num_steps, t, dt, (float)v_mean, (float)v_std, v_min, v_max);
        }

        // Euler step: noise += dt * velocity
        for (int64_t i = 0; i < noise_n; i++) {
            noise[i] += dt * vel_spatial[i];
        }

        pretty_progress(step + 1, cfg.num_steps, 0.0f);
    }

    // Create latent tensor from denoised noise for VAE / visualization
    auto* latent = ggml_new_tensor_4d(lat_ctx, GGML_TYPE_F32,
                                       lat_w, lat_h, out_channels, n_frames);
    {
        float* lat = (float*)latent->data;
        for (int d = 0; d < n_frames; d++) {
            for (int h = 0; h < lat_h; h++) {
                for (int w = 0; w < lat_w; w++) {
                    for (int c = 0; c < out_channels; c++) {
                        int src = ((d * lat_h + h) * lat_w + w) * out_channels + c;
                        int dst = ((d * out_channels + c) * lat_h + h) * lat_w + w;
                        lat[dst] = noise[src];
                    }
                }
            }
        }
    }

    // 5. Decode with VAE (if available)
    if (vae_) {
        LOG_INFO("Decoding with VAE...");
        ggml_tensor* image = nullptr;
        if (!vae_->decode(latent, cfg.n_threads, &image, lat_ctx)) {
            LOG_ERROR("VAE decode failed");
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }

        // 6. Convert to RGB uint8
        if (image) {
            int64_t w = image->ne[0], h = image->ne[1];
            *out_w = (int)w;
            *out_h = (int)h;
            *output_rgb = (uint8_t*)malloc(w * h * 3);
            if (!*output_rgb) {
                LOG_ERROR("Failed to allocate image output buffer");
                ggml_free(tok_ctx);
                ggml_free(lat_ctx);
                return false;
            }
            tensor_to_rgb_u8(image, *output_rgb);
        }
    } else {
        // No VAE — output latent as grayscale visualization
        LOG_WARN("No VAE — outputting latent visualization");
        int lat_w_out = cfg.width / 8;
        int lat_h_out = cfg.height / 8;
        *out_w = lat_w_out;
        *out_h = lat_h_out;
        *output_rgb = (uint8_t*)calloc(lat_w_out * lat_h_out * 3, 1);
        if (!*output_rgb) {
            LOG_ERROR("Failed to allocate latent visualization buffer");
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }
        if (latent && latent->data) {
            float* lat_data = (float*)latent->data;
            for (int i = 0; i < lat_w_out * lat_h_out; i++) {
                float val = lat_data[i] * 30.0f + 128.0f;
                val = std::max(0.0f, std::min(255.0f, val));
                (*output_rgb)[i * 3 + 0] = (uint8_t)val;
                (*output_rgb)[i * 3 + 1] = (uint8_t)val;
                (*output_rgb)[i * 3 + 2] = (uint8_t)val;
            }
        }
    }

    ggml_free(tok_ctx);
    ggml_free(lat_ctx);

    LOG_INFO("Generation complete");
    return true;
}

// ── Text-to-Video generation ─────────────────────────────────────────

bool KandinskyPipeline::txt2vid(const std::string& prompt, const std::string& neg_prompt,
                                 Config cfg, uint8_t** output_frames,
                                 int* out_w, int* out_h, int* out_n_frames) {
    if (!loaded_) {
        LOG_ERROR("Pipeline not loaded");
        return false;
    }
    if (!output_frames || !out_w || !out_h || !out_n_frames) {
        LOG_ERROR("txt2vid: output pointers must not be null");
        return false;
    }
    *output_frames = nullptr;
    *out_w = 0;
    *out_h = 0;
    *out_n_frames = 0;
    if (!validate_generate_config(cfg, true, "txt2vid")) {
        return false;
    }
    if (!qwen_tokenizer_.is_loaded()) {
        LOG_ERROR("Qwen tokenizer not loaded. Provide --vocab-dir with text_encoder/vocab.json and merges.txt");
        return false;
    }
    if (!clip_tokenizer_.is_loaded()) {
        LOG_ERROR("CLIP tokenizer not loaded. Provide --vocab-dir with text_encoder_2/vocab.json + merges.txt");
        return false;
    }

    // Temporal compression = 4: n_lat_frames = (n_frames - 1) / 4 + 1
    int n_frames = cfg.num_frames;
    int n_lat_frames = (n_frames - 1) / 4 + 1;
    if (n_lat_frames < 1) n_lat_frames = 1;

    LOG_INFO("Generating %dx%d video, %d frames (%d latent frames), %d steps",
             cfg.width, cfg.height, n_frames, n_lat_frames, cfg.num_steps);

    bool use_cfg = cfg.guidance_scale > 1.0f;

    // 1. Tokenize + encode with Qwen
    LOG_INFO("Encoding text with Qwen...");
    auto qwen_tokens = qwen_tokenizer_.encode_for_k5(prompt, "video");
    int crop_start = qwen_tokenizer_.crop_start("video");
    {
        int64_t checksum = 0;
        for (auto v : qwen_tokens) checksum = checksum * 1315423911u + v;
        LOG_INFO("Qwen(video) tokens: n=%zu crop_start=%d checksum=%lld",
                 qwen_tokens.size(), crop_start, (long long)checksum);
        int s = std::max(0, crop_start - 2);
        int e = std::min((int)qwen_tokens.size(), crop_start + 10);
        std::string window;
        for (int i = s; i < e; i++) {
            if (!window.empty()) window += ",";
            window += std::to_string(qwen_tokens[i]);
        }
        LOG_INFO("Qwen(video) token window [%d:%d): %s", s, e, window.c_str());
    }

    std::vector<int32_t> neg_qwen_tokens;
    if (use_cfg) {
        neg_qwen_tokens = qwen_tokenizer_.encode_for_k5(neg_prompt, "video");
    }

    size_t max_seq = std::max(qwen_tokens.size(), (size_t)128);
    if (use_cfg && neg_qwen_tokens.size() > max_seq)
        max_seq = neg_qwen_tokens.size();

    size_t tok_buf_size = max_seq * 3584 * sizeof(float) * (use_cfg ? 2 : 1)
                        + 768 * sizeof(float) * (use_cfg ? 2 : 1)
                        + max_seq * sizeof(int32_t) * (use_cfg ? 4 : 2)
                        + ggml_tensor_overhead() * (use_cfg ? 32 : 16);
    ggml_init_params tp = { tok_buf_size, nullptr, false };
    auto* tok_ctx = ggml_init(tp);
    if (!tok_ctx) {
        LOG_ERROR("Failed to allocate tokenizer context");
        return false;
    }
    auto* qwen_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, qwen_tokens.size());
    memcpy(qwen_ids->data, qwen_tokens.data(), qwen_tokens.size() * sizeof(int32_t));

    ggml_tensor* qwen_hidden = nullptr;
    if (!qwen_->forward(qwen_ids, cfg.n_threads, &qwen_hidden, tok_ctx)) {
        LOG_ERROR("Qwen forward failed");
        ggml_free(tok_ctx);
        return false;
    }

    if (qwen_hidden && crop_start > 0 && crop_start < qwen_hidden->ne[1]) {
        int64_t new_seq = qwen_hidden->ne[1] - crop_start;
        qwen_hidden = ggml_view_3d(tok_ctx, qwen_hidden,
            qwen_hidden->ne[0], new_seq, qwen_hidden->ne[2],
            qwen_hidden->nb[1], qwen_hidden->nb[2],
            crop_start * qwen_hidden->nb[1]);
    }

    // 2. Tokenize + encode with CLIP
    LOG_INFO("Encoding text with CLIP...");
    auto clip_tokens = clip_tokenizer_.tokenize(prompt);
    {
        int64_t checksum = 0;
        for (auto v : clip_tokens) checksum = checksum * 1315423911u + v;
        LOG_INFO("CLIP tokens: n=%zu checksum=%lld",
                 clip_tokens.size(), (long long)checksum);
    }
    auto* clip_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, clip_tokens.size());
    memcpy(clip_ids->data, clip_tokens.data(), clip_tokens.size() * sizeof(int32_t));

    ggml_tensor* clip_pooled = nullptr;
    if (!clip_->forward(clip_ids, cfg.n_threads, &clip_pooled, tok_ctx)) {
        LOG_ERROR("CLIP forward failed");
        ggml_free(tok_ctx);
        return false;
    }

    // 3. Encode negative prompt (for CFG)
    ggml_tensor* neg_qwen_hidden = nullptr;
    ggml_tensor* neg_clip_pooled = nullptr;
    if (use_cfg) {
        LOG_INFO("Encoding negative prompt...");

        auto* neg_qwen_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, neg_qwen_tokens.size());
        memcpy(neg_qwen_ids->data, neg_qwen_tokens.data(),
               neg_qwen_tokens.size() * sizeof(int32_t));

        if (!qwen_->forward(neg_qwen_ids, cfg.n_threads, &neg_qwen_hidden, tok_ctx)) {
            LOG_ERROR("Qwen forward failed (negative)");
            ggml_free(tok_ctx);
            return false;
        }

        if (neg_qwen_hidden && crop_start > 0 && crop_start < neg_qwen_hidden->ne[1]) {
            int64_t new_seq = neg_qwen_hidden->ne[1] - crop_start;
            neg_qwen_hidden = ggml_view_3d(tok_ctx, neg_qwen_hidden,
                neg_qwen_hidden->ne[0], new_seq, neg_qwen_hidden->ne[2],
                neg_qwen_hidden->nb[1], neg_qwen_hidden->nb[2],
                crop_start * neg_qwen_hidden->nb[1]);
        }

        auto neg_clip_tokens = clip_tokenizer_.tokenize(neg_prompt);
        auto* neg_clip_ids = ggml_new_tensor_1d(tok_ctx, GGML_TYPE_I32, neg_clip_tokens.size());
        memcpy(neg_clip_ids->data, neg_clip_tokens.data(),
               neg_clip_tokens.size() * sizeof(int32_t));

        if (!clip_->forward(neg_clip_ids, cfg.n_threads, &neg_clip_pooled, tok_ctx)) {
            LOG_ERROR("CLIP forward failed (negative)");
            ggml_free(tok_ctx);
            return false;
        }
    }

    // ── Embedding diagnostics ───────────────────────────────────────
    log_tensor_stats("Qwen hidden (pos)", qwen_hidden);
    log_tensor_stats("CLIP pooled (pos)", clip_pooled);
    if (use_cfg) {
        log_tensor_stats("Qwen hidden (neg)", neg_qwen_hidden);
        log_tensor_stats("CLIP pooled (neg)", neg_clip_pooled);
        // Measure how different the CLIP pooled embeddings are (same size)
        float clip_dist = tensor_l2_dist(clip_pooled, neg_clip_pooled);
        LOG_INFO("CLIP pooled L2 distance pos-neg: %.6f (higher = more different)", clip_dist);
        // For Qwen, sequence lengths may differ; compare per-element of min overlap
        if (qwen_hidden && neg_qwen_hidden) {
            int64_t min_seq = std::min(qwen_hidden->ne[1], neg_qwen_hidden->ne[1]);
            int64_t dim = qwen_hidden->ne[0];
            LOG_INFO("Qwen hidden seq lens: pos=%lld neg=%lld (comparing first %lld tokens)",
                     (long long)qwen_hidden->ne[1], (long long)neg_qwen_hidden->ne[1],
                     (long long)min_seq);
        }
    }

    // 4. Generate noise latent
    LOG_INFO("Generating noise latent (%d latent frames)...", n_lat_frames);
    int lat_h = cfg.height / 8;
    int lat_w = cfg.width / 8;

    auto& dit_cfg = dit_->cfg;
    int pd  = dit_cfg.patch_size[0];
    int pph = dit_cfg.patch_size[1];
    int ppw = dit_cfg.patch_size[2];
    int patch_area   = pd * pph * ppw;
    int out_channels = (int)(dit_cfg.out_visual_dim / patch_area);
    int in_channels  = (int)(dit_cfg.in_visual_dim / patch_area);

    int n_patches_h     = lat_h / pph;
    int n_patches_w     = lat_w / ppw;
    int n_visual_tokens = (n_lat_frames / pd) * n_patches_h * n_patches_w;

    LOG_INFO("Latent: %dx%dx%d, patches: %dx%d = %d tokens, out_ch=%d",
             lat_w, lat_h, n_lat_frames, n_patches_w, n_patches_h, n_visual_tokens,
             out_channels);

    int64_t noise_n = (int64_t)n_lat_frames * lat_h * lat_w * out_channels;
    auto noise = rng_->randn((uint32_t)noise_n);
    {
        auto st = mean_std(noise);
        LOG_INFO("Latent stats before denoise: mean=%.5f std=%.5f", st.first, st.second);
    }

    int64_t patched_in_size  = (int64_t)n_visual_tokens * dit_cfg.in_visual_dim;
    int64_t patched_out_size = (int64_t)n_visual_tokens * dit_cfg.out_visual_dim;
    std::vector<float> patched_input(patched_in_size, 0.0f);
    std::vector<float> vel_spatial(noise_n);
    std::vector<float> vel_uncond_spatial;
    if (use_cfg) vel_uncond_spatial.resize(noise_n);

    // VAE output: packed tensor (all frames concatenated) + individual frame tensors
    int64_t vae_frame_bytes = (int64_t)cfg.width * cfg.height * 3 * sizeof(float);
    size_t lat_buf_size =
        patched_in_size * sizeof(float) +
        patched_out_size * sizeof(float) +
        noise_n * sizeof(float) +
        vae_frame_bytes * n_frames * 2 +
        ggml_tensor_overhead() * (32 + n_frames * 2);
    if (use_cfg) {
        lat_buf_size += patched_out_size * sizeof(float) + ggml_tensor_overhead();
    }
    ggml_init_params lp = { lat_buf_size, nullptr, false };
    auto* lat_ctx = ggml_init(lp);
    if (!lat_ctx) {
        LOG_ERROR("Failed to allocate latent context");
        ggml_free(tok_ctx);
        return false;
    }

    auto* dit_input = ggml_new_tensor_2d(lat_ctx, GGML_TYPE_F32,
                                          dit_cfg.in_visual_dim, n_visual_tokens);
    auto* velocity = ggml_new_tensor_2d(lat_ctx, GGML_TYPE_F32,
                                         dit_cfg.out_visual_dim, n_visual_tokens);
    ggml_tensor* velocity_uncond = nullptr;
    if (use_cfg) {
        velocity_uncond = ggml_new_tensor_2d(lat_ctx, GGML_TYPE_F32,
                                              dit_cfg.out_visual_dim, n_visual_tokens);
    }

    // 5. Denoise loop
    LOG_INFO("Denoising with %d steps (guidance=%.1f, scheduler_scale=%.1f)...",
             cfg.num_steps, cfg.guidance_scale, cfg.scheduler_scale);

    auto timesteps = scheduler::flow_schedule(cfg.num_steps, cfg.scheduler_scale);

    for (int step = 0; step < cfg.num_steps; step++) {
        float t = timesteps[step];
        float dt = timesteps[step + 1] - timesteps[step];

        // Patchify
        std::fill(patched_input.begin(), patched_input.end(), 0.0f);
        for (int d = 0; d < n_lat_frames; d++) {
            int di = d / pd, dd = d % pd;
            for (int h = 0; h < lat_h; h++) {
                int hi = h / pph, dh = h % pph;
                for (int w = 0; w < lat_w; w++) {
                    int wi = w / ppw, dw = w % ppw;
                    int token = (di * n_patches_h + hi) * n_patches_w + wi;
                    for (int c = 0; c < out_channels; c++) {
                        int src = ((d * lat_h + h) * lat_w + w) * out_channels + c;
                        int patch_idx = dd * pph * ppw + dh * ppw + dw;
                        int dst = token * (int)dit_cfg.in_visual_dim
                                + patch_idx * in_channels + c;
                        patched_input[dst] = noise[src];
                    }
                }
            }
        }
        memcpy(dit_input->data, patched_input.data(), patched_in_size * sizeof(float));

        // Conditional velocity
        ggml_tensor* vel_ptr = velocity;
        if (!dit_->forward(dit_input, qwen_hidden, clip_pooled,
                           t * 1000.0f, n_lat_frames, lat_h, lat_w,
                           cfg.n_threads, &vel_ptr, lat_ctx)) {
            LOG_ERROR("DiT forward failed at step %d", step);
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }

        // Unconditional velocity (CFG)
        if (use_cfg) {
            ggml_tensor* vel_unc_ptr = velocity_uncond;
            if (!dit_->forward(dit_input, neg_qwen_hidden, neg_clip_pooled,
                               t * 1000.0f, n_lat_frames, lat_h, lat_w,
                               cfg.n_threads, &vel_unc_ptr, lat_ctx)) {
                LOG_ERROR("DiT unconditional forward failed at step %d", step);
                ggml_free(tok_ctx);
                ggml_free(lat_ctx);
                return false;
            }
        }

        // Unpatchify conditional velocity
        unpatchify_velocity((float*)velocity->data, vel_spatial.data(), noise_n,
                            n_lat_frames, lat_h, lat_w, out_channels, patch_area,
                            pd, pph, ppw, n_patches_h, n_patches_w,
                            dit_cfg.out_visual_dim);

        // CFG blending: vel = vel_uncond + scale * (vel_cond - vel_uncond)
        if (use_cfg) {
            unpatchify_velocity((float*)velocity_uncond->data, vel_uncond_spatial.data(),
                                noise_n, n_lat_frames, lat_h, lat_w, out_channels,
                                patch_area, pd, pph, ppw, n_patches_h, n_patches_w,
                                dit_cfg.out_visual_dim);
            if (step == 0) {
                double mae = 0.0;
                for (int64_t i = 0; i < noise_n; i++) {
                    mae += fabs((double)vel_spatial[i] - (double)vel_uncond_spatial[i]);
                }
                LOG_INFO("CFG delta step1 (video): mean_abs=%.8f", (float)(mae / (double)noise_n));
            }
            for (int64_t i = 0; i < noise_n; i++) {
                vel_spatial[i] = vel_uncond_spatial[i] +
                    cfg.guidance_scale * (vel_spatial[i] - vel_uncond_spatial[i]);
            }
        }

        // Log velocity stats before Euler step
        {
            float v_min = vel_spatial[0], v_max = vel_spatial[0];
            double v_sum = 0, v_sum2 = 0;
            for (int64_t i = 0; i < noise_n; i++) {
                float v = vel_spatial[i];
                v_sum += v; v_sum2 += (double)v * v;
                if (v < v_min) v_min = v;
                if (v > v_max) v_max = v;
            }
            double v_mean = v_sum / noise_n;
            double v_std = sqrt(v_sum2 / noise_n - v_mean * v_mean);
            LOG_INFO("Step %d/%d: t=%.4f dt=%.4f vel: mean=%.6f std=%.6f min=%.4f max=%.4f",
                     step+1, cfg.num_steps, t, dt, (float)v_mean, (float)v_std, v_min, v_max);
        }

        // Euler step
        for (int64_t i = 0; i < noise_n; i++) {
            noise[i] += dt * vel_spatial[i];
        }

        pretty_progress(step + 1, cfg.num_steps, 0.0f);
    }
    {
        auto st = mean_std(noise);
        LOG_INFO("Latent stats after denoise:  mean=%.5f std=%.5f", st.first, st.second);
    }

    // Create latent tensor [lat_w, lat_h, 16, n_lat_frames]
    auto* latent = ggml_new_tensor_4d(lat_ctx, GGML_TYPE_F32,
                                       lat_w, lat_h, out_channels, n_lat_frames);
    {
        float* lat = (float*)latent->data;
        for (int d = 0; d < n_lat_frames; d++) {
            for (int h = 0; h < lat_h; h++) {
                for (int w = 0; w < lat_w; w++) {
                    for (int c = 0; c < out_channels; c++) {
                        int src = ((d * lat_h + h) * lat_w + w) * out_channels + c;
                        int dst = ((d * out_channels + c) * lat_h + h) * lat_w + w;
                        lat[dst] = noise[src];
                    }
                }
            }
        }
    }

    // 5. Decode with 3D VAE (if available)
    if (vae3d_) {
        LOG_INFO("Decoding with 3D VAE (%d latent frames)...", n_lat_frames);
        std::vector<ggml_tensor*> decoded_frames;
        if (!vae3d_->decode(latent, n_lat_frames, cfg.n_threads, decoded_frames, lat_ctx)) {
            LOG_ERROR("VAE3D decode failed");
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }

        int n_decoded = (int)decoded_frames.size();
        LOG_INFO("VAE3D produced %d output frames", n_decoded);

        *out_n_frames = n_decoded;
        *out_w = (int)decoded_frames[0]->ne[0];
        *out_h = (int)decoded_frames[0]->ne[1];

        int64_t frame_pixels = (int64_t)(*out_w) * (*out_h) * 3;
        *output_frames = (uint8_t*)malloc(n_decoded * frame_pixels);
        if (!*output_frames) {
            LOG_ERROR("Failed to allocate decoded video buffer");
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }

        for (int f = 0; f < n_decoded; f++) {
            uint8_t* dst = *output_frames + f * frame_pixels;
            tensor_to_rgb_u8(decoded_frames[f], dst);
        }
    } else {
        // No 3D VAE — output latent visualization per frame
        LOG_WARN("No 3D VAE — outputting latent visualization");
        *out_w = lat_w;
        *out_h = lat_h;
        *out_n_frames = n_lat_frames;

        int64_t frame_pixels = (int64_t)lat_w * lat_h * 3;
        *output_frames = (uint8_t*)calloc(n_lat_frames * frame_pixels, 1);
        if (!*output_frames) {
            LOG_ERROR("Failed to allocate latent video visualization buffer");
            ggml_free(tok_ctx);
            ggml_free(lat_ctx);
            return false;
        }

        float* lat_data = (float*)latent->data;
        for (int f = 0; f < n_lat_frames; f++) {
            uint8_t* dst = *output_frames + f * frame_pixels;
            float* src = lat_data + f * lat_w * lat_h * out_channels;
            for (int i = 0; i < lat_w * lat_h; i++) {
                float val = src[i] * 30.0f + 128.0f;
                val = std::max(0.0f, std::min(255.0f, val));
                dst[i * 3 + 0] = (uint8_t)val;
                dst[i * 3 + 1] = (uint8_t)val;
                dst[i * 3 + 2] = (uint8_t)val;
            }
        }
    }

    ggml_free(tok_ctx);
    ggml_free(lat_ctx);

    LOG_INFO("Video generation complete");
    return true;
}

// ── C API implementation ─────────────────────────────────────────────

struct kd_ctx {
    KandinskyPipeline pipeline;
    KandinskyPipeline::Config config;
};

extern "C" {

struct kd_params kd_default_params(void) {
    struct kd_params p = {};
    p.model_dir    = nullptr;
    p.vocab_dir    = nullptr;
    p.dit_filename = nullptr;
    p.n_threads    = 4;
    p.flash_attn   = true;
    p.wtype        = KD_TYPE_F16;
    p.text_cpu     = false;
    p.vae3d_cpu    = false;
    p.dit_gpu_layers = -1;
    p.log_cb       = nullptr;
    p.log_cb_data  = nullptr;
    return p;
}

struct kd_generate_params kd_default_generate_params(void) {
    struct kd_generate_params p = {};
    p.prompt          = nullptr;
    p.negative_prompt = "";
    p.width           = 1024;
    p.height          = 1024;
    p.num_frames      = 1;
    p.num_steps       = 50;
    p.guidance_scale  = 3.5f;
    p.scheduler_scale = 3.0f;
    p.seed            = -1;
    p.mode            = KD_MODE_TXT2IMG;
    p.progress_cb     = nullptr;
    p.progress_cb_data = nullptr;
    return p;
}

kd_ctx_t* kd_ctx_create(struct kd_params params) {
    if (params.log_cb) {
        kd_set_log_callback(params.log_cb, params.log_cb_data);
    }

    auto* ctx = new kd_ctx();
    ctx->config.n_threads = params.n_threads;
    ctx->config.flash_attn = params.flash_attn;
    ctx->config.wtype = (ggml_type)params.wtype;
    ctx->config.text_cpu = params.text_cpu;
    ctx->config.vae3d_cpu = params.vae3d_cpu;
    ctx->config.dit_gpu_layers = params.dit_gpu_layers;
    if (params.vocab_dir) ctx->config.vocab_dir = params.vocab_dir;
    if (params.dit_filename) ctx->config.dit_filename = params.dit_filename;

    if (params.model_dir) {
        if (!ctx->pipeline.load(params.model_dir, ctx->config)) {
            delete ctx;
            return nullptr;
        }
    }

    return ctx;
}

void kd_ctx_free(kd_ctx_t* ctx) {
    if (ctx) {
        ctx->pipeline.unload();
        delete ctx;
    }
}

int kd_generate(kd_ctx_t* ctx, struct kd_generate_params params, kd_image_t** output) {
    if (!ctx || !params.prompt || !output) return -1;
    *output = nullptr;

    KandinskyPipeline::Config cfg = ctx->config;
    cfg.width = params.width;
    cfg.height = params.height;
    cfg.num_frames = params.num_frames;
    cfg.num_steps = params.num_steps;
    cfg.guidance_scale = params.guidance_scale;
    cfg.scheduler_scale = params.scheduler_scale;
    cfg.seed = params.seed;

    if (params.mode == KD_MODE_TXT2VID || params.num_frames > 1) {
        // Video generation
        uint8_t* frame_data = nullptr;
        int w = 0, h = 0, n_frames = 0;

        if (!ctx->pipeline.txt2vid(
                params.prompt,
                params.negative_prompt ? params.negative_prompt : "",
                cfg, &frame_data, &w, &h, &n_frames)) {
            return -1;
        }

        if (!frame_data || n_frames <= 0) return -1;

        int64_t frame_bytes = (int64_t)w * h * 3;
        *output = (kd_image_t*)malloc(n_frames * sizeof(kd_image_t));
        if (!*output) {
            free(frame_data);
            return -1;
        }
        for (int i = 0; i < n_frames; i++) {
            (*output)[i].width = w;
            (*output)[i].height = h;
            (*output)[i].channel = 3;
            (*output)[i].data = (uint8_t*)malloc(frame_bytes);
            if (!(*output)[i].data) {
                for (int j = 0; j < i; j++) {
                    free((*output)[j].data);
                }
                free(*output);
                *output = nullptr;
                free(frame_data);
                return -1;
            }
            memcpy((*output)[i].data, frame_data + i * frame_bytes, frame_bytes);
        }
        free(frame_data);

        return n_frames;
    } else {
        // Image generation
        uint8_t* rgb = nullptr;
        int w = 0, h = 0;

        if (!ctx->pipeline.txt2img(
                params.prompt,
                params.negative_prompt ? params.negative_prompt : "",
                cfg, &rgb, &w, &h)) {
            return -1;
        }

        if (!rgb) return -1;

        *output = (kd_image_t*)malloc(sizeof(kd_image_t));
        if (!*output) {
            free(rgb);
            return -1;
        }
        (*output)->width = w;
        (*output)->height = h;
        (*output)->channel = 3;
        (*output)->data = rgb;

        return 1;
    }
}

void kd_image_free(kd_image_t* images, int count) {
    if (!images) return;
    for (int i = 0; i < count; i++) {
        if (images[i].data) free(images[i].data);
    }
    free(images);
}

} // extern "C"
