#ifndef KD_CLIP_HPP
#define KD_CLIP_HPP

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "util.hpp"
#include "core/module.hpp"
#include "core/executor.hpp"

// ── CLIP BPE Tokenizer ──────────────────────────────────────────────

class CLIPTokenizer {
public:
    static constexpr int MAX_LENGTH    = 77;
    static constexpr int BOS_TOKEN_ID  = 49406;
    static constexpr int EOS_TOKEN_ID  = 49407;
    static constexpr int PAD_TOKEN_ID  = 49407;
    static constexpr int VOCAB_SIZE    = 49408;

    CLIPTokenizer() { init_byte_encoder(); }

    bool load(const std::string& vocab_path, const std::string& merges_path) {
        token_to_id_.clear();
        id_to_token_.clear();
        bpe_merges_.clear();
        bpe_cache_.clear();
        if (!load_from_vocab(vocab_path)) return false;
        if (!load_from_merges(merges_path)) return false;
        return true;
    }

    bool load_from_memory(const std::string& vocab_json, const std::string& merges_text) {
        token_to_id_.clear();
        id_to_token_.clear();
        bpe_merges_.clear();
        bpe_cache_.clear();
        if (!load_from_vocab_string(vocab_json, "<embedded>")) return false;
        if (!load_from_merges_string(merges_text, "<embedded>")) return false;
        return true;
    }

    bool load_from_vocab(const std::string& vocab_path) {
        std::ifstream f(vocab_path);
        if (!f.is_open()) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        return load_from_vocab_string(ss.str(), vocab_path.c_str());
    }

    bool load_from_merges(const std::string& merges_path) {
        std::ifstream f(merges_path);
        if (!f.is_open()) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        return load_from_merges_string(ss.str(), merges_path.c_str());
    }

    std::vector<int32_t> tokenize(const std::string& text, int max_length = MAX_LENGTH) const {
        std::vector<int32_t> tokens;
        tokens.push_back(BOS_TOKEN_ID);

        std::string clean = basic_clean(text);
        clean = whitespace_clean(clean);
        std::transform(clean.begin(), clean.end(), clean.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });

        auto words = pre_tokenize(clean);
        for (auto& word : words) {
            auto byte_encoded = bytes_to_unicode(word);
            auto bpe_tokens = bpe(byte_encoded);
            for (auto& t : bpe_tokens) {
                auto it = token_to_id_.find(t);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                }
            }
        }

        tokens.push_back(EOS_TOKEN_ID);

        // Pad or truncate
        if ((int)tokens.size() > max_length) {
            tokens.resize(max_length);
            tokens.back() = EOS_TOKEN_ID;
        }
        while ((int)tokens.size() < max_length) {
            tokens.push_back(PAD_TOKEN_ID);
        }

        return tokens;
    }

    bool is_loaded() const {
        return !bpe_merges_.empty() && !token_to_id_.empty();
    }

private:
    std::map<std::pair<std::string, std::string>, int> bpe_merges_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    bool load_from_vocab_string(const std::string& vocab_json, const char* source) {
        try {
            auto j = nlohmann::json::parse(vocab_json);
            token_to_id_.clear();
            id_to_token_.clear();
            bpe_cache_.clear();
            int32_t max_id = -1;
            for (auto& [tok, id] : j.items()) {
                int32_t tid = id.get<int32_t>();
                if (tid < 0 || tid >= VOCAB_SIZE) {
                    LOG_WARN("Rejecting CLIP vocab from %s: token id %d is outside CLIP range [0, %d)",
                             source, tid, VOCAB_SIZE);
                    token_to_id_.clear();
                    id_to_token_.clear();
                    return false;
                }
                token_to_id_[tok] = tid;
                id_to_token_[tid] = tok;
                max_id = std::max(max_id, tid);
            }
            if ((int32_t) token_to_id_.size() != VOCAB_SIZE ||
                token_to_id_.count("<|startoftext|>") == 0 ||
                token_to_id_.count("<|endoftext|>") == 0 ||
                max_id != VOCAB_SIZE - 1) {
                LOG_WARN("Rejecting CLIP vocab from %s: incompatible CLIP tokenizer layout", source);
                token_to_id_.clear();
                id_to_token_.clear();
                return false;
            }
            LOG_INFO("Loaded %zu CLIP vocab entries from %s", token_to_id_.size(), source);
            return !token_to_id_.empty();
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to parse CLIP vocab JSON from %s: %s", source, e.what());
            return false;
        }
    }

    bool load_from_merges_string(const std::string& merges_text, const char* source) {
        std::istringstream in(merges_text);
        bpe_merges_.clear();
        bpe_cache_.clear();

        std::string line;
        if (std::getline(in, line)) {
            while (!line.empty() && line.back() == '\r') line.pop_back();
        }

        int rank = 0;
        while (std::getline(in, line)) {
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line[0] == '#') continue;

            auto space = line.find(' ');
            if (space == std::string::npos) continue;

            std::string a = line.substr(0, space);
            std::string b = line.substr(space + 1);
            bpe_merges_[{a, b}] = rank++;
        }

        LOG_INFO("Loaded %d CLIP merges from %s", rank, source);
        return !bpe_merges_.empty();
    }

    static inline bool is_space(char32_t cp) {
        return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' || cp == U'\v' || cp == U'\f';
    }

    static inline bool is_letter(char32_t cp) {
        if (cp < 128) return std::isalpha((unsigned char)cp) != 0;
        return true;
    }

    static inline bool is_digit(char32_t cp) {
        if (cp < 128) return std::isdigit((unsigned char)cp) != 0;
        return false;
    }

    static inline bool is_symbol(char32_t cp) {
        return !is_space(cp) && !is_letter(cp) && !is_digit(cp);
    }

    static inline char32_t lower_ascii(char32_t cp) {
        if (cp >= U'A' && cp <= U'Z') return cp - U'A' + U'a';
        return cp;
    }

    static bool try_match_contraction(const std::u32string& cps, size_t i, size_t* out_len) {
        if (i >= cps.size() || cps[i] != U'\'') return false;
        const size_t n = cps.size() - i;
        if (n >= 2) {
            char32_t c1 = lower_ascii(cps[i + 1]);
            if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') {
                *out_len = 2;
                return true;
            }
        }
        if (n >= 3) {
            char32_t c1 = lower_ascii(cps[i + 1]);
            char32_t c2 = lower_ascii(cps[i + 2]);
            if ((c1 == U'r' && c2 == U'e') ||
                (c1 == U'v' && c2 == U'e') ||
                (c1 == U'l' && c2 == U'l')) {
                *out_len = 3;
                return true;
            }
        }
        return false;
    }

    void init_byte_encoder() {
        std::vector<int> bs;
        for (int b = 33; b <= 126; b++) bs.push_back(b);
        for (int b = 161; b <= 172; b++) bs.push_back(b);
        for (int b = 174; b <= 255; b++) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        byte_encoder_.clear();
        for (size_t i = 0; i < bs.size(); i++) {
            std::u32string u(1, (char32_t)cs[i]);
            byte_encoder_[(uint8_t)bs[i]] = utf32_to_utf8(u);
        }
    }

    static std::string basic_clean(const std::string& text) {
        return text;
    }

    static std::string whitespace_clean(const std::string& text) {
        std::string out;
        out.reserve(text.size());
        bool prev_space = true;
        for (char c : text) {
            bool sp = std::isspace((unsigned char)c) != 0;
            if (sp) {
                if (!prev_space) out.push_back(' ');
            } else {
                out.push_back(c);
            }
            prev_space = sp;
        }
        if (!out.empty() && out.back() == ' ') out.pop_back();
        return out;
    }

    std::string bytes_to_unicode(const std::string& text) const {
        std::string out;
        out.reserve(text.size() * 2);
        for (uint8_t b : std::vector<uint8_t>(text.begin(), text.end())) {
            auto it = byte_encoder_.find(b);
            if (it != byte_encoder_.end()) out += it->second;
        }
        return out;
    }

    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> out;
        auto cps = utf8_to_utf32(text);
        const size_t n = cps.size();
        if (n == 0) return out;

        std::vector<size_t> offsets(n + 1, 0);
        size_t byte_off = 0;
        for (size_t i = 0; i < n; i++) {
            std::u32string one(1, cps[i]);
            byte_off += utf32_to_utf8(one).size();
            offsets[i + 1] = byte_off;
        }
        auto slice = [&](size_t a, size_t b) {
            return text.substr(offsets[a], offsets[b] - offsets[a]);
        };

        auto starts_with_at = [&](size_t pos, const std::string& s) -> bool {
            size_t byte_pos = offsets[pos];
            return byte_pos + s.size() <= text.size() && text.compare(byte_pos, s.size(), s) == 0;
        };

        size_t i = 0;
        while (i < n) {
            if (starts_with_at(i, "<|startoftext|>")) {
                size_t j = i;
                while (j < n && offsets[j] < offsets[i] + 15) j++;
                out.push_back("<|startoftext|>");
                i = j;
                continue;
            }
            if (starts_with_at(i, "<|endoftext|>")) {
                size_t j = i;
                while (j < n && offsets[j] < offsets[i] + 13) j++;
                out.push_back("<|endoftext|>");
                i = j;
                continue;
            }

            size_t len = 0;
            if (try_match_contraction(cps, i, &len)) {
                out.push_back(slice(i, i + len));
                i += len;
                continue;
            }

            if (is_letter(cps[i])) {
                size_t j = i + 1;
                while (j < n && is_letter(cps[j])) j++;
                out.push_back(slice(i, j));
                i = j;
                continue;
            }

            if (is_digit(cps[i])) {
                out.push_back(slice(i, i + 1));
                i++;
                continue;
            }

            if (is_symbol(cps[i])) {
                size_t j = i + 1;
                while (j < n && is_symbol(cps[j])) j++;
                out.push_back(slice(i, j));
                i = j;
                continue;
            }

            // Skip whitespace
            i++;
        }

        return out;
    }

    std::vector<std::string> split_utf8_chars(const std::string& s) const {
        std::vector<std::string> out;
        auto cps = utf8_to_utf32(s);
        out.reserve(cps.size());
        for (char32_t cp : cps) {
            out.push_back(utf32_to_utf8(std::u32string(1, cp)));
        }
        return out;
    }

    std::vector<std::string> bpe(const std::string& token) const {
        auto cache_it = bpe_cache_.find(token);
        if (cache_it != bpe_cache_.end()) return cache_it->second;

        auto word = split_utf8_chars(token);
        if (word.empty()) return {};
        // CLIP uses end-of-word suffix on the last symbol.
        word.back() += "</w>";

        while (word.size() > 1) {
            int best_rank = INT32_MAX;
            int best_idx = -1;
            for (size_t i = 0; i + 1 < word.size(); i++) {
                auto it = bpe_merges_.find({word[i], word[i + 1]});
                if (it != bpe_merges_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = (int)i;
                }
            }
            if (best_idx < 0) break;

            word[(size_t)best_idx] += word[(size_t)best_idx + 1];
            word.erase(word.begin() + best_idx + 1);
        }

        bpe_cache_[token] = word;
        return word;
    }
};

// ── CLIP Model Components ────────────────────────────────────────────

class CLIPEmbeddings : public Module {
public:
    int64_t vocab_size, embed_dim, max_position;

    CLIPEmbeddings() : vocab_size(49408), embed_dim(768), max_position(77) {}
    CLIPEmbeddings(int64_t vocab, int64_t dim, int64_t max_pos = 77)
        : vocab_size(vocab), embed_dim(dim), max_position(max_pos) {}

protected:
    void init_params(ggml_context* ctx, ggml_type wtype, const std::string& prefix) override {
        params["token_embedding.weight"]    = ggml_new_tensor_2d(ctx, wtype, embed_dim, vocab_size);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, max_position);
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* input_ids, ggml_tensor* position_ids) {
        auto* tok_emb = ggml_get_rows(fctx->ctx, params["token_embedding.weight"], input_ids);
        auto* pos_emb = ggml_get_rows(fctx->ctx, params["position_embedding.weight"], position_ids);
        return ggml_add(fctx->ctx, tok_emb, pos_emb);
    }
};

class CLIPMLP : public Module {
public:
    Linear fc1, fc2;

    CLIPMLP() = default;
    CLIPMLP(int64_t dim, int64_t intermediate) : fc1(dim, intermediate), fc2(intermediate, dim) {
        submodules["fc1"] = std::make_shared<Linear>(fc1);
        submodules["fc2"] = std::make_shared<Linear>(fc2);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        x = std::static_pointer_cast<Linear>(submodules["fc1"])->forward(fctx, x);
        x = ggml_gelu(fctx->ctx, x);
        x = std::static_pointer_cast<Linear>(submodules["fc2"])->forward(fctx, x);
        return x;
    }
};

class CLIPLayer : public Module {
public:
    LayerNorm layer_norm1, layer_norm2;
    // Self-attention weights as params
    int64_t embed_dim, num_heads;

    CLIPLayer() : embed_dim(768), num_heads(12) {}
    CLIPLayer(int64_t dim, int64_t heads, int64_t intermediate)
        : embed_dim(dim), num_heads(heads),
          layer_norm1(dim), layer_norm2(dim) {
        submodules["layer_norm1"] = std::make_shared<LayerNorm>(layer_norm1);
        submodules["layer_norm2"] = std::make_shared<LayerNorm>(layer_norm2);

        // Self-attention Q/K/V/Out
        submodules["self_attn.q_proj"] = std::make_shared<Linear>(dim, dim);
        submodules["self_attn.k_proj"] = std::make_shared<Linear>(dim, dim);
        submodules["self_attn.v_proj"] = std::make_shared<Linear>(dim, dim);
        submodules["self_attn.out_proj"] = std::make_shared<Linear>(dim, dim);

        // MLP
        submodules["mlp"] = std::make_shared<CLIPMLP>(dim, intermediate);
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        auto* ctx = fctx->ctx;

        // Self-attention with pre-norm
        auto* residual = x;
        x = std::static_pointer_cast<LayerNorm>(submodules["layer_norm1"])->forward(fctx, x);

        auto* q = std::static_pointer_cast<Linear>(submodules["self_attn.q_proj"])->forward(fctx, x);
        auto* k = std::static_pointer_cast<Linear>(submodules["self_attn.k_proj"])->forward(fctx, x);
        auto* v = std::static_pointer_cast<Linear>(submodules["self_attn.v_proj"])->forward(fctx, x);

        x = ops::attention(ctx, fctx->backend, q, k, v, num_heads, nullptr, fctx->flash_attn, true);
        x = std::static_pointer_cast<Linear>(submodules["self_attn.out_proj"])->forward(fctx, x);
        x = ggml_add(ctx, x, residual);

        // MLP with pre-norm
        residual = x;
        x = std::static_pointer_cast<LayerNorm>(submodules["layer_norm2"])->forward(fctx, x);
        x = std::static_pointer_cast<CLIPMLP>(submodules["mlp"])->forward(fctx, x);
        x = ggml_add(ctx, x, residual);

        return x;
    }
};

class CLIPEncoder : public Module {
public:
    int num_layers;

    CLIPEncoder() : num_layers(0) {}
    CLIPEncoder(int64_t dim, int64_t heads, int64_t intermediate, int layers)
        : num_layers(layers) {
        for (int i = 0; i < layers; i++) {
            submodules["layers." + std::to_string(i)] =
                std::make_shared<CLIPLayer>(dim, heads, intermediate);
        }
    }

    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        for (int i = 0; i < num_layers; i++) {
            x = std::static_pointer_cast<CLIPLayer>(
                submodules["layers." + std::to_string(i)])->forward(fctx, x);
        }
        return x;
    }
};

// ── CLIP Text Model (full model) ────────────────────────────────────
// CLIP-ViT-L/14: embed_dim=768, heads=12, layers=12, intermediate=3072

class CLIPTextModel : public Executor {
public:
    CLIPEmbeddings embeddings;
    CLIPEncoder    encoder;
    LayerNorm      final_layer_norm;

    int64_t embed_dim;
    int     num_layers;

    CLIPTextModel(ggml_backend_t backend,
                  int64_t dim = 768, int heads = 12, int layers = 12,
                  int64_t intermediate = 3072, int64_t vocab = 49408, int64_t max_pos = 77)
        : Executor(backend),
          embeddings(vocab, dim, max_pos),
          encoder(dim, heads, intermediate, layers),
          final_layer_norm(dim),
          embed_dim(dim), num_layers(layers) {
    }

    std::string get_desc() override { return "CLIPTextModel"; }

    bool init(ggml_type wtype = GGML_TYPE_F16) {
        auto* ctx = params_ctx();
        if (!ctx) {
            LOG_ERROR("CLIPTextModel: params context is not initialized");
            return false;
        }
        embeddings.init(ctx, wtype, "text_model.embeddings");
        encoder.init(ctx, wtype, "text_model.encoder");
        final_layer_norm.init(ctx, GGML_TYPE_F32, "text_model.final_layer_norm");
        return alloc_params_buffer();
    }

    void collect_all_params(std::map<std::string, ggml_tensor*>& tensors) {
        embeddings.collect_params(tensors, "text_model.embeddings");
        encoder.collect_params(tensors, "text_model.encoder");
        final_layer_norm.collect_params(tensors, "text_model.final_layer_norm");
    }

    // Forward: returns pooler_output [batch, embed_dim]
    // input_ids: [batch, seq_len] int32
    // eos_pos: position of the EOS token to pool from (-1 = auto-detect)
    bool forward(ggml_tensor* input_ids, int n_threads,
                 ggml_tensor** pooled_output, ggml_context* out_ctx,
                 int eos_pos = -1) {
        int seq_len = (int)input_ids->ne[0];

        // Find EOS position if not provided
        if (eos_pos < 0) {
            int32_t* ids = (int32_t*)input_ids->data;
            for (int i = 1; i < seq_len; i++) {
                if (ids[i] == CLIPTokenizer::EOS_TOKEN_ID) { eos_pos = i; break; }
            }
            if (eos_pos < 0) eos_pos = seq_len - 1;
        }

        // Position IDs: 0, 1, 2, ..., seq_len-1 (persistent through compute)
        std::vector<int32_t> pos_vec(seq_len);
        for (int i = 0; i < seq_len; i++) pos_vec[i] = i;

        auto build_graph = [&]() -> ggml_cgraph* {
            auto* ctx = compute_ctx_;
            auto* gf = ggml_new_graph_custom(ctx, MAX_GRAPH_SIZE, false);

            ForwardContext fctx = get_forward_ctx();

            auto* ids = to_backend(input_ids);

            auto* pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
            set_backend_tensor_data(pos_ids, pos_vec.data());

            auto* x = embeddings.forward(&fctx, ids, pos_ids);
            x = encoder.forward(&fctx, x);
            x = final_layer_norm.forward(&fctx, x);

            // Pooled output: take the EOS token embedding
            int64_t byte_offset = (int64_t)eos_pos * x->nb[1];
            auto* pooled = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], byte_offset);

            ggml_build_forward_expand(gf, pooled);
            return gf;
        };

        return compute(build_graph, n_threads, true, pooled_output, out_ctx);
    }
};

#endif // KD_CLIP_HPP
