#ifndef KD_QWEN_TOKENIZER_HPP
#define KD_QWEN_TOKENIZER_HPP

#include <algorithm>
#include <array>
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

// ── Qwen2.5 BPE Tokenizer ───────────────────────────────────────────
// GPT-2 style BPE with Qwen2 pre-tokenizer regex
// Vocab and merges loaded from files or GGUF metadata

class QwenTokenizer {
public:
    // Special tokens for Qwen2.5
    static constexpr int IM_START_ID = 151644;  // <|im_start|>
    static constexpr int IM_END_ID   = 151645;  // <|im_end|>
    static constexpr int EOS_ID      = 151643;  // <|endoftext|>
    static constexpr int PAD_ID      = 151643;

    QwenTokenizer() { init_byte_encoder(); }

    bool load(const std::string& vocab_path, const std::string& merges_path) {
        token_to_id_.clear();
        id_to_token_.clear();
        merges_.clear();
        bpe_cache_.clear();
        if (!load_vocab(vocab_path)) return false;
        if (!load_merges(merges_path)) return false;
        return true;
    }

    bool load_from_memory(const std::string& vocab_json, const std::string& merges_text) {
        token_to_id_.clear();
        id_to_token_.clear();
        merges_.clear();
        bpe_cache_.clear();
        if (!load_vocab_json_string(vocab_json, "<embedded>")) return false;
        if (!load_merges_string(merges_text, "<embedded>")) return false;
        return true;
    }

    bool load_vocab(const std::string& path) {
        // Try JSON format first (HuggingFace vocab.json: {"token": id, ...})
        if (ends_with(path, ".json")) {
            return load_vocab_json(path);
        }
        // Fall back to simple text format (one token per line)
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_ERROR("Cannot open vocab file: %s", path.c_str());
            return false;
        }

        token_to_id_.clear();
        id_to_token_.clear();
        bpe_cache_.clear();

        std::string tok;
        int id = 0;
        while (std::getline(f, tok)) {
            while (!tok.empty() && tok.back() == '\r') tok.pop_back();
            if (tok.empty()) continue;
            token_to_id_[tok] = id;
            id_to_token_[id] = tok;
            id++;
        }
        LOG_INFO("Loaded %d vocab entries", id);
        return true;
    }

    bool load_vocab_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_ERROR("Cannot open vocab JSON: %s", path.c_str());
            return false;
        }
        return load_vocab_json_stream(f, path.c_str());
    }

    bool load_merges(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_ERROR("Cannot open merges file: %s", path.c_str());
            return false;
        }
        return load_merges_stream(f, path.c_str());
    }

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text) const {
        auto words = pre_tokenize(text);
        std::vector<int32_t> tokens;
        for (auto& word : words) {
            auto byte_encoded = bytes_to_unicode(word);
            auto bpe_tokens = bpe(byte_encoded);
            for (auto& tok : bpe_tokens) {
                auto it = token_to_id_.find(tok);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                } else {
                    // Fallback: byte-level IDs
                    for (uint8_t b : std::vector<uint8_t>(tok.begin(), tok.end())) {
                        auto enc_it = byte_encoder_.find(b);
                        if (enc_it == byte_encoder_.end()) continue;
                        auto bit = token_to_id_.find(enc_it->second);
                        if (bit != token_to_id_.end()) {
                            tokens.push_back(bit->second);
                        }
                    }
                }
            }
        }
        return tokens;
    }

    // Encode with Qwen chat template for K5
    std::vector<int32_t> encode_for_k5(const std::string& prompt,
                                        const std::string& content_type = "image",
                                        int max_length = 512) const {
        std::string system_text;
        if (content_type == "video") {
            system_text =
                "system\n"
                "You are a promt engineer. Describe the video in detail.\n"
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\n"
                "Describe the location of the video, main characters or objects and their action.\n"
                "Describe the dynamism of the video and presented actions.\n"
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.\n"
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.\n"
                "Pay attention to the order of key actions shown in the scene.";
        } else if (content_type == "image2video") {
            system_text =
                "system\n"
                "You are a promt engineer. Your task is to create a highly detailed and effective video description based on a provided input image.\n"
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\n"
                "Describe main characters actions.\n"
                "Describe the dynamism of the video and presented actions.\n"
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.\n"
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.\n"
                "Pay attention to the order of key actions shown in the scene.";
        } else if (content_type == "image_edit") {
            system_text =
                "system\n"
                "You are a promt engineer. Based on the provided source image (first image) and target image (second image), create an interesting text prompt that can be used together with the source image to create the target image:";
        } else {
            system_text =
                "system\n"
                "You are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";
        }

        std::vector<int32_t> tokens;
        tokens.reserve(768);

        auto append_encoded = [&](const std::string& text) {
            auto t = encode(text);
            tokens.insert(tokens.end(), t.begin(), t.end());
        };

        // <|im_start|>system ... <|im_end|>
        tokens.push_back(IM_START_ID);
        append_encoded(system_text);
        tokens.push_back(IM_END_ID);

        // newline between turns
        append_encoded("\n");

        // <|im_start|>user ... <|im_end|>
        tokens.push_back(IM_START_ID);
        append_encoded("user\n" + prompt);
        tokens.push_back(IM_END_ID);

        // Match upstream truncation semantics: max_length applies after crop_start.
        const int total_max = max_length + crop_start(content_type);
        if (total_max > 0 && (int)tokens.size() > total_max) {
            tokens.resize(total_max);
        }
        return tokens;
    }

    int crop_start(const std::string& content_type) const {
        if (content_type == "video") return 129;
        if (content_type == "image") return 41;
        if (content_type == "image_edit") return 55;
        if (content_type == "image2video") return 132;
        return 41;
    }

    std::string decode(const std::vector<int32_t>& ids) const {
        std::string result;
        for (auto id : ids) {
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) result += it->second;
        }
        return result;
    }

    bool is_loaded() const { return !token_to_id_.empty() && !merges_.empty(); }

    int vocab_size() const { return (int)token_to_id_.size(); }

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merges_;
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    bool load_vocab_json_string(const std::string& vocab_json, const char* source) {
        try {
            auto j = nlohmann::json::parse(vocab_json);
            token_to_id_.clear();
            id_to_token_.clear();
            bpe_cache_.clear();
            for (auto& [token, id] : j.items()) {
                int32_t tid = id.get<int32_t>();
                token_to_id_[token] = tid;
                id_to_token_[tid] = token;
            }
            LOG_INFO("Loaded %zu vocab entries from %s", token_to_id_.size(), source);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to parse vocab JSON from %s: %s", source, e.what());
            return false;
        }
    }

    bool load_vocab_json_stream(std::istream& in, const char* source) {
        std::ostringstream ss;
        ss << in.rdbuf();
        return load_vocab_json_string(ss.str(), source);
    }

    bool load_merges_string(const std::string& merges_text, const char* source) {
        std::istringstream in(merges_text);
        return load_merges_stream(in, source);
    }

    bool load_merges_stream(std::istream& in, const char* source) {
        merges_.clear();
        bpe_cache_.clear();

        std::string l;
        int rank = 0;
        while (std::getline(in, l)) {
            while (!l.empty() && l.back() == '\r') l.pop_back();
            if (l.empty()) continue;
            if (l[0] == '#') continue;

            auto space = l.find(' ');
            if (space == std::string::npos) continue;

            std::string a = l.substr(0, space);
            std::string b = l.substr(space + 1);
            merges_[{a, b}] = rank++;
        }
        LOG_INFO("Loaded %d merges from %s", rank, source);
        return !merges_.empty();
    }

    static inline bool is_newline(char32_t cp) {
        return cp == U'\n' || cp == U'\r';
    }

    static inline bool is_space(char32_t cp) {
        return cp == U' ' || cp == U'\t' || cp == U'\v' || cp == U'\f' || is_newline(cp);
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

    static inline bool is_non_crlf_non_alnum(char32_t cp) {
        return cp != U'\n' && cp != U'\r' && !is_letter(cp) && !is_digit(cp);
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
        // GPT-2 bytes_to_unicode mapping used by Qwen/CLIP ByteLevel BPE.
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

        size_t i = 0;
        while (i < n) {
            size_t len = 0;
            if (try_match_contraction(cps, i, &len)) {
                out.push_back(slice(i, i + len));
                i += len;
                continue;
            }

            if (is_letter(cps[i]) || (is_non_crlf_non_alnum(cps[i]) && i + 1 < n && is_letter(cps[i + 1]))) {
                size_t start = i;
                if (!is_letter(cps[i])) i++;
                while (i < n && is_letter(cps[i])) i++;
                out.push_back(slice(start, i));
                continue;
            }

            if (is_digit(cps[i])) {
                out.push_back(slice(i, i + 1));
                i++;
                continue;
            }

            if ((cps[i] == U' ' && i + 1 < n && is_symbol(cps[i + 1])) || is_symbol(cps[i])) {
                size_t start = i;
                if (cps[i] == U' ' && i + 1 < n && is_symbol(cps[i + 1])) i++;
                while (i < n && is_symbol(cps[i])) i++;
                while (i < n && is_newline(cps[i])) i++;
                out.push_back(slice(start, i));
                continue;
            }

            if (is_space(cps[i])) {
                size_t j = i;
                while (j < n && is_space(cps[j]) && !is_newline(cps[j])) j++;
                size_t k = j;
                while (k < n && is_newline(cps[k])) k++;
                if (k > j) {
                    out.push_back(slice(i, k));
                    i = k;
                    continue;
                }
            }

            if (is_space(cps[i])) {
                size_t j = i;
                while (j < n && is_space(cps[j])) j++;
                bool trailing_only = true;
                for (size_t k = j; k < n; k++) {
                    if (!is_space(cps[k])) {
                        trailing_only = false;
                        break;
                    }
                }
                if (trailing_only) {
                    out.push_back(slice(i, j));
                    i = j;
                    continue;
                }
            }

            if (is_space(cps[i])) {
                size_t j = i;
                while (j < n && is_space(cps[j])) j++;
                out.push_back(slice(i, j));
                i = j;
                continue;
            }

            out.push_back(slice(i, i + 1));
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

        while (word.size() > 1) {
            int best_rank = INT32_MAX;
            int best_idx = -1;
            for (size_t i = 0; i + 1 < word.size(); i++) {
                auto it = merges_.find({word[i], word[i + 1]});
                if (it != merges_.end() && it->second < best_rank) {
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

#endif // KD_QWEN_TOKENIZER_HPP
