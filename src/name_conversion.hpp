#ifndef KD_NAME_CONVERSION_HPP
#define KD_NAME_CONVERSION_HPP

#include <string>
#include <map>
#include <vector>

// ── K5 tensor name mapping ───────────────────────────────────────────
// Maps safetensors tensor names (from HuggingFace K5 checkpoints) to
// our internal Module tree names.
//
// For K5, the safetensors names already match our module hierarchy directly,
// so the main job is:
// 1. Stripping any diffusers-style prefixes
// 2. Handling VAE name differences
// 3. Handling CLIP/Qwen name differences

namespace name_conversion {

// DiT tensor names: safetensors names match our structure directly
// e.g. "text_transformer_blocks.0.text_modulation.out_layer.weight"
//      "visual_transformer_blocks.0.visual_modulation.out_layer.weight"
//      "time_embeddings.in_layer.weight"
//      "text_embeddings.in_layer.weight"
//      "pooled_text_embeddings.in_layer.weight"
//      "visual_embeddings.in_layer.weight"
//      "out_layer.modulation.out_layer.weight"
//      "out_layer.norm.weight"
//      "out_layer.out_layer.weight"

inline std::string convert_dit_name(const std::string& name) {
    // K5 DiT names already match our module hierarchy
    return name;
}

// CLIP tensor names from HuggingFace
// HF: "text_model.encoder.layers.0.self_attn.q_proj.weight"
// Our: "text_model.encoder.layers.0.self_attn.q_proj.weight" (same)
inline std::string convert_clip_name(const std::string& name) {
    return name;
}

// Qwen tensor names from HuggingFace
// HF: "model.layers.0.self_attn.q_proj.weight"
// Our: "model.layers.0.self_attn.q_proj.weight"
// Note: Qwen uses different attention naming
inline std::string convert_qwen_name(const std::string& name) {
    std::string result = name;

    // Qwen attention: HF uses q_proj, k_proj, v_proj, o_proj (matches ours)
    // Qwen MLP: HF uses gate_proj, up_proj, down_proj (matches ours)
    return result;
}

// VAE tensor names
// HF FLUX VAE: "encoder.down_blocks.0.resnets.0.norm1.weight"
// Our: "encoder.down.0.block.0.norm1.weight"
inline std::string convert_vae_name(const std::string& name) {
    std::string result = name;

    // Convert HF diffusers-style VAE names to our naming convention
    // "down_blocks.X.resnets.Y" → "down.X.block.Y"
    // "up_blocks.X.resnets.Y" → "up.X.block.Y"
    // "down_blocks.X.downsamplers.0.conv" → "down.X.downsample.conv"
    // "up_blocks.X.upsamplers.0.conv" → "up.X.upsample.conv"
    // "mid_block.resnets.0" → "mid.block_1"
    // "mid_block.attentions.0" → "mid.attn_1"

    auto replace = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.length(), to);
            pos += to.length();
        }
    };

    replace(result, "down_blocks.", "down.");
    replace(result, "up_blocks.", "up.");
    replace(result, "resnets.", "block.");
    replace(result, "downsamplers.0.conv", "downsample.conv");
    replace(result, "upsamplers.0.conv", "upsample.conv");
    replace(result, "mid_block.resnets.0", "mid.block_1");
    replace(result, "mid_block.resnets.1", "mid.block_2");
    replace(result, "mid_block.attentions.0", "mid.attn_1");
    replace(result, "attentions.0.group_norm", "mid.attn_1.norm");

    return result;
}

// VAE3D tensor names — already converted by Python converter, pass through
inline std::string convert_vae3d_name(const std::string& name) {
    return name;
}

// Auto-detect and convert based on prefix
inline std::string convert_name(const std::string& name,
                                 const std::string& model_type = "") {
    if (model_type == "dit" || model_type.empty()) {
        // Check if it looks like a DiT tensor
        if (name.find("transformer_blocks") != std::string::npos ||
            name.find("time_embeddings") != std::string::npos ||
            name.find("text_embeddings") != std::string::npos ||
            name.find("visual_embeddings") != std::string::npos ||
            name.find("out_layer") != std::string::npos) {
            return convert_dit_name(name);
        }
    }
    if (model_type == "clip" || model_type.empty()) {
        if (name.find("text_model") != std::string::npos) {
            return convert_clip_name(name);
        }
    }
    if (model_type == "qwen" || model_type.empty()) {
        if (name.find("model.layers") != std::string::npos ||
            name.find("model.embed_tokens") != std::string::npos ||
            name.find("model.norm") != std::string::npos) {
            return convert_qwen_name(name);
        }
    }
    if (model_type == "vae" || model_type.empty()) {
        if (name.find("encoder") != std::string::npos ||
            name.find("decoder") != std::string::npos) {
            return convert_vae_name(name);
        }
    }
    if (model_type == "vae3d") {
        return convert_vae3d_name(name);
    }
    return name;
}

} // namespace name_conversion

#endif // KD_NAME_CONVERSION_HPP
