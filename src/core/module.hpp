#ifndef KD_MODULE_HPP
#define KD_MODULE_HPP

#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "core/ops.hpp"

// Forward declarations
struct TensorStorage;
template<typename K, typename V> class OrderedMap;
typedef OrderedMap<std::string, TensorStorage> String2TensorStorage;

// ── Forward context passed during graph building ─────────────────────

struct ForwardContext {
    ggml_backend_t backend       = nullptr;
    ggml_context*  ctx           = nullptr;
    bool           flash_attn    = false;
    ggml_tensor*   one           = nullptr;  // pre-built scalar 1.0f tensor
    std::function<void(ggml_tensor*, const void*)> set_tensor_data;
};

// ── Module base class (adapted from sd.cpp GGMLBlock) ────────────────

class Module {
protected:
    typedef std::unordered_map<std::string, ggml_tensor*>            ParamMap;
    typedef std::unordered_map<std::string, std::shared_ptr<Module>> SubmoduleMap;

    SubmoduleMap submodules;
    ParamMap     params;

    virtual void init_params(ggml_context* ctx, ggml_type default_type, const std::string& prefix) {}

    void init_submodules(ggml_context* ctx, ggml_type default_type, const std::string& prefix) {
        for (auto& [name, mod] : submodules) {
            mod->init(ctx, default_type, prefix + name);
        }
    }

public:
    virtual ~Module() = default;

    void init(ggml_context* ctx, ggml_type default_type = GGML_TYPE_F16, std::string prefix = "") {
        if (!prefix.empty()) prefix += ".";
        init_params(ctx, default_type, prefix);
        init_submodules(ctx, default_type, prefix);
    }

    size_t param_count() const {
        size_t n = params.size();
        for (auto& [_, mod] : submodules) n += mod->param_count();
        return n;
    }

    size_t param_mem_size() const {
        size_t mem = 0;
        for (auto& [_, t] : params) mem += ggml_nbytes(t);
        for (auto& [_, mod] : submodules) mem += mod->param_mem_size();
        return mem;
    }

    void collect_params(std::map<std::string, ggml_tensor*>& out, std::string prefix = "") const {
        if (!prefix.empty()) prefix += ".";
        for (auto& [name, mod] : submodules) {
            mod->collect_params(out, prefix + name);
        }
        for (auto& [name, t] : params) {
            out[prefix + name] = t;
        }
    }
};

// ── Linear ───────────────────────────────────────────────────────────

class Linear : public Module {
public:
    int64_t in_features, out_features;
    bool    has_bias;

    Linear() : in_features(0), out_features(0), has_bias(true) {}
    Linear(int64_t in, int64_t out, bool bias = true)
        : in_features(in), out_features(out), has_bias(bias) {}

protected:
    void init_params(ggml_context* ctx, ggml_type wtype, const std::string& prefix) override {
        if (in_features % ggml_blck_size(wtype) != 0) wtype = GGML_TYPE_F32;
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (has_bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
        }
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        return ops::linear(fctx->ctx, x, params["weight"],
                           has_bias ? params["bias"] : nullptr);
    }
};

// ── Embedding ────────────────────────────────────────────────────────

class Embedding : public Module {
public:
    int64_t num_embeddings, embedding_dim;

    Embedding() : num_embeddings(0), embedding_dim(0) {}
    Embedding(int64_t num, int64_t dim) : num_embeddings(num), embedding_dim(dim) {}

protected:
    void init_params(ggml_context* ctx, ggml_type wtype, const std::string& prefix) override {
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* ids) {
        return ggml_get_rows(fctx->ctx, params["weight"], ids);
    }
};

// ── LayerNorm ────────────────────────────────────────────────────────

class LayerNorm : public Module {
public:
    int64_t dim;
    float   eps;
    bool    affine;

    LayerNorm() : dim(0), eps(1e-5f), affine(true) {}
    LayerNorm(int64_t dim, float eps = 1e-5f, bool affine = true)
        : dim(dim), eps(eps), affine(affine) {}

protected:
    void init_params(ggml_context* ctx, ggml_type, const std::string& prefix) override {
        if (affine) {
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        }
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        return ops::layer_norm(fctx->ctx, x,
                               affine ? params["weight"] : nullptr,
                               affine ? params["bias"] : nullptr,
                               eps);
    }
};

// ── RMSNorm ──────────────────────────────────────────────────────────

class RMSNorm : public Module {
public:
    int64_t dim;
    float   eps;

    RMSNorm() : dim(0), eps(1e-6f) {}
    RMSNorm(int64_t dim, float eps = 1e-6f) : dim(dim), eps(eps) {}

protected:
    void init_params(ggml_context* ctx, ggml_type, const std::string& prefix) override {
        params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        return ops::rms_norm(fctx->ctx, x, params["weight"], eps);
    }
};

// ── GroupNorm ─────────────────────────────────────────────────────────

class GroupNorm : public Module {
public:
    int64_t num_channels;
    int     num_groups;
    float   eps;

    GroupNorm() : num_channels(0), num_groups(32), eps(1e-6f) {}
    GroupNorm(int groups, int64_t channels, float eps = 1e-6f)
        : num_channels(channels), num_groups(groups), eps(eps) {}

protected:
    void init_params(ggml_context* ctx, ggml_type, const std::string& prefix) override {
        params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        return ops::group_norm(fctx->ctx, x, params["weight"], params["bias"],
                               num_groups, eps);
    }
};

// ── Conv2d ───────────────────────────────────────────────────────────

class Conv2d : public Module {
public:
    int64_t in_channels, out_channels;
    int kh, kw, sh, sw, ph, pw, dh, dw;
    bool has_bias;

    Conv2d() : in_channels(0), out_channels(0),
               kh(1), kw(1), sh(1), sw(1), ph(0), pw(0), dh(1), dw(1),
               has_bias(true) {}

    Conv2d(int64_t in_ch, int64_t out_ch,
           std::pair<int,int> kernel = {1,1},
           std::pair<int,int> stride = {1,1},
           std::pair<int,int> padding = {0,0},
           std::pair<int,int> dilation = {1,1},
           bool bias = true)
        : in_channels(in_ch), out_channels(out_ch),
          kh(kernel.first), kw(kernel.second),
          sh(stride.first), sw(stride.second),
          ph(padding.first), pw(padding.second),
          dh(dilation.first), dw(dilation.second),
          has_bias(bias) {}

protected:
    void init_params(ggml_context* ctx, ggml_type, const std::string& prefix) override {
        params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kw, kh, in_channels, out_channels);
        if (has_bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    ggml_tensor* forward(ForwardContext* fctx, ggml_tensor* x) {
        return ops::conv2d(fctx->ctx, x, params["weight"],
                           has_bias ? params["bias"] : nullptr,
                           sw, sh, pw, ph, dw, dh);
    }
};

#endif // KD_MODULE_HPP
