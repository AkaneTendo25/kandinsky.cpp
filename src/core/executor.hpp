#ifndef KD_EXECUTOR_HPP
#define KD_EXECUTOR_HPP

#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef KD_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef KD_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef KD_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef KD_USE_OPENCL
#include "ggml-opencl.h"
#endif

#include "core/module.hpp"
#include "util.hpp"

#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE        1048576

// ── Executor: manages backend, buffers, and graph computation ────────
// Adapted from sd.cpp GGMLRunner

class Executor {
public:
    typedef std::function<ggml_cgraph*()> GraphBuilder;

    Executor(ggml_backend_t backend, bool offload_params_to_cpu = false)
        : runtime_backend_(backend) {
        if (!ggml_backend_is_cpu(runtime_backend_) && offload_params_to_cpu) {
            params_backend_ = ggml_backend_cpu_init();
            if (!params_backend_) {
                LOG_WARN("Executor: failed to initialize CPU params backend, using runtime backend");
                params_backend_ = runtime_backend_;
            }
        } else {
            params_backend_ = runtime_backend_;
        }
        if (!alloc_params_ctx()) {
            LOG_ERROR("Executor: failed to initialize parameter contexts");
        }
    }

    virtual ~Executor() {
        free_params_buffer();
        free_compute_buffer();
        if (sched_) { ggml_backend_sched_free(sched_); sched_ = nullptr; }
        free_params_ctx();
        free_compute_ctx();
        if (compute_buf_) { free(compute_buf_); compute_buf_ = nullptr; }
        if (params_backend_ != runtime_backend_) {
            ggml_backend_free(params_backend_);
        }
        if (own_split_backend_ && split_backend_ &&
            split_backend_ != runtime_backend_ &&
            split_backend_ != params_backend_) {
            ggml_backend_free(split_backend_);
        }
    }

    virtual std::string get_desc() = 0;

    // ── Parameter context management ─────────────────────────────────

    ggml_context* params_ctx() { return params_ctx_; }

    // Enable a secondary CPU backend used for mixed CPU/GPU execution.
    // Returns runtime backend when runtime is already CPU.
    ggml_backend_t ensure_cpu_split_backend() {
        if (ggml_backend_is_cpu(runtime_backend_)) return runtime_backend_;
        if (!split_backend_) {
            split_backend_ = ggml_backend_cpu_init();
            own_split_backend_ = (split_backend_ != nullptr);
        }
        return split_backend_;
    }

    // Place a specific parameter tensor on a chosen backend.
    void set_param_tensor_backend(ggml_tensor* t, ggml_backend_t backend) {
        if (!t || !backend) return;
        param_tensor_backend_[t] = backend;
    }

    bool alloc_params_buffer() {
        if (!params_ctx_) {
            LOG_ERROR("%s: params context is not initialized", get_desc().c_str());
            return false;
        }
        if (!param_tensor_backend_.empty()) {
            free_params_buffer();

            std::map<ggml_backend_t, size_t> bytes_by_backend;
            size_t tensors_alloc = 0;

            for (auto* t = ggml_get_first_tensor(params_ctx_); t; t = ggml_get_next_tensor(params_ctx_, t)) {
                ggml_backend_t target = params_backend_;
                auto it = param_tensor_backend_.find(t);
                if (it != param_tensor_backend_.end()) {
                    target = it->second;
                }

                auto* buft = ggml_backend_get_default_buffer_type(target);
                size_t sz = ggml_backend_buft_get_alloc_size(buft, t);
                auto* buf = ggml_backend_buft_alloc_buffer(buft, sz);
                if (!buf) {
                    LOG_ERROR("%s: failed to allocate param tensor buffer", get_desc().c_str());
                    free_params_buffer();
                    return false;
                }
                ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                void* base = ggml_backend_buffer_get_base(buf);
                auto st = ggml_backend_tensor_alloc(buf, t, base);
                if (st != GGML_STATUS_SUCCESS) {
                    LOG_ERROR("%s: failed to bind param tensor buffer", get_desc().c_str());
                    ggml_backend_buffer_free(buf);
                    free_params_buffer();
                    return false;
                }
                param_buffers_.push_back(buf);
                bytes_by_backend[target] += sz;
                tensors_alloc++;
            }

            bool mixed = false;
            for (auto& kv : bytes_by_backend) {
                if (kv.first != runtime_backend_) { mixed = true; break; }
            }
            use_sched_ = mixed;

            if (use_sched_) {
                if (!split_backend_) {
                    LOG_ERROR("%s: mixed params requested but split backend is not initialized", get_desc().c_str());
                    return false;
                }
                if (!sched_) {
                    ggml_backend_t backends[2] = { runtime_backend_, split_backend_ };
                    sched_ = ggml_backend_sched_new(backends, nullptr, 2, MAX_GRAPH_SIZE, false, true);
                    if (!sched_) {
                        LOG_ERROR("%s: failed to create backend scheduler", get_desc().c_str());
                        return false;
                    }
                }
            }

            for (auto& kv : bytes_by_backend) {
                LOG_DEBUG("%s: params buffer %.2f MB (%s)", get_desc().c_str(),
                          kv.second / (1024.0 * 1024.0),
                          ggml_backend_is_cpu(kv.first) ? "RAM" : "VRAM");
            }
            LOG_DEBUG("%s: param tensors allocated individually: %zu", get_desc().c_str(), tensors_alloc);
            return true;
        }

        params_buffer_ = ggml_backend_alloc_ctx_tensors(params_ctx_, params_backend_);
        if (!params_buffer_) {
            LOG_ERROR("%s: failed to allocate params buffer", get_desc().c_str());
            return false;
        }
        size_t sz = ggml_backend_buffer_get_size(params_buffer_);
        LOG_DEBUG("%s: params buffer %.2f MB (%s)", get_desc().c_str(),
                  sz / (1024.0 * 1024.0),
                  ggml_backend_is_cpu(params_backend_) ? "RAM" : "VRAM");
        return true;
    }

    // ── Compute ──────────────────────────────────────────────────────

    bool compute(GraphBuilder build_graph, int n_threads,
                 bool free_immediately = true,
                 ggml_tensor** output = nullptr,
                 ggml_context* output_ctx = nullptr) {
        if (use_sched_) {
            reset_compute_ctx();
            auto* gf = get_compute_graph(build_graph);

            if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
                LOG_ERROR("%s: failed to alloc scheduled compute graph", get_desc().c_str());
                return false;
            }

            copy_backend_tensor_data();

            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched_); i++) {
                auto* b = ggml_backend_sched_get_backend(sched_, i);
                if (ggml_backend_is_cpu(b)) {
                    ggml_backend_cpu_set_n_threads(b, n_threads);
                }
            }

            auto status = ggml_backend_sched_graph_compute(sched_, gf);
            if (status != GGML_STATUS_SUCCESS) {
                LOG_ERROR("%s: scheduled compute failed", get_desc().c_str());
                return false;
            }

            if (output) {
                auto* result = ggml_get_tensor(compute_ctx_, "result");
                if (*output == nullptr && output_ctx) {
                    *output = ggml_dup_tensor(output_ctx, result);
                }
                if (*output) {
                    ggml_backend_tensor_get(result, (*output)->data, 0, ggml_nbytes(*output));
                }
            }

            if (free_immediately) free_compute_buffer();
            return true;
        }

        if (!offload_params_to_runtime()) return false;
        if (!alloc_compute_buffer(build_graph)) return false;

        reset_compute_ctx();
        auto* gf = get_compute_graph(build_graph);
        if (!ggml_gallocr_alloc_graph(compute_allocr_, gf)) {
            LOG_ERROR("%s: failed to alloc compute graph", get_desc().c_str());
            return false;
        }

        copy_backend_tensor_data();

        if (ggml_backend_is_cpu(runtime_backend_)) {
            ggml_backend_cpu_set_n_threads(runtime_backend_, n_threads);
        }

        auto status = ggml_backend_graph_compute(runtime_backend_, gf);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERROR("%s: compute failed", get_desc().c_str());
            return false;
        }

        if (output) {
            auto* result = ggml_get_tensor(compute_ctx_, "result");
            if (*output == nullptr && output_ctx) {
                *output = ggml_dup_tensor(output_ctx, result);
            }
            if (*output) {
                ggml_backend_tensor_get(result, (*output)->data, 0, ggml_nbytes(*output));
            }
        }

        if (free_immediately) free_compute_buffer();
        return true;
    }

    // ── Helpers ──────────────────────────────────────────────────────

    ForwardContext get_forward_ctx() {
        ForwardContext fctx;
        fctx.backend = runtime_backend_;
        fctx.ctx = compute_ctx_;
        fctx.flash_attn = flash_attn_;
        fctx.one = one_tensor_;
        fctx.set_tensor_data = [this](ggml_tensor* t, const void* data) {
            this->set_backend_tensor_data(t, data);
        };
        return fctx;
    }

    void set_flash_attn(bool enabled) { flash_attn_ = enabled; }

    // Copy an input tensor to GPU if needed
    ggml_tensor* to_backend(ggml_tensor* t) {
        if (!t) return nullptr;
        if (!ggml_backend_is_cpu(runtime_backend_) &&
            (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer))) {
            auto* bt = ggml_dup_tensor(compute_ctx_, t);
            set_backend_tensor_data(bt, t->data);
            return bt;
        }
        return t;
    }

    void set_backend_tensor_data(ggml_tensor* t, const void* data) {
        if (use_sched_ && sched_) {
            ggml_backend_sched_set_tensor_backend(sched_, t, runtime_backend_);
        }
        backend_tensor_data_[t] = data;
    }

    ggml_backend_t runtime_backend() { return runtime_backend_; }

protected:
    ggml_backend_t  params_backend_   = nullptr;
    ggml_backend_t  runtime_backend_  = nullptr;
    ggml_backend_t  split_backend_    = nullptr;
    bool            own_split_backend_ = false;

    ggml_context*        params_ctx_       = nullptr;
    ggml_backend_buffer* params_buffer_    = nullptr;
    std::vector<ggml_backend_buffer_t> param_buffers_;
    std::map<ggml_tensor*, ggml_backend_t> param_tensor_backend_;
    ggml_context*        offload_ctx_      = nullptr;
    ggml_backend_buffer* runtime_params_   = nullptr;
    bool                 params_offloaded_ = false;

    ggml_context*  compute_ctx_    = nullptr;
    ggml_gallocr*  compute_allocr_ = nullptr;
    ggml_backend_sched_t sched_    = nullptr;
    bool use_sched_ = false;

    bool flash_attn_ = false;

    std::map<ggml_tensor*, const void*> backend_tensor_data_;

    // Pre-allocated memory buffer for compute context (reused across resets)
    void*  compute_buf_      = nullptr;
    size_t compute_buf_size_ = 0;

    // Built-in tensors for graph building
    std::vector<float> one_vec_ = {1.0f};
    ggml_tensor* one_tensor_ = nullptr;

    bool alloc_params_ctx() {
        ggml_init_params p = {};
        p.mem_size   = MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead();
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        params_ctx_ = ggml_init(p);
        if (!params_ctx_) {
            LOG_ERROR("Executor: ggml_init failed for params context");
            return false;
        }
        if (params_backend_ != runtime_backend_) {
            offload_ctx_ = ggml_init(p);
            if (!offload_ctx_) {
                LOG_ERROR("Executor: ggml_init failed for offload params context");
                ggml_free(params_ctx_);
                params_ctx_ = nullptr;
                return false;
            }
        }
        return true;
    }

    void free_params_ctx() {
        if (params_ctx_) { ggml_free(params_ctx_); params_ctx_ = nullptr; }
        if (offload_ctx_) { ggml_free(offload_ctx_); offload_ctx_ = nullptr; }
    }

    void alloc_compute_ctx() {
        size_t needed = ggml_tensor_overhead() * MAX_GRAPH_SIZE
                      + ggml_graph_overhead_custom(MAX_GRAPH_SIZE, false);

        // Pre-allocate the memory buffer once; reuse it on subsequent calls
        if (!compute_buf_ || compute_buf_size_ < needed) {
            if (compute_buf_) { free(compute_buf_); compute_buf_ = nullptr; }
            compute_buf_size_ = needed;
            compute_buf_ = malloc(compute_buf_size_);
            if (!compute_buf_) {
                LOG_ERROR("%s: failed to malloc compute buffer (%.2f MB)",
                          get_desc().c_str(), needed / (1024.0 * 1024.0));
                return;
            }
            LOG_DEBUG("%s: allocated compute context buffer %.2f MB",
                      get_desc().c_str(), needed / (1024.0 * 1024.0));
        }

        ggml_init_params p = {};
        p.mem_size   = compute_buf_size_;
        p.mem_buffer = compute_buf_;   // externally managed, ggml won't free it
        p.no_alloc   = true;
        compute_ctx_ = ggml_init(p);
        if (!compute_ctx_) {
            LOG_ERROR("%s: ggml_init failed for compute context", get_desc().c_str());
        }
    }

    void free_compute_ctx() {
        if (compute_ctx_) { ggml_free(compute_ctx_); compute_ctx_ = nullptr; }
        // Note: compute_buf_ is NOT freed here — it's reused across resets
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    void free_params_buffer() {
        if (params_buffer_) { ggml_backend_buffer_free(params_buffer_); params_buffer_ = nullptr; }
        for (auto* b : param_buffers_) {
            ggml_backend_buffer_free(b);
        }
        param_buffers_.clear();
    }

    void free_compute_buffer() {
        if (compute_allocr_) { ggml_gallocr_free(compute_allocr_); compute_allocr_ = nullptr; }
        if (sched_) {
            ggml_backend_sched_reset(sched_);
        }
        offload_params_to_params_backend();
    }

    ggml_cgraph* get_compute_graph(GraphBuilder build) {
        one_tensor_ = ggml_new_tensor_1d(compute_ctx_, GGML_TYPE_F32, 1);
        ggml_set_name(one_tensor_, "kd_builtin:one");
        set_backend_tensor_data(one_tensor_, one_vec_.data());

        auto* gf = build();
        if (ggml_graph_n_nodes(gf) > 0) {
            ggml_set_name(ggml_graph_node(gf, -1), "result");
        }
        ggml_build_forward_expand(gf, one_tensor_);
        return gf;
    }

    bool alloc_compute_buffer(GraphBuilder build) {
        if (compute_allocr_) return true;
        reset_compute_ctx();
        auto* gf = get_compute_graph(build);
        backend_tensor_data_.clear();

        compute_allocr_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(runtime_backend_));

        if (!ggml_gallocr_reserve(compute_allocr_, gf)) {
            LOG_ERROR("%s: failed to reserve compute buffer", get_desc().c_str());
            free_compute_buffer();
            return false;
        }

        size_t sz = ggml_gallocr_get_buffer_size(compute_allocr_, 0);
        LOG_DEBUG("%s: compute buffer %.2f MB (%s)", get_desc().c_str(),
                  sz / (1024.0 * 1024.0),
                  ggml_backend_is_cpu(runtime_backend_) ? "RAM" : "VRAM");
        return true;
    }

    void copy_backend_tensor_data() {
        for (auto& [t, data] : backend_tensor_data_) {
            ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        }
        backend_tensor_data_.clear();
    }

    bool offload_params_to_runtime() {
        if (!param_tensor_backend_.empty()) return true;
        if (params_backend_ == runtime_backend_) return true;
        if (params_offloaded_) return true;
        if (!params_ctx_ || !offload_ctx_) {
            LOG_ERROR("%s: params contexts are not initialized for offload", get_desc().c_str());
            return false;
        }

        assert(!runtime_params_);
        size_t num = 0;
        for (auto* t = ggml_get_first_tensor(params_ctx_); t; t = ggml_get_next_tensor(params_ctx_, t))
            num++;

        if (num == 0) return true;

        // Create mirror tensors in offload context
        for (auto* t = ggml_get_first_tensor(params_ctx_); t; t = ggml_get_next_tensor(params_ctx_, t))
            ggml_dup_tensor(offload_ctx_, t);

        runtime_params_ = ggml_backend_alloc_ctx_tensors(offload_ctx_, runtime_backend_);
        if (!runtime_params_) {
            LOG_ERROR("%s: failed to alloc runtime params", get_desc().c_str());
            return false;
        }

        auto* t  = ggml_get_first_tensor(params_ctx_);
        auto* ot = ggml_get_first_tensor(offload_ctx_);
        while (t && ot) {
            ggml_backend_tensor_copy(t, ot);
            std::swap(t->buffer, ot->buffer);
            std::swap(t->data, ot->data);
            std::swap(t->extra, ot->extra);
            t  = ggml_get_next_tensor(params_ctx_, t);
            ot = ggml_get_next_tensor(offload_ctx_, ot);
        }

        params_offloaded_ = true;
        LOG_INFO("%s: offloaded params to %s", get_desc().c_str(),
                 ggml_backend_name(runtime_backend_));
        return true;
    }

    void offload_params_to_params_backend() {
        if (!param_tensor_backend_.empty()) return;
        if (!params_offloaded_) return;
        if (!params_ctx_ || !offload_ctx_) return;
        auto* t  = ggml_get_first_tensor(params_ctx_);
        auto* ot = ggml_get_first_tensor(offload_ctx_);
        while (t && ot) {
            t->buffer = ot->buffer;
            t->data   = ot->data;
            t->extra  = ot->extra;
            ot->buffer = nullptr;
            ot->data   = nullptr;
            ot->extra  = nullptr;
            t  = ggml_get_next_tensor(params_ctx_, t);
            ot = ggml_get_next_tensor(offload_ctx_, ot);
        }
        if (runtime_params_) {
            ggml_backend_buffer_free(runtime_params_);
            runtime_params_ = nullptr;
        }
        params_offloaded_ = false;
    }
};

#endif // KD_EXECUTOR_HPP
