#include "core/model_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>

#include "gguf.h"

// We include nlohmann json for safetensors header parsing
#include "json.hpp"
using json = nlohmann::json;

// ── Safetensors format ───────────────────────────────────────────────
// Layout: [8-byte header_size LE] [JSON header] [tensor data]
// JSON header maps tensor names to {dtype, shape, data_offsets: [begin, end]}

static ggml_type dtype_str_to_ggml(const std::string& dtype) {
    if (dtype == "F32"  || dtype == "float32")  return GGML_TYPE_F32;
    if (dtype == "F16"  || dtype == "float16")  return GGML_TYPE_F16;
    if (dtype == "BF16" || dtype == "bfloat16") return GGML_TYPE_BF16;
    if (dtype == "I8")                          return GGML_TYPE_I8;
    if (dtype == "I16")                         return GGML_TYPE_I16;
    if (dtype == "I32")                         return GGML_TYPE_I32;
    return GGML_TYPE_COUNT;
}

bool ModelLoader::init_from_safetensors(const std::string& path, const std::string& prefix) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERROR("Cannot open %s", path.c_str());
        return false;
    }

    // Read header size
    uint64_t header_size = 0;
    if (fread(&header_size, sizeof(header_size), 1, f) != 1) {
        fclose(f);
        return false;
    }

    if (header_size > 100 * 1024 * 1024) { // sanity check: 100MB max header
        LOG_ERROR("Safetensors header too large: %llu", (unsigned long long)header_size);
        fclose(f);
        return false;
    }

    // Read header JSON
    std::string header_str(header_size, '\0');
    if (fread(header_str.data(), 1, header_size, f) != header_size) {
        fclose(f);
        return false;
    }
    fclose(f);

    uint64_t data_offset = 8 + header_size;
    size_t file_idx = file_paths_.size();
    file_paths_.push_back(path);

    try {
        auto header = json::parse(header_str);
        for (auto& [key, val] : header.items()) {
            if (key == "__metadata__") continue;

            std::string dtype = val["dtype"].get<std::string>();
            auto shape = val["shape"].get<std::vector<int64_t>>();
            auto offsets = val["data_offsets"].get<std::vector<uint64_t>>();

            if (shape.size() > KD_MAX_DIMS) {
                LOG_ERROR("Tensor %s has %zu dimensions, but loader supports at most %d",
                          key.c_str(), shape.size(), KD_MAX_DIMS);
                return false;
            }
            if (offsets.size() != 2 || offsets[1] < offsets[0]) {
                LOG_ERROR("Tensor %s has invalid data_offsets in safetensors header", key.c_str());
                return false;
            }

            ggml_type type = dtype_str_to_ggml(dtype);
            if (type == GGML_TYPE_COUNT) {
                LOG_WARN("Unsupported dtype %s for tensor %s, skipping", dtype.c_str(), key.c_str());
                continue;
            }

            TensorStorage ts;
            ts.name = prefix.empty() ? key : prefix + "." + key;
            ts.type = type;
            ts.n_dims = (int)shape.size();
            // Safetensors uses [outer...inner] order, ggml uses [inner...outer]
            for (int i = 0; i < ts.n_dims; i++) {
                ts.ne[i] = shape[ts.n_dims - 1 - i];
            }
            ts.file_index = file_idx;
            ts.offset = data_offset + offsets[0];

            tensor_storage_.insert(ts.name, ts);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse safetensors header: %s", e.what());
        return false;
    }

    LOG_INFO("Loaded %zu tensors from %s", tensor_storage_.size(), path.c_str());
    return true;
}

// ── GGUF format ──────────────────────────────────────────────────────

bool ModelLoader::init_from_gguf(const std::string& path, const std::string& prefix) {
    ggml_context* meta_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta_ctx,
    };

    auto* gguf = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf) {
        LOG_ERROR("Failed to load GGUF: %s", path.c_str());
        return false;
    }

    size_t file_idx = file_paths_.size();
    file_paths_.push_back(path);

    int64_t n_tensors = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf, i);
        size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);

        TensorStorage ts;
        ts.name = prefix.empty() ? name : prefix + "." + std::string(name);
        ts.type = gguf_get_tensor_type(gguf, i);
        ts.file_index = file_idx;
        ts.offset = offset;

        // Get shape from the meta context
        if (meta_ctx) {
            ggml_tensor* meta_tensor = ggml_get_tensor(meta_ctx, name);
            if (meta_tensor) {
                ts.n_dims = ggml_n_dims(meta_tensor);
                if (ts.n_dims > KD_MAX_DIMS) {
                    LOG_ERROR("Tensor %s has %d dimensions, but loader supports at most %d",
                              name, ts.n_dims, KD_MAX_DIMS);
                    if (meta_ctx) ggml_free(meta_ctx);
                    gguf_free(gguf);
                    return false;
                }
                for (int d = 0; d < ts.n_dims; d++) {
                    ts.ne[d] = meta_tensor->ne[d];
                }
            }
        }

        tensor_storage_.insert(ts.name, ts);
    }

    if (meta_ctx) ggml_free(meta_ctx);
    gguf_free(gguf);

    LOG_INFO("Loaded %zu tensors from GGUF %s", tensor_storage_.size(), path.c_str());
    return true;
}

// ── Dispatch ─────────────────────────────────────────────────────────

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    if (ends_with(file_path, ".safetensors")) {
        return init_from_safetensors(file_path, prefix);
    } else if (ends_with(file_path, ".gguf")) {
        return init_from_gguf(file_path, prefix);
    } else {
        LOG_ERROR("Unsupported model format: %s", file_path.c_str());
        return false;
    }
}

// ── Read tensor data from file ───────────────────────────────────────

bool ModelLoader::read_tensor_data(const TensorStorage& ts, void* dst, bool use_mmap) {
    const auto& path = file_paths_[ts.file_index];
    size_t bytes = ts.nbytes();

    if (use_mmap) {
        auto mf = MmapFile::open(path);
        if (mf && ts.offset + bytes <= mf->size()) {
            memcpy(dst, mf->data() + ts.offset, bytes);
            return true;
        }
        // Fall through to fread path if mmap fails
    }

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERROR("Cannot open %s for reading tensor %s", path.c_str(), ts.name.c_str());
        return false;
    }

    // Validate offset + size against file size
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    int64_t file_size = _ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    int64_t file_size = ftello(f);
#endif
    if ((int64_t)(ts.offset + bytes) > file_size) {
        LOG_ERROR("Tensor %s: offset+size (%llu+%zu=%llu) exceeds file size (%lld) in %s",
                  ts.name.c_str(),
                  (unsigned long long)ts.offset, bytes, (unsigned long long)(ts.offset + bytes),
                  (long long)file_size, path.c_str());
        fclose(f);
        return false;
    }

#ifdef _WIN32
    _fseeki64(f, ts.offset, SEEK_SET);
#else
    fseeko(f, ts.offset, SEEK_SET);
#endif

    size_t read = fread(dst, 1, bytes, f);
    fclose(f);

    if (read != bytes) {
        LOG_ERROR("Failed to read tensor %s: expected %zu bytes, got %zu", ts.name.c_str(), bytes, read);
        return false;
    }

    return true;
}

// ── Load tensors into existing ggml tensors ──────────────────────────

bool ModelLoader::load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                std::set<std::string> ignore, bool use_mmap) {
    int loaded = 0, skipped = 0, failed = 0;
    std::set<std::string> loaded_names;

    for (auto& [name, ts] : tensor_storage_) {
        if (ignore.count(name)) { skipped++; continue; }

        auto it = tensors.find(name);
        if (it == tensors.end()) {
            skipped++;
            continue;
        }

        auto* tensor = it->second;
        bool shape_ok = ts.n_dims == ggml_n_dims(tensor);
        if (shape_ok) {
            for (int d = 0; d < ts.n_dims; d++) {
                if (ts.ne[d] != tensor->ne[d]) {
                    shape_ok = false;
                    break;
                }
            }
        }
        if (!shape_ok) {
            LOG_ERROR("Shape mismatch for %s: model=[%lld,%lld,%lld,%lld] n_dims=%d, target=[%lld,%lld,%lld,%lld] n_dims=%d",
                      name.c_str(),
                      (long long)ts.ne[0], (long long)ts.ne[1], (long long)ts.ne[2], (long long)ts.ne[3], ts.n_dims,
                      (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3], ggml_n_dims(tensor));
            failed++;
            continue;
        }
        size_t bytes = ts.nbytes();

        if (ts.type == tensor->type) {
            // Direct load
            if (tensor->buffer) {
                std::vector<uint8_t> buf(bytes);
                if (!read_tensor_data(ts, buf.data(), use_mmap)) {
                    LOG_ERROR("FATAL: failed to read tensor %s — aborting load", name.c_str());
                    failed++;
                    continue;
                }
                ggml_backend_tensor_set(tensor, buf.data(), 0, bytes);
            } else if (tensor->data) {
                if (!read_tensor_data(ts, tensor->data, use_mmap)) {
                    LOG_ERROR("FATAL: failed to read tensor %s — aborting load", name.c_str());
                    failed++;
                    continue;
                }
            }
        } else {
            // Type conversion needed — load as original type, then convert
            LOG_DEBUG("Type conversion for %s: %s -> %s", name.c_str(),
                      ggml_type_name(ts.type), ggml_type_name(tensor->type));
            std::vector<uint8_t> buf(bytes);
            if (!read_tensor_data(ts, buf.data(), use_mmap)) {
                LOG_ERROR("FATAL: failed to read tensor %s — aborting load", name.c_str());
                failed++;
                continue;
            }

            // Convert through F32 for both CPU tensors and backend tensors.
            size_t n = ts.nelements();
            std::vector<float> f32_buf(n);
            if (ts.type == GGML_TYPE_F32) {
                memcpy(f32_buf.data(), buf.data(), n * sizeof(float));
            } else {
                const auto* src_traits = ggml_get_type_traits(ts.type);
                if (!src_traits || !src_traits->to_float) {
                    LOG_ERROR("FATAL: no to_float conversion for type %s (tensor %s)",
                              ggml_type_name(ts.type), name.c_str());
                    failed++;
                    continue;
                }
                src_traits->to_float(buf.data(), f32_buf.data(), n);
            }

            bool all_zero = true;
            bool has_nan = false;
            for (size_t i = 0; i < std::min(n, (size_t)1024); i++) {
                if (f32_buf[i] != 0.0f) all_zero = false;
                if (std::isnan(f32_buf[i])) has_nan = true;
            }
            if (all_zero && n > 1) {
                LOG_WARN("Tensor %s: all zeros after type conversion (possible corruption)", name.c_str());
            }
            if (has_nan) {
                LOG_ERROR("Tensor %s: contains NaN after type conversion (corrupted weights)", name.c_str());
                failed++;
                continue;
            }

            std::vector<uint8_t> dst_buf(ggml_nbytes(tensor));
            const auto* dst_traits = ggml_get_type_traits(tensor->type);
            if (dst_traits && dst_traits->from_float_ref) {
                dst_traits->from_float_ref(f32_buf.data(), dst_buf.data(), n);
            } else if (tensor->type == GGML_TYPE_F32) {
                memcpy(dst_buf.data(), f32_buf.data(), ggml_nbytes(tensor));
            } else {
                LOG_ERROR("FATAL: no from_float_ref for type %s (tensor %s)",
                          ggml_type_name(tensor->type), name.c_str());
                failed++;
                continue;
            }

            if (tensor->buffer) {
                ggml_backend_tensor_set(tensor, dst_buf.data(), 0, ggml_nbytes(tensor));
            } else if (tensor->data) {
                memcpy(tensor->data, dst_buf.data(), ggml_nbytes(tensor));
            }
        }
        loaded++;
        loaded_names.insert(name);
    }

    std::vector<std::string> missing_names;
    for (const auto& kv : tensors) {
        const std::string& name = kv.first;
        if (ignore.count(name)) continue;
        if (!loaded_names.count(name)) {
            missing_names.push_back(name);
        }
    }

    if (!missing_names.empty()) {
        LOG_ERROR("Tensor loading FAILED: %zu required tensors were not loaded", missing_names.size());
        const size_t limit = std::min<size_t>(missing_names.size(), 32);
        for (size_t i = 0; i < limit; i++) {
            LOG_ERROR("  MISSING: %s", missing_names[i].c_str());
        }
        if (missing_names.size() > limit) {
            LOG_ERROR("  ... and %zu more", missing_names.size() - limit);
        }
        return false;
    }

    if (failed > 0) {
        LOG_ERROR("Tensor loading FAILED: %d tensors could not be read (loaded %d, skipped %d)",
                  failed, loaded, skipped);
        return false;
    }

    LOG_INFO("Loaded %d tensors, skipped %d", loaded, skipped);
    return loaded > 0;
}

bool ModelLoader::load_tensors(OnNewTensorCb cb, bool use_mmap) {
    int loaded = 0, failed = 0;
    for (auto& [name, ts] : tensor_storage_) {
        ggml_tensor* tensor = nullptr;
        if (cb(ts, &tensor) && tensor) {
            if (tensor->buffer) {
                std::vector<uint8_t> buf(ts.nbytes());
                if (!read_tensor_data(ts, buf.data(), use_mmap)) {
                    LOG_ERROR("FATAL: failed to read tensor %s via callback", name.c_str());
                    failed++;
                    continue;
                }
                ggml_backend_tensor_set(tensor, buf.data(), 0, ts.nbytes());
            } else if (tensor->data) {
                if (!read_tensor_data(ts, tensor->data, use_mmap)) {
                    LOG_ERROR("FATAL: failed to read tensor %s via callback", name.c_str());
                    failed++;
                    continue;
                }
            }
            loaded++;
        }
    }
    if (failed > 0) {
        LOG_ERROR("Callback tensor loading FAILED: %d tensors could not be read", failed);
        return false;
    }
    return true;
}
