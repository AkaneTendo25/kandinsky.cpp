#ifndef KD_MODEL_LOADER_HPP
#define KD_MODEL_LOADER_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "util.hpp"

#define KD_MAX_DIMS 5

// ── Tensor storage metadata ──────────────────────────────────────────

struct TensorStorage {
    std::string name;
    ggml_type   type          = GGML_TYPE_F32;
    ggml_type   expected_type = GGML_TYPE_COUNT;
    int64_t     ne[KD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int         n_dims        = 0;
    size_t      file_index    = 0;
    uint64_t    offset        = 0;

    TensorStorage() = default;

    TensorStorage(std::string name_, ggml_type type_, const int64_t* ne_, int nd,
                  size_t fi = 0, uint64_t off = 0)
        : name(std::move(name_)), type(type_), n_dims(nd), file_index(fi), offset(off) {
        for (int i = 0; i < nd; i++) ne[i] = ne_[i];
    }

    int64_t nelements() const {
        int64_t n = 1;
        for (int i = 0; i < KD_MAX_DIMS; i++) n *= ne[i];
        return n;
    }

    int64_t nbytes() const {
        return nelements() * ggml_type_size(type) / ggml_blck_size(type);
    }

    void reverse_ne() {
        int64_t tmp[KD_MAX_DIMS] = {1,1,1,1,1};
        for (int i = 0; i < n_dims; i++) tmp[i] = ne[n_dims - 1 - i];
        for (int i = 0; i < n_dims; i++) ne[i] = tmp[i];
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << name << " | " << ggml_type_name(type) << " | " << n_dims << " [";
        for (int i = 0; i < KD_MAX_DIMS; i++) {
            ss << ne[i];
            if (i < KD_MAX_DIMS - 1) ss << ", ";
        }
        ss << "]";
        return ss.str();
    }
};

// ── Ordered map preserving insertion order ───────────────────────────

template<typename K, typename V>
class OrderedMap {
public:
    using iterator = typename std::vector<std::pair<K, V>>::iterator;
    using const_iterator = typename std::vector<std::pair<K, V>>::const_iterator;

    void insert(const K& key, const V& val) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            entries_[it->second].second = val;
        } else {
            index_[key] = entries_.size();
            entries_.push_back({key, val});
        }
    }

    V* find(const K& key) {
        auto it = index_.find(key);
        return it != index_.end() ? &entries_[it->second].second : nullptr;
    }

    const V* find(const K& key) const {
        auto it = index_.find(key);
        return it != index_.end() ? &entries_[it->second].second : nullptr;
    }

    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

    iterator begin() { return entries_.begin(); }
    iterator end() { return entries_.end(); }
    const_iterator begin() const { return entries_.begin(); }
    const_iterator end() const { return entries_.end(); }

private:
    std::vector<std::pair<K, V>> entries_;
    std::map<K, size_t>          index_;
};

typedef OrderedMap<std::string, TensorStorage> TensorStorageMap;
typedef std::function<bool(const TensorStorage&, ggml_tensor**)> OnNewTensorCb;

// ── Model Loader ─────────────────────────────────────────────────────

class ModelLoader {
public:
    bool init_from_file(const std::string& file_path, const std::string& prefix = "");
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore = {},
                      bool use_mmap = false);
    bool load_tensors(OnNewTensorCb cb, bool use_mmap = false);

    TensorStorageMap& tensor_storage() { return tensor_storage_; }

    std::vector<std::string> get_tensor_names() const {
        std::vector<std::string> names;
        for (auto& [name, _] : tensor_storage_) names.push_back(name);
        return names;
    }

protected:
    std::vector<std::string> file_paths_;
    TensorStorageMap          tensor_storage_;

    bool init_from_safetensors(const std::string& path, const std::string& prefix = "");
    bool init_from_gguf(const std::string& path, const std::string& prefix = "");

    bool read_tensor_data(const TensorStorage& ts, void* dst, bool use_mmap = false);
};

#endif // KD_MODEL_LOADER_HPP
