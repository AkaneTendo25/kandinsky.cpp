#ifndef KD_UTIL_HPP
#define KD_UTIL_HPP

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "kandinsky.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// ── Logging ──────────────────────────────────────────────────────────

static kd_log_cb_t g_log_cb      = nullptr;
static void*       g_log_cb_data = nullptr;

inline void kd_set_log_callback(kd_log_cb_t cb, void* data) {
    g_log_cb      = cb;
    g_log_cb_data = data;
}

inline void kd_log_printf(kd_log_level_t level, const char* file, int line, const char* fmt, ...) {
    char buf[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (g_log_cb) {
        g_log_cb(level, buf, g_log_cb_data);
        return;
    }

    const char* level_str = "???";
    switch (level) {
        case KD_LOG_DEBUG: level_str = "DEBUG"; break;
        case KD_LOG_INFO:  level_str = "INFO";  break;
        case KD_LOG_WARN:  level_str = "WARN";  break;
        case KD_LOG_ERROR: level_str = "ERROR"; break;
    }

    // Extract filename from path
    const char* fname = file;
    const char* sep = strrchr(file, '/');
    if (!sep) sep = strrchr(file, '\\');
    if (sep) fname = sep + 1;

    fprintf(stderr, "[%s] %s:%d - %s\n", level_str, fname, line, buf);
}

#define LOG_DEBUG(fmt, ...) kd_log_printf(KD_LOG_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  kd_log_printf(KD_LOG_INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  kd_log_printf(KD_LOG_WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) kd_log_printf(KD_LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// ── String helpers ───────────────────────────────────────────────────

inline bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool contains(const std::string& str, const std::string& sub) {
    return str.find(sub) != std::string::npos;
}

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline std::string path_join(const std::string& p1, const std::string& p2) {
    if (p1.empty()) return p2;
    if (p2.empty()) return p1;
    char last = p1.back();
    if (last == '/' || last == '\\') return p1 + p2;
    return p1 + "/" + p2;
}

inline bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}

inline std::vector<std::string> split_string(const std::string& str, char delim) {
    std::vector<std::string> result;
    std::string current;
    for (char c : str) {
        if (c == delim) {
            result.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

// ── Unicode helpers ──────────────────────────────────────────────────

inline std::u32string utf8_to_utf32(const std::string& utf8) {
    std::u32string result;
    size_t i = 0;
    while (i < utf8.size()) {
        uint32_t cp = 0;
        uint8_t c = (uint8_t)utf8[i];
        if (c < 0x80) {
            cp = c; i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; i += 1;
            if (i < utf8.size()) cp = (cp << 6) | (utf8[i++] & 0x3F);
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; i += 1;
            for (int j = 0; j < 2 && i < utf8.size(); j++) cp = (cp << 6) | (utf8[i++] & 0x3F);
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07; i += 1;
            for (int j = 0; j < 3 && i < utf8.size(); j++) cp = (cp << 6) | (utf8[i++] & 0x3F);
        } else {
            i += 1; continue;
        }
        result.push_back(cp);
    }
    return result;
}

inline std::string utf32_to_utf8(const std::u32string& utf32) {
    std::string result;
    for (char32_t cp : utf32) {
        if (cp < 0x80) {
            result += (char)cp;
        } else if (cp < 0x800) {
            result += (char)(0xC0 | (cp >> 6));
            result += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            result += (char)(0xE0 | (cp >> 12));
            result += (char)(0x80 | ((cp >> 6) & 0x3F));
            result += (char)(0x80 | (cp & 0x3F));
        } else {
            result += (char)(0xF0 | (cp >> 18));
            result += (char)(0x80 | ((cp >> 12) & 0x3F));
            result += (char)(0x80 | ((cp >> 6) & 0x3F));
            result += (char)(0x80 | (cp & 0x3F));
        }
    }
    return result;
}

// ── Memory-mapped file ───────────────────────────────────────────────

class MmapFile {
public:
    static std::unique_ptr<MmapFile> open(const std::string& path) {
        auto mf = std::unique_ptr<MmapFile>(new MmapFile());
        if (!mf->init(path)) return nullptr;
        return mf;
    }

    ~MmapFile() {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (data_ && data_ != MAP_FAILED) munmap(data_, size_);
        if (fd_ >= 0) close(fd_);
#endif
    }

    const uint8_t* data() const { return (const uint8_t*)data_; }
    size_t size() const { return size_; }

private:
    MmapFile() = default;

    bool init(const std::string& path) {
#ifdef _WIN32
        file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER file_size;
        GetFileSizeEx(file_, &file_size);
        size_ = (size_t)file_size.QuadPart;

        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) return false;

        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        return data_ != nullptr;
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;

        struct stat st;
        if (fstat(fd_, &st) != 0) return false;
        size_ = (size_t)st.st_size;

        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        return data_ != MAP_FAILED;
#endif
    }

#ifdef _WIN32
    HANDLE file_    = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    void*  data_    = nullptr;
#else
    int    fd_      = -1;
    void*  data_    = nullptr;
#endif
    size_t size_    = 0;
};

// ── Progress display ─────────────────────────────────────────────────

inline void pretty_progress(int step, int steps, float time_s) {
    if (steps <= 0) return;
    int pct = (int)(100.0f * step / steps);
    fprintf(stderr, "\r  [%3d%%] step %d/%d (%.2fs/step)", pct, step, steps, time_s);
    if (step >= steps) fprintf(stderr, "\n");
    fflush(stderr);
}

#endif // KD_UTIL_HPP
