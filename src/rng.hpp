#ifndef KD_RNG_HPP
#define KD_RNG_HPP

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// ── RNG interface ────────────────────────────────────────────────────

class RNG {
public:
    virtual ~RNG() = default;
    virtual void manual_seed(uint64_t seed) = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;
};

// ── STD default RNG ──────────────────────────────────────────────────

class STDDefaultRNG : public RNG {
public:
    void manual_seed(uint64_t seed) override {
        gen_.seed((unsigned int)seed);
    }

    std::vector<float> randn(uint32_t n) override {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> result(n);
        for (uint32_t i = 0; i < n; i++) {
            result[i] = dist(gen_);
        }
        return result;
    }

private:
    std::default_random_engine gen_;
};

// ── Philox RNG (matches torch CUDA randn) ───────────────────────────
// Port from: https://github.com/AUTOMATIC1111/stable-diffusion-webui

class PhiloxRNG : public RNG {
public:
    PhiloxRNG(uint64_t seed = 0) : seed_(seed), offset_(0) {}

    void manual_seed(uint64_t seed) override {
        seed_   = seed;
        offset_ = 0;
    }

    std::vector<float> randn(uint32_t n) override {
        std::vector<std::vector<uint32_t>> counter(4, std::vector<uint32_t>(n, 0));
        for (uint32_t i = 0; i < n; i++) {
            counter[0][i] = offset_;
            counter[2][i] = i;
        }
        offset_ += 1;

        std::vector<uint64_t> key(n, seed_);
        auto key_u32 = to_uint32_vec(key);

        auto g = philox4_32(counter, key_u32);

        std::vector<float> result(n);
        for (uint32_t i = 0; i < n; i++) {
            result[i] = box_muller((float)g[0][i], (float)g[1][i]);
        }
        return result;
    }

private:
    uint64_t seed_;
    uint32_t offset_;

    static constexpr uint32_t PHILOX_M0 = 0xD2511F53;
    static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57;
    static constexpr uint32_t PHILOX_W0 = 0x9E3779B9;
    static constexpr uint32_t PHILOX_W1 = 0xBB67AE85;

    static constexpr float TWO_POW32_INV      = 2.3283064e-10f;
    static constexpr float TWO_POW32_INV_2PI  = 2.3283064e-10f * 6.2831855f;

    static std::vector<uint32_t> to_uint32(uint64_t x) {
        return { (uint32_t)(x & 0xFFFFFFFF), (uint32_t)(x >> 32) };
    }

    static std::vector<std::vector<uint32_t>> to_uint32_vec(const std::vector<uint64_t>& x) {
        uint32_t N = (uint32_t)x.size();
        std::vector<std::vector<uint32_t>> result(2, std::vector<uint32_t>(N));
        for (uint32_t i = 0; i < N; i++) {
            result[0][i] = (uint32_t)(x[i] & 0xFFFFFFFF);
            result[1][i] = (uint32_t)(x[i] >> 32);
        }
        return result;
    }

    static void philox4_round(std::vector<std::vector<uint32_t>>& counter,
                               const std::vector<std::vector<uint32_t>>& key) {
        uint32_t N = (uint32_t)counter[0].size();
        for (uint32_t i = 0; i < N; i++) {
            auto v1 = to_uint32((uint64_t)counter[0][i] * (uint64_t)PHILOX_M0);
            auto v2 = to_uint32((uint64_t)counter[2][i] * (uint64_t)PHILOX_M1);
            counter[0][i] = v2[1] ^ counter[1][i] ^ key[0][i];
            counter[1][i] = v2[0];
            counter[2][i] = v1[1] ^ counter[3][i] ^ key[1][i];
            counter[3][i] = v1[0];
        }
    }

    static std::vector<std::vector<uint32_t>> philox4_32(
            std::vector<std::vector<uint32_t>>& counter,
            std::vector<std::vector<uint32_t>>& key,
            int rounds = 10) {
        uint32_t N = (uint32_t)counter[0].size();
        for (int i = 0; i < rounds - 1; i++) {
            philox4_round(counter, key);
            for (uint32_t j = 0; j < N; j++) {
                key[0][j] += PHILOX_W0;
                key[1][j] += PHILOX_W1;
            }
        }
        philox4_round(counter, key);
        return counter;
    }

    static float box_muller(float x, float y) {
        float u = x * TWO_POW32_INV + TWO_POW32_INV / 2;
        float v = y * TWO_POW32_INV_2PI + TWO_POW32_INV_2PI / 2;
        return sqrtf(-2.0f * logf(u)) * sinf(v);
    }
};

#endif // KD_RNG_HPP
