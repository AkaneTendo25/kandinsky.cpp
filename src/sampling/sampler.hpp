#ifndef KD_SAMPLER_HPP
#define KD_SAMPLER_HPP

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "sampling/scheduler.hpp"
#include "util.hpp"

// ── Sampler: Euler and DPM++2M for velocity prediction ───────────────

namespace sampler {

// Model prediction function: takes (latent, timestep) → velocity
// The timestep is scaled by 1000 before passing to the model
using VelocityFn = std::function<bool(ggml_tensor* x, float t, ggml_tensor* out)>;

// ── Euler sampler ────────────────────────────────────────────────────
// Simple first-order integration: x += dt * v(x, t)

inline bool euler(ggml_tensor* latent,
                  VelocityFn predict_velocity,
                  int num_steps,
                  float scheduler_scale,
                  float guidance_weight = 1.0f,
                  VelocityFn predict_velocity_uncond = nullptr) {
    auto timesteps = scheduler::flow_schedule(num_steps, scheduler_scale);
    auto dt_vec = scheduler::get_dt(timesteps);

    int64_t n = ggml_nelements(latent);
    std::vector<float> velocity(n);
    std::vector<float> velocity_uncond(n);
    std::vector<float> x(n);

    // Read initial latent
    ggml_backend_tensor_get(latent, x.data(), 0, n * sizeof(float));

    for (int step = 0; step < num_steps; step++) {
        float t = timesteps[step];
        float dt = dt_vec[step];

        // Set current x
        ggml_backend_tensor_set(latent, x.data(), 0, n * sizeof(float));

        // Create temp output tensor on CPU for velocity
        ggml_init_params p = { n * sizeof(float) + ggml_tensor_overhead(), nullptr, false };
        auto* tmp_ctx = ggml_init(p);
        auto* vel_tensor = ggml_new_tensor(tmp_ctx, GGML_TYPE_F32,
                                           ggml_n_dims(latent), latent->ne);

        // Predict velocity
        if (!predict_velocity(latent, t, vel_tensor)) {
            ggml_free(tmp_ctx);
            return false;
        }
        memcpy(velocity.data(), vel_tensor->data, n * sizeof(float));

        // CFG: combine conditional + unconditional predictions
        if (guidance_weight != 1.0f && predict_velocity_uncond) {
            auto* vel_uncond = ggml_new_tensor(tmp_ctx, GGML_TYPE_F32,
                                               ggml_n_dims(latent), latent->ne);
            if (!predict_velocity_uncond(latent, t, vel_uncond)) {
                ggml_free(tmp_ctx);
                return false;
            }
            memcpy(velocity_uncond.data(), vel_uncond->data, n * sizeof(float));

            // v = v_uncond + guidance * (v_cond - v_uncond)
            for (int64_t i = 0; i < n; i++) {
                velocity[i] = velocity_uncond[i] +
                              guidance_weight * (velocity[i] - velocity_uncond[i]);
            }
        }

        ggml_free(tmp_ctx);

        // Euler step: x += dt * v
        for (int64_t i = 0; i < n; i++) {
            x[i] += dt * velocity[i];
        }

        int64_t t_ms = ggml_time_ms();
        pretty_progress(step + 1, num_steps, 0.0f);
    }

    // Write result back
    ggml_backend_tensor_set(latent, x.data(), 0, n * sizeof(float));
    return true;
}

// ── DPM++2M sampler ──────────────────────────────────────────────────
// Second-order multistep method for improved quality

inline bool dpm_pp_2m(ggml_tensor* latent,
                       VelocityFn predict_velocity,
                       int num_steps,
                       float scheduler_scale,
                       float guidance_weight = 1.0f,
                       VelocityFn predict_velocity_uncond = nullptr) {
    auto timesteps = scheduler::flow_schedule(num_steps, scheduler_scale);
    auto dt_vec = scheduler::get_dt(timesteps);

    int64_t n = ggml_nelements(latent);
    std::vector<float> x(n);
    std::vector<float> velocity(n);
    std::vector<float> velocity_uncond(n);
    std::vector<float> prev_velocity(n, 0.0f);
    bool has_prev = false;

    ggml_backend_tensor_get(latent, x.data(), 0, n * sizeof(float));

    for (int step = 0; step < num_steps; step++) {
        float t = timesteps[step];
        float dt = dt_vec[step];

        ggml_backend_tensor_set(latent, x.data(), 0, n * sizeof(float));

        ggml_init_params p = { n * sizeof(float) * 2 + ggml_tensor_overhead() * 2, nullptr, false };
        auto* tmp_ctx = ggml_init(p);
        auto* vel_tensor = ggml_new_tensor(tmp_ctx, GGML_TYPE_F32,
                                           ggml_n_dims(latent), latent->ne);

        if (!predict_velocity(latent, t, vel_tensor)) {
            ggml_free(tmp_ctx);
            return false;
        }
        memcpy(velocity.data(), vel_tensor->data, n * sizeof(float));

        if (guidance_weight != 1.0f && predict_velocity_uncond) {
            auto* vel_uncond = ggml_new_tensor(tmp_ctx, GGML_TYPE_F32,
                                               ggml_n_dims(latent), latent->ne);
            if (!predict_velocity_uncond(latent, t, vel_uncond)) {
                ggml_free(tmp_ctx);
                return false;
            }
            memcpy(velocity_uncond.data(), vel_uncond->data, n * sizeof(float));
            for (int64_t i = 0; i < n; i++) {
                velocity[i] = velocity_uncond[i] +
                              guidance_weight * (velocity[i] - velocity_uncond[i]);
            }
        }

        ggml_free(tmp_ctx);

        if (has_prev) {
            // DPM++2M: x += dt * (3/2 * v_n - 1/2 * v_{n-1})
            for (int64_t i = 0; i < n; i++) {
                x[i] += dt * (1.5f * velocity[i] - 0.5f * prev_velocity[i]);
            }
        } else {
            // First step: Euler
            for (int64_t i = 0; i < n; i++) {
                x[i] += dt * velocity[i];
            }
        }

        prev_velocity = velocity;
        has_prev = true;

        pretty_progress(step + 1, num_steps, 0.0f);
    }

    ggml_backend_tensor_set(latent, x.data(), 0, n * sizeof(float));
    return true;
}

} // namespace sampler

#endif // KD_SAMPLER_HPP
