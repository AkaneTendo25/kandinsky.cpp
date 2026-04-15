#ifndef KD_SCHEDULER_HPP
#define KD_SCHEDULER_HPP

#include <cstdint>
#include <vector>

// ── Flow matching schedule with shift ────────────────────────────────
// K5 uses velocity prediction with a shifted timestep schedule:
//   t_raw = linspace(1, 0, num_steps + 1)
//   t = scale * t_raw / (1 + (scale - 1) * t_raw)
// Default scale: 3.0 for images, 5.0 for video

namespace scheduler {

inline std::vector<float> flow_schedule(int num_steps, float scale = 3.0f) {
    std::vector<float> timesteps(num_steps + 1);
    for (int i = 0; i <= num_steps; i++) {
        float t_raw = 1.0f - (float)i / (float)num_steps;
        timesteps[i] = scale * t_raw / (1.0f + (scale - 1.0f) * t_raw);
    }
    return timesteps;
}

// Compute dt for each step: timesteps[i+1] - timesteps[i]
inline std::vector<float> get_dt(const std::vector<float>& timesteps) {
    std::vector<float> dt(timesteps.size() - 1);
    for (size_t i = 0; i < dt.size(); i++) {
        dt[i] = timesteps[i + 1] - timesteps[i];
    }
    return dt;
}

} // namespace scheduler

#endif // KD_SCHEDULER_HPP
