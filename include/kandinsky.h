#ifndef KANDINSKY_H
#define KANDINSKY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KD_LOG_DEBUG,
    KD_LOG_INFO,
    KD_LOG_WARN,
    KD_LOG_ERROR,
} kd_log_level_t;

typedef void (*kd_log_cb_t)(kd_log_level_t level, const char* text, void* user_data);
typedef void (*kd_progress_cb_t)(int step, int steps, float time, void* user_data);

typedef enum {
    KD_TYPE_F32  = 0,
    KD_TYPE_F16  = 1,
    KD_TYPE_Q8_0 = 8,
    KD_TYPE_Q4_0 = 2,
    KD_TYPE_Q4_1 = 3,
    KD_TYPE_Q5_0 = 6,
    KD_TYPE_Q5_1 = 7,
} kd_type_t;

typedef enum {
    KD_MODE_TXT2IMG,
    KD_MODE_TXT2VID,
    KD_MODE_IMG2IMG,
    KD_MODE_IMG2VID,
} kd_mode_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} kd_image_t;

typedef struct kd_ctx kd_ctx_t;

struct kd_params {
    const char* model_dir;
    const char* vocab_dir;       // Optional: directory with tokenizer files (text_encoder/, text_encoder2/)
    const char* dit_filename;    // Optional: DiT GGUF filename (default: "dit.gguf")
    int n_threads;
    bool flash_attn;
    kd_type_t wtype;
    bool text_cpu;               // Run Qwen+CLIP on CPU to save VRAM
    bool vae3d_cpu;              // Run VAE3D decoder on CPU to save VRAM
    int dit_gpu_layers;          // Keep last N DiT blocks on GPU (-1 = all)
    kd_log_cb_t log_cb;
    void* log_cb_data;
};

struct kd_generate_params {
    const char* prompt;
    const char* negative_prompt;
    int width;
    int height;
    int num_frames;
    int num_steps;
    float guidance_scale;
    float scheduler_scale;
    int64_t seed;
    kd_mode_t mode;
    kd_progress_cb_t progress_cb;
    void* progress_cb_data;
};

kd_ctx_t* kd_ctx_create(struct kd_params params);
void kd_ctx_free(kd_ctx_t* ctx);

// Generate image(s) or video frames
// Returns number of output images, or -1 on error
// Caller must free output images with kd_image_free()
int kd_generate(kd_ctx_t* ctx, struct kd_generate_params params, kd_image_t** output);

void kd_image_free(kd_image_t* images, int count);

// Get default params with sensible defaults
struct kd_params kd_default_params(void);
struct kd_generate_params kd_default_generate_params(void);

#ifdef __cplusplus
}
#endif

#endif // KANDINSKY_H
