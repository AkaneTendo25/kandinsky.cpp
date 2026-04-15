#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <string>

#include "kandinsky.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ── Argument parsing ─────────────────────────────────────────────────

struct Args {
    std::string model_dir;
    std::string vocab_dir;
    std::string dit_filename;
    std::string prompt;
    std::string negative_prompt;
    std::string output = "output.png";
    int         width  = 1024;
    int         height = 1024;
    int         steps  = 50;
    int         frames = 1;
    int         fps    = 8;
    float       cfg_scale = 3.5f;
    float       scheduler_scale = 3.0f;
    int64_t     seed   = -1;
    int         threads = 4;
    int         dit_gpu_layers = -1;
    std::string type   = "f16";
    bool        flash_attn = true;
    bool        text_cpu = false;
    bool        vae3d_cpu = false;
    bool        help   = false;
    bool        parse_error = false;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  -m, --model-dir <path>     Model directory with GGUF files (required)\n"
        "  -d, --vocab-dir <path>     Optional tokenizer data directory for legacy bundles\n"
        "  --dit <filename>           DiT GGUF filename (default: dit.gguf)\n"
        "  -p, --prompt <text>        Prompt text (required)\n"
        "  -n, --negative-prompt <t>  Negative prompt\n"
        "  -o, --output <path>        Output file (default: output.png)\n"
        "  -W, --width <int>          Width (default: 1024)\n"
        "  -H, --height <int>         Height (default: 1024)\n"
        "  -s, --steps <int>          Number of steps (default: 50)\n"
        "  --frames <int>             Number of frames for video (default: 1 = image)\n"
        "  --fps <int>                Output video FPS for container encoding (default: 8)\n"
        "  --cfg-scale <float>        CFG guidance scale (default: 3.5)\n"
        "  --scheduler-scale <float>  Scheduler scale (default: 3.0)\n"
        "  --seed <int>               Random seed (default: -1 = random)\n"
        "  -t, --threads <int>        Number of threads (default: 4)\n"
        "  --type <str>               Weight type: f16, q8_0, q4_0 (default: f16)\n"
        "  --dit-gpu-layers <int>     Keep last N DiT blocks on GPU (-1 = all)\n"
        "  --text-cpu                 Run Qwen+CLIP on CPU (lower VRAM, slower)\n"
        "  --vae3d-cpu                Run VAE3D on CPU (much lower VRAM, much slower)\n"
        "  --no-flash-attn            Disable flash attention\n"
        "  -h, --help                 Show this help\n\n"
        "Examples:\n"
        "  %s -m models/ -p \"a photo of a cat\" -o cat.png --type q4_0\n"
        "  %s -m models/ -p \"a cat walking\" -o cat.mp4 --frames 61 --fps 24 --type q4_0\n",
        prog, prog, prog);
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 < argc) return argv[++i];
            fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
            exit(1);
            return nullptr;
        };

        if (arg == "-m" || arg == "--model-dir")       args.model_dir = next();
        else if (arg == "-d" || arg == "--vocab-dir")   args.vocab_dir = next();
        else if (arg == "--dit")                        args.dit_filename = next();
        else if (arg == "-p" || arg == "--prompt")      args.prompt = next();
        else if (arg == "-n" || arg == "--negative-prompt") args.negative_prompt = next();
        else if (arg == "-o" || arg == "--output")      args.output = next();
        else if (arg == "-W" || arg == "--width")       args.width = atoi(next());
        else if (arg == "-H" || arg == "--height")      args.height = atoi(next());
        else if (arg == "-s" || arg == "--steps")       args.steps = atoi(next());
        else if (arg == "--frames")                     args.frames = atoi(next());
        else if (arg == "--fps")                        args.fps = atoi(next());
        else if (arg == "--cfg-scale")                  args.cfg_scale = (float)atof(next());
        else if (arg == "--scheduler-scale")            args.scheduler_scale = (float)atof(next());
        else if (arg == "--seed")                       args.seed = atoll(next());
        else if (arg == "-t" || arg == "--threads")     args.threads = atoi(next());
        else if (arg == "--dit-gpu-layers")             args.dit_gpu_layers = atoi(next());
        else if (arg == "--type")                       args.type = next();
        else if (arg == "--text-cpu")                   args.text_cpu = true;
        else if (arg == "--vae3d-cpu")                  args.vae3d_cpu = true;
        else if (arg == "--no-flash-attn")              args.flash_attn = false;
        else if (arg == "-h" || arg == "--help")        args.help = true;
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            args.parse_error = true;
        }
    }
    return args;
}

static kd_type_t type_from_string(const std::string& s) {
    if (s == "f32")  return KD_TYPE_F32;
    if (s == "f16")  return KD_TYPE_F16;
    if (s == "q8_0") return KD_TYPE_Q8_0;
    if (s == "q4_0") return KD_TYPE_Q4_0;
    if (s == "q4_1") return KD_TYPE_Q4_1;
    if (s == "q5_0") return KD_TYPE_Q5_0;
    if (s == "q5_1") return KD_TYPE_Q5_1;
    fprintf(stderr, "Unknown type: %s, using f16\n", s.c_str());
    return KD_TYPE_F16;
}

static std::string to_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return to_lower(s.substr(s.size() - suffix.size())) == to_lower(suffix);
}

static bool is_video_path(const std::string& p) {
    return ends_with_ci(p, ".mp4") || ends_with_ci(p, ".mov") ||
           ends_with_ci(p, ".mkv") || ends_with_ci(p, ".webm");
}

static std::string quote_arg(const std::string& s) {
    std::string q = "\"";
    for (char c : s) {
        if (c == '"') q += "\\\"";
        else q += c;
    }
    q += "\"";
    return q;
}

// ── Logging callback ─────────────────────────────────────────────────

static void log_callback(kd_log_level_t level, const char* text, void*) {
    const char* prefix = "";
    switch (level) {
        case KD_LOG_DEBUG: prefix = "[DEBUG] "; break;
        case KD_LOG_INFO:  prefix = "[INFO]  "; break;
        case KD_LOG_WARN:  prefix = "[WARN]  "; break;
        case KD_LOG_ERROR: prefix = "[ERROR] "; break;
    }
    fprintf(stderr, "%s%s\n", prefix, text);
}

// ── Main ─────────────────────────────────────────────────────────────

static bool ensure_parent_dir(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path parent = fs::path(path).parent_path();
    if (parent.empty()) return true;
    return fs::create_directories(parent, ec) || fs::exists(parent);
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.help || args.parse_error || args.model_dir.empty() || args.prompt.empty()) {
        print_usage(argv[0]);
        return (args.help && !args.parse_error) ? 0 : 1;
    }
    if (args.frames > 1 && args.fps <= 0) {
        fprintf(stderr, "Invalid --fps value: %d\n", args.fps);
        return 1;
    }

    fprintf(stderr, "kandinsky-cli — Kandinsky 5 inference engine\n\n");
    fprintf(stderr, "  Model:    %s\n", args.model_dir.c_str());
    fprintf(stderr, "  Prompt:   %s\n", args.prompt.c_str());
    fprintf(stderr, "  Size:     %dx%d\n", args.width, args.height);
    fprintf(stderr, "  Frames:   %d%s\n", args.frames,
            args.frames > 1 ? " (video)" : " (image)");
    if (args.frames > 1) {
        fprintf(stderr, "  FPS:      %d\n", args.fps);
    }
    fprintf(stderr, "  Steps:    %d\n", args.steps);
    fprintf(stderr, "  CFG:      %.1f\n", args.cfg_scale);
    fprintf(stderr, "  Seed:     %lld\n", (long long)args.seed);
    fprintf(stderr, "  Type:     %s\n", args.type.c_str());
    fprintf(stderr, "  DiT GPU:  %d\n", args.dit_gpu_layers);
    fprintf(stderr, "  Threads:  %d\n", args.threads);
    fprintf(stderr, "  TextCPU:  %s\n", args.text_cpu ? "on" : "off");
    fprintf(stderr, "  VAE3DCPU: %s\n", args.vae3d_cpu ? "on" : "off");
    fprintf(stderr, "  Output:   %s\n\n", args.output.c_str());

    // Create context
    struct kd_params params = kd_default_params();
    params.model_dir   = args.model_dir.c_str();
    params.vocab_dir   = args.vocab_dir.empty() ? nullptr : args.vocab_dir.c_str();
    params.dit_filename = args.dit_filename.empty() ? nullptr : args.dit_filename.c_str();
    params.n_threads   = args.threads;
    params.flash_attn  = args.flash_attn;
    params.wtype       = type_from_string(args.type);
    params.dit_gpu_layers = args.dit_gpu_layers;
    params.text_cpu    = args.text_cpu;
    params.vae3d_cpu   = args.vae3d_cpu;
    params.log_cb      = log_callback;

    kd_ctx_t* ctx = kd_ctx_create(params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    // Generate
    struct kd_generate_params gen = kd_default_generate_params();
    gen.prompt          = args.prompt.c_str();
    gen.negative_prompt = args.negative_prompt.c_str();
    gen.width           = args.width;
    gen.height          = args.height;
    gen.num_frames      = args.frames;
    gen.num_steps       = args.steps;
    gen.guidance_scale  = args.cfg_scale;
    gen.scheduler_scale = args.scheduler_scale;
    gen.seed            = args.seed;
    gen.mode            = (args.frames > 1) ? KD_MODE_TXT2VID : KD_MODE_TXT2IMG;

    kd_image_t* images = nullptr;
    int n_images = kd_generate(ctx, gen, &images);

    if (n_images <= 0 || !images) {
        fprintf(stderr, "Generation failed\n");
        kd_ctx_free(ctx);
        return 1;
    }

    // Save output
    const bool save_video_file = (n_images > 1) && is_video_path(args.output);
    std::string video_base = args.output;
    std::string video_ext;
    std::string frame_pattern;
    if (save_video_file) {
        auto dot = video_base.rfind('.');
        video_ext = video_base.substr(dot);
        video_base = video_base.substr(0, dot);
        frame_pattern = video_base + "_%04d.png";
        if (!ensure_parent_dir(frame_pattern)) {
            fprintf(stderr, "Failed to create output directory for %s\n", frame_pattern.c_str());
            kd_image_free(images, n_images);
            kd_ctx_free(ctx);
            return 1;
        }
    }

    for (int i = 0; i < n_images; i++) {
        std::string path;
        if (save_video_file) {
            char buf[32];
            snprintf(buf, sizeof(buf), "_%04d.png", i);
            path = video_base + buf;
        } else {
            path = args.output;
        }

        if (!save_video_file && n_images > 1) {
            // Insert frame number before extension
            auto dot = path.rfind('.');
            if (dot != std::string::npos) {
                char buf[32];
                snprintf(buf, sizeof(buf), "_%04d", i);
                path = path.substr(0, dot) + buf + path.substr(dot);
            }
        }

        if (!ensure_parent_dir(path)) {
            fprintf(stderr, "Failed to create output directory for %s\n", path.c_str());
            kd_image_free(images, n_images);
            kd_ctx_free(ctx);
            return 1;
        }

        int ok = 0;
        if (ends_with_ci(path, ".png")) {
            ok = stbi_write_png(path.c_str(), images[i].width, images[i].height,
                                images[i].channel, images[i].data,
                                images[i].width * images[i].channel);
        } else if (ends_with_ci(path, ".jpg")) {
            ok = stbi_write_jpg(path.c_str(), images[i].width, images[i].height,
                                images[i].channel, images[i].data, 95);
        } else if (ends_with_ci(path, ".bmp")) {
            ok = stbi_write_bmp(path.c_str(), images[i].width, images[i].height,
                                images[i].channel, images[i].data);
        } else {
            // Default to PNG
            path += ".png";
            if (!ensure_parent_dir(path)) {
                fprintf(stderr, "Failed to create output directory for %s\n", path.c_str());
                kd_image_free(images, n_images);
                kd_ctx_free(ctx);
                return 1;
            }
            ok = stbi_write_png(path.c_str(), images[i].width, images[i].height,
                                images[i].channel, images[i].data,
                                images[i].width * images[i].channel);
        }

        if (ok) {
            fprintf(stderr, "Saved: %s (%dx%d)\n", path.c_str(),
                    images[i].width, images[i].height);
        } else {
            fprintf(stderr, "Failed to save: %s\n", path.c_str());
        }
    }

    if (save_video_file) {
        std::string ffmpeg_cmd =
            "ffmpeg -y -loglevel error -framerate " + std::to_string(args.fps) +
            " -i " + quote_arg(frame_pattern) +
            " -c:v libx264 -pix_fmt yuv420p -crf 18 -r " + std::to_string(args.fps) +
            " " + quote_arg(args.output);

        int rc = std::system(ffmpeg_cmd.c_str());
        if (rc == 0) {
            fprintf(stderr, "Saved video: %s\n", args.output.c_str());
        } else {
            fprintf(stderr, "Failed to encode video with ffmpeg (code %d)\n", rc);
            fprintf(stderr, "Frame sequence remains at: %s\n", frame_pattern.c_str());
        }
    }

    kd_image_free(images, n_images);
    kd_ctx_free(ctx);

    return 0;
}
