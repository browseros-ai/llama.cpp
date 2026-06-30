#include "molmo2-rrt.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t IMAGE_PATCH_TOKEN_ID = 151938;

static void usage(char ** argv) {
    std::fprintf(stderr,
            "usage: %s --model text.gguf --vision vision.gguf --image image.png --prompt text "
            "[--vision-backend cpu|gpu|metal|mps] [--threads N] [-ngl gpu_layers] [-n predict] "
            "[--top-k K] [--sample-vocab N] [--no-fa]\n",
            argv[0]);
}

static std::string normalize_backend(std::string backend) {
    if (backend == "metal" || backend == "mps") {
        return "gpu";
    }
    return backend;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string vision_path;
    std::string image_path;
    std::string prompt;
    std::string vision_backend = "cpu";
    int n_threads = 8;
    int n_gpu_layers = 0;
    int n_predict = 96;
    int top_k = 0;
    int sample_vocab = 0;
    bool flash_attn = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
            vision_path = argv[++i];
        } else if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--vision-backend") == 0 && i + 1 < argc) {
            vision_backend = normalize_backend(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sample-vocab") == 0 && i + 1 < argc) {
            sample_vocab = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-fa") == 0) {
            flash_attn = false;
        } else {
            usage(argv);
            return 1;
        }
    }

    if (model_path.empty() || vision_path.empty() || image_path.empty() || prompt.empty() ||
        n_threads <= 0 || n_predict <= 0) {
        usage(argv);
        return 1;
    }

    molmo2_rrt_preprocess_result prep;
    if (!molmo2_rrt_preprocess_image(image_path, &prep)) {
        return 1;
    }
    const size_t n_image_patch_tokens = (size_t) std::count(
            prep.image_tokens.begin(), prep.image_tokens.end(), IMAGE_PATCH_TOKEN_ID);
    std::fprintf(stderr,
            "preprocess: image=%dx%d tiling=%dx%d high_grid=%dx%d image_tokens=%zu image_patch_tokens=%zu\n",
            prep.image_w,
            prep.image_h,
            prep.tile_h,
            prep.tile_w,
            prep.high_grid_h,
            prep.high_grid_w,
            prep.image_tokens.size(),
            n_image_patch_tokens);

    std::vector<float> vision_features;
    if (!molmo2_rrt_run_vision(
                vision_path,
                prep.images,
                prep.token_pooling,
                vision_backend,
                n_threads,
                &vision_features)) {
        return 1;
    }

    std::vector<int64_t> input_ids;
    if (!molmo2_rrt_build_input_ids(model_path, prompt, prep.image_tokens, &input_ids)) {
        return 1;
    }
    std::fprintf(stderr, "prompt_tokens=%zu\n", input_ids.size());

    std::vector<float> prompt_embeds;
    if (!molmo2_rrt_compose_prompt_embeddings(
                model_path,
                input_ids,
                vision_features,
                &prompt_embeds)) {
        return 1;
    }

    std::string generated;
    if (!molmo2_rrt_decode_embeddings(
                model_path,
                prompt_embeds,
                (int) input_ids.size(),
                n_predict,
                n_gpu_layers,
                flash_attn,
                top_k,
                sample_vocab,
                &generated)) {
        return 1;
    }

    std::printf("%s\n", generated.c_str());
    return 0;
}
