#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct molmo2_rrt_preprocess_result {
    int image_w = 0;
    int image_h = 0;
    int tile_h = 0;
    int tile_w = 0;
    int high_grid_h = 0;
    int high_grid_w = 0;
    std::vector<uint8_t> images;
    std::vector<int64_t> token_pooling;
    std::vector<int64_t> image_tokens;
};

bool molmo2_rrt_preprocess_image(
        const std::string & image_path,
        molmo2_rrt_preprocess_result * result);

bool molmo2_rrt_run_vision(
        const std::string & vision_path,
        const std::vector<uint8_t> & images_u8,
        const std::vector<int64_t> & pooling_i64,
        const std::string & backend_name,
        int n_threads,
        std::vector<float> * result);

bool molmo2_rrt_build_input_ids(
        const std::string & model_path,
        const std::string & prompt,
        const std::vector<int64_t> & image_tokens,
        std::vector<int64_t> * input_ids);

bool molmo2_rrt_compose_prompt_embeddings(
        const std::string & model_path,
        const std::vector<int64_t> & input_ids,
        const std::vector<float> & vision_features,
        std::vector<float> * prompt_embeds);

bool molmo2_rrt_decode_embeddings(
        const std::string & model_path,
        const std::vector<float> & prompt_embeds,
        int n_prompt,
        int n_predict,
        int n_gpu_layers,
        bool flash_attn,
        int top_k,
        int sample_vocab,
        std::string * generated);
