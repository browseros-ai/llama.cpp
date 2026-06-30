#include "molmo2-rrt.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int EMBD = 1024;
constexpr int64_t IMAGE_PATCH_TOKEN_ID = 151938;
constexpr int64_t QWEN_IM_START_TOKEN_ID = 151644;

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct gguf_context_deleter {
    void operator()(gguf_context * ctx) const {
        if (ctx) {
            gguf_free(ctx);
        }
    }
};

struct llama_model_deleter {
    void operator()(llama_model * model) const {
        if (model) {
            llama_model_free(model);
        }
    }
};

using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;
using gguf_context_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;
using llama_model_ptr = std::unique_ptr<llama_model, llama_model_deleter>;

static void usage(char ** argv) {
    std::fprintf(stderr,
            "usage: %s --model text.gguf (--input-ids input_ids.i64.bin | --prompt text --image-tokens image_tokens.i64.bin) --vision-features vision_features.f32.bin --out prompt_embeds.f32.bin [--dump-input-ids input_ids.i64.bin] [--ref prompt_embeds.f32.bin]\n",
            argv[0]);
}

template<typename T>
static bool read_file(const std::string & path, std::vector<T> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "error: failed to open '%s'\n", path.c_str());
        return false;
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || size % (std::streamsize) sizeof(T) != 0) {
        std::fprintf(stderr, "error: bad file size for '%s'\n", path.c_str());
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize((size_t) size / sizeof(T));
    if (size > 0 && !in.read(reinterpret_cast<char *>(out.data()), size)) {
        std::fprintf(stderr, "error: failed to read '%s'\n", path.c_str());
        return false;
    }
    return true;
}

template<typename T>
static bool write_file(const std::string & path, const std::vector<T> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "error: failed to create '%s'\n", path.c_str());
        return false;
    }
    if (!values.empty()) {
        out.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) (values.size() * sizeof(T)));
    }
    return (bool) out;
}

static void compare(const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "prompt_embed size mismatch: got=%zu ref=%zu\n", got.size(), ref.size());
        return;
    }
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    size_t max_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        sum_abs += d;
        sum_sq += (double) d * d;
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
    }
    std::fprintf(stderr, "prompt_embed_diff: mean_abs=%.8g rmse=%.8g max_abs=%.8g max_i=%zu got=%.8g ref=%.8g\n",
            sum_abs / got.size(),
            std::sqrt(sum_sq / got.size()),
            max_abs,
            max_i,
            got[max_i],
            ref[max_i]);
}

static bool append_tokenized(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special,
        std::vector<int64_t> & out) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    if (n < 0) {
        std::fprintf(stderr, "error: failed to size tokenization for '%s'\n", text.c_str());
        return false;
    }
    std::vector<llama_token> tmp((size_t) n);
    const int actual = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tmp.data(), (int32_t) tmp.size(), add_special, parse_special);
    if (actual < 0) {
        std::fprintf(stderr, "error: failed to tokenize '%s'\n", text.c_str());
        return false;
    }
    tmp.resize((size_t) actual);
    out.reserve(out.size() + tmp.size());
    for (llama_token tok : tmp) {
        out.push_back(tok);
    }
    return true;
}

static bool build_prompt_input_ids(
        const std::string & model_path,
        const std::string & prompt,
        const std::string & image_tokens_path,
        std::vector<int64_t> & input_ids) {
    std::vector<int64_t> image_tokens;
    if (!read_file(image_tokens_path, image_tokens)) {
        return false;
    }

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model_ptr tokenizer_model(llama_model_load_from_file(model_path.c_str(), params));
    if (!tokenizer_model) {
        std::fprintf(stderr, "error: failed to load tokenizer from '%s'\n", model_path.c_str());
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(tokenizer_model.get());
    if (!vocab) {
        std::fprintf(stderr, "error: failed to get tokenizer vocab\n");
        return false;
    }

    input_ids.clear();
    input_ids.reserve(1 + image_tokens.size() + prompt.size() / 3 + 16);
    const int64_t im_end = llama_vocab_bos(vocab);
    input_ids.push_back(im_end);
    input_ids.insert(input_ids.end(), image_tokens.begin(), image_tokens.end());
    input_ids.push_back(QWEN_IM_START_TOKEN_ID);
    if (!append_tokenized(vocab, "user\n", false, false, input_ids) ||
        !append_tokenized(vocab, prompt, false, false, input_ids)) {
        return false;
    }
    input_ids.push_back(im_end);
    if (!append_tokenized(vocab, "\n", false, false, input_ids)) {
        return false;
    }
    input_ids.push_back(QWEN_IM_START_TOKEN_ID);
    if (!append_tokenized(vocab, "assistant\n", false, false, input_ids)) {
        return false;
    }
    return true;
}

} // namespace

bool molmo2_rrt_build_input_ids(
        const std::string & model_path,
        const std::string & prompt,
        const std::vector<int64_t> & image_tokens,
        std::vector<int64_t> * input_ids_out) {
    if (!input_ids_out) {
        std::fprintf(stderr, "error: null input id output\n");
        return false;
    }

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model_ptr tokenizer_model(llama_model_load_from_file(model_path.c_str(), params));
    if (!tokenizer_model) {
        std::fprintf(stderr, "error: failed to load tokenizer from '%s'\n", model_path.c_str());
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(tokenizer_model.get());
    if (!vocab) {
        std::fprintf(stderr, "error: failed to get tokenizer vocab\n");
        return false;
    }

    std::vector<int64_t> input_ids;
    input_ids.reserve(1 + image_tokens.size() + prompt.size() / 3 + 16);
    const int64_t im_end = llama_vocab_bos(vocab);
    input_ids.push_back(im_end);
    input_ids.insert(input_ids.end(), image_tokens.begin(), image_tokens.end());
    input_ids.push_back(QWEN_IM_START_TOKEN_ID);
    if (!append_tokenized(vocab, "user\n", false, false, input_ids) ||
        !append_tokenized(vocab, prompt, false, false, input_ids)) {
        return false;
    }
    input_ids.push_back(im_end);
    if (!append_tokenized(vocab, "\n", false, false, input_ids)) {
        return false;
    }
    input_ids.push_back(QWEN_IM_START_TOKEN_ID);
    if (!append_tokenized(vocab, "assistant\n", false, false, input_ids)) {
        return false;
    }
    *input_ids_out = std::move(input_ids);
    return true;
}

bool molmo2_rrt_compose_prompt_embeddings(
        const std::string & model_path,
        const std::vector<int64_t> & input_ids,
        const std::vector<float> & vision_features,
        std::vector<float> * prompt_embeds) {
    if (!prompt_embeds) {
        std::fprintf(stderr, "error: null prompt embedding output\n");
        return false;
    }
    if (vision_features.size() % EMBD != 0) {
        std::fprintf(stderr, "error: vision feature size is not divisible by %d\n", EMBD);
        return false;
    }
    const int n_vision = (int) (vision_features.size() / EMBD);

    ggml_context * weights_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx =*/ &weights_raw,
    };
    gguf_context_ptr gguf(gguf_init_from_file(model_path.c_str(), params));
    ggml_context_ptr weights(weights_raw);
    if (!gguf || !weights) {
        std::fprintf(stderr, "error: failed to load text GGUF '%s'\n", model_path.c_str());
        return false;
    }

    ggml_tensor * token_embd = ggml_get_tensor(weights.get(), "token_embd.weight");
    if (!token_embd) {
        std::fprintf(stderr, "error: missing token_embd.weight\n");
        return false;
    }
    if (token_embd->ne[0] != EMBD) {
        std::fprintf(stderr, "error: token embedding dim is %lld, expected %d\n", (long long) token_embd->ne[0], EMBD);
        return false;
    }

    std::vector<float> out((size_t) input_ids.size() * EMBD);
    int image_row = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const int64_t tok = input_ids[i];
        if (tok < 0 || tok >= token_embd->ne[1]) {
            std::fprintf(stderr, "error: token id %lld out of range at position %zu\n", (long long) tok, i);
            return false;
        }
        float * dst = out.data() + i * EMBD;
        if (token_embd->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *) token_embd->data + tok * EMBD;
            ggml_fp16_to_fp32_row(src, dst, EMBD);
        } else if (token_embd->type == GGML_TYPE_F32) {
            const float * src = (const float *) token_embd->data + tok * EMBD;
            std::memcpy(dst, src, EMBD * sizeof(float));
        } else {
            std::fprintf(stderr, "error: unsupported token_embd type %s\n", ggml_type_name(token_embd->type));
            return false;
        }
        if (tok == IMAGE_PATCH_TOKEN_ID) {
            if (image_row >= n_vision) {
                std::fprintf(stderr, "error: more image tokens than vision rows\n");
                return false;
            }
            const float * vf = vision_features.data() + (size_t) image_row * EMBD;
            for (int d = 0; d < EMBD; ++d) {
                dst[d] += vf[d];
            }
            ++image_row;
        }
    }

    if (image_row != n_vision) {
        std::fprintf(stderr, "error: consumed %d vision rows, expected %d\n", image_row, n_vision);
        return false;
    }

    std::fprintf(stderr, "prompt_embeds=[%zu,%d] image_rows=%d\n", input_ids.size(), EMBD, image_row);
    *prompt_embeds = std::move(out);
    return true;
}

#ifndef MOLMO2_RRT_NO_MAIN
int main(int argc, char ** argv) {
    std::string model_path;
    std::string input_ids_path;
    std::string prompt;
    std::string image_tokens_path;
    std::string vision_features_path;
    std::string out_path;
    std::string dump_input_ids_path;
    std::string ref_path;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--input-ids") == 0 && i + 1 < argc) {
            input_ids_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--image-tokens") == 0 && i + 1 < argc) {
            image_tokens_path = argv[++i];
        } else if (std::strcmp(argv[i], "--vision-features") == 0 && i + 1 < argc) {
            vision_features_path = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-input-ids") == 0 && i + 1 < argc) {
            dump_input_ids_path = argv[++i];
        } else if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        } else {
            usage(argv);
            return 1;
        }
    }

    if (model_path.empty() || vision_features_path.empty() || out_path.empty() ||
        (input_ids_path.empty() && (prompt.empty() || image_tokens_path.empty()))) {
        usage(argv);
        return 1;
    }

    std::vector<int64_t> input_ids;
    std::vector<float> vision_features;
    if (!input_ids_path.empty()) {
        if (!read_file(input_ids_path, input_ids)) {
            return 1;
        }
    } else if (!build_prompt_input_ids(model_path, prompt, image_tokens_path, input_ids)) {
        return 1;
    }
    if (!dump_input_ids_path.empty() && !write_file(dump_input_ids_path, input_ids)) {
        return 1;
    }
    if (!read_file(vision_features_path, vision_features)) {
        return 1;
    }
    if (vision_features.size() % EMBD != 0) {
        std::fprintf(stderr, "error: vision feature size is not divisible by %d\n", EMBD);
        return 1;
    }
    const int n_vision = (int) (vision_features.size() / EMBD);

    ggml_context * weights_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx =*/ &weights_raw,
    };
    gguf_context_ptr gguf(gguf_init_from_file(model_path.c_str(), params));
    ggml_context_ptr weights(weights_raw);
    if (!gguf || !weights) {
        std::fprintf(stderr, "error: failed to load text GGUF '%s'\n", model_path.c_str());
        return 1;
    }

    ggml_tensor * token_embd = ggml_get_tensor(weights.get(), "token_embd.weight");
    if (!token_embd) {
        std::fprintf(stderr, "error: missing token_embd.weight\n");
        return 1;
    }
    if (token_embd->ne[0] != EMBD) {
        std::fprintf(stderr, "error: token embedding dim is %lld, expected %d\n", (long long) token_embd->ne[0], EMBD);
        return 1;
    }

    std::vector<float> out((size_t) input_ids.size() * EMBD);
    int image_row = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const int64_t tok = input_ids[i];
        if (tok < 0 || tok >= token_embd->ne[1]) {
            std::fprintf(stderr, "error: token id %lld out of range at position %zu\n", (long long) tok, i);
            return 1;
        }
        float * dst = out.data() + i * EMBD;
        if (token_embd->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *) token_embd->data + tok * EMBD;
            ggml_fp16_to_fp32_row(src, dst, EMBD);
        } else if (token_embd->type == GGML_TYPE_F32) {
            const float * src = (const float *) token_embd->data + tok * EMBD;
            std::memcpy(dst, src, EMBD * sizeof(float));
        } else {
            std::fprintf(stderr, "error: unsupported token_embd type %s\n", ggml_type_name(token_embd->type));
            return 1;
        }
        if (tok == IMAGE_PATCH_TOKEN_ID) {
            if (image_row >= n_vision) {
                std::fprintf(stderr, "error: more image tokens than vision rows\n");
                return 1;
            }
            const float * vf = vision_features.data() + (size_t) image_row * EMBD;
            for (int d = 0; d < EMBD; ++d) {
                dst[d] += vf[d];
            }
            ++image_row;
        }
    }

    if (image_row != n_vision) {
        std::fprintf(stderr, "error: consumed %d vision rows, expected %d\n", image_row, n_vision);
        return 1;
    }
    if (!write_file(out_path, out)) {
        return 1;
    }
    std::printf("prompt_embeds=[%zu,%d] image_rows=%d\n", input_ids.size(), EMBD, image_row);

    if (!ref_path.empty()) {
        std::vector<float> ref;
        if (!read_file(ref_path, ref)) {
            return 1;
        }
        compare(out, ref);
    }
    return 0;
}
#endif
