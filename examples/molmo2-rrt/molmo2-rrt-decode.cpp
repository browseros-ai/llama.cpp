#include "molmo2-rrt.h"

#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

static void print_usage(char ** argv) {
    std::fprintf(stderr,
            "usage: %s -m model.gguf -e prompt_embeds.f32.bin --n-tokens N [-n predict] [-ngl gpu_layers]\n",
            argv[0]);
}

static bool read_f32_file(const std::string & path, std::vector<float> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "error: failed to open embeddings file '%s'\n", path.c_str());
        return false;
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || size % (std::streamsize) sizeof(float) != 0) {
        std::fprintf(stderr, "error: embeddings file size is not a float32 multiple\n");
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize((size_t) size / sizeof(float));
    if (!in.read(reinterpret_cast<char *>(out.data()), size)) {
        std::fprintf(stderr, "error: failed to read embeddings file '%s'\n", path.c_str());
        return false;
    }
    return true;
}

static void fill_common_batch_fields(llama_batch & batch, int i, llama_pos pos, bool logits) {
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits ? 1 : 0;
}

static llama_token greedy_sample_prefix(const float * logits, int n_logits) {
    int best = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_logits; ++i) {
        if (logits[i] > best_logit) {
            best = i;
            best_logit = logits[i];
        }
    }
    return best;
}

bool molmo2_rrt_decode_embeddings(
        const std::string & model_path,
        const std::vector<float> & prompt_embd,
        int n_prompt,
        int n_predict,
        int n_gpu_layers,
        bool flash_attn,
        int top_k,
        int sample_vocab,
        std::string * generated_out) {
    if (!generated_out) {
        std::fprintf(stderr, "error: null generated output\n");
        return false;
    }
    if (model_path.empty() || n_prompt <= 0 || n_predict <= 0) {
        std::fprintf(stderr, "error: bad decode arguments\n");
        return false;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (sample_vocab <= 0) {
        sample_vocab = n_vocab;
    }
    if (sample_vocab > n_vocab) {
        std::fprintf(stderr, "error: sample vocab %d exceeds tokenizer vocab %d\n", sample_vocab, n_vocab);
        llama_model_free(model);
        return false;
    }
    if ((int64_t) prompt_embd.size() != (int64_t) n_prompt * n_embd) {
        std::fprintf(stderr,
                "error: embeddings vector has %zu floats, expected %lld for %d tokens x %d dims\n",
                prompt_embd.size(), (long long) n_prompt * n_embd, n_prompt, n_embd);
        llama_model_free(model);
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.n_ubatch = n_prompt;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: failed to create llama_context\n");
        llama_model_free(model);
        return false;
    }

    llama_batch embd_batch = llama_batch_init(n_prompt, n_embd, 1);
    embd_batch.n_tokens = n_prompt;
    std::memcpy(embd_batch.embd, prompt_embd.data(), prompt_embd.size() * sizeof(float));
    for (int i = 0; i < n_prompt; ++i) {
        fill_common_batch_fields(embd_batch, i, i, i == n_prompt - 1);
    }

    if (llama_decode(ctx, embd_batch) != 0) {
        std::fprintf(stderr, "error: llama_decode failed during embedding prefill\n");
        llama_batch_free(embd_batch);
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }
    llama_batch_free(embd_batch);

    if (top_k > 0) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "error: failed to read prefill logits\n");
        } else {
            std::vector<int> ids((size_t) sample_vocab);
            std::iota(ids.begin(), ids.end(), 0);
            const int k = std::min(top_k, sample_vocab);
            std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
                return logits[a] > logits[b];
            });
            std::fprintf(stderr, "top_logits:");
            for (int i = 0; i < k; ++i) {
                const int tok = ids[i];
                char buf[512];
                int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
                std::string piece;
                if (n >= 0) {
                    piece.assign(buf, (size_t) n);
                }
                std::fprintf(stderr, " [%d '%s' %.6f]", tok, piece.c_str(), logits[tok]);
            }
            std::fprintf(stderr, "\n");
        }
    }

    std::string generated;
    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "error: failed to read logits before token %d\n", i);
            break;
        }
        llama_token new_token = greedy_sample_prefix(logits, sample_vocab);
        if (new_token < 0 || new_token >= n_vocab) {
            std::fprintf(stderr, "error: sampled invalid token %d\n", new_token);
            break;
        }
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[512];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::vector<char> big((size_t) -n);
            n = llama_token_to_piece(vocab, new_token, big.data(), big.size(), 0, true);
            if (n < 0) {
                std::fprintf(stderr, "error: failed to convert token %d to piece\n", new_token);
                break;
            }
            generated.append(big.data(), (size_t) n);
        } else {
            generated.append(buf, (size_t) n);
        }

        llama_batch tok_batch = llama_batch_init(1, 0, 1);
        tok_batch.n_tokens = 1;
        tok_batch.token[0] = new_token;
        fill_common_batch_fields(tok_batch, 0, n_prompt + i, true);

        if (llama_decode(ctx, tok_batch) != 0) {
            std::fprintf(stderr, "error: llama_decode failed during token decode\n");
            llama_batch_free(tok_batch);
            break;
        }
        llama_batch_free(tok_batch);
    }

    *generated_out = generated;
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

#ifndef MOLMO2_RRT_NO_MAIN
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string embd_path;
    int n_prompt = -1;
    int n_predict = 96;
    int ngl = 0;
    int top_k = 0;
    int sample_vocab = 0;
    bool flash_attn = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            embd_path = argv[++i];
        } else if (std::strcmp(argv[i], "--n-tokens") == 0 && i + 1 < argc) {
            n_prompt = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sample-vocab") == 0 && i + 1 < argc) {
            sample_vocab = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-fa") == 0) {
            flash_attn = false;
        } else {
            print_usage(argv);
            return 1;
        }
    }

    if (model_path.empty() || embd_path.empty() || n_prompt <= 0 || n_predict <= 0) {
        print_usage(argv);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (sample_vocab <= 0) {
        sample_vocab = n_vocab;
    }
    if (sample_vocab > n_vocab) {
        std::fprintf(stderr, "error: sample vocab %d exceeds tokenizer vocab %d\n", sample_vocab, n_vocab);
        llama_model_free(model);
        return 1;
    }

    std::vector<float> prompt_embd;
    if (!read_f32_file(embd_path, prompt_embd)) {
        llama_model_free(model);
        return 1;
    }
    if ((int64_t) prompt_embd.size() != (int64_t) n_prompt * n_embd) {
        std::fprintf(stderr,
                "error: embeddings file has %zu floats, expected %lld for %d tokens x %d dims\n",
                prompt_embd.size(), (long long) n_prompt * n_embd, n_prompt, n_embd);
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.n_ubatch = n_prompt;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    llama_batch embd_batch = llama_batch_init(n_prompt, n_embd, 1);
    embd_batch.n_tokens = n_prompt;
    std::memcpy(embd_batch.embd, prompt_embd.data(), prompt_embd.size() * sizeof(float));
    for (int i = 0; i < n_prompt; ++i) {
        fill_common_batch_fields(embd_batch, i, i, i == n_prompt - 1);
    }

    if (llama_decode(ctx, embd_batch) != 0) {
        std::fprintf(stderr, "error: llama_decode failed during embedding prefill\n");
        llama_batch_free(embd_batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    llama_batch_free(embd_batch);

    if (top_k > 0) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "error: failed to read prefill logits\n");
        } else {
            std::vector<int> ids((size_t) sample_vocab);
            std::iota(ids.begin(), ids.end(), 0);
            const int k = std::min(top_k, sample_vocab);
            std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
                return logits[a] > logits[b];
            });
            std::fprintf(stderr, "top_logits:");
            for (int i = 0; i < k; ++i) {
                const int tok = ids[i];
                char buf[512];
                int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
                std::string piece;
                if (n >= 0) {
                    piece.assign(buf, (size_t) n);
                }
                std::fprintf(stderr, " [%d '%s' %.6f]", tok, piece.c_str(), logits[tok]);
            }
            std::fprintf(stderr, "\n");
        }
    }

    std::string generated;
    std::vector<llama_token> generated_tokens;
    llama_token new_token = LLAMA_TOKEN_NULL;

    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "error: failed to read logits before token %d\n", i);
            break;
        }
        new_token = greedy_sample_prefix(logits, sample_vocab);
        if (new_token < 0 || new_token >= n_vocab) {
            std::fprintf(stderr, "error: sampled invalid token %d\n", new_token);
            break;
        }
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        generated_tokens.push_back(new_token);
        char buf[512];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::vector<char> big((size_t) -n);
            n = llama_token_to_piece(vocab, new_token, big.data(), big.size(), 0, true);
            if (n < 0) {
                std::fprintf(stderr, "error: failed to convert token %d to piece\n", new_token);
                break;
            }
            generated.append(big.data(), (size_t) n);
        } else {
            generated.append(buf, (size_t) n);
        }

        llama_batch tok_batch = llama_batch_init(1, 0, 1);
        tok_batch.n_tokens = 1;
        tok_batch.token[0] = new_token;
        fill_common_batch_fields(tok_batch, 0, n_prompt + i, true);

        if (llama_decode(ctx, tok_batch) != 0) {
            std::fprintf(stderr, "error: llama_decode failed during token decode\n");
            llama_batch_free(tok_batch);
            break;
        }
        llama_batch_free(tok_batch);
    }

    std::printf("%s\n", generated.c_str());
    std::fprintf(stderr, "generated_tokens:");
    for (llama_token tok : generated_tokens) {
        std::fprintf(stderr, " %d", tok);
    }
    std::fprintf(stderr, "\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
#endif
