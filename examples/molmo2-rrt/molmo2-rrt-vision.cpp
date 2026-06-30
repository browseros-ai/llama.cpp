#include "molmo2-rrt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int VIT_DIM = 1152;
constexpr int VIT_LAYERS = 25;
constexpr int VIT_HEADS = 16;
constexpr int VIT_HEAD_DIM = 72;
constexpr int VIT_PATCHES = 729;
constexpr int PATCH_PIXELS = 588;
constexpr int POOL = 4;
constexpr int POOL_DIM = VIT_DIM * 2;
constexpr int OUT_DIM = 1024;
constexpr float NORM_EPS = 1.0e-6f;

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

struct backend_deleter {
    void operator()(ggml_backend * backend) const {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct backend_buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;
using gguf_context_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using backend_buffer_ptr = std::unique_ptr<ggml_backend_buffer, backend_buffer_deleter>;

static void usage(char ** argv) {
    std::fprintf(stderr,
            "usage: %s --vision vision.gguf --artifacts-dir artifacts [--ref vision_features.f32.bin] [--out native_features.f32.bin] [--threads N] [--backend cpu|gpu]\n",
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
        std::fprintf(stderr, "error: file '%s' has bad size\n", path.c_str());
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

static std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

static ggml_tensor * require_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        std::fprintf(stderr, "error: missing tensor '%s'\n", name);
        std::exit(1);
    }
    return t;
}

static backend_buffer_ptr load_weights_to_backend(
        const std::string & path,
        gguf_context * gguf,
        ggml_context * weights,
        ggml_backend * backend) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(weights, buft));
    if (!buffer) {
        std::fprintf(stderr, "error: failed to allocate weight buffer on %s\n", ggml_backend_name(backend));
        std::exit(1);
    }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "error: failed to open '%s' for tensor loading\n", path.c_str());
        std::exit(1);
    }
    std::vector<uint8_t> tmp;
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = require_tensor(weights, name);
        const size_t nbytes = ggml_nbytes(t);
        const size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
        tmp.resize(nbytes);
        in.seekg((std::streamoff) offset, std::ios::beg);
        if (!in.read(reinterpret_cast<char *>(tmp.data()), (std::streamsize) nbytes)) {
            std::fprintf(stderr, "error: failed to read tensor '%s'\n", name);
            std::exit(1);
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
    }
    return buffer;
}

static std::string block_name(int il, const char * suffix) {
    return "v.blk." + std::to_string(il) + "." + suffix;
}

static ggml_tensor * wt(ggml_context * weights, const std::string & name) {
    return require_tensor(weights, name.c_str());
}

static ggml_tensor * linear(
        ggml_context * ctx,
        ggml_context * weights,
        const std::string & w_name,
        const std::string & b_name,
        ggml_tensor * x,
        bool has_bias = true) {
    ggml_tensor * y = ggml_mul_mat(ctx, wt(weights, w_name), x);
    if (has_bias) {
        y = ggml_add(ctx, y, wt(weights, b_name));
    }
    return y;
}

static ggml_tensor * layer_norm(
        ggml_context * ctx,
        ggml_context * weights,
        ggml_tensor * x,
        const std::string & w_name,
        const std::string & b_name) {
    ggml_tensor * y = ggml_norm(ctx, x, NORM_EPS);
    y = ggml_mul(ctx, y, wt(weights, w_name));
    y = ggml_add(ctx, y, wt(weights, b_name));
    return y;
}

static ggml_tensor * attention(
        ggml_context * ctx,
        ggml_context * weights,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        int n_q,
        int n_k,
        int batch,
        const std::string & wo_name,
        const std::string & wo_b_name,
        ggml_tensor * mask) {
    q = ggml_reshape_4d(ctx, q, VIT_HEAD_DIM, VIT_HEADS, n_q, batch);
    k = ggml_reshape_4d(ctx, k, VIT_HEAD_DIM, VIT_HEADS, n_k, batch);
    v = ggml_reshape_4d(ctx, v, VIT_HEAD_DIM, VIT_HEADS, n_k, batch);

    q = ggml_cast(ctx, q, GGML_TYPE_F32);
    k = ggml_cast(ctx, k, GGML_TYPE_F32);
    v = ggml_cast(ctx, v, GGML_TYPE_F32);

    ggml_tensor * q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, k_perm, q_perm);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float) VIT_HEAD_DIM), 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(ctx, v_perm, kq);
    ggml_tensor * out = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), VIT_DIM, n_q * batch);
    out = linear(ctx, weights, wo_name, wo_b_name, out, true);
    return out;
}

static ggml_tensor * vit_block(ggml_context * ctx, ggml_context * weights, ggml_tensor * x, int il, int n_crops) {
    const std::string p = "v.blk." + std::to_string(il) + ".";
    ggml_tensor * residual = x;
    ggml_tensor * cur = layer_norm(ctx, weights, x, p + "attn_norm.weight", p + "attn_norm.bias");

    ggml_tensor * q = linear(ctx, weights, p + "attn_wq.weight", p + "attn_wq.bias", cur);
    ggml_tensor * k = linear(ctx, weights, p + "attn_wk.weight", p + "attn_wk.bias", cur);
    ggml_tensor * v = linear(ctx, weights, p + "attn_wv.weight", p + "attn_wv.bias", cur);
    cur = attention(ctx, weights, q, k, v, VIT_PATCHES, VIT_PATCHES, n_crops,
            p + "attn_wo.weight", p + "attn_wo.bias", nullptr);
    cur = ggml_add(ctx, cur, residual);

    residual = cur;
    cur = layer_norm(ctx, weights, cur, p + "ffn_norm.weight", p + "ffn_norm.bias");
    cur = linear(ctx, weights, p + "ffn_up.weight", p + "ffn_up.bias", cur);
    cur = ggml_gelu(ctx, cur);
    cur = linear(ctx, weights, p + "ffn_down.weight", p + "ffn_down.bias", cur);
    cur = ggml_add(ctx, residual, cur);
    return cur;
}

static ggml_tensor * build_graph(
        ggml_context * ctx,
        ggml_context * weights,
        int n_crops,
        int n_tokens,
        ggml_tensor ** inp_images,
        ggml_tensor ** inp_pool_idx,
        ggml_tensor ** inp_pool_mask,
        ggml_tensor ** inp_inv_denom,
        ggml_tensor ** inp_attn_mask) {
    *inp_images = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PATCH_PIXELS, VIT_PATCHES, n_crops);
    ggml_set_name(*inp_images, "molmo2_images_f32");
    ggml_set_input(*inp_images);

    *inp_pool_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens * POOL);
    ggml_set_name(*inp_pool_idx, "molmo2_pool_idx");
    ggml_set_input(*inp_pool_idx);

    *inp_pool_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, POOL, 1, n_tokens);
    ggml_set_name(*inp_pool_mask, "molmo2_pool_mask");
    ggml_set_input(*inp_pool_mask);

    *inp_inv_denom = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_name(*inp_inv_denom, "molmo2_inv_denom");
    ggml_set_input(*inp_inv_denom);

    *inp_attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, POOL, 1, 1, n_tokens);
    ggml_set_name(*inp_attn_mask, "molmo2_pool_attn_mask");
    ggml_set_input(*inp_attn_mask);

    ggml_tensor * cur = ggml_reshape_2d(ctx, *inp_images, PATCH_PIXELS, VIT_PATCHES * n_crops);
    cur = linear(ctx, weights,
            "v.patch.weight",
            "v.patch.bias",
            cur,
            true);
    cur = ggml_reshape_3d(ctx, cur, VIT_DIM, VIT_PATCHES, n_crops);
    cur = ggml_add(ctx, cur, ggml_cast(ctx, wt(weights, "v.pos"), GGML_TYPE_F32));
    cur = ggml_reshape_2d(ctx, cur, VIT_DIM, VIT_PATCHES * n_crops);

    ggml_tensor * layer18 = nullptr;
    ggml_tensor * layer24 = nullptr;
    for (int il = 0; il < VIT_LAYERS; ++il) {
        cur = vit_block(ctx, weights, cur, il, n_crops);
        if (il == 18) {
            layer18 = cur;
        } else if (il == 24) {
            layer24 = cur;
        }
    }
    if (!layer18 || !layer24) {
        std::fprintf(stderr, "internal error: missing captured ViT layers\n");
        std::exit(1);
    }

    // Python order for vit_layers=[-3,-9] is [24,18].
    ggml_tensor * features = ggml_concat(ctx, layer24, layer18, 0);

    ggml_tensor * gathered = ggml_get_rows(ctx, features, *inp_pool_idx);
    gathered = ggml_reshape_3d(ctx, gathered, POOL_DIM, POOL, n_tokens);
    ggml_tensor * grouped = ggml_cont(ctx, ggml_permute(ctx, gathered, 1, 0, 2, 3)); // [4, 2304, n_tokens]
    ggml_tensor * masked_grouped = ggml_mul(ctx, grouped, *inp_pool_mask);

    ggml_tensor * query = ggml_sum_rows(ctx, masked_grouped); // [1, 2304, n_tokens]
    query = ggml_mul(ctx, query, *inp_inv_denom);
    query = ggml_reshape_2d(ctx, query, POOL_DIM, n_tokens);

    ggml_tensor * to_pool = ggml_cont(ctx, ggml_permute(ctx, masked_grouped, 1, 0, 2, 3)); // [2304, 4, n_tokens]
    ggml_tensor * to_pool_2d = ggml_reshape_2d(ctx, to_pool, POOL_DIM, POOL * n_tokens);

    ggml_tensor * q = linear(ctx, weights,
            "v.pool.wq.weight",
            "v.pool.wq.bias",
            query,
            true);
    ggml_tensor * k = linear(ctx, weights,
            "v.pool.wk.weight",
            "v.pool.wk.bias",
            to_pool_2d,
            true);
    ggml_tensor * v = linear(ctx, weights,
            "v.pool.wv.weight",
            "v.pool.wv.bias",
            to_pool_2d,
            true);
    cur = attention(ctx, weights, q, k, v, 1, POOL, n_tokens,
            "v.pool.wo.weight",
            "v.pool.wo.bias",
            *inp_attn_mask);

    ggml_tensor * sw = linear(ctx, weights,
            "v.proj.w1.weight",
            "",
            cur,
            false);
    sw = ggml_silu(ctx, sw);
    ggml_tensor * gate = linear(ctx, weights,
            "v.proj.w3.weight",
            "",
            cur,
            false);
    cur = ggml_mul(ctx, sw, gate);
    cur = linear(ctx, weights,
            "v.proj.w2.weight",
            "",
            cur,
            false);
    ggml_set_name(cur, "molmo2_vision_features");
    ggml_set_output(cur);
    return cur;
}

static void compare_features(const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "feature size mismatch: got=%zu ref=%zu\n", got.size(), ref.size());
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
    std::fprintf(stderr, "feature_diff: mean_abs=%.8g rmse=%.8g max_abs=%.8g max_i=%zu got=%.8g ref=%.8g\n",
            sum_abs / got.size(),
            std::sqrt(sum_sq / got.size()),
            max_abs,
            max_i,
            got[max_i],
            ref[max_i]);
}

} // namespace

bool molmo2_rrt_run_vision(
        const std::string & vision_path,
        const std::vector<uint8_t> & images_u8,
        const std::vector<int64_t> & pooling_i64,
        const std::string & backend_name,
        int n_threads,
        std::vector<float> * result_out) {
    if (!result_out) {
        std::fprintf(stderr, "error: null vision result\n");
        return false;
    }
    if (images_u8.size() % (VIT_PATCHES * PATCH_PIXELS) != 0 || pooling_i64.size() % POOL != 0) {
        std::fprintf(stderr, "bad artifact shapes\n");
        return false;
    }
    const int n_crops = (int) (images_u8.size() / (VIT_PATCHES * PATCH_PIXELS));
    const int n_tokens = (int) (pooling_i64.size() / POOL);

    std::vector<float> images_f32(images_u8.size());
    for (size_t i = 0; i < images_u8.size(); ++i) {
        images_f32[i] = ((float) images_u8[i] / 255.0f) * 2.0f - 1.0f;
    }

    std::vector<int32_t> pool_idx(pooling_i64.size());
    std::vector<float> pool_mask(pooling_i64.size());
    std::vector<float> attn_mask(pooling_i64.size());
    std::vector<float> inv_denom(n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        int valid = 0;
        for (int j = 0; j < POOL; ++j) {
            const int64_t src = pooling_i64[(size_t) t * POOL + j];
            pool_idx[(size_t) t * POOL + j] = (int32_t) std::max<int64_t>(src, 0);
            const bool ok = src >= 0;
            pool_mask[(size_t) t * POOL + j] = ok ? 1.0f : 0.0f;
            attn_mask[(size_t) t * POOL + j] = ok ? 0.0f : -1.0e30f;
            valid += ok ? 1 : 0;
        }
        inv_denom[t] = 1.0f / (float) std::max(valid, 1);
    }

    ggml_backend_load_all();
    backend_ptr backend;
    if (backend_name == "cpu") {
        backend.reset(ggml_backend_cpu_init());
    } else if (backend_name == "gpu") {
        backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
        if (!backend) {
            backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr));
        }
    } else {
        std::fprintf(stderr, "error: unknown backend '%s'\n", backend_name.c_str());
        return false;
    }
    if (!backend) {
        std::fprintf(stderr, "error: failed to init backend '%s'\n", backend_name.c_str());
        return false;
    }
    if (backend_name == "cpu") {
        ggml_backend_cpu_set_n_threads(backend.get(), n_threads);
    }

    ggml_context * weights_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &weights_raw,
    };
    gguf_context_ptr gguf(gguf_init_from_file(vision_path.c_str(), params));
    ggml_context_ptr weights(weights_raw);
    if (!gguf || !weights) {
        std::fprintf(stderr, "error: failed to load vision GGUF metadata '%s'\n", vision_path.c_str());
        return false;
    }
    backend_buffer_ptr weights_buffer = load_weights_to_backend(vision_path, gguf.get(), weights.get(), backend.get());
    std::fprintf(stderr, "vision_backend: %s weights_buffer=%.2f MiB\n",
            ggml_backend_name(backend.get()),
            (double) ggml_backend_buffer_get_size(weights_buffer.get()) / (1024.0 * 1024.0));

    ggml_init_params gparams = {
        /*.mem_size =*/ 256ull * 1024ull * 1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ggml_context_ptr gctx(ggml_init(gparams));
    ggml_cgraph * graph = ggml_new_graph_custom(gctx.get(), 8192, false);

    ggml_tensor * inp_images = nullptr;
    ggml_tensor * inp_pool_idx = nullptr;
    ggml_tensor * inp_pool_mask = nullptr;
    ggml_tensor * inp_inv_denom = nullptr;
    ggml_tensor * inp_attn_mask = nullptr;
    ggml_tensor * out = build_graph(
            gctx.get(), weights.get(), n_crops, n_tokens,
            &inp_images, &inp_pool_idx, &inp_pool_mask, &inp_inv_denom, &inp_attn_mask);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()));
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        std::fprintf(stderr, "error: failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        return false;
    }
    std::fprintf(stderr, "vision_graph: crops=%d tokens=%d compute_buffer=%.2f MiB\n",
            n_crops, n_tokens, (double) ggml_gallocr_get_buffer_size(allocr, 0) / (1024.0 * 1024.0));

    ggml_backend_tensor_set(inp_images, images_f32.data(), 0, images_f32.size() * sizeof(float));
    ggml_backend_tensor_set(inp_pool_idx, pool_idx.data(), 0, pool_idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pool_mask, pool_mask.data(), 0, pool_mask.size() * sizeof(float));
    ggml_backend_tensor_set(inp_inv_denom, inv_denom.data(), 0, inv_denom.size() * sizeof(float));
    ggml_backend_tensor_set(inp_attn_mask, attn_mask.data(), 0, attn_mask.size() * sizeof(float));

    const enum ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "error: graph compute failed with status %d\n", (int) status);
        ggml_gallocr_free(allocr);
        return false;
    }

    std::vector<float> result((size_t) OUT_DIM * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    std::fprintf(stderr, "vision_features=[%d,%d]\n", n_tokens, OUT_DIM);

    ggml_gallocr_free(allocr);
    *result_out = std::move(result);
    return true;
}

#ifndef MOLMO2_RRT_NO_MAIN
int main(int argc, char ** argv) {
    std::string vision_path;
    std::string artifacts_dir;
    std::string ref_path;
    std::string out_path;
    std::string backend_name = "cpu";
    int n_threads = 8;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
            vision_path = argv[++i];
        } else if (std::strcmp(argv[i], "--artifacts-dir") == 0 && i + 1 < argc) {
            artifacts_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else {
            usage(argv);
            return 1;
        }
    }

    if (vision_path.empty() || artifacts_dir.empty()) {
        usage(argv);
        return 1;
    }

    std::vector<uint8_t> images_u8;
    std::vector<int64_t> pooling_i64;
    if (!read_file(join_path(artifacts_dir, "images.bin"), images_u8) ||
        !read_file(join_path(artifacts_dir, "token_pooling.i64.bin"), pooling_i64)) {
        return 1;
    }
    if (images_u8.size() % (VIT_PATCHES * PATCH_PIXELS) != 0 || pooling_i64.size() % POOL != 0) {
        std::fprintf(stderr, "bad artifact shapes\n");
        return 1;
    }
    const int n_crops = (int) (images_u8.size() / (VIT_PATCHES * PATCH_PIXELS));
    const int n_tokens = (int) (pooling_i64.size() / POOL);

    std::vector<float> images_f32(images_u8.size());
    for (size_t i = 0; i < images_u8.size(); ++i) {
        images_f32[i] = ((float) images_u8[i] / 255.0f) * 2.0f - 1.0f;
    }

    std::vector<int32_t> pool_idx(pooling_i64.size());
    std::vector<float> pool_mask(pooling_i64.size());
    std::vector<float> attn_mask(pooling_i64.size());
    std::vector<float> inv_denom(n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        int valid = 0;
        for (int j = 0; j < POOL; ++j) {
            const int64_t src = pooling_i64[(size_t) t * POOL + j];
            pool_idx[(size_t) t * POOL + j] = (int32_t) std::max<int64_t>(src, 0);
            const bool ok = src >= 0;
            pool_mask[(size_t) t * POOL + j] = ok ? 1.0f : 0.0f;
            attn_mask[(size_t) t * POOL + j] = ok ? 0.0f : -1.0e30f;
            valid += ok ? 1 : 0;
        }
        inv_denom[t] = 1.0f / (float) std::max(valid, 1);
    }

    ggml_backend_load_all();
    backend_ptr backend;
    if (backend_name == "cpu") {
        backend.reset(ggml_backend_cpu_init());
    } else if (backend_name == "gpu") {
        backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
        if (!backend) {
            backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr));
        }
    } else {
        std::fprintf(stderr, "error: unknown backend '%s'\n", backend_name.c_str());
        return 1;
    }
    if (!backend) {
        std::fprintf(stderr, "error: failed to init backend '%s'\n", backend_name.c_str());
        return 1;
    }
    if (backend_name == "cpu") {
        ggml_backend_cpu_set_n_threads(backend.get(), n_threads);
    }

    ggml_context * weights_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &weights_raw,
    };
    gguf_context_ptr gguf(gguf_init_from_file(vision_path.c_str(), params));
    ggml_context_ptr weights(weights_raw);
    if (!gguf || !weights) {
        std::fprintf(stderr, "error: failed to load vision GGUF metadata '%s'\n", vision_path.c_str());
        return 1;
    }
    backend_buffer_ptr weights_buffer = load_weights_to_backend(vision_path, gguf.get(), weights.get(), backend.get());
    std::fprintf(stderr, "vision_backend: %s weights_buffer=%.2f MiB\n",
            ggml_backend_name(backend.get()),
            (double) ggml_backend_buffer_get_size(weights_buffer.get()) / (1024.0 * 1024.0));

    ggml_init_params gparams = {
        /*.mem_size =*/ 256ull * 1024ull * 1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ggml_context_ptr gctx(ggml_init(gparams));
    ggml_cgraph * graph = ggml_new_graph_custom(gctx.get(), 8192, false);

    ggml_tensor * inp_images = nullptr;
    ggml_tensor * inp_pool_idx = nullptr;
    ggml_tensor * inp_pool_mask = nullptr;
    ggml_tensor * inp_inv_denom = nullptr;
    ggml_tensor * inp_attn_mask = nullptr;
    ggml_tensor * out = build_graph(
            gctx.get(), weights.get(), n_crops, n_tokens,
            &inp_images, &inp_pool_idx, &inp_pool_mask, &inp_inv_denom, &inp_attn_mask);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()));
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        std::fprintf(stderr, "error: failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        return 1;
    }
    std::fprintf(stderr, "vision_graph: crops=%d tokens=%d compute_buffer=%.2f MiB\n",
            n_crops, n_tokens, (double) ggml_gallocr_get_buffer_size(allocr, 0) / (1024.0 * 1024.0));

    ggml_backend_tensor_set(inp_images, images_f32.data(), 0, images_f32.size() * sizeof(float));
    ggml_backend_tensor_set(inp_pool_idx, pool_idx.data(), 0, pool_idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pool_mask, pool_mask.data(), 0, pool_mask.size() * sizeof(float));
    ggml_backend_tensor_set(inp_inv_denom, inv_denom.data(), 0, inv_denom.size() * sizeof(float));
    ggml_backend_tensor_set(inp_attn_mask, attn_mask.data(), 0, attn_mask.size() * sizeof(float));

    const enum ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "error: graph compute failed with status %d\n", (int) status);
        ggml_gallocr_free(allocr);
        return 1;
    }

    std::vector<float> result((size_t) OUT_DIM * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    std::printf("vision_features=[%d,%d]\n", n_tokens, OUT_DIM);

    if (!out_path.empty() && !write_file(out_path, result)) {
        ggml_gallocr_free(allocr);
        return 1;
    }

    if (!ref_path.empty()) {
        std::vector<float> ref;
        if (!read_file(ref_path, ref)) {
            ggml_gallocr_free(allocr);
            return 1;
        }
        compare_features(result, ref);
    }

    ggml_gallocr_free(allocr);
    return 0;
}
#endif
