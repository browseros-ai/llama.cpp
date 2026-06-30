#include "molmo2-rrt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace {

constexpr int IMAGE_SIZE = 378;
constexpr int PATCH_SIZE = 14;
constexpr int CROP_PATCHES = IMAGE_SIZE / PATCH_SIZE;
constexpr int PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * 3;
constexpr int POOL_H = 2;
constexpr int POOL_W = 2;
constexpr int LEFT_MARGIN = 4;
constexpr int RIGHT_MARGIN = 4;
constexpr int MAX_CROPS = 48;
constexpr int CROP_WINDOW_PATCHES = CROP_PATCHES - LEFT_MARGIN - RIGHT_MARGIN;
constexpr int CROP_WINDOW_SIZE = CROP_WINDOW_PATCHES * PATCH_SIZE;
constexpr int TOTAL_MARGIN_PIXELS = (LEFT_MARGIN + RIGHT_MARGIN) * PATCH_SIZE;

constexpr int64_t TOK_IM_START = 151936;
constexpr int64_t TOK_IM_END = 151937;
constexpr int64_t TOK_IM_PATCH = 151938;
constexpr int64_t TOK_IM_COL = 151939;
constexpr int64_t TOK_LOW_RES_IM_START = 151940;

struct image_u8 {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> data;

    const uint8_t * pixel(int x, int y) const {
        return &data[((size_t) y * w + x) * 3];
    }

    uint8_t * pixel(int x, int y) {
        return &data[((size_t) y * w + x) * 3];
    }
};

struct native_preprocess {
    int tile_h = 0;
    int tile_w = 0;
    int high_grid_h = 0;
    int high_grid_w = 0;
    std::vector<uint8_t> images;
    std::vector<int64_t> token_pooling;
    std::vector<int64_t> image_tokens;
};

static void print_usage(char ** argv) {
    std::fprintf(stderr,
            "usage: %s --image image.png [--ref-dir artifacts] [--dump-native out_dir]\n",
            argv[0]);
}

static bool read_file_u8(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "error: failed to open '%s'\n", path.c_str());
        return false;
    }
    const std::streamsize size = in.tellg();
    if (size < 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize((size_t) size);
    if (size > 0 && !in.read(reinterpret_cast<char *>(out.data()), size)) {
        std::fprintf(stderr, "error: failed to read '%s'\n", path.c_str());
        return false;
    }
    return true;
}

template<typename T>
static bool read_file_t(const std::string & path, std::vector<T> & out) {
    std::vector<uint8_t> bytes;
    if (!read_file_u8(path, bytes)) {
        return false;
    }
    if (bytes.size() % sizeof(T) != 0) {
        std::fprintf(stderr, "error: file '%s' size is not an element multiple\n", path.c_str());
        return false;
    }
    out.resize(bytes.size() / sizeof(T));
    if (!bytes.empty()) {
        std::memcpy(out.data(), bytes.data(), bytes.size());
    }
    return true;
}

template<typename T>
static bool write_file_t(const std::string & path, const std::vector<T> & values) {
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

static image_u8 load_image(const std::string & path) {
    int w = 0;
    int h = 0;
    int c = 0;
    stbi_uc * pixels = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!pixels) {
        std::fprintf(stderr, "error: stbi_load failed for '%s': %s\n", path.c_str(), stbi_failure_reason());
        std::exit(1);
    }
    image_u8 out;
    out.w = w;
    out.h = h;
    out.data.assign(pixels, pixels + (size_t) w * h * 3);
    stbi_image_free(pixels);
    return out;
}

static uint8_t round_to_u8(float x) {
    x = std::min(255.0f, std::max(0.0f, x));
    return (uint8_t) std::nearbyint(x);
}

static image_u8 resize_bilinear_torch_u8(const image_u8 & src, int dst_h, int dst_w) {
    image_u8 dst;
    dst.w = dst_w;
    dst.h = dst_h;
    dst.data.resize((size_t) dst_w * dst_h * 3);

    const float scale_x = (float) src.w / (float) dst_w;
    const float scale_y = (float) src.h / (float) dst_h;

    for (int y = 0; y < dst_h; ++y) {
        float src_y = ((float) y + 0.5f) * scale_y - 0.5f;
        src_y = std::max(0.0f, src_y);
        const int y0 = std::min((int) std::floor(src_y), src.h - 1);
        const int y1 = std::min(y0 + 1, src.h - 1);
        const float wy = src_y - (float) y0;
        for (int x = 0; x < dst_w; ++x) {
            float src_x = ((float) x + 0.5f) * scale_x - 0.5f;
            src_x = std::max(0.0f, src_x);
            const int x0 = std::min((int) std::floor(src_x), src.w - 1);
            const int x1 = std::min(x0 + 1, src.w - 1);
            const float wx = src_x - (float) x0;

            const uint8_t * p00 = src.pixel(x0, y0);
            const uint8_t * p10 = src.pixel(x1, y0);
            const uint8_t * p01 = src.pixel(x0, y1);
            const uint8_t * p11 = src.pixel(x1, y1);
            uint8_t * out = dst.pixel(x, y);
            for (int ch = 0; ch < 3; ++ch) {
                const float top = (1.0f - wx) * (float) p00[ch] + wx * (float) p10[ch];
                const float bot = (1.0f - wx) * (float) p01[ch] + wx * (float) p11[ch];
                out[ch] = round_to_u8((1.0f - wy) * top + wy * bot);
            }
        }
    }

    return dst;
}

static std::pair<int, int> select_tiling(int h, int w, int patch_size, int max_num_crops) {
    std::vector<std::pair<int, int>> tilings;
    for (int i = 1; i <= max_num_crops; ++i) {
        for (int j = 1; j <= max_num_crops; ++j) {
            if (i * j <= max_num_crops) {
                tilings.push_back({i, j});
            }
        }
    }
    std::sort(tilings.begin(), tilings.end(), [](const auto & a, const auto & b) {
        if (a.first * a.second != b.first * b.second) {
            return a.first * a.second < b.first * b.second;
        }
        return a.first < b.first;
    });

    std::vector<float> scales;
    scales.reserve(tilings.size());
    bool all_downscale = true;
    for (const auto & t : tilings) {
        const float scale_h = (float) (t.first * patch_size) / (float) h;
        const float scale_w = (float) (t.second * patch_size) / (float) w;
        const float scale = std::min(scale_h, scale_w);
        scales.push_back(scale);
        if (scale >= 1.0f) {
            all_downscale = false;
        }
    }

    int best = 0;
    if (all_downscale) {
        float best_scale = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < (int) scales.size(); ++i) {
            if (scales[i] > best_scale) {
                best_scale = scales[i];
                best = i;
            }
        }
    } else {
        float best_scale = std::numeric_limits<float>::infinity();
        for (int i = 0; i < (int) scales.size(); ++i) {
            const float scale = scales[i] < 1.0f ? 1.0e10f : scales[i];
            if (scale < best_scale) {
                best_scale = scale;
                best = i;
            }
        }
    }
    return tilings[best];
}

static void append_patchified_crop(std::vector<uint8_t> & out, const image_u8 & src, int x0, int y0) {
    for (int py = 0; py < CROP_PATCHES; ++py) {
        for (int px = 0; px < CROP_PATCHES; ++px) {
            for (int iy = 0; iy < PATCH_SIZE; ++iy) {
                for (int ix = 0; ix < PATCH_SIZE; ++ix) {
                    const uint8_t * p = src.pixel(x0 + px * PATCH_SIZE + ix, y0 + py * PATCH_SIZE + iy);
                    out.push_back(p[0]);
                    out.push_back(p[1]);
                    out.push_back(p[2]);
                }
            }
        }
    }
}

static std::vector<int64_t> arange_for_pooling(const std::vector<int64_t> & idx, int h, int w, int pool_h, int pool_w) {
    const int h_pad = pool_h * ((h + pool_h - 1) / pool_h) - h;
    const int w_pad = pool_w * ((w + pool_w - 1) / pool_w) - w;
    const int pad_top = h_pad / 2;
    const int pad_left = w_pad / 2;
    const int h2 = h + h_pad;
    const int w2 = w + w_pad;
    std::vector<int64_t> padded((size_t) h2 * w2, -1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            padded[(size_t) (y + pad_top) * w2 + (x + pad_left)] = idx[(size_t) y * w + x];
        }
    }

    std::vector<int64_t> out;
    out.reserve((size_t) (h2 / pool_h) * (w2 / pool_w) * pool_h * pool_w);
    for (int y = 0; y < h2 / pool_h; ++y) {
        for (int x = 0; x < w2 / pool_w; ++x) {
            for (int dy = 0; dy < pool_h; ++dy) {
                for (int dx = 0; dx < pool_w; ++dx) {
                    out.push_back(padded[(size_t) (y * pool_h + dy) * w2 + (x * pool_w + dx)]);
                }
            }
        }
    }
    return out;
}

static std::vector<int64_t> build_high_res_patch_mapping(int tile_h, int tile_w) {
    std::vector<int64_t> crop_maps((size_t) tile_h * tile_w * CROP_PATCHES * CROP_PATCHES);
    int on_crop = 0;
    for (int i = 0; i < tile_h; ++i) {
        for (int j = 0; j < tile_w; ++j) {
            for (int py = 0; py < CROP_PATCHES; ++py) {
                for (int px = 0; px < CROP_PATCHES; ++px) {
                    int64_t val = (int64_t) on_crop * CROP_PATCHES * CROP_PATCHES + py * CROP_PATCHES + px;
                    if (i != 0 && py < LEFT_MARGIN) {
                        val = -1;
                    }
                    if (j != 0 && px < LEFT_MARGIN) {
                        val = -1;
                    }
                    if (i != tile_h - 1 && py >= CROP_PATCHES - RIGHT_MARGIN) {
                        val = -1;
                    }
                    if (j != tile_w - 1 && px >= CROP_PATCHES - RIGHT_MARGIN) {
                        val = -1;
                    }
                    crop_maps[((size_t) on_crop * CROP_PATCHES + py) * CROP_PATCHES + px] = val;
                }
            }
            ++on_crop;
        }
    }

    std::vector<int64_t> filtered;
    filtered.reserve((size_t) (tile_h * CROP_WINDOW_PATCHES + LEFT_MARGIN + RIGHT_MARGIN) *
                     (tile_w * CROP_WINDOW_PATCHES + LEFT_MARGIN + RIGHT_MARGIN));
    for (int i = 0; i < tile_h; ++i) {
        for (int py = 0; py < CROP_PATCHES; ++py) {
            for (int j = 0; j < tile_w; ++j) {
                const int crop = i * tile_w + j;
                for (int px = 0; px < CROP_PATCHES; ++px) {
                    const int64_t val = crop_maps[((size_t) crop * CROP_PATCHES + py) * CROP_PATCHES + px];
                    if (val >= 0) {
                        filtered.push_back(val);
                    }
                }
            }
        }
    }
    return filtered;
}

static native_preprocess preprocess_molmo2(const image_u8 & original) {
    native_preprocess out;
    const auto tiling = select_tiling(
            std::max(original.h - TOTAL_MARGIN_PIXELS, 1),
            std::max(original.w - TOTAL_MARGIN_PIXELS, 1),
            CROP_WINDOW_SIZE,
            MAX_CROPS);
    out.tile_h = tiling.first;
    out.tile_w = tiling.second;

    const int high_h = out.tile_h * CROP_WINDOW_SIZE + TOTAL_MARGIN_PIXELS;
    const int high_w = out.tile_w * CROP_WINDOW_SIZE + TOTAL_MARGIN_PIXELS;
    out.high_grid_h = high_h / PATCH_SIZE;
    out.high_grid_w = high_w / PATCH_SIZE;

    image_u8 global = resize_bilinear_torch_u8(original, IMAGE_SIZE, IMAGE_SIZE);
    image_u8 high = resize_bilinear_torch_u8(original, high_h, high_w);

    const int n_crops = 1 + out.tile_h * out.tile_w;
    out.images.reserve((size_t) n_crops * CROP_PATCHES * CROP_PATCHES * PATCH_PIXELS);
    append_patchified_crop(out.images, global, 0, 0);
    for (int i = 0; i < out.tile_h; ++i) {
        const int y0 = i * CROP_WINDOW_SIZE;
        for (int j = 0; j < out.tile_w; ++j) {
            const int x0 = j * CROP_WINDOW_SIZE;
            append_patchified_crop(out.images, high, x0, y0);
        }
    }

    std::vector<int64_t> global_idx(CROP_PATCHES * CROP_PATCHES);
    for (int i = 0; i < (int) global_idx.size(); ++i) {
        global_idx[i] = i;
    }
    std::vector<int64_t> global_pool = arange_for_pooling(global_idx, CROP_PATCHES, CROP_PATCHES, POOL_H, POOL_W);

    std::vector<int64_t> high_idx = build_high_res_patch_mapping(out.tile_h, out.tile_w);
    if ((int) high_idx.size() != out.high_grid_h * out.high_grid_w) {
        std::fprintf(stderr, "error: high-res mapping has %zu entries, expected %d\n",
                high_idx.size(), out.high_grid_h * out.high_grid_w);
        std::exit(1);
    }
    std::vector<int64_t> high_pool = arange_for_pooling(high_idx, out.high_grid_h, out.high_grid_w, POOL_H, POOL_W);
    for (int64_t & v : high_pool) {
        if (v >= 0) {
            v += CROP_PATCHES * CROP_PATCHES;
        }
    }

    out.token_pooling.reserve(global_pool.size() + high_pool.size());
    out.token_pooling.insert(out.token_pooling.end(), global_pool.begin(), global_pool.end());
    out.token_pooling.insert(out.token_pooling.end(), high_pool.begin(), high_pool.end());

    const int global_tokens_h = (CROP_PATCHES + 1) / 2;
    const int global_tokens_w = (CROP_PATCHES + 1) / 2;
    const int high_tokens_h = (out.high_grid_h + 1) / 2;
    const int high_tokens_w = (out.high_grid_w + 1) / 2;

    out.image_tokens.reserve((size_t) 2 + global_tokens_h * global_tokens_w +
                             2 + high_tokens_h * (high_tokens_w + 1));
    out.image_tokens.push_back(TOK_LOW_RES_IM_START);
    for (int i = 0; i < global_tokens_h * global_tokens_w; ++i) {
        out.image_tokens.push_back(TOK_IM_PATCH);
    }
    out.image_tokens.push_back(TOK_IM_END);
    out.image_tokens.push_back(TOK_IM_START);
    for (int y = 0; y < high_tokens_h; ++y) {
        for (int x = 0; x < high_tokens_w; ++x) {
            out.image_tokens.push_back(TOK_IM_PATCH);
        }
        out.image_tokens.push_back(TOK_IM_COL);
    }
    out.image_tokens.push_back(TOK_IM_END);

    return out;
}

template<typename T>
static bool compare_vector(const char * name, const std::vector<T> & actual, const std::vector<T> & expected, int max_print = 8) {
    bool ok = true;
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s size mismatch: actual=%zu expected=%zu\n", name, actual.size(), expected.size());
        ok = false;
    }
    const size_t n = std::min(actual.size(), expected.size());
    int printed = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (actual[i] != expected[i]) {
            ++mismatches;
            ok = false;
            if (printed < max_print) {
                std::fprintf(stderr, "%s mismatch[%zu]: actual=%lld expected=%lld\n",
                        name, i, (long long) actual[i], (long long) expected[i]);
                ++printed;
            }
        }
    }
    if (mismatches > 0) {
        std::fprintf(stderr, "%s mismatches: %zu / %zu\n", name, mismatches, n);
    } else if (ok) {
        std::fprintf(stderr, "%s: exact match (%zu elements)\n", name, actual.size());
    }
    return ok;
}

static std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

} // namespace

bool molmo2_rrt_preprocess_image(
        const std::string & image_path,
        molmo2_rrt_preprocess_result * result) {
    if (!result) {
        std::fprintf(stderr, "error: null preprocess result\n");
        return false;
    }

    image_u8 image = load_image(image_path);
    native_preprocess native = preprocess_molmo2(image);

    result->image_w = image.w;
    result->image_h = image.h;
    result->tile_h = native.tile_h;
    result->tile_w = native.tile_w;
    result->high_grid_h = native.high_grid_h;
    result->high_grid_w = native.high_grid_w;
    result->images = std::move(native.images);
    result->token_pooling = std::move(native.token_pooling);
    result->image_tokens = std::move(native.image_tokens);
    return true;
}

#ifndef MOLMO2_RRT_NO_MAIN
int main(int argc, char ** argv) {
    std::string image_path;
    std::string ref_dir;
    std::string dump_dir;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (std::strcmp(argv[i], "--ref-dir") == 0 && i + 1 < argc) {
            ref_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-native") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else {
            print_usage(argv);
            return 1;
        }
    }

    if (image_path.empty()) {
        print_usage(argv);
        return 1;
    }

    image_u8 image = load_image(image_path);
    native_preprocess native = preprocess_molmo2(image);

    const size_t n_crops = native.images.size() / (CROP_PATCHES * CROP_PATCHES * PATCH_PIXELS);
    std::printf("image=%dx%d tiling=%dx%d crops=%zu high_grid=%dx%d\n",
            image.w, image.h, native.tile_h, native.tile_w, n_crops, native.high_grid_h, native.high_grid_w);
    std::printf("images=[%zu,%d,%d] token_pooling=[%zu,4] image_tokens=%zu image_patch_tokens=%zu\n",
            n_crops, CROP_PATCHES * CROP_PATCHES, PATCH_PIXELS,
            native.token_pooling.size() / 4,
            native.image_tokens.size(),
            (size_t) std::count(native.image_tokens.begin(), native.image_tokens.end(), TOK_IM_PATCH));

    bool ok = true;
    if (!ref_dir.empty()) {
        std::vector<uint8_t> ref_images;
        std::vector<int64_t> ref_pooling;
        std::vector<int64_t> ref_input_ids;
        ok = read_file_u8(join_path(ref_dir, "images.bin"), ref_images) && ok;
        ok = read_file_t(join_path(ref_dir, "token_pooling.i64.bin"), ref_pooling) && ok;
        ok = read_file_t(join_path(ref_dir, "input_ids.i64.bin"), ref_input_ids) && ok;
        if (ok) {
            ok = compare_vector("images", native.images, ref_images) && ok;
            ok = compare_vector("token_pooling", native.token_pooling, ref_pooling) && ok;
            size_t image_token_offset = std::numeric_limits<size_t>::max();
            for (size_t off = 0; off < std::min<size_t>(8, ref_input_ids.size()); ++off) {
                if (off + native.image_tokens.size() > ref_input_ids.size()) {
                    break;
                }
                bool match = true;
                for (size_t j = 0; j < native.image_tokens.size(); ++j) {
                    if (ref_input_ids[off + j] != native.image_tokens[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    image_token_offset = off;
                    break;
                }
            }
            if (image_token_offset == std::numeric_limits<size_t>::max()) {
                std::fprintf(stderr, "could not find image-token span in first input_ids positions\n");
                std::vector<int64_t> ref_image_tokens;
                if (ref_input_ids.size() >= native.image_tokens.size()) {
                    ref_image_tokens.assign(
                            ref_input_ids.begin(),
                            ref_input_ids.begin() + (std::ptrdiff_t) native.image_tokens.size());
                }
                ok = compare_vector("image_token_prefix", native.image_tokens, ref_image_tokens) && ok;
                ok = false;
            } else {
                std::vector<int64_t> ref_image_tokens(
                        ref_input_ids.begin() + (std::ptrdiff_t) image_token_offset,
                        ref_input_ids.begin() + (std::ptrdiff_t) (image_token_offset + native.image_tokens.size()));
                ok = compare_vector("image_tokens", native.image_tokens, ref_image_tokens) && ok;
                std::fprintf(stderr, "image_tokens offset in input_ids: %zu\n", image_token_offset);
            }
        }
    }

    if (!dump_dir.empty()) {
        write_file_t(join_path(dump_dir, "native_images.bin"), native.images);
        write_file_t(join_path(dump_dir, "native_token_pooling.i64.bin"), native.token_pooling);
        write_file_t(join_path(dump_dir, "native_image_tokens.i64.bin"), native.image_tokens);
        write_file_t(join_path(dump_dir, "images.bin"), native.images);
        write_file_t(join_path(dump_dir, "token_pooling.i64.bin"), native.token_pooling);
        write_file_t(join_path(dump_dir, "image_tokens.i64.bin"), native.image_tokens);
    }

    return ok ? 0 : 2;
}
#endif
