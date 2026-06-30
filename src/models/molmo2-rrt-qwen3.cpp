#include "models.h"

#include <stdexcept>

static ggml_tensor * rrt_lora_mm(
        ggml_context * ctx0,
        const llama_model_molmo2_rrt_qwen3::lora_pair & lora,
        ggml_tensor * cur) {
    return ggml_mul_mat(ctx0, lora.b, ggml_mul_mat(ctx0, lora.a, cur));
}

static ggml_tensor * view_2d_prefix(
        ggml_context * ctx0,
        ggml_tensor * cur,
        int64_t n0,
        int64_t n1,
        size_t offset) {
    return ggml_view_2d(ctx0, cur, n0, n1, cur->nb[1], offset);
}

void llama_model_molmo2_rrt_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key("molmo2.rrt.unique_layer_count",    n_unique_layers);
    ml.get_key("molmo2.rrt.effective_layer_count", n_effective_layers);
    ml.get_key("molmo2.rrt.loop_count",            n_loops);
    ml.get_key("molmo2.rrt.prelude_layer_count",   n_prelude);
    ml.get_key("molmo2.rrt.core_layer_count",      n_core);
    ml.get_key("molmo2.rrt.coda_layer_count",      n_coda);
    ml.get_key("molmo2.rrt.adapter_rank",          n_adapter_rank);

    if (n_unique_layers != hparams.n_layer_all) {
        throw std::runtime_error("Molmo2 RRT: GGUF block_count must equal unique layer count");
    }
    if (n_unique_layers != n_prelude + n_core + n_coda) {
        throw std::runtime_error("Molmo2 RRT: inconsistent prelude/core/coda counts");
    }
    if (n_effective_layers != n_prelude + n_core * n_loops + n_coda) {
        throw std::runtime_error("Molmo2 RRT: inconsistent effective layer count");
    }
    if (n_effective_layers > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("Molmo2 RRT: effective layer count exceeds LLAMA_MAX_LAYERS");
    }

    for (uint32_t il = n_unique_layers; il < n_effective_layers; ++il) {
        hparams.n_head_arr[il]    = hparams.n_head_arr[0];
        hparams.n_head_kv_arr[il] = hparams.n_head_kv_arr[0];
        hparams.n_ff_arr[il]      = hparams.n_ff_arr[0];
        hparams.is_swa_impl[il]   = false;
        hparams.is_recr_impl[il]  = false;
    }
    hparams.n_layer_all = n_effective_layers;

    type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_UNKNOWN;
}

void llama_model_molmo2_rrt_qwen3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   { n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < (int) n_unique_layers; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
    }

    rrt_core.resize(n_loops);
    for (uint32_t loop = 0; loop < n_loops; ++loop) {
        rrt_core[loop].resize(n_core);
        for (uint32_t core = 0; core < n_core; ++core) {
            auto & rrt = rrt_core[loop][core];

            rrt.attn_qkv.a = create_tensor(tn(LLM_TENSOR_RRT_ATTN_QKV_LORA_A, "weight", loop, core), { n_embd, n_adapter_rank }, 0);
            rrt.attn_qkv.b = create_tensor(tn(LLM_TENSOR_RRT_ATTN_QKV_LORA_B, "weight", loop, core), { n_adapter_rank, n_embd_head_k * n_head + 2 * n_embd_gqa }, 0);
            rrt.attn_out.a = create_tensor(tn(LLM_TENSOR_RRT_ATTN_OUT_LORA_A, "weight", loop, core), { n_embd_head_k * n_head, n_adapter_rank }, 0);
            rrt.attn_out.b = create_tensor(tn(LLM_TENSOR_RRT_ATTN_OUT_LORA_B, "weight", loop, core), { n_adapter_rank, n_embd }, 0);

            rrt.ffn_gate_up.a = create_tensor(tn(LLM_TENSOR_RRT_FFN_GATE_UP_LORA_A, "weight", loop, core), { n_embd, n_adapter_rank }, 0);
            rrt.ffn_gate_up.b = create_tensor(tn(LLM_TENSOR_RRT_FFN_GATE_UP_LORA_B, "weight", loop, core), { n_adapter_rank, 2 * n_ff }, 0);
            rrt.ffn_down.a    = create_tensor(tn(LLM_TENSOR_RRT_FFN_DOWN_LORA_A, "weight", loop, core), { n_ff, n_adapter_rank }, 0);
            rrt.ffn_down.b    = create_tensor(tn(LLM_TENSOR_RRT_FFN_DOWN_LORA_B, "weight", loop, core), { n_adapter_rank, n_embd }, 0);

            rrt.attn_norm = create_tensor(tn(LLM_TENSOR_RRT_ATTN_NORM, "weight", loop, core), { n_embd }, 0);
            rrt.ffn_norm  = create_tensor(tn(LLM_TENSOR_RRT_FFN_NORM,  "weight", loop, core), { n_embd }, 0);
        }
    }

    index_embd = create_tensor(tn(LLM_TENSOR_RRT_INDEX_EMBD, "weight"), { n_embd, (int64_t) n_loops * n_core }, 0);

    deep_residuals.resize(n_loops > 0 ? n_loops - 1 : 0);
    for (uint32_t loop = 0; loop + 1 < n_loops; ++loop) {
        deep_residuals[loop] = create_tensor(tn(LLM_TENSOR_RRT_DEEP_RESIDUAL, "weight", loop), { 2 * n_embd, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_molmo2_rrt_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_molmo2_rrt_qwen3::graph::graph(const llama_model & base_model, const llm_graph_params & params)
        : llm_graph_context(params) {
    const auto & model = static_cast<const llama_model_molmo2_rrt_qwen3 &>(base_model);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_ff        = hparams.n_ff();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * x0   = inpL;

    ggml_tensor * inp_pos     = build_inp_pos();
    auto *        inp_attn    = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    int cache_il = 0;

    auto add_index_embedding = [&](ggml_tensor * x, uint32_t loop, uint32_t core) {
        const int64_t row = (int64_t) loop * model.n_core + core;
        ggml_tensor * idx = ggml_view_1d(ctx0, model.index_embd, n_embd, row * ggml_row_size(model.index_embd->type, n_embd));
        return ggml_add(ctx0, x, ggml_repeat(ctx0, idx, x));
    };

    auto split_qkv = [&](ggml_tensor * qkv) {
        ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_head * n_head));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_head * (n_head + n_head_kv)));
        return llm_graph_qkv { Qcur, Kcur, Vcur };
    };

    auto run_block = [&](int block_idx, const core_adapters * rrt) {
        const int il = cache_il++;
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        ggml_tensor * attn_norm_w = rrt ? rrt->attn_norm : model.layers[block_idx].attn_norm;
        cur = build_norm(inpL, attn_norm_w, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * qkv = build_lora_mm(model.layers[block_idx].wqkv, cur, model.layers[block_idx].wqkv_s);
            if (rrt) {
                qkv = ggml_add(ctx0, qkv, rrt_lora_mm(ctx0, rrt->attn_qkv, cur));
            }
            cb(qkv, "wqkv", il);

            auto [Qcur, Kcur, Vcur] = split_qkv(qkv);

            Qcur = build_norm(Qcur, model.layers[block_idx].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[block_idx].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            ggml_tensor * attn_out = build_lora_mm(model.layers[block_idx].wo, cur, model.layers[block_idx].wo_s);
            if (rrt) {
                attn_out = ggml_add(ctx0, attn_out, rrt_lora_mm(ctx0, rrt->attn_out, cur));
            }
            cur = attn_out;
        }

        if (il == (int) model.n_effective_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        ggml_tensor * ffn_norm_w = rrt ? rrt->ffn_norm : model.layers[block_idx].ffn_norm;
        cur = build_norm(ffn_inp, ffn_norm_w, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * up   = build_lora_mm(model.layers[block_idx].ffn_up,   cur, model.layers[block_idx].ffn_up_s);
        ggml_tensor * gate = build_lora_mm(model.layers[block_idx].ffn_gate, cur, model.layers[block_idx].ffn_gate_s);
        if (rrt) {
            ggml_tensor * gate_up_delta = rrt_lora_mm(ctx0, rrt->ffn_gate_up, cur);
            ggml_tensor * up_delta      = view_2d_prefix(ctx0, gate_up_delta, n_ff, n_tokens, 0);
            ggml_tensor * gate_delta    = view_2d_prefix(ctx0, gate_up_delta, n_ff, n_tokens, ggml_row_size(gate_up_delta->type, n_ff));
            up   = ggml_add(ctx0, up,   up_delta);
            gate = ggml_add(ctx0, gate, gate_delta);
        }

        cb(up, "ffn_up", il);
        cb(gate, "ffn_gate", il);
        cur = ggml_swiglu_split(ctx0, gate, up);
        cb(cur, "ffn_swiglu", il);

        ggml_tensor * ffn_out = build_lora_mm(model.layers[block_idx].ffn_down, cur, model.layers[block_idx].ffn_down_s);
        if (rrt) {
            ffn_out = ggml_add(ctx0, ffn_out, rrt_lora_mm(ctx0, rrt->ffn_down, cur));
        }
        cb(ffn_out, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_out, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    };

    for (uint32_t block = 0; block < model.n_prelude; ++block) {
        run_block((int) block, nullptr);
    }

    for (uint32_t loop = 0; loop < model.n_loops; ++loop) {
        for (uint32_t core = 0; core < model.n_core; ++core) {
            inpL = add_index_embedding(inpL, loop, core);
            run_block((int) (model.n_prelude + core), &model.rrt_core[loop][core]);
        }

        if (loop + 1 < model.n_loops && loop < model.deep_residuals.size()) {
            ggml_tensor * joined = ggml_concat(ctx0, x0, inpL, 0);
            ggml_tensor * inj = ggml_mul_mat(ctx0, model.deep_residuals[loop], joined);
            inj = ggml_scale(ctx0, inj, 0.5f);
            inpL = ggml_add(ctx0, inpL, inj);
        }
    }

    for (uint32_t i = 0; i < model.n_coda; ++i) {
        run_block((int) (model.n_prelude + model.n_core + i), nullptr);
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
