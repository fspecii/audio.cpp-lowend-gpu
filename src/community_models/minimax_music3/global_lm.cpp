#include "engine/community_models/minimax_music3/global_lm.h"

#include "engine/framework/modules/weight_binding.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace engine::models::minimax_music3 {
namespace {

namespace binding = engine::modules::binding;

void validate_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
        case assets::TensorStorageType::Q4_0:
        case assets::TensorStorageType::Q4_K:
            return;
        default:
            throw std::runtime_error("MiniMax Music 3 weight_type supports native, bf16, f16, q8_0, q4_0, and q4_k");
    }
}

modules::QwenDecoderActivationCastPolicy activation_cast_policy(core::BackendType backend_type) {
    modules::QwenDecoderActivationCastPolicy policy;
    if (backend_type == core::BackendType::Cpu || backend_type == core::BackendType::Vulkan ||
        backend_type == core::BackendType::Metal) {
        return policy;
    }
    if (std::getenv("MM3_NO_ACT_CAST") != nullptr) {
        // Skip the bf16 round-trip casts (~214k kernel launches / ~2 ms per frame).
        // Costs exact bf16-parity with the reference; compute runs in F32 throughout.
        return policy;
    }
    policy.enabled = true;
    policy.type = GGML_TYPE_BF16;
    policy.after_input_norm = true;
    policy.after_qkv_projection = true;
    policy.after_qk_norm = true;
    policy.after_rope = true;
    policy.after_static_cache_update = true;
    policy.after_attention = true;
    policy.after_attention_output = true;
    policy.after_residual = true;
    policy.after_ffn_norm = true;
    policy.after_mlp_projection = true;
    policy.after_mlp_silu = true;
    policy.after_mlp_mul = true;
    policy.after_output = true;
    return policy;
}

modules::QwenDecoderLayerWeights load_qwen_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MiniMaxMusic3QwenConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    // Fused QKV (default; MM3_SPLIT_QKV reverts). GQA-safe: PackedQKV slices
    // q_out + kv_out*2 in exactly this concatenation order, and all three tensors are
    // row-major {rows, hidden} of the same dtype, so raw byte concatenation is the packed
    // weight. 36 layers x 2 fewer GEMM launches on every AR frame.
    if (std::getenv("MM3_FUSE_LM_QKV") != nullptr) {
        const auto q_raw = source.require_tensor_data(prefix + ".self_attn.q_proj.weight");
        const auto k_raw = source.require_tensor_data(prefix + ".self_attn.k_proj.weight");
        const auto v_raw = source.require_tensor_data(prefix + ".self_attn.v_proj.weight");
        if (q_raw.metadata.dtype != k_raw.metadata.dtype || q_raw.metadata.dtype != v_raw.metadata.dtype ||
            k_raw.bytes.size() != v_raw.bytes.size()) {
            throw std::runtime_error("MiniMax Music 3 global qkv fuse requires matching dtypes");
        }
        const ggml_type qkv_type = assets::ggml_type_for_tensor_dtype(q_raw.metadata.dtype);
        std::vector<std::byte> packed(q_raw.bytes.size() + k_raw.bytes.size() + v_raw.bytes.size());
        std::memcpy(packed.data(), q_raw.bytes.data(), q_raw.bytes.size());
        std::memcpy(packed.data() + q_raw.bytes.size(), k_raw.bytes.data(), k_raw.bytes.size());
        std::memcpy(packed.data() + q_raw.bytes.size() + k_raw.bytes.size(), v_raw.bytes.data(), v_raw.bytes.size());
        const int64_t rows = config.attention_heads * config.head_dim + 2 * config.kv_heads * config.head_dim;
        out.self_attention.qkv_weight = store.make_tensor(
            core::TensorShape::from_dims({rows, config.hidden_size}),
            qkv_type,
            packed.data(),
            packed.size());
    } else {
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    }
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".post_attention_layernorm",
        config.hidden_size);
    // Packed gate/up (default; MM3_SPLIT_MLP reverts): one GEMM instead of two, and it
    // unlocks ggml's fused swiglu kernel (silu+mul in one op) - eligible because the
    // bf16 activation casts are off. 36 layers x 3 fewer launches on every AR frame.
    if (std::getenv("MM3_PACK_MLP") != nullptr) {
        const auto g_raw = source.require_tensor_data(prefix + ".mlp.gate_proj.weight");
        const auto u_raw = source.require_tensor_data(prefix + ".mlp.up_proj.weight");
        if (g_raw.metadata.dtype != u_raw.metadata.dtype || g_raw.bytes.size() != u_raw.bytes.size()) {
            throw std::runtime_error("MiniMax Music 3 gate/up fuse requires matching tensors");
        }
        const ggml_type mlp_type = assets::ggml_type_for_tensor_dtype(g_raw.metadata.dtype);
        std::vector<std::byte> packed(g_raw.bytes.size() * 2);
        std::memcpy(packed.data(), g_raw.bytes.data(), g_raw.bytes.size());
        std::memcpy(packed.data() + g_raw.bytes.size(), u_raw.bytes.data(), u_raw.bytes.size());
        modules::LinearWeights gate_up;
        gate_up.weight = store.make_tensor(
            core::TensorShape::from_dims({2 * config.intermediate_size, config.hidden_size}),
            mlp_type,
            packed.data(),
            packed.size());
        out.mlp.gate_up_proj = gate_up;
    } else {
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    }
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

}  // namespace

modules::QwenCausalDecodeRuntimeConfig make_minimax_music3_global_lm_runtime_config(
    const MiniMaxMusic3Config & config,
    core::BackendType backend_type,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = "minimax_music3.ar";
    out.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
    out.decode_graph_arena_bytes = decode_graph_arena_bytes;
    out.decoder.stack.hidden_size = config.qwen.hidden_size;
    out.decoder.stack.num_attention_heads = config.qwen.attention_heads;
    out.decoder.stack.num_key_value_heads = config.qwen.kv_heads;
    out.decoder.stack.head_dim = config.qwen.head_dim;
    out.decoder.stack.intermediate_size = config.qwen.intermediate_size;
    out.decoder.stack.layers = config.qwen.layers;
    out.decoder.stack.rms_norm_eps = config.qwen.rms_norm_eps;
    out.decoder.stack.rope_theta = config.qwen.rope_theta;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.use_qk_norm = true;
    // Packed gate/up measured 45.4 -> 50.1 s (a 10% regression): the 24576x4096 packed GEMV
    // falls off ggml's MMVQ fast path that the two 12288x4096 halves each hit, and the slice
    // views add copies. Same lesson as the LM QKV fusion - opt-in only.
    out.decoder.stack.runtime.mlp.mode = std::getenv("MM3_PACK_MLP") != nullptr
        ? modules::QwenDecoderMLPMode::PackedGateUp
        : modules::QwenDecoderMLPMode::Exact;
    // Fused QKV measured neutral/slightly worse for the 8B LM (its GEMMs are already large
    // enough that packing adds slice overhead without helping occupancy) - opt-in only.
    out.decoder.stack.qkv_layout = std::getenv("MM3_FUSE_LM_QKV") != nullptr
        ? modules::QwenDecoderQKVLayout::PackedQKV
        : modules::QwenDecoderQKVLayout::Separate;
    out.decoder.stack.activation_cast = activation_cast_policy(backend_type);
    out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode =
        modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.decoder.logits_size = config.qwen.vocab_size;
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.decoder.lm_head_precision = GGML_PREC_DEFAULT;
    out.readback_round_type = GGML_TYPE_BF16;
    if (backend_type == core::BackendType::Vulkan || backend_type == core::BackendType::Metal) {
        out.decoder.lm_head_input_type = GGML_TYPE_F16;
    } else if (backend_type != core::BackendType::Cpu) {
        out.decoder.lm_head_input_type = GGML_TYPE_BF16;
    }
    return out;
}

MiniMaxMusic3GlobalLMWeights load_minimax_music3_global_lm_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    validate_storage_type(storage_type);
    const auto & config = assets.config.qwen;
    const auto & source = *assets.language_model_weights;
    MiniMaxMusic3GlobalLMWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.global_lm.weights",
        weight_context_bytes);
    out.token_embedding = out.store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    out.qwen.token_embedding = out.token_embedding;
    out.qwen.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        out.qwen.stack.layers.push_back(load_qwen_layer(*out.store, source, config, storage_type, layer));
    }
    out.qwen.final_norm = binding::norm_weight_from_source(*out.store, source, "model.norm", config.hidden_size);
    // Compact head (default on; MM3_FULL_HEAD reverts): the AR stage can only ever sample
    // the 16384 semantic-audio rows plus the audio-end row, but the stock graph pays for all
    // 200000 - measured 11.7 ms/frame (a full q4_0 dequantization + FP16 GEMM per frame,
    // because the BF16 lm_head input disqualifies the MMVQ path). Slice the head to the
    // reachable rows at load time; the row order matches the compact-logits convention the
    // sampler already auto-detects, so downstream code is unchanged.
    if (std::getenv("MM3_FULL_HEAD") == nullptr) {
        constexpr int64_t kAudioCodeOffset = 151675;   // checkpoint contract (see prompt.h)
        constexpr int64_t kSemanticVocab = 16384;
        constexpr int64_t kAudioEndTokenId = 151670;
        const auto raw = source.require_tensor_data("lm_head.weight");
        const ggml_type head_type = assets::ggml_type_for_tensor_dtype(raw.metadata.dtype);
        const size_t row_bytes = ggml_row_size(head_type, config.hidden_size);
        if (raw.bytes.size() != row_bytes * static_cast<size_t>(config.vocab_size)) {
            throw std::runtime_error("MiniMax Music 3 lm_head raw size mismatch for compact head");
        }
        const int64_t compact_rows = kSemanticVocab + 1;
        std::vector<std::byte> compact(static_cast<size_t>(compact_rows) * row_bytes);
        std::memcpy(
            compact.data(),
            raw.bytes.data() + static_cast<size_t>(kAudioCodeOffset) * row_bytes,
            static_cast<size_t>(kSemanticVocab) * row_bytes);
        std::memcpy(
            compact.data() + static_cast<size_t>(kSemanticVocab) * row_bytes,
            raw.bytes.data() + static_cast<size_t>(kAudioEndTokenId) * row_bytes,
            row_bytes);
        out.qwen.lm_head = modules::LinearWeights{};
        out.qwen.lm_head->weight = out.store->make_tensor(
            core::TensorShape::from_dims({compact_rows, config.hidden_size}),
            head_type,
            compact.data(),
            compact.size());
    } else {
        out.qwen.lm_head = binding::linear_from_source(
            *out.store,
            source,
            "lm_head",
            storage_type,
            config.vocab_size,
            config.hidden_size,
            false);
    }
    out.store->upload();
    assets.language_model_weights->release_storage();
    return out;
}

}  // namespace engine::models::minimax_music3
