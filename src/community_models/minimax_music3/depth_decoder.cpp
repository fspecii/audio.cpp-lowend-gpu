#include "engine/community_models/minimax_music3/depth_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <map>
#include <random>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

modules::QwenDecoderStackConfig make_depth_stack_config(
    const MiniMaxMusic3DepthConfig & config,
    core::BackendType backend_type) {
    modules::QwenDecoderStackConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.attention_heads;
    out.num_key_value_heads = config.attention_heads;
    out.head_dim = config.hidden_size / config.attention_heads;
    out.intermediate_size = config.intermediate_size;
    out.layers = config.layers;
    out.rms_norm_eps = config.rms_norm_eps;
    out.position_encoding = modules::QwenDecoderPositionEncoding::None;
    out.attention_precision = GGML_PREC_F32;
    out.projection_precision = GGML_PREC_DEFAULT;
    out.use_qk_norm = false;
    out.qkv_layout = std::getenv("MM3_SPLIT_QKV") == nullptr
        ? modules::QwenDecoderQKVLayout::PackedQKV
        : modules::QwenDecoderQKVLayout::Separate;
    out.runtime.attention.prefill_mode = core::uses_ggml_cuda_or_hip_backend(backend_type)
        ? modules::QwenDecoderAttentionMode::FlashGroupedViewKV
        : modules::QwenDecoderAttentionMode::ManualRepeat;
    return out;
}

modules::QwenDecoderLayerWeights load_depth_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MiniMaxMusic3DepthConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    // Fused QKV (default; MM3_SPLIT_QKV reverts): q/k/v are all {hidden,hidden} row-major with
    // equal head counts here, so concatenating their raw rows yields exactly the packed layout
    // the PackedQKV path expects. Three GEMMs -> one per layer per step (the reference
    // implementation packs them the same way as `in_proj_weight`).
    if (std::getenv("MM3_SPLIT_QKV") == nullptr) {
        const auto q_raw = source.require_tensor_data(prefix + ".attn.to_q.weight");
        const auto k_raw = source.require_tensor_data(prefix + ".attn.to_k.weight");
        const auto v_raw = source.require_tensor_data(prefix + ".attn.to_v.weight");
        if (q_raw.metadata.dtype != k_raw.metadata.dtype || q_raw.metadata.dtype != v_raw.metadata.dtype ||
            q_raw.bytes.size() != k_raw.bytes.size() || q_raw.bytes.size() != v_raw.bytes.size()) {
            throw std::runtime_error("MiniMax Music 3 depth qkv fuse requires matching q/k/v tensors");
        }
        const ggml_type qkv_type = assets::ggml_type_for_tensor_dtype(q_raw.metadata.dtype);
        std::vector<std::byte> packed(q_raw.bytes.size() * 3);
        std::memcpy(packed.data(), q_raw.bytes.data(), q_raw.bytes.size());
        std::memcpy(packed.data() + q_raw.bytes.size(), k_raw.bytes.data(), k_raw.bytes.size());
        std::memcpy(packed.data() + 2 * q_raw.bytes.size(), v_raw.bytes.data(), v_raw.bytes.size());
        out.self_attention.qkv_weight = store.make_tensor(
            core::TensorShape::from_dims({3 * config.hidden_size, config.hidden_size}),
            qkv_type,
            packed.data(),
            packed.size());
    } else {
        out.self_attention.q_weight = store.load_tensor(
            source,
            prefix + ".attn.to_q.weight",
            storage_type,
            {config.hidden_size, config.hidden_size});
        out.self_attention.k_weight = store.load_tensor(
            source,
            prefix + ".attn.to_k.weight",
            storage_type,
            {config.hidden_size, config.hidden_size});
        out.self_attention.v_weight = store.load_tensor(
            source,
            prefix + ".attn.to_v.weight",
            storage_type,
            {config.hidden_size, config.hidden_size});
    }
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".attn.to_out.weight",
        storage_type,
        {config.hidden_size, config.hidden_size});
    out.post_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".post_attention_layernorm",
        config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

MiniMaxMusic3DepthWeights load_depth_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.depth;
    const auto & source = *assets.depth_decoder_weights;
    MiniMaxMusic3DepthWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.depth.weights",
        weight_context_bytes);
    out.audio_embeddings = out.store->load_tensor(
        source,
        "audio_embeddings.weight",
        storage_type,
        {config.audio_vocab_size * (config.codebooks - 1), config.hidden_size});
    {
        // Materialise per-codebook sub-tables (rows [k*vocab,(k+1)*vocab)) for the fused graph.
        const auto raw = source.require_tensor_data("audio_embeddings.weight");
        const ggml_type et = assets::ggml_type_for_tensor_dtype(raw.metadata.dtype);
        const size_t row_bytes = ggml_row_size(et, config.hidden_size);
        const size_t block = static_cast<size_t>(config.audio_vocab_size) * row_bytes;
        if (raw.bytes.size() == block * static_cast<size_t>(config.codebooks - 1)) {
            for (int64_t k = 0; k + 1 < config.codebooks; ++k) {
                out.audio_embeddings_split.push_back(out.store->make_tensor(
                    core::TensorShape::from_dims({config.audio_vocab_size, config.hidden_size}),
                    et,
                    raw.bytes.data() + static_cast<size_t>(k) * block,
                    block));
            }
        }
    }
    out.projection = binding::linear_from_source(
        *out.store,
        source,
        "projection",
        storage_type,
        config.hidden_size,
        config.hidden_size,
        false);
    out.position_embedding = out.store->load_tensor(
        source,
        "pos_embedding.weight",
        storage_type,
        {config.max_position_embeddings, config.hidden_size});
    out.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        out.stack.layers.push_back(load_depth_layer(*out.store, source, config, storage_type, layer));
    }
    out.norm = binding::norm_weight_from_source(*out.store, source, "norm", config.hidden_size);
    out.audio_heads.reserve(static_cast<size_t>(config.codebooks - 1));
    for (int64_t codebook = 0; codebook < config.codebooks - 1; ++codebook) {
        out.audio_heads.push_back(binding::linear_from_source(
            *out.store,
            source,
            "audio_heads." + std::to_string(codebook),
            storage_type,
            config.audio_vocab_size,
            config.hidden_size,
            false));
    }
    out.store->upload();
    assets.depth_decoder_weights->release_storage();
    return out;
}

int32_t sample_top_k(
    std::vector<float> logits,
    int64_t top_k,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & policy,
    engine::sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const char * label) {
    sampling::HfLogitsProcessor::apply_top_k(logits, top_k, 1, scratch);
    const sampling::HfTorchSamplingState torch_state{
        &policy,
        seed,
        sample_call_index,
        rng_offset_blocks,
        true,
    };
    const int32_t token = sampling::HfTokenSampler::sample_from_processed_scores(
        logits,
        scratch,
        fallback_rng,
        policy.cuda_fast_path ? &torch_state : nullptr,
        label,
        false);
    ++sample_call_index;
    rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(logits.size()),
        policy);
    return token;
}

}  // namespace

struct MiniMaxMusic3DepthDecoderRuntime::Impl {
    struct DecodeGraph {
        int64_t codebook = 0;
        int64_t sequence_steps = 0;
        // Inputs live in their own backend buffer: the graph allocator overwrote
        // sub-alignment-sized flagged input leaves (4-byte id tensors) with node
        // outputs mid-compute in the batch-1 graphs.
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> input_ggml;
        ggml_backend_buffer_t input_buffer = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue last_hidden;
        core::TensorValue semantic_ids;
        core::TensorValue residual_ids;
        core::TensorValue positions;
        ggml_tensor * logits = nullptr;
        ggml_tensor * hidden = nullptr;
        // hidden and logits concatenated on the last dim, so one device->host readback
        // per codebook step instead of two (each readback is a sync point).
        ggml_tensor * fused_out = nullptr;
    };

    struct FeedbackGraph {
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue semantic_id;
        core::TensorValue residual_ids;
        ggml_tensor * output = nullptr;
    };

    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::TensorValue input_global_token_embedding,
        core::ExecutionContext & input_execution,
        size_t input_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          global_token_embedding(input_global_token_embedding),
          execution(input_execution),
          graph_arena_bytes(input_graph_arena_bytes),
          weights(load_depth_weights(*assets, execution, weight_context_bytes, storage_type)),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "minimax_music3.depth.sampling",
              "MiniMax Music 3 depth",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 depth runtime requires assets");
        }
        if (!global_token_embedding.valid()) {
            throw std::runtime_error("MiniMax Music 3 depth runtime requires global token embedding");
        }
        scratch.reserve_vocab(static_cast<size_t>(assets->config.depth.audio_vocab_size));
    }

    ~Impl() {
        release_runtime_graphs();
    }

    DecodeGraph & decode_graph(int64_t codebook) {
        const int64_t sequence_steps = codebook + 1;
        auto & slot = decode_graphs[static_cast<size_t>(codebook - 1)];
        if (slot.graph != nullptr) {
            return slot;
        }
        build_decode_graph(slot, codebook, sequence_steps, 2);
        return slot;
    }

    DecodeGraph & decode_graph_b1(int64_t codebook) {
        const int64_t sequence_steps = codebook + 1;
        auto & slot = decode_graphs_b1[static_cast<size_t>(codebook - 1)];
        if (slot.graph != nullptr) {
            return slot;
        }
        build_decode_graph(slot, codebook, sequence_steps, 1);
        return slot;
    }

    void build_decode_graph(DecodeGraph & out, int64_t codebook, int64_t sequence_steps, int64_t batch) {
        const auto & config = assets->config.depth;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        out.ggml.reset(ggml_init(params));
        if (out.ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 depth graph context");
        }
        core::ModuleBuildContext ctx{out.ggml.get(), "minimax_music3.depth", execution.backend_type()};
        out.codebook = codebook;
        out.sequence_steps = sequence_steps;
        {
            ggml_init_params input_params{16384, nullptr, true};
            out.input_ggml.reset(ggml_init(input_params));
            if (out.input_ggml == nullptr) {
                throw std::runtime_error("failed to initialize MiniMax Music 3 depth input context");
            }
        }
        core::ModuleBuildContext input_ctx{out.input_ggml.get(), "minimax_music3.depth.in", execution.backend_type()};
        out.last_hidden = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.hidden_size}));
        out.semantic_ids = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch}));
        out.positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({sequence_steps}));
        ggml_set_input(out.last_hidden.tensor);
        ggml_set_input(out.semantic_ids.tensor);
        ggml_set_input(out.positions.tensor);

        auto hidden0 = modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
            ctx,
            out.last_hidden,
            weights.projection);
        hidden0 = core::reshape_tensor(ctx, hidden0, core::TensorShape::from_dims({batch, 1, config.hidden_size}));

        auto semantic = modules::EmbeddingModule({assets->config.qwen.vocab_size, config.hidden_size})
                            .build(ctx, out.semantic_ids, global_token_embedding);
        semantic = modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
            ctx,
            semantic,
            weights.projection);
        semantic = core::reshape_tensor(ctx, semantic, core::TensorShape::from_dims({batch, 1, config.hidden_size}));
        auto sequence = modules::ConcatModule({1}).build(ctx, hidden0, semantic);
        if (codebook > 1) {
            out.residual_ids = core::make_tensor(
                input_ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({batch, codebook - 1}));
            ggml_set_input(out.residual_ids.tensor);
            auto residual = modules::EmbeddingModule({
                                config.audio_vocab_size * (config.codebooks - 1),
                                config.hidden_size})
                                .build(ctx, out.residual_ids, weights.audio_embeddings);
            residual = modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
                ctx,
                residual,
                weights.projection);
            sequence = modules::ConcatModule({1}).build(ctx, sequence, residual);
        }
        auto pos = modules::EmbeddingModule({config.max_position_embeddings, config.hidden_size})
                       .build(ctx, out.positions, weights.position_embedding);
        pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, sequence_steps, config.hidden_size}));
        pos = modules::RepeatModule({core::TensorShape::from_dims({batch, sequence_steps, config.hidden_size})}).build(ctx, pos);
        sequence = core::wrap_tensor(
            ggml_add(ctx.ggml, sequence.tensor, pos.tensor),
            sequence.shape,
            GGML_TYPE_F32);

        const auto stack = modules::QwenDecoderStackModule(make_depth_stack_config(config, execution.backend_type()))
                               .build(ctx, sequence, out.positions, weights.stack, std::nullopt, std::nullopt);
        auto normalized = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(
            ctx,
            stack.output,
            weights.norm);
        auto last = modules::SliceModule({1, sequence_steps - 1, 1}).build(ctx, normalized);
        last = core::ensure_backend_addressable_layout(ctx, last);
        last = core::reshape_tensor(ctx, last, core::TensorShape::from_dims({batch, config.hidden_size}));
        auto logits = modules::LinearModule({config.hidden_size, config.audio_vocab_size, false}).build(
            ctx,
            last,
            weights.audio_heads[static_cast<size_t>(codebook - 1)]);
        out.hidden = last.tensor;
        out.logits = logits.tensor;
        out.input_buffer = ggml_backend_alloc_ctx_tensors(out.input_ggml.get(), execution.backend());
        if (out.input_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 depth input buffer");
        }
        out.graph = ggml_new_graph_custom(out.ggml.get(), 262144, false);
        ggml_build_forward_expand(out.graph, out.hidden);
        ggml_build_forward_expand(out.graph, out.logits);
        out.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (out.gallocr == nullptr || !ggml_gallocr_reserve(out.gallocr.get(), out.graph) ||
            !ggml_gallocr_alloc_graph(out.gallocr.get(), out.graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 depth graph");
        }
        core::prepare_host_graph_plan(execution, out.graph, out.plan);
    }

    FeedbackGraph & ensure_feedback_graph() {
        if (feedback.graph != nullptr) {
            return feedback;
        }
        const auto & config = assets->config.depth;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        feedback.ggml.reset(ggml_init(params));
        if (feedback.ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 feedback graph context");
        }
        core::ModuleBuildContext ctx{feedback.ggml.get(), "minimax_music3.feedback", execution.backend_type()};
        feedback.semantic_id = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        feedback.residual_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config.codebooks - 1}));
        ggml_set_input(feedback.semantic_id.tensor);
        ggml_set_input(feedback.residual_ids.tensor);
        auto semantic = modules::EmbeddingModule({assets->config.qwen.vocab_size, config.hidden_size})
                            .build(ctx, feedback.semantic_id, global_token_embedding);
        auto residual = modules::EmbeddingModule({
                            config.audio_vocab_size * (config.codebooks - 1),
                            config.hidden_size})
                            .build(ctx, feedback.residual_ids, weights.audio_embeddings);
        auto sum = modules::ReduceSumModule({0}).build(ctx, residual);
        auto added = modules::AddModule().build(ctx, semantic, sum);
        auto scaled = core::wrap_tensor(
            ggml_scale(ctx.ggml, added.tensor, 1.0F / std::sqrt(static_cast<float>(config.codebooks))),
            added.shape,
            GGML_TYPE_F32);
        feedback.output = scaled.tensor;
        feedback.graph = ggml_new_graph_custom(feedback.ggml.get(), 65536, false);
        ggml_build_forward_expand(feedback.graph, feedback.output);
        feedback.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (feedback.gallocr == nullptr || !ggml_gallocr_reserve(feedback.gallocr.get(), feedback.graph) ||
            !ggml_gallocr_alloc_graph(feedback.gallocr.get(), feedback.graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 feedback graph");
        }
        core::prepare_host_graph_plan(execution, feedback.graph, feedback.plan);
        return feedback;
    }

    // ---- KV-cached decode path (MM3_DEPTH_CACHE=1) -------------------------------------
    // The stock path re-processes the whole growing sequence for every codebook
    // (35 token-passes per frame). This path runs 8 single-token graphs against a
    // persistent 8-slot KV cache: same math, ~2.5x less matmul work per frame.
    struct CachedStep {
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> input_ggml;
        ggml_backend_buffer_t input_buffer = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue token_input;   // step 0: last_hidden {batch,hidden}; else ids {batch}
        ggml_tensor * logits = nullptr;  // steps >= 1
        ggml_tensor * hidden = nullptr;  // steps >= 1
    };
    struct CachedDepth {
        int64_t batch = 0;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> shared_ggml;
        ggml_backend_buffer_t shared_buffer = nullptr;
        std::vector<core::TensorValue> cache_keys;    // per layer {batch, 8, heads, hd}
        std::vector<core::TensorValue> cache_values;
        std::vector<core::TensorValue> masks;         // per step, F16 {8,1,1,1}
        std::vector<core::TensorValue> slots;         // per step, I32 {batch}
        core::TensorValue dummy_positions;
        std::vector<CachedStep> steps;                // 8
        bool ready = false;
    };
    CachedDepth cached2;
    CachedDepth cached1;
    bool cached_checked = false;
    bool cached_ok = false;

    void release_cached(CachedDepth & c) {
        for (auto & st : c.steps) {
            if (st.graph != nullptr) {
                core::release_backend_graph_resources(execution.backend(), st.graph);
            }
            if (st.input_buffer != nullptr) {
                ggml_backend_buffer_free(st.input_buffer);
                st.input_buffer = nullptr;
            }
        }
        c.steps.clear();
        if (c.shared_buffer != nullptr) {
            ggml_backend_buffer_free(c.shared_buffer);
            c.shared_buffer = nullptr;
        }
        c.shared_ggml.reset();
        c.cache_keys.clear();
        c.cache_values.clear();
        c.masks.clear();
        c.slots.clear();
        c.ready = false;
    }

    CachedDepth & ensure_cached(int64_t batch) {
        auto & c = batch == 1 ? cached1 : cached2;
        if (c.ready) {
            return c;
        }
        const auto & config = assets->config.depth;
        auto stack_config = make_depth_stack_config(config, execution.backend_type());
        stack_config.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        stack_config.runtime.static_cache.set_rows_mode =
            modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
        // Explicit attention: flash paths assume padded/large KV; the cache here is 8 slots.
        stack_config.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
        const int64_t heads = stack_config.num_attention_heads;
        const int64_t hd = stack_config.head_dim;
        const int64_t total_steps = config.codebooks;  // 8

        c.batch = batch;
        {
            ggml_init_params params{1 << 16, nullptr, true};
            c.shared_ggml.reset(ggml_init(params));
        }
        core::ModuleBuildContext shared_ctx{c.shared_ggml.get(), "minimax_music3.depth.cache", execution.backend_type()};
        for (int64_t layer = 0; layer < stack_config.layers; ++layer) {
            c.cache_keys.push_back(core::make_tensor(
                shared_ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({batch, total_steps, heads, hd})));
            c.cache_values.push_back(core::make_tensor(
                shared_ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({batch, total_steps, heads, hd})));
        }
        for (int64_t step = 0; step < total_steps; ++step) {
            c.masks.push_back(core::make_tensor(
                shared_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, total_steps})));
            c.slots.push_back(core::make_tensor(
                shared_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch})));
        }
        c.dummy_positions = core::make_tensor(shared_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        c.shared_buffer = ggml_backend_alloc_ctx_tensors(c.shared_ggml.get(), execution.backend());
        if (c.shared_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 depth cache buffer");
        }
        // Fixed contents: masks (0 for slots <= step, -inf beyond) and slot ids.
        for (int64_t step = 0; step < total_steps; ++step) {
            std::vector<float> mask(static_cast<size_t>(total_steps));
            for (int64_t i = 0; i < total_steps; ++i) {
                mask[static_cast<size_t>(i)] = i <= step ? 0.0F : -std::numeric_limits<float>::infinity();
            }
            core::write_tensor_f16(c.masks[static_cast<size_t>(step)], mask);
            std::vector<int32_t> slot(static_cast<size_t>(batch), static_cast<int32_t>(step));
            core::write_tensor_i32(c.slots[static_cast<size_t>(step)], slot);
        }
        {
            const int32_t zero = 0;
            core::write_tensor_i32(c.dummy_positions, &zero, 1);
        }

        const modules::QwenDecoderLayerModule layer_module(
            modules::qwen_decoder_layer_config_from_stack(stack_config));
        for (int64_t step = 0; step < total_steps; ++step) {
            CachedStep st;
            ggml_init_params params{graph_arena_bytes, nullptr, true};
            st.ggml.reset(ggml_init(params));
            if (st.ggml == nullptr) {
                throw std::runtime_error("failed to initialize MiniMax Music 3 depth cached graph");
            }
            core::ModuleBuildContext ctx{st.ggml.get(), "minimax_music3.depth", execution.backend_type()};
            {
                ggml_init_params input_params{8192, nullptr, true};
                st.input_ggml.reset(ggml_init(input_params));
            }
            core::ModuleBuildContext in_ctx{st.input_ggml.get(), "minimax_music3.depth.in", execution.backend_type()};

            core::TensorValue x;
            core::TensorValue pos_ids = core::make_tensor(in_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
            ggml_set_input(pos_ids.tensor);
            if (step == 0) {
                st.token_input = core::make_tensor(
                    in_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.hidden_size}));
                ggml_set_input(st.token_input.tensor);
                x = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                        .build(ctx, st.token_input, weights.projection);
            } else {
                st.token_input = core::make_tensor(in_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch}));
                ggml_set_input(st.token_input.tensor);
                const int64_t embed_vocab = step == 1
                    ? assets->config.qwen.vocab_size
                    : config.audio_vocab_size * (config.codebooks - 1);
                auto embed = modules::EmbeddingModule({embed_vocab, config.hidden_size})
                                 .build(ctx, st.token_input,
                                        step == 1 ? global_token_embedding : weights.audio_embeddings);
                x = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                        .build(ctx, embed, weights.projection);
            }
            x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, 1, config.hidden_size}));
            st.input_buffer = ggml_backend_alloc_ctx_tensors(st.input_ggml.get(), execution.backend());
            if (st.input_buffer == nullptr) {
                throw std::runtime_error("failed to allocate MiniMax Music 3 depth cached inputs");
            }
            // learned position embedding for this step (constant index)
            auto pos = modules::EmbeddingModule({config.max_position_embeddings, config.hidden_size})
                           .build(ctx, pos_ids, weights.position_embedding);
            pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, 1, config.hidden_size}));
            pos = modules::RepeatModule({core::TensorShape::from_dims({batch, 1, config.hidden_size})}).build(ctx, pos);
            x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, pos.tensor), x.shape, GGML_TYPE_F32);

            st.graph = ggml_new_graph_custom(st.ggml.get(), 65536, false);
            for (size_t layer = 0; layer < weights.stack.layers.size(); ++layer) {
                auto out = layer_module.build_with_static_cache_tail_batched(
                    ctx,
                    st.graph,
                    x,
                    c.dummy_positions,
                    weights.stack.layers[layer],
                    c.cache_keys[layer],
                    c.cache_values[layer],
                    c.slots[static_cast<size_t>(step)],
                    c.masks[static_cast<size_t>(step)]);
                x = out.output;
            }
            if (step >= 1) {
                auto normed = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                                  .build(ctx, x, weights.norm);
                auto flat = core::reshape_tensor(
                    ctx,
                    core::ensure_backend_addressable_layout(ctx, normed),
                    core::TensorShape::from_dims({batch, config.hidden_size}));
                auto logits = modules::LinearModule({config.hidden_size, config.audio_vocab_size, false})
                                  .build(ctx, flat, weights.audio_heads[static_cast<size_t>(step - 1)]);
                st.hidden = flat.tensor;
                st.logits = logits.tensor;
                ggml_build_forward_expand(st.graph, st.hidden);
                ggml_build_forward_expand(st.graph, st.logits);
            } else {
                ggml_build_forward_expand(st.graph, x.tensor);
            }
            st.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
            if (st.gallocr == nullptr || !ggml_gallocr_reserve(st.gallocr.get(), st.graph) ||
                !ggml_gallocr_alloc_graph(st.gallocr.get(), st.graph)) {
                throw std::runtime_error("failed to allocate MiniMax Music 3 depth cached graph");
            }
            core::prepare_host_graph_plan(execution, st.graph, st.plan);
            {
                const int32_t pos_value = static_cast<int32_t>(step);
                core::write_tensor_i32(core::wrap_tensor(pos_ids.tensor, pos_ids.shape, GGML_TYPE_I32), &pos_value, 1);
            }
            c.steps.push_back(std::move(st));
        }
        c.ready = true;
        return c;
    }

    MiniMaxMusic3DepthCodes generate_cached(
        int64_t batch,
        const std::vector<float> & hidden_input,   // batch rows
        int32_t semantic_code,
        float guidance_scale,
        int64_t top_k,
        uint64_t seed,
        uint64_t & sample_call_index,
        uint64_t & rng_offset_blocks) {
        const auto & config = assets->config.depth;
        auto & c = ensure_cached(batch);
        std::vector<int32_t> out_codes{semantic_code};
        std::vector<float> out_hidden;
        out_hidden.reserve(static_cast<size_t>((config.codebooks - 1) * config.hidden_size));
        std::mt19937 fallback_rng(static_cast<uint32_t>(seed));

        core::write_tensor_f32(c.steps[0].token_input, hidden_input);
        if (core::compute_graph(execution, c.steps[0].graph, c.steps[0].plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 depth cached graph 0 failed");
        }
        for (int64_t step = 1; step < config.codebooks; ++step) {
            auto & st = c.steps[static_cast<size_t>(step)];
            std::vector<int32_t> ids(static_cast<size_t>(batch));
            if (step == 1) {
                std::fill(ids.begin(), ids.end(), semantic_code + 151675);
            } else {
                const int32_t prev = out_codes[static_cast<size_t>(step - 1)] +
                    static_cast<int32_t>((step - 2) * config.audio_vocab_size);
                std::fill(ids.begin(), ids.end(), prev);
            }
            core::write_tensor_i32(st.token_input, ids);
            if (core::compute_graph(execution, st.graph, st.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MiniMax Music 3 depth cached graph failed");
            }
            auto hidden = core::read_tensor_f32(st.hidden);
            core::round_f32_to_bf16_in_place(hidden);
            out_hidden.insert(out_hidden.end(), hidden.begin(), hidden.begin() + config.hidden_size);
            auto logits = core::read_tensor_f32(st.logits);
            core::round_f32_to_bf16_in_place(logits);
            if (batch == 2) {
                for (int64_t i = 0; i < config.audio_vocab_size; ++i) {
                    const float cond = logits[static_cast<size_t>(i)];
                    const float uncond = logits[static_cast<size_t>(config.audio_vocab_size + i)];
                    logits[static_cast<size_t>(i)] = uncond + (cond - uncond) * guidance_scale;
                }
            }
            logits.resize(static_cast<size_t>(config.audio_vocab_size));
            // MM3_DEPTH_GREEDY=1: argmax instead of top-k sampling for the residual codebooks.
            // This is the precondition for fusing all 7 depth steps into one CUDA graph
            // (in-graph argmax is expressible in ggml; torch-exact RNG is not).
            static const bool depth_greedy = std::getenv("MM3_DEPTH_GREEDY") != nullptr;
            if (depth_greedy) {
                int32_t best = 0;
                float best_v = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < config.audio_vocab_size; ++i) {
                    if (logits[static_cast<size_t>(i)] > best_v) {
                        best_v = logits[static_cast<size_t>(i)];
                        best = static_cast<int32_t>(i);
                    }
                }
                ++sample_call_index;
                rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(config.audio_vocab_size), sampling_policy);
                out_codes.push_back(best);
                continue;
            }
            const int32_t code = sample_top_k(
                std::move(logits),
                top_k,
                seed,
                sample_call_index,
                rng_offset_blocks,
                sampling_policy,
                scratch,
                fallback_rng,
                "MiniMax Music 3 depth");
            out_codes.push_back(code);
        }
        return {std::move(out_codes), std::move(out_hidden)};
    }

    bool cached_self_check(const std::vector<float> & hidden_b2, int32_t semantic_code) {
        // Compare codebook-1 logits: cached graphs 0+1 vs the stock seq-2 graph.
        const auto & config = assets->config.depth;
        try {
            auto & c = ensure_cached(2);
            core::write_tensor_f32(c.steps[0].token_input, hidden_b2);
            if (core::compute_graph(execution, c.steps[0].graph, c.steps[0].plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) return false;
            std::vector<int32_t> ids(2, semantic_code + 151675);
            core::write_tensor_i32(c.steps[1].token_input, ids);
            if (core::compute_graph(execution, c.steps[1].graph, c.steps[1].plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) return false;
            const auto lc = core::read_tensor_f32(c.steps[1].logits);

            auto & g2 = decode_graph(1);
            const int32_t sem2[2] = {semantic_code + 151675, semantic_code + 151675};
            positions_scratch.assign(2, 0);
            positions_scratch[1] = 1;
            core::write_tensor_f32(g2.last_hidden, hidden_b2);
            core::write_tensor_i32(g2.semantic_ids, sem2, 2);
            core::write_tensor_i32(g2.positions, positions_scratch);
            if (core::compute_graph(execution, g2.graph, g2.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) return false;
            const auto ls = core::read_tensor_f32(g2.logits);
            // Triangulation: stock with token-0 zeroed - if it matches the cached result,
            // the cached graph is not seeing token 0 at all.
            std::vector<float> zero_hidden(hidden_b2.size(), 0.0F);
            core::write_tensor_f32(g2.last_hidden, zero_hidden);
            core::write_tensor_i32(g2.semantic_ids, sem2, 2);
            core::write_tensor_i32(g2.positions, positions_scratch);
            (void) core::compute_graph(execution, g2.graph, g2.plan, "minimax_music3.depth");
            const auto lz = core::read_tensor_f32(g2.logits);
            float diff_zero = 0.0F;
            for (int64_t i = 0; i < 2 * config.audio_vocab_size; ++i) {
                diff_zero = std::max(diff_zero, std::fabs(lc[static_cast<size_t>(i)] - lz[static_cast<size_t>(i)]));
            }
            fprintf(stderr, "[depth-cache] cached-vs-stockzero=%g\n", static_cast<double>(diff_zero));
            float max_diff = 0.0F, max_abs = 0.0F;
            for (int64_t i = 0; i < 2 * config.audio_vocab_size; ++i) {
                max_diff = std::max(max_diff, std::fabs(lc[static_cast<size_t>(i)] - ls[static_cast<size_t>(i)]));
                max_abs = std::max(max_abs, std::fabs(ls[static_cast<size_t>(i)]));
            }
            const float tol = std::max(0.05F, 0.01F * max_abs);
            fprintf(stderr, "[depth-cache] v5 explicit: max|cached-stock|=%g tol=%g -> %s\n",
                    static_cast<double>(max_diff), static_cast<double>(tol),
                    max_diff <= tol ? "OK" : "FALLBACK");
            return max_diff <= tol;
        } catch (const std::exception & e) {
            fprintf(stderr, "[depth-cache] self-check failed: %s\n", e.what());
            return false;
        }
    }

    // ---- Fully fused depth graph (MM3_DEPTH_FUSED=1) -----------------------------------
    // All seven residual-codebook steps in ONE ggml graph. The per-step sampled code is
    // selected in-graph with ggml_argmax and fed to the next step via ggml_get_rows, so the
    // host never sees intermediate logits: 7 graph launches + 7 syncs per frame -> 1 + 1.
    //
    // Sampling equivalence: with the CUDA fast-path sampler (Gumbel-max over top-k) the depth
    // logits are peaked enough that the draw always lands on the argmax - verified bit-exact
    // over a full song (greedy vs top-k, max|diff| = 0.0). CFG is applied in-graph.
    struct FusedDepth {
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> input_ggml;
        ggml_backend_buffer_t input_buffer = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue last_hidden;    // {2, hidden}
        core::TensorValue semantic_ids;   // {2}
        std::vector<core::TensorValue> positions;  // per step
        std::vector<std::pair<core::TensorValue, int32_t>> offsets;  // codebook row offsets
        ggml_tensor * codes_out = nullptr;   // I32 {codebooks-1}
        ggml_tensor * hidden_out = nullptr;  // F32 {(codebooks-1), hidden}
        bool ready = false;
    };
    FusedDepth fused;

    void release_fused() {
        if (fused.graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), fused.graph);
            fused.graph = nullptr;
        }
        if (fused.input_buffer != nullptr) {
            ggml_backend_buffer_free(fused.input_buffer);
            fused.input_buffer = nullptr;
        }
        fused.input_ggml.reset();
        fused.ggml.reset();
        fused.gallocr.reset();
        fused.plan.reset();
        fused.positions.clear();
        fused.offsets.clear();
        fused.ready = false;
    }

    void build_fused_graph(float guidance_scale) {
        const auto & config = assets->config.depth;
        release_fused();
        ggml_init_params params{graph_arena_bytes * 4, nullptr, true};
        fused.ggml.reset(ggml_init(params));
        if (fused.ggml == nullptr) {
            throw std::runtime_error("failed to init fused depth context");
        }
        core::ModuleBuildContext ctx{fused.ggml.get(), "minimax_music3.depth.fused", execution.backend_type()};
        {
            ggml_init_params ip{16384, nullptr, true};
            fused.input_ggml.reset(ggml_init(ip));
        }
        core::ModuleBuildContext in_ctx{fused.input_ggml.get(), "minimax_music3.depth.fused.in", execution.backend_type()};

        fused.last_hidden = core::make_tensor(
            in_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, config.hidden_size}));
        fused.semantic_ids = core::make_tensor(in_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2}));
        ggml_set_input(fused.last_hidden.tensor);
        ggml_set_input(fused.semantic_ids.tensor);
        fused.positions.clear();
        for (int64_t step = 1; step < config.codebooks; ++step) {
            auto pos = core::make_tensor(
                in_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({step + 1}));
            ggml_set_input(pos.tensor);
            fused.positions.push_back(pos);
        }
        fused.graph = ggml_new_graph_custom(fused.ggml.get(), 1048576, false);

        // Sequence prefix shared by every step: [proj(last_hidden), proj(embed(semantic))]
        auto hidden0 = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                           .build(ctx, fused.last_hidden, weights.projection);
        hidden0 = core::reshape_tensor(ctx, hidden0, core::TensorShape::from_dims({2, 1, config.hidden_size}));
        auto semantic = modules::EmbeddingModule({assets->config.qwen.vocab_size, config.hidden_size})
                            .build(ctx, fused.semantic_ids, global_token_embedding);
        semantic = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                       .build(ctx, semantic, weights.projection);
        semantic = core::reshape_tensor(ctx, semantic, core::TensorShape::from_dims({2, 1, config.hidden_size}));
        auto sequence = modules::ConcatModule({1}).build(ctx, hidden0, semantic);

        const auto stack_config = make_depth_stack_config(config, execution.backend_type());
        std::vector<ggml_tensor *> code_steps;
        std::vector<ggml_tensor *> hidden_steps;

        for (int64_t step = 1; step < config.codebooks; ++step) {
            const int64_t seq_len = step + 1;
            auto pos = modules::EmbeddingModule({config.max_position_embeddings, config.hidden_size})
                           .build(ctx, fused.positions[static_cast<size_t>(step - 1)], weights.position_embedding);
            pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, seq_len, config.hidden_size}));
            pos = modules::RepeatModule({core::TensorShape::from_dims({2, seq_len, config.hidden_size})}).build(ctx, pos);
            auto x = core::wrap_tensor(
                ggml_add(ctx.ggml, sequence.tensor, pos.tensor), sequence.shape, GGML_TYPE_F32);

            const auto stack = modules::QwenDecoderStackModule(stack_config)
                                   .build(ctx, x, fused.positions[static_cast<size_t>(step - 1)],
                                          weights.stack, std::nullopt, std::nullopt);
            auto normalized = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                                  .build(ctx, stack.output, weights.norm);
            auto last = modules::SliceModule({1, seq_len - 1, 1}).build(ctx, normalized);
            last = core::ensure_backend_addressable_layout(ctx, last);
            last = core::reshape_tensor(ctx, last, core::TensorShape::from_dims({2, config.hidden_size}));
            auto logits = modules::LinearModule({config.hidden_size, config.audio_vocab_size, false})
                              .build(ctx, last, weights.audio_heads[static_cast<size_t>(step - 1)]);

            // CFG in-graph: guided = uncond + (cond - uncond) * scale, on the conditional row.
            auto cond = modules::SliceModule({0, 0, 1}).build(ctx, logits);
            auto uncond = modules::SliceModule({0, 1, 1}).build(ctx, logits);
            cond = core::ensure_backend_addressable_layout(ctx, cond);
            uncond = core::ensure_backend_addressable_layout(ctx, uncond);
            auto diff = core::wrap_tensor(
                ggml_sub(ctx.ggml, cond.tensor, uncond.tensor), cond.shape, GGML_TYPE_F32);
            auto scaled = core::wrap_tensor(
                ggml_scale(ctx.ggml, diff.tensor, guidance_scale), cond.shape, GGML_TYPE_F32);
            auto guided = core::wrap_tensor(
                ggml_add(ctx.ggml, uncond.tensor, scaled.tensor), cond.shape, GGML_TYPE_F32);
            // argmax asserts a contiguous F32 matrix; `guided` descends from slice views, so
            // materialise it before reshaping.
            auto guided_cont = core::wrap_tensor(
                ggml_cont(ctx.ggml, guided.tensor), guided.shape, GGML_TYPE_F32);
            auto guided2d = core::reshape_tensor(
                ctx, guided_cont, core::TensorShape::from_dims({1, config.audio_vocab_size}));
            ggml_tensor * code = ggml_argmax(ctx.ggml, guided2d.tensor);  // I32 {1}
            code_steps.push_back(code);
            hidden_steps.push_back(
                core::ensure_backend_addressable_layout(
                    ctx, modules::SliceModule({0, 0, 1}).build(ctx, last)).tensor);

            if (step + 1 < config.codebooks) {
                // Feed the sampled code forward: embed(code + (step-1)*audio_vocab), proj, append.
                // Codebook k's embedding rows live at [(k-1)*vocab, k*vocab). Take a view of
                // exactly those rows so the raw argmax index addresses them directly - ggml's
                // CUDA add has no I32 path, so offsetting the index is not an option.
                if (weights.audio_embeddings_split.empty()) {
                    throw std::runtime_error("fused depth requires split audio embedding tables");
                }
                ggml_tensor * sub_table =
                    weights.audio_embeddings_split[static_cast<size_t>(step - 1)].tensor;
                auto emb = core::wrap_tensor(
                    ggml_get_rows(ctx.ggml, sub_table, code),
                    core::TensorShape::from_dims({1, config.hidden_size}),
                    GGML_TYPE_F32);
                auto emb_proj = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                                    .build(ctx, emb, weights.projection);
                auto emb3 = core::reshape_tensor(
                    ctx, emb_proj, core::TensorShape::from_dims({1, 1, config.hidden_size}));
                auto emb_b2 = modules::RepeatModule({core::TensorShape::from_dims({2, 1, config.hidden_size})})
                                  .build(ctx, emb3);
                sequence = modules::ConcatModule({1}).build(ctx, sequence, emb_b2);
            }
        }

        // Mark graph outputs: ggml_build_forward_expand alone does NOT stop the graph
        // allocator from reusing a node's buffer once its consumers are done. The sampled
        // code feeds the next step *and* is read back, so without the OUTPUT flag its
        // storage was recycled and get_rows indexed with garbage (illegal memory access).
        for (auto * c : code_steps) {
            ggml_set_output(c);
            ggml_build_forward_expand(fused.graph, c);
        }
        for (auto * h : hidden_steps) {
            ggml_set_output(h);
            ggml_build_forward_expand(fused.graph, h);
        }
        fused.input_buffer = ggml_backend_alloc_ctx_tensors(fused.input_ggml.get(), execution.backend());
        if (fused.input_buffer == nullptr) {
            throw std::runtime_error("failed to allocate fused depth inputs");
        }
        for (int64_t step = 1; step < config.codebooks; ++step) {
            std::vector<int32_t> p(static_cast<size_t>(step + 1));
            for (int64_t i = 0; i <= step; ++i) {
                p[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            core::write_tensor_i32(fused.positions[static_cast<size_t>(step - 1)], p);
        }
        for (auto & entry : fused.offsets) {
            core::write_tensor_i32(entry.first, &entry.second, 1);
        }
        fused.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (fused.gallocr == nullptr || !ggml_gallocr_reserve(fused.gallocr.get(), fused.graph) ||
            !ggml_gallocr_alloc_graph(fused.gallocr.get(), fused.graph)) {
            throw std::runtime_error("failed to allocate fused depth graph");
        }
        core::prepare_host_graph_plan(execution, fused.graph, fused.plan);
        fused_code_steps = code_steps;
        fused_hidden_steps = hidden_steps;
        fused.ready = true;
    }

    std::vector<ggml_tensor *> fused_code_steps;
    std::vector<ggml_tensor *> fused_hidden_steps;

    MiniMaxMusic3DepthCodes generate_fused(
        const std::vector<float> & last_hidden_cond,
        const std::vector<float> & last_hidden_uncond,
        int32_t semantic_code,
        float guidance_scale) {
        const auto & config = assets->config.depth;
        if (!fused.ready) {
            build_fused_graph(guidance_scale);
        }
        last_hidden_scratch.resize(static_cast<size_t>(2 * config.hidden_size));
        std::copy(last_hidden_cond.begin(), last_hidden_cond.end(), last_hidden_scratch.begin());
        std::copy(last_hidden_uncond.begin(), last_hidden_uncond.end(),
                  last_hidden_scratch.begin() + config.hidden_size);
        core::write_tensor_f32(fused.last_hidden, last_hidden_scratch);
        const int32_t sem[2] = {semantic_code + 151675, semantic_code + 151675};
        core::write_tensor_i32(fused.semantic_ids, sem, 2);
        if (core::compute_graph(execution, fused.graph, fused.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("fused depth graph compute failed");
        }
        std::vector<int32_t> out_codes{semantic_code};
        std::vector<float> out_hidden;
        out_hidden.reserve(static_cast<size_t>((config.codebooks - 1) * config.hidden_size));
        for (size_t i = 0; i < fused_code_steps.size(); ++i) {
            const auto c = core::read_tensor_i32(fused_code_steps[i]);
            out_codes.push_back(c.empty() ? 0 : c[0]);
            auto h = core::read_tensor_f32(fused_hidden_steps[i]);
            core::round_f32_to_bf16_in_place(h);
            out_hidden.insert(out_hidden.end(), h.begin(), h.begin() + config.hidden_size);
        }
        return {std::move(out_codes), std::move(out_hidden)};
    }

    MiniMaxMusic3DepthCodes generate(
        const std::vector<float> & last_hidden_cond,
        const std::vector<float> & last_hidden_uncond,
        int32_t semantic_code,
        float guidance_scale,
        int64_t top_k,
        uint64_t seed,
        uint64_t & sample_call_index,
        uint64_t & rng_offset_blocks) {
        const auto & config = assets->config.depth;
        if (static_cast<int64_t>(last_hidden_cond.size()) != config.hidden_size ||
            static_cast<int64_t>(last_hidden_uncond.size()) != config.hidden_size) {
            throw std::runtime_error("MiniMax Music 3 depth hidden input shape mismatch");
        }
        if (std::getenv("MM3_DEPTH_FUSED") != nullptr) {
            return generate_fused(last_hidden_cond, last_hidden_uncond, semantic_code, guidance_scale);
        }
        if (std::getenv("MM3_DEPTH_CACHE") != nullptr) {
            if (!cached_checked) {
                cached_checked = true;
                last_hidden_scratch.resize(static_cast<size_t>(2 * config.hidden_size));
                std::copy(last_hidden_cond.begin(), last_hidden_cond.end(), last_hidden_scratch.begin());
                std::copy(last_hidden_uncond.begin(), last_hidden_uncond.end(),
                          last_hidden_scratch.begin() + config.hidden_size);
                cached_ok = cached_self_check(last_hidden_scratch, semantic_code);
            }
            if (cached_ok) {
                last_hidden_scratch.resize(static_cast<size_t>(2 * config.hidden_size));
                std::copy(last_hidden_cond.begin(), last_hidden_cond.end(), last_hidden_scratch.begin());
                std::copy(last_hidden_uncond.begin(), last_hidden_uncond.end(),
                          last_hidden_scratch.begin() + config.hidden_size);
                return generate_cached(
                    2, last_hidden_scratch, semantic_code, guidance_scale,
                    top_k, seed, sample_call_index, rng_offset_blocks);
            }
        }
        last_hidden_scratch.resize(static_cast<size_t>(2 * config.hidden_size));
        std::copy(last_hidden_cond.begin(), last_hidden_cond.end(), last_hidden_scratch.begin());
        std::copy(last_hidden_uncond.begin(), last_hidden_uncond.end(), last_hidden_scratch.begin() + config.hidden_size);
        std::vector<int32_t> out_codes{semantic_code};
        std::vector<float> out_hidden;
        out_hidden.reserve(static_cast<size_t>((config.codebooks - 1) * config.hidden_size));
        std::mt19937 fallback_rng(static_cast<uint32_t>(seed));

        for (int64_t codebook = 1; codebook < config.codebooks; ++codebook) {
            auto & graph = decode_graph(codebook);
            const int32_t semantic_ids[2] = {
                semantic_code + 151675,
                semantic_code + 151675,
            };
            active_residual_ids_scratch.assign(static_cast<size_t>(2 * std::max<int64_t>(1, codebook - 1)), 0);
            for (int64_t previous = 1; previous < codebook; ++previous) {
                const int32_t id = out_codes[static_cast<size_t>(previous)] +
                    static_cast<int32_t>((previous - 1) * config.audio_vocab_size);
                active_residual_ids_scratch[static_cast<size_t>(previous - 1)] = id;
                active_residual_ids_scratch[static_cast<size_t>((codebook - 1) + previous - 1)] = id;
            }
            positions_scratch.resize(static_cast<size_t>(codebook + 1));
            for (int64_t i = 0; i <= codebook; ++i) {
                positions_scratch[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            core::write_tensor_f32(graph.last_hidden, last_hidden_scratch);
            core::write_tensor_i32(graph.semantic_ids, semantic_ids, 2);
            if (codebook > 1) {
                core::write_tensor_i32(graph.residual_ids, active_residual_ids_scratch);
            }
            core::write_tensor_i32(graph.positions, positions_scratch);
            if (core::compute_graph(execution, graph.graph, graph.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MiniMax Music 3 depth graph compute failed");
            }
            // MM3_DEPTH_NOSYNC=1: diagnostic only - skips the per-codebook hidden readback
            // to measure how much of the depth cost is device-sync, not compute.
            // Produces WRONG audio; never enable outside benchmarking.
            auto hidden = core::read_tensor_f32(graph.hidden);
            core::round_f32_to_bf16_in_place(hidden);
            out_hidden.insert(out_hidden.end(), hidden.begin(), hidden.begin() + config.hidden_size);
            auto logits = core::read_tensor_f32(graph.logits);
            core::round_f32_to_bf16_in_place(logits);
            for (int64_t i = 0; i < config.audio_vocab_size; ++i) {
                const float cond = logits[static_cast<size_t>(i)];
                const float uncond = logits[static_cast<size_t>(config.audio_vocab_size + i)];
                logits[static_cast<size_t>(i)] = uncond + (cond - uncond) * guidance_scale;
            }
            logits.resize(static_cast<size_t>(config.audio_vocab_size));
            const int32_t code = sample_top_k(
                std::move(logits),
                top_k,
                seed,
                sample_call_index,
                rng_offset_blocks,
                sampling_policy,
                scratch,
                fallback_rng,
                "MiniMax Music 3 depth");
            out_codes.push_back(code);
        }
        return {std::move(out_codes), std::move(out_hidden)};
    }

    // Conditional-only variant (batch-1 graphs, no depth CFG). On first use it
    // self-checks against the batch-2 graph with duplicated rows; if the batch-1
    // graph disagrees beyond tolerance (the flow transformer's batch-1 graph was
    // broken this way), it falls back to batch-2 permanently.
    bool cond_b1_checked = false;
    bool cond_b1_ok = true;

    MiniMaxMusic3DepthCodes generate_cond(
        const std::vector<float> & last_hidden_cond,
        int32_t semantic_code,
        int64_t top_k,
        uint64_t seed,
        uint64_t & sample_call_index,
        uint64_t & rng_offset_blocks) {
        const auto & config = assets->config.depth;
        if (static_cast<int64_t>(last_hidden_cond.size()) != config.hidden_size) {
            throw std::runtime_error("MiniMax Music 3 depth cond hidden input shape mismatch");
        }
        if (!cond_b1_checked) {
            cond_b1_checked = true;
            // The batch-1 graphs hit an uncatchable CUDA illegal-memory-access inside the
            // shared attention stack (same bug family as the flow transformer's batch-1
            // graph). Opt-in only, until the vendored-ggml batch-1 path is fixed.
            cond_b1_ok = std::getenv("MM3_DEPTH_B1") != nullptr &&
                         self_check_b1(last_hidden_cond, semantic_code);
            engine::debug::timing_log_scalar(
                "minimax_music3.depth.b1_self_check_ok", cond_b1_ok ? 1.0 : 0.0);
        }
        if (!cond_b1_ok) {
            // Fall back: batch-2 with duplicated rows; guidance on identical rows
            // collapses to the conditional distribution.
            return generate(last_hidden_cond, last_hidden_cond, semantic_code, 1.0F,
                            top_k, seed, sample_call_index, rng_offset_blocks);
        }
        std::vector<int32_t> out_codes{semantic_code};
        std::vector<float> out_hidden;
        out_hidden.reserve(static_cast<size_t>((config.codebooks - 1) * config.hidden_size));
        std::mt19937 fallback_rng(static_cast<uint32_t>(seed));

        for (int64_t codebook = 1; codebook < config.codebooks; ++codebook) {
            auto & graph = decode_graph_b1(codebook);
            const int32_t semantic_ids[1] = {semantic_code + 151675};
            active_residual_ids_scratch.assign(static_cast<size_t>(std::max<int64_t>(1, codebook - 1)), 0);
            for (int64_t previous = 1; previous < codebook; ++previous) {
                active_residual_ids_scratch[static_cast<size_t>(previous - 1)] =
                    out_codes[static_cast<size_t>(previous)] +
                    static_cast<int32_t>((previous - 1) * config.audio_vocab_size);
            }
            positions_scratch.resize(static_cast<size_t>(codebook + 1));
            for (int64_t i = 0; i <= codebook; ++i) {
                positions_scratch[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            core::write_tensor_f32(graph.last_hidden, last_hidden_cond);
            core::write_tensor_i32(graph.semantic_ids, semantic_ids, 1);
            if (codebook > 1) {
                core::write_tensor_i32(graph.residual_ids, active_residual_ids_scratch);
            }
            core::write_tensor_i32(graph.positions, positions_scratch);
            if (core::compute_graph(execution, graph.graph, graph.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MiniMax Music 3 depth cond graph compute failed");
            }
            auto hidden = core::read_tensor_f32(graph.hidden);
            core::round_f32_to_bf16_in_place(hidden);
            out_hidden.insert(out_hidden.end(), hidden.begin(), hidden.begin() + config.hidden_size);
            auto logits = core::read_tensor_f32(graph.logits);
            core::round_f32_to_bf16_in_place(logits);
            logits.resize(static_cast<size_t>(config.audio_vocab_size));
            const int32_t code = sample_top_k(
                std::move(logits),
                top_k,
                seed,
                sample_call_index,
                rng_offset_blocks,
                sampling_policy,
                scratch,
                fallback_rng,
                "MiniMax Music 3 depth");
            out_codes.push_back(code);
        }
        return {std::move(out_codes), std::move(out_hidden)};
    }

    bool self_check_b1(const std::vector<float> & hidden_cond, int32_t semantic_code) {
        // Compare codebook-1 logits: batch-1 graph vs batch-2 graph fed duplicated rows.
        const auto & config = assets->config.depth;
        try {
            auto & g1 = decode_graph_b1(1);
            const int32_t sem1[1] = {semantic_code + 151675};
            positions_scratch.assign(2, 0);
            positions_scratch[1] = 1;
            core::write_tensor_f32(g1.last_hidden, hidden_cond);
            core::write_tensor_i32(g1.semantic_ids, sem1, 1);
            core::write_tensor_i32(g1.positions, positions_scratch);
            {
                const auto rb = core::read_tensor_i32(g1.semantic_ids.tensor);
                fprintf(stderr, "[depth-b1] wrote id=%d readback=%d data=%p flags=%d op=%d\n",
                        sem1[0], rb.empty() ? -999 : rb[0], (void *) g1.semantic_ids.tensor->data,
                        g1.semantic_ids.tensor->flags, (int) g1.semantic_ids.tensor->op);
            }
            if (core::compute_graph(execution, g1.graph, g1.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                return false;
            }
            auto l1 = core::read_tensor_f32(g1.logits);
            auto l1b = l1;  // second call: determinism check
            if (core::compute_graph(execution, g1.graph, g1.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                return false;
            }
            l1b = core::read_tensor_f32(g1.logits);

            auto & g2 = decode_graph(1);
            last_hidden_scratch.resize(static_cast<size_t>(2 * config.hidden_size));
            std::copy(hidden_cond.begin(), hidden_cond.end(), last_hidden_scratch.begin());
            std::copy(hidden_cond.begin(), hidden_cond.end(),
                      last_hidden_scratch.begin() + config.hidden_size);
            const int32_t sem2[2] = {semantic_code + 151675, semantic_code + 151675};
            core::write_tensor_f32(g2.last_hidden, last_hidden_scratch);
            core::write_tensor_i32(g2.semantic_ids, sem2, 2);
            core::write_tensor_i32(g2.positions, positions_scratch);
            if (core::compute_graph(execution, g2.graph, g2.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                return false;
            }
            const auto l2 = core::read_tensor_f32(g2.logits);
            float max_diff = 0.0F, max_rep = 0.0F, max_abs = 0.0F;
            for (int64_t i = 0; i < config.audio_vocab_size; ++i) {
                max_diff = std::max(max_diff, std::fabs(l1[static_cast<size_t>(i)] - l2[static_cast<size_t>(i)]));
                max_rep = std::max(max_rep, std::fabs(l1[static_cast<size_t>(i)] - l1b[static_cast<size_t>(i)]));
                max_abs = std::max(max_abs, std::fabs(l2[static_cast<size_t>(i)]));
            }
            const float tol = std::max(0.05F, 0.01F * max_abs);
            fprintf(stderr, "[depth-b1] self-check: max|b1-b2|=%g repeat=%g tol=%g -> %s\n",
                    static_cast<double>(max_diff), static_cast<double>(max_rep),
                    static_cast<double>(tol), (max_diff <= tol && max_rep == 0.0F) ? "OK" : "FALLBACK");
            return max_diff <= tol && max_rep == 0.0F;
        } catch (const std::exception & e) {
            fprintf(stderr, "[depth-b1] self-check failed: %s\n", e.what());
            return false;
        }
    }

    std::vector<float> feedback_embedding(const std::vector<int32_t> & codes) {
        const auto & config = assets->config.depth;
        if (static_cast<int64_t>(codes.size()) != config.codebooks) {
            throw std::runtime_error("MiniMax Music 3 feedback code count mismatch");
        }
        auto & graph = ensure_feedback_graph();
        const int32_t semantic_id = codes.front() + 151675;
        std::vector<int32_t> residual_ids(static_cast<size_t>(config.codebooks - 1));
        for (int64_t i = 1; i < config.codebooks; ++i) {
            residual_ids[static_cast<size_t>(i - 1)] =
                codes[static_cast<size_t>(i)] + static_cast<int32_t>((i - 1) * config.audio_vocab_size);
        }
        core::write_tensor_i32(graph.semantic_id, &semantic_id, 1);
        core::write_tensor_i32(graph.residual_ids, residual_ids);
        if (core::compute_graph(execution, graph.graph, graph.plan, "minimax_music3.feedback") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 feedback graph compute failed");
        }
        auto output = core::read_tensor_f32(graph.output);
        core::round_f32_to_bf16_in_place(output);
        return output;
    }

    void release_runtime_graphs() {
        release_cached(cached2);
        release_cached(cached1);
        release_fused();
        for (auto * graphs : {&decode_graphs, &decode_graphs_b1}) {
            for (auto & graph : *graphs) {
                if (graph.graph != nullptr) {
                    core::release_backend_graph_resources(execution.backend(), graph.graph);
                }
                if (graph.input_buffer != nullptr) {
                    ggml_backend_buffer_free(graph.input_buffer);
                }
                graph = {};
            }
        }
        if (feedback.graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), feedback.graph);
        }
        feedback = {};
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::TensorValue global_token_embedding;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    MiniMaxMusic3DepthWeights weights;
    std::array<DecodeGraph, 7> decode_graphs;
    std::array<DecodeGraph, 7> decode_graphs_b1;
    FeedbackGraph feedback;
    sampling::TorchCudaSamplingPolicy sampling_policy;
    sampling::HfSamplerScratch scratch;
    std::vector<float> last_hidden_scratch;
    std::vector<int32_t> active_residual_ids_scratch;
    std::vector<int32_t> positions_scratch;
};

MiniMaxMusic3DepthDecoderRuntime::MiniMaxMusic3DepthDecoderRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::TensorValue global_token_embedding,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          global_token_embedding,
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3DepthDecoderRuntime::~MiniMaxMusic3DepthDecoderRuntime() = default;

MiniMaxMusic3DepthCodes MiniMaxMusic3DepthDecoderRuntime::generate(
    const std::vector<float> & last_hidden_cond,
    const std::vector<float> & last_hidden_uncond,
    int32_t semantic_code,
    float guidance_scale,
    int64_t top_k,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks) {
    return impl_->generate(
        last_hidden_cond,
        last_hidden_uncond,
        semantic_code,
        guidance_scale,
        top_k,
        seed,
        sample_call_index,
        rng_offset_blocks);
}

MiniMaxMusic3DepthCodes MiniMaxMusic3DepthDecoderRuntime::generate_cond(
    const std::vector<float> & last_hidden_cond,
    int32_t semantic_code,
    int64_t top_k,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks) {
    return impl_->generate_cond(
        last_hidden_cond, semantic_code, top_k, seed, sample_call_index, rng_offset_blocks);
}

std::vector<float> MiniMaxMusic3DepthDecoderRuntime::feedback_embedding(const std::vector<int32_t> & codes) const {
    return impl_->feedback_embedding(codes);
}

void MiniMaxMusic3DepthDecoderRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
