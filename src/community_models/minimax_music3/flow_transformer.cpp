#include "engine/community_models/minimax_music3/flow_transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

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

MiniMaxMusic3FlowWeights load_flow_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.flow;
    const auto & source = *assets.transformer_weights;
    MiniMaxMusic3FlowWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.flow.weights",
        weight_context_bytes);
    const int64_t concat_channels = 2 * config.in_channels + config.condition_dim;
    out.preprocess_conv = binding::conv1d_from_source(
        *out.store,
        source,
        "preprocess_conv",
        conv_safe_storage_type(source, "preprocess_conv", storage_type),
        concat_channels,
        concat_channels,
        1,
        false);
    out.proj_in = binding::linear_from_source(*out.store, source, "proj_in", storage_type, config.attention_heads * config.head_dim, concat_channels, false);
    out.time_proj_weight = out.store->load_f32_tensor(source, "time_proj.weight", {config.fourier_embedding_dim / 2, 1});
    out.time_linear_1 = binding::linear_from_source(*out.store, source, "time_embed.linear_1", storage_type, config.attention_heads * config.head_dim, config.fourier_embedding_dim, true);
    out.time_linear_2 = binding::linear_from_source(*out.store, source, "time_embed.linear_2", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, true);
    out.blocks.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        const std::string prefix = "transformer_blocks." + std::to_string(layer);
        MiniMaxMusic3FlowBlockWeights block;
        block.norm1 = binding::norm_from_source(*out.store, source, prefix + ".norm1", config.attention_heads * config.head_dim);
        block.q = binding::linear_from_source(*out.store, source, prefix + ".attn.to_q", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.k = binding::linear_from_source(*out.store, source, prefix + ".attn.to_k", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.v = binding::linear_from_source(*out.store, source, prefix + ".attn.to_v", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.out = binding::linear_from_source(*out.store, source, prefix + ".attn.to_out.0", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.norm2 = binding::norm_from_source(*out.store, source, prefix + ".norm2", config.attention_heads * config.head_dim);
        block.ff_in = binding::linear_from_source(*out.store, source, prefix + ".ff_in", storage_type, 2 * config.ff_inner_dim, config.attention_heads * config.head_dim, true);
        block.ff_out = binding::linear_from_source(*out.store, source, prefix + ".ff_out", storage_type, config.attention_heads * config.head_dim, config.ff_inner_dim, true);
        out.blocks.push_back(std::move(block));
    }
    out.proj_out = binding::linear_from_source(*out.store, source, "proj_out", storage_type, config.in_channels, config.attention_heads * config.head_dim, false);
    out.postprocess_conv = binding::conv1d_from_source(
        *out.store,
        source,
        "postprocess_conv",
        conv_safe_storage_type(source, "postprocess_conv", storage_type),
        config.in_channels,
        config.in_channels,
        1,
        false);
    out.store->upload();
    assets.transformer_weights->release_storage();
    return out;
}

std::vector<float> fourier_embedding(const std::vector<float> & weights, int64_t dim, float timestep) {
    constexpr float kTwoPi = 6.2831853071795864769F;
    std::vector<float> out(static_cast<size_t>(dim), 0.0F);
    const int64_t half = dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float angle = kTwoPi * timestep * weights[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = std::cos(angle);
        out[static_cast<size_t>(half + i)] = std::sin(angle);
    }
    return out;
}

std::vector<float> split_rope_table(int64_t steps, int64_t heads, int64_t rotary_dim, bool cosine) {
    std::vector<float> out(static_cast<size_t>(steps * heads * rotary_dim / 2), 0.0F);
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t i = 0; i < rotary_dim / 2; ++i) {
            const float inv_freq = std::pow(10000.0F, -static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
            const float value = cosine ? std::cos(static_cast<float>(step) * inv_freq) : std::sin(static_cast<float>(step) * inv_freq);
            for (int64_t head = 0; head < heads; ++head) {
                out[static_cast<size_t>((step * heads + head) * (rotary_dim / 2) + i)] = value;
            }
        }
    }
    return out;
}

core::TensorValue apply_partial_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t rotary_dim) {
    const auto rotary = modules::SliceModule({3, 0, rotary_dim}).build(ctx, input);
    const auto rest = modules::SliceModule({3, rotary_dim, input.shape.dims[3] - rotary_dim}).build(ctx, input);
    const auto rotated = modules::SplitRoPEModule({rotary_dim}).build(ctx, rotary, cos, sin);
    return modules::ConcatModule({3}).build(ctx, rotated, rest);
}

}  // namespace

struct MiniMaxMusic3FlowTransformerRuntime::Impl {
    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::ExecutionContext & input_execution,
        size_t input_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          execution(input_execution),
          graph_arena_bytes(input_graph_arena_bytes),
          weights(load_flow_weights(*assets, execution, weight_context_bytes, storage_type)) {
        time_proj = assets->transformer_weights->require_f32(
            "time_proj.weight",
            {assets->config.flow.fourier_embedding_dim / 2, 1});
    }

    // One fully independent graph per batch size: sharing a ggml arena between the
    // batch-2 and batch-1 graphs made rebuilds alias each other in the backend's
    // CUDA-graph cache (nodes land at identical addresses) and corrupted replays.
    struct GraphSlot {
        // Inputs live in a dedicated backend buffer: the graph allocator was caught
        // overwriting flagged input leaves with node outputs (root cause of the
        // batch-1 corruption; see depth_decoder.cpp for the same fix).
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> input_ggml;
        ggml_backend_buffer_t input_buffer = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue latents;
        core::TensorValue condition;
        core::TensorValue time_features;
        core::TensorValue rope_cos;
        core::TensorValue rope_sin;
        ggml_tensor * output = nullptr;
        int64_t latent_frames = 0;
        int64_t condition_version = -1;
        std::vector<float> latent_batch;
        std::vector<float> time_batch;
        std::vector<float> rope_cos_table;
        std::vector<float> rope_sin_table;
    };


    ~Impl() {
        release_runtime_graphs();
    }

    void release_runtime_graphs() {
        for (GraphSlot * slot : {&slot2, &slot1}) {
            if (slot->graph != nullptr) {
                core::release_backend_graph_resources(execution.backend(), slot->graph);
            }
            slot->graph = nullptr;
            slot->latents = {};
            slot->condition = {};
            slot->time_features = {};
            slot->rope_cos = {};
            slot->rope_sin = {};
            slot->output = nullptr;
            slot->gallocr.reset();
            if (slot->input_buffer != nullptr) {
                ggml_backend_buffer_free(slot->input_buffer);
                slot->input_buffer = nullptr;
            }
            slot->input_ggml.reset();
            slot->ggml.reset();
            slot->plan.reset();
            slot->latent_frames = 0;
            slot->rope_cos_table.clear();
            slot->rope_sin_table.clear();
        }
    }

    void ensure_graph(GraphSlot & slot, int64_t frames, int64_t batch) {
        if (slot.graph != nullptr && slot.latent_frames == frames) {
            return;
        }
        if (slot.graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), slot.graph);
            slot.graph = nullptr;
        }
        if (slot.input_buffer != nullptr) {
            ggml_backend_buffer_free(slot.input_buffer);
            slot.input_buffer = nullptr;
        }
        slot.input_ggml.reset();
        const auto & config = assets->config.flow;
        const int64_t inner = config.attention_heads * config.head_dim;
        const int64_t steps = frames + 1;
        const int64_t concat_channels = 2 * config.in_channels + config.condition_dim;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        slot.ggml.reset(ggml_init(params));
        if (slot.ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 flow slot.graph context");
        }
        core::ModuleBuildContext ctx{slot.ggml.get(), "minimax_music3.flow", execution.backend_type()};
        {
            ggml_init_params input_params{16384, nullptr, true};
            slot.input_ggml.reset(ggml_init(input_params));
            if (slot.input_ggml == nullptr) {
                throw std::runtime_error("failed to initialize MiniMax Music 3 flow input context");
            }
        }
        core::ModuleBuildContext input_ctx{slot.input_ggml.get(), "minimax_music3.flow.in", execution.backend_type()};
        slot.latents = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.in_channels, frames}));
        slot.condition = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.condition_dim, frames}));
        slot.time_features = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.fourier_embedding_dim}));
        slot.rope_cos = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.attention_heads, config.rotary_dim / 2}));
        slot.rope_sin = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.attention_heads, config.rotary_dim / 2}));
        slot.input_buffer = ggml_backend_alloc_ctx_tensors(slot.input_ggml.get(), execution.backend());
        if (slot.input_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 flow input buffer");
        }
        ggml_set_input(slot.latents.tensor);
        ggml_set_input(slot.condition.tensor);
        ggml_set_input(slot.time_features.tensor);
        ggml_set_input(slot.rope_cos.tensor);
        ggml_set_input(slot.rope_sin.tensor);

        auto zeros = core::wrap_tensor(ggml_scale(ctx.ggml, slot.latents.tensor, 0.0F), slot.latents.shape, GGML_TYPE_F32);
        auto x = modules::ConcatModule({1}).build(ctx, slot.latents, zeros);
        x = modules::ConcatModule({1}).build(ctx, x, slot.condition);
        auto conv = modules::Conv1dModule({concat_channels, concat_channels, 1, 1, 0, 1, false}).build(
            ctx,
            x,
            weights.preprocess_conv);
        x = core::wrap_tensor(ggml_add(ctx.ggml, conv.tensor, x.tensor), conv.shape, GGML_TYPE_F32);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        x = modules::LinearModule({concat_channels, inner, false}).build(ctx, x, weights.proj_in);

        auto temb = modules::LinearModule({config.fourier_embedding_dim, inner, true}).build(
            ctx,
            slot.time_features,
            weights.time_linear_1);
        temb = modules::SiluModule().build(ctx, temb);
        temb = modules::LinearModule({inner, inner, true}).build(ctx, temb, weights.time_linear_2);
        temb = core::reshape_tensor(ctx, temb, core::TensorShape::from_dims({batch, 1, inner}));
        x = modules::ConcatModule({1}).build(ctx, temb, x);

        for (const auto & block : weights.blocks) {
            auto normed = modules::LayerNormModule({inner, 1.0e-5F, true, true, false}).build(ctx, x, block.norm1);
            auto q = modules::LinearModule({inner, inner, false}).build(ctx, normed, block.q);
            auto k = modules::LinearModule({inner, inner, false}).build(ctx, normed, block.k);
            auto v = modules::LinearModule({inner, inner, false}).build(ctx, normed, block.v);
            q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({batch, steps, config.attention_heads, config.head_dim}));
            k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({batch, steps, config.attention_heads, config.head_dim}));
            v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({batch, steps, config.attention_heads, config.head_dim}));
            q = apply_partial_rope(ctx, q, slot.rope_cos, slot.rope_sin, config.rotary_dim);
            k = apply_partial_rope(ctx, k, slot.rope_cos, slot.rope_sin, config.rotary_dim);
            q = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
            auto attn = modules::ScaledDotProductAttentionModule({
                config.head_dim,
                // Flash lowering is nondeterministic/garbage on the batch-1 slot.graph shape
                // (b1-repeat probe diff ~8.0); use the explicit path there.
                (core::uses_ggml_cuda_or_hip_backend(execution.backend_type()) && batch > 1)
                    ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
                    : modules::ScaledDotProductAttentionLowering::Explicit,
                GGML_PREC_F32,
                modules::AttentionCausality::NonCausal,
            }).build(ctx, q, k, v);
            attn = core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(ctx, attn),
                core::TensorShape::from_dims({batch, steps, inner}));
            attn = modules::LinearModule({inner, inner, false}).build(ctx, attn, block.out);
            x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, attn.tensor), x.shape, GGML_TYPE_F32);

            auto ffn = modules::LayerNormModule({inner, 1.0e-5F, true, true, false}).build(ctx, x, block.norm2);
            ffn = modules::LinearModule({inner, 2 * config.ff_inner_dim, true}).build(ctx, ffn, block.ff_in);
            const auto gate_states = modules::SliceModule({2, 0, config.ff_inner_dim}).build(ctx, ffn);
            auto gate = modules::SliceModule({2, config.ff_inner_dim, config.ff_inner_dim}).build(ctx, ffn);
            gate = modules::SiluModule().build(ctx, gate);
            auto gated = core::wrap_tensor(
                ggml_mul(ctx.ggml, gate_states.tensor, gate.tensor),
                gate_states.shape,
                GGML_TYPE_F32);
            gated = modules::LinearModule({config.ff_inner_dim, inner, true}).build(ctx, gated, block.ff_out);
            x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, gated.tensor), x.shape, GGML_TYPE_F32);
        }

        x = modules::SliceModule({1, 1, frames}).build(ctx, x);
        x = modules::LinearModule({inner, config.in_channels, false}).build(ctx, x, weights.proj_out);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        auto post = modules::Conv1dModule({config.in_channels, config.in_channels, 1, 1, 0, 1, false}).build(
            ctx,
            x,
            weights.postprocess_conv);
        x = core::wrap_tensor(ggml_add(ctx.ggml, post.tensor, x.tensor), post.shape, GGML_TYPE_F32);
        slot.output = x.tensor;
        slot.graph = ggml_new_graph_custom(slot.ggml.get(), 524288, false);
        ggml_build_forward_expand(slot.graph, slot.output);
        slot.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (slot.gallocr == nullptr || !ggml_gallocr_reserve(slot.gallocr.get(), slot.graph) ||
            !ggml_gallocr_alloc_graph(slot.gallocr.get(), slot.graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 flow graph");
        }
        core::prepare_host_graph_plan(execution, slot.graph, slot.plan);
        slot.latent_frames = frames;
        slot.condition_version = -1;
        slot.rope_cos_table = split_rope_table(steps, config.attention_heads, config.rotary_dim, true);
        slot.rope_sin_table = split_rope_table(steps, config.attention_heads, config.rotary_dim, false);
        core::write_tensor_f32(slot.rope_cos, slot.rope_cos_table);
        core::write_tensor_f32(slot.rope_sin, slot.rope_sin_table);
        slot.latent_batch.resize(static_cast<size_t>(batch * config.in_channels * frames));
        slot.time_batch.resize(static_cast<size_t>(batch * config.fourier_embedding_dim));
    }

    void prepare_chunk_condition(
        const std::vector<float> & input_condition,
        int64_t frames) {
        const auto & config = assets->config.flow;
        if (static_cast<int64_t>(input_condition.size()) != config.condition_dim * frames) {
            throw std::runtime_error("MiniMax Music 3 flow condition shape mismatch");
        }
        condition_batch.assign(static_cast<size_t>(2 * config.condition_dim * frames), 0.0F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < config.condition_dim; ++channel) {
                condition_batch[static_cast<size_t>(channel * frames + frame)] =
                    input_condition[static_cast<size_t>(frame * config.condition_dim + channel)];
            }
        }
        condition_frames = frames;
        ++condition_epoch;
    }

    std::vector<float> predict_velocity_branches(
        const std::vector<float> & input_latents,
        const std::vector<float> & input_condition,
        int64_t frames,
        float timestep) {
        const auto & config = assets->config.flow;
        if (static_cast<int64_t>(input_latents.size()) != config.in_channels * frames ||
            static_cast<int64_t>(input_condition.size()) != config.condition_dim * frames) {
            throw std::runtime_error("MiniMax Music 3 flow input shape mismatch");
        }
        ensure_graph(slot2, frames, 2);
        if (condition_frames != frames ||
            condition_batch.size() != static_cast<size_t>(2 * config.condition_dim * frames)) {
            throw std::runtime_error("MiniMax Music 3 flow slot2.condition was not prepared for this chunk");
        }
        std::copy(input_latents.begin(), input_latents.end(), slot2.latent_batch.begin());
        std::copy(input_latents.begin(), input_latents.end(), slot2.latent_batch.begin() + input_latents.size());
        const auto time_feature = fourier_embedding(time_proj, config.fourier_embedding_dim, timestep);
        std::copy(time_feature.begin(), time_feature.end(), slot2.time_batch.begin());
        std::copy(time_feature.begin(), time_feature.end(), slot2.time_batch.begin() + time_feature.size());
        core::write_tensor_f32(slot2.latents, slot2.latent_batch);
        if (slot2.condition_version != condition_epoch) {
            // The chunk condition is constant across denoise steps - upload once per chunk
            // instead of pushing 11 MB over PCIe on every forward.
            core::write_tensor_f32(slot2.condition, condition_batch);
            slot2.condition_version = condition_epoch;
        }
        core::write_tensor_f32(slot2.time_features, slot2.time_batch);
        if (core::compute_graph(execution, slot2.graph, slot2.plan, "minimax_music3.flow") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 flow slot2.graph compute failed");
        }
        return core::read_tensor_f32(slot2.output);
    }

    // Conditional-only forward (batch 1). Used after the guidance cutoff, where the
    // unconditional branch no longer changes the trajectory (measured corr > 0.999).
    std::vector<float> predict_velocity_cond(
        const std::vector<float> & input_latents,
        const std::vector<float> & input_condition,
        int64_t frames,
        float timestep) {
        const auto & config = assets->config.flow;
        if (static_cast<int64_t>(input_latents.size()) != config.in_channels * frames ||
            static_cast<int64_t>(input_condition.size()) != config.condition_dim * frames) {
            throw std::runtime_error("MiniMax Music 3 flow input shape mismatch");
        }
        ensure_graph(slot1, frames, 1);
        if (condition_frames != frames ||
            condition_batch.size() != static_cast<size_t>(2 * config.condition_dim * frames)) {
            throw std::runtime_error("MiniMax Music 3 flow slot1.condition was not prepared for this chunk");
        }
        std::copy(input_latents.begin(), input_latents.end(), slot1.latent_batch.begin());
        const auto time_feature = fourier_embedding(time_proj, config.fourier_embedding_dim, timestep);
        std::copy(time_feature.begin(), time_feature.end(), slot1.time_batch.begin());
        // condition_batch row 0 holds the conditional features; row 1 is the CFG zeros.
        // Keep the staging buffer alive on the Impl - a stack temporary here raced the
        // backend's copy and produced nondeterministic garbage.
        core::write_tensor_f32(slot1.latents, slot1.latent_batch);
        if (slot1.condition_version != condition_epoch) {
            condition_single.assign(
                condition_batch.begin(),
                condition_batch.begin() + static_cast<std::ptrdiff_t>(config.condition_dim * frames));
            core::write_tensor_f32(slot1.condition, condition_single);
            slot1.condition_version = condition_epoch;
        }
        core::write_tensor_f32(slot1.time_features, slot1.time_batch);
        if (core::compute_graph(execution, slot1.graph, slot1.plan, "minimax_music3.flow") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 flow slot1.graph compute failed");
        }
        return core::read_tensor_f32(slot1.output);
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    MiniMaxMusic3FlowWeights weights;
    std::vector<float> time_proj;
    int64_t condition_frames = 0;
    int64_t condition_epoch = 0;
    std::vector<float> condition_batch;
    std::vector<float> condition_single;
    GraphSlot slot2;
    GraphSlot slot1;
};

MiniMaxMusic3FlowTransformerRuntime::MiniMaxMusic3FlowTransformerRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3FlowTransformerRuntime::~MiniMaxMusic3FlowTransformerRuntime() = default;

std::vector<float> MiniMaxMusic3FlowTransformerRuntime::predict_velocity_branches(
    const std::vector<float> & latents,
    const std::vector<float> & condition,
    int64_t latent_frames,
    float timestep) {
    return impl_->predict_velocity_branches(latents, condition, latent_frames, timestep);
}

std::vector<float> MiniMaxMusic3FlowTransformerRuntime::predict_velocity_cond(
    const std::vector<float> & latents,
    const std::vector<float> & condition,
    int64_t latent_frames,
    float timestep) {
    return impl_->predict_velocity_cond(latents, condition, latent_frames, timestep);
}

void MiniMaxMusic3FlowTransformerRuntime::prepare_chunk_condition(
    const std::vector<float> & condition,
    int64_t latent_frames) {
    impl_->prepare_chunk_condition(condition, latent_frames);
}

void MiniMaxMusic3FlowTransformerRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
