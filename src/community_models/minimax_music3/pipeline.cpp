#include "engine/community_models/minimax_music3/pipeline.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kCropLeftLatent = 86;
// Stitching geometry: a chunk yields `chunk_frames * latents_per_frame` latents and consecutive
// chunks advance `hop * latents_per_frame`, so the kept span must equal the hop's latents.
// The stock constant (344 - 86) is exactly this for the default hop of 100 (344 latents/hop);
// deriving it keeps a non-default MM3_CHUNK_HOP from silently truncating the song.
inline int64_t crop_right_latent(const MiniMaxMusic3Config & config) {
    int64_t hop = config.chunk_hop_frames;
    if (const char * env = std::getenv("MM3_CHUNK_HOP")) {
        const int64_t parsed = std::atoll(env);
        if (parsed > 0 && parsed <= config.chunk_frames) {
            hop = parsed;
        }
    }
    const int64_t chunk_latents = 2 * 344;  // chunk_frames(200) * latents_per_frame(3.44)
    const int64_t hop_latents = (hop * chunk_latents) / config.chunk_frames;
    return chunk_latents - kCropLeftLatent - hop_latents;
}

std::vector<int64_t> chunk_starts(int64_t frames, const MiniMaxMusic3Config & config) {
    if (frames <= 0) {
        throw std::runtime_error("MiniMax Music 3 requires positive AR frame count");
    }
    if (frames <= config.chunk_frames) {
        return {0};
    }
    // MM3_CHUNK_HOP: the flow denoises `chunk_frames` (200) windows every `chunk_hop` (100)
    // frames, so every frame is denoised ~2x. The blend only consumes ~50 frames of overlap,
    // so a larger hop cuts chunks (and flow time) at the cost of left context per window.
    int64_t hop = config.chunk_hop_frames;
    if (const char * env = std::getenv("MM3_CHUNK_HOP")) {
        const int64_t parsed = std::atoll(env);
        if (parsed > 0 && parsed <= config.chunk_frames) {
            hop = parsed;
        }
    }
    std::vector<int64_t> out;
    for (int64_t start = 0; start < frames - hop; start += hop) {
        out.push_back(start);
    }
    return out;
}

std::vector<float> condition_slice(
    const std::vector<float> & condition,
    int64_t start,
    int64_t frames,
    int64_t dim) {
    if (start < 0 || frames <= 0 || dim <= 0 ||
        static_cast<int64_t>(condition.size()) < (start + frames) * dim) {
        throw std::runtime_error("MiniMax Music 3 condition slice is out of range");
    }
    return std::vector<float>(
        condition.begin() + static_cast<std::ptrdiff_t>(start * dim),
        condition.begin() + static_cast<std::ptrdiff_t>((start + frames) * dim));
}

std::vector<float> crop_interleaved_audio(
    const runtime::AudioBuffer & audio,
    int64_t left_samples,
    int64_t right_samples) {
    if (audio.channels != 2 || audio.sample_rate <= 0 ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MiniMax Music 3 vocoder returned invalid stereo audio");
    }
    const int64_t frames = static_cast<int64_t>(audio.samples.size()) / audio.channels;
    const int64_t start = std::min(std::max<int64_t>(0, left_samples), frames);
    const int64_t end = std::max(start, frames - std::max<int64_t>(0, right_samples));
    std::vector<float> out(static_cast<size_t>((end - start) * audio.channels));
    std::copy(
        audio.samples.begin() + static_cast<std::ptrdiff_t>(start * audio.channels),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(end * audio.channels),
        out.begin());
    return out;
}

struct DenoisedChunk {
    std::vector<float> latents;
    int64_t latent_frames = 0;
};

}  // namespace

struct MiniMaxMusic3PipelineRuntime::Impl {
    Impl(
        core::ExecutionContext & input_execution,
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool memory_saver)
        : execution(input_execution),
          assets(std::move(input_assets)),
          graph_arena_bytes(graph_arena_bytes),
          weight_context_bytes(weight_context_bytes),
          storage_type(storage_type),
          memory_saver(memory_saver),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "minimax_music3.flow.sampling",
              "MiniMax Music 3 flow",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 pipeline requires assets");
        }
        if (!memory_saver) {
            ar = make_ar();
            condition = make_condition();
            flow = make_flow();
            vocoder = make_vocoder();
        }
    }

    ~Impl() {
        release_runtime_graphs();
    }

    runtime::AudioBuffer generate(const MiniMaxMusic3Request & request) {
        if (std::getenv("MM3_PIPELINE") != nullptr) {
            return generate_pipelined(request);
        }
        if (request.duration_sec <= 0.0) {
            throw std::runtime_error("MiniMax Music 3 duration_sec must be positive");
        }
        if (request.num_inference_steps <= 0) {
            throw std::runtime_error("MiniMax Music 3 num_inference_steps must be positive");
        }
        if (request.guidance_scale <= 0.0F || request.ar_guidance_scale <= 0.0F) {
            throw std::runtime_error("MiniMax Music 3 guidance scales must be positive");
        }
        if (request.top_k <= 0) {
            throw std::runtime_error("MiniMax Music 3 top_k must be positive");
        }
        const int64_t target_frames = std::min<int64_t>(
            assets->config.max_audio_frames,
            static_cast<int64_t>(request.duration_sec * static_cast<double>(assets->config.frame_rate)));
        uint64_t rng_offset_blocks = 0;
        std::vector<float> frame_hiddens;
        {
            auto & ar_runtime = ensure_ar();
            frame_hiddens = ar_runtime.generate_frame_hiddens(request, target_frames, rng_offset_blocks);
            release_ar_after_phase();
        }
        const int64_t generated_frames =
            static_cast<int64_t>(frame_hiddens.size()) /
            (assets->config.condition.condition_layers * assets->config.qwen.hidden_size);
        if (static_cast<int64_t>(frame_hiddens.size()) !=
            generated_frames * assets->config.condition.condition_layers * assets->config.qwen.hidden_size) {
            throw std::runtime_error("MiniMax Music 3 frame hidden shape mismatch");
        }
        if (generated_frames <= 0) {
            throw std::runtime_error("MiniMax Music 3 AR produced no frames");
        }

        const auto denoise_start = Clock::now();
        const auto starts = chunk_starts(generated_frames, assets->config);
        std::vector<DenoisedChunk> denoised;
        denoised.reserve(starts.size());
        // MM3_CPU_VOCODER=1: vocode each chunk on the CPU (24 threads) concurrently with
        // the GPU denoising the next chunk. Hides the vocoder stage entirely if CPU decode
        // stays under one flow-chunk time, and frees its ~1.5 GB GPU compute buffer.
        const bool cpu_vocoder = std::getenv("MM3_CPU_VOCODER") != nullptr;
        std::unique_ptr<core::ExecutionContext> cpu_execution;
        std::unique_ptr<MiniMaxMusic3VocoderRuntime> cpu_vocoder_runtime;
        std::vector<runtime::AudioBuffer> cpu_chunks;
        std::thread cpu_worker;
        std::exception_ptr cpu_error;
        if (cpu_vocoder) {
            core::BackendConfig cpu_config;
            cpu_config.type = core::BackendType::Cpu;
            cpu_config.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency() - 2));
            cpu_execution = std::make_unique<core::ExecutionContext>(cpu_config);
            cpu_vocoder_runtime = std::make_unique<MiniMaxMusic3VocoderRuntime>(
                assets, *cpu_execution, graph_arena_bytes, weight_context_bytes,
                assets::TensorStorageType::F32);
        }
        std::vector<float> previous_latent;
        std::vector<float> previous_condition;
        {
            auto & condition_runtime = ensure_condition();
            auto & flow_runtime = ensure_flow();
            for (size_t chunk_index = 0; chunk_index < starts.size(); ++chunk_index) {
                const int64_t start = starts[chunk_index];
                const int64_t end = std::min(start + assets->config.chunk_frames, generated_frames);
                const int64_t frame_count = end - start;
                int64_t condition_frames = 0;
                const auto condition_start = Clock::now();
                auto condition_values = condition_runtime.encode(
                    condition_slice(
                        frame_hiddens,
                        start,
                        frame_count,
                        assets->config.condition.condition_layers * assets->config.qwen.hidden_size),
                    frame_count,
                    condition_frames);
                core::round_f32_to_bf16_in_place(condition_values);
                engine::debug::timing_log_scalar(
                    "minimax_music3.condition.total_ms",
                    engine::debug::elapsed_ms(condition_start, Clock::now()));
                std::vector<float> carry_condition;
                std::vector<float> carry_latent;
                const auto flow_start = Clock::now();
                auto latents = flow_runtime.denoise_chunk(
                    condition_values,
                    condition_frames,
                    previous_latent,
                    previous_condition,
                    request,
                    rng_offset_blocks,
                    sampling_policy,
                    carry_condition,
                    carry_latent);
                engine::debug::timing_log_scalar(
                    "minimax_music3.flow.total_ms",
                    engine::debug::elapsed_ms(flow_start, Clock::now()));
                rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(latents.size()),
                    sampling_policy);
                previous_latent = std::move(carry_latent);
                previous_condition = std::move(carry_condition);
                denoised.push_back({std::move(latents), condition_frames});
                if (cpu_vocoder) {
                    if (cpu_worker.joinable()) {
                        cpu_worker.join();
                    }
                    if (cpu_error) {
                        std::rethrow_exception(cpu_error);
                    }
                    const auto & job = denoised.back();
                    cpu_chunks.emplace_back();
                    auto * slot_audio = &cpu_chunks.back();
                    const auto * job_ptr = &job;
                    cpu_worker = std::thread([&, slot_audio, job_ptr] {
                        try {
                            const auto vocoder_start = Clock::now();
                            *slot_audio = cpu_vocoder_runtime->decode(job_ptr->latents, job_ptr->latent_frames);
                            engine::debug::timing_log_scalar(
                                "minimax_music3.cpu_vocoder.chunk_ms",
                                engine::debug::elapsed_ms(vocoder_start, Clock::now()));
                        } catch (...) {
                            cpu_error = std::current_exception();
                        }
                    });
                }
            }
            if (cpu_worker.joinable()) {
                cpu_worker.join();
            }
            if (cpu_error) {
                std::rethrow_exception(cpu_error);
            }
            release_flow_after_phase();
        }
        std::vector<runtime::AudioBuffer> chunks;
        chunks.reserve(denoised.size());
        if (cpu_vocoder) {
            for (size_t chunk_index = 0; chunk_index < cpu_chunks.size(); ++chunk_index) {
                auto & audio = cpu_chunks[chunk_index];
                const int64_t left = chunk_index == 0 ? 0 : kCropLeftLatent * assets->config.vocoder.hop_length;
                const int64_t right =
                    chunk_index + 1 == cpu_chunks.size() ? 0 : crop_right_latent(assets->config) * assets->config.vocoder.hop_length;
                audio.samples = crop_interleaved_audio(audio, left, right);
                chunks.push_back(std::move(audio));
            }
        } else {
            auto & vocoder_runtime = ensure_vocoder();
            for (size_t chunk_index = 0; chunk_index < denoised.size(); ++chunk_index) {
                const auto vocoder_start = Clock::now();
                auto audio = vocoder_runtime.decode(
                    denoised[chunk_index].latents,
                    denoised[chunk_index].latent_frames);
                engine::debug::timing_log_scalar(
                    "minimax_music3.vocoder.total_ms",
                    engine::debug::elapsed_ms(vocoder_start, Clock::now()));
                const int64_t left = chunk_index == 0 ? 0 : kCropLeftLatent * assets->config.vocoder.hop_length;
                const int64_t right =
                    chunk_index + 1 == denoised.size() ? 0 : crop_right_latent(assets->config) * assets->config.vocoder.hop_length;
                audio.samples = crop_interleaved_audio(audio, left, right);
                chunks.push_back(std::move(audio));
            }
            release_vocoder_after_phase();
        }
        (void) 0;
        engine::debug::timing_log_scalar(
            "minimax_music3.flow_vocoder.total_ms",
            engine::debug::elapsed_ms(denoise_start, Clock::now()));

        runtime::AudioBuffer out;
        out.sample_rate = assets->config.vocoder.sample_rate;
        out.channels = 2;
        for (const auto & chunk : chunks) {
            if (chunk.sample_rate != out.sample_rate || chunk.channels != out.channels) {
                throw std::runtime_error("MiniMax Music 3 chunk audio format mismatch");
            }
            out.samples.insert(out.samples.end(), chunk.samples.begin(), chunk.samples.end());
        }
        for (float & sample : out.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return out;
    }

    void release_runtime_graphs() {
        if (ar != nullptr) {
            ar->release_runtime_graphs();
        }
        if (condition != nullptr) {
            condition->release_runtime_graphs();
        }
        if (flow != nullptr) {
            flow->release_runtime_graphs();
        }
        if (vocoder != nullptr) {
            vocoder->release_runtime_graphs();
        }
    }

    // Pipelined generation (MM3_PIPELINE=1): the flow/vocoder consumer runs on its own
    // thread with its own ExecutionContext (own CUDA stream) and starts denoising chunk k
    // as soon as its 200 frames exist, while the AR keeps producing. The AR stage is
    // bandwidth-bound and the flow stage compute-bound, so the streams genuinely overlap.
    // Flow noise uses a fixed RNG base offset (the sequential offset depends on the AR
    // total, unknown mid-flight), so output differs from the sequential mode by noise
    // draw but stays deterministic for a fixed seed.
    runtime::AudioBuffer generate_pipelined(const MiniMaxMusic3Request & request) {
        if (request.duration_sec <= 0.0) {
            throw std::runtime_error("MiniMax Music 3 duration_sec must be positive");
        }
        if (request.num_inference_steps <= 0) {
            throw std::runtime_error("MiniMax Music 3 num_inference_steps must be positive");
        }
        const int64_t target_frames = std::min<int64_t>(
            assets->config.max_audio_frames,
            static_cast<int64_t>(request.duration_sec * static_cast<double>(assets->config.frame_rate)));
        const int64_t frame_values =
            assets->config.condition.condition_layers * assets->config.qwen.hidden_size;
        const int64_t chunk_frames = assets->config.chunk_frames;
        const int64_t chunk_hop = assets->config.chunk_hop_frames;

        if (flow_execution == nullptr) {
            flow_execution = std::make_unique<core::ExecutionContext>(execution.config());
        }
        auto flow_condition = std::make_unique<MiniMaxMusic3ConditionEncoderRuntime>(
            assets, *flow_execution, graph_arena_bytes, weight_context_bytes, storage_type);
        auto flow_sampler = std::make_unique<MiniMaxMusic3FlowSamplerRuntime>(
            assets, *flow_execution, graph_arena_bytes, weight_context_bytes, storage_type);

        std::mutex m;
        std::condition_variable cv;
        std::vector<float> shared_hiddens;
        shared_hiddens.reserve(static_cast<size_t>(target_frames * frame_values));
        int64_t produced = 0;
        bool ar_done = false;
        std::exception_ptr consumer_error;
        std::vector<runtime::AudioBuffer> chunks;
        std::vector<DenoisedChunk> pending;

        const auto pipeline_start = Clock::now();
        std::thread consumer([&] {
            try {
                uint64_t flow_rng_offset = (1ull << 40);  // fixed base, disjoint from AR offsets
                std::vector<float> previous_latent;
                std::vector<float> previous_condition;
                int64_t chunk_index = 0;
                while (true) {
                    const int64_t start = chunk_index * chunk_hop;
                    int64_t total = 0;
                    {
                        std::unique_lock<std::mutex> lock(m);
                        cv.wait(lock, [&] {
                            return ar_done || produced >= start + chunk_frames;
                        });
                        total = produced;
                        if (ar_done && total <= 0) {
                            return;
                        }
                        // Is there a chunk at this start? (mirrors chunk_starts())
                        const bool have_full = total >= start + chunk_frames;
                        const bool is_chunk =
                            start == 0 ? total > 0
                                       : (ar_done ? start < total - chunk_hop : have_full);
                        if (!is_chunk) {
                            if (ar_done) {
                                break;
                            }
                            continue;
                        }
                        if (!have_full && !ar_done) {
                            continue;
                        }
                    }
                    const int64_t end = std::min<int64_t>(start + chunk_frames, total);
                    const int64_t frame_count = end - start;
                    std::vector<float> slice;
                    {
                        std::lock_guard<std::mutex> lock(m);
                        slice.assign(
                            shared_hiddens.begin() + static_cast<std::ptrdiff_t>(start * frame_values),
                            shared_hiddens.begin() + static_cast<std::ptrdiff_t>(end * frame_values));
                    }
                    int64_t condition_frames = 0;
                    auto condition_values = flow_condition->encode(slice, frame_count, condition_frames);
                    core::round_f32_to_bf16_in_place(condition_values);
                    std::vector<float> carry_condition;
                    std::vector<float> carry_latent;
                    const auto flow_start = Clock::now();
                    auto latents = flow_sampler->denoise_chunk(
                        condition_values,
                        condition_frames,
                        previous_latent,
                        previous_condition,
                        request,
                        flow_rng_offset,
                        sampling_policy,
                        carry_condition,
                        carry_latent);
                    engine::debug::timing_log_scalar(
                        "minimax_music3.pipeline.flow_chunk_ms",
                        engine::debug::elapsed_ms(flow_start, Clock::now()));
                    flow_rng_offset += sampling::torch_cuda_tensor_iterator_offset_blocks(
                        static_cast<uint64_t>(latents.size()),
                        sampling_policy);
                    previous_latent = std::move(carry_latent);
                    previous_condition = std::move(carry_condition);

                    // Vocode after the pipeline: the vocoder's compute buffer is ~1.5 GB,
                    // which does not coexist with the AR + flow stages on a 12 GB card.
                    pending.push_back({std::move(latents), condition_frames});
                    ++chunk_index;

                    bool finished = false;
                    {
                        std::lock_guard<std::mutex> lock(m);
                        const int64_t next_start = chunk_index * chunk_hop;
                        finished = ar_done && !(next_start == 0 ? produced > 0
                                                                 : next_start < produced - chunk_hop);
                    }
                    if (finished) {
                        break;
                    }
                }
            } catch (...) {
                consumer_error = std::current_exception();
            }
        });

        uint64_t rng_offset_blocks = 0;
        {
            auto & ar_runtime = ensure_ar();
            ar_runtime.generate_frame_hiddens(
                request,
                target_frames,
                rng_offset_blocks,
                [&](int64_t frames_done, const float * frame_data, int64_t values) {
                    std::lock_guard<std::mutex> lock(m);
                    shared_hiddens.insert(shared_hiddens.end(), frame_data, frame_data + values);
                    produced = frames_done;
                    cv.notify_all();
                });
        }
        {
            std::lock_guard<std::mutex> lock(m);
            ar_done = true;
            cv.notify_all();
        }
        consumer.join();
        ar->release_runtime_graphs();
        flow_condition->release_runtime_graphs();
        flow_sampler->release_runtime_graphs();
        if (consumer_error) {
            std::rethrow_exception(consumer_error);
        }
        if (pending.empty()) {
            throw std::runtime_error("MiniMax Music 3 pipelined generation produced no chunks");
        }
        {
            auto flow_vocoder = std::make_unique<MiniMaxMusic3VocoderRuntime>(
                assets, *flow_execution, graph_arena_bytes, weight_context_bytes, storage_type);
            for (auto & chunk : pending) {
                const auto vocoder_start = Clock::now();
                auto audio = flow_vocoder->decode(chunk.latents, chunk.latent_frames);
                engine::debug::timing_log_scalar(
                    "minimax_music3.vocoder.total_ms",
                    engine::debug::elapsed_ms(vocoder_start, Clock::now()));
                chunks.push_back(std::move(audio));
            }
        }
        engine::debug::timing_log_scalar(
            "minimax_music3.pipeline.total_ms",
            engine::debug::elapsed_ms(pipeline_start, Clock::now()));

        runtime::AudioBuffer out;
        out.sample_rate = assets->config.vocoder.sample_rate;
        out.channels = 2;
        for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
            auto & chunk = chunks[chunk_index];
            const int64_t left = chunk_index == 0 ? 0 : kCropLeftLatent * assets->config.vocoder.hop_length;
            const int64_t right =
                chunk_index + 1 == chunks.size() ? 0 : crop_right_latent(assets->config) * assets->config.vocoder.hop_length;
            chunk.samples = crop_interleaved_audio(chunk, left, right);
            out.samples.insert(out.samples.end(), chunk.samples.begin(), chunk.samples.end());
        }
        for (float & sample : out.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return out;
    }

    MiniMaxMusic3ArRuntime & ensure_ar() {
        if (ar == nullptr) {
            ar = make_ar();
        }
        return *ar;
    }

    MiniMaxMusic3ConditionEncoderRuntime & ensure_condition() {
        if (condition == nullptr) {
            condition = make_condition();
        }
        return *condition;
    }

    MiniMaxMusic3FlowSamplerRuntime & ensure_flow() {
        if (flow == nullptr) {
            flow = make_flow();
        }
        return *flow;
    }

    MiniMaxMusic3VocoderRuntime & ensure_vocoder() {
        if (vocoder == nullptr) {
            vocoder = make_vocoder();
        }
        return *vocoder;
    }

    void release_ar_after_phase() {
        if (ar == nullptr) {
            return;
        }
        ar->release_runtime_graphs();
        if (memory_saver) {
            ar.reset();
        }
    }

    void release_flow_after_phase() {
        if (condition != nullptr) {
            condition->release_runtime_graphs();
        }
        if (flow != nullptr) {
            flow->release_runtime_graphs();
        }
        if (memory_saver) {
            flow.reset();
            condition.reset();
        }
    }

    void release_vocoder_after_phase() {
        if (vocoder == nullptr) {
            return;
        }
        vocoder->release_runtime_graphs();
        if (memory_saver) {
            vocoder.reset();
        }
    }

    std::unique_ptr<MiniMaxMusic3ArRuntime> make_ar() {
        const auto t0 = Clock::now();
        auto out = std::make_unique<MiniMaxMusic3ArRuntime>(
            assets, execution, graph_arena_bytes, weight_context_bytes, storage_type);
        engine::debug::timing_log_scalar(
            "minimax_music3.load.ar_ms", engine::debug::elapsed_ms(t0, Clock::now()));
        return out;
    }

    std::unique_ptr<MiniMaxMusic3ArRuntime> make_ar_unused() {
        return std::make_unique<MiniMaxMusic3ArRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    std::unique_ptr<MiniMaxMusic3ConditionEncoderRuntime> make_condition() {
        return std::make_unique<MiniMaxMusic3ConditionEncoderRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    std::unique_ptr<MiniMaxMusic3FlowSamplerRuntime> make_flow() {
        const auto t0 = Clock::now();
        auto out = std::make_unique<MiniMaxMusic3FlowSamplerRuntime>(
            assets, execution, graph_arena_bytes, weight_context_bytes, storage_type);
        engine::debug::timing_log_scalar(
            "minimax_music3.load.flow_ms", engine::debug::elapsed_ms(t0, Clock::now()));
        return out;
    }

    std::unique_ptr<MiniMaxMusic3FlowSamplerRuntime> make_flow_unused() {
        return std::make_unique<MiniMaxMusic3FlowSamplerRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    std::unique_ptr<MiniMaxMusic3VocoderRuntime> make_vocoder() {
        return std::make_unique<MiniMaxMusic3VocoderRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    core::ExecutionContext & execution;
    std::unique_ptr<core::ExecutionContext> flow_execution;
    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    size_t graph_arena_bytes = 0;
    size_t weight_context_bytes = 0;
    assets::TensorStorageType storage_type = assets::TensorStorageType::Native;
    bool memory_saver = true;
    std::unique_ptr<MiniMaxMusic3ArRuntime> ar;
    std::unique_ptr<MiniMaxMusic3ConditionEncoderRuntime> condition;
    std::unique_ptr<MiniMaxMusic3FlowSamplerRuntime> flow;
    std::unique_ptr<MiniMaxMusic3VocoderRuntime> vocoder;
    sampling::TorchCudaSamplingPolicy sampling_policy;
};

MiniMaxMusic3PipelineRuntime::MiniMaxMusic3PipelineRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type,
    bool memory_saver)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          memory_saver)) {}

MiniMaxMusic3PipelineRuntime::~MiniMaxMusic3PipelineRuntime() = default;

runtime::AudioBuffer MiniMaxMusic3PipelineRuntime::generate(const MiniMaxMusic3Request & request) {
    return impl_->generate(request);
}

void MiniMaxMusic3PipelineRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
