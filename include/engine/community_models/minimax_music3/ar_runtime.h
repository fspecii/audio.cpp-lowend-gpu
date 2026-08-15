#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/depth_decoder.h"
#include "engine/community_models/minimax_music3/global_lm.h"
#include "engine/community_models/minimax_music3/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3ArRuntime {
public:
    MiniMaxMusic3ArRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiniMaxMusic3ArRuntime();

    // on_frame (optional): called after each emitted frame with the number of complete
    // frames so far and a pointer to that frame's hidden block (condition_layers x hidden).
    using FrameCallback = std::function<void(int64_t frames_done, const float * frame_data, int64_t frame_values)>;

    std::vector<float> generate_frame_hiddens(
        const MiniMaxMusic3Request & request,
        int64_t target_frames,
        uint64_t & rng_offset_blocks,
        const FrameCallback & on_frame = {});

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
