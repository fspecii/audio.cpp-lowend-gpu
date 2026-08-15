# Bug report: graph-allocator reuses flagged input tensors' memory mid-compute (batch-1 MiniMax Music 3 graphs)

**Repo:** 0xShug0/audio.cpp, branch `preview/minimax-music-3` (commit cc82a5b)
**Severity:** silent output corruption / hard crash, backend-independent (reproduced on CPU **and** CUDA)

## Symptom

Building batch-1 variants of the MiniMax Music 3 depth-decoder and flow-transformer graphs
(for conditional-only classifier-free-guidance inference) produces:

* **depth decoder:** `GGML_ASSERT(i01 >= 0 && i01 < ne01)` in `ggml_compute_forward_get_rows`
  (CPU) / CUDA illegal memory access — the ids read by `get_rows` are float garbage
  (e.g. `-1071465402` = bit pattern of ≈ -2.53f)
* **flow transformer:** no crash, but output is corrupted: systematic ~0.7% error on first
  call and **consecutive identical calls differ by up to 60%** of value magnitude
  (`max|call1-call2| ≈ 6` on values ≈ 12), while the batch-2 graph is bit-deterministic

The stock batch-2 graphs are unaffected (bit-exact across runs).

## Root cause (established by probe)

Input tensors created in the graph context and allocated by `ggml_gallocr` get their memory
**reused for node outputs during the same compute**, despite `GGML_TENSOR_FLAG_INPUT` being set:

```
[depth-b1] wrote id=157376  readback=157376  data=0x...400  flags=1 op=0   <- write + readback OK
[get_rows-oob] src0=embed(200000 rows) src1=(reshaped) idx=-1071465402
               src1.data=0x...400  == vsrc.data                            <- same address, now floats
```

i.e. write → readback correct → graph executes → an earlier node's output has been placed
over the input's bytes by the allocator. `ggml_gallocr_free_node` protects only
`GGML_TENSOR_FLAG_OUTPUT`; inputs are freed after their last consumer and the batch-1 graph
topologies (ids tensors of 4-8 bytes, consumed through a reshape view) end up with a
reuser that clobbers the input before/while its consumer reads it. The batch-2 versions of
the same graphs dodge the collision by allocation-offset luck.

## Fix that works (verified)

Allocate graph inputs in a dedicated context/buffer instead of the graph allocator — the same
pattern already used for weights:

```cpp
// per graph:
ggml_init_params input_params{16384, nullptr, true};
input_ctx = ggml_init(input_params);
last_hidden  = make_tensor(input_ctx, ...);   // all ggml_set_input tensors
semantic_ids = make_tensor(input_ctx, ...);
positions    = make_tensor(input_ctx, ...);
input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx, backend);
// compute graph ctx references these tensors; gallocr never manages them
```

After this change:
* depth batch-1 vs batch-2 (duplicated rows): **max|diff| = 0**, repeat-determinism = 0, CPU & CUDA
* flow batch-1: deterministic across calls and across CUDA-graph replays
* full pipeline output bit-identical to the pre-change engine when the new paths are disabled

Patch available (applies to `src/community_models/minimax_music3/{depth_decoder,flow_transformer}.cpp`).

## Repro sketch

1. Parameterise `build_decode_graph` / the flow graph build by batch and build a batch-1 instance.
2. Write inputs, read them back (correct), compute → get_rows assert (CPU) or garbage output.
3. Move inputs to a dedicated buffer → bit-exact vs batch-2.

Context: found while adding conditional-only CFG paths for single-GPU (RTX 3060 12 GB)
MiniMax Music 3 inference; happy to share the full working branch.
