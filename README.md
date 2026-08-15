# audio.cpp — low-end GPU fork (MiniMax Music 3 on a single RTX 3060)

A modified fork of [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) tuned to run
**MiniMax Music 3 fast on low-end consumer GPUs**.

Upstream MiniMax Music 3 documents **two CUDA GPUs**. This fork runs the full 11 B-parameter
stack on **one RTX 3060 (12 GB)** and generates a 20-second song in **43 seconds** — **3.2x
faster** than the first working single-GPU configuration.

**Built by [Ambsd](https://x.com/Ambsd)** · [github.com/fspecii](https://github.com/fspecii)

Upstream's own README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).

---

## Performance (RTX 3060 12 GB, PCIe 3.0 x4)

| | first working build | this fork |
|---|---|---|
| 20 s song | ~136 s | **43.1 s** (3.2x) |
| 75 s song | ~339 s | **171 s** — 2.28x realtime (2.0x) |
| 2:24 song | — | **361 s** |
| Peak VRAM | 8.2 GiB | ~9 GiB of 11.9 |

Where the 43.1 s goes, and how close each stage is to the hardware limit:

| stage | time | headroom left |
|---|---|---|
| Flow matching | 17.8 s | ~79% of the card's bf16 FLOP ceiling — near tapped out |
| Autoregressive (LM 8.7 + depth 7.2) | 17.4 s | 71% / 55% of memory bandwidth |
| Model load | 4.3 s | 6.4 GB over a PCIe 3.0 **x4** link |
| Vocoder | 2.9 s | GPU F32 convnet |

**This is close to the practical floor for this architecture on a 3060** without writing new
CUDA kernels. Both heavy stages sit within ~20-25% of their hardware limits.

---

## Quick start

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then run with the tuned preset (weights: `audio-cpp/MiniMax-Music3-GGUF`):

```bash
export MM3_SOLVER=ab2          # 2nd-order flow solver: 15 steps ≈ 30 Euler steps
export MM3_AR_CFG_FRAMES=50    # full CFG for 2 s, conditional-only after
export MM3_NO_ACT_CAST=1       # skip bf16 parity casts
export MM3_DEPTH_B1=1          # batch-1 depth decoder after the guidance switch
export GGML_CUDA_MMVQ_MAX=4    # route ncols 5-8 matmuls to MMQ

./build/bin/audiocpp_cli --task gen --family minimax_music3 \
  --model /path/to/MiniMax-Music3-GGUF --backend cuda --metrics \
  --session-option "minimax_music3.rvq_depth_decoder_gguf=rvq_depth_decoder_q4_k.gguf" \
  --text "<structured caption>" \
  --request-option "lyrics=[verse]
Your lyrics here" \
  --request-option duration_sec=60 \
  --request-option num_inference_steps=15 \
  --out song.wav
```

> **Song length comes from your lyrics, not `duration_sec`** — that value is only an upper bound.
> A four-line lyric yields ~30 s even if you ask for five minutes. Write a full lyric sheet.

---

## What this fork changes

All flags default to upstream behaviour unless noted, so the stock paths are unaffected.

### Speed
| change | effect | how |
|---|---|---|
| **Compact `lm_head`** *(default on)* | **−5.6 s** | The AR stage can only ever sample 16385 of the checkpoint's 200000 logit rows, but a full q4_0 dequant + cuBLAS GEMM over the whole vocabulary ran every frame (11.7 ms). The head is sliced to the reachable rows at load time — mathematically identical output |
| **AB2 flow solver** (`MM3_SOLVER=ab2`) | **−8.5 s** | Adams-Bashforth 2nd order reuses the previous velocity (`x += dt·(1.5vₙ − 0.5vₙ₋₁)`) at the same one model evaluation per step, so 15 steps match 30 Euler steps (measured: low band −0.6 dB, corr 0.961) |
| **MMVQ→MMQ routing** (`GGML_CUDA_MMVQ_MAX`) | **−1.6 s** | ggml picks MMVQ whenever `ncols ≤ 8`, but its multi-column path ran at ~45% of memory bandwidth on the depth decoder. Making the cutoff runtime-tunable lets ncols 5-8 reach the tensor-core MMQ path |
| **Fused QKV in the depth decoder** *(default on)* | **−3.2 s** | q/k/v are identically shaped and row-major, so raw byte concatenation *is* the packed layout. Three GEMMs become one per layer per codebook step |
| **Skip bf16 parity casts** (`MM3_NO_ACT_CAST`) | **−2.2 s** | Removes 213,846 kernel launches per song that exist only for bit-parity with the reference |
| **AR guidance truncation** (`MM3_AR_CFG_FRAMES`) | LM 36.5 → 28 ms/frame | Full classifier-free guidance for the first N frames, then conditional-only on the non-batched decode path |
| **Flow CFG truncation** (`MM3_CFG_END`, default 0.40) | −4.4 s | Past ~40% of the flow trajectory the unconditional branch no longer changes the result (corr 0.9997) |
| **Cache-growth bucketing** *(default on)* | −9.5 s on a 75 s song | Fixed-shape CUDA graphs attend over every *allocated* KV slot, so allocating for the whole song taxes every frame. The decode graph now grows in 512-slot steps |

### Correctness / capability
| fix | why it matters |
|---|---|
| **Graph inputs in dedicated backend buffers** | The graph allocator was recycling live, `ggml_set_input`-flagged tensors mid-compute — silent output corruption in the flow transformer and an out-of-bounds crash in the depth decoder. See [`BUGREPORT.md`](BUGREPORT.md) |
| **`ggml_set_output` on consumed-and-returned nodes** | `ggml_build_forward_expand` alone does not protect a node from the allocator |
| **F16 KV cache** (was F32) | Halves the decode-graph allocation; raised the maximum song length ~50% on a 12 GB card |
| **Padded `conv_transpose_1d` lowering** | `ggml_conv_transpose_1d` asserts `p0 == 0`; the CPU backend could not run this model's vocoder at all without it |
| **Per-part AR instrumentation** | `ar.lm_ms`, `ar.depth_ms`, `ar.sample_ms`, `load.ar_ms` TIMING keys — how every number above was measured |

---

## Things that did **not** work

Measured and rejected, so you can skip them:

| attempt | result |
|---|---|
| Larger chunk hop (150 vs 100) | 5 s faster, **broke the music** — cut words, repeated phrases. Every spectral metric passed it; only listening caught it |
| Fused depth graph (7 steps → 1, in-graph argmax) | Built and working — **exactly zero gain**. The stage is bandwidth-bound, not launch-bound |
| Prompt-lookup speculative decoding | Dead end: music semantic tokens repeat ~3.6% of the time, unlike text |
| AR↔flow pipelining on one GPU | Net negative — the stages contend for the same memory system |
| CPU vocoding in parallel with the GPU | 27 s/chunk against a 9 s budget |
| F16 vocoder | **15x slower** — the conv path re-casts weights every forward |
| int8 / packed GEMMs on the 8 B LM | 0.35–0.86x. Packing helps only where GEMMs are small enough to be launch-bound |
| q4_0 depth decoder | 1.6% faster, vocal-band energy 87% → 61%. Rejected |
| 3rd-order (AB3) solver | Worse than AB2 — the overlap-blend breaks the smoothness it needs |

**Rule learned the hard way: spectral metrics can *reject* a change, never *approve* one.**
Band energies, RMS and silence detection all passed a render that was audibly chopped to pieces.
Anything touching chunking, stitching, context or the AR trajectory is judged by ear on a
**full-length** song — short clips hide seam damage.

---

## Why this model can't be ACE-Step fast

ACE-Step 1.5 reaches 27x realtime; MiniMax Music 3 is ~2x *slower* than realtime. The gap is
architectural. For 60 s of audio ACE-Step runs roughly **30** sequential model passes — its LM
writes a plan once, then a DiT denoises the whole song in parallel. Music 3 runs about
**12,900**: an 8 B language model producing one semantic token per 40 ms of audio, each frame
strictly dependent on the last, plus seven residual codebooks per frame. No kernel closes a
430x algorithmic gap. Music 3 spends that cost on vocal quality.

---

## Licence

Apache 2.0, inherited from upstream `0xShug0/audio.cpp` (Copyright 2026 ShugoAI LLC).
Modifications listed above are released under the same licence.
