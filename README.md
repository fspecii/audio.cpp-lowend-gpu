<p align="center">
  <h1 align="center">🎛️ audio.cpp — Low-End GPU Fork</h1>
</p>

<p align="center">
  <strong>MiniMax Music 3 on a single RTX 3060. 20-second song in 43 seconds.</strong><br>
  <em>A tuned fork of <a href="https://github.com/0xShug0/audio.cpp">audio.cpp</a> — upstream documents <b>two</b> CUDA GPUs, this runs the full 11B stack on <b>one</b> 12&nbsp;GB card</em>
</p>

<p align="center">
  <img alt="speedup" src="https://img.shields.io/badge/speedup-3.2x-brightgreen?style=for-the-badge">
  <img alt="gpu" src="https://img.shields.io/badge/GPU-RTX%203060%2012GB-76b900?style=for-the-badge&logo=nvidia&logoColor=white">
  <img alt="vram" src="https://img.shields.io/badge/peak%20VRAM-9%20GiB-blue?style=for-the-badge">
  <img alt="python" src="https://img.shields.io/badge/python-0%25-yellow?style=for-the-badge">
  <img alt="licence" src="https://img.shields.io/badge/licence-Apache%202.0-lightgrey?style=for-the-badge">
</p>

<p align="center">
  Built by <a href="https://x.com/AmbsdOP"><b>@AmbsdOP</b></a> · <a href="https://github.com/fspecii">github.com/fspecii</a>
</p>

---

## 🎧 Listen — made on the 3060 with this fork

<p align="center">
  <a href="https://fspecii.github.io/audio.cpp-lowend-gpu/">
    <img alt="Play the samples" src="https://img.shields.io/badge/▶%20PLAY%20THE%20SAMPLES-in%20your%20browser-22c55e?style=for-the-badge">
  </a>
</p>

*(GitHub strips `<audio>` and `<video>` tags from READMEs, so the player lives on the
project page — one click, no download.)*

**“Thirty Sixty”** — 2:24, generated in **361 s**. A trap track *about* this optimisation work,
written and rendered end-to-end on one RTX 3060:

> *Ambsd on the build, running on a thirty sixty*
> *Three times faster and the music still pretty*
> *Forty three seconds where it used to be a city*
> *Low end GPU but the output not gritty*

**Trap demo** — 2:00, generated in **293 s**. 140 BPM half-time, F minor.

Direct files: [`ambsd-3060-trap.mp3`](docs/ambsd-3060-trap.mp3) · [`trap-demo.mp3`](docs/trap-demo.mp3)

---

## ⚡ Performance

Measured on **RTX 3060 12 GB**, PCIe 3.0 **x4**, driver 570 / CUDA 12.8.

| | first working build | **this fork** |
|---|---|---|
| 20-second song | ~136 s | **43.1 s** — 3.2x |
| 75-second song | ~339 s | **171 s** — 2.28x realtime |
| 2:24 song | — | **361 s** |
| Peak VRAM | 8.2 GiB | ~9 GiB of 11.9 |

**Where the 43 seconds goes** — and how close each stage is to the silicon:

| stage | time | headroom left |
|---|---|---|
| 🌊 Flow matching | 17.8 s | ~79% of the card's bf16 FLOP ceiling — near tapped out |
| 🔁 Autoregressive (LM 8.7 + depth 7.2) | 17.4 s | 71% / 55% of memory bandwidth |
| 📦 Model load | 4.3 s | 6.4 GB over a PCIe 3.0 x4 link |
| 🔊 Vocoder | 2.9 s | GPU F32 convnet |

> This is close to the **practical floor** for this architecture on a 3060 without writing new CUDA kernels. Both heavy stages sit within ~20-25% of their hardware limits.

---

## 🚀 Quick Start

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Weights: [`audio-cpp/MiniMax-Music3-GGUF`](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF)

```bash
export MM3_SOLVER=ab2          # 2nd-order flow solver — 15 steps ≈ 30 Euler steps
export MM3_AR_CFG_FRAMES=50    # full guidance for 2 s, conditional-only after
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

> **⚠️ Song length comes from your lyrics, not `duration_sec`** — that value is only an upper bound. A four-line lyric yields ~30 s even if you ask for five minutes. Write a full lyric sheet.

---

## 🔧 What this fork changes

Defaults preserve upstream behaviour except where noted — stock paths are unaffected.

### Speed

| change | gain | how |
|---|---|---|
| **Compact `lm_head`** | **−5.6 s** | The AR stage can only ever sample **16,385 of 200,000** logit rows, yet a full q4_0 dequant + cuBLAS GEMM over the whole vocabulary ran **every frame** (11.7 ms). Sliced to the reachable rows at load time — mathematically identical output |
| **AB2 flow solver** | **−8.5 s** | 2nd-order Adams-Bashforth reuses the previous velocity at the *same* one model eval per step, so **15 steps match 30 Euler steps** (low band −0.6 dB, corr 0.961) |
| **MMVQ→MMQ routing** | **−1.6 s** | ggml picks MMVQ whenever `ncols ≤ 8`, but its multi-column path ran at **~45% of memory bandwidth**. A runtime-tunable cutoff lets ncols 5-8 reach the tensor-core MMQ path |
| **Fused QKV** (depth decoder) | **−3.2 s** | q/k/v are identically shaped and row-major, so raw byte concatenation *is* the packed layout |
| **Skip bf16 parity casts** | **−2.2 s** | Removes **213,846 kernel launches** per song that exist only for bit-parity with the reference |
| **Cache-growth bucketing** | **−9.5 s** *(75 s song)* | Fixed-shape CUDA graphs attend over every *allocated* KV slot, so sizing for a whole song taxed every frame |
| **Guidance truncation** (AR + flow) | LM 36.5 → 28 ms/frame | Past a point the unconditional branch stops changing the result (flow: corr 0.9997) |

### Correctness & capability

| fix | why it matters |
|---|---|
| **Graph inputs in dedicated buffers** | The graph allocator was recycling live `ggml_set_input` tensors mid-compute — silent output corruption in the flow transformer, out-of-bounds crash in the depth decoder. See [`BUGREPORT.md`](BUGREPORT.md) |
| **`ggml_set_output` on returned nodes** | `ggml_build_forward_expand` alone does **not** protect a node from the allocator |
| **F16 KV cache** (was F32) | Halves the decode-graph allocation — **~50% longer songs** on a 12 GB card |
| **Padded `conv_transpose_1d`** | `ggml_conv_transpose_1d` asserts `p0 == 0`; the CPU backend couldn't run this vocoder at all |
| **Per-part AR instrumentation** | `ar.lm_ms`, `ar.depth_ms`, `load.ar_ms` — how every number here was measured |

---

## 🚫 Things that did **not** work

Measured and rejected, so you can skip them:

| attempt | result |
|---|---|
| Larger chunk hop (150 vs 100) | 5 s faster, **broke the music** — cut words, repeated phrases. Every spectral metric passed it; only listening caught it |
| Fused depth graph (7 steps → 1, in-graph argmax) | Built and working — **exactly zero gain**. The stage is bandwidth-bound, not launch-bound |
| Prompt-lookup speculative decoding | Dead: music tokens repeat ~3.6% of the time, unlike text |
| AR↔flow pipelining on one GPU | Net negative — the stages contend for the same memory system |
| CPU vocoding in parallel | 27 s/chunk against a 9 s budget |
| F16 vocoder | **15x slower** — the conv path re-casts weights every forward |
| int8 / packed GEMMs on the 8B LM | 0.35–0.86x. Packing helps only where GEMMs are small enough to be launch-bound |
| q4_0 depth decoder | 1.6% faster, vocal-band energy 87% → 61%. Rejected |
| 3rd-order (AB3) solver | Worse than AB2 — the overlap-blend breaks the smoothness it needs |

> **The rule that came out of this: spectral metrics can *reject* a change, never *approve* one.** Band energies, RMS and silence detection all passed a render that was audibly chopped to pieces. Anything touching chunking, stitching, context or the AR trajectory gets judged **by ear, on a full-length song** — short clips hide seam damage.

---

## 🧠 Why this model can't be ACE-Step fast

ACE-Step 1.5 reaches **27x realtime**; MiniMax Music 3 is ~2x *slower* than realtime. The gap is architectural, not implementation quality.

For 60 seconds of audio:

| | sequential model passes |
|---|---|
| ACE-Step 1.5 | **~30** — the LM writes a plan once, then a DiT denoises the whole song in parallel |
| MiniMax Music 3 | **~12,900** — an 8B LM emitting one semantic token per 40 ms, each frame dependent on the last, plus 7 residual codebooks per frame |

No kernel closes a **430x** algorithmic gap. Music 3 spends that cost on vocal quality.

---

## 📜 Licence

Apache 2.0, inherited from upstream [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) (Copyright 2026 ShugoAI LLC). Modifications above released under the same licence. Upstream's original README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).

<p align="center">
  <sub>Built by <a href="https://x.com/AmbsdOP">@AmbsdOP</a> · <a href="https://github.com/fspecii">github.com/fspecii</a></sub>
</p>
