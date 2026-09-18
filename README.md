# Qwen3.8-Flash-Next on 3× Radeon PRO V620 (gfx1030) — vLLM PP=3 working tree

**Temporary public repo, 17 Sept 2026.** A shared working base for the two RDNA2 vLLM forks this work builds on:

- [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen) (branch `rdna2/qwen38-flash-next`, image `ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b`) — the tree we serve from;
- [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) (branch `rdna_extras`, `50120e1`) — the tree whose fused W4A16 MoE HIP kernel we ported into the leapdragon tree, and whose Flash-Next decode-corruption root cause we found.

Nothing here is a fork of either repo: it is the **delta** (overlay files, patches, kernel extension, harness, reports) needed to reproduce the configuration below on three V620s in pipeline parallel, plus the full record of how we got there. Everything is Apache-2.0 like the trees it comes from; the HIP kernel and its Python method are opengfx1030's code, the serving tree is leapdragon's.

## The result (17–18 Sept)

PP=3 / TP=1 (nothing in this model divides by 3, so TP is impossible), fp16, AWQ-W4A16 checkpoint (`wtdcode`), int4 n-gram sidecar in host RAM, power cap 160 W per card.

| configuration (same harness, `scripts/perf3.py`) | prefill 4K / 16K / 64K / 130K tok/s | decode tok/s | context |
|---|---|---|---|
| leapdragon image 0915, cudagraphs (as shipped, PP3) | 409 / 738 / 1 349 / — | 32 (48 with `DENSE_INT8=1`) | 262K |
| opengfx1030 `50120e1` + our fix, eager, cache off | 1 010 / 1 568 / 1 798 / 1 829 | 25–26 | 262K ×2 |
| leapdragon + tuned int4 MoE config `E=512` + TunableOp | 747 / 1 109 / — / — | 48 | — |
| leapdragon + **opengfx1030 MoE HIP kernel** (this repo) | 1 201 / 1 794 / — / — | 48 | — |
| **leapdragon + MoE HIP + cudagraphs + MTP k=2** (W4A16 drafter, VRAM 1 075 MHz, 160 W) | **1 152 / 1 705 / 1 936 / 1 933** | **62–68** | **262K** (1 full request) |
| same at a **200 W** cap | **1 306 / 1 943 / 2 221 / 2 254** | 62–68 | 262K |

> **Correction (18 Sept).** This table first showed 1 078 / 1 863 / 1 989 / 2 493 for the combined line. Those prefill figures were taken with prefix caching on by a harness that started every prompt size at the same corpus offset, so the ≥ 16K values included cache hits. The line above is the clean re-measurement (cache off, one prompt per size). Details and the full current-state table: [`docs/RESULTS.md`](docs/RESULTS.md) §0.

Multi-stream, decode only (512-token prompts, 18 Sept): **without MTP 48.8 → 157 tok/s at 4 streams → 242 at 8**; with MTP k=2 62.5 → 159 at 4 (staggered arrivals; simultaneous arrivals lock-step into one batch and give 56) → 91 at 8; with the MTP drafter's experts quantised to W4A16 (`scripts/quant_mtp_experts.py`, same HIP kernel) 65–68 single stream, 160–173 at 4, ~90 wall at 8 (+17 % over the bf16 drafter; an earlier "123" compared two different `--max-num-seqs`) — so MTP k=2 (quantised drafter) up to ~4 streams, plain cudagraphs for more (249 at 8). A 200 W power cap adds +13–17 % prefill (decode unchanged); VRAM at 1 075 MHz (driver-side OverDrive unlock, `patches/host/`) adds +4 % decode — the only memory clock validated on all three cards, see RESULTS §2c before going higher. See [`docs/RESULTS.md`](docs/RESULTS.md) §2b–3 for what else was tried (k=3, P2P level, batched tokens, memory clock) and why 262K is the ceiling.

The combined line ran 30 minutes of random-size/burst traffic at 131K context (134 iterations, 0 errors, 0 corrupted outputs), 4-stream bursts clean, and a 262K server (KV pool 293K tokens at `--kv-cache-memory-bytes 3.5e9`, VRAM 33.2 / 32.2 / 32.1 GB). Decode above 32K is measured on the streaming client and should be confirmed with server counters; no clean prefill figure exists yet at 261K (the earlier 2 716 tok/s included prefix-cache hits). Full tables: [`docs/RESULTS.md`](docs/RESULTS.md).

## What is in here

```
overlay/            files mounted over /app/vllm in the leapdragon image (vllm-pp3.sh does the mounts)
                    PP3 patches, MTP-under-PP relay (#46994) and #54044 backports, MoE HIP glue, E=512 MoE config
rdna2-moe/          opengfx1030's moe_q_gemm_rdna2.cu + headers, torch-extension binding, setup.py, build.sh
patches/leapdragon/ the same overlay as one unified diff against the 20260915-g1bdbbef4b image
patches/rdna_extras/ 16 git patches on opengfx1030 50120e1: PP3 support + MTP fixes (V2 runner) + diagnostics
scripts/            launchers (container + native TheRock venv), harness (perf3/stab/burst/cachetest3/sweep), test drivers, quant_mtp_experts.py (W4A16 MTP drafter)
docs/               JOURNEY.md (every step, 15–17 Sept), RESULTS.md, the opengfx1030 report (§0–13), the llama.cpp PP3 guide
prompts/            the exact prompts of the corruption repro (4 336 / 1 614 tokens) and the GDN sanitizer patch
```

## Run it (leapdragon image + this overlay)

Host: Proxmox 9.2 / kernel 7.0, `amdgpu.ras_enable=0` (ECC off → 32 752 MiB usable per card), no GIM driver. Unprivileged LXC with `/dev/kfd` + the three `renderD*`/`card*` passed through, 112 GB RAM, Docker inside. Model: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` (+ leapdragon's int4 PLE sidecar, `ples_int4`).

```bash
# 1. build the MoE HIP extension inside the leapdragon image (36 s)
docker run --rm -v $PWD/rdna2-moe:/build -w /build --entrypoint bash \
  ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b /build/build.sh
cp rdna2-moe/v620_moe_rdna2.so overlay/vllm/v620_moe_rdna2.so     # the launcher mounts it at /app/vllm/vllm/

# 2. serve — scripts/vllm-pp3.sh mounts every overlay/vllm/**/*.py|json|so over /app/vllm
MOE_HIP=1 TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" \
IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image \
DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=131072 \
EXTRA='--kv-cache-memory-bytes 3500000000 --speculative-config {"method":"mtp","num_speculative_tokens":2}' \
scripts/vllm-pp3.sh start
```

Knobs (all in `scripts/vllm-pp3.sh`): `MOE_HIP=1` opengfx1030 MoE kernel (default off → leapdragon's Triton path); `TUNEOP=1` rocBLAS TunableOp lookup-only with the rows shipped in the image (`tunableop/rocblas-9847aecc4bf8`); `CG_SIZES` piecewise CUDA-graph capture sizes for prefill batches (leapdragon §8e); `PART` = `VLLM_PP_LAYER_PARTITION` (17,18,13 with the MTP drafter on the last stage; 18,17,13 without); `EAGER=1`, `NOPC="--no-enable-prefix-caching"`, `P2P=` (`NCCL_P2P_LEVEL`, PHB here: no shared PCIe switch, PXB disables P2P).

Edit `MODEL`, `PLE`, `CACHE` at the top of the launcher for your paths. The launcher expects the overlay at `/root/vllm-leap-img` — set `OVERLAY=` or symlink.

## What the overlay changes (leapdragon tree)

1. **PP=3 on a Flash-Next model** (15 Sept): lift the two PP guards, keep the PLE-offload connector on the PLE-layer rank only, ignore `hyper_connection_mixer.*` on non-last stages, offload worker ignores `VLLM_PP_LAYER_PARTITION`.
2. **MTP under PP** (15–17 Sept): drafter embed/lm_head as CPU placeholders (VRAM), draft head embeds on the last stage, padded sampled-token broadcast (warmup deadlock), then the real fix: **draft-token relay** from the last stage to the other ranks — recipe 0009 / vLLM [#46994](https://github.com/vllm-project/vllm/pull/46994), backported from opengfx1030 `22bb2e8` (`broadcast_draft`, `vllm/v1/worker/gpu/pp_utils.py`).
3. **MTP + cudagraphs + prefix caching**: backport of vLLM [#54044](https://github.com/vllm-project/vllm/pull/54044) (reset cached Mamba `align` metadata on profiling teardown) in `gpu/cudagraph_utils.py` — without it the first real request after capture fails.
4. **Prefill** (17 Sept evening):
   - leapdragon's tuned int4 Triton MoE config is keyed `E=128,N=640,…` (experts ÷ EP4) and is never loaded under PP3 without EP; an `E=512` copy alone is +68 % / +40 % prefill (409 → 689 at 4K, 738 → 1 032 at 16K);
   - TunableOp lookup + capture sizes 1…256: +8 %;
   - **opengfx1030's `moe_gptq_gemm_rdna2`** built as a standalone torch extension (own op namespace `_v620_rdna2`), `CompressedTensorsWNA16RDNA2MoEMethod` (byte-identical to the RDNA3 method already in the leapdragon tree, op relocated, `create_weights` supplies `intermediate_size_full`), a gfx10x branch in `rocm_moe_rdna.py`, and the RDNA2 class name in the two exact-name allow-lists of `routed_experts.py`: 747 → **1 201** at 4K, 1 109 → **1 794** at 16K, decode unchanged.

`patches/leapdragon/overlay-vs-image-20260915-g1bdbbef4b.diff` is the whole overlay as one diff. It also carries the opt-in, currently non-functional QSA host-KV offload port (`VLLM_QSA_KV_OFFLOAD=1`, see JOURNEY 18 Sept) — inert unless the variable is set.

## What the rdna_extras patches do (opengfx1030 tree)

`patches/rdna_extras/0001–0006`: the same PP3 support. `0007–0009`: diagnostics (Triton A/B gates, `V620_DUMP` layer dumps) used to find the corruption root cause. `0010–0016`: MTP on that tree — it works, but only on the V2 model runner (the relay lives in `vllm/v1/worker/gpu/`), and the V2 runner decodes at half speed there because its PLE connector receives GPU input ids and serialises the CPU n-gram lookup behind every forward (13.4 vs 25.4 tok/s; profile in the report §13). V2 + cudagraphs dies with `hipErrorIllegalAddress` on rank 2. So on opengfx1030 the fast path stays V1 eager without MTP.

The **decode corruption** (garbage from the first decode token on prompts ≥ ~4K, `!!!` = NaN logits) was root-caused on 17 Sept 00:30 to two one-shot `conv_state.zero_() / ssm_state.zero_()` "sanitizers" in `qwen_gdn_linear_attn.py` that wipe the paged GDN state cache at the first decode; the maintainer upstreamed the fix (`388a61b`). A second, separate defect remains on that tree: prefix caching + ≥3 concurrent requests corrupts one stream deterministically (`mamba` `align` cache mode; leapdragon on the same seed/sequence is clean), and its prefix cache never hits — so cache off is free there. Report: [`docs/opengfx1030-report.md`](docs/opengfx1030-report.md), prompts in `prompts/`.

## The journey

Day by day, with every dead end: [`docs/JOURNEY.md`](docs/JOURNEY.md). Short version: PP3 in leapdragon on 15 Sept (32 tok/s, prefill 400–1 350) → opengfx1030 corrupt from the first decode token, root cause found and upstreamed on 17 Sept → MTP under PP made to work with the #46994 relay and #54044 → leapdragon's own prefill levers (MoE config, TunableOp, capture sizes) applied to PP3 → opengfx1030's MoE kernel ported → one configuration that has both trees' strengths.

## Credits

- **leapdragon** — the serving tree, the PLE int4 sidecar, the RDNA2 platform work, the MTP recipe (0009) and two days of prefill measurements we reused (`docs/rdna2/CHANGES.md` §8d–8e).
- **opengfx1030** — the fused W4A16 MoE HIP kernel and its Python method (ported unchanged), the draft-token relay (`22bb2e8`), the fast GDN/conv HIP paths, and fast turnaround on the sanitizer fix.
- **Karl0007** — vLLM PR #55506 (mamba spec-decode block tables) tested here (does not fix the align corruption on this tree).

Hardware: Threadripper PRO 3975WX, 128 GB, 3× Radeon PRO V620 32 GB (no shared PCIe switch), TheRock ROCm 10.0 host venv for the native tree, ROCm 7.14.1 in the leapdragon image.
