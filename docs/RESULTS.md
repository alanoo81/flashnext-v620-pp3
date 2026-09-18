# Measured results — Qwen3.8-Flash-Next, 3× Radeon PRO V620, PP=3 (17 Sept 2026)

Harness: `scripts/perf3.py` (prompts cut at the token from one corpus at distinct offsets, greedy, 200–300 generated tokens, client-side TTFT and throughput, single stream unless stated), `scripts/stab.py` (random sizes, 4-stream burst every 6th iteration, seed 1, output sanity classifier), `scripts/burst.py` (4 concurrent streams then the same prompt again), `scripts/cachetest3.py` (same prompt 3× on an idle server, `/metrics` prefix-cache counters). Power cap 160 W per card, undervolt −40 mV. "tok/s" = tokens per second.

## 1. Single stream, by context depth

| stack | mode | 1K | 4K | 16K | 32K | 64K | 128K | 262K |
|---|---|---|---|---|---|---|---|---|
| llama.cpp `v620-qsa` (-sm layer 15,16,17, KV q8_0) | decode | — | — | 28.8 | 28.0 | 27.1 | 26.3 (96K) | 22.4 |
| | prefill | — | — | 1 233 | 1 199 | 1 128 | 1 072 (96K) | 894 |
| leapdragon 0907/0915, cudagraphs, cache on | decode | 32.3 | 32.1 | 32.3 | 32.8 | 32.3 | 32.2 | 32.2 |
| | prefill | 200 | 409 | 738 | 1 169 | 1 349 | 779 | 804 |
| leapdragon, eager | decode | 14.7 | 14.7 | 14.8 | 14.5 | 15.1 | — | — |
| | prefill | 201 | 428 | 760 | 1 227 | 1 361 | — | — |
| opengfx1030 `50120e1` + sanitizer fix, V1 runner, eager, cache **off** | decode | 25.5 | 26.5 | 26.1 | 26.2 | 26.2 | 25.9 | 26.2 |
| | prefill | 515 | 1 010 | 1 568 | 1 719 | 1 798 | 1 829 | 1 827 |
| leapdragon + MTP k=2, eager, cache on | decode | — | 24.9 | 28.3 | — | — | — | — |
| | prefill | — | 516 | 681 | — | — | — | — |
| leapdragon + MTP k=1, cudagraphs, cache on (#54044) | decode | — | 51.1–52.2 | 52.1 | — | ✱ | ✱ | — |
| | prefill | — | 391–519 | 686 | — | 647 | — | — |
| leapdragon + MTP k=2, cudagraphs, cache on (#54044) | decode | — | 56.9 | 58.7–60.0 | — | 59.2 | — | — |
| | prefill | — | 517 | 605–680 | — | 743 | — | — |
| **leapdragon + MoE HIP** (this repo), cudagraphs, cache off, E=512 config, TunableOp | decode | — | 48.4 | 48.5 | — | — | — | — |
| | prefill | — | **1 201** | **1 794** | — | — | — | — |
| **leapdragon + MoE HIP + MTP k=2**, cudagraphs, cache on, 17,18,13, KV 3.5e9 | decode | — | **57.2** | **62.0** | 58.9 ‡ | 62.8 ‡ | 60.7 ‡ | 63.4 ‡ |
| | prefill | — | **1 078** | **1 863** | **1 843** | **1 989** | **2 493** | 2 716 § |

✱ measured but invalid (the streaming client sees nothing until the end with MTP above ~32K). ‡ streaming measurement on 100 tokens, consistent with 4K/16K, to be confirmed with server counters. § 262K server (KV 293K tokens = 1.12 full requests), prefix caching on during the measurement — may include partial hits; 1 859 at 131K and 1 313 at 200K on the same server.

Decode 48 vs 32 on the leapdragon lines: image 20260915 + `VLLM_RDNA_DENSE_INT8=1 VLLM_RDNA_DENSE_INT8_ONLY=1` (int8 shadows of the dense projections), not the MoE kernel.

## 2. The prefill ladder on leapdragon (PP3, cudagraphs, cache off, 4K / 16K)

| step | prefill 4K | prefill 16K | decode |
|---|---|---|---|
| image 0915 as shipped (default Triton MoE config under PP3) | 409 | 738 | 48 |
| + leapdragon's tuned int4 MoE config copied to `E=512` (`num_stages=1`) | 689 (+68 %) | 1 032 (+40 %) | 48.1 |
| + TunableOp lookup (rocBLAS rows in the image) + capture sizes 1…256 | 747 | 1 109 | 48.0 |
| + opengfx1030 `moe_gptq_gemm_rdna2` as a torch extension | **1 201** | **1 794** | 48.4 |
| + MTP k=2 + prefix caching (ctx 131K) | 1 078 | 1 863 | 57–63 |

## 2b. What else was tried on the combined configuration (18 Sept, 65K ctx, cache on, 4K / 16K / 32K / 60K prompts)

| change | prefill tok/s | decode tok/s | verdict |
|---|---|---|---|
| reference, 160 W cap | 1 089 / 1 879 / 2 029 / 2 544 | 58–65 | — |
| **200 W cap** (undervolt −40 mV kept) | 1 206 / 2 096 / 2 262 / 2 870 | 60–66 | **+11–13 % prefill** (compute-bound: cards sit at the cap with sclk throttled to 1.5–2.3 GHz), decode within noise |
| MTP k=3 | 1 075 / 1 841 / 2 003 / 2 511 | 56–67 | wash (2.96 tokens/step at 65 % acceptance), 8 % less KV |
| `NCCL_P2P_LEVEL=SYS` | 1 078 / 1 847 / 2 008 / 2 510 | 59–65 | −1–2 % (leapdragon's +8 % came from TP all-reduces; PP has none) |
| `--max-num-batched-tokens 4096` | 903 / 1 406 / 1 484 / 2 105 | 58–65 | −17–27 %, keep 2 048 |
| memory clock forced to 1 000 MHz (`manual`, `pp_dpm_mclk 3`) | 1 082 / 1 871 / 2 018 / 2 544 | 56–63 | identical: mclk already sits at 1 000 MHz during active phases (0.25 s sampling), the 96 MHz dips are idle gaps |
| 262K ctx, `--kv-cache-memory-bytes 4.5e9` | 1 862 (131K) / 1 247 (261K) | — | KV 377K tokens = 1.44 full requests, GPU0 holds |
| 262K ctx, no MTP, KV auto (0.95) | boot fails | — | profiler leaves 2.2 GiB < 2.21 needed — always pin the KV size |

Memory clock: GDDR6 DPM levels 96 / 456 / 673 / 1 000 MHz, no `OD_MCLK` in the OverDrive table of this VBIOS (only `OD_VDDGFX_OFFSET`). An in-driver unlock of the OverDrive capability flags (the same four bytes as [Tamalero/amd-v620-soft-unlock](https://github.com/Tamalero/amd-v620-soft-unlock) flips in the ROM, done in `sienna_cichlid_patch_pptable_quirk` for subsystem `0x0e34`) builds cleanly against the patched amdgpu; it would expose UCLK 674–1 075 MHz — not yet applied (needs a host reboot).

## 3. Concurrency — decode only (18 Sept, 512-token prompts, 300 generated tokens per stream, 160 W)

`perf3 --conc N`: N streams started together; "overlapped decode" = Σ of each stream's own decode rate (the streams' decode windows overlap almost entirely with 512-token prompts); "wall" divides all generated tokens by the wall time including the serialised prefills.

| configuration | c=1 | c=4, all at once | c=4, arrivals 0.7 s apart | c=8, all at once | c=8, 0.7 s apart |
|---|---|---|---|---|---|
| **leapdragon + MoE HIP + cudagraphs, no MTP** | 48.8 | **157** (127) — 39 per stream | — | **242** (184) — 30 per stream | 207 (156) |
| leapdragon + MoE HIP + cudagraphs + MTP k=2 | 62.5 | 56 (51) — 14 per stream | **159** (116) — 40 per stream | 90 (77) | 91 (77) |

The MTP "collapse" at c=4 is a scheduling lock-step, not a kernel problem: requests that arrive at the same instant land in one batch and stay there, and with speculative decoding a request cannot be rescheduled until its draft tokens come back from the last stage, so only one batch is ever in flight and the three stages run serially (56 ≈ the single-stream 62). Without MTP the async scheduler keeps `pp_size` batches in flight with placeholder tokens. Evidence: `--max-num-seqs 2` at c=4 → 95 tok/s wall (two batches in flight, each stream at its single-stream 58), `--max-num-seqs 1` → 53 (fully serial), and arrivals 0.7 s apart with `--max-num-seqs 8` → 159 overlapped, the same as without MTP. Triton JIT during inference was ruled out (5 compilations over a whole run). At 8 streams MTP stays at ~90 whatever the arrival pattern — the verify batches grow and the drafter's fp16 MoE (Triton `fused_moe_kernel`, 12 % of last-stage GPU time in the c=4 profile) weighs — so the rule is **MTP k=2 up to ~4 streams, no MTP beyond**. For reference, leapdragon TP4+EP reports 64 → 127 tok/s at 12 streams and Minachist (3× RTX 3090, PP3) 60 → 155 at 4.

Two knobs from Minachist's write-up checked on ROCm: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` fixes the KV auto-sizing (262K, KV auto: 363K tokens without MTP, 344K with k=2, where the default profiler refused to boot) — use it instead of pinning `--kv-cache-memory-bytes`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks the PLE offload worker's cross-process registration on ROCm (`hipErrorInvalidValue`) — do not set it.

## 3a. Older concurrency numbers (17 Sept, 4 streams, 4K prompts, 200 tokens each — prefill-contaminated, kept for the record)

| stack | aggregate tok/s | per stream | mean TTFT |
|---|---|---|---|
| leapdragon cudagraphs + MTP k=1 (131K) | 29.4 | 21.2 | 22.6 s |
| leapdragon cudagraphs | 28.0 | 21.0 | 16.6 s |
| leapdragon eager | 21.1 | 10.4 | 16.2 s |
| opengfx1030 eager (cache off) | 24.8 | 15.2 | 15.9 s |
| opengfx1030 V1 (cache on — corrupts) | 27.1 | 16.2 | 14.7 s |

The combined configuration's 4-stream bursts (2 808 / 3 783 / 4 776 / 713-token prompts) complete in 9.3 s at 65K and 14.2 s at 131K with all four outputs correct; not yet run through the c=4 throughput harness.

## 4. Stability (`stab.py`, random sizes 512–16K, burst every 6th iteration, seed 1)

| stack | duration | requests | corrupted (`!!!`) | notes |
|---|---|---|---|---|
| opengfx1030 + fix, cache **off** | 40 min | 225 | 0 | 5 "suspects" = the model continuing the corpus |
| opengfx1030 + fix, cache **on** (V2, V1, +PR #55506, `--max-num-seqs 2`) | — | — | **1 stream at iteration 12, deterministic** | second defect, cache must stay off |
| leapdragon cudagraphs, cache on, same seed | 20 min | 74 | 0 | |
| **leapdragon + MoE HIP + MTP k=2 + cache on, ctx 131K** | 30 min | 134 iterations / 196 outputs | 0 | 4 suspects (corpus continuations in bursts), 0 errors |

## 5. Prefix caching (same 9 645-token prompt sent 3× to an idle server)

| stack | 2nd send | 3rd send |
|---|---|---|
| leapdragon cudagraphs, no MTP | 97–99 % hit | 97–99 % |
| leapdragon + MTP k=2 (with or without the MoE HIP kernel) | 0 % | 34–91 % (83 % measured today) |
| opengfx1030 | 0 % | 0 % (and corrupts under concurrency) |

## 6. Where the prefill time went (torch profiler, one 4 336-token prefill, rank 0, same container)

| kernel | opengfx1030 (HIP MoE) | leapdragon 0915 (Triton MoE, default config) |
|---|---|---|
| fused MoE | `moe_gemm_q4_kernel_rdna2` **582 ms** | `fused_moe_kernel_gptq_awq` **3 849 ms** |
| everything else | within noise | within noise |

## 7. MTP directly on opengfx1030 (worktree on `50120e1`, PP3, eager, cache off, 4K / 16K)

| runner | MTP | decode tok/s | acceptance |
|---|---|---|---|
| V1 (default for Qwen4Exp) | none | 25.4 / 26.3 | — |
| V2 | none | 13.4 / 13.4 | — (PLE lookup serialised, see report §13) |
| V2 | k=1 | 18.3 / 18.8 | 87 % (1.87) |
| V2 | k=2 | 23.8 / 23.2 | 72.6 % (2.45) |
| V1 | k=1/2 | assert (async) / hang (sync) | — |
| V2 + cudagraphs | none | `hipErrorIllegalAddress` rank 2 | — |

## 8. VRAM (combined configuration, `--kv-cache-memory-bytes 3.5e9`)

| ctx | KV pool | full requests | VRAM used GPU0 / 1 / 2 |
|---|---|---|---|
| 65 536 | 219K tokens | 3.3 | — |
| 131 072 | 263K | 2.0 | 33.7 / 32.6 / 32.4 GB |
| 262 144 | 293K | 1.12 | 33.2 / 32.2 / 32.1 GB |

Weights 23.8 / 24.1 / 25.4 GiB per stage (17,18,13 layers; stage 0 also hosts the PLE connector, stage 2 the MTP drafter). opengfx1030 eager without MTP keeps 546K tokens of KV (two full 262K requests).
