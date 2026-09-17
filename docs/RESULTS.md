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

## 3. Concurrency (4 streams, 4K prompts, 200 tokens each)

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
