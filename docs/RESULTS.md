# Measured results — Qwen3.8-Flash-Next, 3× Radeon PRO V620, PP=3 (17–18 Sept 2026)

Harness: `scripts/perf3.py` (prompts cut at the token from one corpus, greedy, 200–300 generated tokens, client-side TTFT and throughput, single stream unless stated), `scripts/stab.py` (random sizes, 4-stream burst every 6th iteration, seed 1, output sanity classifier), `scripts/burst.py` (4 concurrent streams then the same prompt again), `scripts/cachetest3.py` (same prompt 3× on an idle server, `/metrics` prefix-cache counters). Power cap 160 W per card, undervolt −40 mV unless stated. "tok/s" = tokens per second.

> **Correction, 18 Sept 17:20 — read this before quoting any prefill number.** Until 18 Sept, `perf3.py` run with `--reps 1` started **every prompt size at corpus offset 0**, and so did the warm-up probe. With prefix caching **on**, each size therefore re-used the blocks of the smaller sizes sent before it, and every cache-on prefill figure at ≥ 16K in this file was inflated (16K: 1 863 published, 1 705 clean; 130K: **2 493 published, 1 933 clean**; the 261K "2 716" is unverified and should be ignored). Affected: the last row of §1 and §2, all of §2b (its *relative* verdicts hold, every line carried the same bias). Not affected: anything measured with the cache off (the §2 ladder, the opengfx1030 rows), all decode and concurrency numbers. §0 below is the clean re-measurement; the harness now uses one offset per size. The real prefill advantage of the combined configuration over opengfx1030 at ~130K is +6 % at 160 W (1 933 vs 1 829), not +36 %.

## 0. Current state — clean campaign, 18 Sept 16:00–17:35

One script (`scripts/host/campaign-1075.sh`), one boot, 13 runs, 0 GPU events: leapdragon 0915 + overlay + MoE HIP + cudagraphs + `DENSE_INT8`, partition 17,18,13, **VRAM 1 075 MHz**, −40 mV, the two power caps back to back. "Ref." = the same measurement at 1 000 MHz / 160 W earlier the same day (§3).

Decode, tok/s — multi-stream as "overlapped (wall)", 512-token prompts, 300 tokens per stream:

| | 1 075 MHz, 160 W | 1 075 MHz, 200 W | ref. 1 000 MHz, 160 W |
|---|---|---|---|
| no MTP, 1 stream | **50.9** | 51.1 | 48.8 |
| no MTP, 4 streams at once | 139.5 (114) | 139.5 (117) | 157 (127) † |
| no MTP, 8 streams at once (`--max-num-seqs 8`) | **249 (191)** | 253 (195) | 242 (184) |
| MTP k=2, bf16 drafter, 1 stream (4K / 16K) | 62.8 / 61.8 | 62.6 / 61.3 | 59.8 / 62.5 |
| MTP k=2, W4A16 drafter, 1 stream (4K / 16K) | **66.2 / 67.9** | 64.9 / 64.4 | 64.9 / 66.3 |
| MTP k=2, W4A16 drafter, 4 streams 0.7 s apart | **173 (125)** | 166 (126) | 160 (121) |
| MTP k=2, W4A16 drafter, 8 streams 0.7 s apart, `--max-num-seqs 8` | 112 (90) | 109 (88) | bf16 drafter: 91 (77) |

Prefill, tok/s — MTP k=2 with the W4A16 drafter, ctx 131K, prefix caching **off**, one prompt per size:

| | 4K | 16K | 65K | 130K |
|---|---|---|---|---|
| 1 075 MHz, 160 W | 1 152 | 1 705 | 1 936 | 1 933 |
| 1 075 MHz, 200 W | **1 306** | **1 943** | **2 221** | **2 254** |
| gain | +13.4 % | +14.0 % | +14.7 % | +16.6 % |

What holds: **200 W = +13–17 % prefill and nothing on decode** (decode is memory-bound, prefill sits at the cap); **1 075 MHz ≈ +4 % decode without MTP** (three samples at 50.7–50.9 vs 48.8); the quantised drafter ≈ +5–10 % single stream and +17 % wall at 8 streams. Single-run MTP figures move by ±5 % from run to run (acceptance varies with the text). † The 4-stream no-MTP figure reproduced at exactly 139.5 on both caps and we cannot explain the gap to the morning's 157 — open. Thermals during the 130K prefill at 200 W: junction ≤ 80 °C, memory ≤ 64 °C (passive cards, server airflow).

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
| | prefill (cache on — ≥ 16K inflated, see the correction above) | — | **1 078** | ~~1 863~~ | ~~1 843~~ | ~~1 989~~ | ~~2 493~~ | ~~2 716~~ § |
| same, W4A16 drafter, VRAM 1 075 MHz, **cache off** (§0) | prefill | — | **1 152** | **1 705** | — | **1 936** | **1 933** | — |

✱ measured but invalid (the streaming client sees nothing until the end with MTP above ~32K). ‡ streaming measurement on 100 tokens, consistent with 4K/16K, to be confirmed with server counters. § 262K server (KV 293K tokens = 1.12 full requests), prefix caching on during the measurement — may include partial hits; 1 859 at 131K and 1 313 at 200K on the same server.

Decode 48 vs 32 on the leapdragon lines: image 20260915 + `VLLM_RDNA_DENSE_INT8=1 VLLM_RDNA_DENSE_INT8_ONLY=1` (int8 shadows of the dense projections), not the MoE kernel.

## 2. The prefill ladder on leapdragon (PP3, cudagraphs, cache off, 4K / 16K)

| step | prefill 4K | prefill 16K | decode |
|---|---|---|---|
| image 0915 as shipped (default Triton MoE config under PP3) | 409 | 738 | 48 |
| + leapdragon's tuned int4 MoE config copied to `E=512` (`num_stages=1`) | 689 (+68 %) | 1 032 (+40 %) | 48.1 |
| + TunableOp lookup (rocBLAS rows in the image) + capture sizes 1…256 | 747 | 1 109 | 48.0 |
| + opengfx1030 `moe_gptq_gemm_rdna2` as a torch extension | **1 201** | **1 794** | 48.4 |
| + MTP k=2 (ctx 131K, cache off, clean re-measurement §0) | 1 152 | 1 705 | 62–68 |

## 2b. What else was tried on the combined configuration (18 Sept, 65K ctx, cache on, 4K / 16K / 32K / 60K prompts)

Absolute prefill values at ≥ 16K in this table are inflated by prefix-cache hits (see the correction at the top); every line carried the same bias, so the verdicts stand. The 200 W line was re-measured cleanly in §0: +13–17 %.

| change | prefill tok/s | decode tok/s | verdict |
|---|---|---|---|
| reference, 160 W cap | 1 089 / 1 879 / 2 029 / 2 544 | 58–65 | — |
| **200 W cap** (undervolt −40 mV kept) | 1 206 / 2 096 / 2 262 / 2 870 | 60–66 | **+11–13 % prefill** (compute-bound: cards sit at the cap with sclk throttled to 1.5–2.3 GHz), decode within noise |
| MTP k=3 | 1 075 / 1 841 / 2 003 / 2 511 | 56–67 | wash (2.96 tokens/step at 65 % acceptance), 8 % less KV |
| `NCCL_P2P_LEVEL=SYS` | 1 078 / 1 847 / 2 008 / 2 510 | 59–65 | −1–2 % (leapdragon's +8 % came from TP all-reduces; PP has none) |
| `--max-num-batched-tokens 4096` | 903 / 1 406 / 1 484 / 2 105 | 58–65 | −17–27 %, keep 2 048 |
| memory clock forced to the top DPM level (`manual`, `pp_dpm_mclk 3`) | 1 082 / 1 871 / 2 018 / 2 544 | 56–63 | identical: mclk already sits there during active phases (0.25 s sampling), the 96 MHz dips are idle gaps |
| 262K ctx, `--kv-cache-memory-bytes 4.5e9` | 1 862 (131K) / 1 247 (261K) | — | KV 377K tokens = 1.44 full requests, GPU0 holds |
| 262K ctx, no MTP, KV auto (0.95) | boot fails | — | profiler leaves 2.2 GiB < 2.21 needed — always pin the KV size |

## 2c. VRAM clock (18 Sept, afternoon)

GDDR6 DPM levels are 96 / 456 / 673 / 1 000 MHz and the stock driver exposes no `OD_MCLK` on the V620: the VBIOS PowerPlay table carries a complete OverDrive section (GFXCLK 500–2 650, UCLK 674–1 075 MHz) with every capability flag cleared. `patches/host/v620-amdgpu-powercap-odcaps-uclk1250.patch` sets the four flags in the **driver's copy** of the table for PCI subsystem `0x0e34` (the in-driver equivalent of the four ROM bytes [Tamalero/amd-v620-soft-unlock](https://github.com/Tamalero/amd-v620-soft-unlock) flips through QEMU `romfile=`; we run LXC on the host driver, so no romfile) and widens the driver-side UCLK ceiling to 1 250 MHz for exploration. Nothing is written to the card; it needs a module rebuild and a host reboot.

Tested with `memtest_vulkan` v0.5.0, one card at a time (`scripts/host/vram-clk-climb.sh`, `scripts/host/mtv.py`). Write throughput in GB/s:

| card (bus) | 1 000 | 1 075 | 1 100 | 1 125 | 1 150 |
|---|---|---|---|---|---|
| card0 (63:00) | 448 | **481** | 490 | 478 — throughput drops, no data error (EDR link retries) | — |
| card2 (46:00) | — | **479** | 487 | 499 | 508, still tracking the clock; not pushed further |
| card1 (43:00) | — | **476** | 476 | 490 | `ERROR_DEVICE_LOST` (gfx ring timeout) |

**Only 1 075 MHz is validated**: 5 min 30 per card, twice (before and after the incident below), 0 errors, +7.4 % memory throughput on card0 (the only card with a verified 1 000 MHz reference: the earlier reference passes on the other two had silently tested card0, see the pitfalls below), memory at 62–64 °C on these passive cards. Everything above 1 075 is a single 90-second screening pass, and the 90-second throughput figure is noisier than it looks (card1 read 476 at both 1 075 and 1 100, then 490 at 1 125). The SMU firmware does not clamp above the VBIOS ceiling — throughput follows the clock — and the three cards do not have the same margin.

Serving effect of 1 075 MHz, same configuration as §3: decode without MTP 48.8 → **50.7–50.9 tok/s (+4 %, three samples)**, with MTP k=2 and the quantised drafter 64.9 / 66.3 → 66–68, c=4 staggered 160 → 163–173 (full table in §0). Forcing the memory clock to the top DPM level (`manual`, `pp_dpm_mclk 3`) changes nothing: it already sits there during active phases. In PP=3 the stages run in series for each token, so a per-card setting (1 100 / 1 075 / 1 125) would buy well under 1 % — we run **1 075 on all three**, made persistent by `scripts/host/gpu-undervolt` (voltage offset and memory clock live in the same OverDrive table: both are written, then committed **once** per card).

**Incident, for anyone repeating this.** When card1 hung at 1 150 the driver's ring reset succeeded (`device wedged, but recovered through reset`). Our script then wrote an OverDrive setting to that freshly reset card — it only `break`-ed out of its climb loop and carried on — and six seconds later the SMU stopped answering (`SMU: No response msg_reg: 29`, sysfs silent, a process stuck in D state in `amdgpu_dpm_get_sclk`). There is no FLR on these cards: host reboot. The script in this repo now exits without any OverDrive write on a memtest error or any amdgpu ring-timeout / reset / wedged / SMU message, and `gpu-undervolt` reads the table with a timeout before writing. **After a GPU hang, write nothing to `pp_od_clk_voltage` until the host has rebooted.**

`memtest_vulkan` pitfalls here: it ignores the device choice on stdin, as a CLI argument and via `DRI_PRIME` (always tests the first card) — drive it through a pseudo-terminal (`mtv.py`) and check the `Bus=` it reports; concurrent instances fail with `Failed determining memory budget`; a backgrounded instance ignores `SIGINT`. Memory voltage: the OverDrive table sent to the SMU has a single voltage field (`VddGfxOffset`); `MemMvddVoltage[]` / `MemVddciVoltage[]` exist in the PowerPlay table but there is no memory-voltage sensor and we did not touch them.

## 3. Concurrency — decode only (18 Sept, 512-token prompts, 300 generated tokens per stream, 160 W)

`perf3 --conc N`: N streams started together; "overlapped decode" = Σ of each stream's own decode rate (the streams' decode windows overlap almost entirely with 512-token prompts); "wall" divides all generated tokens by the wall time including the serialised prefills.

| configuration | c=1 | c=4, all at once | c=4, arrivals 0.7 s apart | c=8, all at once | c=8, 0.7 s apart |
|---|---|---|---|---|---|
| **leapdragon + MoE HIP + cudagraphs, no MTP** | 48.8 | **157** (127) — 39 per stream | — | **242** (184) — 30 per stream | 207 (156) |
| leapdragon + MoE HIP + cudagraphs + MTP k=2 | 62.5 | 56 (51) — 14 per stream | **159** (116) — 40 per stream | 90 (77) | 91 (77) |

The MTP "collapse" at c=4 is a scheduling lock-step, not a kernel problem: requests that arrive at the same instant land in one batch and stay there, and with speculative decoding a request cannot be rescheduled until its draft tokens come back from the last stage, so only one batch is ever in flight and the three stages run serially (56 ≈ the single-stream 62). Without MTP the async scheduler keeps `pp_size` batches in flight with placeholder tokens. Evidence: `--max-num-seqs 2` at c=4 → 95 tok/s wall (two batches in flight, each stream at its single-stream 58), `--max-num-seqs 1` → 53 (fully serial), and arrivals 0.7 s apart with `--max-num-seqs 8` → 159 overlapped, the same as without MTP. Triton JIT during inference was ruled out (5 compilations over a whole run). At 8 streams MTP stays at ~90 whatever the arrival pattern — the verify batches grow and the drafter's fp16 MoE (Triton `fused_moe_kernel`, 12 % of last-stage GPU time in the c=4 profile) weighs — so the rule is **MTP k=2 up to ~4 streams, no MTP beyond**. For reference, leapdragon TP4+EP reports 64 → 127 tok/s at 12 streams and Minachist (3× RTX 3090, PP3) 60 → 155 at 4.

**Quantised MTP drafter (18 Sept, 10:30).** The checkpoint leaves the draft head's 512 experts in bf16 (`mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`, 4.9 GB), so under MTP they ran through the generic Triton `fused_moe_kernel`. `scripts/quant_mtp_experts.py` quantises them offline (RTN, symmetric int4, group 128, packed exactly like the main experts) into a derived checkpoint (1.38 GB shard, hard links for the rest, `quantization_config.ignore` narrowed to `re:^mtp\.(?!layers\.\d+\.mlp\.experts(\.|$)).*` — the drafter is built as `mtp.layers.48.*` and `get_moe_method` probes `experts.0.<proj>`), and the drafter experts then go through the same HIP kernel:

| MTP k=2, MoE HIP, cudagraphs | c=1 (4K / 16K) | acceptance | c=4, 0.7 s apart | c=8, 0.7 s apart (wall) |
|---|---|---|---|---|
| drafter experts bf16 (Triton) | 59.8 / 62.5 | 70 % (2.41) | 159 (116 wall) | 91 (77) |
| **drafter experts W4A16 (HIP)** | **64.9 / 66.3** | 75 % (2.50) | 160 (121 wall) | **~110 (88–90 wall)**, two samples |

No measurable acceptance loss (RTN error ~13 % relative on the weights), +4–7 % single stream, **+17 % wall / +23 % overlapped at 8 streams** (no-MTP at 8 streams, same arrival pattern: 156 wall — so beyond ~4 streams, no MTP). *Corrected 18 Sept:* this line first read "123 wall, +60 %"; that run had been launched with `--max-num-seqs 4` against 8 for the bf16 line. Like for like it is 88–90. The acceptance column is the server's rolling counter, indicative only.

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
