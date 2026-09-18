# The journey — Qwen3.8-Flash-Next on 3× V620, 15–17 Sept 2026

Everything below was measured on the same box (Threadripper PRO 3975WX, 128 GB, 3× Radeon PRO V620 32 GB, Proxmox 9.2 host, unprivileged LXC, power cap 160 W / −40 mV) with the same model (`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`, leapdragon int4 PLE sidecar). Times are local. Commit ids refer to `overlay-git-log.txt` (leapdragon overlay) and `patches/rdna_extras/LOG.txt` (opengfx1030 tree).

## Why PP=3 at all

Flash-Next (`qwen4exp`): 36 GDN + 12 QSA layers, 512 experts, hidden 2 560, 2 KV heads of 256, a 51 B-parameter n-gram (PLE) table, MTP draft head. Nothing divides by 3 — not the heads, not the experts, not `moe_intermediate` 640 with group size 128 — so TP=3 is impossible and EP needs a divisor of 512. Three cards means pipeline parallel, one stage per card, which neither fork had run: leapdragon serves TP=4 + EP, opengfx1030 TP=4.

## 15 Sept — PP3 works on the leapdragon tree

- Image `ghcr.io/leapdragon/vllm-rdna2-qwen:latest` (20260907), tree at `/app/vllm`, overlay of patched files mounted over it.
- Four changes to get a PP=3 boot: lift two PP guards (Flash-Next asserted `pp_size == 1`), create the PLE-offload connector only on the rank that holds the PLE layer (stage 0), ignore `hyper_connection_mixer.*` weights on non-last stages, make the offload worker ignore `VLLM_PP_LAYER_PARTITION` (it inherited the env and tried to split its own 1-layer model).
- Result: **32.3 tok/s decode, 848 tok/s prefill at 4K, 29.8–30.3 GiB per card**, PLE table in host RAM. Greedy output not bit-deterministic across restarts (software non-determinism, later confirmed on all trees).
- MTP under PP: **dead end that day**. Memory solved (drafter embed/lm_head as CPU placeholders, partition 17,18,13, `VLLM_ROCM_MOE_PADDING=0`), warmup deadlock solved (padded sampled-token broadcast), but the other ranks never learned which draft tokens the last stage proposed — every step ran with garbage drafts. V1-runner attempt abandoned (`79a8e65`).
- Same day on `opengfx1030/vllm-rdna` `f663686` with the same PP patches: boots, prefill fast, **output corrupt from the first decode token** on prompts ≥ ~4K (1 614-token prompt fine, 4 336 garbage). First report sent to the maintainer.

## 16 Sept — the corruption is not going away by itself

- Maintainer says fixes were pushed: re-tested at `8960a3b` (20 commits later): **byte-for-byte the same garbage**. Report v2 with a simple English repro (`docs/opengfx1030-report.md` §0–7): setup, launch, the exact prompts, what was already ruled out (P2P level PXB vs PHB — no shared PCIe switch here so PXB disables P2P; both reproduce), what could not be tested.
- `c350fa2` routes gfx1030 `n ≤ 5` GEMV to `wvSplitK`, compiled only for GFX9/GFX11+ → device-side assert at the first decode on every gfx1030 build. Workaround `VLLM_RDNA_DENSE_GEMV=1` (§0).
- Maintainer asks whether we really use *their* model/drafter: yes — and reproduced with **their** checkpoint too (`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`, needed `V620_MOE_TRITON=1` + a missing `w2_zp` argument, §8).
- Evening: the user asks for the real cause ("aide la communauté à avancer"). Layer-by-layer dumps of both trees (`V620_DUMP`), same input, same weights: the first divergence is the GDN recurrent state handed to the first decode step. KV blocks are fine; the state is zero.

## 17 Sept, 00:30 — root cause

`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` on `rdna_extras` has two one-shot "sanitizers" (`_rdna2_cache_sanitized`, `_rdna2_ssm_sanitized`) that call `conv_state.zero_()` / `ssm_state.zero_()` on the whole paged cache the first time the decode path runs — after the prefill has already written the state. Any request whose prefill finishes before the first decode of the process loses its state; short prompts survive because their state is rebuilt from the conv window. Dropping the sanitizers fixes it (`prompts/opengfx1030-fix-gdn-sanitizer.patch`); the maintainer upstreamed `388a61b` the same morning. Report §9.

## 17 Sept, morning — campaign on the fixed tree, and a second defect

- Common harness written (`scripts/perf3.py`, `stab.py`, `burst.py`, `cachetest3.py`): prompts cut at the token, distinct corpus offsets (no cache hits between runs), greedy, 200–300 tokens, client TTFT/throughput; stability = random sizes + a 4-stream burst every 6th iteration, seed 1.
- opengfx1030 `50120e1` + fix, PP3, eager, cache off: **prefill 515 / 1 010 / 1 568 / 1 719 / 1 798 / 1 829 / 1 827 tok/s at 1K…262K, decode 25.5–26.5 flat, 40 min stable (225 requests, 0 corruption)**. Two 262K requests fit (KV 546K tokens).
- With prefix caching on (the default): a 4-stream burst corrupts one stream **deterministically** (`!!!` = NaN logits → token 0), at the same iteration on the V2 runner, the V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`, maintainer's advice), with vLLM PR #55506 cherry-picked, and with `--max-num-seqs 2`. Its prefix cache also never hits (0 % on identical resent prompts). leapdragon on the same seed/sequence with cache on: 74 requests, 0 corruption. So on opengfx1030 cache off is free and mandatory. Report §10.
- The maintainer's production serve config adapted to PP3 (V1 runner, PIECEWISE graphs, block size 16, EP, KV 5e9): dies in graph capture on `wvSplitK`; with `DENSE_GEMV=1` the first request takes 93 s and the second hangs a worker. Report §10.

## 17 Sept, 11:00–12:30 — MTP under PP, for real

- The user points at vLLM #46994 (draft-token relay under PP) as referenced by GeorgeMA-Strong `7da0de5` and opengfx1030 `22bb2e8` (`broadcast_draft`, recipe 0009). Backported into the leapdragon overlay (`38a2c9f`): **MTP works under PP3, eager**: k=1 21.2–21.7 tok/s (acceptance 89.5 %), k=2 24.9–28.3 (2.62 tokens/step).
- With cudagraphs the first real request after capture failed until vLLM #54044 (reset cached Mamba `align` metadata on profiling teardown, merged 30 Aug, missing from leapdragon's 29 Aug base) was backported (`5697c39`): **MTP k=1 51–52 tok/s at 131K, k=2 57–60 tok/s at 65K** (KV under-allocation with MTP+graphs fixed by `--kv-cache-memory-bytes 3.5e9`). Prefix-cache hit rate with MTP: 0 % on the 2nd send, 34–91 % on the 3rd (state checkpoints only at block boundaries). Report §11. Leapdragon cudagraphs pushed to 262K without MTP (KV 344K tokens).
- llama.cpp PP3 guide for the Discord written (`docs/llamacpp-pp3-guide.md`): our `v620-qsa` fork, `-sm layer -ts 15,16,17`, 28.8 tok/s at 16K, 22.4 at 262K.

## 17 Sept, 15:00 — why opengfx1030 prefills 2× faster

Torch profiler on one 4 336-token prefill, rank 0, both trees in the same container: **opengfx1030's HIP W4A16 MoE kernel 582 ms vs leapdragon's Triton fused MoE 3 849 ms (6.6×)**; everything else within noise. Report §12 (with an erratum: our earlier "Triton MoE A/B" gate only changed a log line).

## 17 Sept, 16:00–18:00 — option A: MTP directly on opengfx1030 (not a shortcut)

Worktree on `50120e1` + our fixes. Six things stood in the way (`patches/rdna_extras/0011–0016`): the drafter's `embed_tokens` stays a CPU placeholder under PP (`_maybe_share_embeddings` is gated on `pp world_size == 1`); `llm_base_proposer.py` bypasses `Qwen4ExpMTPProposer._get_hidden_size()` so the drafter buffer is 2 560 wide instead of 4 × 2 560; the RDNA2 `causal_conv1d` HIP kernels require `state_len == width − 1` while spec decode widens the conv state; `image_token_index`; and the relay only exists in the V2 runner. Then:

- V2 runner + MTP: **works** — k=1 18.3 tok/s (acc. 87 %), k=2 23.8 (72.6 %, 2.45), bursts clean.
- But V2 without MTP decodes at **13.4 tok/s vs 25.4–26.3 on V1**, same worktree. Profile: `ple_offload/connector.py:_launch_inline` 87 ms median per step on V2, absent on V1. V2 hands the PLE connector GPU input ids (`input_buffers.input_ids`), V1 hands `input_ids.cpu`; with GPU ids the ~30 ms CPU n-gram lookup can only start after the previous forward and is serialised (42 + 30 ≈ 75 ms). `PLE_OFFLOAD_DOORBELL=0` changes nothing. This is exactly leapdragon's eager regime (14.7 tok/s); cudagraphs are what save leapdragon.
- V1 + MTP: async → `PP+async expects sampled_token_ids [num_reqs, 1]`; `--no-async-scheduling` → hangs after the first prefill.
- V2 + cudagraphs on this tree: `hipErrorIllegalAddress` on rank 2 at the first request.

Conclusion: each tree had half — opengfx1030 the prefill (MoE HIP), leapdragon the decode (graphs + MTP). Option B it is.

## 17 Sept, 18:00–19:20 — option B: the MoE kernel goes to leapdragon

- Read leapdragon's `RESULTS.md` / `CHANGES.md` first (user's request): their prefill levers are a tuned int4 Triton MoE config (`num_stages=1`, the "gfx1030 num_stages rule"), TunableOp rows for rocBLAS, and larger piecewise capture sizes. The config file is keyed `E=128` (experts ÷ EP4) and is **never loaded under PP3 without EP**: an `E=512` copy → 409 → 689 tok/s at 4K, 738 → 1 032 at 16K (`1cf8bb2`). TunableOp lookup (rows for the image's rocBLAS `9847aecc4bf8` are shipped) + capture sizes 1…256 → 747 / 1 109.
- The kernel: `csrc/rocm/moe_q_gemm_rdna2.cu` + `q_gemm_rdna2_common.cuh` + `qdq_4_rdna2.cuh` compile unchanged as a standalone torch extension inside the leapdragon image (hipcc ROCm 7.14, 36 s). The Python method `CompressedTensorsWNA16RDNA2MoEMethod` is byte-identical to `compressed_tensors_moe_wna16_rdna3.py` already in the leapdragon tree (`rdna3 → rdna2`), so the glue is: op relocated to `torch.ops._v620_rdna2`, a gfx10x branch in `rocm_moe_rdna.py` (`e3718ac`), `create_weights` supplying `intermediate_size_full` (`cfcde9d`), and the RDNA2 class name in `routed_experts.py`'s two exact-name allow-lists — without which the compressed-tensors weights load untransposed (`80 vs 2560`) (`392ab6b`).
- **leapdragon + MoE HIP, cudagraphs: 1 201 / 1 794 tok/s prefill at 4K / 16K, decode 48 unchanged**, bursts clean.
- **+ MTP k=2 + prefix caching (17,18,13, KV 3.5e9): 1 078 / 1 863 prefill, 57–62 decode at 4K/16K; at 131K: 1 843 / 1 989 / 2 493 prefill at 32K/65K/130K; 30 min stability (134 iterations, 196 ok, 4 "suspects" = the model continuing the corpus in bursts, 0 errors); 262K server boots (KV 293K tokens, 1.12 full requests, VRAM 33.2/32.2/32.1 GB), 1 859 / 1 313 / 2 716 prefill at 131K/200K/261K.**

## 18 Sept, morning — everything else on the combined configuration

- Audit of leapdragon's `CHANGES.md` / `RESULTS.md` against PP3: T43 (fp16 GEMV), T45/T45b (int8 shadows, `DENSE_INT8=1 …_ONLY=1`), T46 (launch fusion), §8a/8c (PLE protocol, doorbell, prefault), §8d (MoE config, TunableOp, QSA tiles), §8e (capture sizes) all applied; T44 (one-shot all-reduce) is TP-only; the PCIe stability kernel line was never needed here.
- Power: **200 W cap = +11–13 % prefill**, decode unchanged. MTP k=3, `NCCL_P2P_LEVEL=SYS`, `--max-num-batched-tokens 4096`, forced memory clock: none helps (details in RESULTS §2b).
- 262K: `--kv-cache-memory-bytes 4.5e9` gives 377K tokens (1.44 requests); the model's native window is 262 144 (no rope scaling in the checkpoint), beyond that would be YaRN through `--hf-overrides`, unvalidated on the AMD path — not attempted.
- **Concurrency, measured properly** (decode only, 512-token prompts): without MTP 48.8 → 157 (c=4) → 242 tok/s (c=8); with MTP k=2 62.5 → 53 → 90. The earlier "c=4 = 32 tok/s" was the harness feeding 4K prompts whose serialised prefills sat inside the measurement window. Rule: MTP for one user, no MTP from two streams.
- **Minachist's QSA K/V host offload** (`Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, `vllm-patch/`): UVA works on ROCm inside the leapdragon image (sparse 2 048-row gather over PCIe at 16 GB/s, Triton reads host memory at 21.6 GB/s), and the patch transposes to the AMD path (3 of 4 hunks apply, `kv_cache_utils` by hand; overlay `ff0134d`, opt-in `VLLM_QSA_KV_OFFLOAD=1`, no MTP). It does not boot: leapdragon's "CSA+linear" KV grouping stores each request's GDN state inside a **per-layer** main-KV page, and with 2-byte GPU slots that page is 24 KB against a 3.2 MB state (`kv_cache_utils.get_kv_cache_groups`). Minachist's vLLM 0.29 base sizes the mamba page against the sum over layers. Next step if wanted: give the mamba states their own pool in leapdragon's grouping — a few hours, worth ~4× the KV capacity (multi-262K), not speed.
- **VRAM 1 075 MHz** (Tamalero): in-driver equivalent written and built (`v620-odcaps-unlock.patch` on the patched amdgpu), pending a host reboot.

## What is still open

1. Re-measure the 261K prefill with prefix caching off (possible partial hits) and decode > 32K with server counters (the streaming client mis-measures MTP at long context).
2. Propose to leapdragon: the `E=512` config (or key the config by global experts under PP), the MoE HIP extension, the #46994 relay and #54044 backports.
3. Tell opengfx1030: the deterministic `!!!` under prefix caching + concurrency (repro in the report), the never-hitting prefix cache, the V2-runner PLE serialisation (§13), the V2 + graphs crash.
4. `moe_align_block_size` is called per layer per step on the HIP path — the decode budget on leapdragon's 4-card TP is 15.6 ms/step; ours under PP3 is ~20 ms with graphs. The PP send/recv and the PLE round-trip are the next things to look at.
