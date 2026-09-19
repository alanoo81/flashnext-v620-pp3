# Flash-Next on gfx1030: wrong output on a fresh server for ~4.3K-token prompts (rdna_extras @ 8960a3b)

**UPDATE 17 Sept: root cause found and fixed — see section 9.** Follow-up to the 2026-09-15 report. Re-tested 16 Sept on `rdna_extras` @ `8960a3b` (2026-09-16 18:18): **the corruption is unchanged**, same output byte-for-byte as at `f663686`.

## 0. Blocker first: `c350fa2` crashes every gfx1030 build at the first decode step

`c350fa2` routes gfx1030 decode GEMMs (n ≤ 5 rows) to `ops.wvSplitK`, but in `csrc/rocm/skinny_gemms.cu` the real kernel is compiled only under `__HIP__GFX9__ || __HIP__GFX1X__`, and `__HIP__GFX1X__` is defined only for `__GFX11__ || __GFX12__`. A `VLLM_GPU_ARCHES=gfx1030` build gets the `#else` stub:

```
csrc/rocm/skinny_gemms.hip:583: void wvSplitK_hf_sml_(...) [scalar_t = __half, THRDS = 64, YTILE = 4, WvPrGrp = 16, A_CHUNK = 8, UNRL = 1, N = 4]: Device-side assertion `false' failed.
```

The engine dies during startup (first decode of the warmup). Workaround for everything below: `VLLM_RDNA_DENSE_GEMV=1` (restores the previous `gemv_f16_rdna2` path).

## 1. Setup

- Hardware: 3× Radeon PRO V620 32 GB (gfx1030), Threadripper PRO 3975WX, 128 GB RAM. Power cap 160 W, undervolt −40 mV (also reproduced at 0 mV).
- OS: Proxmox VE 9.2 (kernel 7.0.14), Ubuntu 24.04 unprivileged LXC with `/dev/kfd` + `/dev/dri` passed through.
- Two independent software stacks, same result:
  - **Container**: `ghcr.io/leapdragon/vllm-rdna2-qwen:latest` (ROCm 7.14.1, PyTorch 2.12.0, Python 3.12) with the `rdna_extras` tree bind-mounted and `pip install -e . --no-build-isolation --no-deps`.
  - **Native**: TheRock stable 10.0 wheels (`rocm[libraries,devel,device-gfx1030]`, `torch[device-gfx1030]==2.12.0+rocm10.0.0`), `PYTORCH_ROCM_ARCH=gfx1030`, `pip install --no-build-isolation -e .`.
- Weights (Hugging Face checkpoints, downloaded fresh on 15 Sept with the `hf` CLI, sizes verified against the hub): backbone `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125b` (only revision, 2026-08-27) — shards 2–5, `model_mtp.safetensors`, config and tokenizer as published; `model-00001-of-00005.safetensors` (the 102 GB BF16 PLE n-gram table) not downloaded and its 128 `layers.1.ple…ngram_embedding.shard_N` index entries removed, the PLE comes from `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` `ples_int4` @ `a0fa93f2` (2026-09-14, 129 files, `group16_int4_fp16scale_lownibblefirst`) via `VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_QUANT_DIR=<ples_int4>`. This is the pair your `docs/rdna2/qwen4_exp_hip_path.md` and `flash-next-ple-ipc.md` describe, and the same files serve the leapdragon control tree. **No drafter**: no `--speculative-config`, MTP never enabled in any of these runs. Your `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` checkpoint, with the sidecar and with the original BF16 PLE from its shard 16, is covered in section 8: same bug.
- Parallelism: **PP=3, TP=1, no EP** (the only layout that fits this model on 3 cards; TP=3 is impossible for it). This needs the small PP patch series from the 15 Sept report (lift the two PP guards, PLE offload connector only on the rank owning the PLE layer, ignore `hyper_connection_mixer.*` on non-last stages, offload worker ignores `VLLM_PP_LAYER_PARTITION`). The identical patches on leapdragon `rdna2/qwen38-flash-next` in the same container serve correctly, so they are not the cause.

## 2. Launch

```bash
export VLLM_RDNA_DENSE_GEMV=1            # workaround for c350fa2 (section 0)
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_QUANT_DIR=/ples_int4 VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
export VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HSA_NO_SCRATCH_RECLAIM=1 NCCL_P2P_LEVEL=PHB   # reproduced with both PHB (P2P/IPC) and PXB (SHM fallback here, no shared PCIe switch) PYTORCH_ROCM_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0,1,2
# container only: HSA_OVERRIDE_GFX_VERSION=10.3.0

python -m vllm.entrypoints.openai.api_server --model /model --served-model-name flash-next \
  --dtype float16 --tensor-parallel-size 1 --pipeline-parallel-size 3 --distributed-executor-backend mp \
  --max-model-len 32768 --gpu-memory-utilization 0.95 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --skip-mm-profiling --enable-prefix-caching --enforce-eager \
  --host 127.0.0.1 --port 8086
```

Log shows the default gfx1030 paths: `RDNA2 HIP GDN prefill kernel`, `'RDNA2_W4A16' WNA16 MoE backend`, `GDN decode kernel: triton`.

## 3. Trigger

**On a fresh server, make the first request a prompt of about 4 300 tokens** (2 full 2 048-token prefill chunks + a short tail of ~240 tokens), greedy, 100+ output tokens:

The exact failing prompts are attached: `prompt-4336.txt` (16 000 chars of llama.cpp documentation — the DeepSeek-V4 ROCm harness README — plus the instruction line; 4 336 tokens with this tokenizer) and `prompt-5108.txt` (same corpus, other offset; 5 108 tokens). Both already end with the instruction line, so send them as-is:

```bash
python3 - <<'EOF'
import json, urllib.request
prompt = open("prompt-4336.txt", encoding="utf-8").read()      # or prompt-5108.txt
body = {"model": "flash-next", "prompt": prompt, "max_tokens": 150, "temperature": 0, "seed": 42, "ignore_eos": True}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8086/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}))
print(json.load(r)["choices"][0]["text"])
EOF
```

The prompt asks for a French summary so that a correct answer is obvious at a glance. Any long technical text of the same length shape (2 full 2 048-token chunks + a short tail) is likely to do; the corpus is public llama.cpp documentation. Files: `/root/opengfx1030-repro-prompts/`.

**Observed** (identical at `f663686` and `8960a3b`, container and native):

```
        assert "gpt4" in processors
        # `gpt4` is always slow
        # but we always prefer the official tokenizer
        # which is now deprecated at the same time
```

Unrelated to the prompt. Other ~5K prompts from other offsets of the same corpus give `</parameter></function></tool_call> user <system-reminder>` or an empty `<think></think>`.

**Expected** (same prompt on leapdragon `rdna2/qwen38-flash-next`, same container, same PP patches): `The user is asking me to provide a detailed and structured summary…` then a correct French summary.

## 4. Same server, other prompt lengths (fresh server each time, first request)

| prompt tokens | chunks (2 048) | output |
|---|---|---|
| 1 614 | 1 | correct French summary |
| 3 187 | 2 | coherent |
| **4 336** | 2 + 240 | **garbage** (above) |
| 5 108 (other offset) | 2 + 1 012 | garbage (`</parameter></function>…`) |
| 17 925 | 8 + 1 541 | coherent (thinking) |

Decode 26.3 t/s, prefill 1 460 t/s at 18K, KV 433K tokens at 0.95.

## 5. The KV blocks are fine, the state handed to decode is not

- Fresh server → 4 336-token prompt → garbage.
- Then send a 17 925-token prompt whose prefix covers those 4 336 tokens → coherent.
- Send the very same 4 336-token prompt again (blocks now come from the cache) → **coherent**.
- Conversely, a 17 925-token request that reuses the blocks computed by the garbage run is coherent too.

So the cached KV/state blocks are correct; what is wrong is the recurrent state (GDN/conv, or QSA/PLE tail) produced when the prompt is prefilled from scratch and handed to decode.

## 6. Already ruled out (each A/B on the 4 336-token case, 1 614 correct every time)

| variable | tested | result |
|---|---|---|
| RDNA2 HIP GDN prefill | forced `Triton/FLA GDN prefill` (`VLLM_GDN_HIP_PREFILL=0`) | still garbage |
| native W4A16 MoE kernel | forced `'TRITON' WNA16 MoE backend` | still garbage |
| RDNA2 FA | `VLLM_USE_RDNA2_FA=0` | still garbage |
| model runner | V1 (`VLLM_USE_V2_MODEL_RUNNER=0`) and V2 | still garbage |
| chunked prefill | `--max-num-batched-tokens 8192` (single chunk) | still garbage |
| prefix caching | `--no-enable-prefix-caching` | still garbage |
| ROCm / PyTorch stack | container ROCm 7.14.1 vs native TheRock 10.0 | same |
| RCCL transport (PP send/recv) | `NCCL_P2P_LEVEL=PHB` (P2P/IPC) vs `PXB` (SHM fallback on this topology) | still garbage |
| GPU voltage | −40 mV vs 0 mV | same |
| `empty` → `zeros` fixes in `143e2bf`, `3092d63`, `1ff7359` | included in `8960a3b` | no change |
| control tree | leapdragon `rdna2/qwen38-flash-next`, same container, same PP patches | **correct** |

Every HIP-specific gfx1030 kernel path is excluded. What is left is Python-side in the `rdna_extras` Qwen4Exp port vs leapdragon's: chunked-prefill / state hand-off, PLE tail, QSA changes.

## 7. What we could not test

Only PP=3/TP=1 is possible here (the model does not fit TP=1 or PP=2 on 32 GB cards, TP=3 is not divisible). If you can, on 4× V620 TP=4, run the section 3 probe **as the first request of a fresh server** with a ~4 300-token prompt; the `!!!` corruption you documented in `docs/rdna2/flash-next-prefix-cache-corruption.md` appears after a load, ours appears on the very first request, so they may or may not be the same bug.

Compiled mode (no `--enforce-eager`) at `f663686` failed with an Inductor stride assertion on PP stages 1/2 (`assert_size_stride(arg3_1, (s26, 4), (336, 1))`); **still fails at `8960a3b`** with the same assertion (`RuntimeError: Worker failed with error 'expected size 8==8, stride 4==336 at dim=0'`, right after `torch.compile` on the PP stages). Not seen with the leapdragon tree in the same container.

## 8. Your checkpoint: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (added 16 Sept evening)

To answer the "are you sure you use our model" question we downloaded the checkpoint from `V620-CANDIDATE.md` (all 17 shards + `model_extra_tensors.safetensors`, sizes verified against the hub) and ran the same PP3 eager probe. No drafter here either.

Two more rdna_extras bugs on this path (both hit before the first request, native gfx1030 build, `8960a3b`):

- `convert_to_wna16_moe_kernel_format` (`fused_moe/oracle/int_wna16.py:1737`) has no branch for the backend it just selected: `ValueError: Unsupported wna16 MoE backend: RDNA2_W4A16`. The AutoRound checkpoint goes through `moe_wna16` (GPTQ packing), and the RDNA2 W4A16 backend is only wired for compressed-tensors. Worked around by forcing the Triton WNA16 backend.
- `try_rocm_moe_skinny_decode()` requires the keyword-only `w2_zp` (`rocm_moe_skinny.py:139`) but three call sites pass only `w1_zp`: `fused_moe.py:1636`, `experts/triton_moe.py:245` and `:661` → `TypeError: missing 1 required keyword-only argument: 'w2_zp'` at warmup. Fixed locally by passing `w2_zp=quant_config.w2_zp` (commit `8249722` on our branch).

| backbone | PLE table | 1 614 tk | 4 336 tk (fresh server, first request) |
|---|---|---|---|
| wtdcode AWQ-W4A16 | `ples_int4` sidecar | correct | **garbage** (`assert "gpt4" in processors …`) |
| **Intel AutoRound** | `ples_int4` sidecar | correct | **garbage** (`<think> assert isinstance(src, str) / assert src.endswith('.js') …`) |
| **Intel AutoRound** | **BF16 from shard 16** (`VLLM_PLE_DISK_OFFLOAD_DIR`, no sidecar) | empty `<think></think>` then nothing in 150 tokens (suspect); 3 187 tk correct structured summary | **garbage** (`<think> assert isinstance(encoder.added_tokens, list) / assert 'encoder' is always added after the encoder …`) |

So neither the backbone quantization nor the PLE table (int4 sidecar vs the original BF16 table) is the variable; the bug follows the `rdna_extras` code. Decode 24–26 t/s, prefill 139 t/s at 4.3K on the Triton MoE path (first request, cold).

## 9. ROOT CAUSE FOUND (17 Sept, 00:30) — one-shot `ssm_state.zero_()` in the gfx1030 GDN decode path wipes the first request's state

**The bug is not in any kernel.** In `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `_forward_core_decode_non_spec()` runs, once per process, right before the first `torch.ops._rocm_C.gdn_decode_rdna2` call:

```python
if not self._rdna2_ssm_sanitized:
    ssm_state.zero_()          # the WHOLE paged GDN state cache
    self._rdna2_ssm_sanitized = True
```

(`_bind_decode_state_arenas()` has the same pattern for `conv_state` + `ssm_state` behind `_rdna2_cache_sanitized`, active on the cudagraph/arena path.) The comment says the paged caches are `torch.empty` on RDNA2 and the first read would see uncommitted pages. But `allocate_kv_cache` (`vllm/v1/worker/utils.py:404`) already allocates the cache with `torch.zeros`, so nothing is uncommitted — and the `zero_()` fires at the **first decode step of the first request**, i.e. immediately after that request's prefill wrote its final recurrent state into the cache. The first request therefore decodes from a zero GDN state. Short prompts survive because the 12 QSA attention layers still see the whole prompt; beyond ~4K tokens the model depends on the GDN memory and collapses (unrelated `assert` loops, or "I don't see any question, you only provided a title" once the HIP conv1d is switched off). Every later request is fine because the flag is set — which is exactly why "the same prompt again", "after a longer request" and "after `/reset_prefix_cache`" all looked like cache effects.

**Fixed upstream:** `388a61b` (17 Sept 00:57, "remove the one-shot ssm/conv state zero_() wipes that corrupted the first request") removes both sanitizers, same change as the attached patch; maintainer-validated on TP=4 (4 189-token fresh-server prompt, was constant `duct`). Re-validated here on upstream `50120e1` (17 Sept 08:11) in PP=3 eager, native build: fresh-server 4 336 → correct, then 1 614 / 17 925 / 4 336 correct.

How it was pinned down:
- Fresh server → 4 336 garbage → same prompt again (even after `POST /reset_prefix_cache`) → correct. So a one-shot effect, not the prefix cache.
- Layer-by-layer dump of both trees on the same prompt (forward hooks on every decoder layer and its `ple` / `linear_attn` / `self_attn` / `mlp`): end-of-prefill outputs agree (token 100 to 1e-3, last token within MoE-routing noise); at the **first decode step, `layers.0.linear_attn` already diverges (cos 0.55) with an identical input** → the state read at decode is wrong, at layer 0.
- Kernel A/Bs, each on a fresh server: `VLLM_CAUSAL_CONV1D_RDNA2_FWD=0/UPDATE=0`, `--linear-backend rdna_hybrid`, `_C` RMSNorms, `VLLM_GDN_HIP_PREFILL=0` + Triton conv1d together → still wrong on the first request (the failure mode changes, the first-request-only property does not).
- `VLLM_GDN_DECODE_RDNA2=0` (skips the branch that contains the sanitizer) → **first request correct**.
- Fix applied (both sanitizers removed, HIP decode kernel kept): two fresh servers, first requests 4 336 and 5 108, then 1 614 / 3 187 / 17 925 / 4 336 / 5 108 → **all correct**. Patch attached: `opengfx1030-fix-gdn-sanitizer.patch` (commit `b649be8` on our branch, on top of `8960a3b`; the sanitizer is still present at `b78006a`).

Side findings from the same investigation, for the tracker:
- `V620_GDN_TRITON=1`-style checks are not enough: only `VLLM_GDN_HIP_PREFILL=0` actually switches the GDN prefill chain (the backend label in the log is resolved separately from the dispatch in `_gdn_prefill_dispatch_available()`), so the 15 Sept "Triton A/B" only changed the log line. (Moot now, and `cd1231f` defaults prefill to Triton anyway.)
- With the fix in, the eager PP3 configuration serves Flash-Next correctly on 3× V620 at 26 t/s decode / ~1 460 t/s prefill@18K. Compiled mode still fails with the Inductor stride assertion (section 7) — separate issue.
- The `</parameter></function></tool_call>` prefixes we reported for the 5 108-token prompt are the model imitating the llama.cpp tool-call documentation in that part of the corpus (leapdragon produces the same prefix and then a correct summary); not corruption. Only the fresh-server first-request failure was real.

## 10. 17 Sept, afternoon — prefix caching + concurrency corruption, and the production config on PP3

**Prefix-cache corruption (`!!!`, i.e. all-NaN logits → token 0) is deterministic on this tree and specific to it.** Sequence: fresh server, `--enable-prefix-caching` (mamba cache mode `align`), then our stability loop `stab.py --seed 1` (random 512-65K prompts, one burst of 4 concurrent requests every 6th iteration). The burst at iteration 12 always poisons the state pool; from then on 60-100 % of requests return `!!!…`. Tested at `50120e1` + PP patches, PP=3/TP=1, eager:

| config | result |
|---|---|
| V2 runner (default) | corrupt at it. 12 |
| `VLLM_USE_V2_MODEL_RUNNER=0` (your recommendation) | corrupt at it. 12 |
| + vLLM PR #55506 commit 1 (request-slot block tables) | corrupt at it. 12 |
| `--max-num-seqs 2` | corrupt (it. 13-14, right after the burst) |
| `--no-enable-prefix-caching` | **clean**: 40 min, 225 requests incl. 25 bursts, 0 corruption |
| leapdragon `rdna2/qwen38-flash-next` (image 20260907), same PP patches, cache ON, cudagraphs, **same seed/sequence** | **clean**: 20 min, 74 requests, 0 corruption |

So the align-mode state copies are broken in `rdna_extras` (V1 and V2 alike) but not in the leapdragon tree at its base (vLLM main 6cddad414). Upstream references that look related: #55766 (Qwen3.5/3.8 GDN NaN after a prefix-cache hit, single request), #53142 / #53798 / #54076 (mamba block size used for align seeding / chunk splitting), tracking #26201.

**Your production config (`scripts/serve_gfx1030_flashnext.sh`, 17 Sept) adapted to PP=3/TP=1** (everything else verbatim: V1 runner, `PIECEWISE`, `--block-size 16`, prefix caching, `--enable-expert-parallel`, `--kv-cache-memory-bytes 5e9`, `--gpu-memory-utilization 0.88`, `--max-num-seqs 6`, the RCCL/HSA env, TunableOp; our stack is the TheRock 10.0 venv):
- as-is: dies during CUDA-graph capture on rank 2 with `HSA_STATUS_ERROR_EXCEPTION 0x1016` in `wvSplitK_hf_big_<__half,64,2,16,8,2,4>` — the `#else` stub of section 0 (`c350fa2`). With `VLLM_RDNA_FUSED_HC=0` the fp16 hyper-connection projections go through `rocm_unquantized_gemm` → `wvSplitK` at n ≤ 5. How does your gfx1030 build get past this?
- with `VLLM_RDNA_DENSE_GEMV=1` added: capture completes on all 3 ranks, the first request (4 336 tokens) is correct but takes 93 s, and the second request (1 614 tokens) hangs a worker → `EngineCore encountered a fatal error` (mq.dequeue timeout, no GPU error). Not usable here.

**MTP under PP works** — on the leapdragon tree, once the draft-token relay from your `22bb2e8` (recipe 0009 / vLLM #46994, `broadcast_draft`) is backported: PP=3, eager, prefix caching on, k=1: 21.2-21.7 t/s, acceptance 89.5 %; k=2: 24.9-28.3 t/s, 2.62 tokens/step, 86.5 %; fresh-server 4 336-token prompt correct, 4-way bursts correct. With cudagraphs it fails at startup on the drafter rank (`mamba_utils.py: expected 4 block tables, got 3` in `_ensure_align_ctx`). Not tested on `rdna_extras` (a worktree with the same MTP patches is ready).

## 11. 17 Sept, evening — MTP under PP with cudagraphs (leapdragon tree) and prefix-cache hit rates

- **MTP + cudagraphs + prefix caching under PP=3 works on the leapdragon tree** once vLLM **#54044** ("Reset cached Mamba align metadata on profiling teardown", merged 30 Aug, missing from leapdragon's 29 Aug base) is backported on top of the 0009 draft relay: k=1 → 51-52 t/s decode (acceptance 79-90 %), k=2 → 57-60 t/s (81-87 %, 2.45-2.62 tokens/step), outputs correct, 4-way bursts correct, 10 min of mixed load clean. The automatic KV sizing collapses to 61K tokens with MTP + graphs (graph-pool overestimate + per-rank minimum); `--kv-cache-memory-bytes 3.5e9` gives 219-277K tokens with 0.4-2 GB left per card (partition 18,17,13).
- **Prefix-cache hit rate, same prompt sent 3 times on an idle server** (`vllm:prefix_cache_hits_total` deltas): leapdragon cudagraphs without MTP → 97-99 % from the 2nd request; leapdragon + MTP k=2 → 0 % on the 2nd, 34-91 % on the 3rd (state checkpoints only published on the second computation, only at 1 568-token boundaries — cf. #54076); **`rdna_extras` with `--enable-prefix-caching` → 0 % on every repeat (19K and 4.7K prompts)**. So on your tree the prefix cache currently yields nothing while the align copies corrupt state under concurrency (section 10) — worth checking what your mamba_utils rewrite does to block publication.

## 12. 17 Sept, late — where the prefill time goes (and an erratum)

Torch-profiler traces of one 4 336-token prefill, rank 0, same container (your tree rebuilt in the leapdragon ROCm 7.14 image; leapdragon `rdna2/qwen38-flash-next`): `moe_gemm_q4_kernel_rdna2` (your HIP W4A16 MoE) **582 ms** vs leapdragon's Triton `fused_moe_kernel_gptq_awq` **3 849 ms** for the same 96 launches — 6.6×. Everything else (dense GEMMs ~700 ms, QSA ~310 ms, FLA GDN ~135 ms) is within noise. That single kernel is the whole prefill gap between the two trees (1 700 vs 740 t/s at 16K). Your HIP GDN prefill chain, forced back on (`VLLM_GDN_HIP_PREFILL=1`), is 15 % slower than Triton/FLA here (1 420 vs 1 650 t/s at 16K), consistent with `cd1231f`. conv1d HIP, AWQ prefill HIP, `_rocm_C` RMSNorms and FA-RDNA2 each switched off: no measurable effect on prefill at 16K.

Erratum on section 4: our `V620_MOE_TRITON=1` diagnostic gate changed the log line but not the kernel (the quant method had already selected `CompressedTensorsWNA16RDNA2MoEMethod`), so the 15 Sept "Triton MoE A/B" was not one. Moot since section 9, but stated for the record.

## 13. 17 Sept, evening — MTP on `rdna_extras`, the V2 runner, and your MoE kernel on the leapdragon tree

**MTP directly on your tree works, but only on the V2 model runner.** Worktree on `50120e1` + the fixes above (`patches/rdna_extras/0010–0016` in this repo); PP=3, eager, cache off, `VLLM_ROCM_MOE_PADDING=0`. The draft-token relay of your `22bb2e8` (`broadcast_draft`) lives only in `vllm/v1/worker/gpu/`, so `VLLM_USE_V2_MODEL_RUNNER=1` is required. Six small fixes were needed: the drafter's `embed_tokens` stays a CPU placeholder under PP because `_maybe_share_embeddings` is gated on `pp world_size == 1`; `llm_base_proposer.py` reads `draft_model_config.get_hidden_size()` directly, bypassing `Qwen4ExpMTPProposer._get_hidden_size()` (×`hc_mult` streams → `shape [2048, 4, 2560] invalid for 5242880`); your `causal_conv1d_{fwd,update}_rdna2` require `state_len == width−1` while spec decode allocates `width−1+num_spec` (we fall back to Triton in that case); `image_token_index` vs `image_token_id`; the `_v620_mark` import (ours). Result: **k=1: 18.3 t/s, acceptance 87 %; k=2: 23.8 t/s, 72.6 % (2.45 tokens/step)**, 4-stream bursts correct.

**But the V2 runner decodes at half speed on your tree: 13.4 t/s without MTP vs 25.4–26.3 on V1, same worktree.** Torch profile (rank 0, 64 decode steps): `ple_offload/connector.py: prepare_forward → _launch_inline` takes **87 ms median per step** on V2 and does not appear on V1; the model forward is identical (41.8 ms). Cause: `gpu/model_runner.py._setup_ple_offload` hands the connector `self.input_buffers.input_ids` (GPU), while the V1 runner hands `self.input_ids.cpu`. With GPU input ids the step-N+1 lookup can only start after forward N has produced them, so the ~30 ms CPU n-gram lookup is serialised behind every forward (42 + 30 ≈ 75 ms = 13.4 t/s); with CPU ids the V1 runner overlaps it. `PLE_OFFLOAD_DOORBELL=0` changes nothing (13.3–13.5). This is the same regime as leapdragon's eager mode (14.7 t/s) — what saves leapdragon is cudagraphs. Two dead ends for the record: V1 + MTP async → `PP+async expects sampled_token_ids to have shape [num_reqs, 1]`; V1 + MTP + `--no-async-scheduling` → hangs after the first prefill (`No available shared memory broadcast block`). And **V2 + cudagraphs on your tree** (your serve env, `FULL_AND_PIECEWISE`, `VLLM_RDNA_DENSE_GEMV=1`, TheRock 10.0) captures fine and dies on the first request with `hipErrorIllegalAddress` on rank 2 (log on request).

**Your MoE kernel on the leapdragon tree.** `csrc/rocm/moe_q_gemm_rdna2.cu` + `q_gemm_rdna2_common.cuh` + `qdq_4_rdna2.cuh` build unchanged as a standalone torch extension in 36 s inside the leapdragon ROCm 7.14 image (own op namespace, `rdna2-moe/` here); your `CompressedTensorsWNA16RDNA2MoEMethod` is byte-identical to leapdragon's RDNA3 one (`rdna3→rdna2`), so the glue is a gfx10x branch in `rocm_moe_rdna.py` plus the RDNA2 class name in two exact-name allow-lists of leapdragon's `routed_experts.py` (`intermediate_size_full`, compressed-tensors transposed load). Measured on the same PP=3 harness, cudagraphs: **prefill 409 → 1 201 tok/s at 4K, 738 → 1 794 at 16K, decode unchanged at 48 t/s; with MTP k=2: 1 078 / 1 863 prefill and 57–62 t/s decode**, bursts clean, 30 min stable at 131K, 262K boots. So the 6.6× MoE gap of section 12 is real and your kernel closes it. Side finding for leapdragon: their tuned int4 Triton MoE config is keyed `E=128` (experts ÷ EP4) and is never loaded in PP3 without EP — an `E=512` copy alone gives +68 % / +40 % prefill.

## 14. `rdna_extras` @ `b33f9b6` with the maintainer's production configuration, transposed to PP=3 (19 Sept)

Tree: `b33f9b6` + our six PP3 patches (`patches/rdna_extras-b33f9b6/`, they replay without conflict), compiled extensions from the 17 Sept build (the csrc delta since `50120e1` is `rdna_allreduce` + an arch guard, unused here). Launcher: `scripts/vllm-native-pp3.sh` with `OGFX_PROD=1` = the published TP4 configuration minus what is TP-only (custom all-reduce, `RCCL_P2P_*`, `NCCL_PROTO`, expert parallel), `NCCL_P2P_LEVEL=PHB` (no shared PCIe switch here, `pix` disables P2P), `--language-model-only`, partition 18,17,13, V1 runner, `FULL_AND_PIECEWISE`, breakable CUDA graphs.

| what | result |
|---|---|
| boot with CUDA graphs under PP3 | **works now** (155 s; on `50120e1` the production configuration did not survive capture) |
| `--kv-cache-memory-bytes` | 3.5e9 → OOM at KV allocation on stage 0; 2.5e9 boots (232K tokens) but stage 0 runs out of memory in `qsa_mqa_paged` on the first long prompt; **1.5e9 is stable** |
| **first request whose prefill batch fits a captured graph size** (74-token prompt, or the 240-token tail of a 4 336-token prompt; 2 048 / 4 096-token prompts pass) | **engine dies**: `scheduler.py:1882 update_from_output` → `KeyError: '<req id>'` in `model_runner_output.req_id_to_index` (path `step_with_batch_queue`, i.e. pipeline parallel only). Same with `cudagraph_mode=PIECEWISE`, with prefix caching off, with TunableOp off. Eager is fine. No worker-side error. |
| workaround | `"cudagraph_capture_sizes":[1,2,4,8]` (`CGSIZES=1,2,4,8`): every prefill batch runs eager, decode replays graphs — 74 / 1 614 / 4 336-token probes correct |
| single stream, workaround config | prefill 1 077 / 1 611 tok/s (4K / 16K), decode **27.8** tok/s — against 26.7 eager on the same tree: the decode graphs buy ~4 % under PP3 (leapdragon's tree goes 15 → 51 with graphs + int8 dense) |
| eager, cache off (our 17 Sept settings) on `b33f9b6` | 1 079 / 1 617 tok/s, 26.7 decode — unchanged from `50120e1` |
| **prefix caching + concurrency** (`stab.py --seed 1`, 8 min, 45 iterations / 61 outputs, 4-stream bursts) | **0 corrupted output** — the deterministic `!!!` at iteration 12 of §10 is gone (`741e5bc` is in). The 5 errors are our harness asking for 65 536-token prompts on a 65 536 window (HTTP 400). |
| prefix cache hit rate | still **0.0 %** over the whole run |
