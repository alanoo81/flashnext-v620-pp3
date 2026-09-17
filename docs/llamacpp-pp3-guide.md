# Qwen3.8-Flash-Next on 3× Radeon PRO V620 with llama.cpp (layer split, 262K ctx)

**TL;DR** — stock llama.cpp master, HIP build for `gfx1030`, unsloth `UD-Q4_K_XL`, and:
`-ngl 99 -sm layer -ts 15,16,17 --fit off -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512`.
The 51B n-gram table stays in host RAM automatically. Do **not** use `-ot`, `--fit` or `-sm tensor`. ~26 GiB VRAM per card, ≥64 GB RAM (112-128 better).

**1. Host** — any Linux with in-kernel `amdgpu` (we're on Proxmox 9.2 / kernel 7.0). Kernel cmdline: `amdgpu.ras_enable=0` (ECC off → 32 752 MiB usable per card). Never install the GIM/SR-IOV driver. Three `renderD*` + `/dev/kfd` must exist.

**2. Container (LXC unprivileged, Ubuntu 24.04)** — ROCm supports gfx1030 on Ubuntu 24.04 only. `/etc/pve/lxc/<id>.conf`:
```
dev0: /dev/dri/renderD128,gid=993
dev1: /dev/kfd,gid=993
dev2: /dev/dri/card0,gid=44
dev3: /dev/dri/renderD129,gid=993
dev4: /dev/dri/card1,gid=44
dev5: /dev/dri/renderD130,gid=993
dev6: /dev/dri/card2,gid=44
memory: 114688
```
(993 = `render`, 44 = `video` inside the CT.) `rocminfo | grep -c gfx1030` → 3. No `HSA_OVERRIDE_GFX_VERSION`.

**3. ROCm user space** — `amdgpu-install --usecase=hiplibsdk --no-dkms` (7.2.4 and 10.0 both fine).

**4. Build**
```
cmake -S . -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j$(nproc) --target llama-server llama-cli llama-bench
```
No `GGML_HIP_ROCWMMA_FATTN` on RDNA2 (`-fa on` still works via the vector kernels). Flag is `GPU_TARGETS` (ex-`AMDGPU_TARGETS`).

**5. Model** — `hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --local-dir /root/models/flash-next` (104 GiB, 4 shards; 76.2 GiB of GPU weights; the only quant that leaves margin at 262K on 3×32 GB).

**6. Run**
```
HSA_NO_SCRATCH_RECLAIM=1 setsid nohup ./build-hip/bin/llama-server \
  -m /root/models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 99 -sm layer -ts 15,16,17 --fit off -fa on -ctk q8_0 -ctv q8_0 \
  -c 131072 -np 4 -b 2048 -ub 512 --jinja --host 0.0.0.0 --port 8085 > flashnext.log 2>&1 < /dev/null &
```
`-ts 15,16,17` = layers 0-15 / 16-31 / 32-47+output (26.0 / 25.6 / 26.5 GiB). Full 262K: `-c 262144 -np 1` (1-2 GiB free per card; `-ub 1024` OOMs there). Load takes 2-9 min cold; `setsid nohup` or it dies with your session.

**7. Expected (our fork, 131K/262K, 200 gen, q8_0 KV)** — prefill / decode t/s at 16K / 32K / 64K / 96K: 1233/28.8, 1199/28.0, 1128/27.1, 1072/26.3; 247K: 894/22.4. Stock master: same decode ≤32K, −20-25 % past 64K, and prefill ~500 t/s (see traps).

**8. Traps**
- `-sm tensor` → refused for `qwen4exp`. Use `-sm layer`.
- `-ot` / `--fit on` (default in recent builds) → OOM or 4-8 t/s. `--fit off`, no `-ot`.
- Only 1-2 GPUs visible → each card needs its `renderD12x` **and** `cardN` passed through, right gids.
- Prefill ~500 t/s instead of ~1 250 at 16K → stock-master regression (MoE weighted-reduction fusion re-decided on 0-token ubatches). Fixed in our fork (`504c099`), not upstream yet; Lemonade b1328 binaries show the same 526 t/s.
- Decode 27 → 20 t/s past 64K → stock QSA recomputes pooled indexer keys every ubatch; our fork caches them + block-level top-k.
- "KV cache shifting is not supported" → expected (recurrent GDN); the server truncates at `-c`.
- `pkill -f llama-server` kills your own shell → `pkill -x llama-server`.
- First request slow after start → cold mmap of the n-gram table, warms up.
- Garbled output → never seen with stock llama.cpp here; check KV types (`q8_0` both) and no rocWMMA build.

**9. Our fork `v620-qsa`** = master `9e717162` + 9 generic commits (FA vec RDNA tuning, prefill fusion fix, narrow `get_rows`, cached pooled keys, block-level top-k, deterministic radix top-k, f32 pooled keys, indexer V half, fused rms_norm+scale): +5 % decode at 16K, +29 % at 96K, deterministic greedy, PPL unchanged (2.7078 vs 2.7141 on 32×8192, ±0.015). MTP (PR ggml-org/llama.cpp#28243) works as a draft (+55-70 % decode) but halves prefill under layer split — experimental.

**10. Monitoring** — `amdgpu_top` / `nvtop` in the CT. In decode only one card works at a time (layer split): ~30 % utilisation per card at 28 t/s is normal.
