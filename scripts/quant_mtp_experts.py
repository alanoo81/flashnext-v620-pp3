#!/usr/bin/env python3
"""Quantifie les experts du drafter MTP (bf16 fusionnés) en W4A16 compressed-tensors
(int4 symétrique, groupes de 128, uint4b8 packé en int32 le long de K), au format
per-expert des experts principaux, et fabrique un dossier checkpoint dérivé.

Usage: quant_mtp_experts.py <src_dir> <dst_dir>
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
GS = 128
os.makedirs(DST, exist_ok=True)


def quant_rtn(w: torch.Tensor):
    """w [N, K] float → (packed int32 [N, K/8], scale bf16 [N, K/GS])."""
    n, k = w.shape
    assert k % GS == 0 and k % 8 == 0
    wf = w.float().reshape(n, k // GS, GS)
    scale = wf.abs().amax(dim=-1).clamp_min(1e-8) / 7.0  # [N, K/GS]
    q = torch.round(wf / scale[..., None]).clamp_(-8, 7).reshape(n, k)
    nib = (q + 8).to(torch.int32)  # uint4b8 : valeur stockée = q + 8, dequant = (nib - 8) * scale
    nib = nib.reshape(n, k // 8, 8)
    packed = torch.zeros(n, k // 8, dtype=torch.int32)
    for i in range(8):
        packed |= nib[:, :, i] << (4 * i)  # nibble 0 dans les bits de poids faible
    return packed.contiguous(), scale.to(torch.bfloat16).contiguous()


def unpack(packed: torch.Tensor, scale: torch.Tensor):
    n, k8 = packed.shape
    p = packed.to(torch.int64) & 0xFFFFFFFF
    nib = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(n, k8 * 8)
    q = nib.float() - 8
    return q.reshape(n, -1, GS) * scale.float()[..., None]


# --- validation de la convention sur un expert principal : dépaqueter → requantifier → repaqueter
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
base = "model.language_model.layers.2.mlp.experts.0.gate_proj."
f = safe_open(os.path.join(SRC, wm[base + "weight_packed"]), "pt")
p0, s0 = f.get_tensor(base + "weight_packed"), f.get_tensor(base + "weight_scale")
w0 = unpack(p0, s0).reshape(p0.shape[0], -1)
p1, s1 = quant_rtn(w0)
print("validation packing (expert principal, aller-retour) : packed identiques =", torch.equal(p0, p1),
      "| scales max rel diff =", ((s1.float() - s0.float()).abs() / s0.float().abs().clamp_min(1e-9)).max().item())

# --- experts du drafter
src_mtp = os.path.join(SRC, wm["mtp.layers.0.mlp.experts.gate_up_proj"])
fm = safe_open(src_mtp, "pt")
out: dict[str, torch.Tensor] = {}
for name in fm.keys():
    if name in ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"):
        continue
    out[name] = fm.get_tensor(name)  # autres tenseurs mtp inchangés (bf16)
gu = fm.get_tensor("mtp.layers.0.mlp.experts.gate_up_proj")  # [E, 2*inter, hidden]
dn = fm.get_tensor("mtp.layers.0.mlp.experts.down_proj")  # [E, hidden, inter]
E, two_inter, hidden = gu.shape
inter = two_inter // 2
assert dn.shape == (E, hidden, inter), dn.shape
print(f"experts drafter : E={E} inter={inter} hidden={hidden}")
err_sum, err_n = 0.0, 0
for e in range(E):
    for proj, w in (("gate_proj", gu[e, :inter]), ("up_proj", gu[e, inter:]), ("down_proj", dn[e])):
        packed, scale = quant_rtn(w)
        pre = f"mtp.layers.0.mlp.experts.{e}.{proj}."
        out[pre + "weight_packed"] = packed
        out[pre + "weight_scale"] = scale
        out[pre + "weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
        if e < 4:
            wq = unpack(packed, scale).reshape(w.shape)
            err_sum += ((wq - w.float()).abs().mean() / w.float().abs().mean()).item()
            err_n += 1
    if e % 64 == 0:
        print(f"  expert {e}/{E}", flush=True)
print(f"erreur relative moyenne |Δw|/|w| (4 premiers experts) : {err_sum/err_n:.4f}")

new_shard = "model_mtp_q.safetensors"
save_file(out, os.path.join(DST, new_shard), metadata={"format": "pt"})
print("shard écrit :", new_shard, f"{os.path.getsize(os.path.join(DST, new_shard))/2**30:.2f} GiB, {len(out)} tenseurs")

# --- index + config + liens vers les autres fichiers
new_wm = {k: v for k, v in wm.items() if v != os.path.basename(src_mtp)}
for k in out:
    new_wm[k] = new_shard
idx["weight_map"] = new_wm
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(SRC, "config.json")))
ign = cfg["quantization_config"]["ignore"]
ign = [p for p in ign if p not in ("re:.*mtp\\..*", "re:^mtp.*")]
ign.append(r"re:^mtp\.(?!layers\.\d+\.mlp\.experts(\.|$)).*")  # tout mtp sauf les experts (construit sous layers.48, testé sur experts.0.<proj>)
cfg["quantization_config"]["ignore"] = ign
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
print("ignore :", ign)
for fn in os.listdir(SRC):
    if fn in ("config.json", "model.safetensors.index.json", os.path.basename(src_mtp)) or fn.startswith("."):
        continue
    dst = os.path.join(DST, fn)
    if not os.path.exists(dst):
        os.symlink(os.path.join(SRC, fn), dst)
print("dossier prêt :", DST)
