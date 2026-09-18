#!/usr/bin/env python3
# Chronologie de ~2 pas de décodage : segments de calcul GPU (hors NCCL, fusionnés à 0,3 ms près) et noyaux NCCL, par rang.
import sys, json, gzip, glob, os
d = sys.argv[1]; data = {}
for f in sorted(glob.glob(os.path.join(d, "*rank*.json*"))):
    r = int(os.path.basename(f).split("rank")[1][0])
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    data[r] = sorted(((e["ts"], e["ts"] + e.get("dur", 0), "nccl" in e["name"].lower()) for e in ev if e.get("ph") == "X" and (e.get("cat") or "").lower() == "kernel"), key=lambda x: x[0])
t1 = min(v[-1][1] for v in data.values()); T0 = t1 - 400000; T1 = T0 + 50000
rows = []
for r, k in data.items():
    seg = None
    for s, e, n in k:
        if e < T0 or s > T1: continue
        if n: rows.append((s, e, r, "NCCL")); continue
        if seg and s - seg[1] <= 300: seg[1] = max(seg[1], e); seg[2] += e - s
        else:
            if seg: rows.append((seg[0], seg[1], r, f"calcul (occupé {seg[2]/1e3:.2f} ms)"))
            seg = [s, e, e - s]
    if seg: rows.append((seg[0], seg[1], r, f"calcul (occupé {seg[2]/1e3:.2f} ms)"))
for s, e, r, w in sorted(rows):
    if e - s < 150 and w != "NCCL": continue
    print(f"{(s-T0)/1e3:8.2f} -> {(e-T0)/1e3:8.2f} ms  ({(e-s)/1e3:6.2f})  {'    '*r}rang {r}  {w}")
