#!/usr/bin/env python3
# Budget d un pas de décodage en PP3 à partir des traces torch profiler (une par rang).
# Pour chaque rang, sur la fenêtre de décodage stable (derniers 60 % de la trace) : part du temps mur passée en noyaux de
# calcul, en noyaux NCCL (= attente/transfert inter-étages), en copies, et inoccupée ; top noyaux ; plus gros trous GPU
# et ce que le CPU faisait pendant ces trous (rang 0 : aller-retour PLE).
# usage: decstep.py <dir> [ms_par_pas]
import sys, json, gzip, glob, os, collections, bisect
d = sys.argv[1]; step_ms = float(sys.argv[2]) if len(sys.argv) > 2 else None
files = sorted(glob.glob(os.path.join(d, "*rank*.json*")))
summary = []
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    tr = json.load(op(f, "rt")); ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    X = [e for e in ev if e.get("ph") == "X" and "ts" in e]
    gpu = [e for e in X if (e.get("cat") or "").lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not gpu: print(f, "aucun noyau GPU"); continue
    t0 = min(e["ts"] for e in gpu); t1 = max(e["ts"] + e.get("dur", 0) for e in gpu)
    w0 = t0 + 0.4 * (t1 - t0); W = t1 - w0
    g = sorted((e for e in gpu if e["ts"] >= w0), key=lambda e: e["ts"])
    cat = collections.Counter(); names = collections.Counter(); cnt = collections.Counter()
    for e in g:
        n = e["name"]; du = e.get("dur", 0)
        c = "nccl" if "nccl" in n.lower() else ("copie" if (e.get("cat") or "").lower() != "kernel" else "calcul")
        cat[c] += du; key = n.split("(")[0][:70]; names[key] += du; cnt[key] += 1
    # union des intervalles occupés -> temps inoccupé et trous
    busy = 0; gaps = []; cur_s, cur_e = g[0]["ts"], g[0]["ts"] + g[0].get("dur", 0)
    for e in g[1:]:
        s, en = e["ts"], e["ts"] + e.get("dur", 0)
        if s > cur_e:
            busy += cur_e - cur_s; gaps.append((s - cur_e, cur_e, s)); cur_s, cur_e = s, en
        else: cur_e = max(cur_e, en)
    busy += cur_e - cur_s
    idle = W - busy
    rank = os.path.basename(f).split("rank")[1][0]
    print(f"\n##### rang {rank} — fenêtre {W/1e3:.0f} ms : calcul {100*cat['calcul']/W:.1f} %, nccl {100*cat['nccl']/W:.1f} %, copies {100*cat['copie']/W:.1f} %, GPU inoccupé {100*idle/W:.1f} %")
    if step_ms:
        print(f"      par pas de {step_ms:.1f} ms : calcul {step_ms*cat['calcul']/W:.2f} ms, nccl {step_ms*cat['nccl']/W:.2f} ms, inoccupé {step_ms*idle/W:.2f} ms")
    summary.append((rank, cat["calcul"] / W, cat["nccl"] / W, idle / W))
    for n, du in names.most_common(14):
        print(f"   {100*du/W:5.1f} %  x{cnt[n]:<6d} {n}")
    # trous : histogramme + ce que fait le CPU pendant les 200 plus gros
    gaps.sort(reverse=True)
    big = [x for x in gaps if x[0] >= 200]
    print(f"   trous GPU >= 0,2 ms : {len(big)}, total {sum(x[0] for x in big)/1e3:.0f} ms ({100*sum(x[0] for x in big)/W:.1f} % de la fenêtre) ; médiane des 50 plus gros : {sorted(x[0] for x in gaps[:50])[len(gaps[:50])//2]/1e3:.2f} ms")
    cpu = [e for e in X if (e.get("cat") or "").lower() in ("cpu_op", "user_annotation", "python_function", "cuda_runtime") and e.get("dur", 0) >= 100]
    cpu.sort(key=lambda e: e["ts"]); starts = [e["ts"] for e in cpu]
    during = collections.Counter()
    for gl, gs, ge in gaps[:300]:
        i = bisect.bisect_left(starts, gs - 50000)
        best = collections.Counter()
        while i < len(cpu) and cpu[i]["ts"] < ge:
            e = cpu[i]; ov = min(ge, e["ts"] + e["dur"]) - max(gs, e["ts"])
            if ov > 0.5 * gl: best[e["name"][:80]] = max(best[e["name"][:80]], ov)
            i += 1
        for n, ov in best.items(): during[n] += ov
    for n, ov in during.most_common(12):
        print(f"      pendant les trous : {ov/1e3:8.1f} ms  {n}")
print("\n##### synthèse (part du temps mur)")
for r, c, n, i in summary: print(f"rang {r}: calcul {100*c:.1f} %  nccl {100*n:.1f} %  inoccupé {100*i:.1f} %")
print(f"somme des parts de calcul : {100*sum(c for _, c, _, _ in summary):.1f} %  (100 % = pipeline série sans aucune perte)")
