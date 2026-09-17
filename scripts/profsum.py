#!/usr/bin/env python3
# Agrège une trace torch profiler (json / json.gz, format Chrome) : temps GPU par noyau, top N.
import sys, json, gzip, glob, os, collections
d = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
files = sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True))
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    try:
        tr = json.load(op(f, "rt"))
    except Exception as e:
        print(f, "illisible:", e); continue
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = collections.Counter(); cnt = collections.Counter(); total = 0.0
    tmin = min((e["ts"] for e in ev if e.get("ph") == "X"), default=0); tmax = max((e["ts"] + e.get("dur", 0) for e in ev if e.get("ph") == "X"), default=0)
    for e in ev:
        if e.get("ph") != "X": continue
        cat = (e.get("cat") or "").lower()
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            name = e["name"]
            name = name.split("(")[0][:90]
            kern[name] += e.get("dur", 0); cnt[name] += 1; total += e.get("dur", 0)
    print(f"\n##### {os.path.basename(f)}  fenêtre {(tmax-tmin)/1e3:.0f} ms, GPU occupé {total/1e3:.0f} ms ({100*total/max(1,(tmax-tmin)):.0f} %)")
    for name, dur in kern.most_common(top):
        print(f"{dur/1e3:9.1f} ms {100*dur/max(1,total):5.1f}%  x{cnt[name]:<6d} {name}")
