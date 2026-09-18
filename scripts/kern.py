import sys, json, gzip, glob, os, collections
f = sorted(glob.glob(os.path.join(sys.argv[1], "*rank1*.json*")))[0]; steps = 116
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
k = [e for e in ev if e.get("ph") == "X" and (e.get("cat") or "").lower() == "kernel" and "nccl" not in e["name"].lower()]
t0 = min(e["ts"] for e in k); t1 = max(e["ts"] for e in k); k = [e for e in k if e["ts"] >= t0 + 0.4 * (t1 - t0)]
W = (t1 - t0) * 0.6; steps = W / 23420
d = collections.Counter(); c = collections.Counter()
for e in k: d[e["name"][:150]] += e["dur"]; c[e["name"][:150]] += 1
print(f"rang 1 : {steps:.0f} pas ; {sum(c.values())/steps:.0f} noyaux par pas ; occupé {sum(d.values())/steps/1e3:.2f} ms par pas")
for n, du in d.most_common(16): print(f"{du/steps/1e3:6.3f} ms/pas  x{c[n]/steps:5.1f}  moy {du/c[n]:5.0f} µs  {n}")
