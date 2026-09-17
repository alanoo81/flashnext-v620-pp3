#!/usr/bin/env python3
# Test de prefix cache sur serveur IDLE : même prompt 3 fois (max_tokens=5), TTFT + compteurs vllm:prefix_cache_* entre chaque.
import json, urllib.request, time, sys, re
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8086
chars = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
U = f"http://127.0.0.1:{port}"
text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
p = text[750000:750000 + chars] + "\n\nRésumé détaillé et structuré du texte ci-dessus, en français :\n"
def metrics():
    m = urllib.request.urlopen(U + "/metrics", timeout=10).read().decode()
    q = re.search(r"vllm:prefix_cache_queries_total\S* ([0-9.e+]+)", m); h = re.search(r"vllm:prefix_cache_hits_total\S* ([0-9.e+]+)", m)
    return (float(q.group(1)) if q else 0, float(h.group(1)) if h else 0)
def go(mx):
    t0 = time.time()
    j = json.loads(urllib.request.urlopen(urllib.request.Request(U + "/v1/completions", data=json.dumps({"model": "flash-next", "prompt": p, "max_tokens": mx, "temperature": 0, "ignore_eos": True}).encode(), headers={"Content-Type": "application/json"}), timeout=900).read())
    return time.time() - t0, j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]
q0, h0 = metrics()
for k in range(3):
    dt, pt, ct = go(5); q1, h1 = metrics()
    print(f"cache-test essai {k+1}: prompt {pt} tk, {ct} gen, {dt:5.1f} s | queries +{q1-q0:.0f} hits +{h1-h0:.0f} ({100*(h1-h0)/max(1,q1-q0):.0f} % hit)", flush=True)
    q0, h0 = q1, h1
