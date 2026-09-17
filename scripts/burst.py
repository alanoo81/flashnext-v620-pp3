#!/usr/bin/env python3
# Rafale concurrente puis sondes : reproduit la corruption "!!!" après charge. usage: burst.py <port> [c=4] [--reset] [--tag x]
import sys, json, urllib.request, time, threading, re
port = int(sys.argv[1]); c = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
U = f"http://127.0.0.1:{port}"; MODEL = "flash-next"
def post(path, body, timeout=3600):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
SUF = "\n\nRésumé détaillé et structuré du texte ci-dessus, en français :\n"
GOOD = re.compile(r"(user is asking|user wants|Résumé|résumé|summary|texte|structur|Contexte|Objectif|The text|document|Voici|Ce texte|## |\*\*)", re.I)
def gen(p, mx=64):
    j = post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": mx, "temperature": 0, "ignore_eos": True})
    return j["choices"][0]["text"], j["usage"]["prompt_tokens"]
def probe(label, chars=8000, off=0):
    out, n = gen(text[off:off + chars] + SUF, 48)
    v = "ok " if GOOD.search(out) else ("!!!" if out.count("!") > 20 else "?? ")
    print(f"{tag} {label:34s} N={n:5d} {v} {out[:90]!r}", flush=True)
    return v
probe("avant rafale (8K chars)")
ps = [text[off:off + ch] + SUF for off, ch in ((100000, 6000), (300000, 12000), (620000, 16000), (900000, 3000))][:c]
outs = [None] * c
def w(i):
    try: outs[i] = gen(ps[i], 64)
    except Exception as e: outs[i] = e
t0 = time.time(); th = [threading.Thread(target=w, args=(i,)) for i in range(c)]; [t.start() for t in th]; [t.join() for t in th]
for i, o in enumerate(outs):
    if isinstance(o, Exception): print(f"{tag}   rafale flux {i}: ERREUR {o}", flush=True)
    else: print(f"{tag}   rafale flux {i}: N={o[1]:5d} {'ok ' if GOOD.search(o[0]) else ('!!!' if o[0].count('!') > 20 else '?? ')} {o[0][:80]!r}", flush=True)
print(f"{tag}   rafale c={c} en {time.time()-t0:.1f}s", flush=True)
probe("après rafale, même prompt qu avant")
probe("après rafale, prompt neuf (offset 50000)", 8000, 50000)
if "--reset" in sys.argv:
    urllib.request.urlopen(urllib.request.Request(U + "/reset_prefix_cache", method="POST"), timeout=60).read()
    probe("après reset_prefix_cache (offset 50000)", 8000, 50000)
    probe("après reset, prompt neuf (offset 150000)", 8000, 150000)
