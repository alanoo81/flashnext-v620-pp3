#!/usr/bin/env python3
# Harnais de perf commun (vLLM et llama-server) : prompts coupés au token près, offsets distincts (pas de cache),
# /v1/completions en streaming, TTFT client -> prefill t/s, gen t/s ; contrôle de cohérence de la sortie.
# usage: perf3.py <port> <api: vllm|llama> "<N1,N2,...>" [ngen=200] [--reps 2] [--conc 4] [--tag x]
import sys, json, urllib.request, time, re, threading
args = [a for a in sys.argv[1:] if not a.startswith("--")]
port, api, ns = int(args[0]), args[1], [int(x) for x in args[2].split(",")]
ngen = int(args[3]) if len(args) > 3 else 200
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2
conc = int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else 0
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else api
twoshot = "--twoshot" in sys.argv  # prefill mesuré avec max_tokens=1, gen sur préfixe en cache (cache ON requis)
U = f"http://127.0.0.1:{port}"
MODEL = "flash-next"
def post(path, body, timeout=3600):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)
def ntok(p):
    if api == "llama":
        return len(json.loads(post("/tokenize", {"content": p}).read())["tokens"])
    return len(json.loads(post("/tokenize", {"model": MODEL, "prompt": p}).read())["tokens"])
SUF = "\n\nRésumé détaillé et structuré du texte ci-dessus, en français :\n"
text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
def prompt_for(n, off):
    lo, hi = 1, min(len(text) - off, 8 * n)
    while lo < hi:
        mid = (lo + hi) // 2
        if ntok(text[off:off + mid] + SUF) >= n: hi = mid
        else: lo = mid + 1
    p = text[off:off + lo] + SUF
    return p, ntok(p)
GOOD = re.compile(r"(user is asking|user wants|Résumé|résumé|summary|texte|structur|Contexte|Objectif|The text|document|Voici|Ce texte|## )", re.I)
def run_two(p, mx):
    t0 = time.time(); j = json.loads(post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 1, "temperature": 0}).read()); ta = time.time() - t0
    ptoks = j["usage"]["prompt_tokens"]
    t0 = time.time(); j = json.loads(post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": mx, "temperature": 0, **({"ignore_eos": True} if api == "vllm" else {})}).read()); tb = time.time() - t0
    out = j["choices"][0]["text"]; ctoks = j["usage"]["completion_tokens"]
    return dict(ptoks=ptoks, ttft=ta, pp=ptoks / ta, gen=(ctoks - 1) / tb, ctoks=ctoks, out=out, wall=ta + tb)
def run_one(p, mx):
    if twoshot: return run_two(p, mx)
    body = {"model": MODEL, "prompt": p, "max_tokens": mx, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if api == "vllm": body["ignore_eos"] = True
    t0 = time.time(); t1 = None; n = 0; txt = []; usage = None
    r = post("/v1/completions", body)
    for line in r:
        line = line.decode(errors="ignore").strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        j = json.loads(d)
        if j.get("usage"): usage = j["usage"]
        for c in j.get("choices", []):
            if c.get("text"):
                if t1 is None: t1 = time.time()
                n += 1; txt.append(c["text"])
    t2 = time.time()
    out = "".join(txt)
    ctoks = usage["completion_tokens"] if usage else n
    ptoks = usage["prompt_tokens"] if usage else -1
    ttft = (t1 - t0) if t1 else 0
    gen = (ctoks - 1) / (t2 - t1) if t1 and t2 > t1 and ctoks > 1 else 0
    return dict(ptoks=ptoks, ttft=ttft, pp=ptoks / ttft if ttft else 0, gen=gen, ctoks=ctoks, out=out, wall=t2 - t0)
print(f"# {tag} port={port} ngen={ngen} reps={reps}")
for n in ns:
    offs = [0, 300000][:reps]
    if n > 150000: offs = [0]
    res = []
    for k, off in enumerate(offs):
        try:
            p, real = prompt_for(n, off)
            r = run_one(p, ngen)
            ok = "ok " if GOOD.search(r["out"]) else "?? "
            res.append(r)
            print(f"{tag:12s} N={real:6d} off={off:6d} PP={r['pp']:7.1f} t/s (TTFT {r['ttft']:6.1f}s) GEN={r['gen']:5.1f} t/s ({r['ctoks']} tk) {ok} {r['out'][:70]!r}", flush=True)
        except Exception as e:
            print(f"{tag:12s} N={n:6d} off={off:6d} ERREUR {type(e).__name__}: {str(e)[:120]}", flush=True)
    if res:
        pp = sum(r["pp"] for r in res) / len(res); gen = sum(r["gen"] for r in res) / len(res)
        print(f"== {tag:10s} N={n:6d} PP={pp:7.1f} GEN={gen:5.1f}", flush=True)
if conc:
    n = 4096; ps = [prompt_for(n, off)[0] for off in (0, 300000, 620000, 900000)][:conc]
    results = [None] * conc
    def work(i):
        try: results[i] = run_one(ps[i], ngen)
        except Exception as e: results[i] = e
    t0 = time.time(); th = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
    [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    good = [r for r in results if isinstance(r, dict)]
    tot = sum(r["ctoks"] for r in good); ok = sum(1 for r in good if GOOD.search(r["out"]))
    for i, r in enumerate(results):
        if isinstance(r, dict): print(f"   flux {i} off={(0,300000,620000,900000)[i]} gen={r['gen']:.1f} ttft={r['ttft']:.1f}s {'ok ' if GOOD.search(r['out']) else '?? '} {r['out'][:140]!r}")
        else: print(f"   flux {i} ERREUR {r}")
    print(f"== {tag:10s} CONC c={conc} N=4096: {tot} tokens en {wall:.1f}s = {tot/wall:.1f} t/s agrégés, gen moyen/flux {sum(r['gen'] for r in good)/max(1,len(good)):.1f}, TTFT moyen {sum(r['ttft'] for r in good)/max(1,len(good)):.1f}s, {ok}/{len(good)} cohérents, {conc-len(good)} erreurs", flush=True)
