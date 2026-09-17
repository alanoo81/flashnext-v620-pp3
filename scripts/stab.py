#!/usr/bin/env python3
# Boucle de stabilité : tailles et offsets aléatoires, rafales concurrentes périodiques, contrôle santé + cohérence.
# usage: stab.py <port> <api> <minutes> [--sizes 512,2048,4096,8192,16384,32768,65536] [--tag x] [--seed 1]
import sys, json, urllib.request, time, re, random, threading, traceback
args = [a for a in sys.argv[1:] if not a.startswith("--")]
port, api, minutes = int(args[0]), args[1], float(args[2])
sizes = [int(x) for x in (sys.argv[sys.argv.index("--sizes") + 1] if "--sizes" in sys.argv else "512,2048,4096,8192,16384,32768,65536").split(",")]
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else api
random.seed(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1)
U = f"http://127.0.0.1:{port}"; MODEL = "flash-next"
def post(path, body, timeout=3600):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)
def ntok(p):
    if api == "llama": return len(json.loads(post("/tokenize", {"content": p}).read())["tokens"])
    return len(json.loads(post("/tokenize", {"model": MODEL, "prompt": p}).read())["tokens"])
SUF = "\n\nRésumé détaillé et structuré du texte ci-dessus, en français :\n"
text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
def prompt_for(n, off):
    lo, hi = 1, min(len(text) - off, 8 * n)
    while lo < hi:
        mid = (lo + hi) // 2
        if ntok(text[off:off + mid] + SUF) >= n: hi = mid
        else: lo = mid + 1
    return text[off:off + lo] + SUF
GOOD = re.compile(r"(user is asking|user wants|Résumé|résumé|summary|texte|structur|Contexte|Objectif|The text|document|Voici|Ce texte|## |\*\*)", re.I)
def gen(p, mx=96):
    body = {"model": MODEL, "prompt": p, "max_tokens": mx, "temperature": 0}
    if api == "vllm": body["ignore_eos"] = True
    j = json.loads(post("/v1/completions", body).read())
    return j["choices"][0]["text"], j.get("usage", {}).get("prompt_tokens", -1)
def health():
    try: return urllib.request.urlopen(U + "/health", timeout=5).status == 200
    except Exception: return False
t_end = time.time() + minutes * 60; it = 0; nok = 0; nsus = 0; nerr = 0; sus = []
print(f"# stab {tag} {minutes} min, tailles {sizes}", flush=True)
while time.time() < t_end:
    it += 1
    n = random.choice(sizes); off = random.randrange(0, max(1, len(text) - 8 * n - 1000))
    try:
        if it % 6 == 0:  # rafale concurrente de 4 petites requêtes
            ps = [prompt_for(random.choice([512, 2048, 4096]), random.randrange(0, 800000)) for _ in range(4)]
            outs = [None] * 4
            def w(i):
                try: outs[i] = gen(ps[i], 64)
                except Exception as e: outs[i] = e
            th = [threading.Thread(target=w, args=(i,)) for i in range(4)]; [t.start() for t in th]; [t.join() for t in th]
            for i, o in enumerate(outs):
                if isinstance(o, Exception): nerr += 1; print(f"[{it}] BURST ERREUR {o}", flush=True)
                elif GOOD.search(o[0]): nok += 1
                else: nsus += 1; sus.append((it, "burst", o[1], o[0][:100])); print(f"[{it}] BURST SUSPECT N={o[1]} {o[0][:100]!r}", flush=True)
        else:
            p = prompt_for(n, off); t0 = time.time(); out, pt = gen(p); dt = time.time() - t0
            if GOOD.search(out): nok += 1; print(f"[{it}] ok  N={pt:6d} off={off:7d} {dt:5.1f}s", flush=True)
            else: nsus += 1; sus.append((it, n, off, out[:100])); print(f"[{it}] SUSPECT N={pt:6d} off={off:7d} {out[:100]!r}", flush=True)
    except Exception as e:
        nerr += 1; print(f"[{it}] ERREUR N={n} off={off} {type(e).__name__}: {str(e)[:150]}", flush=True)
        if not health(): print("SERVEUR KO — arrêt", flush=True); break
        time.sleep(5)
print(f"== {tag} stabilité: {it} itérations, {nok} ok, {nsus} suspects, {nerr} erreurs, santé={'ok' if health() else 'KO'}", flush=True)
for s in sus[:20]: print("   suspect:", s, flush=True)
