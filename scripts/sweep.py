#!/usr/bin/env python3
# Sonde de longueurs exactes (tokens) sur prompt TEXTE (comme vllmperf.py) : texte[:c] + instruction, c ajusté par bissection.
# usage: sweep.py "N1,N2,..." [off_chars] [max_tokens] [port] [--reset] [--restart "cmd"] [--show]
#   --reset   : POST /reset_prefix_cache avant chaque essai
#   --restart : exécute la commande shell avant chaque essai (redémarrage serveur) puis attend /health
import sys, json, urllib.request, time, re, subprocess
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ns = [int(x) for x in args[0].split(",")]
off = int(args[1]) if len(args) > 1 else 0
mx = int(args[2]) if len(args) > 2 else 60
port = int(args[3]) if len(args) > 3 else 8086
reset = "--reset" in sys.argv; show = "--show" in sys.argv
restart = sys.argv[sys.argv.index("--restart") + 1] if "--restart" in sys.argv else None
U = f"http://127.0.0.1:{port}"
def post(path, body, timeout=1800):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
def ntok(p): return len(post("/tokenize", {"model": "flash-next", "prompt": p})["tokens"])
def wait_up():
    for _ in range(360):
        try:
            if urllib.request.urlopen(U + "/health", timeout=2).status == 200: return True
        except Exception: pass
        time.sleep(5)
    return False
SUF = "\n\nRésumé détaillé et structuré du texte ci-dessus, en français :\n"
text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
def prompt_for(n):
    lo, hi = 1, min(len(text) - off, 8 * n)
    while lo < hi:  # plus petit c tel que ntok >= n
        mid = (lo + hi) // 2
        if ntok(text[off:off + mid] + SUF) >= n: hi = mid
        else: lo = mid + 1
    p = text[off:off + lo] + SUF
    return p, ntok(p)
GOOD = re.compile(r"(user is asking|Résumé|résumé|summary|texte|structur|Contexte|Objectif|The text|document)", re.I)
BAD = re.compile(r"(assert |</parameter>|</function>|system-reminder|<tool_call>)")
if restart is None: wait_up()
for n in ns:
    if restart:
        subprocess.run(restart, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); wait_up()
    p, real = prompt_for(n)
    if reset: urllib.request.urlopen(urllib.request.Request(U + "/reset_prefix_cache", method="POST"), timeout=60).read()
    t0 = time.time()
    r = post("/v1/completions", {"model": "flash-next", "prompt": p, "max_tokens": mx, "temperature": 0, "ignore_eos": True, "seed": 42})
    out = r["choices"][0]["text"]
    stripped = out.replace("<think>", "").replace("</think>", "").replace("assistant", "").strip()
    v = "BAD " if BAD.search(out) else "ok  " if GOOD.search(out) else "EMPTY" if len(stripped) < 8 else "??  "
    print(f"{v} N={real:5d} (asked {n}) mod256={real % 256:3d} tail2048={real % 2048:4d} {time.time()-t0:5.1f}s {out[:100]!r}", flush=True)
    if show: print(out[:400])
