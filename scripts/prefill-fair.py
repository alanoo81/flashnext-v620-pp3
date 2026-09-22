#!/usr/bin/env python3
"""Équité prefill/décodage : une requête longue (prefill de ~N jetons) occupe le serveur ; 2 s après, deux requêtes
courtes arrivent (style « résultat d'outil »). On mesure leur TTFT et leur débit pendant que le long prompt est lu,
plus le débit de prefill total. usage: prefill-fair.py <port> [--long 100000] [--tag x]"""
import sys, json, time, threading, urllib.request, re
port = int(sys.argv[1])
def opt(k, d): return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
LONG = int(opt("--long", 100000)); TAG = opt("--tag", "fair")
text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
def metrics():
    m = {}
    for l in urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode().splitlines():
        mm = re.match(r"^vllm:(request_prompt_tokens_sum|request_prefill_time_seconds_sum|num_preemptions_total)(?:\{[^}]*\})?\s+([0-9.eE+]+)", l)
        if mm: m[mm.group(1)] = m.get(mm.group(1), 0) + float(mm.group(2))
    return m
res = {}
def req(name, prompt, maxtok, delay):
    time.sleep(delay)
    t0 = time.time(); first = None; n = 0
    body = {"model": "flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0, "stream": True}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            for line in resp:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
                d = json.loads(line[6:]); delta = d["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning"):
                    if first is None: first = time.time()
                    n += 1
        t2 = time.time()
        res[name] = (first - t0 if first else None, n, (n-1)/(t2-first) if first and t2 > first and n > 1 else 0, t2 - t0)
    except Exception as e:
        res[name] = (None, 0, 0, time.time() - t0); print(f"  {name}: ERREUR {type(e).__name__} {str(e)[:70]}")
m0 = metrics(); t_start = time.time()
th = [threading.Thread(target=req, args=("long", "Contexte :\n" + text[:LONG*4] + "\nRésume en 3 points.", 80, 0)),
      threading.Thread(target=req, args=("court1", "Donne la capitale de la France en un mot.", 40, 2.0)),
      threading.Thread(target=req, args=("court2", "Combien font 17 x 23 ? Réponds par le nombre seul.", 40, 4.0))]
for t in th: t.start()
for t in th: t.join()
m1 = metrics(); wall = time.time() - t_start
pt = m1.get("request_prompt_tokens_sum",0) - m0.get("request_prompt_tokens_sum",0)
pti = m1.get("request_prefill_time_seconds_sum",0) - m0.get("request_prefill_time_seconds_sum",0)
f = lambda x: "—" if x is None else f"{x:.1f}"
print(f"== {TAG} : mur {wall:.1f}s | prefill serveur {pt/pti if pti>0.05 else 0:.0f} j/s ({pt:.0f} jetons) | préemptions {int(m1.get('num_preemptions_total',0)-m0.get('num_preemptions_total',0))}")
for k in ("long", "court1", "court2"):
    if k in res:
        ttft, n, gen, tot = res[k]
        print(f"   {k:7s} TTFT {f(ttft):>7s}s  {n:3d} jetons à {gen:5.1f} j/s  total {tot:6.1f}s")
