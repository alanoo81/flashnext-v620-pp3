#!/usr/bin/env python3
# Banc de qualité relatif : mêmes tokens envoyés à chaque configuration de serveur, on garde pour chaque position le
# logprob du vrai token et le top-20 (prompt_logprobs de vLLM), puis on compare deux relevés : perplexité, KL, accord top-1.
#   qual.py tokens  <port> [nchunks=40] [len=2048]   -> /root/qual/tokens.json (une fois pour toutes, offsets fixes)
#   qual.py collect <port> <nom>                      -> /root/qual/<nom>.npz
#   qual.py compare <ref> <autre> [...]               -> tableau
import sys, json, os, urllib.request, time
import numpy as np
D = "/root/qual"; os.makedirs(D, exist_ok=True); K = 20
def post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
cmd = sys.argv[1]
if cmd == "tokens":
    port = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 40; L = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    text = open("/root/ppl.txt", encoding="utf-8", errors="ignore").read()
    step = (len(text) - 12 * L) // n; chunks = []
    for i in range(n):
        ids = post(port, "/tokenize", {"model": "flash-next", "prompt": text[i * step: i * step + 12 * L], "add_special_tokens": False})["tokens"]
        assert len(ids) >= L, (i, len(ids)); chunks.append(ids[:L])
    json.dump(chunks, open(f"{D}/tokens.json", "w")); print(f"{n} blocs de {L} tokens, pas de {step} caractères dans un corpus de {len(text)}")
elif cmd == "collect":
    port = int(sys.argv[2]); name = sys.argv[3]; chunks = json.load(open(f"{D}/tokens.json"))
    lp_true, top_ids, top_lp = [], [], []; t0 = time.time()
    for ci, ids in enumerate(chunks):
        r = post(port, "/v1/completions", {"model": "flash-next", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": K})
        pl = r["choices"][0]["prompt_logprobs"]; assert len(pl) == len(ids), (len(pl), len(ids))
        for pos in range(1, len(ids)):
            d = pl[pos]; items = sorted(((int(t), v["logprob"], v.get("rank", 0)) for t, v in d.items()), key=lambda x: -x[1])
            lp_true.append(d[str(ids[pos])]["logprob"])
            top = items[:K]; top += [(-1, -1e9, 0)] * (K - len(top))
            top_ids.append([t for t, _, _ in top]); top_lp.append([l for _, l, _ in top])
        if ci % 10 == 0: print(f"  bloc {ci+1}/{len(chunks)}  {time.time()-t0:.0f}s  ppl courante {np.exp(-np.mean(lp_true)):.4f}", flush=True)
    np.savez_compressed(f"{D}/{name}.npz", lp_true=np.array(lp_true, np.float32), top_ids=np.array(top_ids, np.int32), top_lp=np.array(top_lp, np.float32))
    print(f"== {name}: {len(lp_true)} positions, perplexité {np.exp(-np.mean(lp_true)):.4f}, {time.time()-t0:.0f}s")
elif cmd == "compare":
    ref = np.load(f"{D}/{sys.argv[2]}.npz"); pr = np.exp(-ref["lp_true"].mean())
    print(f"{'config':22s} {'perplexité':>10s} {'Δppl':>8s} {'KL moy':>9s} {'KL p99':>8s} {'top-1 ≠':>8s} {'|Δlogp| moy':>11s}")
    print(f"{sys.argv[2]:22s} {pr:10.4f} {'réf.':>8s}")
    for name in sys.argv[3:]:
        o = np.load(f"{D}/{name}.npz"); po = np.exp(-o["lp_true"].mean())
        A_id, A_lp, B_id, B_lp = ref["top_ids"], ref["top_lp"].astype(np.float64), o["top_ids"], o["top_lp"].astype(np.float64)
        # q sur le support top-20 de la réf. ; token absent du top-20 de l autre -> son 20e logprob (majorant de q, donc KL minorée)
        match = (A_id[:, :, None] == B_id[:, None, :]); has = match.any(2); idx = match.argmax(2)
        q = np.where(has, np.take_along_axis(B_lp, idx, 1), B_lp[:, -1:]); p = A_lp
        P = np.exp(p); Q = np.exp(q); pt = np.clip(1 - P.sum(1), 1e-9, 1); qt = np.clip(1 - Q.sum(1), 1e-9, 1)
        kl = (P * (p - q)).sum(1) + pt * (np.log(pt) - np.log(qt)); kl = np.clip(kl, 0, None)
        dis = (A_id[:, 0] != B_id[:, 0]).mean(); dl = np.abs(ref["lp_true"] - o["lp_true"]).mean()
        print(f"{name:22s} {po:10.4f} {100*(po/pr-1):+7.2f}% {kl.mean():9.5f} {np.percentile(kl,99):8.4f} {100*dis:7.2f}% {dl:11.4f}")
