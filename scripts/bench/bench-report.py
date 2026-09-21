#!/usr/bin/env python3
"""Compare un journal de bench.sh aux chiffres publiés (RESULTS §0 du dépôt, 18 sept. 2026, 160 W, VRAM 1 075 MHz).
usage: bench-report.py <journal>"""
import re, sys
REF = {  # étiquette -> (mesure, valeur publiée, tolérance relative)
    ("nomtp-c1", "gen", 512):        ("décodage 1 flux sans MTP (jetons/s)", 50.9, 0.05),
    ("mtpbf16-c1", "gen", 4096):     ("décodage 1 flux MTP k=2, drafter bf16, prompt 4K", 62.8, 0.08),
    ("mtpbf16-c1", "gen", 16384):    ("… prompt 16K", 61.8, 0.08),
    ("mtpq-c1", "gen", 4096):        ("décodage 1 flux MTP k=2, drafter W4A16, prompt 4K", 66.2, 0.08),
    ("mtpq-c1", "gen", 16384):       ("… prompt 16K", 67.9, 0.08),
    ("mtpq-prefill", "pp", 4096):    ("prefill 4K, cache OFF (jetons/s)", 1152, 0.06),
    ("mtpq-prefill", "pp", 16384):   ("prefill 16K, cache OFF", 1705, 0.06),
    ("mtpq-prefill", "pp", 65536):   ("prefill 65K, cache OFF", 1936, 0.06),
    ("mtpq-prefill", "pp", 130000):  ("prefill 130K, cache OFF", 1933, 0.06),
    ("nomtp-c4", "conc", 4):         ("4 flux simultanés sans MTP : décodage chevauché (jetons/s)", 139.5, 0.15),
    ("nomtp-c8", "conc", 8):         ("8 flux simultanés sans MTP : décodage chevauché", 249, 0.10),
    ("mtpq-c4s", "conc", 4):         ("4 flux décalés, MTP W4A16 : décodage chevauché", 173, 0.10),
    ("mtpq-c8s", "conc", 8):         ("8 flux décalés, MTP W4A16 : décodage chevauché", 112, 0.15),
}
WALL = {("nomtp-c4", 4): 114, ("nomtp-c8", 8): 191, ("mtpq-c4s", 4): 125, ("mtpq-c8s", 8): 90}
got, extra = {}, []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.match(r"^== (\S+)\s+N=\s*(\d+) PP=\s*([\d.]+) GEN=\s*([\d.]+)", line)
    if m:
        tag, n, pp, gen = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        n = min(REF, key=lambda k: abs(k[2] - n) if k[0] == tag else 1e9)[2] if any(k[0] == tag for k in REF) else n
        got[(tag, "pp", n)] = pp; got[(tag, "gen", n)] = gen
    m = re.match(r"^== (\S+)\s+CONC c=(\d+).*?= ([\d.]+) t/s agrégés.*?chevauché ≈ ([\d.]+)", line)
    if m:
        tag, c = m.group(1), int(m.group(2)); got[(tag, "conc", c)] = float(m.group(4)); got[(tag, "wall", c)] = float(m.group(3))
    m = re.match(r"^== (\S+): (\d+) positions, perplexité ([\d.]+)", line)
    if m:
        extra.append(("perplexité de la configuration servie (corpus fixe, 81 880 positions)", float(m.group(3)), 4.022, 0.01))
    m = re.search(r"stabilité: (\d+) itérations, (\d+) ok, (\d+) suspects, (\d+) erreurs", line)
    if m:
        extra.append(("soak 30 min : sorties correctes / erreurs", f"{m.group(2)} ok, {m.group(4)} erreurs (suspects {m.group(3)} = continuations du corpus)", "223 ok, 0 erreur", None))
print("\n| mesure | ce run | publié (18 sept.) | écart |\n|---|---|---|---|")
verdict_ok = True
for key, (label, ref, tol) in REF.items():
    if key not in got:
        continue
    v = got[key]; d = v / ref - 1
    flag = "✅" if abs(d) <= tol else ("⚠️" if abs(d) <= 2 * tol else "❌")
    if flag != "✅": verdict_ok = False
    wall = f" ({got.get((key[0], 'wall', key[2]), 0):.0f} wall, publié {WALL.get((key[0], key[2]), '?')})" if key[1] == "conc" else ""
    print(f"| {label} | **{v:g}**{wall} | {ref:g} | {d:+.1%} {flag} |")
for label, v, ref, tol in extra:
    if tol is None:
        print(f"| {label} | {v} | {ref} | — |")
    else:
        d = v / ref - 1; flag = "✅" if abs(d) <= tol else "❌"; verdict_ok &= flag == "✅"
        print(f"| {label} | **{v:.4f}** | {ref} | {d:+.2%} {flag} |")
if not got and not extra:
    print("| (aucune mesure trouvée dans le journal) | | | |")
print("\nTolérances : ±5-8 % en décodage mono-flux (une seule mesure par point, MTP sensible au texte), ±6 % en prefill, ±10-15 % en multi-flux (dépend de la répartition des batches en vol). ✅ dans la tolérance · ⚠️ jusqu'au double · ❌ au-delà.")
print("Verdict :", "conforme aux chiffres publiés" if verdict_ok else "au moins un écart hors tolérance — vérifier cap 160 W / VRAM 1075 MHz / rien d'autre sur les GPU")
