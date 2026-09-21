#!/bin/bash
# Enveloppe hôte du banc : vérifie qu'aucun événement GPU n'est dans dmesg avant/après, lance /root/bench/bench.sh dans le CT 100.
# usage : run-bench.sh [quick|decode|conc|prefill|full|quality|soak]   (journal : /root/bench/results/ dans le CT)
GPU_EVT='ring .* timeout|GPU reset|device wedged|SMU: No response|VRAM is lost'
D0=$(dmesg | wc -l)
dmesg | grep -qiE "$GPU_EVT" && { echo "!! événement GPU déjà présent dans dmesg : redémarrer l'hôte avant de mesurer (règle SMU)"; exit 2; }
pct status 100 | grep -q running || { echo "!! CT 100 arrêté"; exit 2; }
pct exec 100 -- /root/bench/bench.sh "${1:-quick}"
dmesg | tail -n +$((D0+1)) | grep -iE "$GPU_EVT" && echo "!! événement GPU pendant le banc : résultats à ignorer, redémarrer l'hôte" || echo "(aucun événement GPU pendant le banc)"
