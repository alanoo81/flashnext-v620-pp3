#!/bin/bash
# Soak 30 min de la config complète : leapdragon + MoE HIP + cudagraphs + MTP k=2 (drafter W4A16) + prefix caching, ctx 131K.
# stab.py = tailles aléatoires 512-16K + rafale c=4 toutes les 6 itérations, classifieur de sorties. Pas de mesure de prefill ici
# (cache ON + prompts à préfixe commun = chiffres biaisés, cf. 18/09).
cd /root
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
echo "=== $(date +%H:%M:%S) soak : démarrage du serveur (ctx 131072, MTP k=2 drafter W4A16, cache ON)"
env MODEL=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16-mtpq VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 MOE_HIP=1 TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=131072 EXTRA="--kv-cache-memory-bytes 3500000000 --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" ./vllm-pp3.sh start >/dev/null 2>&1
waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|Traceback" | tail -5 | cut -c1-200; ./vllm-pp3.sh stop >/dev/null 2>&1; exit 1; }
docker logs fn-pp3 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1
echo "=== $(date +%H:%M:%S) stab.py 30 min"
python3 /root/stab.py 8086 vllm 30 --tag soak-best | grep -E "SUSPECT|^==|!!!" | tail -12
echo "=== $(date +%H:%M:%S) rafale finale"
python3 /root/burst.py 8086 4 --tag soak-best | grep -E "rafale|après"
echo "erreurs réelles dans le log serveur (hors avertissements transformers/default_loader): $(docker logs fn-pp3 2>&1 | grep -iE "error|traceback|hang|fault" | grep -viE "rope_parameters|default_loader|Unrecognized keys" | wc -l)"
./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN SOAK $(date +%H:%M:%S)"
