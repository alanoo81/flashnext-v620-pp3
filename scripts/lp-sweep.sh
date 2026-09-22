#!/bin/bash
# Effet de --long-prefill-token-threshold sur l équité prefill/décodage (profil Agentique + vision).
cd /root; MB=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16
waitup() { for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
for LP in ${LPS:-0 1024 512 256}; do
  echo "=== $(date +%H:%M:%S) LONGPREFILL=$LP"
  ./vllm-pp3.sh stop >/dev/null 2>&1; sleep 5
  env LONGPREFILL=$LP VISION=1 VISION_IMAGES=1 VISION_MAX_PIXELS=401408 DEVS=1,2,0 SEQS=3 MODEL=$MB \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 MOE_HIP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" \
    IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 \
    MOE_PADDING=0 PART=16,16,16 CTX=229376 ./vllm-pp3.sh start >/dev/null 2>&1
  if ! waitup; then echo "MORT : $(docker logs fn-pp3 2>&1 | grep -E 'ValueError|Error:' | tail -1 | cut -c1-140)"; continue; fi
  docker inspect fn-pp3 --format '{{.Args}}' | grep -o "long-prefill-token-threshold [0-9]*" || echo "   (option absente de la ligne de commande)"
  python3 /root/sweep.py 4336 0 20 8086 >/dev/null 2>&1
  python3 /root/prefill-fair.py 8086 --long 100000 --tag lp$LP
done
./vllm-pp3.sh stop >/dev/null 2>&1
echo "=== FIN LP $(date +%H:%M:%S)"
