#!/bin/bash
# MTP PP3 sur rdna_extras (worktree v620-mtp-rdna), eager, cache OFF, KV explicite
cd /root
waitup() { for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; pgrep -f "openai.api_serve[r]" >/dev/null || return 1; sleep 5; done; return 1; }
for k in 2 0; do
  if [ "$k" = 0 ]; then SPEC=""; LBL="sans MTP (référence V2)"; else SPEC="$SPEC"; LBL="MTP k=$k"; fi
  echo "=== $(date +%H:%M:%S) rdna_extras $LBL eager, PART 18,17,13, KV 1.5e9, runner V2, cache OFF"
  TREE=/root/vllm-rdna-mtp CTX=65536 PART=18,17,13 VLLM_ROCM_MOE_PADDING=0 VLLM_USE_V2_MODEL_RUNNER=1 NOPC="--no-enable-prefix-caching" EXTRA="--kv-cache-memory-bytes 1500000000 --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$k}" ./restart-ref.sh
  waitup || { echo MORT; cp /root/vllm-native.log /root/test-mtp-rdna-k$k.log; grep -nE "Error:|Exception|assert|out of memory" /root/vllm-native.log | grep -vE "WARNING|initialization failed|failed with error" | head -4 | cut -c1-240; continue; }
  grep -oE "GPU KV cache size: [0-9,]+ tokens" /root/vllm-native.log | tail -1
  python3 /root/sweep.py 4336,1614 0 120 8086 --show | cut -c1-220
  python3 /root/perf3.py 8086 vllm 4096,16384 300 --reps 1 --tag rdna-mtp$k | grep -E "^== "
  grep -oE "Draft acceptance rate: [0-9.]+%|Mean acceptance length: [0-9.]+" /root/vllm-native.log | tail -2 | tr "\n" " "; echo
  python3 /root/burst.py 8086 4 --tag rdna-mtp$k | grep -E "rafale|après"
done
./vllm-native-pp3.sh stop >/dev/null; pkill -9 -f "VLLM:[:]"; pkill -9 -f "multiprocessing.spaw[n]"; echo "=== FIN $(date +%H:%M:%S)"
