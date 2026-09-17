#!/bin/bash
# combiné leapdragon MoE HIP + cudagraphs + MTP k=2 + cache ON à 262K (KV fixe 3,5 Go = 263K tokens → 1 requête pleine)
cd /root
until grep -q "=== FIN" /root/test-leap-best-stab.log; do sleep 15; done; sleep 5
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
echo "=== $(date +%H:%M:%S) leap-best ctx 262144 (MoE HIP + graphs + MTP k=2 + cache ON, KV 3.5e9)"
MOE_HIP=1 TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=262144 EXTRA="--kv-cache-memory-bytes 3500000000 --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" ./vllm-pp3.sh start >/dev/null 2>&1
waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|error|Traceback|out of memory|OOM" -A2 | head -20 | cut -c1-220; docker logs fn-pp3 > /root/test-leap-best-262k-fail.log 2>&1; ./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"; exit 1; }
docker logs fn-pp3 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tail -2
rocm-smi --showmeminfo vram 2>/dev/null | grep "Used" | cut -c1-70
python3 /root/sweep.py 4336,1614 0 120 8086 --show | cut -c1-160
python3 /root/perf3.py 8086 vllm 131072,200000,261000 100 --reps 1 --tag leap-best-262k | grep -E "^== "
docker logs fn-pp3 2>&1 | grep -ciE "error|fault|hang"
./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"
