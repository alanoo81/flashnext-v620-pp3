#!/bin/bash
# leapdragon + MoE HIP + cudagraphs + MTP k=2 + prefix caching ON : le candidat "meilleur des deux mondes"
cd /root
until grep -q "=== FIN" /root/test-leap-moehip.log; do sleep 15; done; sleep 5
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
echo "=== $(date +%H:%M:%S) leap-cg + MoE HIP + MTP k=2 + cache ON (TUNEOP, CG 1..256, PART 17,18,13, KV 3.5e9, ctx 65536)"
MOE_HIP=1 TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=65536 EXTRA="--kv-cache-memory-bytes 3500000000 --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" ./vllm-pp3.sh start >/dev/null 2>&1
waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|error|Traceback" -A3 | head -30 | cut -c1-220; docker logs fn-pp3 > /root/test-leap-best-fail.log 2>&1; ./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"; exit 1; }
docker logs fn-pp3 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1
python3 /root/sweep.py 4336,1614 0 120 8086 --show | cut -c1-160
python3 /root/perf3.py 8086 vllm 4096,16384 300 --reps 1 --tag leap-best-mtp2 | grep -E "^== "
docker logs fn-pp3 2>&1 | grep -oE "Draft acceptance rate: [0-9.]+%|Mean acceptance length: [0-9.]+" | tail -2 | tr "\n" " "; echo
python3 /root/perf3.py 8086 vllm 65536 100 --reps 1 --tag leap-best-mtp2-pp | grep -E "^== "
python3 /root/burst.py 8086 4 --tag leap-best-mtp2 | grep -E "rafale|après"
python3 /root/cachetest3.py 8086 30000 | grep "cache-test"
./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"
