#!/bin/bash
# leapdragon + port MoE HIP RDNA2 (MOE_HIP=1) : cudagraphs, cache OFF, PP3 — correction + perf + rafale
cd /root
until grep -q "=== FIN" /root/test-leap-prefill2.log; do sleep 20; done; sleep 5
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
echo "=== $(date +%H:%M:%S) leap-cg + MoE HIP (MOE_HIP=1 TUNEOP=1 CG_SIZES 1..256)"
MOE_HIP=1 TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=65536 NOPC="--no-enable-prefix-caching" ./vllm-pp3.sh start >/dev/null 2>&1
waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|error|Traceback" -A3 | head -30 | cut -c1-220; docker logs fn-pp3 > /root/test-leap-moehip-fail.log 2>&1; ./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"; exit 1; }
docker logs fn-pp3 2>&1 | grep -E "RDNA2MoEMethod|WNA16 MoE backend|Using configuration from|Config file not found|TunableOp|v620" | sed "s/(Worker[^)]*) //" | sort | uniq -c | cut -c1-200 | head -8
python3 /root/sweep.py 4336,1614 0 120 8086 --show | cut -c1-160
python3 /root/perf3.py 8086 vllm 4096,16384 300 --reps 1 --tag leap-cg-moehip | grep -E "^== "
python3 /root/burst.py 8086 4 --tag leap-cg-moehip | grep -E "rafale|après"
./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"
