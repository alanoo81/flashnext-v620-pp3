#!/bin/bash
# leapdragon image 0915, cudagraphs, cache OFF, PP3 : effet du config MoE E=512 (num_stages=1) + TunableOp lookup + tailles de capture
cd /root
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
run() { # $1 tag ; env TUNEOP/CG_SIZES hérité de l appelant
  echo "=== $(date +%H:%M:%S) $1 [TUNEOP=${TUNEOP:-0} CG_SIZES=${CG_SIZES:-}]"
  IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=65536 NOPC="--no-enable-prefix-caching" ./vllm-pp3.sh start >/dev/null 2>&1
  waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|error" | head -4 | cut -c1-200; ./vllm-pp3.sh stop >/dev/null 2>&1; return; }
  docker logs fn-pp3 2>&1 | grep -E "Using configuration from|Config file not found|TunableOp|tunableop|Capturing CUDA graphs \(PIECEWISE\): 100" | sed "s/(Worker[^)]*) //" | sort | uniq -c | cut -c1-200 | head -6
  python3 /root/sweep.py 4336,1614 0 120 8086 --show | cut -c1-160
  python3 /root/perf3.py 8086 vllm 4096,16384 300 --reps 1 --tag "$1" | grep -E "^== "
  ./vllm-pp3.sh stop >/dev/null 2>&1; sleep 5
}
run "leap-cg-moecfg"
TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" run "leap-cg-moecfg-tuneop-cg256"
echo "=== FIN $(date +%H:%M:%S)"
