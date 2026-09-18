#!/bin/bash
# Un run leapdragon combiné (MoE HIP + graphs + MTP) : TAG puis variables K (spec tokens, 0 = sans MTP), CTX, KV (octets), P2P, MNBT, SIZES, GEN, CONC
cd /root
TAG=$1; K=${K:-2}; CTX=${CTX:-65536}; KV=${KV:-3500000000}; SIZES=${SIZES:-4096,16384,32768,60000}; GEN=${GEN:-200}
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
SPEC=""; [ "$K" != 0 ] && SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$K}"
KVARG=""; [ "$KV" != auto ] && KVARG="--kv-cache-memory-bytes $KV"
echo "=== $(date +%H:%M:%S) $TAG | K=$K CTX=$CTX KV=$KV P2P=${P2P:-PHB} MNBT=${MNBT:-2048} PART=${PART:-17,18,13} cap=$(cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)"
env MOE_HIP=1 TUNEOP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=${PART:-17,18,13} CTX=$CTX P2P=${P2P:-PHB} MNBT=${MNBT:-2048} ${GPUUTIL:+GPUUTIL=$GPUUTIL} EXTRA="$KVARG $SPEC $EXTRA_ARGS" ./vllm-pp3.sh start > /root/leap-run-launch.log 2>&1
waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|error|Traceback|out of memory|OOM" -A2 | head -12 | cut -c1-200; docker logs fn-pp3 > /root/leap-run-$TAG-fail.log 2>&1; ./vllm-pp3.sh stop >/dev/null 2>&1; echo "=== FIN $(date +%H:%M:%S)"; exit 1; }
docker logs fn-pp3 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tail -2 | tr "\n" " "; echo
python3 /root/sweep.py 4336 0 60 8086 | cut -c1-100
python3 /root/perf3.py 8086 vllm $SIZES $GEN --reps 1 --tag $TAG ${CONC:+--conc $CONC} | grep -E "^== |flux"
[ "$K" != 0 ] && { docker logs fn-pp3 2>&1 | grep -oE "Draft acceptance rate: [0-9.]+%|Mean acceptance length: [0-9.]+" | tail -2 | tr "\n" " "; echo; }
rocm-smi --showpower 2>/dev/null | grep -oE "GPU\[[0-9]\].*Power \(W\): [0-9.]+" | sed "s/.*GPU\[\([0-9]\)\].*: /P\1=/" | tr "\n" " "; echo
./vllm-pp3.sh stop >/dev/null 2>&1; sleep 5; echo "=== FIN $(date +%H:%M:%S)"
