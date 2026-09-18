#!/bin/bash
# Profil torch d un décodage mono-flux, leapdragon + MoE HIP + cudagraphs, SANS MTP (51 t/s = 19,6 ms/pas) : où vont les ms ?
# usage: test-prof-dec-leap.sh [TUNEOP=1|pp3]   -> traces dans /root/vllm-cache/image/prof-dec
cd /root
T=${1:-1}
waitup() { for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && return 0; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || return 1; sleep 5; done; return 1; }
P=/root/vllm-cache/image/prof-dec; rm -rf $P; mkdir -p $P
echo "=== $(date +%H:%M:%S) profil décodage c=1 sans MTP (TUNEOP=$T)"
env MOE_HIP=1 TUNEOP=$T CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=65536 NOPC=--no-enable-prefix-caching EXTRA="--kv-cache-memory-bytes 3500000000 --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/prof-dec\"}" ./vllm-pp3.sh start >/dev/null 2>&1
waitup || { echo MORT; docker logs fn-pp3 2>&1 | grep -E "Error|Traceback" | tail -5 | cut -c1-200; ./vllm-pp3.sh stop >/dev/null 2>&1; exit 1; }
python3 /root/perf3.py 8086 vllm 512 300 --reps 1 --tag ref | grep -E "^== |GEN="
curl -s -X POST http://127.0.0.1:8086/start_profile -o /dev/null -w "start_profile=%{http_code}\n"
python3 /root/perf3.py 8086 vllm 512 160 --reps 1 --tag prof | grep -E "GEN="
curl -s -m 600 -X POST http://127.0.0.1:8086/stop_profile -o /dev/null -w "stop_profile=%{http_code}\n"
./vllm-pp3.sh stop >/dev/null 2>&1; sleep 3; ls -la $P | head -6
/root/vllm-native-venv/bin/python /root/decstep.py $P
echo "=== FIN $(date +%H:%M:%S)"
