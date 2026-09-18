#!/bin/bash
# Série de relevés de qualité : une config par démarrage, deux relevés par config (plancher de bruit propre à chaque config).
# usage: qual-series.sh "<nom> VAR=val ..." ...
cd /root; P=/root/vllm-native-venv/bin/python
for spec in "$@"; do
  set -- $spec; name=$1; shift
  echo "=== $(date +%H:%M:%S) $name : $*"
  env MOE_HIP=1 DENSE_INT8=1 DENSE_INT8_ONLY=1 "$@" CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image MOE_PADDING=0 PART=17,18,13 CTX=8192 NOPC=--no-enable-prefix-caching EXTRA="--kv-cache-memory-bytes 1500000000 --max-logprobs 20" ./vllm-pp3.sh start >/dev/null 2>&1
  up=0; for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && { up=1; break; }; docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || break; sleep 5; done
  if [ $up = 1 ]; then
    $P /root/qual.py collect 8086 $name | tail -1; [ -n "$NOREP" ] || $P /root/qual.py collect 8086 $name-rep | tail -1
    docker logs fn-pp3 2>&1 | grep -m2 FAKEQ | cut -c1-200
  else echo "MORT $name"; docker logs fn-pp3 2>&1 | grep -E "Error|Traceback|ERROR" | tail -4 | cut -c1-220; fi
  ./vllm-pp3.sh stop >/dev/null 2>&1
done
echo "=== FIN SERIE $(date +%H:%M:%S)"
