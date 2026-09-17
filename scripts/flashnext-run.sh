#!/bin/bash
# Qwen3.8-Flash-Next UD-Q4_K_XL — llama.cpp upstream HIP, 3 GPU en split LAYER (arch qwen4exp : -sm tensor refuse)
# Table n-gram 51B laissee sur CPU automatiquement (mmap) : ne PAS ajouter -ot ni --fit (casse le pipeline / OOM)
# Usage: flashnext-run.sh [ctx=131072] [np=1] [ub=512] [ts=15,16,17]   -> port 8085, log /root/flashnext-<ctx>.log
#   env BIN=<llama-server> (defaut /root/llama.cpp/build-hip/bin), KV=q8_0|f16 (defaut q8_0), FA=on|off (defaut on), EXTRA="options supplementaires", LLAMA_QSA_POOL_CACHE=0 (cache cles poolees off)
N=${1:-131072}; NP=${2:-1}; UB=${3:-512}; TS=${4:-15,16,17}; KV=${KV:-q8_0}; FA=${FA:-on}; EXTRA=${EXTRA:-}
M=/root/models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
pkill -x llama-server; sleep 3
BIN=${BIN:-/root/llama.cpp/build-hip/bin/llama-server}
HSA_NO_SCRATCH_RECLAIM=1 setsid nohup $BIN -m $M \
  -ngl 99 -sm layer -ts $TS --fit off -fa $FA -ctk $KV -ctv $KV -c $N -b 2048 -ub $UB -np $NP \
  --jinja $EXTRA --host 127.0.0.1 --port 8085 > /root/flashnext-$N.log 2>&1 < /dev/null & disown
for i in $(seq 1 150); do
    curl -s -m 2 http://127.0.0.1:8085/health 2>/dev/null | grep -q ok && { echo "UP apres $((i*4)) s (KV=$KV FA=$FA ctx=$N ub=$UB np=$NP)"; exit 0; }
    pgrep -x llama-server >/dev/null || { echo "MORT AU CHARGEMENT"; grep -iE "error|out of memory|APERTURE|abort" /root/flashnext-$N.log | tail -3; exit 1; }
    sleep 4
done
echo "TIMEOUT chargement"; exit 1
