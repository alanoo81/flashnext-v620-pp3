#!/bin/bash
# Tuning TunableOp (rocBLAS) pour les formes de GEMM de notre PP3/TP1 : les rangées livrées par leapdragon ont été
# réglées en TP4 (matrices denses coupées en 4). On amorce avec leurs rangées, on laisse PyTorch régler toute forme
# nouvelle, on pousse des longueurs de prompt variées, puis arrêt propre (les rangées s écrivent à la sortie).
# Usage : tune-pp3.sh   (CT 100, ~30-45 min). Résultat : /root/vllm-cache/image/tunableop/pp3/tunableop_results{0,1,2}.csv
cd /root
IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b
D=/root/vllm-cache/image/tunableop/pp3
mkdir -p $D
if [ ! -s $D/tunableop_results0.csv ]; then
  docker run --rm --entrypoint bash -v $D:/out $IMG -c 'cp /app/vllm/tunableop/rocblas-9847aecc4bf8/tunableop_results[012].csv /out/'
fi
echo "=== $(date +%H:%M:%S) rangées de départ : $(wc -l < $D/tunableop_results0.csv) / $(wc -l < $D/tunableop_results1.csv) / $(wc -l < $D/tunableop_results2.csv)"
cp $D/tunableop_results0.csv $D/seed0.csv; cp $D/tunableop_results1.csv $D/seed1.csv; cp $D/tunableop_results2.csv $D/seed2.csv
env MODEL=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16-mtpq VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 MOE_HIP=1 TUNEOP=tune CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=$IMG TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=65536 NOPC=--no-enable-prefix-caching EXTRA="--kv-cache-memory-bytes 3500000000 --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" ./vllm-pp3.sh start >/dev/null 2>&1
up=0
for i in $(seq 1 720); do
  curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && { up=1; break; }
  docker ps --format "{{.Names}}" | grep -q "^fn-pp3$" || break
  sleep 5
done
[ $up = 1 ] || { echo "MORT au démarrage"; docker logs fn-pp3 2>&1 | grep -E "Error|Traceback|ERROR" | tail -8 | cut -c1-220; ./vllm-pp3.sh stop >/dev/null 2>&1; exit 1; }
echo "=== $(date +%H:%M:%S) serveur prêt, envoi des longueurs"
k=0
for n in 74 150 300 450 600 800 1000 1300 1600 1900 2048 2500 3000 4336 6000 8192 12000 16384 22883; do
  k=$((k+1))
  python3 /root/sweep.py $n $((k*7001)) 24 2>&1 | tail -1 | cut -c1-110
done
echo "=== $(date +%H:%M:%S) arrêt propre"
./vllm-pp3.sh stop >/dev/null 2>&1
for r in 0 1 2; do echo "rank $r : $(wc -l < $D/seed$r.csv) -> $(wc -l < $D/tunableop_results$r.csv) rangées ; nouvelles : $(comm -13 <(cut -d, -f1,2 $D/seed$r.csv | sort) <(cut -d, -f1,2 $D/tunableop_results$r.csv | sort) | wc -l)"; done
echo "=== FIN TUNE $(date +%H:%M:%S)"
