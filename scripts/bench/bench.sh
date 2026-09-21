#!/bin/bash
# Banc de reproduction des chiffres publiés (RESULTS §0, 18 sept. 2026 : VRAM 1 075 MHz, cap 160 W, −40 mV).
# À lancer DANS le CT 100 (ou via /root/bench/run-bench.sh depuis l'hôte, qui ajoute la surveillance dmesg).
# Usage : bench.sh [quick|decode|conc|prefill|full|quality|soak]
#   quick    ~15 min : décodage 1 flux sans MTP, 1 flux MTP W4A16, prefill 4K/16K cache OFF
#   decode   ~10 min : 1 flux sans MTP, MTP bf16, MTP W4A16 (4K et 16K)
#   conc     ~25 min : 4 et 8 flux sans MTP ; 4 et 8 flux MTP W4A16 (arrivées décalées de 0,7 s)
#   prefill  ~12 min : 4K / 16K / 65K / 130K, cache OFF, ctx 131K, MTP W4A16
#   full     ~45 min : decode + conc + prefill
#   quality  ~10 min : perplexité de la configuration servie sur le corpus fixe (réf. 4,02 ± 0,01)
#   soak     ~35 min : stabilité 30 min de la configuration de production (0 sortie corrompue attendu)
# Résultats : /root/bench/results/<date>-<mode>.log puis tableau comparatif (bench-report.py).
set -u
cd /root
MODE=${1:-quick}
MB=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16
MQ=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16-mtpq
OUT=/root/bench/results; mkdir -p $OUT
LOG=$OUT/$(date +%Y%m%d-%H%M)-$MODE.log
run() { local tag=$1; shift; env "$@" /root/leap-run-stag.sh "$tag" 2>&1 | tee -a $LOG | grep -E "^=== |^== |MORT" | cut -c1-200; }

# --- préconditions : mêmes réglages que les chiffres publiés ---------------------------------------------
pre() {
  local ok=1
  for c in 0 1 2; do
    cap=$(( $(cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1) / 1000000 ))
    mclk=$(grep -A2 '^OD_MCLK' /sys/class/drm/card$c/device/pp_od_clk_voltage 2>/dev/null | tail -1 | awk '{print $2}')
    printf 'card%s : cap %s W, VRAM max %s\n' $c "$cap" "${mclk:-inconnu}"
    [ "$cap" = 160 ] || { echo "  !! cap attendu 160 W (les chiffres publiés sont à 160 W ; 200 W = +13-17 % de prefill)"; ok=0; }
    [ "$mclk" = 1075MHz ] || { echo "  !! VRAM attendue 1075MHz (module amdgpu patché + gpu-undervolt) : à 1000 MHz compter −4 % en décodage"; ok=0; }
  done
  [ -d $MB ] && [ -d $MQ ] || { echo "!! modèles absents : $MB et $MQ"; exit 2; }
  docker image inspect ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b >/dev/null 2>&1 || { echo "!! image leapdragon 20260915 absente"; exit 2; }
  docker ps --format '{{.Names}}' | grep -q '^fn-pp3$' && { echo "!! un serveur fn-pp3 tourne déjà : ./vllm-pp3.sh stop d'abord"; exit 2; }
  pgrep -f "openai.api_serve[r]" >/dev/null && { echo "!! un serveur natif tourne déjà : ./vllm-native-pp3.sh stop d'abord"; exit 2; }
  [ $ok = 1 ] || echo "(préconditions non remplies : le banc tourne quand même, les écarts seront annotés)"
}
echo "######## BANC $MODE — $(date '+%F %T') — journal $LOG" | tee -a $LOG
pre | tee -a $LOG
[ "$(cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_cap | head -1)" = 160000000 ] && echo "cap=160" >> $LOG

case $MODE in
  quick)
    run nomtp-c1     K=0 SIZES=512 GEN=300
    run mtpq-c1      K=2 MODEL=$MQ SIZES=4096,16384 GEN=300
    run mtpq-prefill K=2 MODEL=$MQ CTX=131072 NOPC=--no-enable-prefix-caching SIZES=4096,16384 GEN=200 ;;
  decode)
    run nomtp-c1     K=0 SIZES=512 GEN=300
    run mtpbf16-c1   K=2 SIZES=4096,16384 GEN=300
    run mtpq-c1      K=2 MODEL=$MQ SIZES=4096,16384 GEN=300 ;;
  conc)
    run nomtp-c4     K=0 SIZES=512 GEN=300 CONC=4
    run nomtp-c8     K=0 SIZES=512 GEN=300 CONC=8 SEQS=8
    run mtpq-c4s     K=2 MODEL=$MQ SIZES=512 GEN=300 CONC=4 STAGGER=0.7
    run mtpq-c8s     K=2 MODEL=$MQ SIZES=512 GEN=300 CONC=8 STAGGER=0.7 SEQS=8 ;;
  prefill)
    run mtpq-prefill K=2 MODEL=$MQ CTX=131072 NOPC=--no-enable-prefix-caching SIZES=4096,16384,65536,130000 GEN=200 ;;
  full)
    $0 decode; $0 conc; $0 prefill; exit ;;
  quality)
    P=/root/vllm-native-venv/bin/python
    env MOE_HIP=1 CG_SIZES="1 2 4 8 16 32 64 128 256" IMG=ghcr.io/leapdragon/vllm-rdna2-qwen:20260915-g1bdbbef4b TREE=image DENSE_INT8=1 DENSE_INT8_ONLY=1 MOE_PADDING=0 PART=17,18,13 CTX=8192 NOPC=--no-enable-prefix-caching EXTRA="--kv-cache-memory-bytes 1500000000 --max-logprobs 20" ./vllm-pp3.sh start >/dev/null 2>&1
    for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && break; sleep 5; done
    [ -s /root/qual/tokens.json ] || $P /root/qual.py tokens 8086 40 2048 | tee -a $LOG
    $P /root/qual.py collect 8086 bench-$(date +%Y%m%d-%H%M) 2>&1 | tail -1 | tee -a $LOG
    ./vllm-pp3.sh stop >/dev/null 2>&1 ;;
  soak)
    /root/soak-best.sh 2>&1 | tee -a $LOG | grep -E "^==|^===|erreurs|SUSPECT" | cut -c1-160 ;;
  *) echo "mode inconnu : $MODE"; exit 1 ;;
esac
echo "######## FIN BANC $MODE — $(date +%T)" | tee -a $LOG
python3 /root/bench/bench-report.py $LOG
