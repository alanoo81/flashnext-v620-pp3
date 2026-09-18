#!/bin/bash
# Montée en fréquence VRAM au-delà de 1075 MHz (module amdgpu v2 : plafond OD élargi à 1250).
# memtest_vulkan (CT 100), une carte à la fois. memtest: 1=63:00 (card0), 2=46:00 (card2), 3=43:00 (card1)
declare -A CARD=( [1]=0 [2]=2 [3]=1 )
declare -A OKF LASTBW STOP
GPU_EVT='ring .* timeout|GPU reset|device wedged|SMU: No response|VRAM is lost|Failed to (disable gfxoff|export SMU)'
DMESG0=$(dmesg | wc -l)
# Après un hang/reset GPU : NE PLUS RIEN ÉCRIRE dans pp_od_clk_voltage (SMU wedge du 18/09) et tout arrêter.
gpu_event() { dmesg | tail -n +$((DMESG0+1)) | grep -qiE "$GPU_EVT"; }
abort_if_gpu_event() { if gpu_event; then echo "!!! événement amdgpu dans dmesg : ARRÊT COMPLET, aucune écriture OD. Reboot hôte recommandé avant tout nouveau test."; dmesg | tail -n +$((DMESG0+1)) | grep -iE "$GPU_EVT" | tail -4 | cut -c1-160; exit 1; fi; }
STEPS="${STEPS:-1075 1100 1125 1150 1175 1200}"
setclk() { d=/sys/class/drm/card$1/device; echo "m 1 $2" > $d/pp_od_clk_voltage 2>/dev/null && echo c > $d/pp_od_clk_voltage 2>/dev/null; }
getclk() { grep -A2 OD_MCLK /sys/class/drm/card$1/device/pp_od_clk_voltage | tail -1 | grep -o '[0-9]*M[Hh]z'; }
temps() { h=$(ls -d /sys/class/drm/card$1/device/hwmon/hwmon*); echo "junc$(( $(cat $h/temp2_input)/1000 ))/mem$(( $(cat $h/temp3_input)/1000 ))°C $(( $(cat $h/power1_average 2>/dev/null || echo 0)/1000000 ))W"; }
# run <durée> <index memtest> → positionne RC (0 ok / 1 erreur / 2 pas d'itération), BW (GB/s écriture, entier)
run() {
  T=$1; i=$2
  pct exec 100 -- python3 /root/memtest_vulkan/mtv.py $i $T /root/memtest_vulkan/d$i/run.log   # pseudo-terminal : seul moyen fiable de choisir la carte
  sleep 2
  BUS=$(pct exec 100 -- bash -c "grep -a 'test of' /root/memtest_vulkan/d$i/run.log | grep -o 'Bus=0x[0-9A-Fa-f:]*' | head -1")
  E=$(pct exec 100 -- bash -c "grep -aciE 'error|failed|mismatch' /root/memtest_vulkan/d$i/run.log")
  N=$(pct exec 100 -- bash -c "grep -ac 'iteration. Passed' /root/memtest_vulkan/d$i/run.log")
  BW=$(pct exec 100 -- bash -c "grep -a 'iteration. Passed' /root/memtest_vulkan/d$i/run.log | tail -1 | grep -o 'written:[^G]*GB *[0-9.]*GB/sec' | grep -o '[0-9.]*GB/sec' | cut -d. -f1")
  BW=${BW:-0}
  if [ "$N" -lt 1 ]; then RC=2; elif [ "$E" != 0 ]; then RC=1; else RC=0; fi
  [ "$RC" != 0 ] && pct exec 100 -- bash -c "grep -aiE 'error|failed|mismatch' /root/memtest_vulkan/d$i/run.log | head -2 | tr -d '\r' | cut -c1-150"
}
echo "=== $(date +%H:%M:%S) plafond OD: $(grep -A3 OD_RANGE /sys/class/drm/card0/device/pp_od_clk_voltage | grep MCLK)"
for i in 1 2 3; do OKF[$i]=1000; LASTBW[$i]=0; STOP[$i]=0; done
for F in $STEPS; do
  for i in 1 2 3; do
    [ "${STOP[$i]}" = 1 ] && continue
    c=${CARD[$i]}
    if ! setclk $c $F; then echo "card$c $F MHz : REFUSÉ par le pilote/SMU"; STOP[$i]=1; setclk $c ${OKF[$i]}; continue; fi
    run 90 $i
    echo "card$c [$BUS] $F MHz (lu: $(getclk $c)) : rc=$RC écriture=${BW} GB/s  $(temps $c)"
    abort_if_gpu_event
    if [ "$RC" != 0 ]; then echo "   -> ERREUR memtest à $F sur card$c (dernier palier propre: ${OKF[$i]}) : ARRÊT COMPLET sans réécriture de la carte."; exit 1;
    elif [ "$BW" -lt $(( ${LASTBW[$i]} - 2 )) ]; then echo "   -> débit en baisse (${LASTBW[$i]} → $BW GB/s : retransmissions EDR ?), arrêt de la montée pour card$c"; STOP[$i]=1; setclk $c ${OKF[$i]};
    else OKF[$i]=$F; LASTBW[$i]=$BW; fi
  done
  abort_if_gpu_event
done
echo "=== $(date +%H:%M:%S) validation longue (5 min 30) au dernier palier propre, descente par 10 MHz si erreur"
for i in 1 2 3; do
  c=${CARD[$i]}; F=${OKF[$i]}
  while [ "$F" -ge 1000 ]; do
    abort_if_gpu_event; setclk $c $F; run 330 $i; abort_if_gpu_event
    echo "card$c [$BUS] $F MHz 5min30 : rc=$RC écriture=${BW} GB/s  $(temps $c)"
    [ "$RC" = 0 ] && break
    F=$((F-10))
  done
  OKF[$i]=$F; echo "RESULTAT card$c : $F MHz"
done
echo "=== FIN MONTEE $(date +%H:%M:%S)"
