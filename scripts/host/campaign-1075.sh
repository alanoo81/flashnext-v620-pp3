#!/bin/bash
# Campagne de mesure à VRAM 1075 MHz : phase A (cap 160 W) puis phase B (cap 200 W), mêmes mesures.
# Hôte PVE ; les serveurs tournent dans le CT 100 (/root/leap-run-stag.sh). Journal : /root/campaign-1075.log
#
# Sécurité (leçon du 18/09) : après tout événement GPU (timeout d'anneau, reset, SMU muet), on arrête TOUT
# et on n'écrit plus rien — ni OverDrive ni power cap, qui passent tous deux par le SMU. Reboot hôte d'abord.
GPU_EVT='ring .* timeout|GPU reset|device wedged|SMU: No response|VRAM is lost|Failed to (disable gfxoff|export SMU)'
DMESG0=$(dmesg | wc -l)
GPU_BAD=0
MQ=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16-mtpq

gpu_event() { dmesg | tail -n +$((DMESG0+1)) | grep -qiE "$GPU_EVT"; }
check_gpu() {
  if gpu_event; then
    GPU_BAD=1
    echo "!!! ÉVÉNEMENT GPU dans dmesg : arrêt complet, AUCUNE écriture (cap laissé tel quel). Reboot hôte recommandé."
    dmesg | tail -n +$((DMESG0+1)) | grep -iE "$GPU_EVT" | tail -4 | cut -c1-160
    pct exec 100 -- /root/vllm-pp3.sh stop >/dev/null 2>&1
    exit 2
  fi
}
setcap() {
  check_gpu
  for h in /sys/class/drm/card[0-2]/device/hwmon/hwmon*; do echo $(( $1 * 1000000 )) > $h/power1_cap || { echo "!!! échec écriture cap $1 W sur $h"; exit 3; }; done
  echo "--- cap réglé : $(for c in 0 1 2; do echo -n "card$c=$(( $(cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_cap)/1000000 ))W "; done)"
}
check_mclk() {
  for c in 0 1 2; do
    m=$(timeout 5 grep -A2 '^OD_MCLK:' /sys/class/drm/card$c/device/pp_od_clk_voltage | tail -1 | awk '{print $2}')
    [ "$m" = "1075MHz" ] || { echo "!!! card$c : mclk max = '$m' au lieu de 1075MHz, arrêt"; exit 4; }
  done
  echo "--- mclk max vérifié : 1075MHz sur les 3 cartes, offset $(grep -A1 '^OD_VDDGFX_OFFSET:' /sys/class/drm/card0/device/pp_od_clk_voltage | tail -1)"
}
# Retour à 160 W en sortie, SAUF si un événement GPU a été vu (ne rien écrire sur un SMU douteux).
cleanup() {
  if [ "$GPU_BAD" = 0 ] && ! gpu_event; then
    for h in /sys/class/drm/card[0-2]/device/hwmon/hwmon*; do echo 160000000 > $h/power1_cap 2>/dev/null; done
    echo "--- sortie : cap remis à 160 W"
  else
    echo "--- sortie sur événement GPU : cap NON modifié"
  fi
}
trap cleanup EXIT

run() { # run <tag> VAR=val ...   (variables passées à leap-run-stag.sh dans le CT)
  local tag=$1; shift
  pct exec 100 -- env "$@" /root/leap-run-stag.sh "$tag"
  check_gpu
}

phase() { # phase <étiquette> <watts>
  local L=$1 W=$2
  echo "######## PHASE $L : cap $W W, VRAM 1075 MHz — $(date +%H:%M:%S)"
  check_mclk; setcap $W
  run nomtp-c4-$L      K=0 SIZES=512 GEN=300 CONC=4
  run nomtp-c8-$L      K=0 SIZES=512 GEN=300 CONC=8 SEQS=8
  run mtpbf16-c1-$L    K=2 SIZES=4096,16384 GEN=300
  run mtpq-prefill-$L  K=2 MODEL=$MQ CTX=131072 NOPC=--no-enable-prefix-caching SIZES=4096,16384,65536,130000 GEN=200
  run mtpq-c4s-$L      K=2 MODEL=$MQ SIZES=512 GEN=300 CONC=4 STAGGER=0.7
  run mtpq-c8s-$L      K=2 MODEL=$MQ SIZES=512 GEN=300 CONC=8 STAGGER=0.7 SEQS=8
  echo "######## FIN PHASE $L — $(date +%H:%M:%S)"
}

echo "######## ÉCHAUFFEMENT (cache disque froid après reboot, résultats à ignorer) — $(date +%H:%M:%S)"
check_mclk; setcap 160
run warmup K=2 MODEL=$MQ SIZES=4096 GEN=100
phase A160 160
phase B200 200
setcap 160
echo "######## CAMPAGNE FINIE $(date +%H:%M:%S)"
