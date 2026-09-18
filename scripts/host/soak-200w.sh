#!/bin/bash
# Soak 30 min à cap 200 W / VRAM 1075 MHz, avec chien de garde thermique. Hôte PVE ; serveur dans le CT 100.
# Journal : /root/soak-200w.log ; relevé thermique : /root/soak-200w-temps.log
# Sécurité : après tout événement GPU, arrêt complet et AUCUNE écriture (OverDrive ni cap) — leçon du 18/09.
W=${1:-200}
J_MAX=95; M_MAX=90          # seuils d'arrêt (°C) ; limites pilote : jonction 100/105, mémoire 98/103
GPU_EVT='ring .* timeout|GPU reset|device wedged|SMU: No response|VRAM is lost|Failed to (disable gfxoff|export SMU)'
DMESG0=$(dmesg | wc -l)
gpu_event() { dmesg | tail -n +$((DMESG0+1)) | grep -qiE "$GPU_EVT"; }
WD_PID=""
cleanup() {
  [ -n "$WD_PID" ] && kill $WD_PID 2>/dev/null
  if gpu_event; then
    echo "--- sortie sur ÉVÉNEMENT GPU : cap NON modifié, reboot hôte recommandé"
    dmesg | tail -n +$((DMESG0+1)) | grep -iE "$GPU_EVT" | tail -4 | cut -c1-160
  else
    for h in /sys/class/drm/card[0-2]/device/hwmon/hwmon*; do echo 160000000 > $h/power1_cap 2>/dev/null; done
    echo "--- sortie : cap remis à 160 W ($(for c in 0 1 2; do echo -n "$(( $(cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_cap)/1000000 ))W "; done))"
  fi
}
trap cleanup EXIT

if gpu_event; then echo "!!! événement GPU déjà présent, rien n'est lancé"; exit 2; fi
for c in 0 1 2; do
  m=$(timeout 5 grep -A2 '^OD_MCLK:' /sys/class/drm/card$c/device/pp_od_clk_voltage | tail -1 | awk '{print $2}')
  [ "$m" = "1075MHz" ] || { echo "!!! card$c : mclk max = '$m' au lieu de 1075MHz"; exit 4; }
done
for h in /sys/class/drm/card[0-2]/device/hwmon/hwmon*; do echo $(( W * 1000000 )) > $h/power1_cap || { echo "!!! échec écriture cap"; exit 3; }; done
echo "######## SOAK : cap ${W} W, VRAM 1075 MHz, seuils d'arrêt jonction ${J_MAX} °C / mémoire ${M_MAX} °C — $(date +%H:%M:%S)"

# Chien de garde : relevé toutes les 10 s ; arrêt du serveur si seuil thermique dépassé ou événement GPU.
watchdog() {
  : > /root/soak-200w-temps.log
  while true; do
    l="$(date +%H:%M:%S)"; hot=0
    for c in 0 1 2; do
      h=$(ls -d /sys/class/drm/card$c/device/hwmon/hwmon*)
      p=$(( $(cat $h/power1_average)/1000000 )); j=$(( $(cat $h/temp2_input)/1000 )); m=$(( $(cat $h/temp3_input)/1000 ))
      l="$l c$c:${p}W/j${j}/m${m}"
      { [ "$j" -ge "$J_MAX" ] || [ "$m" -ge "$M_MAX" ]; } && hot=1
    done
    echo "$l" >> /root/soak-200w-temps.log
    if [ "$hot" = 1 ]; then echo "!!! SEUIL THERMIQUE ATTEINT : $l — arrêt du serveur"; pct exec 100 -- /root/vllm-pp3.sh stop >/dev/null 2>&1; return; fi
    if gpu_event; then echo "!!! ÉVÉNEMENT GPU pendant le soak — arrêt du serveur"; pct exec 100 -- /root/vllm-pp3.sh stop >/dev/null 2>&1; return; fi
    sleep 10
  done
}
watchdog & WD_PID=$!

pct exec 100 -- /root/soak-best.sh
echo "######## bilan thermique du soak"
python3 - <<'PY'
import re
mx={c:[0,0,0] for c in "012"}; n=0
for l in open("/root/soak-200w-temps.log"):
    n+=1
    for c,p,j,m in re.findall(r"c(\d):(\d+)W/j(\d+)/m(\d+)", l):
        a=mx[c]; a[0]=max(a[0],int(p)); a[1]=max(a[1],int(j)); a[2]=max(a[2],int(m))
print(f"{n} relevés (10 s)")
for c in "012": print(f"card{c}: pic {mx[c][0]} W, jonction max {mx[c][1]} °C, mémoire max {mx[c][2]} °C")
PY
echo "######## SOAK TERMINÉ $(date +%H:%M:%S)"
