#!/bin/bash
# ogfx-try.sh <étiquette> VAR=val ... : démarre l arbre opengfx 0919 avec ces variables, envoie une requête, dit si le moteur survit.
cd /root; tag=$1; shift
./vllm-native-pp3.sh stop >/dev/null 2>&1; sleep 6
env TREE=/root/vllm-rdna-0919 CTX=65536 PART=18,17,13 "$@" ./vllm-native-pp3.sh start >/dev/null
up=0; for i in $(seq 1 360); do curl -s -m 2 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}" | grep -q 200 && { up=1; break; }; pgrep -f "openai.api_serve[r]" >/dev/null || break; sleep 5; done
[ $up = 1 ] || { echo "== $tag : MORT au démarrage : $(grep -E "Error" /root/vllm-native.log | tail -1 | cut -c1-160)"; exit; }
r=$(python3 sweep.py 4336 0 60 2>&1 | tail -1 | cut -c1-90); sleep 2
echo "== $tag : $r | KeyError=$(grep -c KeyError /root/vllm-native.log) health=$(curl -s -m 3 http://127.0.0.1:8086/health -o /dev/null -w "%{http_code}")"
