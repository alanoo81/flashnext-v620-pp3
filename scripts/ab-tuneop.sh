#!/bin/bash
# A/B TunableOp : rangées leapdragon (TP4) vs rangées réglées pour PP3. Prefill cache OFF + décode, sans MTP puis MTP drafter W4A16.
MQ=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16-mtpq
for T in 1 pp3; do
  env TUNEOP=$T K=0 NOPC=--no-enable-prefix-caching SIZES=4096,16384,3000 GEN=300 /root/leap-run-stag.sh ab-nomtp-tune$T
  env TUNEOP=$T K=2 MODEL=$MQ NOPC=--no-enable-prefix-caching SIZES=4096,16384,3000 GEN=300 /root/leap-run-stag.sh ab-mtpq-tune$T
done
echo "=== FIN AB $(date +%H:%M:%S)"
