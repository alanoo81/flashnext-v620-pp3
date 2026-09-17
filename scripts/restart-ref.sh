#!/bin/bash
# redémarre le serveur natif de référence (wtdcode + ples_int4, eager, DENSE_GEMV, dev mode) ; env supplémentaire via EXTRA_ENV
cd /root; ./vllm-native-pp3.sh stop >/dev/null 2>&1; pkill -9 -f "VLLM:[:]"; pkill -9 -f "multiprocessing.spaw[n]"; sleep 5
env VLLM_RDNA_DENSE_GEMV=1 VLLM_SERVER_DEV_MODE=1 $EXTRA_ENV EAGER=1 ./vllm-native-pp3.sh start >/dev/null
