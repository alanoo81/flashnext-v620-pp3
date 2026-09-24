#!/bin/bash
# vLLM Flash-Next sur 3× V620 en PP3/TP1 (table PLE int4 en RAM via le worker CPU)
# Usage: TREE=/root/vllm-rdna|/root/vllm-leapdragon [PP=3] [CTX=65536] [GPUUTIL=0.95] [EXTRA="..."] ./vllm-pp3.sh [start|stop|logs|shell]
#
# Avertissements ATTENDUS dans le journal de démarrage (docker logs fn-pp3), à ne pas « corriger » :
#  - "num_speculative_tokens > 1 ... lower acceptance rate" : générique ; k=2 mesuré meilleur que k=1 (52 -> 57-68 t/s), k=3 sans gain.
#  - "Mamba cache mode is set to 'align'" : requis par le cache de préfixes sur les couches GDN (backport #54044 dans l overlay).
#  - "max_num_scheduled_tokens is set to 2048 ... consider increasing max_num_batched_tokens" : 4096 mesuré -17..-27 % de prefill.
#  - "Using FlashAttention version None" : pas de lib FA pour gfx1030, l attention passe par Triton (FLASH_ATTENTION_TRITON_AMD_ENABLE).
#  - "Op 'sparse_attn_indexer' doesn't exist" : ajouté par vllm/platforms/rocm.py de l arbre leapdragon, sans effet (indexeur QSA en Triton).
#  - "CUDA graph memory profiling is disabled (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0)" : voulu — sans cela le profileur
#    sous-alloue le KV (61K tokens) ; avec, 344-384K tokens à 262K, validé par 30 min de soak (marge ~0,5 Gio sur la carte la plus serrée).
#  - (plus émis) "Auto-prefetch is disabled ... EXT4" : --safetensors-load-strategy=prefetch est passé d office (PREFETCH=0 pour l enlever) :
#    à froid (cache disque vidé) poids en 78 s au lieu de 126, serveur prêt en 196 s au lieu de 266 (mesure du 21/09).
# Vision (VISION=1) : l encodeur d images tient, mais le profilage multimodal coûte ~3,5 Gio sur l étage 0 : à 262K le KV
#   n a plus de place ; à 131K avec 1 image de <= 802 816 px : réservoir 211K jetons (au lieu de 465K en texte seul à 262K).
#   Mesuré le 21/09 : capture d écran 1 000×450 décrite correctement en 9,6 s (556 jetons de prompt).
# API : appels d outils (--enable-auto-tool-choice --tool-call-parser qwen3_coder) et raisonnement dans reasoning_content
#   (--reasoning-parser qwen3) activés par défaut (TOOLS=0 pour les retirer) — requis par les clients agentiques (tool_choice=auto).
# Réseau : le serveur écoute sur ${HOST:-0.0.0.0}:${PORT:-8086} (réseau de l hôte du CT, ex. http://192.168.1.252:8086/v1/models) ;
#   HOST=127.0.0.1 pour le limiter au CT, API_KEY=<secret> pour exiger un jeton Bearer (vLLM --api-key).
# Ordre des cartes : DEVS=<indices HSA> (défaut 0,1,2 = bus 43:00, 46:00, 63:00 -> étages 0,1,2). DEVS=1,2,0 met l étage 2
# (le plus léger : 13-14 couches + drafter) sur la carte 43:00, la plus chaude (face au hub du ventilateur) : -8 W / -1 °C mesurés.
IMG=${IMG:-ghcr.io/leapdragon/vllm-rdna2-qwen:latest}
TREE=${TREE:-/root/vllm-rdna}; NAME=${NAME:-fn-pp3}; PP=${PP:-3}; CTX=${CTX:-65536}; GPUUTIL=${GPUUTIL:-0.95}; PORT=${PORT:-8086}
MODEL=${MODEL:-/root/models/flash-next/vllm/wtdcode-AWQ-W4A16}; PLE=/root/models/flash-next/vllm/ple-quant/ples_int4
CACHE=${CACHE:-/root/vllm-cache/$(basename $TREE)}; mkdir -p $CACHE/compile $CACHE/triton $CACHE/tunableop
COMMON=(--device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 --ipc=host --shm-size=32g --network=host --security-opt seccomp=unconfined --cap-add=SYS_PTRACE
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e ROCR_VISIBLE_DEVICES=${DEVS:-0,1,2} -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_LEVEL=${P2P:-PHB}
  -e VLLM_ROCM_USE_AITER=0 -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_QUANT_DIR=/ples_int4 -e PLE_OFFLOAD_DOORBELL=${DOORBELL:-1} -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 -e V620_PLE_TRACE=${TRACE:-0}
  -e VLLM_CACHE_ROOT=/cache/compile -e TRITON_CACHE_DIR=/cache/triton -e PYTORCH_ROCM_ARCH=gfx1030
  -e V620_FAKEQ_DENSE -e V620_GDN_TRITON -e V620_MOE_TRITON -e VLLM_RDNA_DENSE_GEMV -e VLLM_GDN_HIP_PREFILL -e V620_RMS_C -e VLLM_RDNA_DENSE_INT8=${DENSE_INT8:-1} -e VLLM_RDNA_DENSE_INT8_ONLY=${DENSE_INT8_ONLY:-0} -e VLLM_ROCM_MOE_PADDING=${MOE_PADDING:-1} -e VLLM_DISABLE_COMPILE_CACHE=0 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS -e PYTORCH_CUDA_ALLOC_CONF -e VLLM_QSA_KV_OFFLOAD -e VLLM_QSA_KV_OFFLOAD_MAX_GIB -e VLLM_QSA_KVO_ARENA -e V620_MOE_HIP=${MOE_HIP:-0} ${PART:+-e VLLM_PP_LAYER_PARTITION=$PART} ${V2RUNNER:+-e VLLM_USE_V2_MODEL_RUNNER=$V2RUNNER}
  -v $MODEL:/model -v $PLE:/ples_int4 -v $CACHE:/cache)
# TunableOp (GEMM rocBLAS), rangées liées au build rocBLAS de l image leapdragon 0915 (sha 9847aecc4bf8) -> TREE=image seulement.
#   TUNEOP=1 (défaut avec TREE=image) : lookup-only sur NOS rangées PP3/TP1 = rangées leapdragon (réglées en TP4) + les formes
#             TP1 qui leur manquaient ; source versionnée $OVERLAY/tunableop-pp3/, copiée dans $CACHE/tunableop/pp3/ si absente
#   TUNEOP=leap : lookup-only sur les rangées d origine de l image      TUNEOP=tune : règle toute forme nouvelle (scripts/tune-pp3.sh)
#   TUNEOP=0 : désactivé
OVERLAY=${OVERLAY:-/root/vllm-leap-img}
[ "$TREE" = image ] && TUNEOP=${TUNEOP:-1}; [ "${TUNEOP:-0}" = pp3 ] && TUNEOP=1
case "${TUNEOP:-0}" in
  leap) COMMON+=(-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
          -e PYTORCH_TUNABLEOP_FILENAME=/app/vllm/tunableop/rocblas-9847aecc4bf8/tunableop_results.csv) ;;
  1|tune)
    T=0; [ "$TUNEOP" = tune ] && T=1
    mkdir -p $CACHE/tunableop/pp3
    for r in 0 1 2; do [ -s $CACHE/tunableop/pp3/tunableop_results$r.csv ] || cp $OVERLAY/tunableop-pp3/tunableop_results$r.csv $CACHE/tunableop/pp3/ 2>/dev/null; done
    if [ "$T" = 1 ] || [ -s $CACHE/tunableop/pp3/tunableop_results0.csv ]; then
      COMMON+=(-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=$T -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
        -e PYTORCH_TUNABLEOP_VERBOSE=${TUNEOP_VERBOSE:-0} -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/pp3/tunableop_results.csv)
    else echo "TunableOp : rangées PP3 introuvables ($OVERLAY/tunableop-pp3), désactivé" >&2; fi ;;
esac
# tailles de capture piecewise pour les lots de prefill (leapdragon §8e) — CG_SIZES="1 2 4 8 16 32 64 128 256"
[ -n "${CG_SIZES:-}" ] && EXTRA="$EXTRA --cudagraph-capture-sizes $CG_SIZES"
# Équité prefill/décodage : LONGPREFILL=<jetons> plafonne le morceau de prefill traité par pas d ordonnancement.
# ATTENTION : sans effet au-dessus de MNBT (2048 par défaut), puisque le morceau y est déjà borné — il faut une
# valeur INFÉRIEURE. Plus la valeur est basse, plus les requêtes en décodage avancent pendant qu un long prompt
# est lu, mais plus le prefill total est lent (davantage de pas, chacun avec son coût fixe). 0 = désactivé.
# (--max-num-partial-prefills n existe PAS dans cette build, seul ce seuil est disponible.)
[ "${LONGPREFILL:-0}" != 0 ] && EXTRA="$EXTRA --long-prefill-token-threshold ${LONGPREFILL}"
# Anti-emballement du cache KV : WATERMARK=<0..1> garde cette fraction de blocs libres à l admission d une requête
# en attente ou préemptée. Vise le cycle admission -> remplissage à 100 % -> préemption -> recalcul observé avec 7 agents
# à gros contextes. 0 = désactivé (défaut amont). `scheduler_reserve_full_isl` est déjà à True et couvre le premier cas.
[ -n "${WATERMARK:-}" ] && EXTRA="$EXTRA --watermark ${WATERMARK}"
[ "${PREFETCH:-1}" = 1 ] && EXTRA="$EXTRA --safetensors-load-strategy=prefetch"   # chargement des poids ~40 % plus rapide à froid
# Clients agentiques (appels d outils, raisonnement séparé) : TOOLS=0 pour désactiver. Sans effet sur /v1/completions (harnais de mesure).
# Vision : VISION=1 charge l encodeur d images (captures d écran, photos) — VISION=0 (défaut) = texte seul, sans profilage multimodal.
#   VISION_MAX_PIXELS (défaut 1605632 ≈ 1 460×1 100) et VISION_IMAGES (images max par requête, défaut 2).
if [ "${VISION:-0}" = 1 ]; then MMARGS="--limit-mm-per-prompt {\"image\":${VISION_IMAGES:-2}} --mm-processor-kwargs {\"max_pixels\":${VISION_MAX_PIXELS:-1605632}}"
else MMARGS="--language-model-only --skip-mm-profiling"; fi
# TOOLS=1 : appels d outils (l analyseur ne travaille qu en fin de génération). REASONING=1 : sépare le raisonnement
# dans le champ `reasoning` — ATTENTION, il analyse CHAQUE jeton de CHAQUE flux dans le serveur d API mono-thread.
[ "${TOOLS:-1}" = 1 ] && EXTRA="$EXTRA --enable-auto-tool-choice --tool-call-parser qwen3_coder"
[ "${REASONING:-1}" = 1 ] && EXTRA="$EXTRA --reasoning-parser qwen3"
if [ "${UPSTREAM_ENV:-0}" = 1 ]; then   # pile d environnement de scripts/serve_gfx1030_full.sh (opengfx1030)
  COMMON+=(-e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_RDNA2_FA=1 -e VLLM_ROCM_NO_MIXED_BATCH=0 -e VLLM_ROCM_SKIP_LIVE_TAIL_HASH=1
    -e VLLM_USE_AOT_COMPILE=0 -e VLLM_DISABLE_COMPILE_CACHE=1 -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_RDNA_FORCE_FP16=1
    -e PYTORCH_TUNABLEOP_ENABLED=0 -e VLLM_BATCH_INVARIANT=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    -e GPU_MAX_HW_QUEUES=2 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e HSA_FORCE_FINE_GRAIN_PCIE=1
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_FORCE_CUSTOM_ALL_REDUCE=0 -e VLLM_FA_RDNA2_GQA_MODE=subgroup)
  EXTRA="$EXTRA ${BLOCK:+--block-size $BLOCK} --trust-remote-code --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_ranges_endpoints\":[],\"max_cudagraph_capture_size\":16,\"cudagraph_capture_sizes\":[1,2,4,8,16]}"
fi
if [ "$TREE" = image ]; then   # arbre leapdragon compilé dans l image + nos 3 fichiers patchés
  COMMON+=(-v /root/vllm-leap-img/v620_dump.py:/app/vllm/v620_dump.py:ro -v /root/dump:/dump -e V620_DUMP=${DUMP:-} -e V620_DUMP_STEPS=${DUMP_STEPS:-8})
  for f in $(cd /root/vllm-leap-img && find vllm \( -name "*.py" -o -name "*.json" -o -name "*.so" \) | sort); do
    COMMON+=(-v /root/vllm-leap-img/$f:/app/vllm/$f:ro); done
  COMMON+=(-w /app/vllm)
else
  COMMON+=(-v $TREE:${TREE_MNT:-/src} -w ${TREE_MNT:-/src})
fi
# Garde-fou : un plafond de puissance au-dessus de 200 W signale que le module amdgpu patché n'est pas chargé — ce qui,
# après une mise à jour du noyau, va de pair avec la perte du pilotage du ventilateur (modules hors-arbre reconstruits
# ensemble par v620-rebuild-amdgpu). Pas de charge GPU dans cet état. FORCE=1 pour passer outre en connaissance de cause.
if [ "${1:-start}" = start ] && [ "${FORCE:-0}" != 1 ]; then
  for c in /sys/class/drm/card[0-9]/device/hwmon/hwmon*/power1_cap; do
    w=$(( $(cat "$c" 2>/dev/null || echo 0) / 1000000 ))
    if [ "$w" -gt 200 ]; then
      echo "REFUS : plafond de puissance à ${w} W (attendu 160). Module amdgpu patché absent, ventilation probablement non pilotée." >&2
      echo "        Après une mise à jour du noyau : v620-rebuild-amdgpu puis redémarrage, ou démarrer sur le noyau précédent. FORCE=1 pour passer outre." >&2
      exit 3
    fi
  done
fi
case "${1:-start}" in
  shell) docker run --rm -it --entrypoint bash "${COMMON[@]}" $IMG ;;
  stop)  docker stop -t 60 $NAME; docker rm $NAME ;;
  logs)  docker logs -f $NAME ;;
  start)
    docker rm -f $NAME >/dev/null 2>&1
    docker run -d --name $NAME --entrypoint python3 "${COMMON[@]}" $IMG -m vllm.entrypoints.openai.api_server \
      --model /model --served-model-name flash-next --dtype float16 \
      --tensor-parallel-size 1 --pipeline-parallel-size $PP --distributed-executor-backend mp \
      --max-model-len $CTX --gpu-memory-utilization $GPUUTIL --max-num-seqs ${SEQS:-4} --max-num-batched-tokens ${MNBT:-2048} \
      $MMARGS ${NOPC:---enable-prefix-caching} ${EAGER:+--enforce-eager} $EXTRA \
      --host ${HOST:-0.0.0.0} --port $PORT ${API_KEY:+--api-key $API_KEY}
    echo "conteneur $NAME lancé (arbre $TREE, PP=$PP, ctx=$CTX) ; logs: $0 logs" ;;
esac
