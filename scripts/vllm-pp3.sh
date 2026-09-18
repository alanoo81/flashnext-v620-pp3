#!/bin/bash
# vLLM Flash-Next sur 3× V620 en PP3/TP1 (table PLE int4 en RAM via le worker CPU)
# Usage: TREE=/root/vllm-rdna|/root/vllm-leapdragon [PP=3] [CTX=65536] [GPUUTIL=0.95] [EXTRA="..."] ./vllm-pp3.sh [start|stop|logs|shell]
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
      --language-model-only --skip-mm-profiling ${NOPC:---enable-prefix-caching} ${EAGER:+--enforce-eager} $EXTRA \
      --host 127.0.0.1 --port $PORT
    echo "conteneur $NAME lancé (arbre $TREE, PP=$PP, ctx=$CTX) ; logs: $0 logs" ;;
esac
