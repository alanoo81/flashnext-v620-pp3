#!/bin/bash
# vLLM Flash-Next sur 3× V620 en PP3/TP1 (table PLE int4 en RAM via le worker CPU)
# Usage: TREE=/root/vllm-rdna|/root/vllm-leapdragon [PP=3] [CTX=65536] [GPUUTIL=0.95] [EXTRA="..."] ./vllm-pp3.sh [start|stop|logs|shell]
IMG=${IMG:-ghcr.io/leapdragon/vllm-rdna2-qwen:latest}
TREE=${TREE:-/root/vllm-rdna}; NAME=${NAME:-fn-pp3}; PP=${PP:-3}; CTX=${CTX:-65536}; GPUUTIL=${GPUUTIL:-0.95}; PORT=${PORT:-8086}
MODEL=/root/models/flash-next/vllm/wtdcode-AWQ-W4A16; PLE=/root/models/flash-next/vllm/ple-quant/ples_int4
CACHE=${CACHE:-/root/vllm-cache/$(basename $TREE)}; mkdir -p $CACHE/compile $CACHE/triton $CACHE/tunableop
COMMON=(--device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 --ipc=host --shm-size=32g --network=host --security-opt seccomp=unconfined --cap-add=SYS_PTRACE
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e ROCR_VISIBLE_DEVICES=${DEVS:-0,1,2} -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_LEVEL=${P2P:-PHB}
  -e VLLM_ROCM_USE_AITER=0 -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_QUANT_DIR=/ples_int4 -e PLE_OFFLOAD_DOORBELL=${DOORBELL:-1} -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 -e V620_PLE_TRACE=${TRACE:-0}
  -e VLLM_CACHE_ROOT=/cache/compile -e TRITON_CACHE_DIR=/cache/triton -e PYTORCH_ROCM_ARCH=gfx1030
  -e V620_GDN_TRITON -e V620_MOE_TRITON -e VLLM_RDNA_DENSE_GEMV -e VLLM_GDN_HIP_PREFILL -e V620_RMS_C -e VLLM_RDNA_DENSE_INT8=${DENSE_INT8:-1} -e VLLM_RDNA_DENSE_INT8_ONLY=${DENSE_INT8_ONLY:-0} -e VLLM_ROCM_MOE_PADDING=${MOE_PADDING:-1} -e VLLM_DISABLE_COMPILE_CACHE=0 -e V620_MOE_HIP=${MOE_HIP:-0} ${PART:+-e VLLM_PP_LAYER_PARTITION=$PART} ${V2RUNNER:+-e VLLM_USE_V2_MODEL_RUNNER=$V2RUNNER}
  -v $MODEL:/model -v $PLE:/ples_int4 -v $CACHE:/cache)
# leapdragon serve-qwen38-flash-next.sh : TunableOp lookup-only (rangées rocBLAS de l image, sha 9847aecc4bf8) — TUNEOP=1
if [ "${TUNEOP:-0}" = 1 ]; then
  COMMON+=(-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    -e PYTORCH_TUNABLEOP_FILENAME=/app/vllm/tunableop/rocblas-9847aecc4bf8/tunableop_results.csv)
fi
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
  COMMON+=(-v ${OVERLAY:-/root/vllm-leap-img}/v620_dump.py:/app/vllm/v620_dump.py:ro -v /root/dump:/dump -e V620_DUMP=${DUMP:-} -e V620_DUMP_STEPS=${DUMP_STEPS:-8})
  for f in $(cd ${OVERLAY:-/root/vllm-leap-img} && find vllm \( -name "*.py" -o -name "*.json" -o -name "*.so" \) | sort); do
    COMMON+=(-v ${OVERLAY:-/root/vllm-leap-img}/$f:/app/vllm/$f:ro); done
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
