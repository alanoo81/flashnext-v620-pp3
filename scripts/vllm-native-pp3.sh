#!/bin/bash
# vLLM Flash-Next natif (venv TheRock ROCm 10.0 + arbre opengfx1030 /root/vllm-rdna-native), PP3/TP1, PLE int4 en RAM
# Usage: [PP=3] [CTX=32768] [GPUUTIL=0.95] [EAGER=1] [UPSTREAM_ENV=1] [PART=17,17,14] [EXTRA="..."] ./vllm-native-pp3.sh start|stop|logs|probe
VENV=/root/vllm-native-venv; TREE=${TREE:-/root/vllm-rdna-native}; PP=${PP:-3}; CTX=${CTX:-32768}; GPUUTIL=${GPUUTIL:-0.95}; PORT=${PORT:-8086}
MODEL=${MODEL:-/root/models/flash-next/vllm/wtdcode-AWQ-W4A16}; PLE=${PLE:-/root/models/flash-next/vllm/ple-quant/ples_int4}
LOG=/root/vllm-native.log; CACHE=/root/vllm-cache/native; mkdir -p $CACHE/compile $CACHE/triton
case "${1:-start}" in
  stop) pkill -f "vllm.entrypoints.openai.api_serve[r]" ; sleep 3; pgrep -f "vllm.entrypoint[s]" >/dev/null && pkill -9 -f "vllm.entrypoint[s]"; echo stopped ;;
  logs) tail -f $LOG ;;
  start)
    pkill -f "vllm.entrypoints.openai.api_serve[r]" 2>/dev/null; sleep 2
    . $VENV/bin/activate; R=$(rocm-sdk path --root)
    export ROCM_PATH=$R ROCM_HOME=$R HIP_PATH=$R PATH=$R/bin:$PATH
    SP=$VENV/lib/python3.12/site-packages
    export LD_LIBRARY_PATH=$R/lib:$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib:$SP/torch/lib:${LD_LIBRARY_PATH:-}
    export ROCR_VISIBLE_DEVICES=${DEVS:-0,1,2} HSA_NO_SCRATCH_RECLAIM=1 NCCL_P2P_LEVEL=${P2P:-PHB}
    export VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600; [ "$PLE" = none ] && unset VLLM_PLE_QUANT_DIR || export VLLM_PLE_QUANT_DIR=$PLE; [ -n "$PLE_DISK" ] && export VLLM_PLE_DISK_OFFLOAD_DIR=$PLE_DISK
    export VLLM_CACHE_ROOT=$CACHE/compile TRITON_CACHE_DIR=$CACHE/triton PYTORCH_ROCM_ARCH=gfx1030
    export VLLM_RDNA_DENSE_INT8=${DENSE_INT8:-0} VLLM_DISABLE_COMPILE_CACHE=${COMPILE_CACHE_OFF:-1}
    [ -n "$PART" ] && export VLLM_PP_LAYER_PARTITION=$PART
    [ "${HSA_OVERRIDE:-0}" = 1 ] && export HSA_OVERRIDE_GFX_VERSION=10.3.0
    if [ "${UPSTREAM_ENV:-0}" = 1 ]; then
      export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RDNA2_FA=1 VLLM_ROCM_NO_MIXED_BATCH=0 VLLM_USE_AOT_COMPILE=0 VLLM_ROCM_USE_AITER_MOE=0
      export PYTORCH_TUNABLEOP_ENABLED=0 VLLM_BATCH_INVARIANT=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False GPU_MAX_HW_QUEUES=2
      export VLLM_WORKER_MULTIPROC_METHOD=spawn HSA_FORCE_FINE_GRAIN_PCIE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_FORCE_CUSTOM_ALL_REDUCE=0 VLLM_FA_RDNA2_GQA_MODE=subgroup
      EXTRA="$EXTRA --trust-remote-code --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_ranges_endpoints\":[],\"max_cudagraph_capture_size\":16,\"cudagraph_capture_sizes\":[1,2,4,8,16]}"
    fi
    if [ "${OGFX_PROD:-0}" = 1 ]; then   # config « production » d opengfx1030 du 19/09 (TP4+EP), transposée en PP3/TP1 :
      # retirés car propres au TP : custom all-reduce, RCCL_P2P_*, NCCL_PROTO, EP ; NCCL_P2P_LEVEL reste PHB (pix coupe le P2P ici,
      # pas de switch PCIe commun) ; KV 7e9 -> 3.5e9 (1/3 du modèle par carte au lieu de 1/4) ; vision off (--language-model-only).
      export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0} VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=${BREAKABLE:-1}
      export VLLM_RDNA_FUSED_HC=0 VLLM_ROCM_USE_AITER_MOE=0 VLLM_RDNA_FORCE_FP16=1 VLLM_BATCH_INVARIANT=0 GPU_MAX_HW_QUEUES=2
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False VLLM_WORKER_MULTIPROC_METHOD=spawn HSA_FORCE_FINE_GRAIN_PCIE=1
      export PYTORCH_TUNABLEOP_ENABLED=${TUNABLEOP:-1} PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 PYTORCH_TUNABLEOP_FILENAME=$CACHE/tunableop-ogfx/tunableop_results.csv; mkdir -p $CACHE/tunableop-ogfx
      EXTRA="$EXTRA --trust-remote-code --block-size 16 --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --distributed-timeout-seconds 1800 --compilation-config {\"cudagraph_mode\":\"${CGMODE:-FULL_AND_PIECEWISE}\",\"compile_ranges_endpoints\":[]${CGSIZES:+,\"cudagraph_capture_sizes\":[$CGSIZES],\"max_cudagraph_capture_size\":${CGSIZES##*,}}}"
    fi
    cd $TREE
    setsid nohup python -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name flash-next --dtype float16 \
      --tensor-parallel-size 1 --pipeline-parallel-size $PP --distributed-executor-backend mp \
      --max-model-len $CTX --gpu-memory-utilization $GPUUTIL --max-num-seqs ${SEQS:-4} --max-num-batched-tokens ${MNBT:-2048} \
      --language-model-only --skip-mm-profiling ${NOPC:---enable-prefix-caching} ${EAGER:+--enforce-eager} $EXTRA \
      --host 127.0.0.1 --port $PORT > $LOG 2>&1 < /dev/null & disown
    echo "serveur natif lancé (PP=$PP ctx=$CTX eager=${EAGER:-0} upstream_env=${UPSTREAM_ENV:-0}) ; log $LOG" ;;
esac
