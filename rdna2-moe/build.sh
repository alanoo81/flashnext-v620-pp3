#!/bin/bash
set -x; cd /build; rm -rf build *.so
export PYTORCH_ROCM_ARCH=gfx1030 MAX_JOBS=8
time python3 setup.py build_ext --inplace 2>&1 | tail -60
ls -la /build/*.so && python3 -c "import torch; torch.ops.load_library(\"/build/v620_moe_rdna2.so\"); print(\"op:\", torch.ops._v620_rdna2.moe_gptq_gemm_rdna2)"
