# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoE kernel dispatcher.

Selects architecture-specific native HIP MoE kernels in priority order.
Falls back to the Triton WNA16 path when no native kernel is available.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _v620_rdna2_available() -> bool:
    """v620-pp3: fused W4A16 MoE HIP kernel for gfx1030 (V620_MOE_HIP=0 disables)."""
    import os

    if os.environ.get("V620_MOE_HIP", "1") != "1":
        return False
    if hasattr(torch.ops, "_v620_rdna2") and hasattr(
        torch.ops._v620_rdna2, "moe_gptq_gemm_rdna2"
    ):
        return True
    so = os.environ.get("V620_MOE_SO", "/app/vllm/vllm/v620_moe_rdna2.so")
    if not os.path.exists(so):
        return False
    try:
        torch.ops.load_library(so)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("v620: cannot load %s: %s", so, exc)
        return False
    return hasattr(torch.ops._v620_rdna2, "moe_gptq_gemm_rdna2")


def is_supported(weight_quant) -> bool:
    """Check if a native ROCm MoE kernel is available for this config."""
    if weight_quant.num_bits != 4:
        return False

    from vllm.platforms.rocm import on_gfx10x, on_gfx1100

    # v620-pp3: RDNA2 (gfx1030) via the standalone extension (see _v620_rdna2_available)
    if on_gfx10x() and _v620_rdna2_available():
        return True

    # RDNA3 (gfx1100). Future: add RDNA4 (gfx12x), CDNA (gfx94x), etc.
    return (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3")
    )


def make_method(weight_quant, input_quant, moe_config):
    """Create the native ROCm MoE method. Call only after is_supported()."""
    from vllm.platforms.rocm import on_gfx10x, on_gfx1100

    if on_gfx10x() and _v620_rdna2_available():
        from .compressed_tensors_moe_wna16_rdna2 import (
            CompressedTensorsWNA16RDNA2MoEMethod,
        )
        logger.info_once(
            "Using CompressedTensorsWNA16RDNA2MoEMethod (native gfx1030 HIP kernel "
            "moe_gptq_gemm_rdna2, v620 port from opengfx1030/vllm-rdna)"
        )
        return CompressedTensorsWNA16RDNA2MoEMethod(
            weight_quant, input_quant, moe_config
        )
    if on_gfx1100():
        from .compressed_tensors_moe_wna16_rdna3 import (
            CompressedTensorsWNA16RDNA3MoEMethod,
        )

        logger.info_once(
            "Using CompressedTensorsWNA16RDNA3MoEMethod (native RDNA3 HIP kernel)"
        )
        return CompressedTensorsWNA16RDNA3MoEMethod(
            weight_quant, input_quant, moe_config
        )

    # Future: RDNA4, CDNA, etc.
    raise RuntimeError("is_supported() returned True but no kernel matched")
