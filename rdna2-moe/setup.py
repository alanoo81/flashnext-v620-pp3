import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
setup(
    name="v620_moe_rdna2",
    ext_modules=[CUDAExtension(
        name="v620_moe_rdna2",
        sources=["bindings.cpp", "moe_q_gemm_rdna2.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--offload-arch=gfx1030"]},
    )],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
