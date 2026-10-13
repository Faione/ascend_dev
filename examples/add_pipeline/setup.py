from pathlib import Path
import os
import subprocess

from setuptools import Command, setup, find_packages


ROOT = Path(__file__).resolve().parent


class CMakeBuild(Command):
    description = "Build the standalone Ascend Add extension"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 以当前 Python 环境中的 torch/torch_npu 为准，避免写死安装位置。
        import torch
        import torch_npu

        build_dir = ROOT / "build"
        cmake_args = [
            "cmake", "-S", str(ROOT), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTorch_DIR={Path(torch.utils.cmake_prefix_path) / 'Torch'}",
            f"-DTORCH_NPU_PATH={Path(torch_npu.__file__).resolve().parent}",
        ]
        # CMake 将生成的 _C.so 复制进 ascend_add_pipeline/ 包目录。
        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", str(build_dir), "--parallel", str(os.cpu_count() or 2)])


setup(
    name="ascend-add-pipeline",
    version="0.1.0",
    packages=find_packages(),
    package_data={"ascend_add_pipeline": ["*.so"]},
    cmdclass={"cmake_build": CMakeBuild},
    zip_safe=False,
)
