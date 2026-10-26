# Ascend 算子开发实践

本仓库收录不同的 AscendC 算子开发实践：

- [Add Pipeline 示例](examples/add_pipeline/)：包含代码、构建与验证脚本，以下环境说明服务于这个示例。
- [MXFP8 量化加法示例](examples/add_pipeline/mxfp8_add_pipeline.py)：先逐 32 元素量化为 E4M3FN 与 E8M0 scale，再反量化并复用 FP32 Add 内核；[交互说明](docs/mxfp8-add-pipeline.html)。
- [Sinh 算子设计实验报告](docs/sinh-operator-experiment.md)：记录 `8 × 2048` 的 `half` 输入上的分块、搬运、三地址码、缓冲区与优化思考。算子已通过真机测试，报告尚未收录测试命令和量化数据。

Add 示例从[实验 Gist](https://gist.github.com/Faione/fc06cfce0900e135ffddadd33b71d51f)提取环境脚本和代码，并将 Add 做成独立示例。示例参考 [ops-transformer](https://gitcode.com/Ascend/ops-transformer) 的 Fast Kernel Launch 写法，但构建时不克隆或链接该仓库，也不需要 `opbase`、`ascend-build-deps`。本机尚未接入 Ascend 设备；下列编译和 NPU 结果需在 950 服务器验证。

## 一、环境准备

目标是 x86_64 服务器、Ubuntu 22.04 基础镜像、CANN 9.1.0、PyTorch 2.10.0、torch-npu 2.10.0.post4。910/910B 的安装包也可用同一镜像脚本准备，但本示例的编译和验证目标为 **950**。宿主机需已有匹配设备的驱动和 Docker/Compose；驱动不装进镜像。

在 Windows 浏览器中下载如下文件（也可运行 [`environment/open-downloads-windows.ps1`](environment/open-downloads-windows.ps1) 打开链接），放在同一下载目录：

| 文件 | 下载地址 |
| --- | --- |
| `Ascend-cann-toolkit_9.1.0_linux-x86_64.run` | [CANN 9.1.0 Toolkit](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-x86_64.run) |
| `Ascend-cann-950-ops_9.1.0_linux-x86_64.run` | [950 算子包](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-950-ops_9.1.0_linux-x86_64.run) |
| `torch-2.10.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl` | [PyTorch CPU wheel](https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl) |
| `torch_npu-2.10.0.post4-cp310-cp310-manylinux_2_28_x86_64.whl` | [torch-npu wheel](https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp310-cp310-manylinux_2_28_x86_64.whl) |

若需准备 910/910B 环境，将对应的 [910 ops](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910-ops_9.1.0_linux-x86_64.run) 或 [910B ops](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-x86_64.run) 文件一起下载。Windows 上传脚本要求以上六个文件都存在：

```powershell
powershell -ExecutionPolicy Bypass -File .\environment\upload-windows.ps1 -Server user@server -RemoteRoot /home/user/cann-run -DownloadDir C:\Users\you\Downloads
```

已有文件也可以自行用 `scp` 上传。构建时，Toolkit 和对应型号的 ops `.run` 文件应位于同一个 `cann-run/<soc>/<version>/` 目录；`soc` 是 950、910 等设备型号，`version` 是 9.1.1、9.2.0 等 CANN 版本。例如 `cann-run/950/9.1.1/`。服务器先确认本地基础镜像存在：

```bash
docker image inspect ubuntu:22.04 >/dev/null
```

从仓库根目录构建 950 镜像。将对应版本的 `torch-*.whl` 和 `torch_npu-*.whl` 放在 `cann-run/torch/2.7.1/` 或 `cann-run/torch/2.10.0/`。脚本根据 `--soc`、`--cann-version` 和 `--torch-version` 分别选择 CANN 安装包与两个 wheel，并核对文件名中的基础版本；`torch_npu` 的 `.postN`、`.devN` 后缀不影响选择。目录不在自动查找范围时，分别用 `--cann-root` 和 `--torch-root` 指向 `cann-run` 与 `cann-run/torch`。脚本将安装包复制到临时构建上下文，经 [`environment/Dockerfile`](environment/Dockerfile) 的多阶段构建安装；`.run` 和 wheel 不会留在最终镜像中。APT/PIP 镜像默认使用 `http://mirrors.tools.huawei.com`，可用 `--mirror-domain` 覆盖。

```bash
bash environment/build-image.sh --soc 950 --cann-version 9.1.0 --torch-version 2.10.0
```

镜像名为 `ascend-dev:9.1.0-950-2.10.0`。如选择 `--torch-version 2.7.1`，镜像标签中的 Torch 版本也相应变为 `2.7.1`。`build-image.sh --help` 列出全部参数；`--base-image` 指定本地 Ubuntu 22.04 镜像。构建依赖镜像站安装 apt 包、PyTorch 的附加 Python 依赖。

## 二、后续实验

在 950 服务器上，将本仓库放在可由 Docker 访问的路径，然后启动 [`environment/docker-compose.yaml`](environment/docker-compose.yaml)：

```bash
export PROJECT_ROOT="$(pwd)"
docker compose -f environment/docker-compose.yaml config
docker compose -f environment/docker-compose.yaml up -d
docker compose -f environment/docker-compose.yaml exec ascend-add-pipeline bash
```

Compose 将本仓库和宿主机驱动所需的设备、库及安装信息挂入容器。若宿主机没有列出的设备或驱动路径，应先按该机器实际驱动布局调整 Compose 文件；本机运行正常的 `torch_npu` 运算比 `npu-smi` 更适合作为此容器的验证依据。

容器入口已执行 `set_env.sh`。在容器工作目录中编译并验证：

```bash
python3 setup.py cmake_build
python3 verify_add_pipeline.py --device 0
```

编译使用 CANN 的 Bisheng、PyTorch 和 torch-npu。`CMakeLists.txt` 直接编译本目录的 `csrc/add_pipeline.cpp`，目标架构是 950 的 `dav-3510`；无需执行 `ops-transformer/build.sh`。`verify_add_pipeline.py` 以表格显示 NPU 可用性、不同长度的 Add 结果、输入检查和最终结果；期望各行 `PASS`，数值误差在 `torch.testing.assert_close` 阈值内。内核采用双缓冲 `TPipe/TQue`，每轮依次 CopyIn、Compute、CopyOut，支持非整 tile 长度。

可在线打开 [Add Pipeline 动态示意](https://faione.github.io/ascend_dev/)（[页面源码](docs/add-pipeline.html)）。代码分析部分沿 Python、CPU C++、NPU 内核和硬件设备四条通道展示控制流与数据流；运行时部分可切换向量长度，同时查看 4 个 AI Core 的 tile 分配，未分到 tile 的核心显示“未启动”。页面区分验证脚本的 H2D/D2H 与内核的 CopyIn/Compute/CopyOut，是代码流程说明，不代表真实硬件的执行时序或性能数据。

如果 CMake 找不到 `ASC`/`AICPU` 或 Bisheng，先在容器中检查 `ASCEND_HOME_PATH` 与 `/usr/local/Ascend/cann/set_env.sh`。若 `opdev/bfloat16.h` 缺失，检查 CANN Toolkit 安装和头文件路径；本示例未依赖 `ops-transformer` 的 `opbase`。如 `torch.npu.is_available()` 为 false，优先核对宿主机驱动、Compose 设备映射和驱动库路径。

`csrc/add_pipeline.cpp` 改编自 Gist 中的代码；仓库附有上游 [CANN Open Software License Agreement 2.0](LICENSE)。
