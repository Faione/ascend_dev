ARG BASE_IMAGE=ubuntu:22.04
# 公共层：系统编译工具与 APT/PIP 镜像配置。
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG MIRROR_DOMAIN=http://mirrors.tools.huawei.com
RUN set -eux; \
    [[ "$MIRROR_DOMAIN" =~ ^https?://[A-Za-z0-9.-]+(:[0-9]+)?/?$ ]]; \
    mirror="${MIRROR_DOMAIN%/}"; \
    sed -Ei "s#https?://([a-z0-9.-]*archive.ubuntu.com|security.ubuntu.com)#${mirror}#g" /etc/apt/sources.list; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential cmake ninja-build git \
      libssl-dev zlib1g-dev libffi-dev ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip config --user set global.index "${mirror}/pypi/simple"; \
    python3 -m pip config --user set global.index-url "${mirror}/pypi/simple"; \
    python3 -m pip config --user set global.trusted-host "${mirror#*://}"

FROM base AS torch-install
# wheel 从构建上下文只读挂载；安装结果进入镜像，原始 wheel 不进入层。
ARG TORCH_VERSION=2.10.0
ARG TORCH_NPU_VERSION=2.10.0.post4
ARG TORCH_WHEEL_FILE=
ARG TORCH_NPU_WHEEL_FILE=
ENV PIP_NO_CACHE_DIR=1
RUN --mount=type=bind,source=packages,target=/tmp/cann-packages,readonly \
    set -eux; \
    python3 -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"; \
    python3 -m pip install "/tmp/cann-packages/$TORCH_WHEEL_FILE"; \
    python3 -m pip install "/tmp/cann-packages/$TORCH_NPU_WHEEL_FILE"; \
    python3 -m pip install build pyyaml 'numpy<2' pytest; \
    python3 -c "import importlib.metadata as m; assert m.version('torch') == '${TORCH_VERSION}+cpu'; assert m.version('torch-npu') == '${TORCH_NPU_VERSION}'"

FROM torch-install AS cann-install
# .run 也只读挂载。先检查，再分别安装 Toolkit 与型号对应的算子包。
RUN --mount=type=bind,source=packages,target=/tmp/cann-packages,readonly \
    bash /tmp/cann-packages/toolkit.run --check \
    && bash /tmp/cann-packages/ops.run --check \
    && bash /tmp/cann-packages/toolkit.run --install --quiet \
    && bash /tmp/cann-packages/ops.run --install --quiet \
    && test -f /usr/local/Ascend/cann/set_env.sh

FROM torch-install AS final
# 仅复制 CANN 安装目录，避免把安装包和安装阶段临时文件带入最终镜像。
COPY --from=cann-install /usr/local/Ascend /usr/local/Ascend
COPY entrypoint.sh /usr/local/bin/ascend-entrypoint
RUN chmod +x /usr/local/bin/ascend-entrypoint \
    && source /usr/local/Ascend/cann/set_env.sh \
    && atc --help >/dev/null
ENTRYPOINT ["/usr/local/bin/ascend-entrypoint"]
CMD ["/bin/bash"]
