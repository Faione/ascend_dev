#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'HELP'
Usage: bash build-image.sh --toolkit-run FILE --ops-run FILE --soc MODEL [options]
  --toolkit-run FILE  Original Ascend-cann-toolkit_<version>_linux-x86_64.run
  --ops-run FILE      Original Ascend-cann-<model>-ops_<version>_linux-x86_64.run
  --soc MODEL         Ops model label, e.g. 910b, 910, A3, 950, 310p
  --base-image IMAGE  Local Ubuntu 22.04 base image (default: ubuntu:22.04)
  --mirror-domain URL  APT and pip mirror origin (default: http://mirrors.tools.huawei.com)
  --torch-version VER Torch version to install (default: 2.10.0)
  --torch-npu-version VER TorchNPU package version (default: 2.10.0.post4)
  --torch-wheel FILE  Local Python 3.10 x86_64 CPU wheel (required)
  --torch-npu-wheel FILE  Local matching TorchNPU wheel (required)
  -h, --help          Show help
Supports x86_64 packages and amd64 base images only.
HELP
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 参数可由命令行提供；默认值只用于通用版本和镜像站。
toolkit=""; ops=""; soc=""; torch_wheel=""; torch_npu_wheel=""
base_image="${BASE_IMAGE:-ubuntu:22.04}"
mirror_domain="${MIRROR_DOMAIN:-http://mirrors.tools.huawei.com}"
torch_version="${TORCH_VERSION:-2.10.0}"
torch_npu_version="${TORCH_NPU_VERSION:-2.10.0.post4}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --toolkit-run|--ops-run|--soc|--base-image|--mirror-domain|--torch-version|--torch-npu-version|--torch-wheel|--torch-npu-wheel)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
            case "$1" in
                --toolkit-run) toolkit="$2" ;;
                --ops-run) ops="$2" ;;
                --soc) soc="$2" ;;
                --base-image) base_image="$2" ;;
                --mirror-domain) mirror_domain="$2" ;;
                --torch-version) torch_version="$2" ;;
                --torch-npu-version) torch_npu_version="$2" ;;
                --torch-wheel) torch_wheel="$2" ;;
                --torch-npu-wheel) torch_npu_wheel="$2" ;;
            esac
            shift 2 ;;
        *) usage >&2; fail "Unknown argument: $1" ;;
    esac
done
[[ -s "$toolkit" && -s "$ops" && -n "$soc" ]] || fail 'Specify existing --toolkit-run and --ops-run files plus --soc.'
# 从官方文件名提取 CANN 版本及设备型号，防止 Toolkit/ops 混搭。
toolkit_name="$(basename -- "$toolkit")"
ops_name="$(basename -- "$ops")"
[[ "$toolkit_name" =~ ^Ascend-cann-toolkit_([^_]+)_linux-x86_64\.run$ ]] || fail 'Expected an x86_64 Toolkit .run filename.'
toolkit_version="${BASH_REMATCH[1]}"
[[ "$ops_name" =~ ^Ascend-cann-([A-Za-z0-9]+)-ops_([^_]+)_linux-x86_64\.run$ ]] || fail 'Expected an x86_64 ops .run filename.'
ops_soc="${BASH_REMATCH[1]}"; ops_version="${BASH_REMATCH[2]}"
soc_normalized="$(printf '%s' "$soc" | tr '[:upper:]' '[:lower:]')"
ops_soc_normalized="$(printf '%s' "$ops_soc" | tr '[:upper:]' '[:lower:]')"
[[ "$soc_normalized" == "$ops_soc_normalized" ]] || fail "--soc $soc does not match ops package model $ops_soc."
[[ "$toolkit_version" == "$ops_version" ]] || fail "Version mismatch: Toolkit $toolkit_version, ops $ops_version."
[[ "$torch_version" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]] || fail 'Torch version must be numeric, e.g. 2.10.0.'
[[ "$torch_npu_version" =~ ^[0-9]+(\.[0-9]+){1,2}(\.post[0-9]+)?$ ]] || fail 'TorchNPU version must look like 2.10.0.post4.'
[[ "$torch_npu_version" == "$torch_version" || "$torch_npu_version" == "$torch_version".post* ]] || fail 'TorchNPU package version must match torch version.'
[[ -n "$torch_wheel" && -n "$torch_npu_wheel" ]] || fail 'Specify both --torch-wheel and --torch-npu-wheel.'
# 当前环境固定 Python 3.10 / x86_64，wheel 文件名必须与版本参数一致。
torch_wheel_name=""
[[ -s "$torch_wheel" ]] || fail "Torch wheel $torch_wheel does not exist or is empty."
torch_wheel_name="$(basename -- "$torch_wheel")"
[[ "$torch_wheel_name" == "torch-${torch_version}+cpu-cp310-cp310-manylinux_2_28_x86_64.whl" ]] || fail 'Expected the matching Python 3.10 x86_64 CPU torch wheel filename.'
torch_npu_wheel_name=""
[[ -s "$torch_npu_wheel" ]] || fail "TorchNPU wheel $torch_npu_wheel does not exist or is empty."
torch_npu_wheel_name="$(basename -- "$torch_npu_wheel")"
[[ "$torch_npu_wheel_name" == "torch_npu-${torch_npu_version}-cp310-cp310-manylinux_2_28_x86_64.whl" ]] || fail 'Expected the matching Python 3.10 x86_64 TorchNPU wheel filename.'
tag="ascend-dev:${toolkit_version}-${soc_normalized}-${torch_version}"
[[ -s "$script_dir/Dockerfile" && -s "$script_dir/entrypoint.sh" ]] || fail 'Docker build files are missing.'
command -v docker >/dev/null 2>&1 || fail 'Docker is unavailable.'
docker image inspect "$base_image" >/dev/null 2>&1 || fail "Base image $base_image is not available locally; load or tag it before building."
base_arch="$(docker image inspect --format '{{.Architecture}}' "$base_image")"
[[ "$base_arch" == amd64 ]] || fail "Base image $base_image is $base_arch; an amd64 image is required."
[[ "$mirror_domain" =~ ^https?://[A-Za-z0-9.-]+(:[0-9]+)?/?$ ]] || fail 'Mirror domain must be an http(s) origin without a path.'
context_dir="$(mktemp -d)"
# 临时上下文将任意宿主机路径的安装包统一命名为 toolkit.run/ops.run。
# trap 保证成功或失败后都清理宿主机的临时副本。
trap 'rm -rf "$context_dir"' EXIT
mkdir -p "$context_dir/packages"
cp "$script_dir/Dockerfile" "$script_dir/entrypoint.sh" "$context_dir/"
cp "$toolkit" "$context_dir/packages/toolkit.run"
cp "$ops" "$context_dir/packages/ops.run"
cp "$torch_wheel" "$context_dir/packages/$torch_wheel_name"
cp "$torch_npu_wheel" "$context_dir/packages/$torch_npu_wheel_name"
printf 'Building CANN %s for %s (x86_64)\n' "$toolkit_version" "$ops_soc"
DOCKER_BUILDKIT=1 docker build --pull=false --platform linux/amd64 --progress=plain \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "MIRROR_DOMAIN=$mirror_domain" \
    --build-arg "TORCH_VERSION=$torch_version" \
    --build-arg "TORCH_NPU_VERSION=$torch_npu_version" \
    --build-arg "TORCH_WHEEL_FILE=$torch_wheel_name" \
    --build-arg "TORCH_NPU_WHEEL_FILE=$torch_npu_wheel_name" \
    -f "$context_dir/Dockerfile" -t "$tag" "$context_dir"
printf 'Built %s\n' "$tag"
