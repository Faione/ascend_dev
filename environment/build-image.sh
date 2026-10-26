#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'HELP'
Usage: bash build-image.sh --soc MODEL --cann-version VER --torch-version VER [options]
  --soc MODEL         Device model, e.g. 910b, 910, A3, 950, 310p
  --cann-version VER  Select Toolkit and ops .run files from cann-run/MODEL/VER
                      e.g. 9.1.1 or 9.2.0
  --cann-root DIR     Directory containing model folders (default: auto-detect cann-run)
  --base-image IMAGE  Local Ubuntu 22.04 base image (default: ubuntu:22.04)
  --mirror-domain URL  APT and pip mirror origin (default: http://mirrors.tools.huawei.com)
  --torch-version VER Select wheels from cann-run/torch/VER (e.g. 2.7.1 or 2.10.0)
  --torch-root DIR   Directory containing version folders (default: auto-detect cann-run/torch)
  -h, --help          Show help
Supports x86_64 packages and amd64 base images only.
HELP
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 参数可由命令行提供；镜像站和基础镜像有默认值。
soc=""; cann_version="${CANN_VERSION:-}"; cann_root="${CANN_ROOT:-}"; torch_root="${TORCH_ROOT:-}"
base_image="${BASE_IMAGE:-ubuntu:22.04}"
mirror_domain="${MIRROR_DOMAIN:-http://mirrors.tools.huawei.com}"
torch_version="${TORCH_VERSION:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --soc|--cann-version|--cann-root|--base-image|--mirror-domain|--torch-version|--torch-root)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
            case "$1" in
                --soc) soc="$2" ;;
                --cann-version) cann_version="$2" ;;
                --cann-root) cann_root="$2" ;;
                --base-image) base_image="$2" ;;
                --mirror-domain) mirror_domain="$2" ;;
                --torch-version) torch_version="$2" ;;
                --torch-root) torch_root="$2" ;;
            esac
            shift 2 ;;
        *) usage >&2; fail "Unknown argument: $1" ;;
    esac
done
[[ "$soc" =~ ^[A-Za-z0-9]+$ ]] || fail 'Specify --soc with a model such as 950 or 910.'
[[ "$cann_version" =~ ^[0-9]+(\.[0-9]+){2}$ ]] || fail 'Specify --cann-version, e.g. 9.1.1 or 9.2.0.'
soc_normalized="$(printf '%s' "$soc" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$cann_root" ]]; then
    for root in "$PWD/cann-run" "$PWD" "$(dirname -- "$PWD")" "$(dirname -- "$(dirname -- "$PWD")")" "$script_dir/../../cann-run"; do
        if [[ -d "$root/$soc" ]]; then
            cann_root="$root"; model_dir="$root/$soc"; break
        elif [[ -d "$root/$soc_normalized" ]]; then
            cann_root="$root"; model_dir="$root/$soc_normalized"; break
        fi
    done
    [[ -n "$cann_root" ]] || fail "Cannot find cann-run/$soc; use --cann-root DIR."
else
    model_dir="$cann_root/$soc"
    [[ -d "$model_dir" ]] || model_dir="$cann_root/$soc_normalized"
fi
run_dir="$model_dir/$cann_version"
[[ -d "$run_dir" ]] || fail "CANN package directory $run_dir does not exist."
shopt -s nullglob
toolkit_candidates=("$run_dir"/Ascend-cann-toolkit_*_linux-x86_64.run)
ops_candidates=()
for candidate in "$run_dir"/Ascend-cann-*-ops_*_linux-x86_64.run; do
    candidate_name="$(basename -- "$candidate")"
    [[ "$candidate_name" =~ ^Ascend-cann-([A-Za-z0-9]+)-ops_ ]] || continue
    candidate_soc="$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
    [[ "$candidate_soc" == "$soc_normalized" ]] && ops_candidates+=("$candidate")
done
shopt -u nullglob
[[ ${#toolkit_candidates[@]} -eq 1 ]] || fail "Expected exactly one Toolkit .run in $run_dir, found ${#toolkit_candidates[@]}."
[[ ${#ops_candidates[@]} -eq 1 ]] || fail "Expected exactly one $soc ops .run in $run_dir, found ${#ops_candidates[@]}."
toolkit="${toolkit_candidates[0]}"; ops="${ops_candidates[0]}"
[[ -s "$toolkit" && -s "$ops" ]] || fail 'CANN .run files must not be empty.'
# 从官方文件名提取 CANN 版本及设备型号，防止 Toolkit/ops 混搭。
toolkit_name="$(basename -- "$toolkit")"
ops_name="$(basename -- "$ops")"
[[ "$toolkit_name" =~ ^Ascend-cann-toolkit_([^_]+)_linux-x86_64\.run$ ]] || fail 'Expected an x86_64 Toolkit .run filename.'
toolkit_version="${BASH_REMATCH[1]}"
[[ "$ops_name" =~ ^Ascend-cann-([A-Za-z0-9]+)-ops_([^_]+)_linux-x86_64\.run$ ]] || fail 'Expected an x86_64 ops .run filename.'
ops_soc="${BASH_REMATCH[1]}"; ops_version="${BASH_REMATCH[2]}"
ops_soc_normalized="$(printf '%s' "$ops_soc" | tr '[:upper:]' '[:lower:]')"
[[ "$soc_normalized" == "$ops_soc_normalized" ]] || fail "--soc $soc does not match ops package model $ops_soc."
[[ "$toolkit_version" == "$ops_version" ]] || fail "Version mismatch: Toolkit $toolkit_version, ops $ops_version."
[[ "$toolkit_version" == "$cann_version" ]] || fail "Package version $toolkit_version does not match directory version $cann_version."
[[ "$torch_version" == 2.7.1 || "$torch_version" == 2.10.0 ]] || fail 'Specify --torch-version 2.7.1 or 2.10.0.'
if [[ -z "$torch_root" ]]; then
    if [[ -d "$PWD/cann-run/torch/$torch_version" ]]; then
        torch_root="$PWD/cann-run/torch"
    elif [[ -d "$PWD/torch/$torch_version" ]]; then
        torch_root="$PWD/torch"
    elif [[ "$(basename -- "$PWD")" == "$torch_version" && "$(basename -- "$(dirname -- "$PWD")")" == torch ]]; then
        torch_root="$(dirname -- "$PWD")"
    elif [[ -d "$script_dir/../../cann-run/torch/$torch_version" ]]; then
        torch_root="$script_dir/../../cann-run/torch"
    else
        fail "Cannot find cann-run/torch/$torch_version; use --torch-root DIR."
    fi
fi
wheel_dir="$torch_root/$torch_version"
[[ -d "$wheel_dir" ]] || fail "Torch wheel directory $wheel_dir does not exist."
shopt -s nullglob
torch_candidates=("$wheel_dir"/torch-"$torch_version"*.whl)
npu_candidates=("$wheel_dir"/torch_npu-"$torch_version"*.whl)
shopt -u nullglob
[[ ${#torch_candidates[@]} -eq 1 ]] || fail "Expected exactly one torch wheel in $wheel_dir, found ${#torch_candidates[@]}."
[[ ${#npu_candidates[@]} -eq 1 ]] || fail "Expected exactly one torch_npu wheel in $wheel_dir, found ${#npu_candidates[@]}."
torch_wheel="${torch_candidates[0]}"; torch_npu_wheel="${npu_candidates[0]}"
[[ -s "$torch_wheel" && -s "$torch_npu_wheel" ]] || fail 'Torch wheel files must not be empty.'
torch_wheel_name="$(basename -- "$torch_wheel")"
torch_npu_wheel_name="$(basename -- "$torch_npu_wheel")"
# 当前镜像使用 Python 3.10 / x86_64；TorchNPU 的 post/dev 后缀不参与版本选择。
torch_pattern="^torch-(${torch_version//./\\.}(\+cpu)?)-cp310-cp310-(manylinux[^/]*|linux)_x86_64\.whl$"
npu_pattern="^torch_npu-${torch_version//./\\.}(\.post[0-9]+)?(\.dev[0-9]+)?-cp310-cp310-(manylinux[^/]*|linux)_x86_64\.whl$"
[[ "$torch_wheel_name" =~ $torch_pattern ]] || fail "Unsupported Python/platform or torch wheel filename: $torch_wheel_name"
torch_package_version="${BASH_REMATCH[1]}"
[[ "$torch_npu_wheel_name" =~ $npu_pattern ]] || fail "Unsupported Python/platform or torch_npu wheel filename: $torch_npu_wheel_name"
torch_npu_version="$torch_version"
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
    --build-arg "TORCH_VERSION=$torch_package_version" \
    --build-arg "TORCH_NPU_VERSION=$torch_npu_version" \
    --build-arg "TORCH_WHEEL_FILE=$torch_wheel_name" \
    --build-arg "TORCH_NPU_WHEEL_FILE=$torch_npu_wheel_name" \
    -f "$context_dir/Dockerfile" -t "$tag" "$context_dir"
printf 'Built %s\n' "$tag"
