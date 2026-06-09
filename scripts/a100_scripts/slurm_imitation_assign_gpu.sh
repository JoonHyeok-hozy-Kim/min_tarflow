#!/bin/bash
# ============================================================
# Local Launcher (Slurm Imitation - Manual GPU Assignment)
#
# Usage:
#   bash slurm_imitation.sh <TARGET_GPUS> <TARGET_SCRIPT_PATH>
#
# Example:
#   bash slurm_imitation.sh "6,7" scripts/svd_rank/my_script.sh
# ============================================================

set -euo pipefail

# 1. 인자(Arguments) 검증 및 할당
if [ $# -lt 2 ]; then
    echo "Usage: bash $0 <TARGET_GPUS> <TARGET_SCRIPT_PATH>" >&2
    echo "Example: bash $0 \"6,7\" scripts/svd_rank/my_script.sh" >&2
    exit 1
fi

# 이제 첫 번째 인자는 "개수"가 아니라 "사용할 GPU 번호 문자열"입니다.
ARG_TARGET_GPUS="$1"
ARG_TARGET_SCRIPT="$2"

# 콤마로 구분된 문자열을 파싱해서 실제 사용할 GPU 개수를 구합니다.
IFS=',' read -r -a gpu_array <<< "$ARG_TARGET_GPUS"
CALCULATED_NUM_GPUS=${#gpu_array[@]}

# 작업 이름은 타겟 스크립트의 파일명에서 확장자를 제외하고 자동 생성합니다.
JOB_NAME=$(basename "$ARG_TARGET_SCRIPT" .sh)

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# 입력받은 스크립트 경로가 절대 경로가 아니면 REPO_ROOT를 기준으로 찾습니다.
if [[ "$ARG_TARGET_SCRIPT" = /* ]]; then
    SCRIPT="$ARG_TARGET_SCRIPT"
else
    SCRIPT="${REPO_ROOT}/${ARG_TARGET_SCRIPT}"
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Target script not found at -> $SCRIPT" >&2
    exit 1
fi

# 환경 변수 초기화
NUM_GPUS="${NUM_GPUS:-$CALCULATED_NUM_GPUS}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
FOREGROUND="${FOREGROUND:-0}"

if [ "$MAX_CONCURRENT" -lt 1 ]; then
    echo "MAX_CONCURRENT must be >= 1 (got ${MAX_CONCURRENT})" >&2
    exit 1
fi

CONFIGS=()
LAUNCHED_PIDS=()
LAST_PID=""
LAST_LOGFILE=""

_add() {
    local name="$1" config_path="$2" run_root="$3" overrides="${4:-}"
    CONFIGS+=("${name}|${config_path}|${run_root}|${overrides}")
}

run() {
    local name=$1
    shift

    local env_arr=("EXP_NAME=$name" "REPO_ROOT=$REPO_ROOT")
    local num_gpus="$NUM_GPUS"
    local job_name="$name"

    LAST_PID=""
    LAST_LOGFILE=""

    for kv in "$@"; do
        case "$kv" in
            PARTITION=*|CPUS=*|MEM=*|TIME=*|AFTER=*)
                : ;;
            GPUS=*)
                num_gpus="${kv#GPUS=}"
                ;;
            JOB_NAME=*)
                job_name="${kv#JOB_NAME=}"
                ;;
            *)
                env_arr+=("$kv")
                ;;
        esac
    done

    # ==========================================
    # 핵심 변경점: 빈 GPU를 찾는 로직 대신, 입력받은 GPU 번호를 그대로 사용합니다.
    # ==========================================
    local cuda_devs="$ARG_TARGET_GPUS"

    local master_port=$((29500 + RANDOM % 500))

    local launcher_name=$(basename "$0" .sh)
    local log_dir="${REPO_ROOT}/logs/${launcher_name}/${name}"
    mkdir -p "$log_dir"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)

    local logfile="${log_dir}/${timestamp}.log"

    printf "  %-40s -> GPU(s) [%s]  port=%s\n" \
        "$name" "$cuda_devs" "$master_port" >&2

    if [ "$FOREGROUND" = "1" ]; then
        env \
            CUDA_VISIBLE_DEVICES="$cuda_devs" \
            NUM_GPUS="$num_gpus" \
            MASTER_PORT="$master_port" \
            JOB_ID="${job_name}_${timestamp}" \
            "${env_arr[@]}" \
            bash "$SCRIPT"
    else
        printf "    log: %s\n" "$logfile" >&2
        env \
            CUDA_VISIBLE_DEVICES="$cuda_devs" \
            NUM_GPUS="$num_gpus" \
            MASTER_PORT="$master_port" \
            JOB_ID="${job_name}_${timestamp}" \
            "${env_arr[@]}" \
            bash "$SCRIPT" >"$logfile" 2>&1 &
        LAST_PID=$!
        LAST_LOGFILE="$logfile"
        printf "    PID: %s\n" "$LAST_PID" >&2
    fi
}

_launch_config() {
    local line="$1"
    local name config_path run_root overrides
    IFS='|' read -r name config_path run_root overrides <<< "$line"

    # GPU 탐색(_wait_for_slot) 대기 로직도 제거했습니다. 즉시 실행됩니다.

    run "$name" \
        "CONFIG_PATH=$config_path" \
        "RUN_ROOT=$run_root" \
        "DISABLE_WANDB=${DISABLE_WANDB:-1}" \
        "GPUS=$NUM_GPUS" \
        "JOB_NAME=$name" \
        "CONFIG_OVERRIDES=$overrides"

    if [ "$FOREGROUND" != "1" ]; then
        LAUNCHED_PIDS+=("$LAST_PID")
    fi
}

_wait_for_all_jobs() {
    local failed=0
    local pid

    for pid in "${LAUNCHED_PIDS[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    return "$failed"
}

echo "=== Local Manual Launcher ==="

# ------------------------------------------------------------
# 1. 터미널에서 입력받은 타겟 스크립트 이름으로 단일 작업을 자동 등록합니다.
# ------------------------------------------------------------
_add "$JOB_NAME" "" "" ""

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs generated." >&2
    exit 1
fi

echo "=== Launching ${#CONFIGS[@]} job(s); assigned_gpus=[${ARG_TARGET_GPUS}] ===" >&2
for i in "${!CONFIGS[@]}"; do
    printf "  [%2d] %s\n" "$i" "${CONFIGS[$i]%%|*}" >&2
done

for config_line in "${CONFIGS[@]}"; do
    _launch_config "$config_line"
done

if [ "$FOREGROUND" != "1" ]; then
    if ! _wait_for_all_jobs; then
        echo "=== Some local jobs failed ===" >&2
        exit 1
    fi
fi

echo "=== Done ===" >&2