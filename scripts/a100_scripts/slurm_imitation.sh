#!/bin/bash
# ============================================================
# Local Launcher (Slurm Imitation)
#
# Usage:
#   bash slurm_imitation.sh <NUM_GPUS> <TARGET_SCRIPT_PATH>
#
# Example:
#   bash slurm_imitation.sh 2 scripts/my_train.sh
# ============================================================

set -euo pipefail

# 1. 인자(Arguments) 검증 및 할당
if [ $# -lt 2 ]; then
    echo "Usage: bash $0 <NUM_GPUS> <TARGET_SCRIPT_PATH>" >&2
    echo "Example: bash $0 2 scripts/svd_rank/my_script.sh" >&2
    exit 1
fi

ARG_NUM_GPUS="$1"
ARG_TARGET_SCRIPT="$2"

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

# 환경 변수가 없으면 인자로 받은 값을 우선적으로 사용합니다.
NUM_GPUS="${NUM_GPUS:-$ARG_NUM_GPUS}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
FOREGROUND="${FOREGROUND:-0}"
SCHEDULER_SLEEP_SECONDS="${SCHEDULER_SLEEP_SECONDS:-5}"

if [ "$MAX_CONCURRENT" -lt 1 ]; then
    echo "MAX_CONCURRENT must be >= 1 (got ${MAX_CONCURRENT})" >&2
    exit 1
fi

CONFIGS=()
LAUNCHED_PIDS=()
LAST_PID=""
LAST_LOGFILE=""
LAST_CUDA_DEVS=""
declare -A RESERVED_GPU_PIDS=()

_add() {
    local name="$1" config_path="$2" run_root="$3" overrides="${4:-}"
    CONFIGS+=("${name}|${config_path}|${run_root}|${overrides}")
}

_refresh_gpu_reservations() {
    local gpu pid
    for gpu in "${!RESERVED_GPU_PIDS[@]}"; do
        pid="${RESERVED_GPU_PIDS[$gpu]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            unset "RESERVED_GPU_PIDS[$gpu]"
        fi
    done
}

_active_jobs() {
    _refresh_gpu_reservations

    local count=0
    local pid
    local -A seen=()
    for pid in "${RESERVED_GPU_PIDS[@]}"; do
        if [ -z "${seen[$pid]:-}" ]; then
            seen["$pid"]=1
            count=$((count + 1))
        fi
    done
    printf '%s\n' "$count"
}

pick_free_gpus() {
    local n=$1
    local gpu_idx mem_used
    local -a candidates=()

    _refresh_gpu_reservations

    while IFS=',' read -r gpu_idx mem_used; do
        gpu_idx="${gpu_idx// /}"
        [ -n "$gpu_idx" ] || continue
        if [ -z "${RESERVED_GPU_PIDS[$gpu_idx]:-}" ]; then
            candidates+=("$gpu_idx")
        fi
    done < <(
        nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits \
            | sort -t',' -k2 -n
    )

    if [ ${#candidates[@]} -lt "$n" ]; then
        echo "ERROR: Requested $n GPU(s) but only ${#candidates[@]} unreserved GPU(s) found." >&2
        return 1
    fi

    printf '%s\n' "${candidates[@]:0:$n}" | paste -sd','
}

_wait_for_slot() {
    local requested_gpus="$1"
    while true; do
        if [ "$(_active_jobs)" -lt "$MAX_CONCURRENT" ] && pick_free_gpus "$requested_gpus" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$SCHEDULER_SLEEP_SECONDS"
    done
}

_reserve_gpus_for_pid() {
    local pid="$1" cuda_devs="$2" gpu
    local -a gpu_list=()
    IFS=',' read -r -a gpu_list <<< "$cuda_devs"
    for gpu in "${gpu_list[@]}"; do
        RESERVED_GPU_PIDS["$gpu"]="$pid"
    done
}

run() {
    local name=$1
    shift

    local env_arr=("EXP_NAME=$name" "REPO_ROOT=$REPO_ROOT")
    local num_gpus="$NUM_GPUS"
    local job_name="$name"

    LAST_PID=""
    LAST_LOGFILE=""
    LAST_CUDA_DEVS=""

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

    local cuda_devs
    cuda_devs=$(pick_free_gpus "$num_gpus")

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
        LAST_CUDA_DEVS="$cuda_devs"
        _reserve_gpus_for_pid "$LAST_PID" "$cuda_devs"
        printf "    PID: %s\n" "$LAST_PID" >&2
    fi
}

_launch_config() {
    local line="$1"
    local name config_path run_root overrides
    IFS='|' read -r name config_path run_root overrides <<< "$line"

    if [ "$FOREGROUND" != "1" ]; then
        _wait_for_slot "$NUM_GPUS"
    fi

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

echo "=== Local Slurm-Like Launcher Example ==="

# ------------------------------------------------------------
# 1. 터미널에서 입력받은 타겟 스크립트 이름으로 단일 작업을 자동 등록합니다.
# ------------------------------------------------------------
_add "$JOB_NAME" "" "" ""

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs generated." >&2
    exit 1
fi

echo "=== Launching ${#CONFIGS[@]} job(s); max_concurrent=${MAX_CONCURRENT}; gpus_per_job=${NUM_GPUS} ===" >&2
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