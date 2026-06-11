#!/bin/bash

# Slurm setup
#SBATCH -p gu-compute
#SBATCH -A gu-account
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

FILE_DIR="inference/denoised_sampling/two_dimensional.py"
OUT_DIR="./logs/${FILE_DIR}"
mkdir -p "${OUT_DIR}"

DATE_WITH_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUT_DIR}/${DATE_WITH_TIME}_run_${SLURM_JOB_ID}.log"

exec > "$OUTPUT_FILE" 2>&1

echo "==================================="
date
echo "Job running on node: $(hostname)"
echo "==================================="

source ./venv/bin/activate
echo "[DEBUG] Python check:"
python --version

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"
echo "==================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "${FILE_DIR} starts at $(date)"
echo "==================================="

# exprt WANDB_API_KEY and WANDB_ENTITY from .env
echo "Wandb Settings"
set -a
source .env
set +a

export WANDB_CONFIG_DIR="$(pwd)/.wandb_config"
export WANDB_DIR="$(pwd)/wandb_logs"


echo "==================================="

# Args
WEIGHT="weights/reserved/two_dimensional/spiral/epoch_10000-loss_1.66.pth"
dataset_name=spiral
img_size=8
channel_size=1
num_flow_blocks=8
flow_block_dim=8
permutation_type=flip
num_attn_blocks=8
attn_num_heads=8
attn_head_dim=64
attn_temp=1.0
ffn_expansion=4
cfg_weight=0.0
lr=1e-5
num_samples=3000
annealed_guidance_flag=""


python -u ${FILE_DIR} \
    --pre_trained_weight_path "$WEIGHT" \
    --dataset_name "$dataset_name" \
    --img_size "$img_size" \
    --channel_size "$channel_size" \
    --num_flow_blocks "$num_flow_blocks" \
    --flow_block_dim "$flow_block_dim" \
    --permutation_type "$permutation_type" \
    --num_attn_blocks "$num_attn_blocks" \
    --attn_num_heads "$attn_num_heads" \
    --attn_head_dim "$attn_head_dim" \
    --attn_temp "$attn_temp" \
    --ffn_expansion "$ffn_expansion" \
    --cfg_weight "$cfg_weight" \
    --lr "$lr" \
    --num_samples "$num_samples" \
    $annealed_guidance_flag

echo "-----------------------------------"

echo "==================================="
echo "Fin at $(date)"