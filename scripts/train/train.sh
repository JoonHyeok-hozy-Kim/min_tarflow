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

FILE_NAME="train.py"
OUT_DIR="./logs/${FILE_NAME}"
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

echo "${FILE_NAME} starts at $(date)"
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
dataset_name=imagenet64
img_size=64
channel_size=3
patch_size=4
num_flow_blocks=4
flow_block_dim=512
num_attn_layers=4
perumtation_type=flip
attn_head_dim=64
attn_temp=1.0
ffn_multiplier=4
cfg_weight=0
batch_size=128
epochs=1000
lr=1e-5
lr_schedule_type=wsd
class_dropout_prob=0
sample_freq=10
num_samples=10
sample_batch_size=10
resume_wandb_url=false



python -u ${FILE_NAME} \
    --dataset_name "$dataset_name" \
    --img_size "$img_size" \
    --channel_size "$channel_size" \
    --patch_size "$patch_size" \
    --num_flow_blocks "$num_flow_blocks" \
    --flow_block_dim "$flow_block_dim" \
    --num_attn_layers "$num_attn_layers" \
    --perumtation_type "$perumtation_type" \
    --attn_head_dim "$attn_head_dim" \
    --attn_temp "$attn_temp" \
    --ffn_multiplier "$ffn_multiplier" \
    --cfg_weight "$cfg_weight" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --lr "$lr" \
    --lr_schedule_type "$lr_schedule_type" \
    --class_dropout_prob "$class_dropout_prob" \
    --sample_freq "$sample_freq" \
    --num_samples "$num_samples" \
    --sample_batch_size "$sample_batch_size" \
    --resume_wandb_url "$resume_wandb_url" \

echo "-----------------------------------"

echo "==================================="
echo "Fin at $(date)"