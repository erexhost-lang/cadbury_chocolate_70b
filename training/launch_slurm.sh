#!/bin/bash
# ==============================================================================
# SLURM Cluster Submission Script for Cadbury-Chocolate 70B Pretraining
# ==============================================================================
# Target Cluster Hardware: 4 Nodes x 8 NVIDIA H100/A100 (80GB) GPUs = 32 GPUs Total
# Network: InfiniBand HDR/NDR (400Gbps) with NCCL
# ==============================================================================

#SBATCH --job-name=cadbury-70b-pretrain
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # 1 primary task per node (torchrun manages the 8 GPUs)
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-hopper
#SBATCH --output=logs/cadbury_70b_%j.out
#SBATCH --error=logs/cadbury_70b_%j.err

# Exit immediately if a command exits with a non-zero status
set -e

# Load necessary cluster environment modules
module purge
module load cuda/12.2
module load nccl/2.18
module load cudnn/8.9

# Activate Python virtual environment
source /opt/venvs/cadbury-env/bin/activate

# Discover Master Node Address and Port for distributed synchronization
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent hosts "$MASTER_NODE" | awk '{print $1}')
export MASTER_PORT=29500

echo "=== Cluster Initialization ==="
echo "Total Nodes Allocated: $SLURM_NNODES"
echo "Primary Master Node: $MASTER_ADDR:$MASTER_PORT"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="

# High-Performance NCCL Network Environment Variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                    # Enable InfiniBand
export NCCL_IB_HCA=mlx5                     # Mellanox InfiniBand adapter
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0,ib0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# Memory optimizations for CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch multi-node torchrun via SLURM's srun
srun --output=logs/node_%t.log torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/pretrain_fsdp.py \
    --micro-batch-size 1 \
    --grad-accum-steps 16 \
    --seq-len 8192 \
    --max-steps 100000 \
    --output-dir checkpoints/cadbury-70b
