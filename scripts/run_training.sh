#!/bin/bash
# SSM-ASR Training Launch Script
#
# Usage:
#   ./scripts/run_training.sh                    # Default 2 GPU training
#   ./scripts/run_training.sh --gpus 4           # 4 GPU training
#   ./scripts/run_training.sh --config custom.yaml
#   ./scripts/run_training.sh --resume outputs/ssm_19m_en/checkpoint_epoch_10.pt

set -e

# Default values
GPUS=${GPUS:-2}
CONFIG=${CONFIG:-"configs/ssm_19m_a100.yaml"}
RESUME=""
MASTER_PORT=${MASTER_PORT:-29500}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "==========================================="
echo "SSM-ASR Training"
echo "==========================================="
echo "GPUs: $GPUS"
echo "Config: $CONFIG"
echo "Master Port: $MASTER_PORT"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
fi
echo "==========================================="

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Set CUDA devices if not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Generate comma-separated list of GPUs
    CUDA_DEVICES=$(seq -s, 0 $((GPUS - 1)))
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
fi

# Set environment variables for reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Set environment for better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run training
echo ""
echo "Starting training..."
echo ""

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    scripts/train_distributed.py \
    --config "$CONFIG" \
    $RESUME
