#!/bin/bash
# Metamon Full Actor-Critic Training - 4x H100
# Single-stage training (no critic warmup - it wasn't faster)
# Uses same validated settings as 5090 with larger batch size

set -e  # Exit on error

echo "=========================================="
echo "Metamon Actor-Critic - 4x H100"
echo "=========================================="
echo "Device: 4x H100 (80GB each)"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: google/gemma-3-270m"
echo "  - Agent: GemmaLMAgent"
echo "  - Batch size per GPU: 32"
echo "  - Effective batch: 128 (32 × 4 GPUs)"
echo "  - Gradient accumulation: 1"
echo "  - Epochs: 40"
echo "  - Format: gen9ou"
echo "  - Learning rate: 1.5e-4"
echo "  - Gradient clip: 1.5"
echo "  - offline_coeff: 1.0 (Binary Filtered BC)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Set METAMON_CACHE_DIR
if [ -z "$METAMON_CACHE_DIR" ]; then
    export METAMON_CACHE_DIR="/media/milkkarten/data/.cache/metamon"
    echo "Setting METAMON_CACHE_DIR=$METAMON_CACHE_DIR"
fi

# Environment variables for multi-GPU
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Print GPU info
nvidia-smi || echo "Warning: nvidia-smi not available"
echo ""

# Print Python/PyTorch info
/home/milkkarten/anaconda3/envs/metamon/bin/python --version
/home/milkkarten/anaconda3/envs/metamon/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

echo "Starting actor-critic training..."
echo ""

# Run training
/home/milkkarten/anaconda3/envs/metamon/bin/python metamon/rl/train.py \
    --run_name gemma_actor_critic_h100 \
    --model_gin_config gemma_lm_actor.gin \
    --train_gin_config binary_rl.gin \
    --save_dir ./checkpoints \
    --formats gen9ou \
    --batch_size_per_gpu 32 \
    --grad_accum 1 \
    --epochs 40 \
    --dloader_workers 8 \
    --eval_gens \
    --log

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Checkpoints: ./checkpoints/gemma_actor_critic_h100"
echo "=========================================="
