#!/bin/bash
# Metamon Critic Warmup Training - 4x H100
# Stage 1: Train only the critic (value head) - no actor training
# Uses same settings as 5090 but with larger batch size for H100's 80GB memory

set -e  # Exit on error

echo "=========================================="
echo "Metamon Critic Warmup - 4x H100"
echo "=========================================="
echo "Stage: 1 (Critic Warmup)"
echo "Device: 4x H100 (80GB each)"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Mode: Critic warmup (no actor training)"
echo "  - Model: google/gemma-3-270m"
echo "  - Agent: GemmaLMAgent"
echo "  - Batch size per GPU: 128 (2x 5090)"
echo "  - Gradient accumulation: 1"
echo "  - Epochs: 5 (warmup only)"
echo "  - Format: gen9ou"
echo "  - offline_coeff: 0.0 (no actor loss)"
echo "  - freeze_actor: False (disabled - was slower)"
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

echo "Starting critic warmup training..."
echo ""

# Run training with accelerate for multi-GPU
/home/milkkarten/anaconda3/envs/metamon/bin/python metamon/rl/train.py \
    --run_name gemma_critic_warmup_h100 \
    --model_gin_config gemma_lm_actor.gin \
    --train_gin_config binary_rl_critic_warmup.gin \
    --save_dir ./checkpoints \
    --formats gen9ou \
    --batch_size_per_gpu 128 \
    --grad_accum 1 \
    --epochs 5 \
    --dloader_workers 8 \
    --eval_gens \
    --log

echo ""
echo "=========================================="
echo "Critic warmup complete!"
echo "End time: $(date)"
echo "Checkpoints: ./checkpoints/gemma_critic_warmup_h100"
echo ""
echo "Next step: Run full actor-critic training"
echo "Command: ./train_actor_critic_h100.sh"
echo "=========================================="
