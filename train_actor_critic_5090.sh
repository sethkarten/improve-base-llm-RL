#!/bin/bash
# Metamon Full Actor-Critic Training - Local RTX 5090
# Stage 2: Train both actor and critic together
# Run AFTER train_critic_warmup_5090.sh

set -e  # Exit on error

echo "=========================================="
echo "Metamon Actor-Critic - RTX 5090"
echo "=========================================="
echo "Stage: 2 (Full Training)"
echo "Device: RTX 5090"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Mode: Actor-critic (both train)"
echo "  - Model: google/gemma-3-270m"
echo "  - Agent: GemmaLMAgent"
echo "  - Batch size per GPU: 12"
echo "  - Gradient accumulation: 1"
echo "  - Epochs: 40"
echo "  - Format: gen9ou"
echo "  - offline_coeff: 1.0 (Binary Filtered BC)"
echo "  - freeze_actor: False"
echo "  - Resume from: epoch 5 (critic warmup)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Set METAMON_CACHE_DIR
if [ -z "$METAMON_CACHE_DIR" ]; then
    export METAMON_CACHE_DIR="/media/milkkarten/data/.cache/metamon"
    echo "Setting METAMON_CACHE_DIR=$METAMON_CACHE_DIR"
fi

# Environment variables
export CUDA_VISIBLE_DEVICES=0  # Single RTX 5090
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Print GPU info
nvidia-smi || echo "Warning: nvidia-smi not available"
echo ""

# Print Python/PyTorch info
/home/milkkarten/anaconda3/envs/metamon/bin/python --version
/home/milkkarten/anaconda3/envs/metamon/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "Starting full actor-critic training..."
echo ""

# Run training - resume from critic warmup
/home/milkkarten/anaconda3/envs/metamon/bin/python metamon/rl/train.py \
    --run_name gemma_actor_critic_5090 \
    --model_gin_config gemma_lm_actor.gin \
    --train_gin_config binary_rl.gin \
    --save_dir ./checkpoints \
    --formats gen9ou \
    --batch_size_per_gpu 32 \
    --grad_accum 1 \
    --epochs 40 \
    --dloader_workers 4 \
    --eval_gens \
    --log

echo ""
echo "=========================================="
echo "Actor-critic training complete!"
echo "End time: $(date)"
echo "Checkpoints: ./checkpoints/gemma_actor_critic_5090"
echo "=========================================="
