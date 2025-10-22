#!/bin/bash
# Metamon Full Actor-Critic Training - 4x H100
# Stage 2: Train both actor and critic together
# Run AFTER train_critic_warmup_h100.sh
# Uses same settings as 5090 but with larger batch size for H100's 80GB memory

set -e  # Exit on error

echo "=========================================="
echo "Metamon Actor-Critic - 4x H100"
echo "=========================================="
echo "Stage: 2 (Full Training)"
echo "Device: 4x H100 (80GB each)"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Mode: Actor-critic (both train)"
echo "  - Model: google/gemma-3-270m"
echo "  - Agent: GemmaLMAgent"
echo "  - Batch size per GPU: 32 (2x 5090's 16)"
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

echo "Starting full actor-critic training..."
echo ""

# Run training - resume from critic warmup
/home/milkkarten/anaconda3/envs/metamon/bin/python metamon/rl/train.py \
    --run_name gemma_actor_critic_h100 \
    --model_gin_config gemma_lm_actor.gin \
    --train_gin_config binary_rl.gin \
    --save_dir ./checkpoints \
    --ckpt 5 \
    --formats gen9ou \
    --batch_size_per_gpu 32 \
    --grad_accum 1 \
    --epochs 40 \
    --dloader_workers 8 \
    --eval_gens \
    --log

echo ""
echo "=========================================="
echo "Actor-critic training complete!"
echo "End time: $(date)"
echo "Checkpoints: ./checkpoints/gemma_actor_critic_h100"
echo "=========================================="
