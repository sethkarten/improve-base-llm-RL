#!/bin/bash
#SBATCH --job-name=pokechampv2-gemma-rl
#SBATCH --output=logs/gemma_rl_h100_%j.out
#SBATCH --error=logs/gemma_rl_h100_%j.err
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=72:00:00
#SBATCH --partition=pli
#SBATCH --account=multiagentllm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sk9014@princeton.edu

echo "=========================================="
echo "Metamon Gemma RL Training - 4x H100"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: google/gemma-3-270m + QLoRA"
echo "  - Agent: GemmaLMAgent (LM head as actor)"
echo "  - Batch size per GPU: 32"
echo "  - Gradient accumulation: 1"
echo "  - Effective batch size: 128 (32 * 1 * 4)"
echo "  - Epochs: 40"
echo "  - Learning rate: 1.5e-4"
echo "  - Gradient clip: 1.5"
echo "  - Format: gen9ou"
echo "  - Quantization: QLoRA (4-bit)"
echo "  - Gradient Checkpointing: ENABLED"
echo "  - Mixed Precision: bfloat16"
echo "  - Attention: Flash Attention 2"
echo "  - Mode: OFFLINE (WandB)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Set cache directory
export METAMON_CACHE_DIR=/scratch/gpfs/CHIJ/milkkarten/.pokemon_cache/

# Set environment variables for offline mode
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Disable tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Enable TF32 for faster matmuls on H100
export NVIDIA_TF32_OVERRIDE=1

# Master port for distributed training
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# PyTorch memory allocator optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Print GPU info
nvidia-smi
echo ""

# Print Python/PyTorch info
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); torch.set_float32_matmul_precision('high'); print(f'TF32 matmul precision: {torch.get_float32_matmul_precision()}')"
echo ""

echo "Starting actor-critic training..."
echo ""

# Run training with accelerate for multi-GPU
# AMAGO uses Accelerate internally for distributed training
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --multi_gpu \
    --main_process_port $MASTER_PORT \
    metamon/rl/train.py \
    --run_name gemma_actor_critic_h100 \
    --model_gin_config gemma_lm_actor.gin \
    --train_gin_config binary_rl.gin \
    --save_dir /scratch/gpfs/sk9014/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 32 \
    --grad_accum 1 \
    --epochs 40 \
    --dloader_workers 32 \
    --eval_gens \
    --log

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Checkpoints: /scratch/gpfs/sk9014/metamon_checkpoints/gemma_actor_critic_h100"
echo "WandB logs: /scratch/gpfs/sk9014/metamon_checkpoints/gemma_actor_critic_h100/wandb_logs"
echo ""
echo "To sync WandB logs later:"
echo "  cd /scratch/gpfs/sk9014/metamon_checkpoints/gemma_actor_critic_h100/wandb_logs"
echo "  wandb sync --sync-all"
echo "=========================================="
