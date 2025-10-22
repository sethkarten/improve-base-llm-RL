#!/bin/bash
#SBATCH --job-name=pokechampv2-gemma-lm
#SBATCH --output=logs/gemma_lm_h100_%j.out
#SBATCH --error=logs/gemma_lm_h100_%j.err
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --partition=pli
#SBATCH --account=multiagentllm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sk9014@princeton.edu

echo "=========================================="
echo "Metamon Gemma LM RL Training - 2 GPUs"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: google/gemma-3-270m (FULL FINE-TUNING)"
echo "  - Agent: GemmaLMAgent (LM head as actor)"
echo "  - Batch size per GPU: 16"
echo "  - Gradient accumulation: 8"
echo "  - Effective batch size: 256 (16 * 8 * 2)"
echo "  - Epochs: 10"
echo "  - Max seq length: 16"
echo "  - Format: gen9ou"
echo "  - Quantization: NONE (full bf16)"
echo "  - LoRA: DISABLED (full fine-tuning)"
echo "  - Gradient Checkpointing: ENABLED"
echo "  - Mixed Precision: bfloat16"
echo "  - Attention: Flash Attention"
echo "  - Validation: NONE (Gen 9 OU vs baselines)"
echo "  - Mode: OFFLINE (HuggingFace + WandB)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# # Activate conda environment
# source ~/anaconda3/etc/profile.d/conda.sh
export METAMON_CACHE_DIR=/scratch/gpfs/CHIJ/milkkarten/.pokemon_cache/

# Set environment variables for offline mode
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Disable tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# H100-specific NCCL optimizations
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=5
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_P2P_LEVEL=NVL

# Enable TF32 for faster matmuls on H100 (Ampere+)
export NVIDIA_TF32_OVERRIDE=1

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

echo "Starting training..."
echo ""

# cd showdown
# nohup node pokemon-showdown start --no-security 2>&1 >showdown.log &
# cd ..

# Run training with accelerate for multi-GPU
# AMAGO uses Accelerate internally for distributed training
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --multi_gpu \
    --main_process_port $MASTER_PORT \
    metamon/rl/train.py \
    --run_name gemma_lm_h100_full_ft \
    --model_gin_config gemma_lm_actor_h100.gin \
    --train_gin_config binary_rl.gin \
    --save_dir /scratch/gpfs/sk9014/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 16 \
    --grad_accum 8 \
    --epochs 10 \
    --dloader_workers 32 \
    --eval_gens \
    --log

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Checkpoints: ~/metamon_checkpoints/gemma_lm_h100_full_ft"
echo "WandB logs: ~/metamon_checkpoints/gemma_lm_h100_full_ft/wandb_logs"
echo ""
echo "To sync WandB logs later:"
echo "  cd ~/metamon_checkpoints/gemma_lm_h100_full_ft/wandb_logs"
echo "  wandb sync --sync-all"
echo "=========================================="
