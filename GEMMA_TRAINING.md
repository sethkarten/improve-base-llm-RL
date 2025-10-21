# Training Metamon with Gemma 3 270M

This guide explains how to finetune Gemma 3 270M for Pokemon battle agents using the metamon framework.

## Overview

The Gemma integration replaces metamon's custom tokenizer and transformer with a pretrained Gemma 3 270M LLM. This allows the agent to leverage:
- **Pretrained world knowledge** from Gemma's pretraining
- **Better language understanding** for Pokemon move/ability descriptions
- **Transfer learning** to accelerate training
- **LoRA finetuning** for memory-efficient training

## Architecture

The Gemma-based agent uses a two-stage architecture:

1. **GemmaTstepEncoder**: Processes each battle turn through Gemma
   - Converts Pokemon tokens → text → Gemma embeddings
   - Fuses Gemma embeddings with numerical battle state
   - Output: ~2944-dim embedding (2688 Gemma + 256 numerical)

2. **TformerTrajEncoder**: Processes sequence of turns (unchanged from base metamon)
   - Transformer over turn embeddings
   - Output: 1024-dim (basic) or 1280-dim (SyntheticRLV2-style)

3. **Actor-Critic Heads**: Policy and value networks
   - **Basic**: 4x NCritics (standard ensemble)
   - **SyntheticRLV2**: 6x NCriticsTwoHot (distributional RL)

### Architecture Comparison

| Component | Basic Agent | SyntheticRLV2-Style |
|-----------|-------------|---------------------|
| **Agent type** | Agent | MultiTaskAgent |
| **Critic type** | NCritics | NCriticsTwoHot |
| **Num critics** | 4 | 6 |
| **Value estimation** | Point estimate | Distributional (96 bins) |
| **Trajectory encoder** | 1024-dim, 4 layers | 1280-dim, 9 layers |
| **Max seq len** | 200 | 128 |
| **Performance** | Good | Best (matches SyntheticRLV2) |
| **VRAM usage** | Moderate | High |
| **Training time** | Faster | Slower |

**Recommendation**: Use `gemma_synthetic_multitaskagent.gin` for best performance, or `gemma_agent.gin` for faster experimentation.

## Requirements

Install the required dependencies:

```bash
pip install transformers peft bitsandbytes accelerate
```

For 4-bit quantization support, ensure you have a CUDA-compatible GPU.

## Training Commands

### Standard Training (Full Precision)

For GPUs with ≥24GB VRAM (e.g., RTX 4090, A100):

```bash
python -m metamon.rl.train \
    --run_name gemma_gen9ou_v1 \
    --model_gin_config gemma_agent.gin \
    --train_gin_config il.gin \
    --save_dir ~/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 4 \
    --grad_accum 2 \
    --epochs 10 \
    --log
```

### 4-Bit Quantized Training (Limited VRAM)

For GPUs with 8-16GB VRAM (e.g., RTX 3080, RTX 5090):

```bash
python -m metamon.rl.train \
    --run_name gemma_gen9ou_4bit_v1 \
    --model_gin_config gemma_agent_4bit.gin \
    --train_gin_config il.gin \
    --save_dir ~/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 2 \
    --grad_accum 4 \
    --epochs 10 \
    --log
```

### Offline RL Training (Binary Filtered BC)

For more advanced training with advantage filtering:

```bash
python -m metamon.rl.train \
    --run_name gemma_gen9ou_binary_rl_v1 \
    --model_gin_config gemma_agent.gin \
    --train_gin_config binary_rl.gin \
    --save_dir ~/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 4 \
    --grad_accum 2 \
    --epochs 20 \
    --log
```

### SyntheticRLV2-Style Training (Recommended)

For the strongest performance using MultiTaskAgent with distributional RL:

```bash
python -m metamon.rl.train \
    --run_name gemma_synthetic_gen9ou_v1 \
    --model_gin_config gemma_synthetic_multitaskagent.gin \
    --train_gin_config binary_rl.gin \
    --save_dir ~/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 2 \
    --grad_accum 4 \
    --epochs 40 \
    --log
```

**Note**: The SyntheticRLV2-style architecture is larger (1280-dim trajectory encoder with 9 layers) and requires more VRAM. Use smaller batch sizes or the 4-bit version if needed.

For 4-bit quantized version:

```bash
python -m metamon.rl.train \
    --run_name gemma_synthetic_gen9ou_4bit_v1 \
    --model_gin_config gemma_synthetic_multitaskagent_4bit.gin \
    --train_gin_config binary_rl.gin \
    --save_dir ~/metamon_checkpoints/ \
    --formats gen9ou \
    --batch_size_per_gpu 1 \
    --grad_accum 8 \
    --epochs 40 \
    --log
```

### Resume Training from Checkpoint

To continue training from a previous checkpoint:

```bash
python -m metamon.rl.train \
    --run_name gemma_gen9ou_v1 \
    --model_gin_config gemma_agent.gin \
    --train_gin_config il.gin \
    --save_dir ~/metamon_checkpoints/ \
    --ckpt 5 \
    --formats gen9ou \
    --batch_size_per_gpu 4 \
    --grad_accum 2 \
    --epochs 20 \
    --log
```

This will load the checkpoint from epoch 5 and continue training.

## Configuration Files

### Basic Configurations

#### `gemma_agent.gin`
Standard Gemma configuration with full precision:
- **Agent**: Standard Agent
- **Critic**: 4x NCritics (ensemble value estimation)
- **Model**: google/gemma-3-270m
- **LoRA**: r=16, alpha=32
- **Quantization**: None
- **Trajectory encoder**: 1024-dim, 4 layers
- **Actor/Critic**: 512-dim hidden

#### `gemma_agent_4bit.gin`
Memory-efficient 4-bit quantized version:
- **Agent**: Standard Agent
- **Critic**: 4x NCritics (ensemble value estimation)
- **Model**: google/gemma-3-270m
- **LoRA**: r=16, alpha=32
- **Quantization**: 4-bit NF4
- **Trajectory encoder**: 1024-dim, 4 layers
- **Actor/Critic**: 512-dim hidden

### SyntheticRLV2-Style Configurations

#### `gemma_synthetic_multitaskagent.gin`
Gemma version of the powerful SyntheticRLV2 architecture:
- **Agent**: MultiTaskAgent (multi-task RL)
- **Critic**: 6x NCriticsTwoHot (distributional RL with two-hot encoding)
- **Model**: google/gemma-3-270m
- **LoRA**: r=16, alpha=32
- **Quantization**: None
- **Trajectory encoder**: 1280-dim, 9 layers (matches SyntheticRLV2)
- **Actor/Critic**: 512-dim hidden
- **Value range**: -1100 to +1100 (96 bins)

This is the recommended configuration for achieving the strongest performance, equivalent to the metamon SyntheticRLV2 baseline.

#### `gemma_synthetic_multitaskagent_4bit.gin`
4-bit quantized version of the SyntheticRLV2-style agent:
- Same as above but with 4-bit quantization for limited VRAM

## Training Configuration Files

Use these with the `--train_gin_config` argument:

- **`il.gin`**: Pure imitation learning (behavior cloning)
- **`binary_rl.gin`**: Binary filtered BC (only positive advantages)
- **`exp_rl.gin`**: Exponential advantage weighting (CQL-style)

## Hyperparameter Tuning

### Batch Size and Gradient Accumulation

Total effective batch size = `batch_size_per_gpu * grad_accum * num_gpus`

Recommended settings:
- **24GB GPU**: `batch_size_per_gpu=4`, `grad_accum=2` → effective batch size 8
- **16GB GPU**: `batch_size_per_gpu=2`, `grad_accum=4` → effective batch size 8
- **8GB GPU (4-bit)**: `batch_size_per_gpu=1`, `grad_accum=8` → effective batch size 8

### LoRA Parameters

Edit in the gin config file:
- **`lora_r`**: Rank (higher = more capacity, default: 16)
- **`lora_alpha`**: Scaling (higher = stronger adaptation, default: 32)
- **`lora_dropout`**: Dropout rate (default: 0.05)

For faster convergence on gen9ou only, try:
```gin
GemmaTstepEncoder.lora_r = 32
GemmaTstepEncoder.lora_alpha = 64
```

### Trajectory Encoder Size

For faster training, reduce the trajectory encoder size:
```gin
traj_encoders.TformerTrajEncoder.d_model = 512
traj_encoders.TformerTrajEncoder.n_layers = 3
```

## Expected Training Time

Approximate training times for 10 epochs on gen9ou (~2.8M replays):

| Hardware | Config | Batch Size | Time per Epoch | Total (10 epochs) |
|----------|--------|------------|----------------|-------------------|
| RTX 4090 | Full precision | 4 | ~8 hours | ~3.3 days |
| RTX 4090 | 4-bit | 4 | ~6 hours | ~2.5 days |
| RTX 5090 | Full precision | 8 | ~5 hours | ~2.1 days |
| RTX 5090 | 4-bit | 8 | ~4 hours | ~1.7 days |
| 4x H100 | Full precision | 16 | ~2 hours | ~20 hours |
| 4x H100 | 4-bit | 32 | ~1 hour | ~10 hours |

*Note: Times are approximate and depend on CPU, storage I/O, and other factors.*

## Memory Usage

Expected VRAM usage during training:

| Config | Model Size | Batch Size | VRAM Usage |
|--------|------------|------------|------------|
| Full precision | ~1.1GB | 4 | ~22GB |
| Full precision | ~1.1GB | 2 | ~14GB |
| 4-bit quantized | ~300MB | 4 | ~12GB |
| 4-bit quantized | ~300MB | 2 | ~8GB |
| 4-bit quantized | ~300MB | 1 | ~6GB |

## Evaluation

After training, evaluate your model against baselines:

```bash
python -m metamon.rl.evaluate \
    --run_name gemma_gen9ou_v1 \
    --ckpt_dir ~/metamon_checkpoints/ \
    --ckpt 10 \
    --formats gen9ou
```

## Custom Datasets

To train on your own replay data in addition to the main dataset:

```bash
python -m metamon.rl.train \
    --run_name gemma_gen9ou_custom_v1 \
    --model_gin_config gemma_agent.gin \
    --train_gin_config il.gin \
    --save_dir ~/metamon_checkpoints/ \
    --formats gen9ou \
    --custom_replay_dir ~/my_replays/ \
    --custom_replay_sample_weight 0.25 \
    --log
```

This will sample 25% of each batch from your custom dataset.

## Troubleshooting

### Out of Memory

1. Reduce `batch_size_per_gpu` and increase `grad_accum` proportionally
2. Use 4-bit quantization (`gemma_agent_4bit.gin`)
3. Reduce trajectory encoder size in the gin config
4. Reduce `max_seq_len` in the gin config (default: 200)

### Slow Training

1. Increase `batch_size_per_gpu` if VRAM allows
2. Use 4-bit quantization (faster than full precision)
3. Reduce `dloader_workers` if CPU-bound
4. Disable evaluation during training: `--eval_gens` (no arguments)

### Model Not Learning

1. Try different training configs (`binary_rl.gin`, `exp_rl.gin`)
2. Increase LoRA rank and alpha
3. Train for more epochs (20-30 recommended)
4. Check wandb logs for loss curves
5. Ensure dataset format is gen9ou (other formats may have different dynamics)

## Advanced: Custom Gin Configs

Create your own gin config by copying `gemma_agent.gin` and modifying parameters:

```bash
cp metamon/rl/configs/models/gemma_agent.gin my_custom_gemma.gin
# Edit my_custom_gemma.gin
python -m metamon.rl.train --model_gin_config my_custom_gemma.gin ...
```

Common modifications:
- Change Gemma model: `GemmaTstepEncoder.model_name = "google/gemma-3-1b"`
- Freeze Gemma: `GemmaTstepEncoder.freeze_base_model = True`
- Adjust LoRA: `GemmaTstepEncoder.lora_r = 32`

## Monitoring with WandB

Enable logging with `--log` flag and set environment variables:

```bash
export METAMON_WANDB_PROJECT="pokemon-rl"
export METAMON_WANDB_ENTITY="your_username"

python -m metamon.rl.train --log ...
```

Key metrics to monitor:
- **Loss/Actor**: Policy loss (should decrease)
- **Loss/Critic**: Value loss (should decrease)
- **Win Rate**: Against baseline opponents (should increase to 90%+)
- **Valid Actions**: Percentage of legal actions (should be ~100%)

## Citation

If you use this Gemma integration in your research, please cite:

```bibtex
@software{metamon_gemma,
  title={Gemma Integration for Metamon Pokemon RL},
  year={2025},
  url={https://github.com/your-repo/metamon}
}
```

## Support

For issues or questions:
1. Check this documentation
2. Review existing GitHub issues
3. Open a new issue with training logs and config files
