"""
Gemma with LM head as actor for Metamon RL.

Architecture:
  Text → Gemma → Hidden States → LM Head (action tokens) → Action logits
                              └→ Critic MLP → Values

The actor uses Gemma's built-in language modeling head, constrained to
valid action tokens. This is more natural for an LLM.
"""

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import amago
from amago.nets.tstep_encoders import TstepEncoder
from amago.nets.traj_encoders import TrajEncoder
from amago.nets.utils import symlog


@gin.configurable
class GemmaLMTrajEncoder(TrajEncoder):
    """
    TrajEncoder that uses Gemma's LM head as the actor.

    Instead of a separate actor MLP, we use Gemma's language modeling head
    to predict actions as text. This is more natural for an LLM approach.

    Args:
        tstep_dim: Dummy dimension from TstepEncoder (ignored)
        max_seq_len: Maximum sequence length
        model_name: HuggingFace model name
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_4bit: Whether to use 4-bit quantization
        gradient_checkpointing: Enable gradient checkpointing
        max_tokens_per_obs: Max tokens per observation
        separator: Separator token between timesteps
        action_space_size: Number of discrete actions (e.g., 10)
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        model_name: str = "google/gemma-3-270m",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = False,
        gradient_checkpointing: bool = False,
        max_tokens_per_obs: int = 512,
        separator: str = " | ",
        action_space_size: int = 10,
    ):
        super().__init__(tstep_dim, max_seq_len)

        self.max_tokens_per_obs = max_tokens_per_obs
        self.separator = separator
        self.action_space_size = action_space_size

        # Load Gemma tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get separator token ID
        self.sep_token_id = self.tokenizer.encode(separator, add_special_tokens=False)[0]

        # Load Gemma model (we'll use its LM head)
        # Use SDPA (PyTorch's native scaled dot product attention)
        # Flash Attention 2 doesn't support RTX 5090 (Blackwell) yet
        attn_impl = "sdpa"
        print(f"Using {attn_impl} (PyTorch native fast attention - compatible with RTX 5090)")

        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.gemma = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation=attn_impl,
            )
        else:
            self.gemma = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation=attn_impl,
            )

        # Apply LoRA
        if use_lora:
            if use_4bit:
                self.gemma = prepare_model_for_kbit_training(self.gemma)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.gemma = get_peft_model(self.gemma, lora_config)

            trainable_params = sum(p.numel() for p in self.gemma.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.gemma.parameters())
            print(f"Gemma LoRA: {trainable_params:,} trainable / {total_params:,} total params "
                  f"({100 * trainable_params / total_params:.2f}%)")

        # Enable gradient checkpointing
        if gradient_checkpointing:
            self.gemma.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Note: We skip torch.compile here because it causes issues with Accelerate's model unwrapping.
        # Accelerate will handle compilation at the top level if needed.
        # The critics already use @torch.compile decorator which works fine.

        # Gemma hidden size
        self.gemma_hidden_size = self.gemma.config.hidden_size
        self._emb_dim = self.gemma_hidden_size

        # Define text representations for each action
        # Actions 0-3: move 1-4
        # Actions 4-8: switch 1-5
        # Actions 9-12: tera move 1-4
        self.action_texts, self.action_token_sequences = self._define_action_texts()

        print(f"GemmaLMTrajEncoder: max_seq_len={max_seq_len}, "
              f"max_tokens_per_obs={max_tokens_per_obs}, output_dim={self._emb_dim}")
        print(f"Action 0: '{self.action_texts[0]}' -> {self.action_token_sequences[0].tolist()}")
        print(f"Action 9: '{self.action_texts[9]}' -> {self.action_token_sequences[9].tolist()}")

    def _define_action_texts(self):
        """
        Map each discrete action index to its text representation.

        DefaultActionSpace with 13 actions:
        - 0-3: "move 1" through "move 4"
        - 4-8: "switch 1" through "switch 5"
        - 9-12: "tera move 1" through "tera move 4" (Terastallize + move)

        Returns:
            action_texts: List of text strings
            action_token_sequences: List of token ID tensors
        """
        action_texts = []
        action_token_sequences = []

        # Actions 0-3: Regular moves
        for i in range(4):
            text = f"move {i+1}"
            action_texts.append(text)

        # Actions 4-8: Switches
        for i in range(5):
            text = f"switch {i+1}"
            action_texts.append(text)

        # Actions 9-12: Terastallize moves
        for i in range(4):
            text = f"tera move {i+1}"
            action_texts.append(text)

        # Tokenize all actions
        for text in action_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            action_token_sequences.append(torch.tensor(tokens))

        return action_texts, action_token_sequences

    @property
    def emb_dim(self):
        return self._emb_dim

    def format_sequence(self, text_list, batch_size, seq_len):
        """Format batch of sequences for Gemma."""
        sequences = []

        for b in range(batch_size):
            turns = []
            for t in range(seq_len):
                idx = b * seq_len + t
                turn_text = f"Turn {t+1}: {text_list[idx]}"
                turns.append(turn_text)

            sequence = self.separator.join(turns)
            sequences.append(sequence)

        return sequences

    def forward(self, seq, time_idxs=None, hidden_state=None, log_dict=None):
        """
        Process sequence through Gemma and extract hidden states + action logits.

        Returns:
            (output_dict, None) where output_dict contains:
                - "hidden_states": [batch, seq_len, hidden_dim] for critic
                - "action_logits": [batch, seq_len, action_space_size] for actor
        """
        # Extract text from TstepEncoder
        text_list = seq._metamon_text

        batch_size, seq_len, _ = seq.shape
        device = seq.device

        # Format sequences
        sequences = self.format_sequence(text_list, batch_size, seq_len)

        # Tokenize
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=seq_len * self.max_tokens_per_obs,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Run through Gemma
        with torch.set_grad_enabled(self.training):
            outputs = self.gemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get hidden states for critic
        hidden_states = outputs.hidden_states[-1]  # [batch, total_tokens, hidden_size]

        # Get logits from LM head for actor
        lm_logits = outputs.logits  # [batch, total_tokens, vocab_size]

        # Extract per-timestep hidden states and logits
        per_timestep_hiddens = self.extract_timestep_hiddens(
            hidden_states, input_ids, attention_mask, batch_size, seq_len
        )
        per_timestep_logits = self.extract_timestep_hiddens(
            lm_logits, input_ids, attention_mask, batch_size, seq_len
        )

        # Return both hidden states (for critic) and vocab logits (for LM loss)
        # Don't collapse to 13 actions - keep full vocab space
        output_dict = {
            "hidden_states": per_timestep_hiddens,
            "vocab_logits": per_timestep_logits,  # [batch, seq_len, vocab_size]
        }

        # AMAGO expects a tuple (output, hidden_state)
        # We'll pack the dict into the first element
        return output_dict, None

    def get_action_token_sequences(self):
        """Return the mapping from action indices to token sequences for loss computation."""
        return self.action_token_sequences

    def extract_timestep_hiddens(self, tensor, input_ids, attention_mask, batch_size, seq_len):
        """
        Extract one vector per timestep from Gemma's output.

        Args:
            tensor: [batch, total_tokens, dim] (hidden_states or logits)
            input_ids: [batch, total_tokens]
            attention_mask: [batch, total_tokens]
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Tensor of shape [batch, seq_len, dim]
        """
        # Find separator token positions
        sep_positions = (input_ids == self.sep_token_id).long()

        # For each batch and timestep, extract vectors
        timestep_vectors = []

        for b in range(batch_size):
            batch_vectors = []
            sep_indices = torch.where(sep_positions[b])[0].cpu().tolist()
            sep_indices = [0] + sep_indices + [attention_mask[b].sum().item() - 1]

            for t in range(min(seq_len, len(sep_indices) - 1)):
                start = sep_indices[t]
                end = sep_indices[t + 1]

                # Mean pool over the timestep's tokens
                timestep_tokens = tensor[b, start:end, :]
                if len(timestep_tokens) > 0:
                    timestep_vector = timestep_tokens.mean(dim=0)
                else:
                    timestep_vector = torch.zeros(
                        tensor.shape[-1], device=tensor.device
                    )

                batch_vectors.append(timestep_vector)

            # Pad if needed
            while len(batch_vectors) < seq_len:
                batch_vectors.append(torch.zeros(
                    tensor.shape[-1], device=tensor.device
                ))

            timestep_vectors.append(torch.stack(batch_vectors))

        return torch.stack(timestep_vectors)  # [batch, seq_len, dim]
