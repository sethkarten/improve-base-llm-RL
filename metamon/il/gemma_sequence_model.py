"""
Pure Gemma architecture for Metamon RL.

Architecture:
  Input Sequence → Gemma → Per-timestep Hidden States → Actor/Critic

Implementation:
  - TstepEncoder: Minimal (just formats observations)
  - TrajEncoder: Runs Gemma on full sequence with causal attention
  - Gemma's causal attention handles temporal dependencies
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
class MinimalTstepEncoder(TstepEncoder):
    """
    Minimal TstepEncoder that just formats observations for Gemma.

    Returns raw observations with minimal processing. The TrajEncoder
    will actually run Gemma on the full sequence.

    Args:
        obs_space: Observation space from AMAGO
        rl2_space: RL^2 space (previous reward, action, etc.)
        pokemon_tokenizer: PokemonTokenizer for decoding token IDs
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        pokemon_tokenizer,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        self.pokemon_tokenizer = pokemon_tokenizer

        # Create inverse mapping from token ID to word
        self.id_to_word = {}
        for word in pokemon_tokenizer.all_words:
            token_id = pokemon_tokenizer[word]
            self.id_to_word[token_id] = word

        # Dummy output dim - TrajEncoder will return actual embeddings
        self._emb_dim = 1

    @property
    def emb_dim(self):
        return self._emb_dim

    def pokemon_tokens_to_text(self, token_ids):
        """
        Convert Pokemon token IDs to text.

        Args:
            token_ids: Tensor of shape (batch, seq_len, text_features)

        Returns:
            List of text strings (length = batch * seq_len)
        """
        batch_size, seq_len, text_features = token_ids.shape
        text_list = []

        for b in range(batch_size):
            for t in range(seq_len):
                tokens = token_ids[b, t].cpu().numpy()
                words = []
                for token_id in tokens:
                    if token_id == -1:  # UNKNOWN_TOKEN
                        continue
                    word = self.id_to_word.get(int(token_id), "")
                    if word and word.strip():
                        words.append(word)

                text = " ".join(words) if words else "empty"
                text_list.append(text)

        return text_list

    def inner_forward(self, obs, rl2s, log_dict=None):
        """
        Format observations as text strings, including ALL numerical features.

        Returns a dummy tensor - the real processing happens in TrajEncoder.
        We attach metadata that TrajEncoder will use.
        """
        batch_size, seq_len = obs["text_tokens"].shape[:2]
        device = obs["text_tokens"].device

        # Convert tokens to text
        base_text_list = self.pokemon_tokens_to_text(obs["text_tokens"])

        # Append numerical features and RL2 features as text
        full_text_list = []
        for b in range(batch_size):
            for t in range(seq_len):
                idx = b * seq_len + t
                base_text = base_text_list[idx]

                # Add numerical features as text
                numbers = obs["numbers"][b, t].cpu().numpy()
                num_text = " ".join([f"{x:.2f}" for x in numbers])

                # Add RL2 features (prev reward, action, etc.) as text
                rl2_values = rl2s[b, t].cpu().numpy()
                rl2_text = " ".join([f"{x:.2f}" for x in rl2_values])

                # Combine everything into one text string for Gemma to tokenize
                full_text = f"{base_text} [Nums: {num_text}] [Prev: {rl2_text}]"
                full_text_list.append(full_text)

        # Create a dummy output tensor
        dummy_output = torch.zeros(batch_size, seq_len, 1, device=device)

        # Attach metadata for TrajEncoder to use
        dummy_output._metamon_text = full_text_list

        return dummy_output


@gin.configurable
class GemmaSequenceTrajEncoder(TrajEncoder):
    """
    TrajEncoder that runs Gemma on full sequences with causal attention.

    Gemma processes the entire trajectory at once, using its causal attention
    to capture temporal dependencies. Returns per-timestep hidden states.

    Args:
        tstep_dim: Dummy dimension from TstepEncoder (ignored)
        max_seq_len: Maximum sequence length
        model_name: HuggingFace model name
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_4bit: Whether to use 4-bit quantization
        gradient_checkpointing: Enable gradient checkpointing for memory savings
        max_tokens_per_obs: Max tokens per observation
        separator: Separator token between timesteps
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
    ):
        super().__init__(tstep_dim, max_seq_len)

        self.max_tokens_per_obs = max_tokens_per_obs
        self.separator = separator

        # Load Gemma tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get separator token ID
        self.sep_token_id = self.tokenizer.encode(separator, add_special_tokens=False)[0]

        # Load Gemma model
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
                torch_dtype=torch.float16,
            )
        else:
            self.gemma = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
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

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.gemma.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.gemma.parameters())
            print(f"Gemma LoRA: {trainable_params:,} trainable / {total_params:,} total params "
                  f"({100 * trainable_params / total_params:.2f}%)")

        # Enable gradient checkpointing for memory savings
        if gradient_checkpointing:
            self.gemma.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Gemma hidden size (all inputs including numerical are tokenized by Gemma)
        self.gemma_hidden_size = self.gemma.config.hidden_size
        self._emb_dim = self.gemma_hidden_size  # Pure Gemma output, no separate embeddings

        print(f"GemmaSequenceTrajEncoder: max_seq_len={max_seq_len}, "
              f"max_tokens_per_obs={max_tokens_per_obs}, output_dim={self._emb_dim}")

    @property
    def emb_dim(self):
        return self._emb_dim

    def format_sequence(self, text_list, batch_size, seq_len):
        """
        Format a batch of sequences for Gemma.

        Each sequence is: "Turn 1: <obs_1> | Turn 2: <obs_2> | ..."

        Args:
            text_list: List of text strings (length = batch * seq_len)
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            List of formatted sequences (length = batch_size)
            List of timestep positions (where each timestep starts)
        """
        sequences = []
        timestep_positions = []

        for b in range(batch_size):
            turns = []
            positions = []  # Track where each timestep's text ends

            for t in range(seq_len):
                idx = b * seq_len + t
                turn_text = f"Turn {t+1}: {text_list[idx]}"
                turns.append(turn_text)

            # Join with separator
            sequence = self.separator.join(turns)
            sequences.append(sequence)

            # Tokenize to find positions
            # We'll do this in the forward pass after full tokenization

        return sequences

    def forward(self, seq, time_idxs=None, hidden_state=None, log_dict=None):
        """
        Process full sequence through Gemma.

        Args:
            seq: Dummy tensor from TstepEncoder with text metadata attached

        Returns:
            (hidden_states, None) where hidden_states has shape (batch, seq_len, emb_dim)
        """
        # Extract text from TstepEncoder (includes all features as text)
        text_list = seq._metamon_text

        # Infer batch size and seq_len from tensor shape
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

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, total_tokens, hidden_size]

        # Extract per-timestep hidden states
        # Strategy: Find separator tokens and extract hidden states between them
        per_timestep_hiddens = self.extract_timestep_hiddens(
            hidden_states, input_ids, attention_mask, batch_size, seq_len
        )

        # Return pure Gemma hidden states (all features were tokenized as text)
        return per_timestep_hiddens, None

    def extract_timestep_hiddens(self, hidden_states, input_ids, attention_mask, batch_size, seq_len):
        """
        Extract one hidden state per timestep from Gemma's output.

        Strategy: For each timestep, take the mean of hidden states for that timestep's tokens.

        Args:
            hidden_states: [batch, total_tokens, hidden_size]
            input_ids: [batch, total_tokens]
            attention_mask: [batch, total_tokens]
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Tensor of shape [batch, seq_len, hidden_size]
        """
        # Find separator token positions
        sep_positions = (input_ids == self.sep_token_id).long()

        # For each batch and each timestep, extract hidden states
        timestep_hiddens = []

        for b in range(batch_size):
            batch_hiddens = []
            sep_indices = torch.where(sep_positions[b])[0].cpu().tolist()
            sep_indices = [0] + sep_indices + [attention_mask[b].sum().item() - 1]

            # Extract hidden states between separators
            for t in range(min(seq_len, len(sep_indices) - 1)):
                start = sep_indices[t]
                end = sep_indices[t + 1]

                # Mean pool over the timestep's tokens
                timestep_tokens = hidden_states[b, start:end, :]
                if len(timestep_tokens) > 0:
                    timestep_hidden = timestep_tokens.mean(dim=0)
                else:
                    timestep_hidden = torch.zeros(
                        self.gemma_hidden_size, device=hidden_states.device
                    )

                batch_hiddens.append(timestep_hidden)

            # Pad if needed
            while len(batch_hiddens) < seq_len:
                batch_hiddens.append(torch.zeros(
                    self.gemma_hidden_size, device=hidden_states.device
                ))

            timestep_hiddens.append(torch.stack(batch_hiddens))

        return torch.stack(timestep_hiddens)  # [batch, seq_len, hidden_size]
