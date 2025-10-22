"""
Gemma 3 270M with temporal context for Metamon RL.

Architecture:
- TstepEncoder: Converts observations to text (no Gemma yet)
- TrajEncoder: Runs Gemma on sliding window of N timesteps
- Outputs hidden states for actor/critic heads
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
class GemmaTextTstepEncoder(TstepEncoder):
    """
    Lightweight TstepEncoder that formats observations as text.

    Does NOT run Gemma - just prepares text for the TrajEncoder.
    This allows Gemma in TrajEncoder to see temporal context.

    Args:
        obs_space: Observation space from AMAGO
        rl2_space: RL^2 space (previous reward, action, etc.)
        pokemon_tokenizer: PokemonTokenizer for decoding token IDs
        include_rl2_features: Whether to include prev reward/action in text
        extra_emb_dim: Dimension for numerical feature embeddings
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        pokemon_tokenizer,
        include_rl2_features: bool = True,
        extra_emb_dim: int = 256,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        self.pokemon_tokenizer = pokemon_tokenizer
        self.include_rl2_features = include_rl2_features

        # Create inverse mapping from token ID to word
        self.id_to_word = {}
        for word in pokemon_tokenizer.all_words:
            token_id = pokemon_tokenizer[word]
            self.id_to_word[token_id] = word

        # Embed numerical features separately (can't put in text)
        base_numerical_features = obs_space["numbers"].shape[0]
        if include_rl2_features:
            self.rl2_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim // 2)
            self.numerical_emb = nn.Linear(
                base_numerical_features + extra_emb_dim // 2,
                extra_emb_dim
            )
        else:
            self.numerical_emb = nn.Linear(base_numerical_features, extra_emb_dim)
            self.rl2_emb = None

        # Output is text + numerical embedding
        # Text will be processed by TrajEncoder's Gemma
        self._emb_dim = extra_emb_dim

    @property
    def emb_dim(self):
        """Returns dimension of numerical embeddings only."""
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
        Convert observation to text + numerical embeddings.

        Returns:
            Dictionary with:
                - "text": List of text strings (for TrajEncoder to process)
                - "numerical_emb": Tensor of numerical embeddings
        """
        batch_size, seq_len = obs["text_tokens"].shape[:2]

        # Convert tokens to text
        text_list = self.pokemon_tokens_to_text(obs["text_tokens"])

        # Embed numerical features
        if self.rl2_emb is not None:
            rl2_emb = F.leaky_relu(self.rl2_emb(symlog(rl2s)))
            numerical = torch.cat((obs["numbers"], rl2_emb), dim=-1)
        else:
            numerical = obs["numbers"]

        numerical_emb = F.leaky_relu(self.numerical_emb(numerical))

        # Return both text and numerical embeddings
        # TrajEncoder will process the text through Gemma
        return {
            "text": text_list,
            "numerical_emb": numerical_emb,
        }


@gin.configurable
class GemmaTemporalTrajEncoder(TrajEncoder):
    """
    TrajEncoder that runs Gemma on a sliding window of N timesteps.

    Gemma sees temporal context by processing concatenated observations.

    Args:
        tstep_dim: Dimension of numerical embeddings from TstepEncoder
        max_seq_len: Maximum sequence length (AMAGO's max_seq_len)
        model_name: HuggingFace model name (default: "google/gemma-3-270m")
        window_size: Number of previous timesteps to include (default: 4)
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_4bit: Whether to use 4-bit quantization
        max_tokens_per_timestep: Maximum tokens per observation (for truncation)
        pooling: How to extract hidden states ("last_token", "mean", "cls")
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        model_name: str = "google/gemma-3-270m",
        window_size: int = 4,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = False,
        max_tokens_per_timestep: int = 1024,
        pooling: str = "mean",
    ):
        super().__init__(tstep_dim, max_seq_len)

        self.window_size = window_size
        self.max_tokens_per_timestep = max_tokens_per_timestep
        self.pooling = pooling

        # Load Gemma tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.gemma = get_peft_model(self.gemma, lora_config)

        # Gemma hidden size + numerical embeddings
        self.gemma_hidden_size = self.gemma.config.hidden_size
        self._emb_dim = self.gemma_hidden_size + tstep_dim

        print(f"GemmaTemporalTrajEncoder: window_size={window_size}, "
              f"hidden_size={self.gemma_hidden_size}, output_dim={self._emb_dim}")

    @property
    def emb_dim(self):
        return self._emb_dim

    def format_temporal_prompt(self, text_list, batch_size, seq_len):
        """
        Format a batch of timesteps into prompts with temporal context.

        Args:
            text_list: List of text strings (length = batch * seq_len)
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            List of prompts (length = batch * seq_len), each with window_size context
        """
        prompts = []

        for b in range(batch_size):
            for t in range(seq_len):
                # Get sliding window of previous timesteps
                window_start = max(0, t - self.window_size + 1)
                window_texts = []

                for w in range(window_start, t + 1):
                    idx = b * seq_len + w
                    turn_label = f"Turn {w - t}:" if w < t else "Current:"
                    window_texts.append(f"{turn_label} {text_list[idx]}")

                prompt = "\n".join(window_texts)
                prompts.append(prompt)

        return prompts

    def forward(self, seq, time_idxs=None, hidden_state=None, log_dict=None):
        """
        Process sequence through Gemma with temporal context.

        Args:
            seq: Dictionary from TstepEncoder with:
                - "text": List of text strings
                - "numerical_emb": Tensor of shape (batch, seq_len, tstep_dim)

        Returns:
            (output_seq, None) where output_seq has shape (batch, seq_len, emb_dim)
        """
        # Extract text and numerical embeddings
        text_list = seq["text"]
        numerical_emb = seq["numerical_emb"]
        batch_size, seq_len, _ = numerical_emb.shape
        device = numerical_emb.device

        # Format prompts with temporal context
        prompts = self.format_temporal_prompt(text_list, batch_size, seq_len)

        # Tokenize all prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.window_size * self.max_tokens_per_timestep,
            return_tensors="pt",
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

        # Extract hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer

        # Pool to get one vector per prompt
        if self.pooling == "last_token":
            # Last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            gemma_emb = hidden_states[torch.arange(hidden_states.shape[0]), seq_lengths]
        elif self.pooling == "mean":
            # Mean pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            gemma_emb = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Reshape back to (batch, seq_len, hidden_size)
        gemma_emb = gemma_emb.view(batch_size, seq_len, -1)

        # Concatenate with numerical embeddings
        combined = torch.cat([gemma_emb, numerical_emb], dim=-1)

        return combined, None
