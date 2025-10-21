"""
Gemma 3 270M integration for Metamon.

Provides TstepEncoder implementations that use pretrained Gemma models
instead of the custom Pokemon tokenizer and transformer architecture.
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
from amago.nets.utils import symlog


@gin.configurable
class GemmaTstepEncoder(TstepEncoder):
    """
    Timestep encoder using pretrained Gemma 3 270M model.

    This replaces the custom Pokemon tokenizer + transformer with a pretrained
    LLM that can leverage world knowledge and transfer learning.

    Args:
        obs_space: Observation space from AMAGO
        rl2_space: RL^2 space (previous reward, action, etc.)
        pokemon_tokenizer: PokemonTokenizer instance for decoding token IDs
        model_name: HuggingFace model name (default: "google/gemma-3-270m")
        use_lora: Whether to use LoRA for efficient finetuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        freeze_base_model: Whether to freeze the base model completely
        use_4bit: Whether to use 4-bit quantization (saves VRAM)
        extra_emb_dim: Dimension for extra RL^2 features embedding
        numerical_emb_dim: Dimension for numerical features embedding
        dropout: Dropout probability
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        pokemon_tokenizer,
        model_name: str = "google/gemma-3-270m",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_base_model: bool = False,
        use_4bit: bool = False,
        extra_emb_dim: int = 18,
        numerical_emb_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        self.pokemon_tokenizer = pokemon_tokenizer

        # Create inverse mapping from token ID to word
        self.id_to_word = {}
        for word in pokemon_tokenizer.all_words:
            token_id = pokemon_tokenizer[word]
            self.id_to_word[token_id] = word

        self.model_name = model_name
        self.use_lora = use_lora
        self.use_4bit = use_4bit

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
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

        # Optionally freeze base model
        if freeze_base_model:
            for param in self.gemma.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
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

        # Get Gemma's hidden size
        self.gemma_hidden_size = self.gemma.config.hidden_size

        # Projection layers for RL^2 and numerical features
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        base_numerical_features = obs_space["numbers"].shape[0]
        self.numerical_emb = nn.Linear(
            base_numerical_features + extra_emb_dim,
            numerical_emb_dim
        )

        # Combine Gemma embeddings with numerical embeddings
        self._emb_dim = self.gemma_hidden_size + numerical_emb_dim

        self.dropout = nn.Dropout(dropout)

        # Note: We expect obs["text_tokens"] to contain PRE-TOKENIZED text strings
        # that will be converted to Gemma token IDs. This will require modifying
        # the data loading pipeline.

    @property
    def emb_dim(self):
        return self._emb_dim

    def pokemon_tokens_to_text(self, token_ids):
        """
        Convert Pokemon token IDs back to text.

        Args:
            token_ids: Tensor of shape (batch, seq_len, text_features) with Pokemon token IDs

        Returns:
            List of text strings (length = batch * seq_len)
        """
        batch_size, seq_len, text_features = token_ids.shape
        text_list = []

        for b in range(batch_size):
            for t in range(seq_len):
                # Convert each timestep's tokens to text
                tokens = token_ids[b, t].cpu().numpy()
                words = []
                for token_id in tokens:
                    if token_id == -1:  # UNKNOWN_TOKEN
                        continue
                    # Get the word from the inverse mapping
                    word = self.id_to_word.get(int(token_id), "")
                    if word and word.strip():
                        words.append(word)

                text = " ".join(words) if words else "empty"
                text_list.append(text)

        return text_list

    def process_text_input(self, text_obs):
        """
        Convert text observation to Gemma input format.

        Args:
            text_obs: Tensor of shape (batch, seq_len, text_features) containing
                     text token IDs from the Pokemon tokenizer.

        Returns:
            Tensor of shape (batch, seq_len, gemma_hidden_size)
        """
        batch_size, seq_len, _ = text_obs.shape
        device = text_obs.device

        # Convert Pokemon tokens to text
        text_list = self.pokemon_tokens_to_text(text_obs)

        # Process through Gemma
        text_emb_flat = self.process_text_batch(text_list, device)

        # Reshape to (batch, seq_len, hidden_size)
        text_emb = text_emb_flat.view(batch_size, seq_len, -1)

        return text_emb

    def process_text_batch(self, text_list, device):
        """
        Tokenize and process a batch of text strings through Gemma.

        Args:
            text_list: List of text strings (length = batch * seq_len)
            device: Target device

        Returns:
            Tensor of shape (batch * seq_len, hidden_size)
        """
        # Tokenize all texts in the batch
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward through Gemma (without computing LM loss)
        with torch.set_grad_enabled(self.training):
            outputs = self.gemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling over sequence
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        return pooled

    def inner_forward(self, obs, rl2s, log_dict=None):
        """
        Forward pass through Gemma + numerical fusion.

        Args:
            obs: Dictionary containing:
                - "text_tokens": (batch, seq_len, text_features) - Pokemon token IDs
                - "numbers": (batch, seq_len, numerical_features) - Numerical state
            rl2s: (batch, seq_len, rl2_dim) - RL^2 features (prev reward, action, etc.)

        Returns:
            Tensor of shape (batch, seq_len, emb_dim)
        """
        # Process RL^2 features
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))

        # Combine numerical features with RL^2
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        numerical_emb = F.leaky_relu(self.dropout(self.numerical_emb(numerical)))

        # Process text through Gemma
        text_emb = self.process_text_input(obs["text_tokens"])

        # Combine text and numerical embeddings
        combined_emb = torch.cat([text_emb, numerical_emb], dim=-1)

        return combined_emb


@gin.configurable
class GemmaWithTextTstepEncoder(TstepEncoder):
    """
    Gemma TstepEncoder that expects raw text input instead of pre-tokenized IDs.

    This version requires modifying the observation space to include a "text" field
    with raw battle state descriptions.

    Args:
        obs_space: Observation space from AMAGO
        rl2_space: RL^2 space (previous reward, action, etc.)
        model_name: HuggingFace model name
        use_lora: Whether to use LoRA for efficient finetuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        freeze_base_model: Whether to freeze the base model
        use_4bit: Whether to use 4-bit quantization
        max_text_length: Maximum length for text tokenization
        pool_strategy: How to pool Gemma outputs ("mean", "last", "first")
        extra_emb_dim: Dimension for extra RL^2 features embedding
        numerical_emb_dim: Dimension for numerical features embedding
        dropout: Dropout probability
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        model_name: str = "google/gemma-3-270m",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_base_model: bool = False,
        use_4bit: bool = False,
        max_text_length: int = 512,
        pool_strategy: str = "mean",
        extra_emb_dim: int = 18,
        numerical_emb_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        self.model_name = model_name
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        self.max_text_length = max_text_length
        self.pool_strategy = pool_strategy

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
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

        # Optionally freeze base model
        if freeze_base_model:
            for param in self.gemma.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
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

        # Get Gemma's hidden size
        self.gemma_hidden_size = self.gemma.config.hidden_size

        # Projection layers for RL^2 and numerical features
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        base_numerical_features = obs_space["numbers"].shape[0]
        self.numerical_emb = nn.Linear(
            base_numerical_features + extra_emb_dim,
            numerical_emb_dim
        )

        # Combine Gemma embeddings with numerical embeddings
        self._emb_dim = self.gemma_hidden_size + numerical_emb_dim

        self.dropout = nn.Dropout(dropout)

    @property
    def emb_dim(self):
        return self._emb_dim

    def pool_gemma_output(self, hidden_states, attention_mask):
        """
        Pool Gemma's output to a fixed-size representation.

        Args:
            hidden_states: (batch, text_seq_len, hidden_size)
            attention_mask: (batch, text_seq_len)

        Returns:
            Tensor of shape (batch, hidden_size)
        """
        if self.pool_strategy == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pool_strategy == "last":
            # Take the last non-padded token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        elif self.pool_strategy == "first":
            # Take the first token (like BERT's [CLS])
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pool_strategy: {self.pool_strategy}")

    def process_text_batch(self, text_list, device):
        """
        Tokenize and process a batch of text strings through Gemma.

        Args:
            text_list: List of text strings (length = batch * seq_len)
            device: Target device

        Returns:
            Tensor of shape (batch * seq_len, hidden_size)
        """
        # Tokenize all texts in the batch
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward through Gemma (without computing LM loss)
        with torch.set_grad_enabled(self.training):
            outputs = self.gemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Pool to fixed size
        pooled = self.pool_gemma_output(hidden_states, attention_mask)

        return pooled

    def inner_forward(self, obs, rl2s, log_dict=None):
        """
        Forward pass through Gemma + numerical fusion.

        Args:
            obs: Dictionary containing:
                - "text": List or Tensor of raw text strings describing battle state
                - "numbers": (batch, seq_len, numerical_features) - Numerical state
            rl2s: (batch, seq_len, rl2_dim) - RL^2 features

        Returns:
            Tensor of shape (batch, seq_len, emb_dim)
        """
        batch_size, seq_len = obs["numbers"].shape[:2]
        device = obs["numbers"].device

        # Process RL^2 features
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))

        # Combine numerical features with RL^2
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        numerical_emb = F.leaky_relu(self.dropout(self.numerical_emb(numerical)))

        # Process text through Gemma
        # Flatten batch and sequence dimensions for text processing
        if "text" in obs:
            # Assume obs["text"] is a list of lists: [[text_t0, text_t1, ...], [text_t0, text_t1, ...], ...]
            # Or a flat list of length batch * seq_len
            if isinstance(obs["text"], list):
                if isinstance(obs["text"][0], list):
                    # Nested list - flatten
                    text_flat = [t for batch_texts in obs["text"] for t in batch_texts]
                else:
                    # Already flat
                    text_flat = obs["text"]
            else:
                raise ValueError("obs['text'] must be a list of text strings")

            # Process all texts at once
            text_emb_flat = self.process_text_batch(text_flat, device)

            # Reshape back to (batch, seq_len, hidden_size)
            text_emb = text_emb_flat.view(batch_size, seq_len, -1)
        else:
            # No text provided - use zero embeddings
            text_emb = torch.zeros(
                batch_size, seq_len, self.gemma_hidden_size,
                device=device, dtype=numerical_emb.dtype
            )

        # Combine text and numerical embeddings
        combined_emb = torch.cat([text_emb, numerical_emb], dim=-1)

        return combined_emb
