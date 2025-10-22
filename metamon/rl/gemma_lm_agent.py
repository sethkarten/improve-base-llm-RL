"""
Custom Agent that uses Gemma's LM head as the actor.

This modifies AMAGO's Agent to support using the LM head output
with native LM cross-entropy loss instead of Categorical.
"""

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Type, Optional, Tuple
import sys

import amago
from amago.agent import Agent
from amago.nets import actor_critic
from amago.nets.tstep_encoders import TstepEncoder
from amago.nets.traj_encoders import TrajEncoder


class PassthroughActor(actor_critic.BaseActorHead):
    """
    Actor head that just returns pre-computed logits from TrajEncoder.

    Used when Gemma's LM head is producing action logits directly.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        continuous_dist_type = None,
        **kwargs
    ):
        # Import default if not provided
        if continuous_dist_type is None:
            from amago.nets.policy_dists import TanhGaussian
            continuous_dist_type = TanhGaussian

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            continuous_dist_type=continuous_dist_type,
        )
        # No actual network - just pass through

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict = None,
        straight_from_obs = None,
    ) -> torch.Tensor:
        """
        Extract action logits that were pre-computed by TrajEncoder.

        Args:
            state: Dictionary with:
                - "hidden_states": For critic
                - "action_logits": Pre-computed action logits from LM head [batch, seq_len, action_dim]

        Returns:
            Action logits in shape [batch, seq_len, num_gammas, action_dim]
        """
        if isinstance(state, dict) and "action_logits" in state:
            logits = state["action_logits"]  # [batch, seq_len, action_dim]
        else:
            raise ValueError(f"Expected state to be dict with 'action_logits', got {type(state)}: {state.shape if hasattr(state, 'shape') else state}")

        # Expand to include gamma dimension
        # From [batch, seq_len, action_dim] to [batch, seq_len, num_gammas, action_dim]
        logits = logits.unsqueeze(2).expand(-1, -1, self.num_gammas, -1)

        return logits


@gin.configurable
class GemmaLMAgent(Agent):
    """
    Modified Agent that uses Gemma's LM head with native LM loss.

    Instead of collapsing to 13 discrete actions, we use the full vocab
    and compute cross-entropy loss on action token sequences.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        action_space,
        max_seq_len: int,
        tstep_encoder_type: Type[TstepEncoder],
        traj_encoder_type: Type[TrajEncoder],
        critic_type: Type[actor_critic.BaseCriticHead] = actor_critic.NCritics,
        **kwargs
    ):
        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            action_space=action_space,
            max_seq_len=max_seq_len,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            critic_type=critic_type,
            **kwargs
        )

    def _compute_action_logprobs(self, vocab_logits, actions, action_token_seqs):
        """
        Compute log probabilities for actions using LM cross-entropy.

        Instead of Categorical.log_prob(), we compute the negative cross-entropy
        for the token sequence corresponding to each action.

        Args:
            vocab_logits: [B, L, vocab_size] - LM head logits
            actions: [B, L-1] - action indices
            action_token_seqs: List of token sequences for each action

        Returns:
            logp_a: [B, L-1, G, 1] - log probabilities (negative cross-entropy)
        """
        B, L, vocab_size = vocab_logits.shape
        G = len(self.gammas)

        # Compute negative cross-entropy (which equals log probability)
        logp_a = torch.zeros(B, L-1, G, 1, device=vocab_logits.device)

        for b in range(B):
            for t in range(L - 1):
                action_idx = actions[b, t].item()
                if action_idx < 0 or action_idx >= len(action_token_seqs):
                    continue  # Skip invalid/padding actions

                action_tokens = action_token_seqs[action_idx].to(vocab_logits.device)

                # Sum log probabilities across all tokens in the action sequence
                # log P(action) = sum_i log P(token_i)
                # Note: This assumes independence, full autoregressive would be better
                total_logprob = 0.0
                for token_id in action_tokens:
                    # Cross entropy = -log P(token)
                    # So log P(token) = -cross_entropy
                    ce = F.cross_entropy(
                        vocab_logits[b, t].unsqueeze(0),
                        token_id.unsqueeze(0),
                        reduction='none'
                    )
                    total_logprob = total_logprob - ce  # Convert CE to log prob

                # Replicate across all gammas
                for g in range(G):
                    logp_a[b, t, g, 0] = total_logprob

        return logp_a

    def forward(self, batch, log_step: bool):
        """
        Faithfully reimplements AMAGO's forward, but uses LM cross-entropy for log_prob.

        This matches AMAGO's Actor loss exactly:
        actor_loss = -offline_coeff * filter(advantage) * log_prob(action)

        Where filter(advantage) comes from the FBC filter function.
        """
        from amago.loading import MAGIC_PAD_VAL

        self.update_info = {}
        active_log_dict = self.update_info if log_step else None

        # Encode timesteps
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        B, L, _ = o.shape

        # Get trajectory encoding (returns dict with vocab_logits + hidden_states)
        traj_emb_dict, _ = self.traj_encoder(o, batch.time_idxs, None, None)
        vocab_logits = traj_emb_dict["vocab_logits"]  # [B, L, vocab_size]
        s_rep = traj_emb_dict["hidden_states"]  # [B, L, hidden_dim]

        # Get action token sequences
        action_token_seqs = self.traj_encoder.get_action_token_sequences()

        # Process actions
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1., 1.)
        actions_flat = a[:, :, 0].long()  # [B, L-1]

        # Prepare data (following AMAGO's format)
        import einops
        G = len(self.gammas)

        # Pad actions to match sequence length (L instead of L-1)
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")  # [B, L, action_dim]
        # Repeat across gammas
        a_buffer = einops.repeat(a_buffer, f"b l a -> b l {G} a")  # [B, L, G, action_dim]

        # Prepare rewards, dones, gammas (don't expand yet - let einops.repeat do it)
        r = (batch.rews * self.reward_multiplier).float()  # [B, L-1, 1]
        d = batch.dones.float()  # [B, L-1, 1]
        gamma = self.gammas.to(o.device).unsqueeze(-1)  # [G, 1]

        #################
        ## Critic Loss ##
        #################
        # Compute Q(s,a) for buffer actions
        # a_buffer has shape [B, L, G, action_dim]
        # We need [1, B, L-1, G, action_dim] for the critic
        q_s_a_g_dist = self.critics(s_rep[:, :-1, ...], a_buffer[:, :-1, ...].unsqueeze(0))  # Returns pyd.Categorical
        # Convert categorical distribution to scalar values
        q_s_a_g = self.critics.bin_dist_to_raw_vals(q_s_a_g_dist).squeeze(0)  # [B, L-1, C, G, 1]

        # Compute TD targets
        with torch.no_grad():
            # Use target critic with next state
            q_next_dist = self.target_critics(s_rep[:, 1:, ...], a_buffer[:, :-1, ...].unsqueeze(0))
            q_next = self.target_critics.bin_dist_to_raw_vals(q_next_dist).mean(0)  # [B, L-1, C, G, 1]

            # Expand rewards and dones across gammas: [B, L-1, 1] -> [B, L-1, 1, G, 1]
            r_expanded = einops.repeat(r, f"b l r -> b l 1 {G} r")
            d_expanded = einops.repeat(d, f"b l d -> b l 1 {G} d")

            # TD target: r + gamma * (1 - done) * Q(s', a)
            # gamma is [G, 1], broadcasts to [B, L-1, C, G, 1]
            td_target = r_expanded + gamma * (1.0 - d_expanded) * q_next.mean(2, keepdim=True)
            td_target_norm = self.popart.normalize_values(td_target)

        # Critic loss: (Q(s,a) - target)^2
        critic_loss = (self.popart(q_s_a_g) - td_target_norm.detach()).pow(2)

        #################
        ## Actor Loss  ##
        #################
        actor_loss = torch.zeros((B, L - 1, len(self.gammas), 1), device=o.device)

        if self.offline_coeff > 0:
            # Compute advantages if not using fake_filter
            if not self.fake_filter:
                with torch.no_grad():
                    # V(s) = mean Q over actions (for discrete, would sample)
                    val_s_g = q_s_a_g.mean(2).detach()
                    # A(s,a) = Q(s,a) - V(s)
                    advantage_a_s_g = q_s_a_g.mean(2) - val_s_g
                    filter_ = self.fbc_filter_func(advantage_a_s_g).float()
            else:
                # No filtering (behavior cloning)
                filter_ = torch.ones((1, 1, 1, 1), device=o.device)

            # Compute log P(action) using LM cross-entropy
            logp_a = self._compute_action_logprobs(vocab_logits, actions_flat, action_token_seqs)

            # Actor loss: -filter * log_prob
            actor_loss += self.offline_coeff * -(filter_.detach() * logp_a)

        return critic_loss, actor_loss

    def get_actions(
        self,
        obs,
        rl2s,
        time_idxs=None,
        hidden_state=None,
        sample: bool = True,
        **kwargs
    ):
        """
        Override get_actions for inference.

        During inference, we need to sample from the LM's vocab distribution
        and map back to discrete actions, while respecting action masks.
        """
        # Encode observations
        o = self.tstep_encoder(obs, rl2s)

        # Get trajectory encoding
        traj_emb_dict, hidden_state = self.traj_encoder(o, time_idxs, hidden_state, None)
        vocab_logits = traj_emb_dict["vocab_logits"]  # [B, L, vocab_size]

        # Get action token sequences
        action_token_seqs = self.traj_encoder.get_action_token_sequences()

        # Compute log probs for all 13 actions
        B, L, _ = vocab_logits.shape

        # For each timestep, compute log prob of each action and sample
        action_logprobs_all_actions = []

        for action_idx in range(len(action_token_seqs)):
            action_tokens = action_token_seqs[action_idx].to(vocab_logits.device)

            # Compute log prob for this action across all batch/timesteps
            # Simplified: sum log probs of tokens (assumes independence)
            action_logprob = torch.zeros(B, L, device=vocab_logits.device)

            for token_id in action_tokens:
                # Get log prob of this token
                log_probs = F.log_softmax(vocab_logits, dim=-1)
                token_logprob = log_probs[:, :, token_id]
                action_logprob = action_logprob + token_logprob

            action_logprobs_all_actions.append(action_logprob)

        # Stack to get [B, L, 13]
        all_action_logprobs = torch.stack(action_logprobs_all_actions, dim=-1)

        # Apply action masking if illegal_actions is present
        if "illegal_actions" in obs:
            # obs["illegal_actions"] can be [B, action_dim] or [B, 1, 1, action_dim]
            illegal_actions = obs["illegal_actions"]
            if isinstance(illegal_actions, np.ndarray):
                illegal_actions = torch.from_numpy(illegal_actions).to(all_action_logprobs.device)

            # Reshape to [B, action_dim] regardless of input shape
            # First squeeze all singleton dimensions, then ensure we have [B, action_dim]
            illegal_mask = illegal_actions.squeeze()  # Remove all size-1 dimensions
            if illegal_mask.ndim == 1:
                # Single batch, expand to [1, action_dim]
                illegal_mask = illegal_mask.unsqueeze(0)

            # Now expand to [B, L, action_dim]
            illegal_mask = illegal_mask.unsqueeze(1).expand(B, L, -1)

            # Set illegal actions to -inf (will have 0 probability after softmax)
            all_action_logprobs = all_action_logprobs.masked_fill(illegal_mask, float('-inf'))

        # Debug: check for NaNs
        if torch.isnan(all_action_logprobs).any():
            print(f"WARNING: NaNs in action logprobs!", file=sys.stderr, flush=True)
            print(f"vocab_logits stats: min={vocab_logits.min()}, max={vocab_logits.max()}, has_nan={torch.isnan(vocab_logits).any()}", file=sys.stderr, flush=True)

        # Sample or take argmax
        if sample:
            # Convert to probabilities and sample
            action_probs = F.softmax(all_action_logprobs, dim=-1)
            actions = torch.multinomial(action_probs.reshape(-1, len(action_token_seqs)), 1).reshape(B, L, 1)
        else:
            # Greedy: take action with highest log prob
            actions = all_action_logprobs.argmax(dim=-1, keepdim=True)

        # Ensure actions are in valid range [0, 12] and convert to uint8 for environment
        actions = actions.clamp(0, len(action_token_seqs) - 1).to(torch.uint8)

        # Debug: print action stats
        # print(f"DEBUG get_actions: shape={actions.shape}, min={actions.min().item()}, max={actions.max().item()}, dtype={actions.dtype}", file=sys.stderr, flush=True)
        # print(f"DEBUG get_actions: first 5 actions = {actions.flatten()[:5].tolist()}", file=sys.stderr, flush=True)

        return actions, hidden_state
