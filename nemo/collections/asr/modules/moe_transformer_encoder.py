# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mixture-of-Experts (MoE) Transformer Encoder for ASR.

Extends the standard TransformerEncoder by replacing the feed-forward network
(FFN) in each Transformer layer with a MoE feed-forward module
(MoEFeedForward). Supports both per-layer (switch) routing and shared (omni)
routing across all MoE layers.

References:
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
  with Simple and Efficient Sparsity", 2022
- Gu et al., "Omni-Router: Sharing Routing Decisions in Sparse MoE for ASR", 2025
"""

import math
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.modules.transformer_encoder import FeedForward, TransformerEncoder, TransformerEncoderConfig

__all__ = ['SwitchGate', 'MoEFeedForward', 'MoETransformerEncoder']


class SwitchGate(nn.Module):
    """Softmax routing gate for Mixture-of-Experts layers.

    Computes routing probabilities over N experts for each input token using
    a learned linear projection followed by softmax. Optionally adds Gaussian
    noise during training to encourage exploration (jitter).

    Can be shared across multiple MoE layers (omni-router) or used independently
    per layer (switch-style).

    Args:
        d_model (int): Input feature dimension.
        num_experts (int): Number of experts to route over.
        jitter_eps (float): Standard deviation of Gaussian noise added to logits
            during training. 0.0 disables noise. Defaults to 0.0.
    """

    def __init__(self, d_model: int, num_experts: int, jitter_eps: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities for input tokens.

        Args:
            x: Input tensor of shape (num_tokens, d_model).

        Returns:
            Routing probabilities of shape (num_tokens, num_experts).
        """
        logits = self.w_gate(x)
        if self.training and self.jitter_eps > 0.0:
            logits = logits + torch.randn_like(logits) * self.jitter_eps
        return F.softmax(logits, dim=-1)


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts Feed-Forward module -- drop-in replacement for FeedForward.

    Contains N expert FFNs (each a ``FeedForward`` built from the encoder
    ``TransformerEncoderConfig``) and a router (``SwitchGate``). For each input
    token, the router selects the top-k experts and computes a weighted
    combination of their outputs.

    Each expert mirrors the base ``FeedForward(cfg)`` exactly, so the per-expert
    inner dimension is ``int(cfg.ff_expansion * cfg.d_model)`` -- expert width is
    controlled through ``ff_expansion`` (e.g. a sub-1x value for fine-grained
    experts), matching the NeMo Transformer encoder interface.

    Computes an auxiliary load-balancing loss (GShard-style) stored in
    ``self._aux_loss`` and also stashes the per-expert dispatch counts in
    ``self._expert_counts`` so that downstream code can compute MoE diagnostics
    (load balance, dead experts, routing entropy) without re-running the
    router. Both attributes are reset on every ``forward`` call.

    Args:
        cfg: Encoder config; each expert is a ``FeedForward(cfg)`` and the
            router projects from ``cfg.d_model``.
        num_experts: Number of expert FFNs.
        top_k: Number of experts activated per token. Defaults to 1.
        router: External router (for omni-router sharing). If None, an internal
            router is created. Defaults to None.
        jitter_eps: Jitter noise for the internal router (ignored if an
            external router is provided). Defaults to 0.0.
    """

    def __init__(
        self,
        cfg: TransformerEncoderConfig,
        num_experts: int,
        top_k: int = 1,
        router: Optional[SwitchGate] = None,
        jitter_eps: float = 0.0,
    ):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = int(cfg.ff_expansion * cfg.d_model)

        self.experts = nn.ModuleList([FeedForward(cfg) for _ in range(num_experts)])

        if router is not None:
            self.router = router
        else:
            self.router = SwitchGate(d_model=cfg.d_model, num_experts=num_experts, jitter_eps=jitter_eps)

        self._aux_loss = None
        # Detached per-step routing diagnostics, populated on every forward.
        # All accumulator-friendly:
        #   _expert_counts: (num_experts,) long  -- # of (token, top-k slot)
        #                   pairs routed to each expert during the last forward.
        #   _gate_prob_sum: (num_experts,) float32 -- sum (over tokens) of the
        #                   router softmax probability for each expert. Use
        #                   `_gate_prob_sum / _num_tokens` to get the mean.
        #   _num_tokens:    int -- number of tokens (= B * T) seen in the last
        #                   forward.
        # All three are detached and on-device. Float32 is forced even when the
        # router runs under bf16 autocast so cumulative buffers stay accurate
        # across many steps.
        self._expert_counts: Optional[torch.Tensor] = None
        self._gate_prob_sum: Optional[torch.Tensor] = None
        self._num_tokens: int = 0

    def _compute_load_balancing_loss(
        self,
        gate_probs: torch.Tensor,
        expert_counts: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """GShard / Switch Transformer load-balancing auxiliary loss.

        ``L_load = N * sum_j(f_j * rho_j)`` where:

        - ``f_j``  = fraction of (token, top-k) dispatches that landed on expert j
        - ``rho_j`` = mean router probability allocated to expert j

        Computed from a length-``num_experts`` count vector instead of an
        ``[num_tokens, num_experts]`` one-hot tensor, saving memory.
        """
        f = expert_counts.to(gate_probs.dtype) / num_tokens
        rho = gate_probs.mean(dim=0)
        return self.num_experts * (f * rho).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass -- same signature as ``FeedForward.forward(x)``.

        Implementation notes (perf):

        * The original implementation iterated ``for k in range(top_k): for
          i in range(num_experts):`` (i.e. ``top_k * num_experts`` Python-level
          launches per layer per step). We instead flatten the dispatch into a
          single ``(num_tokens * top_k,)`` list of (token, expert, weight)
          triples and iterate ``num_experts`` times, dropping the inner
          ``top_k`` factor. This is the dominant cost saving.

        * The auxiliary load-balancing loss now uses ``torch.bincount`` over
          the dispatch indices instead of allocating an
          ``[num_tokens, num_experts]`` one-hot tensor.

        Args:
            x: Input tensor of shape ``(B, T, D)``.

        Returns:
            Output tensor of shape ``(B, T, D)``.
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        num_tokens = x_flat.shape[0]

        gate_probs = self.router(x_flat)

        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Preserve top-1 router gradients; otherwise normalize the selected weights.
        if self.top_k > 1:
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Flatten dispatch: each token contributes top_k entries.
        flat_expert = top_k_indices.reshape(-1)  # (num_tokens * top_k,)
        flat_weight = top_k_probs.reshape(-1)  # (num_tokens * top_k,)
        flat_token = (
            torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, self.top_k).reshape(-1)
        )  # (num_tokens * top_k,)

        expert_counts = torch.bincount(flat_expert, minlength=self.num_experts)

        self._aux_loss = self._compute_load_balancing_loss(gate_probs, expert_counts, num_tokens)
        # Stash detached, fp32, on-device per-step diagnostics for the
        # encoder's cumulative MoE stats. We store the SUM of router probs
        # (not the mean) so the parent encoder can correctly weight per-step
        # contributions when accumulating across many steps.
        self._expert_counts = expert_counts.detach()
        self._gate_prob_sum = gate_probs.detach().sum(dim=0).float()
        self._num_tokens = int(num_tokens)

        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = flat_expert == i
            tok_idx = flat_token[mask]
            if tok_idx.numel() == 0:
                # Keep every expert in DDP's autograd graph when it receives no tokens;
                # an FP32 zero also matches ``output`` under autocast.
                if x_flat.shape[0] > 0:
                    zero_idx = torch.zeros(1, dtype=torch.long, device=x_flat.device)
                    anchor_zero = x_flat.new_zeros(1, 1)  # FP32 (matches output)
                    anchor_out = self.experts[i](x_flat[:1]) * anchor_zero
                    output.index_add_(0, zero_idx, anchor_out)
                continue
            expert_in = x_flat.index_select(0, tok_idx)
            expert_out = self.experts[i](expert_in) * flat_weight[mask].unsqueeze(-1)
            output.index_add_(0, tok_idx, expert_out)

        return output.reshape(batch_size, seq_len, d_model)


class MoETransformerEncoder(TransformerEncoder):
    """Transformer Encoder with Mixture-of-Experts (MoE) feed-forward layers.

    Subclasses :class:`TransformerEncoder` and replaces the FFN in each
    Transformer layer with :class:`MoEFeedForward` modules for configurable
    layers.

    Two routing strategies are supported:

    - ``'switch'``: Each MoE layer has its own independent router.
    - ``'omni'``: A single shared router is used across all MoE layers,
      encouraging coordinated expert specialization (Omni-router).

    Args:
        All standard TransformerEncoder args, plus:

        moe_num_experts (int): Number of experts per MoE layer. Defaults to 8.
        moe_top_k (int): Number of experts activated per token. Defaults to 1.
        moe_router_type (str): Router strategy.
            ``'omni'`` = shared router across layers,
            ``'switch'`` = independent router per layer. Defaults to ``'omni'``.
        moe_layer_indices (list[int] or None): Indices of Transformer layers to
            apply MoE to. ``None`` means all layers. Defaults to None.
        moe_load_balance_loss_weight (float): Weight for the auxiliary
            load-balancing loss. Defaults to 0.01.
        moe_jitter_eps (float): Noise scale for the router during training.
            Defaults to 0.0.
        moe_init_from_ffn (bool): If True, initialize all expert FFN weights
            from the original (base) FFN weights created by ``super().__init__()``.
            Defaults to True.
    """

    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        feat_out: int = -1,
        subsampling: str = 'feature_stacking',
        subsampling_factor: int = 4,
        drop_rate: float = 0.1,
        dropout_pre_encoder: float = None,
        dropout_emb: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        self_attention_model: Optional[str] = "rel_pos",
        rope_base: float = 10000.0,
        rotary_fraction: float = 1.0,
        pos_emb_max_len: int = 5000,
        xscaling: bool = False,
        attn_mode: str = "full",
        sync_max_audio_length: bool = True,
        # MoE parameters
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_router_type: str = 'omni',
        moe_layer_indices: Optional[List[int]] = None,
        moe_load_balance_loss_weight: float = 0.01,
        moe_jitter_eps: float = 0.0,
        moe_init_from_ffn: bool = True,
    ):
        super().__init__(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            feat_out=feat_out,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            drop_rate=drop_rate,
            dropout_pre_encoder=dropout_pre_encoder,
            dropout_emb=dropout_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            ff_expansion=ff_expansion,
            pre_block_norm=pre_block_norm,
            self_attention_model=self_attention_model,
            rope_base=rope_base,
            rotary_fraction=rotary_fraction,
            pos_emb_max_len=pos_emb_max_len,
            xscaling=xscaling,
            attn_mode=attn_mode,
            sync_max_audio_length=sync_max_audio_length,
        )

        # Rebuild the (already-validated) per-block config so MoE experts are
        # built identically to the base FeedForward modules created by
        # ``super().__init__()``. ``self.self_attention_model`` has been
        # normalized by the base encoder (e.g. None -> "no_pos").
        cfg = TransformerEncoderConfig(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            ff_expansion=ff_expansion,
            pre_block_norm=pre_block_norm,
            subsampling_factor=subsampling_factor,
            attn_mode=attn_mode,
            self_attention_model=self.self_attention_model,
            rope_base=rope_base,
            rotary_fraction=rotary_fraction,
        )
        self._block_cfg = cfg
        self.ff_hidden_size = int(ff_expansion * d_model)

        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_router_type = moe_router_type
        self.moe_load_balance_loss_weight = moe_load_balance_loss_weight
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_init_from_ffn = moe_init_from_ffn

        if moe_router_type not in ('omni', 'switch'):
            raise ValueError(f"moe_router_type must be one of 'omni', 'switch', got '{moe_router_type}'")

        if moe_layer_indices is not None:
            self.moe_layer_indices = list(moe_layer_indices)
        else:
            self.moe_layer_indices = list(range(n_layers))

        for idx in self.moe_layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"moe_layer_indices contains invalid index {idx} for encoder with {n_layers} layers.")

        # Create shared router for omni-router mode
        if moe_router_type == 'omni':
            self.omni_router = SwitchGate(d_model=d_model, num_experts=moe_num_experts, jitter_eps=moe_jitter_eps)
        else:
            self.omni_router = None

        # Replace FFN modules in selected layers with MoEFeedForward
        for layer_idx in self.moe_layer_indices:
            layer = self.layers[layer_idx]

            original_ffn_state = layer.ffn.state_dict() if moe_init_from_ffn else None

            router = self.omni_router if moe_router_type == 'omni' else None
            moe_ffn = MoEFeedForward(
                cfg=cfg,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                router=router,
                jitter_eps=moe_jitter_eps,
            )

            if original_ffn_state is not None:
                for expert in moe_ffn.experts:
                    expert.load_state_dict(original_ffn_state)

            layer.ffn = moe_ffn

        num_moe_layers = len(self.moe_layer_indices)
        print(
            f"MoETransformerEncoder: Replaced {num_moe_layers} FFN modules with MoE "
            f"(experts={moe_num_experts}, top_k={moe_top_k}, "
            f"router='{moe_router_type}', ffn_hidden_size={self.ff_hidden_size}, "
            f"layers={self.moe_layer_indices})"
        )

        # Plain tensors avoid DDP buffer broadcasts; statistics are reduced explicitly.
        self._cum_counts: Optional[torch.Tensor] = None
        self._cum_prob_sum: Optional[torch.Tensor] = None
        self._cum_tokens: Optional[torch.Tensor] = None

    def forward(self, audio_signal, length, bypass_pre_encode=False):
        """Forward pass identical to :class:`TransformerEncoder.forward`,
        plus an automatic call to :meth:`accumulate_moe_stats` after the
        encoder has run when in training mode.

        Validation / inference paths skip the accumulation so the MoE
        diagnostic window only reflects training-time routing.
        """
        out = super().forward(audio_signal, length, bypass_pre_encode=bypass_pre_encode)
        if self.training:
            self.accumulate_moe_stats()
        return out

    def _ensure_cum_buffers(self, device: torch.device) -> None:
        """Lazily allocate the cumulative MoE diagnostic tensors on ``device``.

        If they already exist on a different device (e.g. the model was moved
        to a new device between calls), they are reallocated and any existing
        accumulation is dropped. In normal training the device is fixed, so
        this allocation happens exactly once per process.
        """
        n = len(self.moe_layer_indices)
        E = self.moe_num_experts
        need_new = (
            self._cum_counts is None
            or self._cum_counts.device != device
            or self._cum_counts.shape[0] != n
            or self._cum_counts.shape[1] != E
        )
        if need_new:
            self._cum_counts = torch.zeros(n, E, dtype=torch.long, device=device)
            self._cum_prob_sum = torch.zeros(n, E, dtype=torch.float32, device=device)
            self._cum_tokens = torch.zeros(n, dtype=torch.long, device=device)

    def accumulate_moe_stats(self) -> None:
        """Sum per-layer routing diagnostics from the most recent forward into
        the rank-local cumulative tensors.

        Called automatically at the end of :meth:`forward` when the encoder is
        in training mode. Safe to call multiple times: it just adds the
        per-layer stats stashed in each :class:`MoEFeedForward` since the last
        forward.
        """
        if not self.moe_layer_indices:
            return
        # Find first valid per-layer stat to discover the device.
        first_counts = None
        for layer_idx in self.moe_layer_indices:
            ffn = self.layers[layer_idx].ffn
            if isinstance(ffn, MoEFeedForward) and ffn._expert_counts is not None:
                first_counts = ffn._expert_counts
                break
        if first_counts is None:
            return
        self._ensure_cum_buffers(first_counts.device)
        for slot, layer_idx in enumerate(self.moe_layer_indices):
            ffn = self.layers[layer_idx].ffn
            if not isinstance(ffn, MoEFeedForward):
                continue
            counts = ffn._expert_counts
            prob_sum = ffn._gate_prob_sum
            n_tokens = ffn._num_tokens
            if counts is None or prob_sum is None or n_tokens == 0:
                continue
            # In-place, on-device, no host sync.
            self._cum_counts[slot].add_(counts.to(self._cum_counts.dtype))
            self._cum_prob_sum[slot].add_(prob_sum.to(self._cum_prob_sum.dtype))
            self._cum_tokens[slot].add_(n_tokens)

    def reset_moe_metrics(self) -> None:
        """Zero the cumulative MoE diagnostic tensors (no-op if not allocated)."""
        if self._cum_counts is not None:
            self._cum_counts.zero_()
        if self._cum_prob_sum is not None:
            self._cum_prob_sum.zero_()
        if self._cum_tokens is not None:
            self._cum_tokens.zero_()

    def get_moe_metrics(self, distributed: bool = True, reset: bool = True) -> Optional[Dict[str, Any]]:
        """Compute MoE routing diagnostics over the current accumulator window.

        Multinode-safe: at most one small all-reduce per call, of size
        ``2 * n_moe_layers * num_experts + n_moe_layers`` (~ a few KB at most
        common configurations). The trigger and reset must be made identically
        on all DDP ranks (handled by callers using the Lightning-synchronized
        ``global_step``).

        Args:
            distributed: If True and ``torch.distributed`` is initialized,
                all-reduce the cumulative buffers (sum across ranks) before
                computing metrics. Set False for single-process use.
            reset: If True, zero the cumulative buffers after reading.

        Returns:
            ``None`` if no MoE forward has been observed yet (all buffers
            empty), otherwise a dict with two sub-dicts:

            * ``scalars``: zero-dim ``torch.Tensor`` values suitable for
              ``self.log(...)``. Keys:

              - ``moe/load_cv``                   : mean coefficient of variation
                of dispatch fractions (0 = perfect balance).
              - ``moe/load_cv_max``               : worst-layer load CV.
              - ``moe/load_max_over_ideal``       : mean of (max(f) / ideal).
              - ``moe/load_min_over_ideal``       : mean of (min(f) / ideal).
              - ``moe/router_entropy_norm``       : mean H(rho) / log(E)
                (1 = uniform router, 0 = collapsed).
              - ``moe/router_entropy_norm_min``   : worst-layer normalized
                entropy.
              - ``moe/dead_experts``              : total # of (layer, expert)
                pairs with zero dispatch in the window.
              - ``moe/dead_experts_pct``          : same, normalized to
                ``[0, 1]``.

            * ``per_layer``: detached CPU tensors for histograms / heatmaps.
              Keys: ``load`` (``[L, E]``), ``rho`` (``[L, E]``), ``cv``
              (``[L]``), ``entropy_norm`` (``[L]``), ``dead`` (``[L]``).
        """
        if not self.moe_layer_indices:
            return None
        if self._cum_counts is None:
            return None

        counts = self._cum_counts.clone()
        prob_sum = self._cum_prob_sum.clone()
        tokens = self._cum_tokens.clone()

        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(prob_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tokens, op=torch.distributed.ReduceOp.SUM)

        if reset:
            self.reset_moe_metrics()

        # Bail if nothing has been observed yet.
        if int(tokens.sum().item()) == 0:
            return None

        n_layers = counts.shape[0]
        E = self.moe_num_experts
        K = self.moe_top_k

        tokens_safe = tokens.clamp_min(1).unsqueeze(-1).to(torch.float32)
        # f[l, j] = (# slots routed to expert j at layer l) / num_tokens
        # Sums to top_k per layer.
        f = counts.to(torch.float32) / tokens_safe
        # rho[l, j] = mean router probability for expert j at layer l.
        # Sums to 1 per layer.
        rho = prob_sum / tokens_safe

        ideal = K / E  # ideal per-expert dispatch fraction at perfect balance

        # Coefficient of variation of dispatch fractions per layer.
        f_mean = f.mean(dim=-1).clamp_min(1e-9)
        f_std = f.std(dim=-1, unbiased=False)
        cv = f_std / f_mean
        f_max = f.max(dim=-1).values
        f_min = f.min(dim=-1).values

        rho_safe = rho.clamp_min(1e-12)
        entropy = -(rho * rho_safe.log()).sum(dim=-1)
        entropy_norm = entropy / math.log(max(E, 2))

        dead_mask = counts == 0  # [L, E]
        dead_per_layer = dead_mask.sum(dim=-1)  # [L]
        total_slots = float(n_layers * E)

        scalars = {
            'moe/load_cv': cv.mean().detach(),
            'moe/load_cv_max': cv.max().detach(),
            'moe/load_max_over_ideal': (f_max / ideal).mean().detach(),
            'moe/load_min_over_ideal': (f_min / ideal).mean().detach(),
            'moe/router_entropy_norm': entropy_norm.mean().detach(),
            'moe/router_entropy_norm_min': entropy_norm.min().detach(),
            'moe/dead_experts': dead_per_layer.sum().to(torch.float32).detach(),
            'moe/dead_experts_pct': (dead_per_layer.to(torch.float32).sum() / total_slots).detach(),
        }

        # Move per-layer tensors to CPU once (only needed for histogram /
        # heatmap rendering on rank 0; the move is a single small DtoH copy).
        per_layer = {
            'load': f.detach().cpu(),
            'rho': rho.detach().cpu(),
            'cv': cv.detach().cpu(),
            'entropy_norm': entropy_norm.detach().cpu(),
            'dead': dead_per_layer.detach().cpu(),
        }

        return {'scalars': scalars, 'per_layer': per_layer}

    def get_moe_auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Collect and return the weighted mean of auxiliary load-balancing losses
        from all MoE feed-forward modules.

        The raw per-layer losses are averaged (not summed) so that
        ``moe_load_balance_loss_weight`` is independent of the number of MoE layers.

        Returns:
            torch.Tensor or None: Weighted scalar auxiliary loss, or None if
                no MoE layers have computed a loss yet.
        """
        total_loss = None
        n_losses = 0

        for layer_idx in self.moe_layer_indices:
            layer = self.layers[layer_idx]
            ff_module = layer.ffn

            if isinstance(ff_module, MoEFeedForward) and ff_module._aux_loss is not None:
                if total_loss is None:
                    total_loss = ff_module._aux_loss
                else:
                    total_loss = total_loss + ff_module._aux_loss
                n_losses += 1

        if total_loss is not None and n_losses > 0:
            total_loss = self.moe_load_balance_loss_weight * (total_loss / n_losses)

        return total_loss

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Override to remap non-MoE FFN state_dict keys to MoE expert keys.

        When loading from a pretrained TransformerEncoder checkpoint, the FFN
        keys have the form::

            layers.{i}.ffn.net.0.weight

        but the MoE encoder expects::

            layers.{i}.ffn.experts.{j}.net.0.weight

        This method detects such mismatches and duplicates the pretrained FFN
        weights into all expert slots, so every expert starts from the pretrained
        FFN weights. The router weights (which have no pretrained counterpart)
        remain at their random initialization.
        """
        if not self.moe_init_from_ffn:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        moe_layer_set = set(self.moe_layer_indices)

        # Pattern: prefix + layers.{layer_idx}.ffn.{param_suffix}
        # where param_suffix does NOT start with "experts." or "router."
        pattern = re.compile(r'^(' + re.escape(prefix) + r'layers\.(\d+)\.ffn)\.((?!experts\.|router\.).+)$')

        keys_to_add = {}
        keys_to_remove = []

        for key in list(state_dict.keys()):
            m = pattern.match(key)
            if m:
                layer_prefix = m.group(1)  # e.g. "layers.0.ffn"
                layer_idx = int(m.group(2))
                param_suffix = m.group(3)  # e.g. "ffn.0.weight"

                if layer_idx in moe_layer_set:
                    value = state_dict[key]
                    for expert_idx in range(self.moe_num_experts):
                        new_key = f"{layer_prefix}.experts.{expert_idx}.{param_suffix}"
                        keys_to_add[new_key] = value.clone()
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]
        state_dict.update(keys_to_add)

        if keys_to_remove:
            print(
                f"MoETransformerEncoder: Remapped {len(keys_to_remove)} pretrained FFN keys "
                f"into {len(keys_to_add)} expert keys for {len(self.moe_layer_indices)} MoE layers."
            )

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
