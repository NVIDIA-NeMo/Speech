# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture-specific acoustic-code predictors for EasyMagpie serving."""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

import torch
from easymagpie_vllm_omni.cfg import CFGOptions, guided_logits
from easymagpie_vllm_omni.config import NUM_SPECIAL_AUDIO_TOKENS, EasyMagpieOmniArch
from easymagpie_vllm_omni.local_transformer import EasyMagpieLocalTransformerPredictor
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.nemotron_h import NemotronHAttentionDecoderLayer, NemotronHMLPDecoderLayer

_DEFAULT_TOP_K = 80
_MIN_SAMPLING_TEMPERATURE = 1e-4


class CodecLatentAudioEmbedder(nn.Module):
    """Embed stacked FSQ indices exactly like NemotronTTS v2."""

    def __init__(self, arch: EasyMagpieOmniArch) -> None:
        super().__init__()
        self.num_codebooks = arch.num_stacked_codebooks
        self.codebook_size = arch.codebook_size
        self.embedding_dim = arch.embedding_dim
        self.latent_dim = len(arch.codec_num_levels_per_group)

        levels = torch.tensor(arch.codec_num_levels_per_group, dtype=torch.long)
        bases = torch.cumprod(torch.tensor([1, *arch.codec_num_levels_per_group[:-1]]), dim=0)
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("bases", bases, persistent=False)
        self.register_buffer("channels", torch.arange(self.num_codebooks).view(1, -1), persistent=False)

        self.audio_code_projection = nn.Linear(self.num_codebooks * self.latent_dim, self.embedding_dim)
        self.audio_special_embeddings = nn.Embedding(
            self.num_codebooks * NUM_SPECIAL_AUDIO_TOKENS,
            self.embedding_dim,
        )

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        nonnegative = torch.div(codes.unsqueeze(-1), self.bases, rounding_mode="floor") % self.levels
        scale = torch.div(self.levels, 2, rounding_mode="floor")
        return ((nonnegative - scale) / scale).to(self.audio_code_projection.weight.dtype)

    def _special_rows(self, channels: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        special_ids = (codes - self.codebook_size).clamp(min=0, max=NUM_SPECIAL_AUDIO_TOKENS - 1)
        return channels * NUM_SPECIAL_AUDIO_TOKENS + special_ids

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed every stacked codebook in one serving frame."""
        if codes.shape[-1] != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} code channels, got {codes.shape[-1]}")
        codes = codes.long()
        is_special = codes >= self.codebook_size
        latent = self._decode(codes.clamp(max=self.codebook_size - 1)).masked_fill(is_special.unsqueeze(-1), 0.0)
        embedded = self.audio_code_projection(latent.flatten(start_dim=1))

        special = self.audio_special_embeddings(self._special_rows(self.channels, codes))
        return embedded + (special * is_special.unsqueeze(-1)).sum(dim=1)

    def embed_codebooks(self, codes: torch.Tensor, codebook_indices: Sequence[int]) -> torch.Tensor:
        """Embed selected predictor channels using their slices of the shared projection."""
        if codes.shape[-1] != len(codebook_indices):
            raise ValueError(f"got {codes.shape[-1]} codes for {len(codebook_indices)} codebook indices")
        codes = codes.long()
        embedded = None
        for input_index, codebook_index in enumerate(codebook_indices):
            channel_codes = codes[:, input_index]
            is_special = (channel_codes >= self.codebook_size).unsqueeze(-1)
            latent = self._decode(channel_codes.clamp(max=self.codebook_size - 1)).masked_fill(is_special, 0.0)
            start = codebook_index * self.latent_dim
            stop = start + self.latent_dim
            weight = self.audio_code_projection.weight[:, start:stop]
            channel_embedded = torch.nn.functional.linear(latent, weight, self.audio_code_projection.bias)
            rows = self._special_rows(codebook_index, channel_codes)
            channel_embedded = channel_embedded + self.audio_special_embeddings(rows) * is_special
            embedded = channel_embedded if embedded is None else embedded + channel_embedded
        assert embedded is not None
        return embedded / len(codebook_indices)


class AcousticCodesPredictorBlock(nn.Module):
    """One cached v2 predictor block: one attention/MLP pair plus one code head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        predictor_config,
        codebook_index: int,
        num_output_tokens: int,
        prefix: str,
        cache_layer_index: int,
    ) -> None:
        super().__init__()
        self.codebook_indices = (codebook_index,)
        self.num_codes = 1
        self.layers = nn.ModuleList(
            [
                NemotronHAttentionDecoderLayer(
                    config=predictor_config,
                    layer_idx=0,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    parallel_config=vllm_config.parallel_config,
                    prefix=f"{prefix}.cache_layers.{cache_layer_index}",
                ),
                NemotronHMLPDecoderLayer(
                    config=predictor_config,
                    layer_idx=1,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    parallel_config=vllm_config.parallel_config,
                    prefix=f"{prefix}.blocks.{codebook_index}.layers.1",
                ),
            ]
        )
        mlp = self.layers[1]
        activation = str(predictor_config.mlp_hidden_act).lower()
        if activation in {"silu", "swish"}:
            mlp.mixer.act_fn = nn.SiLU()
        elif activation == "gelu":
            mlp.mixer.act_fn = nn.GELU()
        elif activation == "relu":
            mlp.mixer.act_fn = nn.ReLU()
        elif activation not in {"relu2", "relu_squared"}:
            raise ValueError(f"unsupported acoustic predictor MLP activation {activation!r}")

        self.norm = RMSNorm(predictor_config.hidden_size, eps=predictor_config.layer_norm_epsilon)
        self.codebook_projection = nn.Linear(predictor_config.hidden_size, num_output_tokens)
        self.num_output_tokens = num_output_tokens

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.codebook_projection(hidden_states)
        return logits.unsqueeze(1)


class EasyMagpieAcousticCodesPredictor(nn.Module):
    """NemotronTTS v2 predictor backed by vLLM-managed global KV caches."""

    requires_all_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        arch = EasyMagpieOmniArch.from_hf_config(vllm_config.model_config.hf_config)
        self.arch = arch
        self.audio_embedder = CodecLatentAudioEmbedder(arch)
        self.num_codebooks = arch.num_stacked_codebooks
        self.num_output_tokens = arch.codebook_size + 1
        self.temperature = 0.7
        self.top_k = _DEFAULT_TOP_K
        self._sample_top_k = min(self.top_k, self.num_output_tokens)

        predictor_config = vllm_config.model_config.hf_config
        self.blocks = nn.ModuleList(
            [
                AcousticCodesPredictorBlock(
                    vllm_config=vllm_config,
                    predictor_config=predictor_config,
                    codebook_index=codebook_index,
                    num_output_tokens=self.num_output_tokens,
                    prefix=prefix,
                    cache_layer_index=predictor_config.num_hidden_layers + codebook_index,
                )
                for codebook_index in range(self.num_codebooks)
            ]
        )

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._gumbel_buf = torch.zeros(
            max_num_tokens,
            self.num_codebooks,
            self._sample_top_k,
            dtype=torch.float32,
        )

        self._init_cfg_buffers(vllm_config)

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_embedder.embed_audio_frame(codes)

    @torch.no_grad()
    def advance(self, hidden_states: torch.Tensor, *, positions: torch.Tensor) -> None:
        """Populate every predictor cache using MASK conditioning between blocks."""
        num_tokens = hidden_states.shape[0]
        previous_codes = None
        previous_indices = None
        for block in self.blocks:
            if previous_codes is not None:
                hidden_states = hidden_states + self.audio_embedder.embed_codebooks(previous_codes, previous_indices)
            hidden_states = block(hidden_states, positions)
            previous_codes = torch.full(
                (num_tokens, block.num_codes),
                self.arch.mask_token_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
            previous_indices = block.codebook_indices

    @torch.no_grad()
    def generate_codes(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor,
        predict: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance every cache row and sample only rows in ``predict``."""
        num_tokens = hidden_states.shape[0]
        if predict is None:
            predict = torch.ones(num_tokens, dtype=torch.bool, device=hidden_states.device)
        if predict.shape != (num_tokens,):
            raise ValueError(f"expected predict mask {(num_tokens,)}, got {tuple(predict.shape)}")

        noise = self._gumbel_buf[:num_tokens]
        noise.uniform_(1e-20, 1.0 - 1e-20)
        noise.log_().neg_().log_().neg_()
        temperature = max(float(self.temperature), _MIN_SAMPLING_TEMPERATURE)

        codes = torch.full(
            (num_tokens, self.num_codebooks),
            self.arch.mask_token_id,
            dtype=torch.long,
            device=hidden_states.device,
        )
        previous_codes = None
        previous_indices = None
        code_offset = 0
        for block in self.blocks:
            if previous_codes is not None:
                hidden_states = hidden_states + self.audio_embedder.embed_codebooks(previous_codes, previous_indices)
            hidden_states = block(hidden_states, positions)
            logits = block.compute_logits(hidden_states)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
            values, indices = torch.topk(logits, self._sample_top_k, dim=-1)
            picked = (values / temperature + noise[:, code_offset : code_offset + block.num_codes]).argmax(
                dim=-1, keepdim=True
            )
            sampled = indices.gather(-1, picked).squeeze(-1)
            sampled = torch.where(sampled == self.arch.codebook_size, self.arch.audio_eos_id, sampled)
            previous_codes = torch.where(
                predict.unsqueeze(-1),
                sampled,
                torch.full_like(sampled, self.arch.mask_token_id),
            )
            codes[:, code_offset : code_offset + block.num_codes] = previous_codes
            previous_indices = block.codebook_indices
            code_offset += block.num_codes
        return codes

    def _init_cfg_buffers(self, vllm_config: VllmConfig) -> None:
        """Stage graph inputs once; keep both global cache branches in one batch."""
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        self._cfg_hidden = torch.zeros(max_tokens, self.arch.embedding_dim, dtype=dtype)
        self._cfg_positions = torch.zeros(max_tokens, dtype=torch.long)
        self._cfg_conditional_rows = torch.zeros(max_tokens, dtype=torch.long)
        self._cfg_unconditional_rows = torch.zeros(max_tokens, dtype=torch.long)
        self._cfg_scales = torch.ones(max_tokens, 1, dtype=torch.float32)
        self._cfg_predict = torch.zeros(max_tokens, dtype=torch.bool)
        self._cfg_temperature = torch.ones(1, dtype=torch.float32)
        self._cfg_trace = bool(os.environ.get('EASYMAGPIE_CFG_TRACE_PATH'))
        self._cfg_forward = (self._generate_cfg,)
        self._cfg_advance = (self.advance,)
        mode = getattr(vllm_config.compilation_config, 'cudagraph_mode', None)
        if (
            CFGOptions.from_hf_config(vllm_config.model_config.hf_config).enabled
            and mode
            and mode.has_full_cudagraphs()
        ):
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            from vllm.config import CUDAGraphMode

            # Full physical rows preserve attention metadata and native KV slots.
            # Capture the complete predictor, avoiding Python launches per block.
            self._cfg_forward = (CUDAGraphWrapper(self._generate_cfg, vllm_config, runtime_mode=CUDAGraphMode.FULL),)
            self._cfg_advance = (CUDAGraphWrapper(self.advance, vllm_config, runtime_mode=CUDAGraphMode.FULL),)

    @torch.no_grad()
    def advance_cfg(self, hidden_states, *, positions):
        """Populate predictor caches with address-stable pure-prefill inputs."""
        count = hidden_states.shape[0]
        hidden = self._cfg_hidden[:count]
        hidden.copy_(hidden_states)
        position_buf = self._cfg_positions[:count]
        position_buf.copy_(positions)
        self._cfg_advance[0](hidden, positions=position_buf)

    @torch.no_grad()
    def generate_codes_cfg(self, hidden_states, *, positions, conditional_rows, unconditional_rows, scales, predict):
        """Advance all cache rows and use one guided sample for each C/U pair.

        Row maps address physical scheduler rows, including prefill and padding.
        Both rows of a pair point to the same conditional row's random draw.
        """
        count = hidden_states.shape[0]
        hidden = self._cfg_hidden[:count]
        hidden.copy_(hidden_states)
        position_buf = self._cfg_positions[:count]
        position_buf.copy_(positions)
        conditional = self._cfg_conditional_rows[:count]
        conditional.copy_(conditional_rows)
        unconditional = self._cfg_unconditional_rows[:count]
        unconditional.copy_(unconditional_rows)
        scale_buf = self._cfg_scales[:count]
        scale_buf.copy_(scales)
        predict_buf = self._cfg_predict[:count]
        predict_buf.copy_(predict)
        noise = self._gumbel_buf[:count]
        noise.uniform_(1e-20, 1.0 - 1e-20)
        noise.log_().neg_().log_().neg_()
        self._cfg_temperature.fill_(max(float(self.temperature), _MIN_SAMPLING_TEMPERATURE))
        return self._cfg_forward[0](
            hidden, position_buf, conditional, unconditional, scale_buf, predict_buf, noise, self._cfg_temperature
        )

    def _generate_cfg(self, hidden, positions, conditional, unconditional, scales, predict, noise, temperature):
        count = hidden.shape[0]
        codes = torch.full(
            (count, self.num_codebooks), self.arch.mask_token_id, dtype=torch.long, device=hidden.device
        )
        diagnostics = hidden.new_zeros(count, 3, dtype=torch.float32)
        previous_codes = previous_indices = None
        offset = 0
        for block in self.blocks:
            if previous_codes is not None:
                hidden = hidden + self.audio_embedder.embed_codebooks(previous_codes, previous_indices)
            # A single batched invocation advances every conditional, unconditional
            # and prefill KV row. Never run a decode-only slice through these blocks.
            hidden = block(hidden, positions)
            logits = torch.nan_to_num(block.compute_logits(hidden), nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits.clamp(-100.0, 100.0)
            cond, uncond = logits[conditional], logits[unconditional]
            guided = guided_logits(cond, uncond, scales.unsqueeze(-1))
            if self._cfg_trace and offset == 0:
                diagnostics = torch.stack(
                    (
                        (cond.float() - uncond.float()).abs().flatten(1).amax(-1),
                        (guided - cond.float()).abs().flatten(1).amax(-1),
                        (guided[:, 0].argmax(-1) != cond[:, 0].argmax(-1)).float(),
                    ),
                    dim=-1,
                )
            values, indices = torch.topk(guided, self._sample_top_k, dim=-1)
            shared_noise = noise[conditional, offset : offset + block.num_codes]
            picked = (values / temperature + shared_noise).argmax(-1, keepdim=True)
            sampled = indices.gather(-1, picked).squeeze(-1)
            sampled = torch.where(sampled == self.arch.codebook_size, self.arch.audio_eos_id, sampled)
            previous_codes = torch.where(predict.unsqueeze(-1), sampled, self.arch.mask_token_id)
            codes[:, offset : offset + block.num_codes] = previous_codes
            previous_indices = block.codebook_indices
            offset += block.num_codes
        return codes, diagnostics

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load NeMo predictor weights, packing q/k/v through vLLM loaders."""
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            for shard_name, shard_id in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
                marker = f".{shard_name}."
                if marker not in name:
                    continue
                mapped = name.replace(marker, ".qkv_proj.")
                param = params[mapped]
                param.weight_loader(param, tensor, shard_id)
                loaded.add(mapped)
                break
            else:
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor)
                loaded.add(name)
        return loaded


def create_acoustic_predictor(*, vllm_config: VllmConfig, prefix: str = "") -> nn.Module:
    """Construct the predictor selected by the converted artifact."""
    arch = EasyMagpieOmniArch.from_hf_config(vllm_config.model_config.hf_config)
    if arch.acoustic_predictor_type == "local_transformer":
        return EasyMagpieLocalTransformerPredictor(vllm_config=vllm_config, prefix=prefix)
    if arch.acoustic_predictor_type == "acoustic_codes_predictor":
        return EasyMagpieAcousticCodesPredictor(vllm_config=vllm_config, prefix=prefix)
    raise ValueError(f"unsupported acoustic predictor {arch.acoustic_predictor_type!r}")
