# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from dataclasses import replace
from typing import List, Optional

import torch

from nemo.collections.tts.modules.nemotron_h_decoder import (
    HybridMambaAttentionDynamicCache,
    NemotronHBlock,
    NemotronHConfig,
    NemotronHRMSNorm,
)
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths


_IGNORE_INDEX = -100


class AcousticCodesPredictorBlock(torch.nn.Module):
    """A causal stack of Nemotron-H attention and feed-forward layer pairs."""

    def __init__(
        self,
        backbone_config: NemotronHConfig,
        first_code: int,
        num_codes: int,
        num_output_tokens: int,
        n_layers: int = 1,
    ):
        super().__init__()
        assert n_layers > 0, "acoustic_codes_predictor_n_layers must be positive"
        self.codebook_indices = tuple(range(first_code, first_code + num_codes))
        self.num_codes = num_codes
        self.num_output_tokens = num_output_tokens
        self.config = replace(
            backbone_config,
            num_hidden_layers=2 * n_layers,
            hybrid_override_pattern="*-" * n_layers,
        )
        self.layers = torch.nn.ModuleList(
            [NemotronHBlock(self.config, layer_idx=idx) for idx in range(self.config.num_hidden_layers)]
        )
        self.norm = NemotronHRMSNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)
        self.codebook_projection = torch.nn.Linear(self.config.hidden_size, num_codes * num_output_tokens)
        self._init_weights()

    def make_cache(self, batch_size: int, device, dtype) -> HybridMambaAttentionDynamicCache:
        return HybridMambaAttentionDynamicCache(
            self.config,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[HybridMambaAttentionDynamicCache] = None,
    ) -> torch.Tensor:
        cached = cache.get_seq_length() if cache is not None else 0
        if cached > 0 and hidden_states.size(1) > 1 and self.config._attn_implementation == "flash_attention_2":
            return torch.cat(
                [self(hidden_states[:, frame : frame + 1], cache=cache) for frame in range(hidden_states.size(1))],
                dim=1,
            )

        attention_mask = self._offset_causal_mask(hidden_states, cache)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cache_params=cache, attention_mask=attention_mask)
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project only the code channels assigned to this predictor block."""
        logits = self.codebook_projection(hidden_states)
        return logits.reshape(hidden_states.shape[:2] + (self.num_codes, self.num_output_tokens))

    @staticmethod
    def _offset_causal_mask(
        hidden_states: torch.Tensor,
        cache: Optional[HybridMambaAttentionDynamicCache],
    ) -> Optional[torch.Tensor]:
        frames = hidden_states.size(1)
        cached = cache.get_seq_length() if cache is not None else 0
        if cached == 0 or frames == 1:
            return None

        device = hidden_states.device
        query_positions = torch.arange(cached, cached + frames, device=device).unsqueeze(1)
        blocked = torch.arange(cached + frames, device=device) > query_positions
        mask = torch.zeros(blocked.shape, dtype=hidden_states.dtype, device=device)
        return mask.masked_fill(blocked, torch.finfo(hidden_states.dtype).min)[None, None]

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        if self.config.rescale_prenorm_residual:
            for name, parameter in self.named_parameters():
                if name.endswith(("o_proj.weight", "down_proj.weight")):
                    with torch.no_grad():
                        parameter /= math.sqrt(self.config.num_hidden_layers)


class AcousticCodesPredictor(torch.nn.Module):
    """Predict all acoustic-code channels iteratively from backbone hidden states."""

    def __init__(
        self,
        backbone_config: NemotronHConfig,
        embed_codes,
        num_audio_codebooks: int,
        audio_eos_id: int,
        mask_token_id: int,
        codebook_size: int,
        prediction_schedule,
        n_layers: int = 1,
    ):
        super().__init__()
        prediction_schedule = tuple(int(num_codes) for num_codes in prediction_schedule)
        assert prediction_schedule, "acoustic_codes_predictor_schedule must not be empty"
        assert all(
            num_codes > 0 for num_codes in prediction_schedule
        ), "acoustic_codes_predictor_schedule values must be positive"
        assert sum(prediction_schedule) == num_audio_codebooks, (
            f"acoustic_codes_predictor_schedule predicts {sum(prediction_schedule)} codes, "
            f"expected {num_audio_codebooks}"
        )

        self.embed_codes = embed_codes
        self.num_audio_codebooks = num_audio_codebooks
        self.audio_eos_id = audio_eos_id
        self.mask_token_id = mask_token_id
        self.codebook_size = codebook_size
        self.num_output_tokens = codebook_size + 1
        self.prediction_schedule = prediction_schedule

        blocks = []
        first_code = 0
        for num_codes in prediction_schedule:
            blocks.append(
                AcousticCodesPredictorBlock(
                    backbone_config=backbone_config,
                    first_code=first_code,
                    num_codes=num_codes,
                    num_output_tokens=self.num_output_tokens,
                    n_layers=n_layers,
                )
            )
            first_code += num_codes
        self.blocks = torch.nn.ModuleList(blocks)

    def make_cache(self, batch_size: int, device, dtype) -> List[HybridMambaAttentionDynamicCache]:
        return [block.make_cache(batch_size, device=device, dtype=dtype) for block in self.blocks]

    @staticmethod
    def cached_frames(cache: List[HybridMambaAttentionDynamicCache]) -> int:
        return cache[0].get_seq_length()

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_codes: torch.Tensor,
        lengths: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss over codec tokens and AUDIO_EOS on the backbone timeline."""
        assert target_codes.size(-1) == sum(self.prediction_schedule), (
            f"target has {target_codes.size(-1)} codes, but acoustic_codes_predictor_schedule "
            f"predicts {sum(self.prediction_schedule)}"
        )
        target_codes = target_codes.long()
        time_mask = get_mask_from_lengths(lengths, x=target_codes[:, :, 0])
        has_target = time_mask.unsqueeze(-1) & (
            ((target_codes >= 0) & (target_codes < self.codebook_size)) | (target_codes == self.audio_eos_id)
        )
        if loss_mask is None:
            supervised = has_target
        else:
            assert loss_mask.shape == target_codes.shape[:2], (
                f"loss mask has shape {tuple(loss_mask.shape)}, expected {tuple(target_codes.shape[:2])}"
            )
            supervised = has_target & loss_mask.to(device=target_codes.device, dtype=torch.bool).unsqueeze(-1)

        compact_targets = torch.where(target_codes == self.audio_eos_id, self.codebook_size, target_codes)
        loss_sum = hidden_states.new_zeros(())
        num_predictions = torch.zeros((), dtype=torch.long, device=target_codes.device)
        previous_codes = None
        previous_codebook_indices = None

        for block in self.blocks:
            if previous_codes is not None:
                hidden_states = hidden_states + self._embed(previous_codes, previous_codebook_indices)
            hidden_states = block(hidden_states)
            logits = block.compute_logits(hidden_states)

            first_code = block.codebook_indices[0]
            last_code = block.codebook_indices[-1] + 1
            block_is_supervised = supervised[:, :, first_code:last_code]
            block_has_target = has_target[:, :, first_code:last_code]
            block_targets = compact_targets[:, :, first_code:last_code]
            step_targets = torch.where(
                block_is_supervised,
                block_targets,
                torch.full_like(block_targets, _IGNORE_INDEX),
            )
            loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.num_output_tokens),
                step_targets.reshape(-1),
                ignore_index=_IGNORE_INDEX,
                reduction='sum',
            )
            num_predictions = num_predictions + block_is_supervised.sum()
            previous_codes = torch.where(
                block_has_target,
                target_codes[:, :, first_code:last_code],
                torch.full_like(block_targets, self.mask_token_id),
            )
            previous_codebook_indices = block.codebook_indices

        return loss_sum / num_predictions.clamp(min=1)

    @torch.no_grad()
    def advance(
        self,
        hidden_states: torch.Tensor,
        cache: List[HybridMambaAttentionDynamicCache],
    ) -> None:
        """Populate every predictor block's cache for positions without audio targets."""
        previous_codes = None
        previous_codebook_indices = None
        for block, block_cache in zip(self.blocks, cache):
            if previous_codes is not None:
                hidden_states = hidden_states + self._embed(previous_codes, previous_codebook_indices)
            hidden_states = block(hidden_states, cache=block_cache)
            previous_codes = torch.full(
                (hidden_states.size(0), hidden_states.size(1), block.num_codes),
                self.mask_token_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
            previous_codebook_indices = block.codebook_indices

    @torch.no_grad()
    def predict_codes(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[List[HybridMambaAttentionDynamicCache]] = None,
        temperature: float = 1.0,
        topk: int = 80,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        sanitize_logits: bool = False,
        predict: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample every acoustic-code channel without using backbone code predictions."""
        num_streams = hidden_states.size(0) // 2 if use_cfg else hidden_states.size(0)
        assert not use_cfg or hidden_states.size(0) == 2 * num_streams
        if predict is None:
            predict = torch.ones(num_streams, dtype=torch.bool, device=hidden_states.device)
        assert predict.shape == (
            num_streams,
        ), f"expected predict mask with shape {(num_streams,)}, got {tuple(predict.shape)}"

        codes = torch.full(
            (num_streams, hidden_states.size(1), self.num_audio_codebooks),
            self.mask_token_id,
            dtype=torch.long,
            device=hidden_states.device,
        )
        predicted_rows = predict.view(-1, 1, 1)
        previous_codes = None
        previous_codebook_indices = None
        for step, block in enumerate(self.blocks):
            if previous_codes is not None:
                embedded = self._embed(previous_codes, previous_codebook_indices)
                if use_cfg:
                    embedded = embedded.repeat(2, 1, 1)
                hidden_states = hidden_states + embedded
            hidden_states = block(hidden_states, cache=None if cache is None else cache[step])
            logits = block.compute_logits(hidden_states)
            if use_cfg:
                logits = cfg_scale * logits[:num_streams] + (1.0 - cfg_scale) * logits[num_streams:]
            if sanitize_logits:
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits.clamp(min=-100.0, max=100.0)

            sampled = self._sample_codes(logits, temperature=temperature, topk=topk)
            sampled = torch.where(sampled == self.codebook_size, self.audio_eos_id, sampled)
            previous_codes = torch.where(
                predicted_rows,
                sampled,
                torch.full_like(sampled, self.mask_token_id),
            )
            first_code = block.codebook_indices[0]
            last_code = block.codebook_indices[-1] + 1
            codes[:, :, first_code:last_code] = previous_codes
            previous_codebook_indices = block.codebook_indices

        return codes

    def _embed(self, codes: torch.Tensor, codebook_indices) -> torch.Tensor:
        return self.embed_codes(codes.transpose(1, 2), codebook_indices=codebook_indices)

    def _sample_codes(self, logits: torch.Tensor, temperature: float, topk: int) -> torch.Tensor:
        cutoff = logits.topk(min(topk, self.num_output_tokens), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs.reshape(-1, self.num_output_tokens), num_samples=1).reshape(probs.shape[:-1])
