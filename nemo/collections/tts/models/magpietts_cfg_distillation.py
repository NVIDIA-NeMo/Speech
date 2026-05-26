# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Adapted MagpieTTSModel class for classifier-free guidance distillation.
"""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

from nemo.collections.tts.losses.magpietts_cfg_distillation import (
    CodesCrossEntropyLoss,
    KLDivergenceLoss,
    NRMSELogitsLoss,
)
from nemo.collections.tts.models.magpietts import ContextTensorsOutput, MagpieTTSModel
from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod, remove_embedded_eos_token
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.lightning.callback_group import CallbackGroup
from nemo.utils import logging

__all__ = ["OnlineCFGDistillation"]

_STATE_DICT_EXCLUDE_NAMES: set[str] = {
    "_codec_model",
    "_teacher_model",
}


@dataclass
class _DefaultParams:

    # Maximum number of decoding steps during audio rollout generation.
    max_decoder_steps: int = 430
    # Sampling temperature during rollout generation.
    rollout_temperature: float = 0.7
    # Top-k sampling limit for token selection.
    rollout_topk: int = 80
    # Whether to use key-value cache during rollout inference.
    use_kv_cache_during_rollout: bool = True
    # Classifier-free guidance (CFG) scale used during distillation.
    distillation_cfg_scale: float = 2.5
    # Temperature used for softening logits during distillation (1.0 means no change).
    distillation_temperature: float = 1.0
    # Weight coefficient in the combined entropy-divergence distillation loss.
    alpha: float = 0.3
    # Weight coefficient for the NRMSE component in the distillation loss.
    beta: float = 2.0
    # Fraction of the ground-truth sequence length used as a cutoff for early rollout truncation.
    truncation_threshold: Optional[float] = 1.25
    # Weight assigned to truncated samples when computing the loss (used to down-weight truncated rollouts).
    truncation_weight: Optional[float] = 0.1
    # Weight coefficient for the MoE loss component.
    moe_loss_weight: float = 1.0


_DEFAULT_PARAMS = _DefaultParams()


def _validate_configuration(cfg: DictConfig) -> None:
    if not hasattr(cfg, "prior_scaling_factor"):
        raise ValueError("Missing required parameter `prior_scaling_factor` in configuration.")

    if cfg.get("prior_scaling_factor") is not None:
        raise ValueError(
            "Attention priors are not applied during distillation. "
            "Please set `prior_scaling_factor = None` in the configuration."
        )

    if hasattr(cfg, "distillation_temperature") and cfg.get("distillation_temperature") <= 0:
        raise ValueError(
            "`distillation_temperature` must be greater than 0. "
            "Typical values for distillation are in the range [1.0, 4.0]."
        )

    if hasattr(cfg, "alpha") and not (0 <= cfg.get("alpha") <= 1):
        raise ValueError(
            "`alpha` must be in the range [0, 1]. "
            "It controls the weighting between KL-divergence and cross-entropy losses."
        )

    if hasattr(cfg, "beta") and cfg.get("beta") < 0:
        raise ValueError("`beta` must be non-negative. It scales the contribution of the NRMSE loss component.")

    if (
        hasattr(cfg, "truncation_threshold")
        and cfg.get("truncation_threshold") is not None
        and cfg.get("truncation_threshold") < 1.0
    ):
        raise ValueError(
            "`truncation_threshold` must be >= 1.0 or `None`. "
            "Values below 1.0 would truncate sequences shorter than the ground truth."
        )

    if (
        hasattr(cfg, "truncation_weight")
        and cfg.get("truncation_weight") is not None
        and cfg.get("truncation_weight") < 0
    ):
        raise ValueError(
            "`truncation_weight` must be non-negative or `None`. "
            "It defines the relative weighting for truncated samples in the loss."
        )


def _get_teacher_model(cfg: DictConfig) -> MagpieTTSModel:
    model_path = Path(cfg.teacher_model_path)
    teacher_model_cfg = copy.deepcopy(cfg)

    with open_dict(teacher_model_cfg):
        teacher_model_cfg.train_ds = None
        teacher_model_cfg.validation_ds = None

    if model_path.suffix == ".ckpt":
        teacher_model = MagpieTTSModel(cfg=teacher_model_cfg)
        ckpt = torch.load(model_path.as_posix(), map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        teacher_model.load_state_dict(state_dict)

    elif model_path.suffix == ".nemo":
        teacher_model = MagpieTTSModel.restore_from(
            restore_path=model_path.as_posix(),
            override_config_path=teacher_model_cfg,
            map_location="cpu",
            strict=cfg.get("init_strict", True),
        )
    else:
        raise ValueError(f"Unsupported teacher model format: {model_path.suffix}")

    teacher_model.freeze()
    teacher_model.eval()
    teacher_model._no_state_dict = True

    return teacher_model


def _prepare_forward_inputs(
    model: MagpieTTSModel,
    context_tensors: ContextTensorsOutput,
    audio_codes_embedded: Tensor,
    audio_codes_mask: Tensor,
) -> dict[str, Tensor | None]:
    additional_decoder_input = context_tensors.additional_decoder_input
    additional_decoder_mask = context_tensors.additional_decoder_mask
    attn_prior = [None, None] if model.model_type == "multi_encoder_context_tts" else None

    if additional_decoder_input is not None:
        dec_input_embedded = torch.cat([additional_decoder_input, audio_codes_embedded], dim=1)
        dec_input_mask = torch.cat([additional_decoder_mask, audio_codes_mask], dim=1)
    else:
        dec_input_embedded = audio_codes_embedded
        dec_input_mask = audio_codes_mask

    return {
        "dec_input_embedded": dec_input_embedded,
        "dec_input_mask": dec_input_mask,
        "cond": context_tensors.cond,
        "cond_mask": context_tensors.cond_mask,
        "attn_prior": attn_prior,
        "multi_encoder_mapping": context_tensors.multi_encoder_mapping,
    }


def _process_moe_routing_info(
    moe_routing_info: Optional[list[dict]],
    dec_input_mask: Tensor,
) -> dict[str, Tensor]:
    all_router_logits, all_router_probs, all_expert_indices = [], [], []

    for layer_routing_info in moe_routing_info:
        all_router_logits.append(layer_routing_info["router_logits"])
        all_router_probs.append(layer_routing_info["router_probs"])
        all_expert_indices.append(layer_routing_info["expert_indices"])

    stacked_logits = torch.stack(all_router_logits, dim=0)
    stacked_probs = torch.stack(all_router_probs, dim=0)
    num_layers, _, seq_len, num_experts = stacked_logits.size()

    merged_logits = stacked_logits.view(-1, seq_len, num_experts)
    merged_probs = stacked_probs.view(-1, seq_len, num_experts)
    merged_mask = dec_input_mask.unsqueeze(0).repeat(num_layers, 1, 1).view(-1, seq_len)

    return {
        "router_logits": merged_logits,
        "router_probs": merged_probs,
        "x_mask": merged_mask,
    }


def _process_batch_student(
    model: MagpieTTSModel,
    batch: dict[str, Tensor | list],
) -> tuple[Tensor, Optional[dict[str, Tensor]]]:
    """Perform a teacher-forced forward decoding pass for a MagpieTTS student model.

    This method runs a standard forward pass without classifier-free guidance (CFG).
    It prepares decoder inputs from the provided audio code sequence, removes the
    terminal EOS token, and computes logits for all positions after the decoder context
    prefix. If the model uses a Mixture-of-Experts (MoE) decoder, aggregated routing
    information is also processed and returned.

    Args:
        model (MagpieTTSModel): Student model instance used for the forward pass.
        batch (dict[str, Tensor | list]): Input batch containing audio token codes,
            audio code lengths, and contextual tensors required for decoding.
            The audio code sequence is expected to already include the special tokens
            required by the decoder input convention.

    Returns:
        tuple[Tensor, Optional[dict[str, Tensor]]]:
            - **logits (Tensor)**: Logits tensor of shape `(B, T', D)`, where `B` is
              batch size, `T'` is the frame-stacked decoder sequence length after
              removing the decoder context prefix, and `D` is the concatenated logit
              dimension across codebooks and frame-stacking positions.
            - **moe_routing_data (Optional[dict[str, Tensor]])**: Aggregated Mixture-of-Experts
              routing data, or `None` if MoE is disabled or routing information is unavailable.
    """
    context_tensors = model.prepare_context_tensors(batch)
    audio_codes = batch["audio_codes"]
    audio_codes_lens = batch["audio_codes_lens"]
    dec_context_size = context_tensors.dec_context_size
    moe_routing_data = None

    audio_codes_embedded_all, audio_codes_lens_all = model.embed_audio_tokens(
        audio_tokens=audio_codes,
        audio_tokens_lens=audio_codes_lens,
    )
    audio_codes_embedded, audio_codes_lens_ = remove_embedded_eos_token(
        embedded=audio_codes_embedded_all,
        embedded_len=audio_codes_lens_all,
    )
    audio_codes_mask = get_mask_from_lengths(audio_codes_lens_)
    inputs = _prepare_forward_inputs(model, context_tensors, audio_codes_embedded, audio_codes_mask)
    logits, _, _, moe_routing_info = model.forward(**inputs)
    logits = logits[:, dec_context_size:, :]

    if model.use_moe and moe_routing_info is not None:
        moe_routing_data = _process_moe_routing_info(
            moe_routing_info=moe_routing_info,
            dec_input_mask=inputs["dec_input_mask"],
        )
    return logits, moe_routing_data


def _infer_batch_teacher(
    model: MagpieTTSModel,
    batch: dict[str, Tensor | list],
    max_decoder_steps: int = 500,
    temperature: float = 0.7,
    topk: int = 80,
    cfg_scale: int = 2.5,
    truncation_threshold: Optional[float] = None,
    truncation_weight: Optional[float] = None,
    use_kv_cache: bool = False,
    eos_detection_method: str = "argmax_or_multinomial_any",
    min_generated_frames: int = 4,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Perform autoregressive batch inference for a MagpieTTS teacher model.

    This method generates audio token rollouts conditioned on text and context inputs.
    It performs autoregressive decoding with classifier-free guidance (CFG) by combining
    conditional and unconditional logits using the specified guidance scale. The function
    also supports optional truncation-based early stopping and per-sample weighting for
    downstream distillation losses.

    Generation is performed in units of stacked frames. When frame stacking is enabled,
    each decoder step predicts a full stacked block of audio codes. If EOS is detected
    inside a stacked block, the entire block is retained in the rollout. This behavior
    is intentional: distillation is performed over complete generated stacked steps rather
    than truncating at the exact EOS frame inside the final block.

    Args:
        model (MagpieTTSModel): The teacher model instance used for autoregressive generation.
        batch (dict[str, Tensor | list]): Input batch containing text tokens, audio code lengths,
            and additional contextual tensors required for decoding.
        max_decoder_steps (int, optional): Maximum number of generated audio frames in the
            unstacked time domain. Defaults to 500.
        temperature (float, optional): Sampling temperature controlling randomness during token
            generation. Defaults to 0.7.
        topk (int, optional): Top-k sampling limit. Only the k most probable tokens are considered
            at each decoding step. Defaults to 80.
        cfg_scale (float, optional): Classifier-free guidance scale factor controlling the strength
            of conditioning. Higher values increase adherence to conditioning. Defaults to 2.5.
        truncation_threshold (Optional[float], optional): Fraction of ground-truth length used as
            a cutoff for early truncation. If set, generation stops once this threshold is reached.
            Defaults to None.
        truncation_weight (Optional[float], optional): Weight assigned to truncated samples for
            downstream loss computation. Defaults to None.
        use_kv_cache (bool, optional): Whether to enable key-value caching during inference.
            Defaults to False.
        eos_detection_method (str, optional): End-of-sequence detection strategy. Must be one of
            the supported `EOSDetectionMethod` values. Defaults to `"argmax_or_multinomial_any"`.
        min_generated_frames (int, optional): Minimum number of generated frames before EOS detection
            is allowed. Prevents premature termination. Defaults to 4.

    Returns:
        tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
            - **predicted_codes (Tensor)**: Generated discrete audio codes of shape `(B, C, T)`,
              where `T` is in the unstacked time domain.
            - **predicted_codes_logits (Tensor)**: Decoder logits collected per stacked decoding step,
              shape `(B, T', D)`, where `T'` is the frame-stacked sequence length.
            - **predicted_codes_lens (Tensor)**: Predicted rollout lengths per batch item in the
              unstacked time domain, shape `(B,)`.
            - **sample_weights (Optional[Tensor])**: Optional per-sample weighting factors, shape `(B,)`.
    """
    model.decoder.reset_cache(use_cache=use_kv_cache)
    eos_detection_method = EOSDetectionMethod(eos_detection_method)
    context_tensors = model.prepare_context_tensors(batch)
    text = context_tensors.text
    bs = text.size(0)
    device = text.device
    fs_factor = model.frame_stacking_factor
    dims = (bs, model.num_audio_codebooks, fs_factor)
    audio_codes_input = torch.full(dims, model.audio_bos_id, device=device).long()
    audio_codes_lens = torch.full((bs,), fs_factor, device=device).long()
    truncation_count = 0

    if truncation_weight is None:
        sample_weights = None
    else:
        sample_weights = torch.ones(bs, dtype=torch.float, device=device)

    if truncation_threshold is not None:
        truncation_thresholds = truncation_threshold * batch["audio_codes_lens"].clone()
        truncation_thresholds = torch.round(truncation_thresholds).long()

    dummy_cond, dummy_cond_mask, dummy_additional_decoder_input, dummy_addition_dec_mask, _ = (
        model.prepare_dummy_cond_for_cfg(
            context_tensors.cond,
            context_tensors.cond_mask,
            context_tensors.additional_decoder_input,
            context_tensors.additional_decoder_mask,
        )
    )
    predicted_codes_logits = []
    predictions = []
    # Stores the start of the final retained stacked block.
    end_indices = {}
    attn_prior = [None, None] if model.model_type == "multi_encoder_context_tts" else None

    with torch.no_grad():
        for idx in range(max_decoder_steps // fs_factor):
            audio_codes_embedded, audio_codes_embedded_lens = model.embed_audio_tokens(
                audio_tokens=audio_codes_input, audio_tokens_lens=audio_codes_lens
            )
            audio_codes_mask = get_mask_from_lengths(audio_codes_embedded_lens)

            if context_tensors.additional_decoder_input is not None:
                _audio_codes_embedded = torch.cat(
                    [context_tensors.additional_decoder_input, audio_codes_embedded], dim=1
                )
                _audio_codes_mask = torch.cat([context_tensors.additional_decoder_mask, audio_codes_mask], dim=1)
            else:
                _audio_codes_embedded = audio_codes_embedded
                _audio_codes_mask = audio_codes_mask

            cfg_audio_codes_embedded = torch.cat([_audio_codes_embedded, _audio_codes_embedded], dim=0)
            cfg_audio_codes_mask = torch.cat([_audio_codes_mask, _audio_codes_mask], dim=0)
            cfg_cond = torch.cat([context_tensors.cond, dummy_cond], dim=0)
            cfg_cond_mask = torch.cat([context_tensors.cond_mask, dummy_cond_mask], dim=0)

            if dummy_additional_decoder_input is not None:
                index = dummy_additional_decoder_input.size(1)
                cfg_audio_codes_embedded[bs:, :index] = dummy_additional_decoder_input
                cfg_audio_codes_mask[bs:, :index] = dummy_addition_dec_mask

            combined_logits, _, _, _ = model.forward(
                dec_input_embedded=cfg_audio_codes_embedded,
                dec_input_mask=cfg_audio_codes_mask,
                cond=cfg_cond,
                cond_mask=cfg_cond_mask,
                attn_prior=attn_prior,
                multi_encoder_mapping=context_tensors.multi_encoder_mapping,
            )
            cond_logits = combined_logits[:bs]
            uncond_logits = combined_logits[bs:]
            all_code_logits = (1 - cfg_scale) * uncond_logits + cfg_scale * cond_logits
            forbid_audio_eos = idx * fs_factor < min_generated_frames
            all_code_logits_t = all_code_logits[:, -1, :]
            predicted_codes_logits.append(all_code_logits_t.unsqueeze(1))

            audio_codes_next = model.sample_codes_from_logits(
                all_code_logits_t,
                temperature=temperature,
                topk=topk,
                forbid_audio_eos=forbid_audio_eos,
            )
            all_codes_next_argmax = model.sample_codes_from_logits(
                all_code_logits_t,
                temperature=0.01,
                topk=1,
                forbid_audio_eos=forbid_audio_eos,
            )

            for item_idx in range(bs):
                if item_idx in end_indices:
                    continue

                global_index = idx * fs_factor

                if truncation_threshold is not None and global_index >= truncation_thresholds[item_idx].item():
                    end_indices[item_idx] = global_index
                    truncation_count += 1

                    if truncation_weight is not None:
                        sample_weights[item_idx] = truncation_weight

                    print(f"Item {item_idx} truncated at decoder timestep: {idx}")

                else:
                    end_frame_index = model.detect_eos(
                        audio_codes_multinomial=audio_codes_next[item_idx],
                        audio_codes_argmax=all_codes_next_argmax[item_idx],
                        eos_detection_method=eos_detection_method,
                    )
                    if end_frame_index != float("inf"):
                        # Intentionally retain the full stacked block containing EOS.
                        end_indices[item_idx] = global_index
                        print(f"End detected for item {item_idx} at decoder timestep: {idx}")

            predictions.append(audio_codes_next)
            audio_codes_input = torch.cat([audio_codes_input, audio_codes_next], dim=-1)
            audio_codes_lens = audio_codes_lens + fs_factor

            if len(end_indices) == bs and len(predictions) >= 4:
                msg = "All ends reached"
                if truncation_threshold is not None:
                    msg += f", truncated samples: {truncation_count}"
                print(msg)
                break

        predicted_codes = torch.cat(predictions, dim=-1)
        predicted_codes_logits = torch.cat(predicted_codes_logits, dim=1)

        max_step = predicted_codes.size(-1)
        predicted_lens = [end_indices[idx] + fs_factor if idx in end_indices else max_step for idx in range(bs)]
        predicted_codes_lens = torch.tensor(predicted_lens, device=text.device).long()

        max_len = predicted_codes_lens.max()
        max_stacked_len = max_len // fs_factor
        predicted_codes = predicted_codes[:, :, :max_len]
        predicted_codes_logits = predicted_codes_logits[:, :max_stacked_len, :]

    model.decoder.reset_cache(use_cache=False)
    torch.cuda.empty_cache()

    return predicted_codes, predicted_codes_logits, predicted_codes_lens, sample_weights


def _collect_validation_outputs(
    outputs: list[dict[str, Tensor]] | list[list[dict[str, Tensor]]],
    key: str,
) -> Optional[Tensor]:
    values = []

    def _add(items: list[dict[str, Tensor]]) -> None:
        for item in items:
            val = item.get(key)
            if val is not None:
                values.append(val)

    if outputs and isinstance(outputs[0], list):
        for items in outputs:
            _add(items)
    else:
        _add(outputs)

    if not values:
        return None

    return torch.stack(values).mean()


class OnlineCFGDistillation(MagpieTTSModel):
    """Implements online classifier-free guidance (CFG) distillation for MagpieTTS."""

    def __init__(
        self,
        cfg: DictConfig,
        trainer: "Trainer" = None,
    ) -> None:
        _validate_configuration(cfg)
        super().__init__(cfg, trainer)
        self._init_extra_attributes()

        if self.alpha != 1.0:
            self._kl_criterion = KLDivergenceLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
            )
        if self.alpha != 0.0:
            self._ce_criterion = CodesCrossEntropyLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
            )
        if self.beta != 0.0:
            self._nrmse_criterion = NRMSELogitsLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
            )
        self._teacher_model: Optional[MagpieTTSModel] = None

    def _load_teacher_model(self) -> None:
        if self._teacher_model is None:
            print("Loading teacher model from checkpoint.")
            self._teacher_model = _get_teacher_model(self.cfg).to(self.device)
            print("Teacher model loaded and frozen.")

    def on_fit_start(self) -> None:
        """See the ModelPT class docstring."""
        super().on_fit_start()
        self._load_teacher_model()

    @rank_zero_only
    def maybe_init_from_pretrained_checkpoint(
        self,
        cfg: OmegaConf,
        map_location: str = "cpu",
    ) -> None:
        """See the ModelPT class docstring."""
        args = ["init_from_nemo_model", "init_from_ptl_ckpt"]
        arg_matches = [(1 if arg in cfg and cfg[arg] is not None else 0) for arg in args]

        if sum(arg_matches) == 0:
            return

        if sum(arg_matches) > 1:
            raise ValueError(
                f"Cannot pass more than one model initialization arguments to config!\n"
                f"Found : {[args[idx] for idx, arg_present in enumerate(arg_matches) if arg_present]}"
            )

        CallbackGroup.get_instance().on_load_checkpoint_start()

        if "init_from_nemo_model" in cfg and cfg.init_from_nemo_model is not None:
            model_path = cfg.init_from_nemo_model
            restore_cfg = copy.deepcopy(self.cfg)

            with open_dict(restore_cfg):
                restore_cfg.train_ds = None
                restore_cfg.validation_ds = None

            if isinstance(model_path, str):
                restored_model = MagpieTTSModel.restore_from(
                    restore_path=model_path,
                    override_config_path=restore_cfg,
                    map_location=map_location,
                    strict=cfg.get("init_strict", True),
                )
                self.load_state_dict(restored_model.state_dict(), strict=False)
                logging.info(f'Model checkpoint restored from nemo file with path : `{model_path}`')
                del restored_model
            else:
                raise TypeError("Invalid type: init_from_nemo_model is not a string!")

        elif "init_from_ptl_ckpt" in cfg and cfg.init_from_ptl_ckpt is not None:
            with open_dict(cfg):
                if isinstance(cfg.init_from_ptl_ckpt, str):
                    ckpt_path = cfg.get("init_from_ptl_ckpt")
                    ckpt = torch.load(ckpt_path, map_location=map_location)
                    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                    self.load_state_dict(state_dict, strict=False)
                    logging.info(
                        f'Model checkpoint restored from pytorch lightning checkpoint with path : `{ckpt_path}`'
                    )
                    del ckpt
                else:
                    raise TypeError("Invalid type: init_from_ptl_ckpt is not a string!")

        CallbackGroup.get_instance().on_load_checkpoint_end()

    def _init_extra_attributes(self) -> None:
        defaults = vars(_DEFAULT_PARAMS)
        for k, v in defaults.items():
            setattr(self, k, self.cfg.get(k, v))

    def state_dict(
        self,
        destination: Optional[dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """See the MagpieTTSModel class docstring."""
        state_dict = super().state_dict(destination, prefix, keep_vars)

        for key in list(state_dict.keys()):
            if any(substring in key for substring in _STATE_DICT_EXCLUDE_NAMES):
                del state_dict[key]

        return state_dict

    def _add_batch_audio_codes(
        self,
        batch: dict[str, Tensor | list],
    ) -> dict[str, Tensor | list]:
        if "audio_codes" in batch:
            return batch

        audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(
            audio=batch["audio"],
            audio_len=batch["audio_lens"],
            sample_rate=batch.get("sample_rate"),
        )
        if self._codec_converter:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            )
        batch["audio_codes"] = audio_codes
        batch["audio_codes_lens"] = audio_codes_lens
        return batch

    def _update_batch(
        self,
        batch: dict[str, Tensor | list],
        rollout_codes: Tensor,
        rollout_lens: Tensor,
    ) -> dict[str, Tensor | list]:
        rollout_codes = torch.nn.functional.pad(
            input=rollout_codes,
            pad=(self.frame_stacking_factor, 0),
            value=self.audio_bos_id,
        )
        batch["audio_codes"] = rollout_codes
        batch["audio_codes_lens"] = rollout_lens + self.frame_stacking_factor
        return batch

    def _compute_loss(
        self,
        teacher_codes: Tensor,
        teacher_logits: Tensor,
        student_logits: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor],
        moe_routing_data: Optional[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}

        if self.alpha != 1.0:
            kl_loss = self._kl_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                sample_weights=sample_weights,
            )
            if self.distillation_temperature != 1.0:
                kl_loss = kl_loss * (self.distillation_temperature**2)

            output["kl_loss"] = kl_loss

        if self.alpha != 0.0:
            ce_loss = self._ce_criterion(
                predicted_logits=student_logits,
                target_codes=teacher_codes,
                mask=mask,
                sample_weights=sample_weights,
            )
            output["ce_loss"] = ce_loss

        loss = (1 - self.alpha) * output.get("kl_loss", 0.0) + self.alpha * output.get("ce_loss", 0.0)

        if self.beta > 0.0:
            nrmse_loss = self._nrmse_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                sample_weights=sample_weights,
            )
            output["nrmse_loss"] = nrmse_loss
            loss = loss + self.beta * nrmse_loss

        if moe_routing_data is not None:
            _, _, moe_loss = self.moe_auxiliary_loss(**moe_routing_data)
            output["moe_loss"] = moe_loss
            loss = loss + self.moe_loss_weight * moe_loss

        output["loss"] = loss

        return output

    def _process_batch_distillation(
        self,
        batch: dict[str, Tensor | list],
        mode: str = "train",
    ) -> dict[str, Tensor]:
        """Perform a knowledge distillation step between teacher and student models.

        This method orchestrates the end-to-end distillation process:
        1. The teacher model generates rollouts.
        2. The student model performs a teacher-forced forward pass on those rollouts.
        3. Multiple loss components are computed, including KL divergence, cross-entropy,
        normalized RMSE (NRMSE), and optionally Mixture-of-Experts (MoE) auxiliary loss.
        4. The total loss is formed from the active loss terms, controlled by the
        coefficients `alpha`, `beta`, and `moe_loss_weight`.

        Args:
            batch (dict[str, Tensor | list]): Input batch containing text tokens, conditioning
                features, and any additional fields required for teacher inference and student
                loss computation.
            mode (str, optional): Execution mode, either `"train"` or `"validation"`.
                In `"train"` mode, truncation and sample weighting may be applied to improve
                efficiency. Defaults to `"train"`.

        Returns:
            dict[str, Tensor]: Dictionary containing computed loss components and auxiliary values:
                - **loss (Tensor)**: Total weighted distillation loss combining all active components.
                - **kl_loss (Tensor, optional)**: KL divergence between student and teacher logits.
                Included when `alpha != 1.0`.
                - **ce_loss (Tensor, optional)**: Cross-entropy loss between student predictions and
                teacher-generated discrete audio codes. Included when `alpha != 0.0`.
                - **nrmse_loss (Tensor, optional)**: Normalized RMSE loss between student and teacher
                logits. Included when `beta > 0.0`.
                - **moe_loss (Tensor, optional)**: Auxiliary Mixture-of-Experts routing loss.
                Included when MoE routing data is available.
        """
        batch = self._add_batch_audio_codes(batch)

        rollout_codes, rollout_logits, rollout_lens, sample_weights = _infer_batch_teacher(
            model=self._teacher_model,
            batch=batch,
            max_decoder_steps=self.max_decoder_steps,
            temperature=self.rollout_temperature,
            topk=self.rollout_topk,
            cfg_scale=self.distillation_cfg_scale,
            truncation_threshold=self.truncation_threshold if mode == "train" else None,
            truncation_weight=self.truncation_weight if mode == "train" else None,
            use_kv_cache=self.use_kv_cache_during_rollout,
        )

        batch = self._update_batch(batch, rollout_codes, rollout_lens)
        student_logits, moe_routing_data = _process_batch_student(model=self, batch=batch)
        mask = get_mask_from_lengths(rollout_lens)

        if self.distillation_temperature != 1.0:
            student_logits = student_logits / self.distillation_temperature
            rollout_logits = rollout_logits / self.distillation_temperature

        output = self._compute_loss(
            teacher_codes=rollout_codes,
            teacher_logits=rollout_logits,
            student_logits=student_logits,
            mask=mask,
            sample_weights=sample_weights,
            moe_routing_data=moe_routing_data,
        )
        return output

    def training_step(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
    ) -> Tensor:
        """Execute a single training step for the model.

        Args:
            batch (dict): Input batch containing text, conditioning, and audio codes.
            batch_idx (int): Index of the current batch within the epoch.

        Returns:
            Tensor: Scalar tensor representing the total training loss for this step.
        """
        outputs = self._process_batch_distillation(batch, mode="train")
        bs = batch["audio_codes"].size(0)
        loss = outputs["loss"]

        self.log(
            name="train/loss",
            value=loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
            on_step=True,
            on_epoch=True,
        )
        for key in ["kl_loss", "ce_loss", "nrmse_loss", "moe_loss"]:
            if key in outputs:
                self.log(
                    name=f"train/{key}",
                    value=outputs[key],
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=bs,
                    on_step=True,
                    on_epoch=True,
                )
        return loss

    def validation_step(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Execute a single validation step for the model.

            Args:
                batch (dict): Validation batch containing required model inputs.
                batch_idx (int): Index of the current validation batch.
                dataloader_idx (int): Index of the dataloader (0 for single dataloader).

        Returns:
            dict[str, Tensor]: Dictionary containing validation loss values returned by
                `_process_batch_distillation()`. Always includes `"loss"` and may also
                include `"kl_loss"`, `"ce_loss"`, `"nrmse_loss"`, and `"moe_loss"`.
        """
        val_output = self._process_batch_distillation(batch, mode="validation")
        self.validation_step_outputs[dataloader_idx].append(val_output)
        return val_output

    def _on_validation_epoch_end_logging(
        self,
        val_outputs: list[dict[str, Tensor]] | list[list[dict[str, Tensor]]],
        prefix: str,
        aggregated: bool = False,
    ) -> None:
        val_loss = _collect_validation_outputs(val_outputs, key="loss")

        if val_loss is not None:
            self.log(
                name=f"{prefix}/loss",
                value=val_loss,
                prog_bar=aggregated,
                sync_dist=True,
            )
            if aggregated:
                self.log(
                    name="val_loss",
                    value=val_loss,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    logger=False,
                    enable_graph=False,
                )

        for key in ["kl_loss", "ce_loss", "nrmse_loss", "moe_loss"]:
            value = _collect_validation_outputs(val_outputs, key)
            if value is not None:
                self.log(
                    name=f"{prefix}/{key}",
                    value=value,
                    prog_bar=aggregated,
                    sync_dist=True,
                )

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log validation metrics at the end of an epoch.

        Computes mean losses across all validation steps and logs them for both progress
        bar and external monitoring systems. Clears cached outputs to free memory.
        """
        self._on_validation_epoch_end_logging(self.validation_step_outputs, prefix="val", aggregated=True)

        if len(self.validation_step_outputs) > 1:
            for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
                if not val_outputs:
                    continue

                prefix = self.get_validation_dataloader_prefix(dataloader_idx)
                self._on_validation_epoch_end_logging(val_outputs, prefix=f"val-{prefix}")

        for val_outputs in self.validation_step_outputs:
            val_outputs.clear()
