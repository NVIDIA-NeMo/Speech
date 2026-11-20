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

import logging

import click

from .cut_id_dataset import CutIdDataset

LOG = logging.getLogger(__name__)


def validate_full_batch(batch, *, step: int) -> None:
    """Fail unless one production SALM batch fully materialized its payload."""
    if not isinstance(batch, dict):
        raise click.ClickException(f"full validation step {step} returned no materialized batch")
    required = ("audio_lens", "input_ids", "loss_mask", "conversations")
    missing = [key for key in required if batch.get(key) is None]
    if missing:
        raise click.ClickException(f"full validation step {step} is missing materialized fields: {', '.join(missing)}")
    has_dense_audio = batch.get("audios") is not None
    has_packed_audio = batch.get("packed_audio_samples") is not None and batch.get("audio_cu_seqlens") is not None
    if not (has_dense_audio or has_packed_audio):
        raise click.ClickException(f"full validation step {step} is missing an audio payload")
    if not batch["conversations"]:
        raise click.ClickException(f"full validation step {step} materialized an empty conversation batch")


def build_validation_dataset(full_cfg, tokenizer, *, mode: str, section: str = "train_ds"):
    """Build the production SALMDataset while keeping decode failures visible."""
    if mode == "fast":
        return CutIdDataset()
    if mode != "full":
        raise ValueError(f"Unknown validation mode: {mode!r}")
    if tokenizer is None:
        raise click.ClickException("full validation requires the production tokenizer")

    from nemo.collections.speechlm2.data.salm_dataset import SALMDataset

    data_cfg = full_cfg.get("data", {})
    model_cfg = full_cfg.get("model", {})
    kwargs = {"tokenizer": tokenizer, "strict_audio_loading": True}
    if (multispeaker_cfg := data_cfg.get("multispeaker_cfg")) is not None:
        kwargs["multispeaker_cfg"] = multispeaker_cfg
    pack_audio = bool(model_cfg.get("use_nemo_automodel", False) and model_cfg.get("packed_encoder_sequences", False))
    if pack_audio:
        kwargs["pack_audio"] = True
    if (batch_tokens := data_cfg.get(section, {}).get("batch_tokens")) is not None:
        kwargs["batch_tokens"] = batch_tokens
    # FallbackDataset would replay the prior batch after a decode failure and
    # hide the exact error this validation mode is intended to detect.
    return SALMDataset(**kwargs)


def build_tokenizer(full_cfg, section_cfg, *, required: bool = False):
    """Mirror production SALM tokenizer construction for sampling and full mode."""
    if not required and not section_cfg.get("use_multimodal_sampling", False):
        return None
    model_cfg = full_cfg.get("model", {})
    tokenizer_src = model_cfg.get("tokenizer_path") or model_cfg.get("pretrained_llm")
    if not tokenizer_src:
        raise click.ClickException(
            "full validation or use_multimodal_sampling=True requires model.tokenizer_path " "or model.pretrained_llm."
        )
    from nemo.collections.common.tokenizers import AutoTokenizer

    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer_kwargs = {"use_fast": True, "trust_remote_code": trust_remote_code}
    if model_cfg.get("use_nemo_automodel", False):
        tokenizer_kwargs["pad_token"] = model_cfg.get("pad_token")
    LOG.info(
        "loading tokenizer for %s (trust_remote_code=%s)",
        tokenizer_src,
        trust_remote_code,
    )
    tokenizer = AutoTokenizer(tokenizer_src, **tokenizer_kwargs)
    audio_tag = model_cfg.get("audio_locator_tag")
    if required and not audio_tag:
        raise click.ClickException("full validation requires model.audio_locator_tag")
    if audio_tag:
        tokenizer.add_special_tokens({"additional_special_tokens": [audio_tag]})
    if model_cfg.get("use_nemo_automodel", False):
        from nemo.collections.speechlm2.parts.multispeaker import build_speaker_tokens

        build_speaker_tokens(model_cfg.get("speaker_tokens"), tokenizer)
    return tokenizer
