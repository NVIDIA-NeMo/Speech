# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Audio-side plumbing for the NeMo Speech LM (SALM) vLLM plugin.

All audio handling lives here: helpers (perception loader, tokenizer special-token
patcher, vocab-size padder), audio constants and TensorSchema, and the trio of
classes that bind to vLLM's multimodal registry to drive prompt expansion and
dummy-input generation. Backbone-agnostic; shared by both transformer and
hybrid backends.

Public surface used by the rest of the package:

* ``_AUDIO_PLACEHOLDER`` -- the audio locator string vLLM emits during prompt
  rendering and the processor expands inline.
* ``_load_nemo_perception``, ``_ensure_special_tokens``, ``_pad_to_vocab_size``
  -- small helpers reused at model init and weight load time.
* ``NeMoSpeechLMAudioInputs`` -- vLLM ``TensorSchema`` describing the parsed
  audio tensors that flow into ``embed_multimodal``.
* ``NeMoSpeechLMProcessingInfo`` / ``NeMoSpeechLMMultiModalProcessor`` /
  ``NeMoSpeechLMDummyInputsBuilder`` -- the trio that vLLM's multimodal
  registry binds to the registered model class.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature, PreTrainedTokenizerBase
from vllm.config.multimodal import BaseDummyOptions

try:
    from vllm.inputs import MultiModalDataDict
except ImportError:
    from vllm.multimodal.inputs import MultiModalDataDict

from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from nemo.collections.speechlm2.vllm.salm.config import _AUDIO_PLACEHOLDER
from nemo.utils import logging

_SAMPLING_RATE = 16000
_AUDIO_CHANNELS = 1
_DUMMY_AUDIO_DURATION_S = 40.0
_DUMMY_AUDIO_MAX_DURATION_S = 3600.0
_DUMMY_AUDIO_TEXT_TOKEN_RESERVE = 64
# FastConformer preprocessor hop length, used to derive the smallest
# chunk that produces ≥ 2 feature frames (per-feature normalization
# breaks on a single frame). Mirrors
# ``encoder_chunking._get_min_chunk_size_samples`` for the canonical
# preprocessor we ship; the chunking helper probes the live featurizer
# at training time, but the prompt processor here runs before the
# perception module is loaded, so we use the same constant the helper
# would derive.
_MIN_CHUNK_SIZE_SAMPLES = 320


# ── Helpers ─────────────────────────────────────────────────────────


def _ensure_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    # NOTE: called per request from _call_hf_processor on the API-server event loop.
    # Use O(1) dict membership; `set(get_vocab().keys())` rebuilt a 131k-entry set
    # every request (~5-6 ms) purely to check one token. get_vocab() returns vLLM's
    # cached dict, so membership is O(1).
    vocab = tokenizer.get_vocab()
    to_add = [t for t in (_AUDIO_PLACEHOLDER,) if t not in vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def _load_nemo_perception(perception_cfg: dict) -> nn.Module:
    try:
        from omegaconf import DictConfig

        from nemo.collections.speechlm2.modules import AudioPerceptionModule
    except ImportError as e:
        raise ImportError(
            "NeMo is required for the audio encoder. " "Install with: pip install 'nemo-toolkit[asr]'"
        ) from e

    cfg = DictConfig(perception_cfg)
    perception = AudioPerceptionModule(cfg)
    perception.eval()
    return perception


def _maybe_mount_pe_encoder(
    perception: nn.Module,
    pe_encoder_path: str | None,
    pe_encoder_config: dict | None = None,
    pe_encoder_overrides: dict | None = None,
) -> bool:
    """Mount a configured perception encoder from a local bundle or model identifier.

    Remote identifiers are resolved through the model cache; invalid local bundles
    fail before the encoder is replaced.
    """
    has_path = pe_encoder_path not in (None, "", False)
    has_config = pe_encoder_config not in (None, {}, "", False)
    if not has_path and not has_config:
        return False
    if has_path and has_config:
        raise ValueError("pe_encoder_path and pe_encoder_config are mutually exclusive.")
    if has_config and pe_encoder_overrides not in (None, {}):
        raise ValueError("pe_encoder_overrides may only be used with pe_encoder_path, not pe_encoder_config.")
    if not hasattr(perception, "encoder"):
        raise RuntimeError(
            "A ParallelExpertEncoder is configured but perception has no `encoder` attribute to replace."
        )

    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT

    if has_config:
        pe_encoder = ParallelExpertEncoderPT.from_inline_config(pe_encoder_config, map_location="cpu")
    else:
        pe_encoder = ParallelExpertEncoderPT.load_from_nemo(
            pe_encoder_path,
            map_location="cpu",
            strict=True,
            config_overrides=pe_encoder_overrides,
        )

    # The outgoing width is unconstrained; unchanged frontend and downstream
    # components must match the replacement encoder.
    existing_d_model = int(getattr(perception.encoder, "d_model", -1))
    if existing_d_model > 0 and int(pe_encoder.d_model) != existing_d_model:
        logging.info(
            "ParallelExpertEncoder d_model=%d replaces a perception encoder of d_model=%d; "
            "the outgoing encoder is discarded.",
            int(pe_encoder.d_model),
            existing_d_model,
        )

    perception_cfg = getattr(perception, "cfg", {})
    preprocessor_cfg = perception_cfg.get("preprocessor", {}) if hasattr(perception_cfg, "get") else {}
    pe_feat_in = int(getattr(pe_encoder, "_feat_in", -1) or -1)
    mel_bins = preprocessor_cfg.get("features", None) if hasattr(preprocessor_cfg, "get") else None
    if pe_feat_in > 0 and mel_bins is not None and int(mel_bins) != pe_feat_in:
        raise ValueError(
            f"ParallelExpertEncoder expects {pe_feat_in} mel bins but the vLLM perception "
            f"preprocessor produces {int(mel_bins)}. The preprocessor is not replaced by "
            "the mount, so these must agree."
        )

    adapter_cfg = perception_cfg.get("modality_adapter", {}) if hasattr(perception_cfg, "get") else {}
    adapter_d_model = adapter_cfg.get("d_model", None) if hasattr(adapter_cfg, "get") else None
    if adapter_d_model is not None and int(adapter_d_model) != int(pe_encoder.d_model):
        raise ValueError(
            f"ParallelExpertEncoder d_model={pe_encoder.d_model} does not match "
            f"vLLM perception modality_adapter.d_model={adapter_d_model}."
        )

    proj = getattr(perception, "proj", None)
    if isinstance(proj, torch.nn.Linear) and int(proj.in_features) != int(pe_encoder.d_model):
        raise ValueError(
            f"ParallelExpertEncoder d_model={pe_encoder.d_model} does not match "
            f"vLLM perception proj.in_features={proj.in_features}."
        )

    # load_from_nemo restores onto CPU; copy the replaced encoder's device/dtype to avoid CPU/dtype mismatches.
    ref_param = next(perception.encoder.parameters(), None)
    if ref_param is not None:
        pe_encoder = pe_encoder.to(device=ref_param.device, dtype=ref_param.dtype)

    # The replacement consumes un-normalised mels and applies ASR normalization internally.
    try:
        perception.preprocessor.featurizer.normalize = None
    except AttributeError:
        # Preprocessor/featurizer layout varies across backends; if the attribute is
        # absent there is no outer normalization to disable, so skipping is correct.
        pass

    perception.encoder = pe_encoder
    perception.eval()
    return True


def _maybe_mount_independent_speaker_encoder(
    perception: nn.Module,
    speaker_encoder_cfg: Mapping | None,
    encoder_chunk_size_seconds: float | None = None,
) -> bool:
    """Reconstruct an exported independent ASR + speaker encoder pair.

    Current dual-encoder exports retain the auxiliary architecture inline in the
    speaker_encoder field. Older exports may still refer to an external artifact.
    The checkpoint's own perception.encoder tensors
    subsequently replace both branches through vLLM's normal weight loader.
    """

    if speaker_encoder_cfg in (None, {}, "", False):
        return False
    if not isinstance(speaker_encoder_cfg, Mapping):
        raise TypeError(
            "speaker_encoder must be a mapping with encoder architecture and chunk settings; "
            f"got {type(speaker_encoder_cfg).__name__}."
        )
    if encoder_chunk_size_seconds is not None:
        raise ValueError(
            "Independent per-encoder chunking requires encoder_chunk_size_seconds=null; "
            "use speaker_encoder.asr_chunk_size_seconds and chunk_size_seconds."
        )
    if not hasattr(perception, "encoder"):
        raise RuntimeError("speaker_encoder is set but perception has no encoder to wrap.")

    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.modules.perception import IdentityConnector, IndependentDualEncoder

    encoder_config = speaker_encoder_cfg.get("encoder_config", None)
    artifact = None
    if encoder_config in (None, {}, "", False):
        artifact = Path(str(speaker_encoder_cfg.get("path", "")))
        config_path = artifact / "model_config.yaml"
        weights_path = artifact / "model.safetensors"
        if not artifact.is_dir() or not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "speaker_encoder must contain encoder_config, or path must contain "
                f"model_config.yaml and model.safetensors; got {artifact}."
            )
    if not isinstance(getattr(perception, "modality_adapter", None), IdentityConnector):
        raise TypeError("IndependentDualEncoder requires IdentityConnector.")
    if getattr(perception, "rote", None) is not None:
        raise ValueError("IndependentDualEncoder requires rote=null.")
    if "encoder_multilayer" in perception._modules:
        raise ValueError("IndependentDualEncoder does not support multi-layer perception adapters.")

    if encoder_config not in (None, {}, "", False):
        speaker_config = OmegaConf.create(encoder_config)
        speaker = perception.from_config_dict(speaker_config)
        source = "inline encoder_config"
    else:
        from safetensors.torch import load_file

        speaker_config = OmegaConf.load(config_path)
        speaker = perception.from_config_dict(speaker_config)
        state = load_file(str(weights_path), device="cpu")
        speaker.load_state_dict(state, strict=True)
        source = str(artifact)

    ref_param = next(perception.encoder.parameters(), None)
    if ref_param is not None:
        speaker = speaker.to(device=ref_param.device, dtype=ref_param.dtype)

    featurizer = perception.preprocessor.featurizer
    frame_shift_seconds = featurizer.hop_length / featurizer.sample_rate
    dual = IndependentDualEncoder(
        perception.encoder,
        speaker,
        frame_shift_seconds=frame_shift_seconds,
        asr_chunk_size_seconds=speaker_encoder_cfg.get("asr_chunk_size_seconds", None),
        auxiliary_chunk_size_seconds=speaker_encoder_cfg.get("chunk_size_seconds", None),
        freeze_auxiliary=speaker_encoder_cfg.get("frozen", True),
    )

    old_proj = getattr(perception, "proj", None)
    if not isinstance(old_proj, torch.nn.Linear):
        raise TypeError(
            "IndependentDualEncoder currently requires perception.proj to be nn.Linear; "
            f"got {type(old_proj).__name__}."
        )
    perception.encoder = dual
    perception.proj = torch.nn.Linear(
        dual.d_model,
        old_proj.out_features,
        bias=old_proj.bias is not None,
        device=old_proj.weight.device,
        dtype=old_proj.weight.dtype,
    )
    perception.eval()
    logging.info(
        "Mounted independent speaker encoder from %s beside ASR encoder "
        "(widths: ASR=%d speaker=%d combined=%d; chunks: ASR=%s speaker=%s seconds; frozen=%s).",
        source,
        IndependentDualEncoder._encoder_width(dual.asr_encoder),
        IndependentDualEncoder._encoder_width(dual.auxiliary_encoder),
        dual.d_model,
        dual.asr_chunk_size_seconds,
        dual.auxiliary_chunk_size_seconds,
        dual.freeze_auxiliary,
    )
    return True


def _pad_to_vocab_size(tensor: torch.Tensor, target_vocab: int) -> torch.Tensor:
    if tensor.shape[0] < target_vocab:
        pad = torch.zeros(
            target_vocab - tensor.shape[0],
            *tensor.shape[1:],
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


# ── Multimodal contract types ───────────────────────────────────────


class NeMoSpeechLMAudioInputs(TensorSchema):
    type: Literal["audio_features"] = "audio_features"
    audio_signal: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "t")]
    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]


class NeMoSpeechLMProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=_SAMPLING_RATE,
            target_channels=_AUDIO_CHANNELS,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def _has_pe_encoder(self) -> bool:
        return _config_has_pe_encoder(self.get_hf_config())

    def _get_encoder_chunk_size_seconds(self) -> float | None:
        """Return the per-encoder-call chunk size baked into the checkpoint.

        Standard perception encoders mirror the training-time
        ``model.encoder_chunk_size_seconds`` field. A mounted
        ParallelExpertEncoder instead owns its context-preserving online
        windowing and the model deliberately bypasses generic waveform
        chunking, so its prompt estimator must also use one full-audio pass.
        ``None`` means the encoder runs once over the full audio.
        """
        config = self.get_hf_config()
        if _config_has_pe_encoder(config):
            return None
        return getattr(config, "encoder_chunk_size_seconds", None)

    def _get_audio_token_estimator_config(self) -> Mapping[str, object] | None:
        """Return the exact training-time audio length arithmetic, when exported."""
        config = getattr(self.get_hf_config(), "audio_token_estimator", None)
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("audio_token_estimator in config.json must be a mapping")
        if config is not None and self._has_pe_encoder():
            # PEE owns context-preserving online windowing and deliberately
            # bypasses generic waveform chunking. Keep that decision
            # authoritative when the exported training estimator still carries
            # the generic chunk size.
            config = {**config, "chunk_size_seconds": None}
        return config

    @staticmethod
    def _estimate_audio_tokens_single_pass(
        audio_length_samples: int,
        estimator_config: Mapping[str, object] | None = None,
    ) -> int:
        """Predict one encoder forward output length using exported training arithmetic."""
        if estimator_config is None:
            preprocessor: Mapping[str, object] = {
                "n_fft": 512,
                "hop_length": 160,
                "stft_pad_amount": 256,
            }
            raw_subsampling: object = {
                "type": "conv",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "repeat": 3,
                "ceil_mode": False,
            }
        else:
            raw_preprocessor = estimator_config.get("preprocessor")
            if not isinstance(raw_preprocessor, Mapping):
                raise TypeError("audio_token_estimator.preprocessor must be a mapping")
            preprocessor = raw_preprocessor
            raw_subsampling = estimator_config.get("subsampling")

        stages = [raw_subsampling] if isinstance(raw_subsampling, Mapping) else raw_subsampling
        if not isinstance(stages, (list, tuple)):
            raise TypeError("audio_token_estimator.subsampling must be a mapping or list")

        n_fft = int(preprocessor["n_fft"])
        hop_length = int(preprocessor["hop_length"])
        stft_pad = int(preprocessor["stft_pad_amount"])
        length = (int(audio_length_samples) + 2 * stft_pad - n_fft) // hop_length
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise TypeError("Each audio_token_estimator.subsampling stage must be a mapping")
            stage_type = stage.get("type", "conv")
            if stage_type == "feature_stacking":
                factor = int(stage["factor"])
                length = (length + factor - 1) // factor
            elif stage_type == "conv":
                kernel = int(stage["kernel_size"])
                stride = int(stage["stride"])
                padding = int(stage["padding"])
                repeat = int(stage.get("repeat", 1))
                ceil_mode = bool(stage.get("ceil_mode", False))
                for _ in range(repeat):
                    numerator = length + 2 * padding - kernel
                    quotient = -(-numerator // stride) if ceil_mode else numerator // stride
                    length = quotient + 1
            else:
                raise ValueError(f"Unsupported audio_token_estimator subsampling type: {stage_type!r}")
        return max(1, length)

    @classmethod
    def _estimate_audio_tokens(
        cls,
        audio_length_samples: int,
        chunk_size_seconds: float | None = None,
        estimator_config: Mapping[str, object] | None = None,
    ) -> int:
        """Predict the encoder's total output frame count for an audio of N samples.

        When ``chunk_size_seconds`` is ``None`` or the audio fits in a single
        chunk, returns the single-pass estimate. Otherwise mirrors
        ``encode_audio_with_optional_chunking``'s split (with the same
        tail-folding rule) and sums the per-chunk frame counts so the
        placeholder count matches what the model emits at forward time.
        """
        if estimator_config is not None and "chunk_size_seconds" in estimator_config:
            configured_chunk_size = estimator_config.get("chunk_size_seconds")
            chunk_size_seconds = None if configured_chunk_size is None else float(configured_chunk_size)
        if chunk_size_seconds is None or audio_length_samples <= 0:
            return cls._estimate_audio_tokens_single_pass(audio_length_samples, estimator_config)
        if chunk_size_seconds <= 0.0:
            raise ValueError("encoder_chunk_size_seconds must be positive when set.")
        chunk_size_samples = max(1, int(round(chunk_size_seconds * _SAMPLING_RATE)))
        chunk_size_samples = max(chunk_size_samples, _MIN_CHUNK_SIZE_SAMPLES)
        if audio_length_samples <= chunk_size_samples:
            return cls._estimate_audio_tokens_single_pass(audio_length_samples, estimator_config)

        spans: list[tuple[int, int]] = []
        for begin in range(0, audio_length_samples, chunk_size_samples):
            end = min(begin + chunk_size_samples, audio_length_samples)
            spans.append((begin, end))
        if spans[-1][1] - spans[-1][0] < _MIN_CHUNK_SIZE_SAMPLES:
            spans[-2] = (spans[-2][0], spans[-1][1])
            spans.pop()

        return sum(cls._estimate_audio_tokens_single_pass(end - begin, estimator_config) for begin, end in spans)

    @classmethod
    def _samples_for_audio_tokens(
        cls,
        target_tokens: int,
        chunk_size_seconds: float | None = None,
        estimator_config: Mapping[str, object] | None = None,
    ) -> int:
        """Return the smallest sample count estimated to produce ``target_tokens``.

        vLLM sizes the multimodal encoder cache from dummy inputs.  The SALM
        plugin supports arbitrarily long audio by chunking the encoder forward,
        but the decoder still receives the concatenated full-audio embedding
        sequence.  This inverse estimator lets ``--limit-mm-per-prompt`` audio
        length hints reserve cache for that full sequence without hard-coding a
        single maximum call duration.
        """
        target_tokens = max(1, int(target_tokens))
        max_samples = int(_DUMMY_AUDIO_MAX_DURATION_S * _SAMPLING_RATE)
        lo, hi = 1, min(_SAMPLING_RATE, max_samples)
        while (
            hi < max_samples and cls._estimate_audio_tokens(hi, chunk_size_seconds, estimator_config) < target_tokens
        ):
            hi = min(hi * 2, max_samples)

        hi_tokens = cls._estimate_audio_tokens(hi, chunk_size_seconds, estimator_config)
        if hi_tokens < target_tokens:
            raise ValueError(
                f"Cannot produce {target_tokens} audio tokens within the "
                f"{_DUMMY_AUDIO_MAX_DURATION_S:g} s dummy-audio cap; "
                f"maximum is {hi_tokens}."
            )

        while lo < hi:
            mid = (lo + hi) // 2
            if cls._estimate_audio_tokens(mid, chunk_size_seconds, estimator_config) >= target_tokens:
                hi = mid
            else:
                lo = mid + 1
        return lo


class NeMoSpeechLMMultiModalProcessor(
    BaseMultiModalProcessor[NeMoSpeechLMProcessingInfo],
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_signal=MultiModalFieldConfig.batched("audio"),
            audio_signal_length=MultiModalFieldConfig.batched("audio"),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        audios = mm_items.get_items("audio", AudioProcessorItems)
        chunk_size_seconds = self.info._get_encoder_chunk_size_seconds()
        estimator_config = self.info._get_audio_token_estimator_config()

        def get_replacement(item_idx: int):
            audio = audios.get(item_idx)
            n_tokens = self.info._estimate_audio_tokens(audio.shape[-1], chunk_size_seconds, estimator_config)
            repl_full = _AUDIO_PLACEHOLDER * n_tokens
            return PromptUpdateDetails.select_text(repl_full, _AUDIO_PLACEHOLDER)

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER,
                replacement=get_replacement,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        _ensure_special_tokens(tokenizer)
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        if audios:
            chunk_size_seconds = self.info._get_encoder_chunk_size_seconds()
            estimator_config = self.info._get_audio_token_estimator_config()
            audio_list: list[torch.Tensor] = []
            audio_lengths: list[int] = []
            parts = re.split(f"({re.escape(_AUDIO_PLACEHOLDER)})", prompt)
            # One placeholder is overwritten with one audio's encoder output
            # at forward time (positional pairing); counts must match or the
            # merge step in get_input_embeddings crashes / silently drops.
            ph_positions = [i for i, p in enumerate(parts) if p == _AUDIO_PLACEHOLDER]
            if len(ph_positions) != len(audios):
                raise ValueError(
                    f"Prompt has {len(ph_positions)} "
                    f"{_AUDIO_PLACEHOLDER!r} placeholders but "
                    f"{len(audios)} audios were provided; counts must match."
                )
            for i, audio in zip(ph_positions, audios):
                audio_tensor = (
                    audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio, dtype=torch.float32)
                )
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.squeeze()
                n_tokens = self.info._estimate_audio_tokens(
                    audio_tensor.shape[-1], chunk_size_seconds, estimator_config
                )
                parts[i] = _AUDIO_PLACEHOLDER * n_tokens
                audio_list.append(audio_tensor)
                audio_lengths.append(audio_tensor.shape[-1])

            prompt = "".join(parts)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        if audios:
            result["audio_signal"] = audio_list
            result["audio_signal_length"] = torch.tensor(audio_lengths)
        return result


class NeMoSpeechLMDummyInputsBuilder(
    BaseDummyInputsBuilder[NeMoSpeechLMProcessingInfo],
):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        dummy_audio_len = int(_DUMMY_AUDIO_DURATION_S * _SAMPLING_RATE)
        audio_options = mm_options.get("audio") if mm_options else None
        requested_audio_len = getattr(audio_options, "length", None)
        if requested_audio_len:
            chunk_size_seconds = self.info._get_encoder_chunk_size_seconds()
            estimator_config = self.info._get_audio_token_estimator_config()
            if seq_len > _DUMMY_AUDIO_TEXT_TOKEN_RESERVE:
                max_audio_tokens = seq_len - _DUMMY_AUDIO_TEXT_TOKEN_RESERVE
                max_audio_len = int(_DUMMY_AUDIO_MAX_DURATION_S * _SAMPLING_RATE)
                max_supported_audio_tokens = NeMoSpeechLMProcessingInfo._estimate_audio_tokens(
                    max_audio_len,
                    chunk_size_seconds,
                    estimator_config,
                )
                if max_audio_tokens < max_supported_audio_tokens:
                    max_audio_len = NeMoSpeechLMProcessingInfo._samples_for_audio_tokens(
                        max_audio_tokens,
                        chunk_size_seconds,
                        estimator_config,
                    )
            else:
                max_audio_len = int(_DUMMY_AUDIO_MAX_DURATION_S * _SAMPLING_RATE)
            dummy_audio_len = min(int(requested_audio_len), max_audio_len)
        return {
            "audio": self._get_dummy_audios(
                length=dummy_audio_len,
                num_audios=num_audios,
            )
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "Transcribe the following: " + _AUDIO_PLACEHOLDER * num_audios


def _config_has_pe_encoder(config) -> bool:
    return getattr(config, "pe_encoder_path", None) not in (None, "", False) or getattr(
        config, "pe_encoder_config", None
    ) not in (None, {}, "", False)
