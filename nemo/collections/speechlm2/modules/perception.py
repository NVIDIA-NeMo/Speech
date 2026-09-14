# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import contextlib
import inspect

import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    pack_encoder_output,
    unpack_encoder_output,
)
from nemo.core import Exportable, NeuralModule, typecheck


class AudioPerceptionModule(NeuralModule, Exportable):
    """Audio perception module that consists of audio encoder(s) and modality adapter."""

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token in the output
        of this module.
        """
        frame_shift = self.preprocessor.featurizer.hop_length / self.preprocessor.featurizer.sample_rate
        encoder_subsampling = self.encoder.subsampling_factor
        adapter_subsampling = getattr(self.modality_adapter, "subsampling_factor", 1.0)
        return frame_shift * encoder_subsampling * adapter_subsampling

    @property
    def encoder_frame_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame at the encoder output
        (but before the modality adapter).
        """
        frame_shift = self.preprocessor.featurizer.hop_length / self.preprocessor.featurizer.sample_rate
        return frame_shift * self.encoder.subsampling_factor

    @property
    def encoder(self) -> nn.Module:
        # When the modality adapter needs per-layer activations (Qformer / MultiLayer
        # projection), the encoder is wrapped inside ``encoder_multilayer`` so
        # ConformerMultiLayerFeatureExtractor can attach hooks. Expose it at the top
        # level here so downstream code (training_step freeze checks, etc.) sees a
        # single logical ``encoder`` submodule regardless of the adapter choice.
        # For the non-multilayer path, the encoder was registered via
        # ``self.encoder = encoder`` through nn.Module.__setattr__ (which bypasses
        # this property); look it up directly in _modules.
        if 'encoder_multilayer' in self._modules:
            return self._modules['encoder_multilayer'].encoder
        return self._modules['encoder']

    @property
    def supports_sequence_packed_output(self) -> bool:
        """Whether this exact encoder/adapter stack can preserve native THD activations."""
        return (
            'encoder_multilayer' not in self._modules
            and isinstance(self.modality_adapter, IdentityConnector)
            and self.rote is None
            and bool(getattr(self.encoder, 'supports_sequence_packed_output', False))
            and callable(getattr(self.encoder, 'forward_sequence_packed', None))
        )

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Initialize components
        self.cfg = cfg
        self.preprocessor = self.from_config_dict(cfg.preprocessor)
        encoder = self.from_config_dict(cfg.encoder)
        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if isinstance(self.modality_adapter, (QformerConnector, MultiLayerProjectionConnector)):
            from nemo.collections.asr.modules.conformer_encoder import ConformerMultiLayerFeatureExtractor

            self.encoder_multilayer = ConformerMultiLayerFeatureExtractor(
                encoder,
                layer_idx_list=cfg.modality_adapter.target_layer_ids,
                detach=False,
                convert_to_cpu=False,
                include_final_output=cfg.modality_adapter.get("include_final_output", True),
            )
        else:
            self.encoder = encoder
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()
        # Optional Rotary Time Embedding (ROTE), applied to the encoder output features
        # at the entrance of the modality adapter. ``None`` (default) is a no-op.
        self.rote = self.from_config_dict(cfg.rote) if cfg.get("rote") is not None else None

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Enable/disable activation checkpointing on the encoder's transformer layers.

        When ``enabled`` is True, wraps each layer in ``self.encoder.layers`` with
        ``torch.distributed.algorithms._checkpoint.checkpoint_wrapper``. Must be
        called before FSDP2 sharding. When ``enabled`` is False, this is a no-op.
        """
        _set_encoder_activation_checkpointing(self.encoder, enabled)

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        input_signal_cu_seqlens=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            if input_signal_cu_seqlens is None:
                processed_signal, processed_signal_length = self.preprocessor(
                    input_signal=input_signal,
                    length=input_signal_length,
                )
            else:
                processed_signal = self.preprocessor.forward_packed(
                    input_signal=input_signal,
                    length=input_signal_length,
                    input_signal_cu_seqlens=input_signal_cu_seqlens,
                )
                processed_signal_length = processed_signal.lengths
        return processed_signal, processed_signal_length

    def _apply_rote(self, encoder_emb, time_offset=None):
        """Apply RoTE to encoder output features ``(B, C, T)`` (or a list thereof).

        ``time_offset`` is an optional per-row start time in seconds, shape ``(B,)``; when
        ``None`` every row starts at time 0. Per-frame absolute time is
        ``time_offset[b] + (frame_idx + 0.5) * encoder_frame_duration``.
        """
        if isinstance(encoder_emb, (list, tuple)):
            return type(encoder_emb)(self._apply_rote(emb, time_offset) for emb in encoder_emb)

        # encoder_emb: (B, C, T) -> rotate over the channel dim with x channel-last.
        b, _, t = encoder_emb.shape
        device = encoder_emb.device
        frame_times = (torch.arange(t, device=device, dtype=torch.float32) + 0.5) * self.encoder_frame_duration
        times = frame_times.unsqueeze(0).expand(b, -1)  # (B, T)
        if time_offset is not None:
            times = times + time_offset.to(device=device, dtype=torch.float32).unsqueeze(1)
        rotated = self.rote(encoder_emb.transpose(1, 2), times)  # (B, T, C)
        return rotated.transpose(1, 2)

    @staticmethod
    def _encoder_accepts_spk_targets(encoder: nn.Module) -> bool:
        if hasattr(encoder, "diarization_model") or hasattr(encoder, "diar_kernel"):
            return True
        try:
            return "spk_targets" in inspect.signature(encoder.forward).parameters
        except (TypeError, ValueError):
            return False

    # disable type checks to avoid type-check errors when using Conformer as modality adapter
    @typecheck.disable_checks()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        return_encoder_emb=False,
        time_offset=None,
        spk_targets=None,
        input_signal_cu_seqlens=None,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal,
            input_signal_length,
            processed_signal,
            processed_signal_length,
            input_signal_cu_seqlens=input_signal_cu_seqlens,
        )
        if isinstance(processed_signal, PackedEncoderActivations):
            processed_signal = unpack_encoder_output(
                processed_signal, total_length=processed_signal.padded_length
            ).transpose(1, 2)

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        if isinstance(self.modality_adapter, (QformerConnector, MultiLayerProjectionConnector)):
            encoder_emb, encoded_len = self.encoder_multilayer(
                audio_signal=processed_signal, length=processed_signal_length
            )
        else:
            encoder_kwargs = {"audio_signal": processed_signal, "length": processed_signal_length}
            if spk_targets is not None:
                if not self._encoder_accepts_spk_targets(self.encoder):
                    raise ValueError(
                        "`spk_targets` were provided, but the mounted perception encoder "
                        f"({type(self.encoder).__name__}) does not support speaker-target inputs. "
                        "spk_targets has no effect when the encoder does not support it."
                    )
                encoder_kwargs["spk_targets"] = spk_targets
            encoder_emb, encoded_len = self.encoder(**encoder_kwargs)
        if self.rote is not None:
            encoder_emb = self._apply_rote(encoder_emb, time_offset)
        encoded, encoded_len = self.modality_adapter(audio_signal=encoder_emb, length=encoded_len)

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))
        result = (encoded, encoded_len)
        if return_encoder_emb:
            result += (encoder_emb.transpose(1, 2),)
        return result

    @typecheck.disable_checks()
    def forward_sequence_packed(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        time_offset=None,
        spk_targets=None,
        input_signal_cu_seqlens=None,
    ) -> PackedEncoderActivations:
        """Encode audio natively as token-major variable-length sequences.

        This opt-in API intentionally supports only an identity modality adapter,
        no multi-layer feature extraction, and no RoTE. Existing ``forward`` and
        checkpoint state dictionaries are unchanged.
        """
        if not self.supports_sequence_packed_output:
            raise ValueError(
                "Packed encoder sequences require an encoder with native packed support, "
                "IdentityConnector, no multi-layer feature extraction, and rote=null."
            )
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal,
            input_signal_length,
            processed_signal,
            processed_signal_length,
            input_signal_cu_seqlens=input_signal_cu_seqlens,
        )
        if self.spec_augmentation is not None and self.training:
            if isinstance(processed_signal, PackedEncoderActivations):
                processed_signal = self.spec_augmentation.forward_packed(processed_signal)
            else:
                processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_kwargs = {"audio_signal": processed_signal, "length": processed_signal_length}
        if spk_targets is not None:
            if not self._encoder_accepts_spk_targets(self.encoder):
                raise ValueError(
                    "`spk_targets` were provided, but the mounted perception encoder "
                    f"({type(self.encoder).__name__}) does not support speaker-target inputs."
                )
            encoder_kwargs["spk_targets"] = spk_targets
        encoded = self.encoder.forward_sequence_packed(**encoder_kwargs)
        return encoded.with_data(self.proj(encoded.data))


class IdentityConnector(nn.Module):
    """User to pass encoder's representations as-is to the LLM."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def forward(self, audio_signal, length=None, *args, **kwargs):
        return audio_signal, length


class IndependentDualEncoder(nn.Module):
    """Run two acoustic encoders independently and concatenate their states.

    Both branches see the same preprocessed features and must have the same input
    width and subsampling factor. Chunking is configured independently for each
    branch. It is applied *after* feature stacking, so chunk boundaries do not
    duplicate STFT padding and both outputs retain exactly the same frame grid.

    This is primarily intended for a trainable ASR encoder paired with a frozen
    auxiliary encoder. A frozen branch is kept in evaluation mode and runs under
    ``no_grad`` even when the parent perception module is training.
    """

    supports_sequence_packed_output = True

    def __init__(
        self,
        asr_encoder: nn.Module,
        auxiliary_encoder: nn.Module,
        *,
        frame_shift_seconds: float,
        asr_chunk_size_seconds: float | None = None,
        auxiliary_chunk_size_seconds: float | None = None,
        freeze_auxiliary: bool = True,
    ) -> None:
        super().__init__()
        self.asr_encoder = asr_encoder
        self.auxiliary_encoder = auxiliary_encoder
        self.frame_shift_seconds = float(frame_shift_seconds)
        self.asr_chunk_size_seconds = asr_chunk_size_seconds
        self.auxiliary_chunk_size_seconds = auxiliary_chunk_size_seconds
        self.freeze_auxiliary = bool(freeze_auxiliary)
        self.auxiliary_encoder_config = None

        if self.frame_shift_seconds <= 0:
            raise ValueError(f"frame_shift_seconds must be positive, got {frame_shift_seconds!r}.")
        for name, value in (
            ("asr_chunk_size_seconds", asr_chunk_size_seconds),
            ("auxiliary_chunk_size_seconds", auxiliary_chunk_size_seconds),
        ):
            if value is not None and float(value) <= 0:
                raise ValueError(f"{name} must be positive or null, got {value!r}.")

        self._feat_in = self._matching_positive_int("_feat_in")
        self.subsampling_factor = self._matching_positive_int("subsampling_factor")
        self.d_model = self._encoder_width(asr_encoder) + self._encoder_width(auxiliary_encoder)

        if not all(
            bool(getattr(encoder, "supports_sequence_packed_output", False))
            and callable(getattr(encoder, "forward_sequence_packed", None))
            for encoder in (asr_encoder, auxiliary_encoder)
        ):
            raise TypeError("IndependentDualEncoder requires two native sequence-packed encoders.")

        if self.freeze_auxiliary:
            self.auxiliary_encoder.requires_grad_(False)
            self.auxiliary_encoder.eval()

    def _matching_positive_int(self, attr: str) -> int:
        values = [int(getattr(encoder, attr, -1) or -1) for encoder in (self.asr_encoder, self.auxiliary_encoder)]
        if values[0] <= 0 or values[0] != values[1]:
            raise ValueError(f"Both encoders must have the same positive {attr}; got {values}.")
        return values[0]

    @staticmethod
    def _encoder_width(encoder: nn.Module) -> int:
        width = int(getattr(encoder, "_feat_out", getattr(encoder, "d_model", -1)) or -1)
        if width <= 0:
            raise ValueError(f"Could not determine output width for {type(encoder).__name__}.")
        return width

    def _chunk_size_tokens(self, chunk_size_seconds: float | None) -> int | None:
        if chunk_size_seconds is None:
            return None
        token_seconds = self.frame_shift_seconds * self.subsampling_factor
        return max(1, round(float(chunk_size_seconds) / token_seconds))

    @staticmethod
    def _chunk_metadata(packed: PackedEncoderActivations, max_tokens: int) -> PackedEncoderActivations:
        chunk_lengths = []
        for length in packed.lengths.detach().cpu().tolist():
            if length == 0:
                chunk_lengths.append(0)
                continue
            chunk_lengths.extend([max_tokens] * (length // max_tokens))
            if length % max_tokens:
                chunk_lengths.append(length % max_tokens)
        lengths = torch.as_tensor(chunk_lengths, dtype=torch.int64, device=packed.data.device)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=packed.data.device),
                lengths.cumsum(0, dtype=torch.int32),
            ]
        ).contiguous()
        return PackedEncoderActivations(
            data=packed.data,
            lengths=lengths,
            cu_seqlens=cu_seqlens,
            max_seqlen=min(max_tokens, packed.max_seqlen),
            padding_value=packed.padding_value,
            padded_length=None,
        )

    def _forward_branch(
        self,
        encoder: nn.Module,
        features: PackedEncoderActivations,
        chunk_size_seconds: float | None,
    ) -> PackedEncoderActivations:
        max_tokens = self._chunk_size_tokens(chunk_size_seconds)
        if max_tokens is None or features.max_seqlen <= max_tokens:
            return encoder.forward_sequence_packed(features, features.lengths)

        pre_encode = getattr(encoder, "pre_encode", None)
        wrapped = getattr(pre_encode, "_checkpoint_wrapped_module", pre_encode)
        if type(wrapped).__name__ != "FeatureStacking":
            raise TypeError(
                "Post-stacking chunking requires subsampling='feature_stacking'; "
                f"got {type(wrapped).__name__} for {type(encoder).__name__}."
            )
        pre_encoded = pre_encode(features)
        chunked = self._chunk_metadata(pre_encoded, max_tokens)
        encoded_chunks = encoder.forward_sequence_packed(
            chunked,
            chunked.lengths,
            bypass_pre_encode=True,
        )
        return pre_encoded.with_data(encoded_chunks.data)

    def forward_sequence_packed(self, audio_signal, length=None) -> PackedEncoderActivations:
        if not isinstance(audio_signal, PackedEncoderActivations):
            if length is None:
                raise ValueError("length is required for padded IndependentDualEncoder input.")
            audio_signal = pack_encoder_output(audio_signal.transpose(1, 2), length)
        elif length is not None and not torch.equal(length.to(audio_signal.lengths), audio_signal.lengths):
            raise ValueError("length must match audio_signal.lengths for packed input.")

        asr = self._forward_branch(self.asr_encoder, audio_signal, self.asr_chunk_size_seconds)
        grad_context = torch.no_grad() if self.freeze_auxiliary else contextlib.nullcontext()
        with grad_context:
            auxiliary = self._forward_branch(
                self.auxiliary_encoder,
                audio_signal,
                self.auxiliary_chunk_size_seconds,
            )
        if not torch.equal(asr.lengths, auxiliary.lengths):
            raise RuntimeError(
                "Independent encoder output lengths diverged despite matching subsampling factors: "
                f"ASR={asr.lengths.detach().cpu().tolist()}, "
                f"auxiliary={auxiliary.lengths.detach().cpu().tolist()}."
            )
        return asr.with_data(torch.cat([asr.data, auxiliary.data], dim=-1))

    def forward(self, audio_signal, length):
        packed = self.forward_sequence_packed(audio_signal, length)
        return unpack_encoder_output(packed).transpose(1, 2), packed.lengths

    def set_activation_checkpointing(self, enabled: bool) -> None:
        _set_encoder_activation_checkpointing(self.asr_encoder, enabled)
        if not self.freeze_auxiliary:
            _set_encoder_activation_checkpointing(self.auxiliary_encoder, enabled)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_auxiliary:
            self.auxiliary_encoder.eval()
        return self


def _set_encoder_activation_checkpointing(encoder: nn.Module, enabled: bool) -> None:
    """Wrap the encoder's subsampling front-end and each transformer layer with
    ``checkpoint_wrapper`` when enabled.

    Covers ``encoder.pre_encode`` (the Conformer fbank→subsampled-activation
    module: ``ConvSubsampling`` / ``StackingSubsampling`` / ``nn.Linear``) and
    each entry in ``encoder.layers``. Missing attributes are skipped so
    non-Conformer architectures degrade gracefully. No-op when ``enabled`` is
    False.

    Encoders whose execution bypasses ``encoder.layers[i](...)`` may implement
    ``set_activation_checkpointing`` and own the policy themselves.
    """
    if not enabled:
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    own_policy = getattr(encoder, "set_activation_checkpointing", None)
    if callable(own_policy):
        own_policy(enabled)
        return

    pre_encode = getattr(encoder, "pre_encode", None)
    # ConformerEncoder.forward dispatches on ``isinstance(pre_encode, nn.Linear)``
    # to choose between positional and (x=, lengths=) kwargs. Wrapping a Linear
    # hides its type and routes it to the wrong branch, so skip that case — the
    # memory win is negligible anyway (one linear vs. a conv/stacking stack).
    if pre_encode is not None and not isinstance(pre_encode, nn.Linear):
        encoder.pre_encode = checkpoint_wrapper(pre_encode)

    layers = getattr(encoder, "layers", None)
    if layers is not None:
        for i in range(len(layers)):
            layers[i] = checkpoint_wrapper(layers[i])


class AudioTranscriptionPerceptionModule(NeuralModule, Exportable):
    """Audio perception module that consists of audio encoder(s) and modality adapter."""

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token in the output
        of this module.
        """
        frame_shift = self.preprocessor.featurizer.hop_length / self.preprocessor.featurizer.sample_rate
        encoder_subsampling = self.encoder.subsampling_factor
        adapter_subsampling = getattr(self.modality_adapter, "subsampling_factor", 1.0)
        return frame_shift * encoder_subsampling * adapter_subsampling

    @property
    def encoder(self) -> nn.Module:
        return self.asr.encoder

    @property
    def preprocessor(self) -> nn.Module:
        return self.asr.preprocessor

    def __init__(self, cfg: DictConfig, pretrained_asr: str):
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.speechlm2.parts.model_loading import load_pretrained_nemo

        super().__init__()
        # Initialize components
        self.cfg = cfg
        self.asr = load_pretrained_nemo(ASRModel, pretrained_asr)
        with open_dict(self.cfg):
            self.cfg.asr = self.asr.cfg
        # self.asr = ASRModel.from_config_dict(cfg.asr)
        self.spec_augmentation = None
        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if isinstance(self.modality_adapter, (QformerConnector, MultiLayerProjectionConnector)):
            from nemo.collections.asr.modules.conformer_encoder import ConformerMultiLayerFeatureExtractor

            self.encoder_multilayer = ConformerMultiLayerFeatureExtractor(
                self.asr.encoder,
                layer_idx_list=cfg.modality_adapter.target_layer_ids,
                detach=False,
                convert_to_cpu=False,
                include_final_output=cfg.modality_adapter.get("include_final_output", True),
            )
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Enable/disable activation checkpointing on the encoder's transformer layers.

        When ``enabled`` is True, wraps each layer in ``self.encoder.layers`` with
        ``torch.distributed.algorithms._checkpoint.checkpoint_wrapper``. Must be
        called before FSDP2 sharding. When ``enabled`` is False, this is a no-op.
        """
        _set_encoder_activation_checkpointing(self.encoder, enabled)

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    def forward_encoder(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        if isinstance(self.modality_adapter, (QformerConnector, MultiLayerProjectionConnector)):
            encoded, encoded_len = self.encoder_multilayer(
                audio_signal=processed_signal, length=processed_signal_length
            )
        else:
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    def transcribe_encoded(self, encoded, encoded_len):
        from nemo.collections.asr.parts.mixins import TranscribeConfig

        if isinstance(encoded, list):
            encoded = encoded[-1]
            encoded_len = encoded_len[-1]
        return self.asr._transcribe_output_processing(
            outputs={"encoded": encoded, "encoded_len": encoded_len}, trcfg=TranscribeConfig()
        )

    # disable type checks to avoid type-check errors when using Conformer as modality adapter
    @typecheck.disable_checks()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        encoded=None,
        encoded_len=None,
    ):
        if encoded is None and encoded_len is None:
            encoded, encoded_len = self.forward_encoder(
                input_signal=input_signal,
                input_signal_length=input_signal_length,
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
            )
        encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        return encoded, encoded_len


class QformerConnector(nn.Module):
    def __init__(
        self,
        prompt_size: int,
        target_layer_ids: list[int],
        qformer_num_hidden_layers: int,
        encoder_config: DictConfig,
        llm_config: DictConfig,
        include_final_output: bool = True,
    ):
        super().__init__()
        self.prompt_size = prompt_size
        self.target_layer_ids = target_layer_ids
        self.include_final_output = include_final_output
        self.qformer_num_hidden_layers = qformer_num_hidden_layers
        self.encoder_config = encoder_config
        self.llm_config = llm_config

        num_inputs = len(self.target_layer_ids) + int(self.include_final_output)
        self.layer_prompts = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.prompt_size, self.encoder_config.d_model)) for _ in range(num_inputs)]
        )
        self.layer_weights = nn.Parameter(torch.zeros(self.prompt_size, num_inputs, dtype=torch.float))

        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = self.qformer_num_hidden_layers
        qformer_config.num_attention_heads = self.encoder_config.encoder_attention_heads
        qformer_config.hidden_size = self.encoder_config.d_model
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True
        if hasattr(qformer_config, "_attn_implementation"):  # fix for newer transformers versions
            qformer_config._attn_implementation = "eager"

        self.qformer = BertEncoder(qformer_config)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.encoder_config.d_model),
            nn.Linear(self.encoder_config.d_model, self.llm_config.hidden_size),  # project to llm hidden size
        )

    def forward(self, audio_signal: list[torch.Tensor], length):
        """
        input:
            audio_signal: layerwise hidden states from the encoder
        """
        layer_prompt_outputs = []
        expected_num = len(self.target_layer_ids) + int(self.include_final_output)
        assert (
            len(audio_signal) == expected_num
        ), f"Expected {expected_num} activations from encoder layers but got {len(audio_signal)}."
        for idx, encoder_hidden_state in enumerate(audio_signal):
            layer_prompt = self.layer_prompts[idx].expand(encoder_hidden_state.size(0), -1, -1)
            qformer_output = self.qformer(
                hidden_states=layer_prompt,
                encoder_hidden_states=encoder_hidden_state.transpose(1, 2),
            )
            layer_prompt_output = qformer_output.last_hidden_state
            layer_prompt_outputs.append(layer_prompt_output)

        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        output = (layer_prompt_outputs * norm_weights).sum(dim=2)  # (b, prompt_size, d_llm)
        output = self.proj(output)
        output = output.transpose(1, 2)

        return output, torch.tensor([output.shape[1]] * output.shape[0], device=output.device, dtype=torch.long)


class MultiLayerProjectionConnector(nn.Module):
    """User to pass encoder's representations as-is to the LLM."""

    def __init__(
        self,
        target_layer_ids: list[int],
        input_dim: int,
        output_dim: int,
        include_final_output: bool = True,
    ):
        super().__init__()
        self.target_layer_ids = target_layer_ids
        self.include_final_output = include_final_output
        self.input_dim = input_dim
        self.output_dim = output_dim
        num_inputs = len(self.target_layer_ids) + int(self.include_final_output)
        self.proj = torch.nn.Linear(self.input_dim * num_inputs, self.output_dim)

    def forward(self, audio_signal: list[torch.Tensor], length):
        expected_num = len(self.target_layer_ids) + int(self.include_final_output)
        assert (
            len(audio_signal) == expected_num
        ), f"Expected {expected_num} activations from encoder layers but got {len(audio_signal)}."
        audio_signal = torch.cat(audio_signal, dim=1).transpose(1, 2)
        projected = self.proj(audio_signal).transpose(1, 2)
        return projected, length[0]
