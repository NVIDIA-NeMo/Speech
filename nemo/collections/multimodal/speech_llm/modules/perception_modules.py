# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder, ConformerMultiLayerFeatureExtractor
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import align_feat_seq_list
from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils.decorators import experimental

__all__ = ["AudioPerceptionModule", "MultiAudioPerceptionModule"]


class AudioPerceptionModule(NeuralModule, Exportable):
    """Audio perception module that consists of audio encoder(s) and modality adapter."""

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        batch_size = torch.randint(low=1, high=max_batch, size=[1]).item()
        max_length = torch.randint(low=min_length, high=max_dim, size=[1]).item()
        signals = torch.rand(size=[batch_size, max_length]) * 2 - 1
        lengths = torch.randint(low=min_length, high=max_dim, size=[batch_size])
        lengths[0] = max_length
        return signals, lengths, None, None

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "input_signal": NeuralType(("B", "T"), AudioSignal(freq=self.preprocessor._sample_rate)),
                "input_signal_length": NeuralType(
                    tuple("B"), LengthsType()
                ),  # Please note that length should be in samples not seconds.
                "processed_signal": NeuralType(("B", "D", "T"), SpectrogramType()),
                "processed_signal_length": NeuralType(tuple("B"), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "encoded": NeuralType(("B", "T", "D"), AcousticEncodedRepresentation()),
                "encoded_len": NeuralType(tuple("B"), LengthsType()),
            }
        )

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Initialize components
        self.cfg = cfg
        self.preprocessor = self.from_config_dict(cfg.preprocessor)
        self.encoder = self.from_config_dict(cfg.encoder)

        if cfg.get("use_multi_layer_feat", False) and cfg.get("multi_layer_feat", None):
            if "_target_" in cfg.multi_layer_feat.aggregator:
                aggregator = self.from_config_dict(cfg.multi_layer_feat.aggregator)
            else:
                aggregator = MultiFeatureAggregator(cfg.multi_layer_feat.aggregator, channel_dim=1)
            self.encoder = ConformerMultiLayerFeatureExtractor(
                encoder=self.encoder, layer_idx_list=cfg.multi_layer_feat.layer_idx_list, aggregator=aggregator
            )

        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()

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

    # disable type checks to avoid type-check errors when using Conformer as modality adapter
    @typecheck.disable_checks()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        return encoded, encoded_len


class MultiFeatureAggregator(nn.Module):
    """
    A module used to aggregate multiple encoded features (from different encoders or different layers) into a single feature sequence.
    """

    def __init__(self, cfg: DictConfig, channel_dim: int = 1):
        super().__init__()
        self.mode = cfg.get("mode", "cat")
        self.channel_dim = channel_dim
        self.pooling = cfg.get("pooling", "mean")
        self.align_mode = cfg.get("align_mode", "min")

    def _have_same_length(self, encoded_len: List[torch.Tensor]) -> bool:
        sample_len = encoded_len[0]
        for x in encoded_len:
            if torch.sum(x - sample_len) != 0:
                return False
        return True

    def forward(
        self,
        encoded: List[torch.Tensor],
        encoded_len: List[torch.Tensor],
        ref_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._have_same_length(encoded_len):
            """Align the length of encoded features if they are different."""
            target_len = encoded[0].size(self.channel_dim)
            if ref_idx is not None:
                target_len = encoded[ref_idx].size(self.channel_dim)
            if self.channel_dim != 1:
                encoded = [x.transpose(1, self.channel_dim) for x in encoded]
            encoded, encoded_len = align_feat_seq_list(
                encoded, encoded_len, mode=self.align_mode, pooling=self.pooling, target_len=target_len
            )
            if self.channel_dim != 1:
                encoded = [x.transpose(1, self.channel_dim) for x in encoded]

        if self.mode == "cat":
            return torch.cat(encoded, dim=self.channel_dim), encoded_len[0]
        elif self.mode == "sum":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).sum(dim=-1), encoded_len[0]
        elif self.mode == "mean" or self.mode == "avg":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).mean(dim=-1), encoded_len[0]
        elif self.mode == "max":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).max(dim=-1), encoded_len[0]
        elif self.mode == "min":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).min(dim=-1), encoded_len[0]
        elif self.mode == "none":
            return encoded, encoded_len
        else:
            raise ValueError(f"Unknown mode {self.mode}")


@experimental
class MultiAudioPerceptionModule(NeuralModule, Exportable):
    """
    Audio perception module that consists of multiple audio encoders and shared modality adapter.
    This module is experimental. An example perception cfg is:
    -------------------
    perception:
        modality_adapter:
            _target_: nemo.collections.multimodal.speechllm.modules.PoolingMLPConnectors
            hidden_dim: 512
            pooling: 'cat'
            pooling_factor: 2
            num_layers: 4
            input_dim: -1
            output_dim: -1

        spec_augment:
            _target_: nemo.collections.asr.modules.SpectrogramAugmentation
            freq_masks: 2 # set to zero to disable it
            time_masks: 10 # set to zero to disable it
            freq_width: 27
            time_width: 0.05

        encoders:
            asr_model:
                _target_: nemo.collections.asr.models.ASRModel
                output_key: d_model
                freeze: True
                pretrained_model: stt_en_fastconformer_transducer_large
            ssl_model:
                _target_: nemo.collections.asr.models.SpeechEncDecSelfSupervisedModel
                output_key: d_model
                freeze: True
                pretrained_model: ssl_en_conformer_large
                use_multi_layer_feat: True
                multi_layer_feat:
                layer_idx_list: [0,16]
                aggregator:
                    mode: "cat"
                    pooling: "avg"
                    rounding: "floor"

            speaker_model:
                segment_length_in_secs: 0.4
                freeze: True
                pretrained_model: titanet_large

            ref_model: asr_model
            aggregator:
                mode: "cat"
                pooling: "mean"
                rounding: "floor"
    -------------------
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Initialize components
        self.aggregator = MultiFeatureAggregator(cfg.aggregator, channel_dim=1)
        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder_cfg = cfg.encoders
        if not isinstance(self.encoder_cfg, DictConfig):
            raise TypeError(f"cfg.encoders must be a DictConfig, got {type(cfg.encoders)}")

        preprocessor = {}
        encoders = {}
        for key, enc_cfg in self.encoder_cfg.items():
            encoder = self.from_config_dict(enc_cfg.model)
            if enc_cfg.get("use_multi_layer_feat", False) and enc_cfg.get("multi_layer_feat", None):
                if not isinstance(encoder, ConformerEncoder):
                    raise TypeError(
                        f"Encoder {key} must be a ConformerEncoder when use_multi_layer_feat is True, got {type(encoder)}"
                    )
                if "_target_" in enc_cfg.multi_layer_feat.aggregator:
                    aggregator = self.from_config_dict(enc_cfg.multi_layer_feat.aggregator)
                else:
                    aggregator = MultiFeatureAggregator(enc_cfg.multi_layer_feat.aggregator, channel_dim=1)
                encoder = ConformerMultiLayerFeatureExtractor(
                    encoder=encoder, layer_idx_list=enc_cfg.multi_layer_feat.layer_idx_list, aggregator=aggregator
                )
            encoders[key] = encoder
            preprocessor[key] = (
                self.from_config_dict(enc_cfg.get("preprocessor"))
                if enc_cfg.get("preprocessor", None) is not None
                else None
            )
        self.encoders = nn.ModuleDict(encoders)
        self.preprocessor = nn.ModuleDict(preprocessor)

        self.speaker_model = None
        self.speaker_seg_len = None
        if "speaker_model" in cfg and cfg.speaker_model.get("model", None) is not None:
            self.speaker_model = EncDecSpeakerLabelModel(cfg=cfg.speaker_model.model)
            self.speaker_model.spec_augmentation = self.spec_augmentation
            self.speaker_seg_len = 1
            if "preprocessor" in cfg.speaker_model.model:
                self.speaker_seg_len = int(
                    cfg.speaker_model.segment_length_in_secs // cfg.speaker_model.model.preprocessor.window_stride
                )
        self.ref_model = cfg.get("ref_model", None)
        if self.ref_model is not None:
            if self.ref_model not in self.encoders and (
                self.ref_model != "speaker_model" and self.speaker_model is not None
            ):
                if self.ref_model == "speaker_model":
                    raise ValueError(f"ref_model is `{self.ref_model}` but speaker_model is None")
                raise ValueError(f"ref_model `{self.ref_model}` not found in encoders [{encoders.keys()}]")

        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()

    def maybe_preprocess_audio(
        self,
        preprocessor,
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

        if not has_processed_signal and preprocessor is not None:
            processed_signal, processed_signal_length = preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        elif not has_processed_signal and preprocessor is None:
            processed_signal, processed_signal_length = input_signal, input_signal_length
        return processed_signal, processed_signal_length

    def forward_speaker(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.speaker_model.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        # encoded has shape [B, D, T], length has shape [B]
        encoded, encoded_len = self.speaker_model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        # pad encoded to be divisible by speaker_seg_len
        if encoded.shape[2] % self.speaker_seg_len != 0:
            encoded = torch.cat(
                [
                    encoded,
                    torch.zeros(
                        encoded.shape[0],
                        encoded.shape[1],
                        self.speaker_seg_len - encoded.shape[2] % self.speaker_seg_len,
                        device=encoded.device,
                    ),
                ],
                dim=2,
            )

        B, D, T = encoded.shape
        num_seg = int(T // self.speaker_seg_len)
        encoded = encoded.view(int(B * num_seg), D, self.speaker_seg_len)  # [B*num_seg, D, seg_len]
        encoded_len_seg = (encoded_len // self.speaker_seg_len).repeat_interleave(num_seg)  # [B*seg_len]

        _, embeds = self.speaker_model.decoder(encoder_output=encoded, length=encoded_len_seg)

        embeds = embeds.view(B, -1, num_seg)  # [B, D, num_seg]

        embeds_len = encoded_len // self.speaker_seg_len  # [B]
        return embeds, embeds_len

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        encoded_list = []
        encoded_len_list = []
        ref_idx = None
        for key, encoder in self.encoders.items():
            curr_processed_signal, curr_processed_signal_length = self.maybe_preprocess_audio(
                self.preprocessor[key], input_signal, input_signal_length, processed_signal, processed_signal_length
            )
            # Spec augment is not applied during evaluation/testing
            if self.spec_augmentation is not None and self.training:
                processed_signal = self.spec_augmentation(
                    input_spec=curr_processed_signal, length=curr_processed_signal_length
                )
            encoded, encoded_len = encoder(audio_signal=curr_processed_signal, length=curr_processed_signal_length)
            if key == self.ref_model:
                ref_idx = len(encoded_list)
            encoded_list.append(encoded)
            encoded_len_list.append(encoded_len)

        if self.speaker_model is not None:
            speaker_embeds, speaker_embeds_len = self.forward_speaker(
                input_signal=input_signal,
                input_signal_length=input_signal_length,
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
            )
            encoded_list.append(speaker_embeds)
            encoded_len_list.append(speaker_embeds_len)
        encoded_list, encoded_len_list = self.aggregator(
            encoded=encoded_list, encoded_len=encoded_len_list, ref_idx=ref_idx
        )
        encoded, encoded_len = self.modality_adapter(audio_signal=encoded_list, length=encoded_len_list)
        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))
        return encoded, encoded_len


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class TransformerCrossAttention(NeuralModule, Exportable):
    """Transformer module for cross-attention between speech and text embeddings.
    The module allows optional projection from the input embeddings to a lower dimension before feeding them to the transformer.
    Args:
        cfg: DictConfig, configuration object for the module which should include:
            xattn: DictConfig, configuration object for the transformer decoder
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        xformer_num_layers = cfg.xattn.get('xformer_num_layers', 2)
        xformer_dims = cfg.xattn.get('xformer_dims', cfg.output_dim)
        self.cfg = cfg
        cross_attn_cfg = cfg.xattn
        if xformer_dims != cfg.output_dim:
            self.input_proj1 = nn.Linear(cfg.output_dim, xformer_dims)
            self.input_proj2 = nn.Linear(cfg.output_dim, xformer_dims)
            self.output_proj = nn.Linear(xformer_dims, cfg.output_dim)
        else:
            self.input_proj1 = nn.Identity()
            self.input_proj2 = nn.Identity()
            self.output_proj = nn.Identity()
        # causal attention decoder by default
        self.xattn_decoder = TransformerDecoder(
            hidden_size=xformer_dims,
            num_layers=xformer_num_layers,
            inner_size=1 * xformer_dims,
            num_attention_heads=cross_attn_cfg.num_attention_heads,
            ffn_dropout=cross_attn_cfg.ffn_dropout,
            attn_score_dropout=cross_attn_cfg.attn_score_dropout,
            attn_layer_dropout=cross_attn_cfg.attn_layer_dropout,
            hidden_act=cross_attn_cfg.hidden_act,
            pre_ln=cross_attn_cfg.pre_ln,
            pre_ln_final_layer_norm=cross_attn_cfg.pre_ln_final_layer_norm,
        )

    def forward(
        self,
        encoder_states,
        encoded_len,
        input_embeds,
        input_lengths,
        decoder_mems_list=None,
        return_mems=False,
    ):
        assert input_embeds.shape[-1] == encoder_states.shape[-1]
        enc_mask = lens_to_mask(encoded_len, encoder_states.shape[1]).to(encoder_states.dtype)
        dec_mask = lens_to_mask(input_lengths, input_embeds.shape[1]).to(input_lengths.dtype)
        y = self.xattn_decoder(
            decoder_states=self.input_proj1(input_embeds),
            decoder_mask=dec_mask,
            encoder_states=self.input_proj2(encoder_states),
            encoder_mask=enc_mask,
            decoder_mems_list=decoder_mems_list,
            return_mems=return_mems,
            return_mems_as_list=False,
        )
        if return_mems:
            extra_outpus = {'decoder_mems_list': y}
            y = y[-1][:, -input_embeds.shape[1] :]
        else:
            extra_outpus = {}
        y = self.output_proj(y) + input_embeds
        assert y.shape == input_embeds.shape
        return y, extra_outpus
