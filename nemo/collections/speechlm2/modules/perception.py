# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules.conformer_encoder import ConformerMultiLayerFeatureExtractor
from nemo.collections.asr.parts.mixins import TranscribeConfig
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

    # disable type checks to avoid type-check errors when using Conformer as modality adapter
    @typecheck.disable_checks()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        return_encoder_emb=False,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        if isinstance(self.modality_adapter, (QformerConnector, MultiLayerProjectionConnector)):
            encoder_emb, encoded_len = self.encoder_multilayer(
                audio_signal=processed_signal, length=processed_signal_length
            )
        else:
            encoder_emb, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.modality_adapter(audio_signal=encoder_emb, length=encoded_len)

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))
        if return_encoder_emb:
            return encoded, encoded_len, encoder_emb.transpose(1, 2)
        else:
            return encoded, encoded_len


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


def _set_encoder_activation_checkpointing(encoder: nn.Module, enabled: bool) -> None:
    """Wrap the encoder's subsampling front-end and each transformer layer with
    ``checkpoint_wrapper`` when enabled.

    Covers ``encoder.pre_encode`` (the Conformer fbank→subsampled-activation
    module: ``ConvSubsampling`` / ``StackingSubsampling`` / ``nn.Linear``) and
    each entry in ``encoder.layers``. Missing attributes are skipped so
    non-Conformer architectures degrade gracefully. No-op when ``enabled`` is
    False.
    """
    if not enabled:
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

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
        from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo

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
