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


import json
import os
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
from megatron.core.inference_params import InferenceParams
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_num_microbatches,
    reconfigure_num_microbatches_calculator,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model.base import GPTConfig, get_packed_seq_params
from nemo.collections.speechlm.data.dataset.data_utils import build_position_ids, pad_or_trim_to_max_length
from nemo.collections.speechlm.models.base import SpeechLanguageModel
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig, HFWrappedEncoder
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils.io import get_nested_attr, import_ckpt, load_distributed_ckpt
from nemo.collections.speechlm.utils.text_generation.audio_text_generation_strategy import (
    SpeechToTextGenerationStrategy,
)
from nemo.collections.speechlm.utils.text_generation.audio_text_generation_utils import (
    clean_end_string,
    default_inference_config,
    generate,
    get_computeprob_response,
)
from nemo.lightning import io
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import AppState, logging, model_utils
from nemo.utils.get_rank import get_last_rank


def set_input_tensor(self, tensor: torch.Tensor):
    """
    Placeholder function for pipeline parallel, not implemented yet.
    """
    pass


def speech_to_text_llm_data_step(dataloader_iter) -> Dict[str, Any]:
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    # used in SpeechToTextLLMConfig
    batch = next(dataloader_iter)
    _batch: dict
    batch_idx, dataloader_idx = None, None

    if isinstance(batch, tuple) and len(batch) == 3:
        _batch, batch_idx, dataloader_idx = batch
    else:
        _batch = batch

    required_keys = set(
        [
            "sample_ids",
            "attention_mask",
            "position_ids",
            "metadata",
            "inference_params",
            "max_length",
        ]
    )
    # "context", "context_length", "answers", "max_length",
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(
            (
                "audio_signal",
                "audio_signal_length",
                "processed_signal",
                "processed_signal_length",
                "tokens",
                "tokens_length",
                "context_start_idx",
                "num_audios",
                "answers",
                "contexts",
                "context_lengths",
            )
        )
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {
        key: move_data_to_device(val, "cuda", non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }

    # inject num_valid_tokens_in_ub for context parallelism,
    # which refers to the total number of valid tokens in the current batch
    if parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in _batch and _batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = _batch["loss_mask"].sum()
        _batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    _batch["dataloader_idx"] = dataloader_idx
    _batch["batch_idx"] = batch_idx
    return _batch


def speech_to_text_llm_forward_step(model: pl.LightningModule, batch: Dict[str, Any]) -> torch.Tensor:
    forward_args = {
        "input_ids": batch.get("tokens", None),
        "input_length": batch.get("tokens_length", None),
        "loss_mask": batch.get("loss_mask", None),
        "attention_mask": batch.get("attention_mask", None),
        "audio_signal": batch.get("audio_signal", None),
        "audio_signal_length": batch.get("audio_signal_length", None),
        "processed_signal": batch.get("processed_signal", None),
        "processed_signal_length": batch.get("processed_signal_length", None),
        "labels": batch.get("labels", None),
        "num_audios": batch.get("num_audios", None),
        "context_start_idx": batch.get("context_start_idx", None),
        "inference_params": batch.get("inference_params", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


@dataclass
class SpeechToTextLLMConfig(TransformerConfig, io.IOMixin):

    num_layers: int = 1  # added to avoid init error, not used!!!
    hidden_size: int = 1  # added to avoid init error, not used!!!
    num_attention_heads: int = 16  # added to avoid init error, not used!!!
    seq_length: int = 1024  # added to avoid init error, not used!!!

    language_model_config: Optional[GPTConfig] = None
    language_model_class: Optional[str] = None
    speech_model_config: Optional[ASRModuleConfig] = None
    modality_adapter_config: Optional[ModalityAdapterConfig] = None

    language_model_from_pretrained: Optional[str] = None
    language_model_hub: Optional[str] = 'hf://'

    freeze_language_model: bool = True
    freeze_speech_model: bool = True
    freeze_modality_adapter: bool = False

    forward_step_fn: Callable = speech_to_text_llm_forward_step
    data_step_fn: Callable = speech_to_text_llm_data_step

    text_generation_strategy: SpeechToTextGenerationStrategy = SpeechToTextGenerationStrategy

    inference_config: Optional[Dict[str, Any]] = None

    data_config: Optional[DictConfig] = None

    resume_speech_model_from_path: Optional[str] = None
    resume_modality_adapter_from_path: Optional[str] = None

    def _freeze_module(self, module: Optional[nn.Module] = None) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _maybe_load_pretrained_llm(self, model: MCoreGPTModel, strict: bool = False) -> MCoreGPTModel:
        if not self.language_model_from_pretrained:
            return model

        logging.info(f"Loading language model weights from {self.language_model_from_pretrained}")

        ckpt_path = self.language_model_from_pretrained

        if dist_checkpointing.check_is_distributed_checkpoint(ckpt_path):
            return ckpt_path

        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed environment is not initialized.")

        rank = torch.distributed.get_rank()
        # Sleep to avoid racing condition when multiple GPUs try to import the same checkpoint
        time.sleep(rank / 2)

        llm_model_cls = model_utils.import_class_by_path(self.language_model_class)  # type: GPTModel
        ckpt_path = import_ckpt(
            llm_model_cls(self.language_model_config), f"{self.language_model_hub}{ckpt_path}", on_import_ckpt=False
        )

        load_path = ckpt_to_weights_subdir(ckpt_path, is_saving=False)
        sharded_sd_metadata = dist_checkpointing.load_content_metadata(load_path)
        if sharded_sd_metadata is None:
            sharded_sd_metadata = {}  # backward-compatibility
        sharded_state_dict = dict(state_dict=model.sharded_state_dict(prefix="module.", metadata=sharded_sd_metadata))

        loaded_state_dict = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=load_path,
            validate_access_integrity=False,
            **({"strict": "log_all"} if not strict else {}),
        )
        loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        model.load_state_dict(loaded_state_dict)
        logging.info(f"Restored language model weights from {self.language_model_from_pretrained}")
        return model

    def _maybe_load_asr_and_modality_adapter(
        self, asr_model: ASRModel, modality_adapter: nn.Module
    ) -> Tuple[ASRModel, nn.Module]:
        if self.resume_speech_model_from_path:
            logging.info(f"Loading speech model weights from {self.resume_from_path}")
            state_dict, _ = load_distributed_ckpt(self.resume_from_path)
            prefix = 'module.speech_model.'
            speech_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            asr_model.load_state_dict(speech_state_dict, strict=True)
            logging.info(f"Restored speech model weights from {self.resume_from_path}")

        if self.resume_modality_adapter_from_path:
            logging.info(f"Loading modality adapter weights from {self.resume_modality_adapter_from_path}")
            state_dict, _ = load_distributed_ckpt(self.resume_modality_adapter_from_path)
            prefix = 'module.modality_adapter.'
            modality_adapter_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            modality_adapter.load_state_dict(modality_adapter_state_dict, strict=True)
            logging.info(f"Restored modality adapter weights from {self.resume_modality_adapter_from_path}")

        return asr_model, modality_adapter

    def _propagate_model_configs(self) -> TransformerConfig:
        """
        propagate key attributes to the language/speech model config
        """
        # LLM
        self.language_model_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_model_config.sequence_parallel = self.sequence_parallel
        self.language_model_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_model_config.context_parallel_size = self.context_parallel_size

        # ASR
        self.speech_model_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.speech_model_config.sequence_parallel = self.sequence_parallel
        self.speech_model_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.speech_model_config.context_parallel_size = self.context_parallel_size

        # modality adapter
        self.modality_adapter_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.modality_adapter_config.sequence_parallel = self.sequence_parallel
        self.modality_adapter_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.modality_adapter_config.context_parallel_size = self.context_parallel_size

    def configure_model(
        self, tokenizer: TokenizerSpec, speech_model: Optional[ASRModel] = None
    ) -> "MCoreSpeechToTextLLM":
        self._propagate_model_configs()
        language_model = self.language_model_config.configure_model(tokenizer=tokenizer)  # type: "MCoreGPTModel"
        language_model = self._maybe_load_pretrained_llm(language_model)

        if speech_model is None:
            # propagate key attributes to the speech model config
            speech_model = self.speech_model_config.configure_model()  # type: MCoreASRModel
        speech_model.set_input_tensor = MethodType(set_input_tensor, speech_model)

        self.modality_adapter_config.output_dim = self.language_model_config.hidden_size
        input_key = self.modality_adapter_config.input_key_from
        if input_key:
            if hasattr(self.speech_model_config.config, input_key):
                self.modality_adapter_config.input_dim = getattr(self.speech_model_config.config, input_key)
            else:
                if isinstance(speech_model.encoder, HFWrappedEncoder):
                    self.modality_adapter_config.input_dim = get_nested_attr(speech_model.encoder.encoder, input_key)
                else:
                    self.modality_adapter_config.input_dim = get_nested_attr(speech_model.encoder, input_key)

        modality_adapter = self.modality_adapter_config.configure_model()
        modality_adapter.set_input_tensor = MethodType(set_input_tensor, modality_adapter)

        speech_model, modality_adapter = self._maybe_load_asr_and_modality_adapter(speech_model, modality_adapter)
        model = MCoreSpeechToTextLLM(
            config=self,
            language_model=language_model,
            speech_model=speech_model,
            modality_adapter=modality_adapter,
            tokenizer=tokenizer,
        )

        if self.freeze_language_model:
            self._freeze_module(model.language_model)
        if self.freeze_speech_model:
            self._freeze_module(model.speech_model)
        if self.freeze_modality_adapter:
            self._freeze_module(model.modality_adapter)

        return model


class MCoreSpeechToTextLLM(MegatronModule, fn.FNMixin):
    def __init__(
        self,
        config: SpeechToTextLLMConfig,
        language_model: MegatronModule,
        speech_model: ASRModel,
        modality_adapter: nn.Module,
        tokenizer: TokenizerSpec,
    ):
        super().__init__(config=config)
        self.language_model = language_model
        self.speech_model = speech_model
        self.modality_adapter = modality_adapter
        self.tokenizer = tokenizer
        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self._language_max_sequence_length = self.language_model.max_sequence_length

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        NOTE: Pipeline parallelism is not supported in this model yet. This is just a placeholder implementation.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        pass

    def _create_attention_mask(self, encoder_input: torch.Tensor):
        """
        Create causal attention mask for whole input
        Args:
            encoder_input: The encoder input tensor of shape [b, t, h].
        Returns:
            attention_mask: The attention mask tensor of shape [b, 1, t, t].
        """
        # Create causal attention mask for whole input
        batch_size = encoder_input.shape[0]
        max_len = encoder_input.shape[1]
        attention_mask = torch.tril(torch.ones((batch_size, max_len, max_len), device=encoder_input.device)).view(
            batch_size, 1, max_len, max_len
        )
        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        # [batch, 1, seq_len, seq_len]
        return attention_mask

    def _concat_features(self, embs1, emb1_lens, embs2, emb2_lens):
        """Concatenate two sets of embeddings and their lengths."""
        concat_emb = []
        concat_len = []
        for emb1, emb1_len, emb2, emb2_len in zip(embs1, emb1_lens, embs2, emb2_lens):
            new_len = emb1_len + emb2_len
            new_emb = torch.concat([emb1[:emb1_len], emb2[:emb2_len]], axis=0)
            padded_new_emb = torch.zeros(
                emb1.shape[0] + emb2.shape[0], emb1.shape[-1], device=emb1.device, dtype=emb1.dtype
            )
            padded_new_emb[:new_len, ...] = new_emb
            concat_emb.append(padded_new_emb)
            concat_len.append(new_len)
        concat_emb = torch.stack(concat_emb, dim=0)
        concat_len = torch.stack(concat_len, dim=0)
        return concat_emb, concat_len

    def _concat_multi_features(
        self,
        encoded: List[torch.Tensor],
        encoded_len: List[torch.Tensor],
        input_embeds: torch.Tensor,
        input_length: torch.Tensor,
        context_start_idx: List[List[int]],
    ):
        """
        Concatenate multiple audio features with text segments.
        This is used when there are more than one audio in a single sample.
        """
        encoder_input_list, encoder_length_list = [], []
        batch_size = input_embeds.size(0)
        max_length = 0
        for i in range(batch_size):
            start_idx_list_i = context_start_idx[i] + [
                input_embeds.size(1)
            ]  # use input_embeds instead of input_length to handle tokens_to_generate in inference
            input_len_list = [start_idx_list_i[j + 1] - start_idx_list_i[j] for j in range(len(start_idx_list_i) - 1)]
            input_emb_list = input_embeds[i].split(input_len_list)
            encoder_input_i = [input_emb_list[0]]
            for j in range(1, len(input_emb_list)):
                encoder_input_i.append(encoded[i][j - 1][: encoded_len[i][j - 1]])
                encoder_input_i.append(input_emb_list[j])
            encoder_input_i = torch.cat(encoder_input_i)  # T, C
            encoder_length_i = encoded_len[i].sum() + input_length[i]  # total length of audio and text features
            max_length = max(max_length, encoder_input_i.size(0))
            encoder_input_list.append(encoder_input_i)
            encoder_length_list.append(encoder_length_i)

        encoder_input = torch.stack(
            [torch.nn.functional.pad(f, (0, 0, 0, max_length - f.size(0))) for f in encoder_input_list]
        )
        encoder_length = torch.LongTensor(encoder_length_list).to(encoder_input.device)
        return encoder_input, encoder_length

    def inject_perception_input(
        self,
        encoded: Union[torch.Tensor, List[torch.Tensor]],
        encoded_len: Union[torch.Tensor, List[torch.Tensor]],
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """
        Inject audio features into the text input and return the final input embeddings to LLM.
        Args:
            encoded: The audio features, tensor of shape [b, t, d] or a tuple/list of such tensors.
            encoded_len: The length of the audio features, tensor of shape [b] or a tuple/list of such tensors.
            input_ids: The input text tokens, tensor of shape [b, t].
            input_length: The length of the input text tokens, tensor of shape [b].
            context_start_idx: The start index of each text segment in the input text tokens, list of list of integers.
        Returns:
            combined_embed: The final input embeddings to the language model, shape [t, b, h].
            attention_mask: The attention mask tensor of shape [b, 1, t, t].
            combined_embed_length: The length of the final input embeddings, shape [b].
            position_ids: The position ids tensor of shape [b, t].
            encoder_max_length: The maximum length of the encoder input, integer.
        """
        # [b, t, c]
        lm_embedding = self.language_model.embedding
        input_embeds = lm_embedding.word_embeddings(input_ids)

        if isinstance(encoded, torch.Tensor):
            # single audio
            combined_embed, combined_embed_length = self._concat_features(
                encoded, encoded_len, input_embeds, input_length
            )
        else:
            # concat multiple audios with text segments
            combined_embed, combined_embed_length = self._concat_multi_features(
                encoded, encoded_len, input_embeds, input_length, context_start_idx
            )

        attention_mask = self._create_attention_mask(combined_embed)
        position_ids = build_position_ids(combined_embed[:, :, 0])

        # Add position embeddings
        if (
            getattr(lm_embedding, "position_embeddings", None) is not None
            and lm_embedding.position_embedding_type == 'learned_absolute'
        ):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            combined_embed = combined_embed + position_embeddings

        encoder_max_length = combined_embed.shape[1]
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        combined_embed = combined_embed.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if lm_embedding.config.fp32_residual_connection:
            combined_embed = combined_embed.float()

        # Dropout.
        if lm_embedding.config.sequence_parallel:
            combined_embed = tensor_parallel.scatter_to_sequence_parallel_region(combined_embed)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if lm_embedding.config.clone_scatter_output_in_embedding:
                combined_embed = combined_embed.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                combined_embed = lm_embedding.embedding_dropout(combined_embed)
        else:
            combined_embed = lm_embedding.embedding_dropout(combined_embed)
        return combined_embed, attention_mask, combined_embed_length, position_ids, encoder_max_length

    def _shift_labels_by_emb_len(self, labels, label_lens, emb_lens, max_len, pad_token=0):
        """Shift labels to the right by the length of the audio embeddings."""
        shifted_labels = []
        for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
            shifted_label = torch.full([max_len], pad_token, device=label.device, dtype=label.dtype)
            shifted_label[emb_len : emb_len + label_len] = label[:label_len]
            shifted_labels.append(shifted_label)
        shifted_labels = torch.stack(shifted_labels, dim=0)
        return shifted_labels

    def _get_text_embeddings(self, text_tokens, position_ids):
        """Get text embeddings for the input text tokens for inference decoding."""
        lm_embedding = self.language_model.embedding
        text_embeddings = lm_embedding.word_embeddings(text_tokens)  # (batch_size, seq_len, hidden_size)
        if hasattr(lm_embedding, 'position_embeddings'):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            text_embeddings = text_embeddings + position_embeddings

        text_embeddings = text_embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if lm_embedding.config.fp32_residual_connection:
            text_embeddings = text_embeddings.float()

        # Dropout.
        if lm_embedding.config.sequence_parallel:
            text_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(text_embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if lm_embedding.config.clone_scatter_output_in_embedding:
                text_embeddings = text_embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                text_embeddings = lm_embedding.embedding_dropout(text_embeddings)
        else:
            text_embeddings = lm_embedding.embedding_dropout(text_embeddings)

        return text_embeddings

    def _get_llm_input_for_context_parallel(
        self,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        labels: torch.Tensor,
        loss_masks: torch.Tensor,
        max_length: int,
    ):
        """
        Prepare context parallel input for the language model, where tensors are padded to the lengths
        divisible by context parallel world size.
        Args:
            attention_mask: The attention mask tensor of shape [b, 1, t, t].
            decoder_input: The decoder input tensor of shape [t, b, h].
            labels: The labels tensor of shape [b, t].
            loss_masks: The loss mask tensor of shape [b, t].
            max_length: The maximum length of the input tensors, integer.
        Returns:
            attention_mask_cp: The attention mask tensor for context parallelism, shape [b, 1, t, t].
            decoder_input_cp: The decoder input tensor for context parallelism, shape [t, b, h].
            labels_cp: The labels tensor for context parallelism, shape [b, t].
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            return attention_mask, decoder_input, labels, loss_masks

        shard_factor = 2 * cp_size  # 2x required by megatron context parallel
        decoder_input = decoder_input.transpose(0, 1).contiguous()  # [t, b, h] -> [b, t, h]
        decoder_input = pad_or_trim_to_max_length(decoder_input, max_length, 0, ceil_to=shard_factor)
        labels = pad_or_trim_to_max_length(labels, max_length, 0, ceil_to=shard_factor)
        loss_masks = pad_or_trim_to_max_length(loss_masks, max_length, 0, ceil_to=shard_factor)
        attention_mask = self._create_attention_mask(decoder_input)

        batch = {
            "attention_mask": attention_mask,
            "decoder_input": decoder_input,
            "labels": labels,
            "loss_mask": loss_masks,
        }

        # Split the batch for context parallelism
        batch_cp = get_batch_on_this_cp_rank(batch)
        attention_mask_cp = batch_cp["attention_mask"]
        decoder_input_cp = batch_cp["decoder_input"].transpose(0, 1).contiguous()  # [b, t, h] -> [t, b, h]
        labels_cp = batch_cp["labels"]
        loss_masks_cp = batch_cp["loss_mask"]
        return attention_mask_cp, decoder_input_cp, labels_cp, loss_masks_cp

    def perception(self, input_signal, input_signal_length, processed_signal, processed_signal_length):
        encoded, encoded_len = self.speech_model(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )
        encoded, encoded_len = self.modality_adapter(encoded, encoded_len)

        return encoded, encoded_len

    def forward(
        self,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_signal: Optional[torch.Tensor] = None,
        audio_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[Dict[str, Any]] = None,
    ):
        encoded, encoded_len = self.perception(
            input_signal=audio_signal,
            input_signal_length=audio_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

        if num_audios is not None:
            # split the encoded and encoded_len by num_audios, used when there're multiple audio files per sample
            encoded = encoded.split(num_audios.tolist())
            encoded_len = encoded_len.split(num_audios.tolist())

        combined_embeddings, attention_mask, _, _, max_length = self.inject_perception_input(
            encoded, encoded_len, input_ids, input_length, context_start_idx
        )

        if num_audios is not None:
            # sum up the audio_feat_lens for each sample in the batch
            encoded_len = torch.stack([torch.sum(lens) for lens in encoded_len])

        if labels is not None:
            # Shift labels to the right
            final_labels = self._shift_labels_by_emb_len(labels, input_length, encoded_len, max_length, pad_token=0)
        else:
            final_labels = None

        if loss_mask is not None:
            # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
            final_loss_mask = self._shift_labels_by_emb_len(
                loss_mask, input_length, encoded_len, max_length, pad_token=0
            )
        else:
            final_loss_mask = None

        attention_mask, combined_embeddings, final_labels, final_loss_mask = self._get_llm_input_for_context_parallel(
            attention_mask, combined_embeddings, final_labels, final_loss_mask, max_length
        )
        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=final_labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )

        if labels is None or loss_mask is None:
            return output

        # [b, t], [b, t]
        return output, final_loss_mask.contiguous()


class SpeechToTextLLM(SpeechLanguageModel):
    def __init__(
        self,
        config: SpeechToTextLLMConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        self._inference_config = None
        self._speech_model = self.config.speech_model_config.configure_model()

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer, self._speech_model)  # type: MCoreSpeechToTextLLM
            self.module.language_model = self.module.language_model.to(self.device)
            self.module.speech_model = self.module.speech_model.to(self.device)
            self.module.modality_adapter = self.module.modality_adapter.to(self.device)
            del self._speech_model

    def setup(self, stage: str):
        super().setup(stage)
        if hasattr(self.cfg.data, "validation_ds"):
            self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
            self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.validation_ds, "metric"):
                self.val_metric_label_key = self.cfg.data.validation_ds.metric.get('label_key', 'labels')

        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.test_ds, "metric"):
                self.test_metric_label_key = self.cfg.data.test_ds.metric.get('label_key', 'labels')

        if self.get_inference_config() is None:
            self.set_inference_config(self.config.inference_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_signal: Optional[torch.Tensor] = None,
        audio_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output = self.module(
            input_ids=input_ids,
            input_length=input_length,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            labels=labels,
            num_audios=num_audios,
            context_start_idx=context_start_idx,
            inference_params=inference_params,
        )
        return output

    def freeze_llm(self):
        module = self.module
        while not hasattr(module, "language_model"):
            module = module.module
        self.freeze_module(module.language_model)

    def freeze_speech(self):
        module = self.module
        while not hasattr(module, "speech_model"):
            module = module.module
        self.freeze_module(module.speech_model)

    def freeze_modality_adapter(self):
        module = self.module
        while not hasattr(module, "modality_adapter"):
            module = module.module
        self.freeze_module(module.modality_adapter)

    def unfreeze_llm(self):
        module = self.module
        while not hasattr(module, "language_model"):
            module = module.module
        self.unfreeze_module(module.language_model)

    def unfreeze_speech(self):
        module = self.module
        while not hasattr(module, "speech_model"):
            module = module.module
        self.unfreeze_module(module.speech_model)

    def unfreeze_modality_adapter(self):
        module = self.module
        while not hasattr(module, "modality_adapter"):
            module = module.module
        self.unfreeze_module(module.modality_adapter)

    def trainable_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        This function returns all trainable parameters of the model,
        including some params that don't require gradients (e.g., batchnorm).
        This function is used for PEFT to determine what parameters to load/save.
        See `nemo/collections/speechlm/utils/model_transform.py` for more details.
        The name of this function is set to align with the PEFT API.
        """
        trainable_params = []
        # must use state_dict() to include params like batchnorm running mean/var
        for name, param in self.state_dict().items():
            if name.startswith("module.speech_model.") and not self.config.freeze_speech_model:
                trainable_params.append((name, param))
            elif name.startswith("module.modality_adapter.") and not self.config.freeze_modality_adapter:
                trainable_params.append((name, param))
            elif name.startswith("module.language_model.") and not self.config.freeze_language_model:
                trainable_params.append((name, param))
            elif (
                name.startswith("module.language_model.")
                and self.config.freeze_language_model
                and (".adapter." in name or name.endswith(".adapters"))
            ):
                trainable_params.append((name, param))

        return trainable_params

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        return self.inference_step(batch, mode='validation')

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric["exact_string_match"]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric_cls = MetricStringToTorchMetric[metric_name]
            if metric_name not in TextMetricsSet:
                metric = [metric_cls(**data_cfg.metric)]
            else:
                metric = [metric_cls()]
        return metric, metric_name

    def inference_step(self, batch, mode):
        """
        Used for validation and test steps, added postprocessing after calling self.predict_step().
        """
        batch_idx = batch.pop("batch_idx", None)
        dataloader_idx = batch.pop("dataloader_idx", None)

        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds

        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        forward_output = self.forward_step(batch)

        if isinstance(forward_output, tuple):
            # reduce validation loss
            loss = self.validation_loss_reduction.forward(batch=batch, forward_out=forward_output)[-1]
            loss = self.validation_loss_reduction.reduce([loss])
        else:
            # no labels provided, use a dummy loss value
            loss = 0.0

        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        preds_text = []
        labels_text = []
        inputs_text = []
        if metric_name != "loss":
            # We need _inference_config to get generation params, tokens_to_generate are set in dataset
            if self.get_inference_config() is None:
                logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
                self.set_inference_config(inference_config=default_inference_config)
            self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

            output = self.predict_step(batch, batch_idx, dataloader_idx)

            inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['contexts']]
            labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['answers']]
            preds_text = [
                self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
                for t, l in zip(output['token_ids'], batch['context_lengths'])
            ]

            if data_cfg.get("end_string", None):
                # sometimes data_cfg.end_string != self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
                # for example when data_cfg.end_string = "<end>", the end_string_re will start with " ?? "
                preds_text = clean_end_string(preds_text, self.tokenizer, data_cfg.end_string)
                labels_text = clean_end_string(labels_text, self.tokenizer, data_cfg.end_string)

            if data_cfg.get("remove_text_pc", False):
                preds_text = [
                    remove_punctuations(pred_text.lower(), data_cfg.get("punctuations", None))
                    for pred_text in preds_text
                ]
                labels_text = [
                    remove_punctuations(label_text.lower(), data_cfg.get("punctuations", None))
                    for label_text in labels_text
                ]

            if data_cfg.get("log_every_n_steps", None) is not None:
                if batch_idx % data_cfg.log_every_n_steps == 0:
                    logging.info(f"Input: `{inputs_text[0]}`")
                    logging.info(f"Label: `{labels_text[0]}`")
                    logging.info(f"Pred: `{preds_text[0]}`")

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }

        if mode == 'validation':
            if self._num_validation_dl > 1:
                self.validation_step_outputs[dataloader_idx].append(outputs)
            else:
                self.validation_step_outputs.append(outputs)
        else:
            if self._num_test_dl > 1:
                self.test_step_outputs[dataloader_idx].append(outputs)
            else:
                self.test_step_outputs.append(outputs)
        return forward_output

    def get_inference_strategy(self):
        return self.config.text_generation_strategy(self.module)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        """
        Used to get LLM predictions for validation and test steps based on the given inference config.
        """
        inference_config = self.get_inference_config()
        if inference_config is not None:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
        else:
            self.set_inference_config(inference_config=default_inference_config)
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            inference_config = self.get_inference_config()

        if self.cfg.data.get('end_string', None):
            inference_config['end_strings'] = [self.cfg.data.end_string]

        inference_config['strategy'] = self.get_inference_strategy()

        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)

        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            elif 'num_audios' in batch:
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                    batch['num_audios'].cuda(),
                    batch['context_start_idx'],
                )
            else:
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                )
            response = generate(self, **inference_config)

        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        # add audio offsets to context lengths for properly decoding only the response
        batch['context_lengths'] = batch['context_lengths'].cuda() + response['audio_feat_lens']

        return response

    def _determine_log_key(self, dataloader_idx, metric_name, mode):
        # If the user provided names for each validation/test dataset, use those.
        if mode == 'validation':
            prefix = self.get_validation_dataloader_prefix(dataloader_idx)
            if prefix.startswith('val_'):
                # no user provided name, use the dataloader idx
                log_key = f'val_{metric_name}_{dataloader_idx}'
            else:
                log_key = f'val_{metric_name}_{prefix}'
        else:
            prefix = self.get_test_dataloader_prefix(dataloader_idx).strip('test_')
            if prefix.startswith('test_'):
                # no user provided name, use the dataloader idx
                log_key = f'test_{metric_name}_{dataloader_idx}'
            else:
                log_key = f'test_{metric_name}_{prefix}'
        return log_key

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs or (all([not x for x in outputs])):
            return None

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'].view(-1, 1) for x in output]  # each loss is [1, B]

            assert (
                getattr(self.config, "virtual_pipeline_model_parallel_size", None) is None
            ), "vpp is not supported yet in SpeechToTextLLMModel"
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                loss = torch.vstack(loss_vals).mean().type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            if metric_name != 'loss':
                self.gather_and_maybe_write_predictions(data_cfg, output, averaged_metric, mode, dataloader_idx)

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 0 else None
        averaged_loss = averaged_loss.to(self.device)
        if averaged_metric is not None:
            averaged_metric = averaged_metric.to(self.device)

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("val_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"val_{self.val_metric_name}", averaged_metric, sync_dist=True, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True, batch_size=1)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        # self._restore_activation_checkpointing_args()
        if hasattr(self.cfg.data, "train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        return averaged_loss, averaged_metric

    def gather_and_maybe_write_predictions(self, data_cfg, output, averaged_metric, mode, dataloader_idx):
        # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
        gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_outputs,
            [
                {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
                for x in output
            ],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        inp_label_set = set()
        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'inputs': [],
            'metadata': [],
        }
        total_size = 0
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_outputs[rank]:
                for pred, label, input, metadata in zip(
                    batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                ):
                    key = input + label + str(metadata)
                    total_size += 1
                    if key not in inp_label_set:
                        inp_label_set.add(key)
                        deduplicated_outputs['preds'].append(pred)
                        deduplicated_outputs['labels'].append(label)
                        deduplicated_outputs['inputs'].append(input)
                        deduplicated_outputs['metadata'].append(metadata)

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key
        if metric_name != 'loss':
            metric_log_key = self._determine_log_key(dataloader_idx, metric_name, mode)
            metric_fn = self.val_metric[0] if mode == 'validation' else self.test_metric[0]
            if metric_label_key in deduplicated_outputs['metadata'][0]:
                labels = [m[metric_label_key] for m in deduplicated_outputs['metadata']]
            else:
                labels = deduplicated_outputs['labels']

            # Compute metrics
            # SacreBLEU does not share the same interface as other metrics. We handle it separately.
            for pred, label in zip(deduplicated_outputs['preds'], labels):
                if metric_name == 'bleu':
                    _ = metric_fn([pred], [[label]])
                else:
                    _ = metric_fn(pred, label)

            metric_result = metric_fn.compute()

            # log the metrics
            if metric_name == 'rouge':
                for k, v in metric_result.items():
                    if 'fmeasure' in k:
                        self.log(metric_log_key + f'_{k}', v.item(), sync_dist=True, batch_size=1)
                        logging.info(f"{metric_log_key}_{k}]: {v.item()}")
                metric_result = metric_result['rouge1_fmeasure']
            else:
                self.log(metric_log_key, metric_result.item(), sync_dist=True, batch_size=1)
                logging.info(f"{metric_log_key}: {metric_result.item()}")

            metric_fn.reset()
            averaged_metric.append(metric_result)

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            filename_log_key = self._determine_log_key(dataloader_idx, metric_name, mode)
            output_dir = data_cfg.get("output_dir", "./")
            self.write_predictions_to_file(deduplicated_outputs, f"speechlm_pred_{filename_log_key}", output_dir)

    # consistent with speech models
    @rank_zero_only
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m in zip(outputs['inputs'], outputs['preds'], outputs['labels'], outputs['metadata']):
                json_string = {'input': i, 'pred_text': p, 'text': l}
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    # Override the parent batch reconfiguring logic.
    def _reconfigure_and_process_inference_batch(self, batch, data_cfg):
        global_batch_size_per_gpu = batch['tokens'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != data_cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=data_cfg.global_batch_size,
                    micro_batch_size=data_cfg.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )

    def set_inference_config(self, inference_config: Optional[Dict] = None):
        self._inference_config = dict(inference_config) if inference_config is not None else None

    def get_inference_config(self):
        return dict(self._inference_config) if self._inference_config is not None else None

    def on_validation_epoch_start(self):
        # self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        # self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_predict_epoch_start(self):
        return self.on_test_epoch_start()

    def on_test_epoch_end(self):
        _ = self.inference_epoch_end(self.test_step_outputs, 'test', self.cfg.data.test_ds)
        # Commenting as on_test_epoch_end was a no-op in PTL 1.9
        # return super().on_test_epoch_end()

    def on_validation_epoch_end(self):
        _ = self.inference_epoch_end(self.validation_step_outputs, 'validation', self.cfg.data.validation_ds)
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # return super().on_validation_epoch_end()

    def on_train_epoch_start(self) -> None:
        # Same logic as validation epoch end, but this may be need if there is no validation sanity check to trigger on_validation_epoch_end()
        self.on_validation_epoch_end()
        return super().on_train_epoch_start()
