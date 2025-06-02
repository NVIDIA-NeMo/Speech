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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.enums import ModelType
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference_params import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params, get_batch_on_this_cp_rank
from torch import nn

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.collections.llm.gpt.model.gemma3 import Gemma3Config
from nemo.collections.vlm.gemma3vl.model.vision import Gemma3VLMultimodalProjectorConfig, Gemma3VLVisionConfig
from nemo.collections.vlm.neva.model.base import MODEL_CONFIG_ATTR, NevaModel, restore_model_weights
from nemo.lightning import io
from nemo.lightning.pytorch.optim import OptimizerModule
from nemo.utils.import_utils import safe_import_from

TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")

HAVE_TEX = True
try:
    import transformer_engine_torch as tex

except ImportError:
    HAVE_TEX = False


def gemma3vl_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    """Gemma3 VL model data setp"""
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.update(("input_ids", "position_ids", "tokens"))
    if ps.is_pipeline_first_stage():
        required_keys.update(("pixel_values",))
    if ps.is_pipeline_last_stage():
        required_keys.update(
            (
                "labels",
                "loss_mask",
            )
        )

    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    return _batch


def gemma3vl_forward_step(model, batch) -> torch.Tensor:
    """Gemma3 VL model forward step"""
    forward_args = {
        "input_ids": batch.get("input_ids", batch["tokens"]),
        "position_ids": batch["position_ids"],
        "pixel_values": batch.get("pixel_values", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def set_input_tensor(self, tensor):
    """Placeholder for `set_input_tensor` method for PP implementation."""
    # pylint: disable=W0107
    pass


@dataclass
class Gemma3VLConfig(TransformerConfig, io.IOMixin):
    """Gemma3 VL model base config"""

    language_transformer_config: Optional[Gemma3Config] = None
    vision_transformer_config: Optional[Gemma3VLVisionConfig] = None
    vision_projection_config: Optional[Gemma3VLMultimodalProjectorConfig] = None

    # 0: encoder and first stage of decoder share the stage
    # 1: only encoder itself is in first stage
    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None
    vision_projection_from_pretrained: Optional[str] = None

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = gemma3vl_forward_step
    data_step_fn: Callable = gemma3vl_data_step

    def __post_init__(self):
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

    def configure_model(self, tokenizer, vp_stage: Optional[int] = None) -> "MCoreGemma3VLModel":
        """Configure Gemma3 VL model"""
        self.language_transformer_config.is_vision_language = True
        # Disable SP scatter to allow combining language and vision embedding.
        self.language_transformer_config.scatter_embedding_sequence_parallel = False
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_transformer_config.context_parallel_size = self.context_parallel_size
        self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size

        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
            self.vision_transformer_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.language_transformer_config.encoder_pipeline_model_parallel_size = (
                self.encoder_pipeline_model_parallel_size
            )
            if self.encoder_tensor_model_parallel_size > 0:
                self.vision_transformer_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size

        assert (
            getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None
        ), "Virtual pipeline model parallel size is not yet supported for Gemma3VL."

        model = MCoreGemma3VLModel(
            config=self,
            tokenizer=tokenizer,
            pre_process=ps.is_pipeline_first_stage()
            or ps.get_pipeline_model_parallel_rank() == self.encoder_pipeline_model_parallel_size,
            post_process=ps.is_pipeline_last_stage(),
            add_encoder=ps.is_pipeline_first_stage(),
            add_decoder=ps.is_pipeline_last_stage()
            or ps.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
        )

        return model


class MCoreGemma3VLModel(MegatronModule):
    """Gemma3 VL model base class"""

    def __init__(
        self,
        config: Gemma3VLConfig,
        tokenizer: Optional[Any] = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        # pylint: disable=C0115,C0116
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        vision_projection_config = config.vision_projection_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.is_sequence_parallel = language_transformer_config.sequence_parallel
        self.tp_comm_overlap = language_transformer_config.tp_comm_overlap
        self.context_parallel_size = language_transformer_config.context_parallel_size
        self.tensor_parallel_size = language_transformer_config.tensor_model_parallel_size
        self.is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
        if self.context_parallel_size > 1:
            self.cp_group = get_context_parallel_group()
            assert (
                torch.distributed.get_world_size(self.cp_group) == self.context_parallel_size
            ), "CP Group size should match the Language Model CP size"
        else:
            self.cp_group = None

        self.max_seq_len = language_transformer_config.seq_length
        self.model_type = ModelType.encoder_or_decoder
        self.image_token_id = vision_transformer_config.image_token_id
        self.vocab_size = language_transformer_config.vocab_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False

        if self.add_encoder:
            self.vision_model = vision_transformer_config.configure_model()
            self.vision_projection = vision_projection_config.configure_model()
            restore_model_weights(self.vision_model, config.vision_model_from_pretrained)

        if self.add_decoder:
            self.language_model = language_transformer_config.configure_model(
                tokenizer=tokenizer,
                pre_process=pre_process,
                post_process=post_process,
            )
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            restore_model_weights(self.language_model, config.language_model_from_pretrained)
        else:
            if config.language_model_from_pretrained is not None:
                dist_checkpointing.load(
                    sharded_state_dict=dict(state_dict={}),
                    checkpoint_dir=config.language_model_from_pretrained,
                    validate_access_integrity=False,
                )

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        # inference_params is deprecated, this is for backward compatibility
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        """Forward of the Gemma3VL model"""
        inference_context = deprecate_inference_params(inference_context, inference_params)
        use_inference_kv_cache = (
            inference_context is not None and "image_tokens_count" in inference_context.key_value_memory_dict
        )

        # Annotation:
        # B: batch size
        # N: number of images per sample
        # U': number of image token before pooling, 4096
        # U: number of image tokens after pooling, 256
        # C: image channels
        # H, W: image size
        # M: vision model hidden size
        # D: language model hidden size
        batch_size = input_ids.shape[0]

        # Compute images embedding
        has_images = pixel_values is not None
        if use_inference_kv_cache:
            # If running inference, we can skip media token computation if they were computed already earlier
            # for this sample.
            image_embedding = None
        elif self.add_encoder:
            if not has_images:
                # If no images provided, use an empty image embeddings tensor.
                image_embedding = None
            else:
                # (B * N, C, H, W)
                pixel_values = pixel_values.reshape(
                    -1,
                    pixel_values.shape[2],
                    pixel_values.shape[3],
                    pixel_values.shape[4],
                ).contiguous()
                # (B * N, U', M)
                image_embedding = self.vision_model(pixel_values)
                # (B * N, U, D)
                image_embedding = self.vision_projection(image_embedding)
                # (B, N, U, D)
                image_embedding = image_embedding.view(
                    batch_size, -1, image_embedding.shape[1], image_embedding.shape[2]
                )

                # TODO: Support batched inference.
                # In inference, the language model KV cache will be updated for image token positions.
                # Store the image tokens sequence length to be used as an offset to the KV cache later.
                if inference_context is not None:
                    inference_context.key_value_memory_dict["media_tokens_count"] = (
                        image_embedding.shape[1] * image_embedding.shape[2]
                    )
        else:
            image_embedding = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embedding, loss_mask

        # Adjust prompt sequence length before any compute
        language_seq_len = input_ids.shape[1]
        # Truncate input with max sequence length
        if language_seq_len > self.max_seq_len:
            if self.pre_process:
                input_ids = input_ids[:, : self.max_seq_len]
                position_ids = position_ids[:, : self.max_seq_len]
            if self.post_process:
                labels = labels[:, : self.max_seq_len]
                loss_mask = loss_mask[:, : self.max_seq_len]
        # Pipeline parallel requires fixed input length
        if self.is_pipeline_parallel and language_seq_len < self.max_seq_len and inference_context is None:
            padded_seq_len = self.max_seq_len - language_seq_len
            if self.pre_process:
                input_ids = F.pad(input_ids, (0, padded_seq_len))
                position_ids = F.pad(position_ids, (0, padded_seq_len))
            if self.post_process:
                labels = F.pad(labels, (0, padded_seq_len))
                loss_mask = F.pad(loss_mask, (0, padded_seq_len))

        # Compute language embedding
        if self.pre_process:
            safe_input_ids = input_ids
            # Replace image_token_id with 0 to avoid embedding index error
            if self.image_token_id >= self.vocab_size:
                image_token_mask = input_ids == self.image_token_id
                safe_input_ids = input_ids.clone()
                safe_input_ids[image_token_mask] = 0
            # (T, B, D)
            language_embedding = self.language_model.embedding(input_ids=safe_input_ids, position_ids=position_ids)
            # (B, T, D)
            language_embedding = language_embedding.transpose(1, 0).contiguous()
            # Ensure T is not sharded here
            assert input_ids.shape[1] == language_embedding.shape[1]
        else:
            language_embedding = None

        # Process inputs
        # (B, T, D), (B, 1, T, T)
        combined_embedding, attention_mask = self._preprocess_data(
            input_ids=input_ids,
            image_embedding=image_embedding,
            language_embedding=language_embedding,
            use_inference_kv_cache=use_inference_kv_cache,
        )
        if self.context_parallel_size > 1 or self.is_sequence_parallel:
            combined_embedding, labels, loss_mask, packed_seq_params = self._process_sequence_parallel(
                combined_embedding, labels, loss_mask, packed_seq_params
            )
        elif combined_embedding is not None:
            combined_embedding = combined_embedding.transpose(1, 0).contiguous()

        # Run decoder model
        output = self.language_model(
            input_ids=None,
            position_ids=None,
            decoder_input=combined_embedding,  # (T, B, D)
            attention_mask=attention_mask,  # (B, 1, T, T)
            labels=labels,  # (B, T)
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        if labels is None or loss_mask is None:
            return output
        return output, loss_mask

    def _preprocess_data(
        self,
        input_ids: torch.Tensor,
        image_embedding: Optional[torch.Tensor] = None,
        language_embedding: Optional[torch.Tensor] = None,
        use_inference_kv_cache: bool = False,
    ):
        if not self.pre_process:
            return None, None

        attention_mask = self._compute_attention_mask(input_ids)

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embedding, attention_mask

        # Merge image and language embedding (for language model first stage)
        # This can handle the case that each sample has different N but padded to N
        # input_ids: (B, T)
        # language_embedding: (B, T, D)
        # image_embedding: (B, N, U, D), N * U <= T
        # final_embedding: (B, T, D)
        final_embedding = language_embedding
        # No images presence mask needed to support the case that
        # each sample has different number of images.
        if image_embedding is not None:
            # (B, T)
            image_mask = input_ids == self.image_token_id
            # (B, T, D)
            image_mask = image_mask.unsqueeze(-1).expand_as(final_embedding)
            # (B, N * U, D)
            image_embedding = image_embedding.view(image_embedding.shape[0], -1, image_embedding.shape[-1])
            # (B, T, D) + (B, N * U, D) = (B, T, D)
            final_embedding = final_embedding.masked_scatter(image_mask, image_embedding)

        # (B, T, D)
        return final_embedding, attention_mask

    def _process_sequence_parallel(
        self,
        combined_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        if not self.pre_process and not self.post_process:
            return combined_embedding, labels, loss_mask, packed_seq_params

        if self.pre_process:
            if self.context_parallel_size > 1 and self.is_sequence_parallel:
                shard_factor = self.tensor_parallel_size * self.context_parallel_size * 2
            elif self.context_parallel_size > 1:
                shard_factor = self.context_parallel_size * 2
            else:
                shard_factor = self.tensor_parallel_size
            assert combined_embedding.shape[1] % shard_factor == 0
            if self.is_sequence_parallel:
                assert combined_embedding.shape[1] % self.tensor_parallel_size == 0
                if self.tp_comm_overlap:
                    assert combined_embedding.shape[1] == self.max_seq_len

        if self.context_parallel_size > 1:
            batch = dict()
            if self.pre_process:
                batch["combined_embedding"] = combined_embedding
            if self.post_process:
                batch["labels"] = labels
                batch["loss_mask"] = loss_mask
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                batch = get_batch_on_this_cp_rank(batch)
            else:
                cp_size = self.cp_group.size()
                cp_rank = self.cp_group.rank()
                for key, data in batch.items():
                    index = tex.thd_get_partitioned_indices(
                        packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
                    )
                    batch[key] = data.index_select(1, index)
            if self.pre_process:
                combined_embedding = batch["combined_embedding"]
            if self.post_process:
                labels = batch["labels"]
                loss_mask = batch["loss_mask"]

        if combined_embedding is not None:
            # After doing CP, the shape needs to be (T / CP, B, D)
            # If not using CP, the shape needs to be (T, B, D).
            combined_embedding = combined_embedding.transpose(1, 0).contiguous()
            if self.is_sequence_parallel and self.pre_process:
                # (T / (CP * TP), B, D)
                combined_embedding = scatter_to_sequence_parallel_region(combined_embedding)
        return combined_embedding, labels, loss_mask, packed_seq_params

    def _compute_attention_mask(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.pre_process:
            return None
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len))).to(input_ids.device)

        image_mask = input_ids == self.image_token_id
        padded_mask = F.pad(image_mask, (1, 0), value=0)
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        numbered_boundary = torch.cumsum(boundary, dim=-1)
        q_block_indices = image_mask * numbered_boundary
        kv_block_indices = q_block_indices
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        # See te.DotProductAttention for the requirement of custom mask
        attention_mask = ~torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        return attention_mask


class Gemma3VLModel(NevaModel):
    """Lightning wrapper for Gemma3VL model"""

    def __init__(
        self,
        config: Gemma3VLConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config=config,
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
        )

    # pylint: disable=W0221
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        output_tensor = self.module(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            loss_mask=loss_mask,
            labels=labels,
            inference_params=inference_params,
        )

        return output_tensor


__all__ = [
    "Gemma3VLModel",
    "Gemma3VLConfig",
    "gemma3vl_data_step",
    "gemma3vl_forward_step",
]
