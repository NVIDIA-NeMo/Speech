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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import lightning.pytorch as L
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import InferenceParams, dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.multimodal.data.energon.config import AudioToken, ImageToken, MultiModalSampleConfig
from nemo.lightning import io
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging
from nemo.utils.app_state import AppState

MODEL_CONFIG_ATTR = [
    'num_layers',
    'hidden_size',
    'num_attention_heads',
    'num_query_groups',
    'ffn_hidden_size',
    'kv_channels',
    'hidden_dropout',
    'attention_dropout',
    'fp32_residual_connection',
    'apply_residual_connection_post_layernorm',
    'layernorm_epsilon',
    'layernorm_zero_centered_gamma',
    'add_bias_linear',
    'add_qkv_bias',
    'gated_linear_unit',
    'activation_func',
    'activation_func_fp8_input_store',
    'num_moe_experts',
    'rotary_interleaved',
    'window_size',
    'normalization',
    'qk_layernorm',
    'test_mode',
    'calculate_per_token_loss',
    'seq_length',
]


def restore_model_weights(model, checkpoint_path, strict=False):
    """
    Restores model weights from a checkpoint.

    Args:
        model: The model to restore weights for.
        checkpoint_path: Path to the checkpoint.
        strict: Whether to restore weights even if they are not the same.
    """
    if checkpoint_path is not None:
        sharded_state_dict = dict(state_dict=model.sharded_state_dict(prefix="module."))
        loaded_state_dict = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_to_weights_subdir(checkpoint_path, is_saving=False),
            validate_access_integrity=False,
            **({"strict": "log_all"} if not strict else {}),
        )
        loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        model.load_state_dict(loaded_state_dict, strict=strict)


def avlm_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    """AVLM Data Step"""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.update(
        (
            "tokens",
            "attention_mask",
            "images",
            "num_image_tiles",
            "image_sizes",
            "audios",
            "audio_lengths",
        )
    )
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("position_ids",))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(
            (
                "labels",
                "loss_mask",
            )
        )

    packed_seq_params = _batch.get("packed_seq_params", None)
    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    if packed_seq_params is not None:
        for attr in ["cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"]:
            value = getattr(packed_seq_params, attr, None)
            if value is not None:
                setattr(packed_seq_params, attr, value.cuda(non_blocking=True))
    _batch["packed_seq_params"] = packed_seq_params
    if ps.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in _batch and _batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = _batch["loss_mask"].sum()

            # In theory, we need to truncate the sequence to the max sequence length here
            # when calculating num_valid_tokens_in_ub
            # e.g.: num_valid_tokens_in_ub = _batch["loss_mask"][:,:max_seq_len].sum()
            # But when we use CP with packed sequence, the sequence is already packed to be
            # less or equal to the max sequence length
            # Therefore, we don't need to truncate the sequence here

        _batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return _batch


def avlm_forward_step(model, batch) -> torch.Tensor:
    """AVLM Forward Step"""
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "loss_mask": batch.get("loss_mask", None),
        "attention_mask": batch.get("attention_mask", None),
        "labels": batch.get("labels", None),
        "images": batch["images"],
        "num_image_tiles": batch.get("num_image_tiles", None),
        # "image_sizes": batch.get("image_sizes", None),
        # "image_token_mask": batch.get("image_token_mask", None),
        "audios": batch["audios"],
        "audio_lengths": batch.get("audio_lengths", None),
        "packed_seq_params": batch.get("packed_seq_params", None),
    }

    return model(**forward_args)


@dataclass
class AVLMConfig(TransformerConfig, io.IOMixin):
    """AVLM Model Base Config"""

    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None
    audio_transformer_config: Optional[TransformerConfig] = None
    audio_projection_config: Optional[TransformerConfig] = None

    drop_vision_class_token: bool = True
    vision_feature_layer: int = -2
    audio_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    seq_length: int = 1024

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None
    vision_projection_from_pretrained: Optional[str] = None  # TODO
    audio_model_from_pretrained: Optional[str] = None
    audio_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    freeze_audio_model: bool = False
    freeze_audio_projection: bool = False

    forward_step_fn: Callable = avlm_forward_step
    data_step_fn: Callable = avlm_data_step

    def __post_init__(self):
        # pylint: disable=C0115,C0116
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

        assert self.calculate_per_token_loss is False, "AVLM does not return normalized loss"

    def configure_model(self, tokenizer) -> "MCoreAVLMModel":
        # pylint: disable=C0115,C0116
        self.language_transformer_config.scatter_embedding_sequence_parallel = False
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        if self.vision_transformer_config is not None:
            self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
            self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        if self.audio_transformer_config is not None:
            self.audio_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
            self.audio_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_transformer_config.context_parallel_size = self.context_parallel_size

        assert (
            self.encoder_pipeline_model_parallel_size == 0
        ), "AVLM `encoder_pipeline_model_parallel_size` has bug for now. Fix will come soon."

        model = MCoreAVLMModel(
            config=self,
            tokenizer=tokenizer,
            pre_process=ps.is_pipeline_first_stage(),
            post_process=ps.is_pipeline_last_stage(),
            add_encoder=ps.is_pipeline_first_stage(),
            add_decoder=ps.is_pipeline_last_stage()
            or ps.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
        )

        return model


class MCoreAVLMModel(MCoreLLaVAModel):
    """AVLM Model Base Model Class"""

    def __init__(
        self,
        config: AVLMConfig,
        tokenizer: Optional = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = True,
    ) -> None:
        # pylint: disable=C0115,C0116
        super(MCoreLLaVAModel, self).__init__(config=config)

        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        vision_projection_config = config.vision_projection_config
        audio_transformer_config = config.audio_transformer_config
        audio_projection_config = config.audio_projection_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.tokenizer = tokenizer
        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.audio_model = None
        self.audio_projection = None
        self.language_model = None

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.add_decoder:
            self.language_model = language_transformer_config.configure_model(
                tokenizer=tokenizer, pre_process=pre_process, post_process=post_process
            )
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            self._language_max_sequence_length = self.language_model.max_sequence_length
            self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
            restore_model_weights(self.language_model, config.language_model_from_pretrained)
            logging.info(f"Restored language model weights from {config.language_model_from_pretrained}")

        else:
            if config.language_model_from_pretrained is not None:
                dist_checkpointing.load(
                    sharded_state_dict=dict(state_dict={}),
                    checkpoint_dir=config.language_model_from_pretrained,
                    validate_access_integrity=False,
                )

        if self.add_encoder:
            if vision_transformer_config is not None:
                self.vision_model = vision_transformer_config.configure_model()
                self.vision_projection = vision_projection_config.configure_model()
                self._drop_vision_class_token = drop_vision_class_token
                restore_model_weights(self.vision_model, config.vision_model_from_pretrained)
                logging.info(f"Restored vision model weights from {config.vision_model_from_pretrained}")
            if audio_transformer_config is not None:
                app_state = AppState()
                # if checkpoint is in NeMo 1.0, we need to temporarily set
                # model_parallel_size to 1 to load audio encoder, because
                # audio encoder does not support model parallel
                # and was saved with model_parallel_size=1
                if config.audio_model_from_pretrained and config.audio_model_from_pretrained.endswith(".nemo"):
                    with temporary_model_parallel_size(app_state, 1):
                        self.audio_model = audio_transformer_config.configure_model()
                        restore_model_weights(self.audio_model, config.audio_model_from_pretrained)
                        logging.info(f"Restored audio model weights from {config.audio_model_from_pretrained}")
                else:
                    self.audio_model = audio_transformer_config.configure_model()
                    restore_model_weights(self.audio_model, config.audio_model_from_pretrained)
                    logging.info(f"Restored audio model weights from {config.audio_model_from_pretrained}")
                self.audio_projection = audio_projection_config.configure_model()

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
            freeze_audio_model=config.freeze_audio_model,
            freeze_audio_projection=config.freeze_audio_projection,
        )

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        # TODO - Modality specific attributes
        self.vision_model_from_hf = hasattr(vision_transformer_config, "image_size")
        self._img_seq_len = vision_transformer_config.num_image_embeddings_per_tile
        if drop_vision_class_token and vision_transformer_config.add_class_token:
            self._img_seq_len -= vision_transformer_config.class_token_len

    def combine_embeddings(
        self,
        input_ids,
        image_embeddings,
        audio_embeddings,
        language_embeddings,
        image_token_index,
        audio_token_index,
        use_inference_kv_cache,
        packed_seq_params,
    ):
        """
        Combine image_embeddings, audio_embeddings, and language_embeddings into a single tensor.
        """

        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No preprocessing needed if not pre_process
        if not self.pre_process:
            return language_embeddings

        # If using the inference KV cache, the image/audio tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings

        # combine image_embeddings, audio_embeddings, language_embeddings
        image_token_mask = input_ids == image_token_index
        audio_token_mask = input_ids == audio_token_index
        combined_embeddings = language_embeddings
        if image_embeddings is not None:
            combined_embeddings = torch.index_put(combined_embeddings, (image_token_mask,), image_embeddings)
        if audio_embeddings is not None:
            combined_embeddings = torch.index_put(combined_embeddings, (audio_token_mask,), audio_embeddings)

        return combined_embeddings

    def pad_sequence(self, combined_embeddings, labels, loss_mask, packed_seq_params):
        """
        Pad the sequence (labels, loss_mask, combined_embeddings, packed_seq_params) to the
        language model's max sequence length.
        combined_embeddings's shape is [batch_size, seq_len, attention head, embed_dim]

        """

        # determine max sequence length to pad
        if self._language_is_pipeline_parallel:
            max_seq_len = self._language_max_sequence_length
        elif self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                # pad to language_max_sequence_length to use TP Comm overlap.
                max_seq_len = self._language_max_sequence_length
            else:
                shard_factor, _ = self._get_shard_factor(packed_seq_params)
                if combined_embeddings is not None:
                    max_seq_len = combined_embeddings.shape[1]
                elif labels is not None:
                    max_seq_len = labels.shape[1]
                else:
                    # no padding needed
                    max_seq_len = None
                max_seq_len = (max_seq_len - 1) // shard_factor * shard_factor + shard_factor
        else:
            # no need to pad
            return combined_embeddings, labels, loss_mask

        # currently, we always use context parallel with sequence packing, which should reach close the max_seq_len
        if self.context_parallel_lm > 1:
            max_seq_len = self._language_max_sequence_length

        # pad labels and loss_mask
        if labels is not None:
            if labels.shape[1] < max_seq_len:
                pad_size = max_seq_len - labels.shape[1]
                labels = F.pad(labels, (0, pad_size), value=MultiModalSampleConfig.ignore_place_holder)
                loss_mask = F.pad(loss_mask, (0, pad_size), value=0)

        # pad combined_embeddings
        if combined_embeddings is not None:
            if combined_embeddings.shape[1] < max_seq_len:
                pad_size = max_seq_len - combined_embeddings.shape[1]
                # Pad along sequence dimension only, using memory-efficient padding
                pad_shape = (0, 0, 0, pad_size)  # Pad last dim by pad_size
                combined_embeddings = torch.nn.functional.pad(
                    combined_embeddings.contiguous(),  # Ensure contiguous memory layout
                    pad=pad_shape,
                    mode='constant',
                    value=0,
                )

        # pad packed_seq_params
        packed_sequence = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if packed_sequence and packed_seq_params.cu_seqlens_q[-1] != max_seq_len:
            last_seqlen = packed_seq_params.cu_seqlens_q[-1] - packed_seq_params.cu_seqlens_q[-2]
            last_seqlen_padded = max_seq_len - packed_seq_params.cu_seqlens_q_padded[-2]
            assert (
                last_seqlen_padded >= last_seqlen
            ), "`language_max_sequence_length` needs to increase for sequence packing to work properly."
            packed_seq_params.cu_seqlens_q_padded[-1] = max_seq_len
            packed_seq_params.cu_seqlens_kv_padded[-1] = max_seq_len
            packed_seq_params.max_seqlen_q = max(last_seqlen_padded, packed_seq_params.max_seqlen_q)
            packed_seq_params.max_seqlen_kv = max(last_seqlen_padded, packed_seq_params.max_seqlen_kv)

        return combined_embeddings, labels, loss_mask

    def truncate_sequence(self, combined_embeddings, labels, loss_mask, packed_seq_params):
        """
        Truncate the sequence (labels, loss_mask, combined_embeddings) to the language model's max sequence length.
            combined_embeddings's shape is [batch_size, seq_len, attention head, embed_dim]
        """

        # truncate labels and loss_mask
        truncate_labels = (labels is not None) and (labels.shape[1] > self._language_max_sequence_length)
        if truncate_labels:
            labels = labels[:, : self._language_max_sequence_length]
            loss_mask = loss_mask[:, : self._language_max_sequence_length]

        # truncate combined_embeddings
        if combined_embeddings is not None:
            # Truncate if exceeding the language model's max sequence length.
            if combined_embeddings.shape[1] > self._language_max_sequence_length:
                combined_embeddings = combined_embeddings[:, : self._language_max_sequence_length]

        # packed_seq_params truncation
        packed_sequence = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if packed_sequence and packed_seq_params.cu_seqlens_q_padded[-1] > self._language_max_sequence_length:
            truncate_len = packed_seq_params.cu_seqlens_q_padded[-1] - self._language_max_sequence_length
            final_seq_len_padded = (
                packed_seq_params.cu_seqlens_q_padded[-1] - packed_seq_params.cu_seqlens_q_padded[-2]
            )
            final_seq_len_unpadded = packed_seq_params.cu_seqlens_q[-1] - packed_seq_params.cu_seqlens_q[-2]
            final_padding = final_seq_len_padded - final_seq_len_unpadded
            truncate_len -= final_padding
            packed_seq_params.cu_seqlens_q_padded[-1] = self._language_max_sequence_length
            packed_seq_params.cu_seqlens_kv_padded[-1] = self._language_max_sequence_length
            # need to truncate the actual sequence as well
            if truncate_len > 0:
                packed_seq_params.cu_seqlens_q[-1] -= truncate_len
                packed_seq_params.cu_seqlens_kv[-1] -= truncate_len
            assert (
                packed_seq_params.cu_seqlens_q[-1] >= packed_seq_params.cu_seqlens_q[-2]
            ), "with packed sequence, the truncation can only truncate on the last sequence."

        return combined_embeddings, labels, loss_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = ImageToken.token_id,
        audios: Optional[torch.Tensor] = None,
        audio_lengths: Optional[List[int]] = None,
        audio_token_index: Optional[int] = AudioToken.token_id,
        inference_params: Optional[InferenceParams] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0301
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [num_tiles, img_h, img_w]. num_tiles means the number of
            image tiles in this batch.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Attention mask for the language model [batch, 1, combined_seq_len,
            combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images. Default None means `image_token_index`
                arg in the constructor will be used.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            packed_seq_params (PackedSeqParams): Dict with padded token information.
                Required for using SP/CP with padding mask type.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """

        # TODO: not sure what to do with this?
        use_inference_kv_cache = (
            inference_params is not None and "media_tokens_count" in inference_params.key_value_memory_dict
        )

        # If running inference, we can skip images/audios token computation if they were computed already earlier
        # for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
            audio_embeddings = None
        elif self.add_encoder:
            # Encode images
            if self.vision_model is not None:
                has_images = images is not None and images.shape[0] > 0
                if not has_images:
                    # If no images provided, set image embeddings tensor to None.
                    image_embeddings = None
                else:
                    # images is in shape of (num_images_in_mbs, c, h, w)
                    # note num_images_in_mbs is not mbs but total images in this mbs.
                    images = images.to(next(self.vision_model.parameters()).dtype)
                    if self.vision_model_from_hf:
                        self.vision_model = self.vision_model.eval()
                        image_embeddings = self.vision_model(images, output_hidden_states=True)
                        image_embeddings = image_embeddings[-1][
                            self.config.vision_feature_layer
                        ]  # [num_images, img_seq_len, h_vision]
                    else:
                        # TODO(yuya): MCore Clip path not yet support taking a specific layer hidden states
                        image_embeddings = self.vision_model(
                            images, num_unused_layers=-self.config.vision_feature_layer - 1
                        )
                    if self._drop_vision_class_token:
                        class_token_len = getattr(self.vision_model, "class_token_len", 1)
                        image_embeddings = image_embeddings[:, class_token_len:, :]
                    # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
                    image_embeddings = image_embeddings.permute(
                        1, 0, 2
                    ).contiguous()  # [img_seq_len, num_tiles, h_vision]

                    # map vision model output size to language model input size.
                    image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, num_tiles, h_language]

                    # TODO: => need to update this. It might actually already be computed with pre-calculated audio embeddings seq length
                    # TODO: Support batched inference.
                    # In inference, the language model KV cache will be updated for image token positions.
                    # Store the image tokens sequence length to be used as an offset to the KV cache later.
                    if inference_params is not None:
                        inference_params.key_value_memory_dict["image_tokens_count"] = (
                            input_ids == image_token_index
                        ).sum()
                        inference_params.key_value_memory_dict["media_tokens_count"] = (
                            inference_params.key_value_memory_dict["image_tokens_count"]
                        )
            else:
                image_embeddings = None

            # Encode audios
            if self.audio_model is not None:
                has_audios = audios is not None and audios.shape[0] > 0
                if not has_audios:
                    # If no audios provided, set audio embeddings tensor to None.
                    audio_embeddings = None
                else:
                    # We don't cast input to bfloat16 here, because processor prefer input audios data in float32.
                    # the output of preprocessing and encoding will be the dtype of encoder
                    # audios is in shape of (num_audios_in_mbs, audio_feature_dim)
                    # note num_audios_in_mbs is not mbs but total audios in this mbs.
                    audio_embeddings, audio_embedding_lens = self.audio_model(
                        input_signal=audios,
                        input_signal_length=audio_lengths,
                        processed_signal=None,
                        processed_signal_length=None,  # what difference between input_signal and processed_signal?
                    )
                    # [num_audios, h_audio, audio_seq_len] -> [audio_seq_len, num_audios, h_audio]
                    # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
                    audio_embeddings = audio_embeddings.permute(2, 0, 1).contiguous()

                    # map audio model output size to language model input size.
                    audio_embeddings = self.audio_projection(audio_embeddings)

                    # TODO: => need to update this. This first, is not correct for audio tokens count, second, it might actually already be computed with pre-calculated audio embeddings seq length
                    # TODO: Support batched inference.
                    # In inference, the language model KV cache will be updated for audio token positions.
                    # Store the audio tokens sequence length to be used as an offset to the KV cache later.
                    if inference_params is not None:
                        inference_params.key_value_memory_dict["audio_tokens_count"] = (
                            input_ids == audio_token_index
                        ).sum()
                        if "media_tokens_count" in inference_params.key_value_memory_dict:
                            inference_params.key_value_memory_dict[
                                "media_tokens_count"
                            ] += inference_params.key_value_memory_dict["audio_tokens_count"]
                        else:
                            inference_params.key_value_memory_dict["media_tokens_count"] = (
                                inference_params.key_value_memory_dict["audio_tokens_count"]
                            )
            else:
                audio_embeddings = None
        else:
            # only need image_embeddings, audio_embeddings if this is the first stage of LLM
            if self.pre_process:
                image_embeddings, audio_embeddings = self.encoder_hidden_state
            else:
                image_embeddings, audio_embeddings = None, None

        if not self.add_decoder:
            return image_embeddings, audio_embeddings

        combined_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be 0 values
            input_ids_text[input_ids_text < 0] = 0

            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

            # processing image and audio embeddings, reshape them to [number_image_tokens or number_audio_tokens, embed_dim]
            # image modality
            if image_embeddings is not None:
                embed_dim = language_embeddings.shape[-1]
                image_embeddings = image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()
            # audio modality
            # (for audio modality, we need to base on audio_embedding_lens to filter out the padded audio embeddings)
            if audio_embeddings is not None:
                audio_embeddings_max_seq_len, _ = audio_embeddings.shape[0], audio_embeddings.shape[1]
                nonpadded_mask = torch.arange(audio_embeddings_max_seq_len).unsqueeze(1).to(
                    audio_embeddings.device
                ) < audio_embedding_lens.unsqueeze(0)
                nonpadded_audio_embeddings = audio_embeddings[nonpadded_mask]
                audio_embeddings = nonpadded_audio_embeddings

            # combine multimodal embeddings to text embeddings
            combined_embeddings = self.combine_embeddings(
                input_ids,
                image_embeddings,
                audio_embeddings,
                language_embeddings,
                image_token_index,
                audio_token_index,
                use_inference_kv_cache,
                packed_seq_params,
            )

        # pad combined_embeddings, labels, loss_mask if needed
        combined_embeddings, labels, loss_mask = self.pad_sequence(
            combined_embeddings, labels, loss_mask, packed_seq_params
        )

        # truncate combined_embeddings, labels, loss_mask if needed
        combined_embeddings, labels, loss_mask = self.truncate_sequence(
            combined_embeddings, labels, loss_mask, packed_seq_params
        )

        # transpose combined_embeddings [b, s, h] -> [s, b, h]
        if combined_embeddings is not None:
            combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()

        # set loss_mask to contiguous
        if loss_mask is not None:
            loss_mask = loss_mask.contiguous()

        # process combined_embeddings, labels, loss_mask for context/sequence parallelism
        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            if self.context_parallel_lm > 1:
                # _process_embedding_token_parallel expects input in shape bshd for cp
                combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()

            combined_embeddings, labels, loss_mask, packed_seq_params = self._process_embedding_token_parallel(
                combined_embeddings, labels, loss_mask, packed_seq_params
            )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        if labels is None or loss_mask is None:
            return output

        return output, loss_mask

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_audio_model: bool,
        freeze_audio_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
            freeze_audio_model (bool): Freeze the audio model module.
            freeze_audio_projection (bool): Freeze the audio projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)
        if freeze_audio_model and self.audio_model is not None:
            modules.append(self.audio_model)
        if freeze_audio_projection and self.audio_projection is not None:
            modules.append(self.audio_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # TODO: rethink for 2 encoders scenario
    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def _get_shard_factor(self, packed_seq_params):
        """Get shard factor of sequence dimension"""

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            seq_dim = 1
        else:
            seq_dim = 0
        if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
            shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
        elif self.context_parallel_lm > 1:
            shard_factor = self.context_parallel_lm * 2
        elif self.sequence_parallel_lm:
            shard_factor = self.tensor_model_parallel_size_lm
        else:
            shard_factor = 1
            seq_dim = 0

        return shard_factor, seq_dim

    def _process_embedding_token_parallel(self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params):
        """Processes the input data for model parallelism support."""

        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                seq_dim = 1
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                seq_dim = 1
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                seq_dim = 0

            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism {combined_embeddings.shape} with dim {seq_dim}"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                    combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), "TP Comm overlap either requires Vision+Text token length \
                == language_max_sequence_length"

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch.update(
                    {
                        "combined_embeddings": combined_embeddings,
                    }
                )
            if self.post_process:
                batch.update(
                    {
                        "new_labels": new_labels,
                        "new_loss_mask": new_loss_mask,
                    }
                )
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.training.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(batch)
            else:
                try:
                    import transformer_engine_torch as tex
                except ModuleNotFoundError as e:
                    logging.error(
                        "Please update Transformer Engine to >= 1.10 to use \
                            Context Parallel with THD format data"
                    )
                    raise e
                cp_size = ps.get_context_parallel_world_size()
                cp_rank = ps.get_context_parallel_rank()
                for key, data in batch.items():
                    index = tex.thd_get_partitioned_indices(
                        packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
                    )
                    batch[key] = data.index_select(1, index)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            if self.post_process:
                new_labels = batch["new_labels"]
                new_loss_mask = batch["new_loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params


class AVLMModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """Lightning Wrapper for AVLM Model"""

    def __init__(
        self,
        config: AVLMConfig,
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

    def configure_model(self) -> None:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = ImageToken.token_id,
        # image_token_mask: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        audio_lengths: Optional[List[int]] = None,
        audio_token_index: Optional[int] = AudioToken.token_id,
        inference_params: Optional[InferenceParams] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116

        output_tensor = self.module(
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            num_image_tiles=num_image_tiles,
            image_token_index=image_token_index,
            # image_token_mask=image_token_mask,
            audios=audios,
            audio_lengths=audio_lengths,
            audio_token_index=audio_token_index,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        # pylint: disable=C0115,C0116
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReductionWithLossMask()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReductionWithLossMask(validation_step=True)

        return self._validation_loss_reduction


@contextmanager
def temporary_model_parallel_size(app_state, temp_value):
    """
    Context manager to temporarily set the model parallel size.
    """
    original_value = app_state.model_parallel_size
    app_state.model_parallel_size = temp_value
    try:
        yield
    finally:
        app_state.model_parallel_size = original_value


__all__ = [
    "AVLMModel",
    "AVLMConfig",
]
