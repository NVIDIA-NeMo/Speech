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
from typing import Callable, Dict, List, Optional

import lightning.pytorch as L
import torch
import torch.distributed
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

from nemo.collections.vlm.neva.data.multimodal_tokens import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from nemo.lightning import io
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging

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
    'init_method_std',
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
    'position_embedding_type',
    'rotary_base',
    'test_mode',
    'calculate_per_token_loss',
    'seq_length',
    'share_embeddings_and_output_weights',
    'moe_token_dispatcher_type',
    'moe_router_load_balancing_type',
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


def neva_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    """Neva Data Step"""
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
            "media",
            "num_media_tiles",
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
        key: (
            (val if isinstance(val, list) else val.cuda(non_blocking=True))
            if key in required_keys and val is not None
            else None
        )
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
        _batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return _batch


def neva_forward_step(model, batch) -> torch.Tensor:
    """Neva Forward Step"""
    forward_args = {
        "images": batch["media"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
        "num_image_tiles": batch.get("num_media_tiles", None),
        "image_token_mask": batch.get("image_token_mask", None),
        "packed_seq_params": batch.get("packed_seq_params", None),
    }

    return model(**forward_args)


@dataclass
class NevaConfig(TransformerConfig, io.IOMixin):
    """Neva Model Base Config"""

    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None

    drop_vision_class_token: bool = True
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    seq_length: int = 1024

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None
    vision_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = neva_forward_step
    data_step_fn: Callable = neva_data_step

    def __post_init__(self):
        # pylint: disable=C0115,C0116
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

    def configure_model(self, tokenizer) -> "MCoreNevaModel":
        # pylint: disable=C0115,C0116
        self.language_transformer_config.scatter_embedding_sequence_parallel = False
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_transformer_config.context_parallel_size = self.context_parallel_size

        assert "NEVA `encoder_pipeline_model_parallel_size` has bug for now. Fix will come soon."
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

        model = MCoreNevaModel(
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


class MCoreNevaModel(MCoreLLaVAModel):
    """Neva Model Base Model Class"""

    def __init__(
        self,
        config: NevaConfig,
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

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.tokenizer = tokenizer
        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
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
            self.vision_model = vision_transformer_config.configure_model()
            self.vision_projection = vision_projection_config.configure_model()
            self._drop_vision_class_token = drop_vision_class_token
            restore_model_weights(self.vision_model, config.vision_model_from_pretrained)
            logging.info(f"Restored vision model weights from {config.vision_model_from_pretrained}")

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        self.vision_model_from_hf = hasattr(vision_transformer_config, "image_size")
        self._img_seq_len = vision_transformer_config.num_image_embeddings_per_tile
        if drop_vision_class_token and vision_transformer_config.add_class_token:
            self._img_seq_len -= vision_transformer_config.class_token_len

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
        runtime_gather_output: Optional[bool] = None,
        image_token_mask: Optional[torch.Tensor] = None,
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
            image_token_mask (torch.Tensor): Tensor indicating the location of
                image token index in input_ids.
            packed_seq_params (PackedSeqParams): Dict with padded token information.
                Required for using SP/CP with padding mask type.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """

        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and len(images) > 0

        # If running inference, we can skip images token computation if they were computed already earlier
        # for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            vision_param = next(self.vision_model.parameters())
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=vision_param.dtype, device=vision_param.device).reshape(0, 0, 0)
        elif self.add_encoder and has_images:
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
                image_embeddings = self.vision_model(images, num_unused_layers=-self.config.vision_feature_layer - 1)
            if self._drop_vision_class_token:
                class_token_len = getattr(self.vision_model, "class_token_len", 1)
                image_embeddings = image_embeddings[:, class_token_len:, :]

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, num_tiles, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, num_tiles, h_language]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0

            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]

            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)
        elif isinstance(num_image_tiles, list):
            num_image_tiles = torch.tensor(num_image_tiles, dtype=torch.int, device=input_ids.device)

        # Preprocess input, labels and loss mask.
        combined_embeddings, final_labels, final_loss_mask, final_attention_mask = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
            num_image_tiles,
            attention_mask,
            packed_seq_params,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            if (
                self.context_parallel_lm > 1
                and packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd"
            ):
                # _process_embedding_token_parallel expects input in shape bshd for cp + thd
                combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()

            combined_embeddings, final_labels, final_loss_mask, packed_seq_params = (
                self._process_embedding_token_parallel(
                    combined_embeddings, final_labels, final_loss_mask, packed_seq_params
                )
            )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=final_attention_mask,
            decoder_input=combined_embeddings,
            labels=final_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        if labels is None or loss_mask is None:
            return output

        return output, final_loss_mask.contiguous()

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

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
        image_token_index,
        num_image_tiles,
        attention_mask,
        packed_seq_params,
    ):
        """Preprocess input data before input to language model.

        This function is adopted from
        https://github.com/huggingface/transformers/blob/85817d98fb60977c97e3014196a462b732d2ed1a/src/transformers/models/llava_next/modeling_llava_next.py#L409
        for our input data conventions.

        image_token_index = -200 indicates the image position in the input_ids = [0, 1, -200, 2, 3]
        and labels = [1, -200, 2, 3, 4], for example.
        We want to replace the image position (-200) with image_embeddings and return the following:
        - final_embeddings = [0, 1, image_embeddings, 2, 3],
        - final_labels = [1, -100, 2, 3, 4]
        - final_loss_mask = [1, 0, 0, 1, 1]

        This function handles samples without images (text-only sample). It also handles samples
        with images that are split into multiples tiles.

        If pipeline parallelism is not used, then self.pre_process and self.post_process
        are both True and we update both input embeddings, labels and loss masks (if available).

        If pipeline parallelism is used, then we do the following
        - the first language model chunk has self.pre_process = True and
          self.post_process = False. We update input embeddings.
        - the middle language model chunk(s) has self.pre_process = False and
          self.post_process = False. We don't need to update anything.
        - the last language model chunk has self.pre_process = False and
          self.post_process = True. We update labels and loss mask.

        TODO: This function should adjust the attention mask too.
        Currently, we assume the language model uses a causal mask.

        Returns:
            final_embedding (torch.Tensor): image and text embeddings [combined_seq_len, b, h].
            final_labels (torch.Tensor): labels for image and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask [b, combined_seq_len].
        """
        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return language_embeddings, loss_mask, labels, attention_mask

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels, attention_mask

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        packed_sequence = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # Create indices for new text and label positions.
        with torch.no_grad():
            image_token_mask = input_ids == image_token_index
            num_images_per_sample = torch.sum(image_token_mask, dim=-1)

            # Number of tiles per sample.
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor([x.sum() for x in num_image_tiles_batch], device=input_ids.device)

            # Sequence length for each sample is the image sequence length multiplied by
            # the number of tiles for that image, minus image token indices,
            # plus text sequence length.
            seq_lens = num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
            max_seq_len = seq_lens.max()
            # padding for SP and CP
            shard_factor, _ = self._get_shard_factor(packed_seq_params)
            max_seq_len = (max_seq_len - 1) // shard_factor * shard_factor + shard_factor
            # Pipeline parallel expects fixed input size. Check if we need to pad.
            if self._language_is_pipeline_parallel and max_seq_len < self._language_max_sequence_length:
                max_seq_len = self._language_max_sequence_length

            if self.sequence_parallel_lm:
                if self.tp_comm_overlap_lm:
                    # If shorter: Pad to language_max_sequence_length to use TP Comm overlap.
                    # If longer: Gets truncated later.
                    if max_seq_len < self._language_max_sequence_length:
                        padded_seq_len = self._language_max_sequence_length
                else:
                    # Pad to multiple of tp size for sequence parallelism
                    tp_world_size = ps.get_tensor_model_parallel_world_size()
                    padded_seq_len = int((max_seq_len + (tp_world_size - 1)) // tp_world_size * tp_world_size)
                max_seq_len = padded_seq_len

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

            batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get
            # new_position_ids = [576, 577, 578, 579]. text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            # +1 is needed here for the cumulative sum. -1 is adjusting for zero-based indexing.
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            text_position_ids = new_position_ids[batch_indices, non_image_indices]

            # Labels are shifted to left by one.
            # So, shift text position ids and non-image indices to left by one.
            if has_labels:
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]

                label_batch_indices = batch_indices[valid_label_text_position_ids]

                label_non_image_indices = non_image_indices - 1
                valid_label_non_image_indices = label_non_image_indices >= 0
                label_non_image_indices = label_non_image_indices[valid_label_non_image_indices]

            # Create a mask for the image embedding positions.
            images_mask = torch.full((batch_size, max_seq_len), True, dtype=torch.bool, device=input_ids.device)
            # No images in the text positions.
            images_mask[batch_indices, text_position_ids] = False
            # Samples can have different amount of images tokens.
            # new_position_ids[:, -1] gives the last text position id for each sample.
            # Padding is needed when the number of image tokens differs.
            first_padding_idx = new_position_ids[:, -1] + 1
            images_mask[
                torch.arange(max_seq_len, device=first_padding_idx.device).repeat(batch_size, 1)
                >= first_padding_idx.unsqueeze(1)
            ] = False

        # Create the final input embedding (if this is the first language model stage).
        final_embedding = None
        if self.pre_process:
            embed_dim = language_embeddings.shape[-1]
            final_embedding = torch.zeros(
                batch_size,
                max_seq_len,
                embed_dim,
                dtype=language_embeddings.dtype,
                device=language_embeddings.device,
            )

            # Put text embeddings to the text positions in the result tensor.
            final_embedding[batch_indices, text_position_ids] = language_embeddings[batch_indices, non_image_indices]

            # Put image embeddings to image positions.
            final_embedding[images_mask] = image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()

        # Create the final labels and loss mask (if this is the last language model stage).
        final_labels, final_loss_mask = None, None
        if has_labels:
            final_labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            final_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device)

            # Put text labels and loss mask to the text positions.
            final_labels[label_batch_indices, label_text_position_ids] = labels[
                label_batch_indices, label_non_image_indices
            ]

            final_loss_mask[batch_indices, text_position_ids] = loss_mask[batch_indices, non_image_indices]

            # For labels, pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the image positions.
            final_loss_mask[images_mask] = 0

            # Loss mask last text position just before an image
            # so that text token does not need to predict the first image token.
            batch_image_indices, image_indices = torch.where(image_token_mask)
            # Indices just before image tokens. If it's -1, skip it.
            before_image_indices = image_indices - 1
            valid = before_image_indices >= 0
            valid_batch_image_indices = batch_image_indices[valid]
            valid_before_image_indices = before_image_indices[valid]
            # Map those indices those position ids.
            valid_before_image_indices = new_position_ids[valid_batch_image_indices, valid_before_image_indices]

            final_loss_mask[valid_batch_image_indices, valid_before_image_indices] = 0

        if final_embedding is not None and has_labels:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        truncate_labels = has_labels and final_labels.shape[1] > self._language_max_sequence_length
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        # truncate final embedding
        if final_embedding is not None:
            # transpose final_embeddings to sbhd
            # note this will also transpose thd, which is fine
            final_embedding = final_embedding.transpose(1, 0).contiguous()
            # Truncate if exceeding the language model's max sequence length.
            if final_embedding.shape[0] > self._language_max_sequence_length:
                final_embedding = final_embedding[: self._language_max_sequence_length]

        # packed seq param truncation
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

        return final_embedding, final_labels, final_loss_mask, attention_mask

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
            shard_factor, seq_dim = self._get_shard_factor(packed_seq_params)

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
                from megatron.core.utils import get_batch_on_this_cp_rank

                if self.pre_process:
                    batch["combined_embeddings"] = batch["combined_embeddings"].transpose(0, 1)
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
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
                combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params


class NevaModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """Lightning Wrapper for Neva Model"""

    def __init__(
        self,
        config: NevaConfig,
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
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
        runtime_gather_output: Optional[bool] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        output_tensor = self.module(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
            num_image_tiles=num_image_tiles,
            image_token_index=image_token_index,
            runtime_gather_output=runtime_gather_output,
            image_token_mask=image_token_mask,
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


__all__ = [
    "NevaModel",
    "NevaConfig",
    "neva_data_step",
    "neva_forward_step",
]
