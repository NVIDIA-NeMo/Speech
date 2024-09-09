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

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as L
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.inference_params import InferenceParams
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model import local_layer_spec, transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.vlm.neva.data.multimodal_tokens import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


def neva_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
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
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("media", "tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def neva_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "media": batch["media"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def set_input_tensor(self, tensor):
    pass


@dataclass
class MultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    """
    For MLP, fc1 in shape of input_size, ffn_hidden_size, fc2 in shape of ffn_hidden_size, hidden_size
    """

    projector_type: str = "mlp2x_gelu"
    layer_spec: Optional[MLPSubmodules] = None
    input_size: Optional[int] = 1024
    hidden_size: int = 1024
    ffn_hidden_size: int = 1024
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!

    def configure_model(self) -> "MCoreMultimodalProjector":
        if self.projector_type.startswith("mcore") and self.layer_spec is None:
            if self.projector_type == "mcore_mlp":
                self.projector_type = "mlp"  # strip "mcore_" for mcore init
                self.layer_spec = ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    ),
                )
                self.layer_spec = self.layer_spec.submodules
            else:
                raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

            return MCoreMultimodalProjector(
                self,
                self.layer_spec,
                projector_type=self.projector_type,
                input_size=self.input_size,
            )

        # e.g. "mlp2x_gelu"
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(self.input_size, self.hidden_size, bias=True)]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True))
            model = torch.nn.Sequential(*modules)
            from types import MethodType

            model.set_input_tensor = MethodType(set_input_tensor, model)
        else:
            raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

        return model


@dataclass
class HFCLIPVisionConfig(CLIPVisionConfig, io.IOMixin):
    """
    https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/clip/configuration_clip.py#L261
    """

    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None

    def configure_hf_config(self, *args, **kwargs) -> None:
        CLIPVisionConfig.__init__(self, *args, **kwargs)

    def configure_model(self) -> "CLIPVisionModel":
        # Monkey patch the method to the vision encoder
        CLIPVisionModel.set_input_tensor = set_input_tensor

        if self.pretrained_model_name_or_path is None:
            model = CLIPVisionModel(self)
        else:
            model = CLIPVisionModel.from_pretrained(self.pretrained_model_name_or_path)
        # Extend all model.config fields to self
        for key, value in model.config.to_dict().items():
            setattr(self, key, value)
        return model


@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    add_class_token: bool = True
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    transformer_layer_spec: ModuleSpec = transformer_engine_layer_spec

    def configure_model(self) -> "MCoreCLIPViTModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)
        return MCoreCLIPViTModel(
            self,
            transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
        )


@dataclass
class NevaConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None
    drop_vision_class_token: bool = True
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!
    vision_feature_layer: int = -2

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None  # TODO
    vision_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = neva_forward_step
    data_step_fn: Callable = neva_data_step

    def configure_model(self, tokenizer) -> "MCoreLLaVAModel":
        language_model = self.language_transformer_config.configure_model(tokenizer=tokenizer)
        vision_model = self.vision_transformer_config.configure_model()
        vision_projection = self.vision_projection_config.configure_model()

        if self.language_model_from_pretrained is not None:
            sharded_state_dict = dict(state_dict=language_model.sharded_state_dict(prefix="module."))
            loaded_state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict, checkpoint_dir=self.language_model_from_pretrained
            )
            loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
            language_model.load_state_dict(loaded_state_dict)
            logging.info(f"Restored language model weights from {self.language_model_from_pretrained}")
        model = MCoreNevaModel(
            transformer_config=self,
            language_model=language_model,
            vision_model=vision_model,
            vision_projection=vision_projection,
            drop_vision_class_token=self.drop_vision_class_token,
        )
        model.freeze(
            freeze_language_model=self.freeze_language_model,
            freeze_vision_model=self.freeze_vision_model,
            freeze_vision_projection=self.freeze_vision_projection,
        )
        return model


class MCoreNevaModel(MCoreLLaVAModel):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        language_model: MegatronModule,
        vision_model: MegatronModule,
        vision_projection: MegatronModule,
        pre_process: bool = True,
        post_process: bool = True,
        drop_vision_class_token: bool = False,
    ) -> None:
        super(MCoreLLaVAModel, self).__init__(config=transformer_config)

        logging.warning("LLaVA model is under development and may be missing features.")

        self.pre_process = pre_process
        self.post_process = post_process

        self.encoder_hidden_state = None
        self.vision_model = vision_model
        self.vision_projection = vision_projection
        self.language_model = language_model
        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.language_model is not None:
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            self._language_max_sequence_length = self.language_model.max_sequence_length

        if self.vision_model is not None:
            self._drop_vision_class_token = drop_vision_class_token

        self.add_encoder = self.vision_model is not None
        self.add_decoder = self.language_model is not None
        self.vision_model_from_hf = str(self.vision_model.__class__.__module__).startswith("transformers.")
        if self.add_decoder:
            vision_config = self.config.vision_transformer_config
            if self.vision_model_from_hf:
                # img_h, img_w, patch_dim, add_class_token, class_token_len
                self._img_seq_len = get_image_sequence_length(
                    img_h=vision_config.image_size,
                    img_w=vision_config.image_size,
                    patch_dim=vision_config.patch_size,
                    add_class_token=not drop_vision_class_token,
                    class_token_len=0 if "siglip" in vision_config.model_type else 1,
                )
            else:
                self._img_seq_len = 576  # TODO(yuya): Fix hardcode
        else:
            self._img_seq_len = 0

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
    ):
        # TODO (yuya): remove this and use the mcore method
        """Preprocess input data before input to language model.

        This function is adopted from
        https://github.com/huggingface/transformers/blob/85817d98fb60977c97e3014196a462b732d2ed1a/src/transformers/models/llava_next/modeling_llava_next.py#L409
        for our input data conventions.

        image_token_index = -200 indicates the image position in the input_ids = [0, 1, -200, 2, 3] and labels = [1, -200, 2, 3, 4], for example.
        We want to replace the image position (-200) with image_embeddings and return the following:
        - final_embeddings = [0, 1, image_embeddings, 2, 3],
        - final_labels = [1, -100, 2, 3, 4]
        - final_loss_mask = [1, 0, 0, 1, 1]

        This function also handles the case where the input does not contain an image (text-only sample). It also handles the case where a single input
        image is split into multiple tiles.

        If pipeline parallelism is not used, then self.pre_process and self.post_process are both True and we update both
        input embeddings, labels and loss masks (if available).

        If pipeline parallelism is used, then we do the following
        - the first language model chunk has self.pre_process = True and self.post_process = False. We update input embeddings.
        - the middle language model chunk(s) has self.pre_process = False and self.post_process = False. We don't need to update anything.
        - the last language model chunk has self.pre_process = False and self.post_process = True. We update labels and loss mask.

        TODO: This function should adjust the attention mask too. Currently, we assume the language model uses a causal mask.

        Returns:
            final_embedding (torch.Tensor): image and text embeddings concated [combined_seq_len, b, h].
            final_labels (torch.Tensor): labels for image and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask for image and text positions [b, combined_seq_len].
        """
        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed. With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return language_embeddings, loss_mask, labels

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        # Create indices for new text and label positions.
        with torch.no_grad():
            image_token_mask = input_ids == image_token_index
            num_image_tokens = torch.sum(image_token_mask, dim=-1)

            # Number of tiles per sample.
            num_image_tiles_batch = num_image_tiles.split(num_image_tokens.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor([x.sum() for x in num_image_tiles_batch], device=input_ids.device)

            # Sequence length for each sample is the image sequence length multiplied by the number of tiles for that image, minus image token indices,
            # plus text sequence length.
            seq_lens = num_image_tiles_batch * img_seq_len - num_image_tokens + text_seq_len
            max_seq_len = seq_lens.max()
            batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get new_position_ids = [576, 577, 578, 579].
            # text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            # +1 is needed here for the cumulative sum. -1 is adjusting for zero-based indexing.
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            text_position_ids = new_position_ids[batch_indices, non_image_indices]

            # Labels are shifted to left by one. So, shift text position ids and non-image indices to left by one.
            if has_labels:
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]

                label_batch_indices = batch_indices[valid_label_text_position_ids]

                label_non_image_indices = non_image_indices - 1
                valid_label_non_image_indices = label_non_image_indices >= 0
                label_non_image_indices = label_non_image_indices[valid_label_non_image_indices]

        # Create a mask for the image embedding positions.
        with torch.no_grad():
            images_mask = torch.full((batch_size, max_seq_len), True, dtype=torch.bool, device=input_ids.device)
            # No images in the text positions.
            images_mask[batch_indices, text_position_ids] = False
            # Samples can have different amount of images tokens. new_position_ids[:, -1] gives the last text position id for each sample.
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
                dtype=image_embeddings.dtype,
                device=image_embeddings.device,
            )

            # Put text embeddings to the text positions in the result tensor.
            final_embedding[batch_indices, text_position_ids] = language_embeddings[batch_indices, non_image_indices]

            # Put image embeddings to image positions.
            final_embedding[images_mask] = image_embeddings.reshape(-1, embed_dim).contiguous()

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

            # For labels, we need to pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the image positions.
            final_loss_mask[images_mask] = 0

            # Loss mask last text position just before an image so that text token does not need to predict the first image token.
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

        if final_embedding is not None:
            final_embedding = final_embedding.transpose(1, 0).contiguous()

        # Truncate if exceeding the language model's max sequence length.
        if final_embedding is not None and final_embedding.shape[0] > self._language_max_sequence_length:
            final_embedding = final_embedding[: self._language_max_sequence_length]

        if has_labels and final_labels.shape[1] > self._language_max_sequence_length:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        return final_embedding, final_labels, final_loss_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        media: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_media_tiles: Optional[List[int]] = None,
        media_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [num_tiles, img_h, img_w]. num_tiles means the number of image tiles in this batch.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_media_tiles (list of int): Number of tiles per image. Default None assumes 1 tile per image.
            image_token_index (int): ID for input images.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip media token computation if they were computed already earlier for this sample.
        if use_inference_kv_cache or media is None:
            media_embeddings = None
        elif self.add_encoder:
            # media is in shape of (num_images_in_mbs, c, h, w)
            # note num_images_in_mbs is not mbs but total images in this mbs.
            if self.vision_model_from_hf:
                media_embeddings = self.vision_model(
                    media, output_hidden_states=True
                )  # [num_images, img_seq_len, h_vision]
                media_embeddings = media_embeddings[-1][
                    self.config.vision_feature_layer
                ]  # take second from last layer
            else:
                # TODO(yuya): MCore Clip path not yet support taking a specific layer hidden states
                media_embeddings = self.vision_model(media)
            if self._drop_vision_class_token:
                class_token_len = getattr(self.vision_model, "class_token_len", 1)
                media_embeddings = media_embeddings[:, class_token_len:, :]

            # map vision model output size to language model input size.
            media_embeddings = self.vision_projection(media_embeddings)  # [img_seq_len, num_tiles, h_vision]

            # If running inference, the language model KV cache will be updated for media token positions.
            # Here we store the media tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["media_tokens_count"] = (
                    media_embeddings.shape[0] * media_embeddings.shape[1]
                )
        else:
            media_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return media_embeddings, loss_mask

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0
            # Note: This adds absolute position embedding but not RoPE. Each image is counted as one position.
            # RoPE is added in language_model forward call. Each image embedding is one position.
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

        if media is None:
            combined_embeddings = language_embeddings.transpose(1, 0).contiguous()
            final_labels = labels
            final_loss_mask = loss_mask
        else:
            # Assume 1 tile per image if the number of tiles is not provided.
            if num_media_tiles is None:
                num_media_tiles = torch.ones(media.shape[0], dtype=torch.int, device=input_ids.device)

            # Preprocess input, labels and loss mask.
            combined_embeddings, final_labels, final_loss_mask = self._preprocess_data(
                media_embeddings,
                language_embeddings,
                input_ids,
                loss_mask,
                labels,
                use_inference_kv_cache,
                media_token_index,
                num_media_tiles,
            )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=final_labels,
            inference_params=inference_params,
        )

        if labels is None or loss_mask is None:
            return output

        return output, final_loss_mask.contiguous()


class NevaModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: NevaConfig,
        # TODO: Add transformer_layer_spec when we update mcore
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
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        media: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            media=media,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReductionWithLossMask()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReductionWithLossMask(validation_step=True)

        return self._validation_loss_reduction


__all__ = [
    "NevaModel",
    "NevaConfig",
    "neva_data_step",
    "neva_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]
