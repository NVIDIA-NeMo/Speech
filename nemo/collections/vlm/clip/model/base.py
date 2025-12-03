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

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import lightning.pytorch as L
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core.enums import ModelType

try:
    from megatron.core.extensions.transformer_engine import TENorm
except ImportError:
    from nemo.utils import logging

    # These Defaults are needed to make sure the code compiles
    TENorm = None
    logging.warning(
        "Failed to import Transformer Engine dependencies. "
        "`from megatron.core.extensions.transformer_engine import TENorm`"
        "If using NeMo Run, this is expected. Otherwise, please verify the Transformer Engine installation."
    )

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tqdm import tqdm

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import default_layer_spec
from nemo.collections.multimodal.data.clip.clip_dataset import build_imagenet_validation_dataloader_params
from nemo.collections.vlm.clip.loss.clip_loss import ClipMegatronLoss
from nemo.lightning import MegatronOptimizerModule, OptimizerModule, get_vocab_size, io
from nemo.utils import logging
from nemo.utils.megatron_utils import average_losses_across_data_parallel_group


# pylint: disable=C0116
def clip_forward_step(model, batch) -> torch.Tensor:
    forward_args = {"images": batch["images"], "captions": batch["captions"]}
    return model(**forward_args)


def clip_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    batch = next(dataloader_iter)

    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    if "captions" in _batch and len(_batch["captions"].shape) == 3:
        _batch["captions"] = _batch["captions"].squeeze()

    _batch = {key: val.cuda(non_blocking=True) if val is not None else None for key, val in _batch.items()}
    return _batch


def set_input_tensor(self, tensor):
    pass


# pylint: enable=C0116
@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    """Clip ViT model config"""

    output_dim: int = 512
    add_class_token: bool = True
    class_token_len: int = 8

    patch_dim: int = 16
    img_h: int = 224
    img_w: int = 224
    vision_model_type: str = "clip"  # ["clip", "siglip"]
    transformer_layer_spec: ModuleSpec = transformer_engine_layer_spec
    gated_linear_unit: bool = False
    attention_softmax_in_fp32: bool = False

    # Without these the init for transformer will give error
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    def configure_model(self) -> "CLIPViTModel":
        # pylint: disable=C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te

            transformer_layer_spec = get_layer_spec_te(is_vit=True)

        transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = MCoreAttnMaskType.no_mask
        self.transformer_layer_spec = transformer_layer_spec

        return CLIPViTModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
            output_dim=self.output_dim,
        )


class CLIPViTModel(MCoreCLIPViTModel):
    """Clip ViT model"""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        add_class_token: bool = True,
        class_token_len: int = 8,
        patch_dim: int = 16,
        img_h: int = 224,
        img_w: int = 224,
        model_subtype: str = "clip",
        output_dim: int = 1024,
    ):
        # pylint: disable=C0116
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = output_dim

        super().__init__(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            add_class_token=add_class_token,
            class_token_len=class_token_len,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            model_subtype=model_subtype,
        )

        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=False,
        )

    def set_input_tensor(self, tensor):
        # pylint: disable=C0116
        pass

    def forward(self, x):
        # pylint: disable=C0116
        x = super().forward(
            x,
        )
        x = self.final_layernorm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


@dataclass
class CLIPTextModelConfig(TransformerConfig, io.IOMixin):
    """Clip text model config"""

    output_dim: int = 512
    make_vocab_size_divisible_by: int = 128
    max_seq_length: int = 1024

    share_embeddings_and_output_weights: bool = False

    # Imported from gpt/base model
    use_transformer_engine_full_layer_spec: bool = False
    transformer_layer_spec: ModuleSpec = default_layer_spec

    # Without these the init for transformer will give error

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "CLIPTextModel":
        # pylint: disable=C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        if hasattr(self, 'vocab_size'):
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

        return CLIPTextModel(
            transformer_config=self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=self.max_seq_length,
            output_dim=self.output_dim,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
        )


class CLIPTextModel(MCoreGPTModel):
    """Clip text model"""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        output_dim: int = 1024,
        share_embeddings_and_output_weights: bool = False,
    ):
        # pylint: disable=C0116
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = output_dim

        # We give post_process as false to get hidden states instead of logits as we have one more layer head
        super().__init__(
            transformer_config,
            transformer_layer_spec,
            vocab_size,
            max_sequence_length,
            True,
            False,
            share_embeddings_and_output_weights,
        )
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=False,
        )
        self.position_ids = None
        if self.pre_process:
            self.position_ids = torch.arange(max_sequence_length).expand(1, -1).cuda()

    def forward(self, input_ids):
        # pylint: disable=C0116
        x = super().forward(input_ids, position_ids=self.position_ids, attention_mask=None)
        x = self.final_layernorm(x)
        x = x[input_ids.argmax(dim=-1), torch.arange(x.shape[1])]
        x = self.head(x)
        return x

    def set_input_tensor(self, tensor):
        # pylint: disable=C0116
        pass


@dataclass
class CLIPConfig(TransformerConfig, io.IOMixin):
    """Clip model config"""

    text_transformer_config: Optional[CLIPTextModelConfig] = None
    vision_transformer_config: Optional[CLIPViTConfig] = None
    get_attention_mask_from_fusion: bool = True
    forward_step_fn: Callable = clip_forward_step
    data_step_fn: Callable = clip_data_step

    # Without these the init for transformer will give error
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!
    hidden_size: int = 768  # Placeholder, NOT used!

    # seq_length is needed for Nemo CI. It is not used anywhere
    seq_length: int = 80

    def configure_model(self, tokenizer, pre_process=True, post_process=True):
        # pylint: disable=C0116
        print(self.kv_channels)
        return MCoreClipModel(
            self,
            tokenizer=tokenizer,
            pre_process=pre_process,
            post_process=post_process,
        )


class MCoreClipModel(MegatronModule):
    """Clip model"""

    def __init__(self, config: CLIPConfig, tokenizer, pre_process=True, post_process=True) -> None:
        # pylint: disable=C0116
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        vision_transformer_config = config.vision_transformer_config
        text_transformer_config = config.text_transformer_config
        self.output_dim = config.vision_transformer_config.output_dim
        self.vision_model = vision_transformer_config.configure_model()
        self.text_model = text_transformer_config.configure_model(
            tokenizer=tokenizer, pre_process=pre_process, post_process=post_process
        )
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.model_type = ModelType.encoder_or_decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        # pylint: disable=C0116
        image_features = self.vision_model(images)
        text_features = self.text_model(captions)
        if self.post_process:
            return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1), self.logit_scale.exp()

        return image_features, text_features

    def set_input_tensor(self, tensor):
        # pylint: disable=C0116
        pass


class CLIPModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """
    CLIPModel is the base class for all CLIP models.

    Args:
        config: CLIPConfig. The configuration of the CLIP model. Please see the `CLIPConfig` for details.
        optim: OptimizerModule. This module is just used for init and the actual optimizer is created via trainer API.
        tokenizer: TokenizerSpec. This module is used for deciding the output length of the language model.

        # These parameters are just for imagenet validation
        imagenet_val: Optional[str] = None: Optional path to imagenet validation dataset.
        mbs: int = 8: Batch size for imagenet validation.
        gbs: int = 8: Global Batch for imagenet validation.
        max_workers: int = 4: Maximum number of workers used for imagenet validation.


    """

    def __init__(
        self,
        config: CLIPConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        imagenet_val: Optional[str] = None,
        mbs: int = 8,
        gbs: int = 8,
        max_workers: int = 4,
    ):
        # pylint: disable=C0116
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

        # These parameters are just for imagenet validation
        self.imagenet_val = imagenet_val
        self.mbs = mbs
        self.gbs = gbs
        self.max_workers = max_workers

    def on_fit_start(self):
        """Initialize the dataloader parameters for imagenet validation"""
        if self.imagenet_val is not None:
            self.imagenet_val = build_imagenet_validation_dataloader_params(
                self.imagenet_val,
                self.config.vision_transformer_config.img_h,
                self.config.vision_transformer_config.img_w,
                self.mbs,
                self.gbs,
                num_workers=self.max_workers,
                max_position_embedding=self.config.text_transformer_config.max_seq_length,
                tokenizer=self.tokenizer,
            )

    def configure_model(self) -> None:
        """Configure the model"""
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        # pylint: disable=C0116
        return self.module(images, captions)

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        # pylint: disable=C0116
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        # pylint: disable=C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        """In mcore the loss-function is part of the forward-pass (when labels are provided)"""
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        """In mcore the loss-function is part of the forward-pass (when labels are provided)"""
        return self.forward_step(batch)

    def zero_shot_classifier(self):
        """Zero shot classifier for imagenet validation"""
        text_encoder = self.module.module.module.text_model
        with torch.no_grad():
            zeroshot_weights = []
            for texts in self.imagenet_val["texts"]:
                texts = texts.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(
                    enabled=True,
                    dtype=torch.bfloat16,
                ):
                    class_embeddings = text_encoder(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights

    def zero_shot_eval(self):
        """Zero shot evaluation for imagenet validation"""

        def accuracy(output, target, topk=(1,)):
            pred = output.topk(max(topk), 1, True, True)[1].t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

        logging.info('Starting zero-shot imagenet.')

        logging.info('Building zero-shot classifier')
        classifier = self.zero_shot_classifier()

        logging.info('Using classifier')

        vision_encoder = self.module.module.module.vision_model

        with torch.no_grad():
            top1, top5, n = 0.0, 0.0, 0.0
            for images, target in tqdm(self.imagenet_val["images"], desc="Imagenet Zero-shot Evaluation", leave=False):
                if images is None or target is None:
                    continue

                images = images.cuda(non_blocking=True).to(torch.bfloat16)
                target = target.cuda(non_blocking=True)

                # predict
                with torch.cuda.amp.autocast(
                    enabled=True,
                    dtype=torch.bfloat16,
                ):

                    image_features = vision_encoder(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100.0 * image_features @ classifier

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        logging.info('Finished zero-shot imagenet.')
        top1 = top1 / n
        top5 = top5 / n
        return top1, top5

    def on_validation_epoch_end(self):
        """Run zero shot evaluation for imagenet validation"""
        if self.imagenet_val is not None:
            imagenet_metric = torch.zeros(2).cuda()
            imagenet_metric[0], imagenet_metric[1] = self.zero_shot_eval()
            imagenet_metric = average_losses_across_data_parallel_group(imagenet_metric)
            self.log('imagenet_top1', imagenet_metric[0], prog_bar=True, rank_zero_only=True, batch_size=1)
            self.log('imagenet_top5', imagenet_metric[1], prog_bar=True, rank_zero_only=True, batch_size=1)

    @property
    def training_loss_reduction(self) -> ClipMegatronLoss:
        # pylint: disable=C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = ClipMegatronLoss()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> ClipMegatronLoss:
        # pylint: disable=C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = ClipMegatronLoss()

        return self._validation_loss_reduction
