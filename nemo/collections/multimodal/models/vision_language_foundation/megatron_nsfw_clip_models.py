# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import functools
import itertools
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.data.clip.clip_dataset import tokenize
from nemo.collections.multimodal.data.nsfw.nsfw_dataset import build_dataset
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    CLIPTextTransformer,
    CLIPVisionTransformer,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.parts.utils_funcs import get_last_rank, torch_dtype_from_precision
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches


class ContentFilteringModel(MegatronModule):
    """Clip based content filtering model for NSFW."""

    def __init__(self, model_cfg: DictConfig, model_parallel_config, padded_vocab_size: int, tokenizer: Optional):
        super(ContentFilteringModel, self).__init__()
        self.cfg = model_cfg
        self.config = model_parallel_config
        self.tokenizer = tokenizer

        self.concept_list = self._load_concept_list(model_cfg.concepts)
        self.concept_count = len(self.concept_list)

        self.vision_encoder = CLIPVisionTransformer(
            model_cfg.vision, model_parallel_config, pre_process=True, post_process=True
        )

        if "text" in model_cfg and model_cfg.text is not None:
            self.text_encoder = CLIPTextTransformer(
                model_cfg.text, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True
            )
        else:
            self.text_encoder = None

        self.mlp_similarity_model = nn.Sequential(
            nn.Linear(model_cfg.output_dim * 2, model_cfg.sim_hidden_dim),
            nn.ReLU(),
            nn.Linear(model_cfg.sim_hidden_dim, 1),
        )

        self.nn_classifier = nn.Sequential(
            nn.Linear(self.concept_count * 2 + model_cfg.output_dim, model_cfg.cls_hidden_dim),
            nn.ReLU(),
            nn.Linear(model_cfg.cls_hidden_dim, 1),
        )

        self.register_buffer("concepts", torch.zeros(self.concept_count, model_cfg.output_dim))

    def initialize_concept_embeddings(self, concepts: torch.Tensor):
        if self.text_encoder is None:
            return

        self.concepts.copy_(concepts.detach())
        del self.text_encoder
        self.text_encoder = None

    def forward(self, image: torch.Tensor, mlp_factor: float = 1.0, emb_factor: float = 1.0) -> torch.Tensor:
        """Perform model forward pass for given image and factor.
        While inferencing, factors should be equal to default value
        """

        with torch.no_grad():
            embedding = self.vision_encoder(image).detach()
        cos_similarity = self.cosine_similarity(embedding, self.concepts)
        mlp_similarity = self.mlp_similarity(embedding, self.concepts)

        features = torch.cat([cos_similarity, mlp_similarity * mlp_factor, embedding * emb_factor], dim=-1)

        return self.nn_classifier(features)

    def cosine_similarity(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between prediction tensor and target tensor
        Args:
            prediction: Tensor of shape [X, H] for prediction embedding
            target: Tensor of shape [Y, H] for target to compare
        Returns:
            Similarity matrix of shape [X, Y] and value range [-1, 1]
        """
        normalized_prediction = F.normalize(prediction)
        normalized_target = F.normalize(target)

        return torch.matmul(normalized_prediction, normalized_target.t())

    def mlp_similarity(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mlp based similarity between prediction tensor and target tensor
        Args:
            prediction: Tensor of shape [X, H] for prediction embedding
            target: Tensor of shape [Y, H] for target to compare
        Returns:
            Similarity matrix of shape [X, Y] and value range [-1, 1]
        """

        prediction, target = torch.broadcast_tensors(prediction.unsqueeze(1), target.unsqueeze(0))

        combined = torch.cat([prediction, target], dim=-1)

        return torch.tanh(self.mlp_similarity_model(combined).squeeze(-1))

    def set_input_tensor(self, input_tensor: torch.Tensor):
        pass

    def _load_concept_list(self, config: Union[str, List[str]]) -> List[str]:
        if isinstance(config, str):
            config = [config]

        result_list = []
        for concept_file in config:
            with open(concept_file, "r") as f:
                result_list += [x.strip() for x in f.readlines() if x.strip() != ""]

        return result_list


class MegatronContentFilteringModel(MegatronBaseModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super(MegatronContentFilteringModel, self).__init__(cfg, trainer)

        self.model = build_model(
            model_provider_func=self.model_provider_func,
            wrap_with_ddp=False,
            on_cpu=isinstance(self.trainer.accelerator, CPUAccelerator),
            virtual_pipeline_model_parallel_size=None,
        )
        self.model = self.model[0]

        self.megatron_amp_O2 = cfg.get("megatron_amp_O2", False)
        if self.megatron_amp_O2:
            if isinstance(self.model, list):
                self.model = [
                    Float16Module(config=self.model_parallel_config, module=x, precision=cfg.precision)
                    for x in self.model
                ]
            else:
                self.model = Float16Module(
                    config=self.model_parallel_config, module=self.model, precision=cfg.precision
                )

        self.autocast_dtype = torch_dtype_from_precision(self.trainer.precision)
        self.enable_autocast = (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16])

        self.init_consumed_samples = 0
        self.mlp_factor = 1.0
        self.emb_factor = 1.0

        self.validation_metrics = None

    def get_module_list(self):
        if isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def model_provider_func(self, pre_process, post_process):
        return ContentFilteringModel(self.cfg, self.model_parallel_config, self.padded_vocab_size, self.tokenizer)

    def forward(self, image: torch.Tensor, mlp_factor: float = 1.0, emb_factor: float = 1.0) -> torch.Tensor:
        return self.model(image, mlp_factor, emb_factor)

    def get_forward_output_and_loss_func(self, with_accuracy: bool = False):
        def loss_fn(prediction: torch.Tensor, target: torch.Tensor):
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            out_dict = {"loss": loss}

            if with_accuracy:
                accuracy_components = torch.stack(
                    [
                        ((prediction > 0) & (target == 1.0)).sum(),  # tp
                        ((prediction < 0) & (target == 0.0)).sum(),  # tn
                        ((prediction > 0) & (target == 0.0)).sum(),  # fp
                        ((prediction < 0) & (target == 1.0)).sum(),  # fn
                    ]
                )
                out_dict["accuracy"] = accuracy_components

            return loss, out_dict

        def forward_step(dataloader_iter, model):
            images, labels = next(dataloader_iter)

            if (
                parallel_state.get_pipeline_model_parallel_world_size() == 1
                or parallel_state.is_pipeline_first_stage()
            ):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:
                images, labels = None, None

            classification = model(images, mlp_factor=self.mlp_factor, emb_factor=self.emb_factor)

            return classification.squeeze(-1), functools.partial(loss_fn, target=labels.float())

        return forward_step

    def get_forward_embedding_func(self):
        def forward_step(dataloader_iter, model):
            concepts = next(dataloader_iter)
            concepts = tokenize(concepts, self.tokenizer, self.cfg.text.max_position_embeddings)
            return (model.text_encoder(concepts.cuda(non_blocking=True)), lambda x: (0.0, {"concepts": x}))

        return forward_step

    def fwd_bwd_step(self, dataloader_iter, batch_idx: int, forward_only: bool):
        fwd_bwd_function = get_forward_backward_func()
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(with_accuracy=forward_only),
            data_iterator=dataloader_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        metrics = None
        if losses_reduced_per_micro_batch:
            loss_mean = torch.stack([l["loss"] for l in losses_reduced_per_micro_batch]).mean()
            if forward_only:
                metrics = torch.stack([l["accuracy"] for l in losses_reduced_per_micro_batch]).sum(dim=0)
        else:
            loss_mean = 0.0

        return loss_mean, metrics

    def training_step(self, dataloader_iter, batch_idx):
        self._optimizer.zero_grad()

        loss_mean, _ = self.fwd_bwd_step(dataloader_iter, batch_idx, forward_only=False)

        if self.megatron_amp_O2:
            self._optimizer.allreduce_main_grads()
        else:
            self.allreduce_gradients()

        torch.distributed.broadcast(loss_mean, get_last_rank())
        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log("loss_scale", loss_scale, batch_size=1, prog_bar=True)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1, prog_bar=True)
        self.log('global_step', self.trainer.global_step + 1, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

        return loss_mean

    def validation_step(self, dataloader_iter, batch_idx):
        loss, metrics = self.fwd_bwd_step(dataloader_iter, batch_idx, forward_only=True)
        if self.validation_metrics is None:
            self.validation_metrics = metrics
        else:
            self.validation_metrics += metrics

        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        torch.distributed.all_reduce(self.validation_metrics, op=torch.distributed.ReduceOp.SUM)
        accuracy = (self.validation_metrics[0] + self.validation_metrics[1]) / self.validation_metrics.sum()
        self.validation_metrics = None

        averaged_metrics = 0
        if parallel_state.is_pipeline_last_stage():
            averaged_metrics = torch.stack(self.validation_step_outputs).mean()
            torch.distributed.broadcast(averaged_metrics, get_last_rank())
        self.log("val_loss", averaged_metrics, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log("accuracy", accuracy, prog_bar=True, rank_zero_only=True, batch_size=1)

        logging.info(f"Current evaluation accuracy: {accuracy}")

        return averaged_metrics

    def test_step(self, dataloader_iter, batch_idx):
        return self.validation_step(dataloader_iter, batch_idx)

    def backward(self, *args, **kwargs):
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        pass

    def on_fit_start(self):
        if self.model.text_encoder is not None:
            fwd_bwd_function = get_forward_backward_func()
            losses_reduced_per_micro_batch = fwd_bwd_function(
                forward_step_func=self.get_forward_embedding_func(),
                data_iterator=iter([self.model.concept_list]),
                model=self.model,
                num_microbatches=get_num_microbatches(),
                forward_only=True,
                seq_length=None,
                micro_batch_size=self.model.concept_count,
            )

            concepts = torch.cat([x["concepts"] for x in losses_reduced_per_micro_batch], dim=0)
            self.model.initialize_concept_embeddings(concepts)
        self._cfg["text"] = None

    def setup(self, stage):
        resume_checkpoint_path = self.trainer.ckpt_path
        self.init_consumed_samples = (
            self._extract_consumed_samples_from_ckpt(resume_checkpoint_path) if resume_checkpoint_path else 0
        )
        self.setup_training_data(self.cfg)
        self.setup_validation_data(self.cfg)

    def setup_training_data(self, cfg: DictConfig) -> None:
        logging.info("Setting up training dataset.")
        train_ds = build_dataset(cfg, self.compute_consumed_samples(0), is_train=True)

        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=True
        )

        self._train_dl = torch.utils.data.DataLoader(
            train_ds,
            sampler=sampler,
            batch_size=cfg.micro_batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=cfg.data.train.get("drop_last", True),
            persistent_workers=True if cfg.data.num_workers > 0 else False,
        )

    def setup_validation_data(self, cfg: DictConfig) -> None:
        logging.info("Setting up validation dataset.")
        val_ds = build_dataset(cfg, self.compute_consumed_samples(0), is_train=False)

        sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=True
        )

        self._validation_dl = torch.utils.data.DataLoader(
            val_ds,
            sampler=sampler,
            batch_size=cfg.micro_batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=cfg.data.validation.get("drop_last", True),
            persistent_workers=True if cfg.data.num_workers > 0 else False,
        )

    def parameters(self):
        return itertools.chain(self.model.mlp_similarity_model.parameters(), self.model.nn_classifier.parameters())

    def on_load_checkpoint(self, checkpoint) -> None:
        if "model.concepts" in checkpoint["state_dict"]:
            self.model.text_encoder = None

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None
