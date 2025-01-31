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

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from transformers import AutoModelForCausalLM

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import fsdp2_strategy_parallelize
from nemo.utils import logging


def masked_cross_entropy(logits, targets, mask=None):
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss * mask.view(-1))
    else:
        return F.cross_entropy(logits, targets)


class HFAutoModelForCausalLM(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        tokenizer=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        model_accelerator=None,
        trust_remote_code=False,
        default_dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="sdpa",
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
        parallelize_fn=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._tokenizer = None
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.model_accelerator = model_accelerator
        self.trust_remote_code = trust_remote_code
        self.default_dtype = default_dtype
        self.load_in_4bit = load_in_4bit
        self.attn_implementation = attn_implementation
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
            cast_forward_inputs=cast_forward_inputs,
        )
        self.parallelize_fn = parallelize_fn

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = HFAutoModelForCausalLM.configure_tokenizer(self.model_name, self.trust_remote_code)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        assert self._tokenizer is None
        self._tokenizer = value

    @staticmethod
    def configure_tokenizer(model_name, use_fast=True, trust_remote_code=False):
        try:
            return AutoTokenizer(model_name, use_fast=use_fast, trust_remote_code=trust_remote_code)
        except:
            return AutoTokenizer(model_name, use_fast=not use_fast, trust_remote_code=trust_remote_code)

    def configure_model(self):
        # create all your layers here
        if self.load_pretrained_weights:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype='auto',
                device_map="cpu",
                trust_remote_code=self.trust_remote_code,
                load_in_4bit=self.load_in_4bit,
                attn_implementation=self.attn_implementation,
            )
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, 'torch_dtype', self.default_dtype)
            self.model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=self.attn_implementation,
            )

        # Apply FSDP2 and TP to the model
        if self.device_mesh is not None:
            if self.parallelize_fn is None:
                self.parallelize_fn = fsdp2_strategy_parallelize
            self.parallelize_fn(self.model, device_mesh=self.device_mesh, mp_policy=self.mp_policy)

        if self.model_accelerator is not None:
            self.model_accelerator(self.model)

        self.model.train()

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx=None):
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        # GPTSFTDataset emits `tokens` instead of `input_ids`
        if not 'input_ids' in batch and 'tokens' in batch:
            batch['input_ids'] = batch['tokens']
        batch = self._remove_extra_batch_keys(batch)

        outputs = self.forward(batch)

        # Prepare for loss calculation
        logits = outputs.logits.float()
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)

        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = self.loss_fn(logits, labels, loss_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        # GPTSFTDataset emits `tokens` instead of `input_ids`
        if not 'input_ids' in batch and 'tokens' in batch:
            batch['input_ids'] = batch['tokens']
        batch = self._remove_extra_batch_keys(batch)

        outputs = self.forward(**batch)

        logits = outputs.logits.float()
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)

        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = self.loss_fn(logits, labels, loss_mask)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def save_pretrained(self, path):
        assert self.model is not None, "Model has to be created first."
        import os

        import torch.distributed as dist
        from torch import Tensor
        from torch.distributed.tensor import DTensor

        is_dist = dist.is_initialized()
        is_rank0 = not is_dist or (is_dist and dist.get_rank() == 0)
        if is_rank0 or type(self.model).__name__.startswith('FSDP'):

            def to_cpu(v):
                if isinstance(v, DTensor):
                    return v.full_tensor().cpu()
                elif isinstance(v, Tensor):
                    return v.cpu()
                else:
                    return v

            cpu_state_dict = {k: to_cpu(v) for k, v in self.model.state_dict().items()}

        if is_rank0:
            self.model.save_pretrained(path, state_dict=cpu_state_dict)
            if self._tokenizer is not None:
                self._tokenizer.save_pretrained(path)
            else:
                logging.warning("A tokenizer wasn't created before to save.")

    def _remove_extra_batch_keys(self, batch, reserved_keys=['labels', 'loss_mask']):
        """Remove extra keys from batch that are not kwargs in model's forward

        Args:
            batch (dict): dictionary of tensors.

        Returns:
            dict: dictionary of tensors; keys that are not in model's forward are removed.
        """
        import inspect

        fwd_signature = inspect.signature(self.model.forward)
        allowed_keys = list(fwd_signature.parameters.keys()) + reserved_keys
        return {k: batch[k] for k in allowed_keys if k in batch}
