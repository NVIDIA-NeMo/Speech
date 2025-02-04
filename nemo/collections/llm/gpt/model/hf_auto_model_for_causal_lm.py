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
    """
    Compute the masked cross-entropy loss between logits and targets.

    If a mask is provided, the loss is computed per element, multiplied by the mask,
    and then averaged. If no mask is provided, the standard cross-entropy loss is used.

    Args:
        logits (torch.Tensor): The predicted logits with shape (N, C) where C is the number of classes.
        targets (torch.Tensor): The ground truth class indices with shape (N,).
        mask (torch.Tensor, optional): A tensor that masks the loss computation. Must be broadcastable
            to the shape of the loss. Defaults to None.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss * mask.view(-1))
    else:
        return F.cross_entropy(logits, targets)


class HFAutoModelForCausalLM(pl.LightningModule, io.IOMixin, fn.FNMixin):
    """
    A LightningModule wrapper for AutoModelForCausalLm.

    This module wraps around a LightningModule around a AutoModelForCausalLm (e.g., GPT-2).
    It provides functionalities for training, validation, and saving, and it supports model parallelization
    techniques including FSDP2 and tensor parallelism.

    Attributes:
        model_name (str): The name or path of the pretrained model.
        load_pretrained_weights (bool): Whether to load pretrained weights.
        loss_fn (callable): The loss function to use during training and validation.
        is_hf_model (bool): Flag indicating that the underlying model is from Hugging Face.
        model_transform (callable, optional): A function to transform the model after creation.
        model_accelerator (callable, optional): A function to apply additional accelerations or modifications.
        trust_remote_code (bool): Whether to trust remote code for model loading.
        default_dtype (torch.dtype): The default data type for model weights.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
        attn_implementation (str): The attention implementation to use.
        mp_policy (MixedPrecisionPolicy): Mixed precision policy for distributed training.
        parallelize_fn (callable, optional): Function for parallelizing the model.
    """

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
        """
        Initialize the HFAutoModelForCausalLM.

        Args:
            model_name (str, optional): The model name or path. Defaults to 'gpt2'.
            load_pretrained_weights (bool, optional): Whether to load pretrained weights. Defaults to True.
            tokenizer (AutoTokenizer, optional): A pre-configured tokenizer. Defaults to None.
            loss_fn (callable, optional): Loss function to use. Defaults to masked_cross_entropy.
            model_transform (callable, optional): Function to transform the model after creation. Defaults to None.
            model_accelerator (callable, optional): Function to accelerate or optimize the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code during model/tokenizer loading.
                Defaults to False.
            default_dtype (torch.dtype, optional): Default data type for the model. Defaults to torch.bfloat16.
            load_in_4bit (bool, optional): Whether to load the model in 4-bit precision. Defaults to False.
            attn_implementation (str, optional): Attention implementation to use. Defaults to "sdpa".
            param_dtype (torch.dtype, optional): Data type for model parameters in mixed precision.
                Defaults to torch.bfloat16.
            reduce_dtype (torch.dtype, optional): Data type for reduction operations in mixed precision.
                Defaults to torch.float32.
            output_dtype (torch.dtype, optional): Data type for model outputs in mixed precision.
                Defaults to None.
            cast_forward_inputs (bool, optional): Whether to cast forward inputs. Defaults to True.
            parallelize_fn (callable, optional): Function for parallelizing the model. Defaults to None.
        """
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
        """
        Get the tokenizer for the model.

        If the tokenizer is not already initialized, it will be created using the
        `configure_tokenizer` static method.

        Returns:
            AutoTokenizer: The tokenizer associated with the model.
        """
        if self._tokenizer is None:
            self._tokenizer = HFAutoModelForCausalLM.configure_tokenizer(self.model_name, self.trust_remote_code)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        Set the tokenizer for the model.

        Args:
            value (AutoTokenizer): The tokenizer to be used.
        """
        assert self._tokenizer is None
        self._tokenizer = value

    @staticmethod
    def configure_tokenizer(model_name, use_fast=True, trust_remote_code=False):
        """
        Configure and return a Hugging Face AutoTokenizer for the given model.

        Args:
            model_name (str): The name or path of the model.
            use_fast (bool, optional): Whether to use the fast tokenizer implementation. Defaults to True.
            trust_remote_code (bool, optional): Whether to trust remote code when loading the tokenizer.
                Defaults to False.

        Returns:
            AutoTokenizer: The instantiated tokenizer.
        """
        try:
            return AutoTokenizer(model_name, use_fast=use_fast, trust_remote_code=trust_remote_code)
        except:
            return AutoTokenizer(model_name, use_fast=not use_fast, trust_remote_code=trust_remote_code)

    def configure_model(self):
        """
        Configure and initialize the Hugging Face model.

        Depending on the `load_pretrained_weights` flag, this method either loads
        a pretrained model or creates a model from configuration. It also applies FSDP2 and
        tensor parallelization if a device mesh is provided, and applies any additional
        accelerator function if specified.

        Raises:
            Exception: If model configuration fails.
        """
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
        """
        Perform a forward pass of the model.

        Args:
            batch (dict): A dictionary of inputs that the model expects.

        Returns:
            ModelOutput: The output of the underlying Hugging Face model.
        """
        return self.model(**batch)

    def training_step(self, batch, batch_idx=None):
        """
        Execute a single training step.

        This method prepares the input batch by ensuring the required keys are present,
        performs a forward pass, reshapes the logits and labels appropriately, computes the loss
        using the defined loss function, and logs the training loss.

        Args:
            batch (dict): A dictionary containing the batch data, including 'labels' and optionally 'loss_mask'.
            batch_idx (int, optional): The index of the batch. Defaults to None.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
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
        """
        Execute a single validation step.

        This method is similar to `training_step` but without gradient computations.
        It processes the input batch, performs a forward pass, computes the loss,
        and logs the validation loss.

        Args:
            batch (dict): A dictionary containing the batch data, including 'labels' and optionally 'loss_mask'.
            batch_idx (int): The index of the batch.
        """
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
        """
        Save the pretrained model and tokenizer to a specified path.

        This method ensures that the model state is gathered (especially in distributed settings)
        and then saves the state dict and tokenizer. Only rank 0 or the appropriate FSDP module saves
        the files to avoid race conditions.

        Args:
            path (str): The directory path where the model and tokenizer should be saved.

        Raises:
            AssertionError: If the model has not been created prior to saving.
        """
        assert self.model is not None, "Model has to be created first."
        import torch.distributed as dist
        from nemo.lightning.pytorch.strategies.utils import to_cpu

        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        is_rank0 = not is_dist or (is_dist and rank == 0)

        if is_rank0 or type(self.model).__name__.startswith('FSDP'):
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
