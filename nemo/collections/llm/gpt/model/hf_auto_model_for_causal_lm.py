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

import inspect
import time

import _io
import lightning.pytorch as pl
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from nemo.automodel.dist_utils import FirstRankPerNode
from nemo.automodel.loss import masked_cross_entropy
from nemo.automodel.loss.linear_ce import HAVE_LINEAR_LOSS_CE, fused_linear_cross_entropy
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.utils import logging
from nemo.utils.import_utils import safe_import


@torch.no_grad()
def count_tail_padding(labels, ignore_label=-100):
    """Counts the total number of padding token in the tail of labels

    e.g.
        labels = torch.tensor([
            [-100, 1, 1, -100, -100],   # 2 tail -100s
            [-100, -100, 2, 3, 4],      # 0 tail -100s
            [5, 6, -100, -100, -100],   # 3 tail -100s
        ])
        count_tail_padding will return 5. Please do note there's more than 5 ignore labels.
    Args:
        labels (torch.Tensor): the labels
        ignore_label (int, optional): ignore label index. Defaults to -100.

    Returns:
        int: total number of ignored tokens in the `labels` input.
    """

    # Flip along the last dimension (seq_len)
    flipped = labels.flip(dims=[1])
    tail_mask = flipped == ignore_label

    # Compute cumulative product to "break" on first non ignore_label
    prod_mask = torch.cumprod(tail_mask.int(), dim=1)

    # Count tail -100s by summing cumprod mask along the sequence dimension
    return prod_mask.view(-1).sum().item()


class HFAutoModelForCausalLM(pl.LightningModule, io.IOMixin, fn.FNMixin):
    """
    A LightningModule wrapper for AutoModelForCausalLm.

    This module wraps a LightningModule around a AutoModelForCausalLM.
    It provides functionalities for training, validation, and checkpoint saving.
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
        use_liger_kernel=False,
        enable_grad_ckpt=False,
        device_map="cpu",
        use_linear_ce_loss=True,
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
            use_liger_kernel (bool, optional): Enables custom kernels from the Liger-Kernel Library. Defaults to False.
            enable_grad_ckpt (bool, optional): Enables gradient checkpoints. Defaults to False.
            device_map (str, optional): Device map to use. Defaults to "cpu".
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._tokenizer = tokenizer
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
        self.use_liger_kernel = use_liger_kernel
        self.device_map = device_map
        # holds loss values until optim step.
        self.loss_buffer = []
        self.n_tok = 0
        self.timestamp = None
        self.enable_grad_ckpt = enable_grad_ckpt
        self.use_linear_ce_loss = use_linear_ce_loss

        if self.use_linear_ce_loss and not HAVE_LINEAR_LOSS_CE:
            logging.warning(
                "Dependency for linear CE loss is not available. \
                    Please refer to https://github.com/apple/ml-cross-entropy."
            )
            self.use_linear_ce_loss = False
        logging.info(f"use_linear_ce_loss: {self.use_linear_ce_loss}")

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
            self._tokenizer = HFAutoModelForCausalLM.configure_tokenizer(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
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

    def _configure_model(self, attn_implementation):
        """helper method; see also configure_model."""
        # create all your layers here
        auto_cls = AutoModelForCausalLM
        if self.use_liger_kernel:
            liger_kernel_trf, HAS_LIGER_KERNEL = safe_import('liger_kernel.transformers')
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
            else:
                auto_cls = liger_kernel_trf.AutoLigerKernelForCausalLM

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.default_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=self.default_dtype,
            )

        if self.load_pretrained_weights:
            m = auto_cls.from_pretrained(
                self.model_name,
                torch_dtype=self.default_dtype,
                device_map=None if self.load_in_4bit else self.device_map,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
            )
            return m
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, 'torch_dtype', self.default_dtype)
            return auto_cls.from_config(
                config,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )

    @FirstRankPerNode()
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
        try:
            self.model = self._configure_model(attn_implementation=self.attn_implementation)
            logging.info("Configuring model with attn_implementation:", self.attn_implementation)
        except ValueError as e:
            # 'does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention'
            if 'does not support an attention' in str(e):
                logging.warning("Falling back to 'eager' attention implementation.")
                self.model = self._configure_model(attn_implementation="eager")
            else:
                raise e
        if self.use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            try:
                _apply_liger_kernel_to_instance(model=self.model)
            except Exception as e:
                logging.warning("Liger failed with: {}. Switching to non-liger path.".format(e))
                self.use_liger_kernel = False
                del self.model
                return self.configure_model()

        if self.model_accelerator is not None:
            from nemo.lightning.pytorch.accelerate.transformer_engine import te_accelerate

            te_accelerate(self.model, self.model_accelerator.fp8_autocast)

        if self.enable_grad_ckpt:
            if getattr(self.model, 'supports_gradient_checkpointing', False):
                self.model.gradient_checkpointing_enable()
            else:
                # TODO(@akoumparouli): custom logic goes here, but for now just a warning
                logging.warning("Asked to use gradient checkpoint, but model does not support it")

        self.model.train()

        # Ugly hack for PEFT: adapters are added here so that can be wrapped correctly with DDP.
        if getattr(self, 'model_transform', None) is not None:
            self.model_transform(self)
            self.model_transform.__num_calls__ = 0

    def forward(self, batch, num_logits_to_keep=None):
        """
        Perform a forward pass of the model.

        Args:
            batch (dict): A dictionary of inputs that the model expects.
            num_logits_to_keep (int, optional): The number of logits to keep. 0 means all logits are kept.
        Returns:
            ModelOutput: The output of the underlying Hugging Face model.
        """
        if num_logits_to_keep is None:
            return self.model(**batch)
        # Check if num_logits_to_keep parameter exists in model's forward method
        model_forward_params = inspect.signature(self.model.forward).parameters
        if 'num_logits_to_keep' in model_forward_params:
            return self.model(**batch, num_logits_to_keep=num_logits_to_keep)
        if 'logits_to_keep' in model_forward_params:
            return self.model(**batch, logits_to_keep=num_logits_to_keep)
        return self.model(**batch)

    def training_step(self, batch, batch_idx=None, context_parallel=False):
        """
        Execute a single training step.

        This method prepares the input batch by ensuring the required keys are present,
        performs a forward pass, reshapes the logits and labels appropriately, computes the loss
        using the defined loss function, and logs the training loss.

        Args:
            batch (dict): A dictionary containing the batch data, including 'labels' and optionally 'loss_mask'.
            batch_idx (int, optional): The index of the batch. Defaults to None.
            context_parallel (bool, optional): Whether to use context parallelism. Defaults to False.
        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        # logging
        if self.timestamp is None:
            self.timestamp = time.perf_counter()

        if isinstance(self.trainer.strategy.checkpoint_io, io.pl.MegatronCheckpointIO):
            logging.warning("Switching CheckpointIO from MegatronCheckpointIO to HFCheckpointIO.")
            self.trainer.strategy.checkpoint_io = self.make_checkpoint_io(self._has_lora_adapter)

        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)
        # GPTSFTDataset emits `tokens` instead of `input_ids`
        if 'input_ids' not in batch and 'tokens' in batch:
            batch['input_ids'] = batch['tokens']

        # TODO(@boxiangw): Refractor. Needed for SP support
        # If 'position_ids' does not exist in batch already then override it. batch in case of Packed sequence
        # contains 'position_ids' and we don't want to override it.
        if 'position_ids' not in batch:
            batch["position_ids"] = torch.arange(0, batch['input_ids'].shape[1]).unsqueeze(0).to(self.model.device)

        batch = self._remove_extra_batch_keys(batch)
        # if attn_mask exists in the batch convert to float. For some reason although torch.bool when created,
        # inside training step it becomes torch.int64 which can lead to error during transformers sdpa call,
        # convert to float.
        if 'attention_mask' in batch:
            batch['attention_mask'] = batch['attention_mask'].float()

        # based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L336
        if context_parallel:

            from nemo.lightning.pytorch.strategies.utils import create_context_parallel_ctx, get_train_context

            input_ids = batch["input_ids"].to(self.model.device)
            batch["position_ids"] = torch.arange(0, input_ids.shape[1]).unsqueeze(0).to(self.model.device)
            position_ids = batch["position_ids"].to(self.model.device)

            context_parallel_ctx = create_context_parallel_ctx(
                cp_mesh=self._device_mesh["context_parallel"],
                cp_buffers=[input_ids, labels, position_ids, loss_mask],
                cp_seq_dims=[1, 1, 1, 1],
                cp_no_restore_buffers={input_ids, labels, loss_mask},
                cp_rotate_method="allgather",  # TODO add "alltoall" option
            )
            train_context = get_train_context(
                False,
                False,
            )
            with train_context(context_parallel_ctx):
                outputs = self.forward(batch)

                # Prepare for loss calculation
                logits = outputs.logits.float()
                n_cls = logits.shape[-1]
                logits = logits.view(-1, n_cls)
                labels = labels.view(-1)
                assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
                loss = self.loss_fn(logits, labels, loss_mask)
        else:
            batch["output_hidden_states"] = True if self.use_linear_ce_loss else False  # Enable hidden states output

            if not self.use_linear_ce_loss:
                outputs = self.forward(batch)
                # Prepare for loss calculation
                logits = outputs.logits
                n_cls = logits.shape[-1]
                logits = logits.view(-1, n_cls)
                labels = labels.view(-1)
                assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
                loss = self.loss_fn(logits, labels, loss_mask)
            else:
                # use num_logits_to_keep=1 to avoid full logits matrix in memory
                # TODO: test CE with CP enabled
                outputs = self.forward(batch, num_logits_to_keep=1)
                hidden_states = outputs.hidden_states[-1]
                lm_head = self.model.get_output_embeddings().weight  # Get the weight matrix
                if loss_mask is not None:
                    # Replace labels with -100 where mask is 0 (don't compute loss for these positions)
                    # -100 is the default ignore index in PyTorch's cross entropy loss
                    labels = labels.masked_fill(loss_mask == 0, -100)
                num_items_in_batch = torch.count_nonzero(labels != -100).item()
                logit_softcapping = 0
                loss = fused_linear_cross_entropy(
                    hidden_states=hidden_states,
                    lm_weight=lm_head.full_tensor() if hasattr(lm_head, 'full_tensor') else lm_head,
                    labels=labels,
                    num_items_in_batch=num_items_in_batch,
                    logit_softcapping=logit_softcapping,
                )
        self.loss_buffer.append(loss.item())
        self.n_tok += labels.numel() - count_tail_padding(labels.view_as(batch['input_ids']))

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Hook triggered befored the optimizer step.
        Used for calculating the average loss across all gradient accumulation steps.

        Args:
            optimizer (torch.optim.Optimizer): the optimizer; unused.
        """
        # Excluding the first/last iter, time_delta is calculated as follows
        # fwd 0 | opt 0 | fwd 1 | opt 1 | fwd 2
        #        ^              ^
        s = time.perf_counter()
        time_delta = s - self.timestamp
        self.timestamp = s

        mean_loss = sum(self.loss_buffer) / len(self.loss_buffer)
        self.loss_buffer = []
        tps = self.n_tok / time_delta
        self.n_tok = 0

        # reduce across ranks
        is_ddp = isinstance(self.trainer.strategy, pl.strategies.DDPStrategy)
        device_mesh = getattr(self, '_device_mesh', None)
        if device_mesh is not None or is_ddp:
            if is_ddp:
                group = dist.group.WORLD  # Default DDP process group
            else:
                # Use the flattened DP / CP device mesh for loss reduction
                # if it exists (CP > 1), else default to the data parallel mesh.
                group = device_mesh[
                    (
                        "dp_cp"
                        if device_mesh.mesh_dim_names is not None and "dp_cp" in device_mesh.mesh_dim_names
                        else "data_parallel"
                    )
                ].get_group()

            def reduce_item(val, op, device, group, dtype):
                """util function"""
                divide_by_world_size = False
                if torch.distributed.get_backend(group) == "gloo" and op == dist.ReduceOp.AVG:
                    # GLOO does not support the `ReduceOp.AVG` operation
                    op = dist.ReduceOp.SUM
                    divide_by_world_size = True

                val = torch.tensor([val], device=device, dtype=dtype).detach()
                dist.all_reduce(val, group=group, op=op)
                val = val.item()
                if divide_by_world_size:
                    val /= dist.get_world_size(group)
                return val

            # Reduce loss across DP (or DP x CP) ranks.
            mean_loss = reduce_item(
                mean_loss, op=dist.ReduceOp.AVG, device=self.device, group=group, dtype=torch.float32
            )
            tps = reduce_item(tps, op=dist.ReduceOp.SUM, device=self.device, group=group, dtype=torch.int64)

        # Log the reduced loss.
        self.log('reduced_train_loss', mean_loss, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=False)
        self.log('tps', tps, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=False)

        # log LR
        # TODO(akoumparouli): move this elsewhere.
        optim = self.optimizers()
        if isinstance(optim, list):
            optim = optim[0]
        self.log('lr', optim.param_groups[0]['lr'], prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=False)

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
        if 'input_ids' not in batch and 'tokens' in batch:
            batch['input_ids'] = batch['tokens']
        batch = self._remove_extra_batch_keys(batch)

        outputs = self.forward(batch)

        logits = outputs.logits
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)

        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = self.loss_fn(logits, labels, loss_mask)
        return loss

    def save_pretrained(self, path, state_dict):
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

        def pop_fqn_prefix(fqn, expected_prefix='model'):
            """pops prefix from FQN"""
            parts = fqn.split('.')
            assert parts[0] == expected_prefix
            return '.'.join(parts[1:])

        # Remove the "model." prefix from FQNs.
        # Context: calling state_dict on an HFAutoModelForCausalLM, will prepend "model." in the
        # state-dict keys. One solution would be to override HFAutoModelForCausalLM's state_dict
        # and `return self.model.state_dict()`, however FSDP2 uses FQNs to acecss modules, therefore
        # removing "model." at the state_dict function level will cause issues with FSDP2.
        keys = list(state_dict.keys())
        io_bytes_state = {}
        for key, new_key in map(lambda x: (x, pop_fqn_prefix(x)), keys):
            val = state_dict.pop(key)
            if isinstance(val, _io.BytesIO):
                io_bytes_state[new_key] = val
            else:
                state_dict[new_key] = val

        if len(io_bytes_state) > 0:
            logging.warning("State-dict contains _io.BytesIO, those will be saved separately to `io_bytes.pt`.")
            torch.save(io_bytes_state, path / 'io_bytes.pt')

        self.model.save_pretrained(path, state_dict=state_dict)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
        else:
            logging.warning("A tokenizer wasn't created before to save.")

    def load_pretrained(self, path):
        """
        Used from HFcheckpointio to load a checkpoint
        TODO(@akoumparouli): refactor
        """

        d = {
            "pretrained_model_name_or_path": path,
            "torch_dtype": torch.bfloat16,  # Always load in bfloat16 first
            "device_map": "cpu",
            "trust_remote_code": self.trust_remote_code,
            "attn_implementation": self.attn_implementation,
            "load_in_4bit": self.load_in_4bit,
        }

        if self.load_in_4bit:
            d["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )

        d["torch_dtype"] = torch.bfloat16
        return AutoModelForCausalLM.from_pretrained(**d).state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Loads the state-dict directly to self.model, therefore FQNs are expected
        not to start with "model." -- referring to HFAutoModelForCausalLM's attribute.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When set to ``False``, the properties of the tensors
                in the current module are preserved whereas setting it to ``True`` preserves
                properties of the Tensors in the state dict. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                for which the value from the module is preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.
        """
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def make_checkpoint_io(self, adapter_only=False):
        """
        Creates a checkpoint_io object for this model;
        the main issue is passing self to the HFCheckpointIO, because it needs
        to call save_pretrained within.
        TODO(@akoumparouli): refactor ^
        """
        from nemo.lightning.io.hf import HFCheckpointIO

        return HFCheckpointIO(model=self, adapter_only=adapter_only)

    def _remove_extra_batch_keys(self, batch, reserved_keys=['labels', 'loss_mask', 'input_ids']):
        """Remove extra keys from batch that are not kwargs in model's forward

        Args:
            batch (dict): dictionary of tensors.

        Returns:
            dict: dictionary of tensors; keys that are not in model's forward are removed.
        """
        fwd_signature = inspect.signature(self.model.forward)
        allowed_keys = list(fwd_signature.parameters.keys()) + reserved_keys
        return {k: batch[k] for k in allowed_keys if k in batch}

    @property
    def _has_lora_adapter(self):
        return any(map(lambda x: 'lora' in x[0].lower(), self.named_modules()))
