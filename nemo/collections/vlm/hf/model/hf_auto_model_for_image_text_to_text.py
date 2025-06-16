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

import _io
import lightning.pytorch as pl
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from nemo.automodel.dist_utils import FirstRankPerNode
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import masked_cross_entropy
from nemo.lightning import io
from nemo.utils import logging


class HFAutoModelForImageTextToText(pl.LightningModule, io.IOMixin, fn.FNMixin):
    """Wrap's HF's AutoModelForImageTextToText in a pl.LightningModule
    for use within NeMo"""

    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        processor=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        trust_remote_code=False,
        default_dtype=torch.bfloat16,
        load_in_4bit=False,
        freeze_language_model=False,
        freeze_vision_model=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._processor = processor
        self.tokenizer = None
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.trust_remote_code = trust_remote_code
        self.default_dtype = default_dtype
        self.load_in_4bit = load_in_4bit
        self.freeze_language_model = freeze_language_model
        self.freeze_vision_model = freeze_vision_model
        self.kwargs = kwargs

    @property
    def processor(self):
        """Return's module processor"""
        if self._processor is None:
            self._processor = HFAutoModelForImageTextToText.configure_processor(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        return self._processor

    @processor.setter
    def processor(self, value):
        """Set's module's processor"""
        assert self._processor is None
        self._processor = value

    @staticmethod
    def configure_processor(model_name, trust_remote_code=False):
        """Initializes an AutoProcessor and returns the instance"""
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    @FirstRankPerNode()
    def configure_model(self):
        """Instantiates the model"""
        # create all your layers here

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.default_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=self.default_dtype,
            )
        if self.load_pretrained_weights:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype='auto',
                trust_remote_code=self.trust_remote_code,
                quantization_config=quantization_config,
                **self.kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, 'torch_dtype', self.default_dtype)
            self.model = AutoModelForImageTextToText.from_config(
                config, torch_dtype=dtype, trust_remote_code=self.trust_remote_code
            )

        self.model.train()
        self.freeze_model()

        # Ugly hack for PEFT: adapters are added here so that can be wrapped correctly with DDP.
        if getattr(self, 'model_transform', None) is not None:
            self.model_transform(self)
            self.model_transform.__num_calls__ = 0

    def forward(self, batch):
        """Runs forward with the model"""
        return self.model(**batch)

    def training_step(self, batch, batch_idx=None):
        """Run one training step"""
        if isinstance(self.trainer.strategy.checkpoint_io, io.pl.MegatronCheckpointIO):
            logging.warning("Switching CheckpointIO from MegatronCheckpointIO to HFCheckpointIO.")
            self.trainer.strategy.checkpoint_io = self.make_checkpoint_io(self._has_lora_adapter)

        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        outputs = self.forward(batch)

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
        """Run one validation step"""
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        outputs = self.forward(**batch)

        logits = outputs.logits.float()
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)

        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = self.loss_fn(logits, labels, loss_mask)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

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
        if self._processor is not None:
            self._processor.save_pretrained(path)
        else:
            logging.warning("A processor wasn't created before to save.")

    @staticmethod
    def extract_skipped_token_ids(tokenizer):
        """Returns list of tokens to mask in labels"""
        # qweb2-2b
        QWEN_TOKENS = [
            '<|im_start|>',
            '<|im_end|>',
            '<|vision_start|>',
            '<|vision_end|>',
            '<|vision_pad|>',
            '<|image_pad|>',
            '<|video_pad|>',
            '<|im_start|>',
            '<|im_end|>',
            '<|vision_start|>',
            '<|vision_end|>',
            '<|vision_pad|>',
            '<|image_pad|>',
            '<|video_pad|>',
        ]
        # llava-1.5-7b-hf, llava-v1.6-mistral-7b-hf
        LLAVA_TOKENS = [
            "<image>",
            "<pad>",
        ]
        LLAMA_TOKENS = [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|finetune_right_pad_id|>',
            '<|step_id|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eom_id|>',
            '<|eot_id|>',
            '<|python_tag|>',
            '<|image|>',
        ]
        GEMMA_TOKENS = [
            '<image_soft_token>',
        ]
        PAD_TOKENS = set(QWEN_TOKENS + LLAVA_TOKENS + LLAMA_TOKENS + GEMMA_TOKENS)
        tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
        skipped_token_ids = []
        for key, val in tokenizer.added_tokens_decoder.items():
            if str(val) in PAD_TOKENS:
                skipped_token_ids.append(key)
        return torch.IntTensor(list(set(skipped_token_ids)))

    def freeze_model(self) -> None:
        '''
        Freezes the language and vision models
        '''
        modules = []

        # Search for language model, atmost one is allowed
        language_model = None
        for attr in dir(self.model):
            if attr.startswith('language') and isinstance(getattr(self.model, attr), torch.nn.Module):
                if language_model is not None:
                    raise ValueError(f"Found multiple language models: {language_model} and {attr}")
                language_model = getattr(self.model, attr)

        # Search for vision model, atmost one is allowed
        vision_model = None
        for attr in dir(self.model):
            if attr.startswith('vision') and isinstance(getattr(self.model, attr), torch.nn.Module):
                if vision_model is not None:
                    raise ValueError(f"Found multiple vision models: {vision_model} and {attr}")
                vision_model = getattr(self.model, attr)

        if self.freeze_language_model and language_model is not None:
            modules.append(language_model)
        if self.freeze_vision_model and vision_model is not None:
            modules.append(vision_model)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    @property
    def _has_lora_adapter(self):
        return any(map(lambda x: 'lora' in x[0].lower(), self.named_modules()))

    def make_checkpoint_io(self, adapter_only=False):
        """
        Creates a checkpoint_io object for this model;
        the main issue is passing self to the HFCheckpointIO, because it needs
        to call save_pretrained within.
        TODO(@akoumparouli): refactor ^
        """
        from nemo.lightning.io.hf import HFCheckpointIO

        return HFCheckpointIO(model=self, adapter_only=adapter_only)
