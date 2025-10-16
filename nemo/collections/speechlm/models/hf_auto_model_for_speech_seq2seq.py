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

import lightning.pytorch as pl
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import masked_cross_entropy
from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import fsdp2_strategy_parallelize
from nemo.utils import logging
from nemo.utils.import_utils import safe_import_from

MixedPrecisionPolicy, _ = safe_import_from(
    "torch.distributed.fsdp", "MixedPrecisionPolicy", fallback_module="torch.distributed._composable.fsdp"
)


class HFAutoModelForSpeechSeq2Seq(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        tokenizer=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        model_accelerator=None,
        trust_remote_code=False,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
        parallelize_fn=None,
    ):
        from nemo.utils.decorators import deprecated_warning

        deprecated_warning(
            old_method="Automodel on NVIDIA/NeMo",
            new_method="https://github.com/NVIDIA-NeMo/Automodel repo",
            wait_seconds=2,
        )
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._tokenizer = None
        self._processor = None
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.model_accelerator = model_accelerator
        self.trust_remote_code = trust_remote_code
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
            self._tokenizer = AutoTokenizer(
                self.model_name, include_special_tokens=True, trust_remote_code=self.trust_remote_code
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        assert self._tokenizer is None
        self._tokenizer = value

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        return self._processor

    @staticmethod
    def configure_tokenizer(model_name):
        return AutoProcessor.from_pretrained(model_name).tokenizer

    def configure_model(self, train=True):
        # create all your layers here
        if self.model is None:
            if self.load_pretrained_weights:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=True,
                )
            else:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
                self.model = AutoModelForSpeechSeq2Seq.from_config(config, trust_remote_code=self.trust_remote_code)

        # Apply FSDP2 and TP to the model
        if self.device_mesh is not None:
            if self.parallelize_fn is not None:
                self.parallelize_fn(self.model, device_mesh=self.device_mesh, mp_policy=self.mp_policy)
            else:
                fsdp2_strategy_parallelize(self.model, device_mesh=self.device_mesh, mp_policy=self.mp_policy)

        if train:
            self.model.train()

    def forward(self, input_features, decoder_input_ids, attention_mask=None):
        return self.model(
            input_features=input_features.to(self.model.device),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    def training_step(self, batch, batch_idx=None):
        outputs = self.forward(input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"])
        loss_mask = batch.get('loss_mask', None)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.model.device).view(-1)
        n_cls = outputs.logits.shape[-1]
        logits = outputs.logits.view(-1, n_cls)
        loss = self.loss_fn(logits, batch["labels"], loss_mask)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        output = self.forward(input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"])
        loss = output.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def save_pretrained(self, path):
        assert self.model is not None, "Model has to be created first."
        self.model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
        else:
            logging.warning("A tokenizer wasn't created before to save.")

        if self._processor is not None:
            self._processor.save_pretrained(path)
        else:
            logging.warning("A processor wasn't created before to save.")
