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


import itertools

import fiddle as fdl
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.lightning.io.mixin import track_io
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform

DATA_PATH = '/home/TestData/lite/hf_cache/squad/'


def make_squad_hf_dataset(data_path, tokenizer):
    tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    def fmt(examples):
        prompt = """
    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
        instruction = examples["context"]
        input = examples["question"]
        output = examples["answers"]['text']
        if isinstance(output, list):
            output = output[0]
        text = prompt.format(instruction, input, output) + "<eos>"
        tokens = tokenizer.text_to_ids(text)
        return {'input_ids': tokens, 'labels': tokens}

    datamodule = llm.HFDatasetDataModule(data_path, split="train[:100]", pad_token_id=tokenizer.eos_id)

    datamodule.map(
        fmt,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )

    return datamodule


@track_io
class OrdTokenizer:
    def __init__(self, vocab_size=30_000, num_reserved_tokens=128, special_token_names=['bos_id', 'eos_id', 'pad_id']):
        self.vocab_size = vocab_size
        self.num_reserved_tokens = num_reserved_tokens
        self.special_token_names = special_token_names
        assert len(self.special_token_names) < num_reserved_tokens

    def __getattr__(self, name):
        if name in self.__dict__.get('special_token_names', {}):
            return self.__dict__['special_token_names'].index(name)
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError

    def text_to_ids(self, text):
        token_ids = list(map(lambda x: self.num_reserved_tokens + ord(x), list(text)))
        assert max(token_ids) < self.vocab_size
        return token_ids


def align_labels(logits, labels):
    logits = logits.float()
    n_cls = logits.shape[-1]
    if logits.shape[-2] == labels.shape[-1]:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
    elif logits.shape[-2] == labels.shape[-1] + 1:
        logits = logits[..., :-1, :].contiguous()
    else:
        raise ValueError("Mismatched labels and logits shapes (" + str(labels.shape) + " " + str(logits.shape))
    return logits.view(-1, n_cls), labels.view(-1)


class DummyJitModel(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        tokenizer=None,
        has_jit=False,
    ):
        super().__init__()
        self.has_jit = has_jit
        self.tokenizer = tokenizer

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = nn.Sequential(
                nn.Embedding(30_000, 512),
                nn.TransformerEncoderLayer(512, 8, 4096, dropout=0.1),
                nn.Linear(512, 30_000),
            )

    def forward(self, batch):
        output = self.module(**batch)
        if self.has_jit:
            assert self.module._compiled_call_impl is not None
            assert callable(self.module._compiled_call_impl)
        else:
            assert self.module._compiled_call_impl is None
        expected_cls = torch.nn.modules.container.Sequential
        assert isinstance(self.module, expected_cls), type(self.module)
        return output

    def training_step(self, batch):
        if self.has_jit:
            assert hasattr(self, '_compiled')
            assert self._compiled == True, self._compiled
        else:
            assert not hasattr(self, '_compiled')
        labels = batch.pop('labels')
        loss_mask = batch.get('loss_mask', None)
        output = self.forward({'input': batch['input_ids']})
        logits, labels = align_labels(output, labels)
        return F.cross_entropy(logits, labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default=1)
    parser.add_argument('--max-steps', type=int, default=1)
    args = parser.parse_args()

    tokenizer = OrdTokenizer()
    data = make_squad_hf_dataset(DATA_PATH, tokenizer)

    for use_torch, use_thunder in itertools.product([True, False], [False, False]):
        if use_torch and use_thunder:
            continue
        model = DummyJitModel(tokenizer=tokenizer, has_jit=use_torch | use_thunder)
        optim = fdl.build(llm.sgd.pytorch_sgd_with_flat_lr(lr=1e-5))

        jit_config = JitConfig(use_torch=use_torch, use_thunder=use_thunder)
        transform = JitTransform(jit_config)

        llm.api.finetune(
            model=model,
            data=data,
            trainer=nl.Trainer(
                devices=args.devices,
                max_steps=args.max_steps,
                accelerator='gpu',
                strategy='auto',
                log_every_n_steps=1,
                limit_val_batches=0.0,
                num_sanity_val_steps=0,
                accumulate_grad_batches=1,
                gradient_clip_val=1.0,
                use_distributed_sampler=False,
                callbacks=[transform],
            ),
            optim=optim,
            log=None,
        )
