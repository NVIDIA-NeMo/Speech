# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoModel, BartForConditionalGeneration, EncoderDecoderModel

from nemo.collections.common.metrics import Perplexity
from nemo.collections.nlp.data.text2sparql import Text2SparqlDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ["Text2SparqlModel"]


class Text2SparqlModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "decoder_input_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "labels": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # must assign tokenizers before init
        if cfg.language_model.pretrained_model_name:
            if cfg.language_model.pretrained_encoder_model_name or cfg.language_model.pretrained_decoder_model_name:
                raise ValueError(
                    "Must have either pretrained_model_name or both pretrained_encoder_model name and "
                    "pretrained_decoder_model_name."
                )
            # setup tokenizer
            self.encoder_tokenizer = self.setup_tokenizer(cfg.encoder_tokenizer)
            self.encoder_add_special_tokens = cfg.encoder_tokenizer.add_special_tokens

            # set decoder to encoder
            self.decoder_tokenizer = self.encoder_tokenizer
            self.decoder_add_special_tokens = self.encoder_add_special_tokens
        else:
            if not (
                cfg.language_model.pretrained_encoder_model_name and cfg.language_model.pretrained_decoder_model_name
            ):
                raise ValueError("Both encoder and decoder must be specified")

            # setup tokenizers
            self.encoder_tokenizer = self.setup_tokenizer(cfg.encoder_tokenizer)
            self.encoder_add_special_tokens = cfg.encoder_tokenizer.add_special_tokens

            self.decoder_tokenizer = self.setup_tokenizer(cfg.decoder_tokenizer)
            self.decoder_add_special_tokens = cfg.decoder_tokenizer.add_special_tokens

        if not self.encoder_tokenizer:
            raise TypeError("encoder_tokenizer failed to initialize")
        if not self.decoder_tokenizer:
            raise TypeError("decoder_tokenizer failed to initialize")

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # must assign modules after init
        if cfg.language_model.pretrained_model_name:
            # Setup end-to-end model
            if "bart" in cfg.language_model.pretrained_model_name:
                self.model = BartForConditionalGeneration.from_pretrained(cfg.language_model.pretrained_model_name)
            else:
                self.model = AutoModel.from_pretrained(cfg.language_model.pretrained_model_name)
        else:
            if not (
                cfg.language_model.pretrained_encoder_model_name and cfg.language_model.pretrained_decoder_model_name
            ):
                raise ValueError("Both encoder and decoder must be specified")

            # Setup encoder/decoder model
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder=cfg.language_model.pretrained_encoder_model_name,
                decoder=cfg.language_model.pretrained_decoder_model_name,
            )

        self.validation_perplexity = Perplexity()

        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=False,
        )
        return outputs

    @typecheck.disable_checks()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Wraps huggingface EncoderDecoder.generate()."""
        outputs = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.encoder_tokenizer.pad_id,
            bos_token_id=self.encoder_tokenizer.bos_id,
            eos_token_id=self.encoder_tokenizer.eos_id,
            decoder_start_token_id=self.decoder_tokenizer.bos_id,
            **self._cfg.generate,
        )
        return outputs

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`. Loss calculation from HuggingFace's BartForConditionalGeneration.
        """
        input_ids, input_mask, decoder_input_ids, labels = batch
        loss = self.forward(
            input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, labels=labels,
        )[0]

        tensorboard_logs = {"train_loss": loss, "lr": self._optimizer.param_groups[0]["lr"]}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`. Loss calculation from HuggingFace's BartForConditionalGeneration.
        """
        input_ids, input_mask, decoder_input_ids, labels = batch
        loss, logits = self.forward(
            input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, labels=labels,
        )[:2]

        self.validation_perplexity(logits=logits)

        tensorboard_logs = {"val_loss": loss}

        return {"val_loss": loss, "log": tensorboard_logs}

    def on_validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        perplexity = self.validation_perplexity.compute()
        tensorboard_logs = {"val_loss": avg_loss, "perplexity": perplexity}
        logging.info(f"evaluation perplexity {perplexity.item()}")
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    @typecheck.disable_checks()
    def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Lightning calls this inside the test loop with data from the test dataloader."""
        input_ids, input_mask, decoder_input_ids, labels = batch
        sequences = self.generate(input_ids=input_ids)
        return sequences

    @typecheck.disable_checks()
    def on_test_epoch_end(self, outputs: List[torch.Tensor]) -> Dict[str, List[str]]:
        """Called at the end of test to aggregate outputs and decode them."""
        texts = [self.encoder_tokenizer.ids_to_text(seq) for batch in outputs for seq in batch]
        self.test_output = [{"texts": texts}]
        return {"texts": texts}

    def setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model=cfg.tokenizer_model,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            vocab_file=cfg.vocab_file,
        )
        return tokenizer

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self.setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self.setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self.setup_dataloader_from_config(cfg=test_data_config)

    def setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = Text2SparqlDataset(
            filepath=cfg.filepath,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            encoder_add_special_tokens=self.encoder_add_special_tokens,
            decoder_add_special_tokens=self.decoder_add_special_tokens,
            max_seq_length=self._cfg.max_seq_length,
            num_samples=cfg.get("num_samples", -1),
            convert_labels=self._cfg.convert_labels,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self._cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
