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

import json
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.utils.data as pt_data
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import SentenceDataset, TarredSentenceDataset
from nemo.collections.nlp.metrics import SequencePerplexity
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_transformer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging, model_utils

__all__ = ["TransformerLMModel"]


class TransformerLMModel(ModelPT):
    """
    Left-to-right Transformer language model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Instantiates tokenizer and register to be saved with NeMo Model archive
        # After this call, ther will be self.tokenizer which can convert between tokens and token_ids.
        self.setup_tokenizer(
            tokenizer_name=cfg.tokenizer.get("tokenizer_name", "sentencepiece"),
            tokenizer_model=cfg.tokenizer.get("tokenizer_model", None),
            vocab_file=cfg.tokenizer.get("vocab_file", None),
            bpe_dropout=cfg.tokenizer.get("bpe_dropout", 0.0),
            special_tokens=OmegaConf.to_container(cfg.tokenizer.special_tokens)
            if cfg.tokenizer.get("special_tokens", None)
            else None,
        )

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # make vocabulary size divisible by 8 for fast fp16 training
        vocab_size = 8 * math.ceil(self.tokenizer.vocab_size / 8)

        # encoder from NeMo, Megatron-LM, or HuggingFace
        encoder_cfg_dict = OmegaConf.to_container(cfg.get('encoder'))
        encoder_cfg_dict['vocab_size'] = vocab_size
        library = encoder_cfg_dict.pop('library', 'nemo')
        model_name = encoder_cfg_dict.pop('model_name', None)
        pretrained = encoder_cfg_dict.pop('pretrained', False)
        self.encoder = get_transformer(
            library=library,
            model_name=model_name,
            pretrained=pretrained,
            config_dict=encoder_cfg_dict,
            encoder=True,
            pre_ln_final_layer_norm=encoder_cfg_dict.get(
                'pre_ln_final_layer_norm', encoder_cfg_dict.get('pre_ln', True)
            ),
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self.encoder.hidden_size,
            num_classes=vocab_size,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.encoder.embedding.token_embedding.weight

        std_init_range = 1 / self.encoder.hidden_size ** 0.5

        # initialize weights if not using pretrained encoder
        if not self._cfg.encoder.get('pretrained', False):
            self.encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.loss_fn = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id, label_smoothing=cfg.label_smoothing)
        self.eval_loss_fn = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)
        self.eval_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)
        self.eval_ppl = SequencePerplexity()

    @typecheck()
    def forward(self, input_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """

        hidden_states = self.encoder(input_ids=input_ids, encoder_mask=attention_mask)
        log_probs = self.log_softmax(hidden_states=hidden_states)

        return log_probs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1
                # added by DataLoader is excess.
                batch[i] = batch[i].squeeze(dim=0)
        ids, mask = batch
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        input_mask = mask[:, :-1]
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)

        tensorboard_logs = {
            "train_loss": train_loss,
            "lr": self._optimizer.param_groups[0]["lr"],
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def eval_step(self, batch, batch_idx):
        mode = 'test' if self.trainer.testing else 'val'
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1
                # added by DataLoader is excess.
                batch[i] = batch[i].squeeze(dim=0)
        ids, mask = batch
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        input_mask, output_mask = mask[:, :-1], mask[:, 1:]
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)
        eval_loss = self.eval_loss_fn(log_probs=log_probs, labels=labels)

        self.eval_loss(loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1])
        self.eval_ppl(log_probs=log_probs, labels=labels, mask=output_mask)
        self.validation_step_outputs.append({}) if mode == 'val' else self.test_step_outputs.append({})
        return {}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx)

    def eval_epoch_end(self, outputs, mode):
        eval_loss = self.eval_loss.compute()
        eval_ppl = self.eval_ppl.compute()
        self.log(f"{mode}_loss", eval_loss, sync_dist=True)
        self.log(f"{mode}_PPL", eval_ppl, sync_dist=True)
        dataset_name = "Validation" if mode == 'val' else "Test"
        logging.info(f"\n\n\n\n{dataset_name} PPL: {np.round(eval_ppl.item(), 2)}")

    def on_validation_epoch_end(self):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.eval_epoch_end(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()  # free memory
        self.eval_loss.reset()
        self.eval_ppl.reset()

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()  # free memory

    def setup_tokenizer(
        self, tokenizer_name=None, tokenizer_model=None, vocab_file=None, bpe_dropout=0.0, special_tokens=None,
    ):

        supported_tokenizers = ['huggingface', 'sentencepiece', 'word']
        if tokenizer_name not in supported_tokenizers:
            raise NotImplementedError(f"Currently we only support tokenizers in {supported_tokenizers}.")

        self.tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            tokenizer_model=self.register_artifact("cfg.tokenizer.tokenizer_model", tokenizer_model),
            vocab_file=vocab_file,
            bpe_dropout=bpe_dropout,
            special_tokens=special_tokens,
            use_fast=False,
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'use_tarred_dataset' in train_data_config and train_data_config['use_tarred_dataset']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches * math.ceil(len(self._train_dl.dataset) / self.world_size)
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'use_tarred_dataset' in val_data_config and val_data_config['use_tarred_dataset']:
            # We also need to check if limit_val_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # validation batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches * math.ceil(len(self._validation_dl.dataset) / self.world_size)
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "validation batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig, predict_last_k=0):

        if cfg.get("use_tarred_dataset", False):
            if cfg.get("metadata_file") is None:
                raise FileNotFoundError("Trying to use tarred data set but could not find metadata path in config.")
            else:
                metadata_file = cfg.get('metadata_file')
                with open(metadata_file) as metadata_reader:
                    metadata = json.load(metadata_reader)
                if cfg.get('tar_files') is None:
                    tar_files = metadata.get('tar_files')
                    if tar_files is not None:
                        logging.info(f'Loading from tarred dataset {tar_files}')
                    else:
                        raise FileNotFoundError("Could not find tarred dataset in config or metadata.")
                else:
                    tar_files = cfg.get('tar_files')
                    if metadata.get('tar_files') is not None:
                        raise ValueError(
                            'Tar files specified in config and in metadata file. Tar files should only be specified once.'
                        )
            dataset = TarredSentenceDataset(
                text_tar_filepaths=tar_files,
                metadata_path=metadata_file,
                tokenizer=self.tokenizer,
                shuffle_n=cfg.get("tar_shuffle_n", 100),
                shard_strategy=cfg.get("shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
        else:
            dataset = SentenceDataset(
                tokenizer=self.tokenizer,
                dataset=cfg.file_name,
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                cache_ids=cfg.get("cache_ids", False),
            )
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
