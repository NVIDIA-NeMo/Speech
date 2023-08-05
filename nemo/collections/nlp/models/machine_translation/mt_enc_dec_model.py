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

import itertools
import json
import os
import random
from collections import OrderedDict
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as pt_data
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.losses import NLLLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.tokenizers.bytelevel_tokenizers import ByteLevelProcessor
from nemo.collections.common.tokenizers.chinese_tokenizers import ChineseProcessor
from nemo.collections.common.tokenizers.en_ja_tokenizers import EnJaProcessor, JaMecabProcessor
from nemo.collections.common.tokenizers.indic_tokenizers import IndicProcessor
from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.data import TarredTranslationDataset, TranslationDataset
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_transformer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator, TopKSequenceGenerator
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging, model_utils, timers

__all__ = ['MTEncDecModel']


class MTEncDecModel(EncDecNLPModel, Exportable):
    """
    Encoder-decoder machine translation model.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = model_utils.maybe_update_config_version(cfg)

        self.src_language = cfg.get("src_language", None)
        self.tgt_language = cfg.get("tgt_language", None)

        self.multilingual = cfg.get("multilingual", False)
        self.multilingual_ids = []
        self.special_tokens = {}

        self.encoder_tokenizer_library = cfg.encoder_tokenizer.get('library', 'yttm')
        self.decoder_tokenizer_library = cfg.decoder_tokenizer.get('library', 'yttm')

        self.validate_input_ids = cfg.get("validate_input_ids", True)
        if self.multilingual:
            if isinstance(self.src_language, ListConfig) and isinstance(self.tgt_language, ListConfig):
                raise ValueError(
                    "cfg.src_language and cfg.tgt_language cannot both be lists. We only support many-to-one or one-to-many multilingual models."
                )
            elif isinstance(self.src_language, ListConfig):
                pass
            elif isinstance(self.tgt_language, ListConfig):
                for lng in self.tgt_language:
                    self.special_tokens["<" + lng + ">"] = "<" + lng + ">"
            else:
                raise ValueError(
                    "Expect either cfg.src_language or cfg.tgt_language to be a list when multilingual=True."
                )
        self.shared_embeddings = cfg.get("shared_embeddings", False)

        # Instantiates tokenizers and register to be saved with NeMo Model archive
        # After this call, ther will be self.encoder_tokenizer and self.decoder_tokenizer
        # Which can convert between tokens and token_ids for SRC and TGT languages correspondingly.

        encoder_tokenizer_model, decoder_tokenizer_model, encoder_vocab_file = None, None, None
        if cfg.encoder_tokenizer.get('tokenizer_model') is not None:
            encoder_tokenizer_model = self.register_artifact(
                "encoder_tokenizer.tokenizer_model", cfg.encoder_tokenizer.get('tokenizer_model')
            )

        if cfg.decoder_tokenizer.get('tokenizer_model') is not None:
            decoder_tokenizer_model = self.register_artifact(
                "decoder_tokenizer.tokenizer_model", cfg.decoder_tokenizer.get('tokenizer_model')
            )

        if cfg.encoder_tokenizer.get('vocab_file') is not None:
            encoder_vocab_file = (
                self.register_artifact("encoder_tokenizer.vocab_file", cfg.encoder_tokenizer.get('vocab_file')),
            )

        encoder_tokenizer, decoder_tokenizer = MTEncDecModel.setup_enc_dec_tokenizers(
            encoder_tokenizer_library=self.encoder_tokenizer_library,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0)
            if cfg.encoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            encoder_model_name=cfg.encoder.get('model_name') if hasattr(cfg.encoder, 'model_name') else None,
            encoder_r2l=cfg.encoder_tokenizer.get('r2l', False),
            decoder_tokenizer_library=self.decoder_tokenizer_library,
            encoder_tokenizer_vocab_file=encoder_vocab_file,
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0)
            if cfg.decoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            decoder_model_name=cfg.decoder.get('model_name') if hasattr(cfg.decoder, 'model_name') else None,
            decoder_r2l=cfg.decoder_tokenizer.get('r2l', False),
            special_tokens=self.special_tokens,
            encoder_sentencepiece_legacy=cfg.encoder_tokenizer.get('sentencepiece_legacy', False),
            decoder_sentencepiece_legacy=cfg.decoder_tokenizer.get('sentencepiece_legacy', False),
        )
        self.encoder_tokenizer, self.decoder_tokenizer = encoder_tokenizer, decoder_tokenizer

        if self.multilingual:
            (
                self.source_processor_list,
                self.target_processor_list,
                self.multilingual_lang_to_id,
            ) = MTEncDecModel.setup_multilingual_ids_and_processors(
                self.src_language,
                self.tgt_language,
                self.encoder_tokenizer,
                self.decoder_tokenizer,
                self.encoder_tokenizer_library,
                self.decoder_tokenizer_library,
            )
            self.multilingual_ids = list(self.multilingual_lang_to_id.values())
        else:
            # After this call, the model will have  self.source_processor and self.target_processor objects
            self.source_processor, self.target_processor = MTEncDecModel.setup_pre_and_post_processing_utils(
                self.src_language, self.tgt_language, self.encoder_tokenizer_library, self.decoder_tokenizer_library
            )
            self.multilingual_ids = [None]

        # TODO: Why is this base constructor call so late in the game?
        super().__init__(cfg=cfg, trainer=trainer)

        # encoder from NeMo, Megatron-LM, or HuggingFace
        encoder_cfg_dict = OmegaConf.to_container(cfg.get('encoder'))
        encoder_cfg_dict['vocab_size'] = self.encoder_vocab_size
        library = encoder_cfg_dict.pop('library', 'nemo')
        model_name = encoder_cfg_dict.pop('model_name', None)
        pretrained = encoder_cfg_dict.pop('pretrained', False)
        checkpoint_file = encoder_cfg_dict.pop('checkpoint_file', None)
        if isinstance(self.encoder_tokenizer, TabularTokenizer):
            # TabularTokenizer does not include a padding token, so this uses the prior default of 0.
            encoder_padding_idx = 0
        else:
            encoder_padding_idx = self.encoder_tokenizer.pad_id
        self.encoder = get_transformer(
            library=library,
            model_name=model_name,
            pretrained=pretrained,
            config_dict=encoder_cfg_dict,
            encoder=True,
            pre_ln_final_layer_norm=encoder_cfg_dict.get('pre_ln_final_layer_norm', False),
            checkpoint_file=checkpoint_file,
            padding_idx=encoder_padding_idx,
        )

        # decoder from NeMo, Megatron-LM, or HuggingFace
        decoder_cfg_dict = OmegaConf.to_container(cfg.get('decoder'))
        decoder_cfg_dict['vocab_size'] = self.decoder_vocab_size
        library = decoder_cfg_dict.pop('library', 'nemo')
        model_name = decoder_cfg_dict.pop('model_name', None)
        pretrained = decoder_cfg_dict.pop('pretrained', False)
        if isinstance(self.decoder_tokenizer, TabularTokenizer):
            # TabularTokenizer does not include a padding token, so this uses the prior default of 0.
            decoder_padding_idx = 0
        else:
            decoder_padding_idx = self.decoder_tokenizer.pad_id
        self.decoder = get_transformer(
            library=library,
            model_name=model_name,
            pretrained=pretrained,
            config_dict=decoder_cfg_dict,
            encoder=False,
            pre_ln_final_layer_norm=decoder_cfg_dict.get('pre_ln_final_layer_norm', False),
            padding_idx=decoder_padding_idx,
        )

        # validate hidden_size of encoder and decoder
        self._validate_encoder_decoder_hidden_size()

        self.log_softmax = TokenClassifier(
            hidden_size=self.decoder.hidden_size,
            num_classes=self.decoder_vocab_size,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.decoder.embedding,
            decoder=self.decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.decoder.max_sequence_length,
            beam_size=cfg.beam_size,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
            len_pen=cfg.len_pen,
            max_delta_length=cfg.max_generation_delta,
        )

        # tie embedding weights
        if self.shared_embeddings:
            if not cfg.get("shared_tokenizer", True):
                raise ValueError("shared_tokenizer cannot be False when shared_embeddings is True")

            # validate vocabulary size and embedding dimension
            if (
                self.encoder.embedding.token_embedding.weight.shape
                != self.decoder.embedding.token_embedding.weight.shape
            ):
                raise ValueError(
                    f"Cannot tie encoder and decoder embeddings due to mismatch in embedding sizes "
                    f"(num_embeddings, embedding_dim): {self.encoder.embedding.token_embedding.weight.shape} (encoder) "
                    f"{self.decoder.embedding.token_embedding.weight.shape} (decoder)"
                )

            self.encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.decoder.embedding.token_embedding.weight

        # TODO: encoder and decoder with different hidden size?
        std_init_range = 1 / self.encoder.hidden_size ** 0.5

        # initialize weights if not using pretrained encoder/decoder
        if not self._cfg.encoder.get('pretrained', False):
            self.encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        if not self._cfg.decoder.get('pretrained', False):
            self.decoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.decoder_tokenizer.pad_id, label_smoothing=cfg.label_smoothing
        )
        self.eval_loss_fn = NLLLoss(ignore_index=self.decoder_tokenizer.pad_id)

    @classmethod
    def setup_multilingual_ids_and_processors(
        cls,
        src_language,
        tgt_language,
        encoder_tokenizer,
        decoder_tokenizer,
        encoder_tokenizer_library,
        decoder_tokenizer_library,
    ):
        multilingual_ids = OrderedDict()

        # Determine all of the language IDs that need to be added as special tokens.
        if isinstance(src_language, ListConfig) and isinstance(tgt_language, ListConfig):
            assert len(src_language) == len(tgt_language)
            all_languages = list(set(tgt_language + src_language))
        elif isinstance(tgt_language, ListConfig):
            all_languages = tgt_language
        elif not isinstance(src_language, ListConfig) and not isinstance(tgt_language, ListConfig):
            all_languages = [src_language, tgt_language]
        else:
            all_languages = []

        # If target is a list config, then add all language ID tokens to the tokenizer.
        # When both src, tgt are lists, we concat and take a unique of all lang IDs.
        # If only tgt lang is a list, then we only add those lang IDs to the tokenizer.
        if all_languages != []:
            for lng in all_languages:
                if len(encoder_tokenizer.text_to_ids(f"<{lng}>")) != 1:
                    encoder_tokenizer.add_special_tokens({f"<{lng}>": f"<{lng}>"})
                if len(decoder_tokenizer.text_to_ids(f"<{lng}>")) != 1:
                    decoder_tokenizer.add_special_tokens({f"<{lng}>": f"<{lng}>"})
                # Make sure that we are adding the same language ID to both tokenizers. If this assert fails it means the tokenizers were different to begin with.
                assert encoder_tokenizer.text_to_ids(f"<{lng}>")[0] == decoder_tokenizer.text_to_ids(f"<{lng}>")[0]
                multilingual_ids[lng] = encoder_tokenizer.text_to_ids(f"<{lng}>")[0]

        if isinstance(src_language, ListConfig) and not isinstance(tgt_language, ListConfig):
            tgt_language = [tgt_language] * len(src_language)
        elif isinstance(tgt_language, ListConfig) and not isinstance(src_language, ListConfig):
            src_language = [src_language] * len(tgt_language)
        else:
            pass

        source_processor_list = []
        target_processor_list = []
        for src_lng, tgt_lng in zip(src_language, tgt_language):
            src_prcsr, tgt_prscr = MTEncDecModel.setup_pre_and_post_processing_utils(
                src_lng, tgt_lng, encoder_tokenizer_library, decoder_tokenizer_library
            )
            source_processor_list.append(src_prcsr)
            target_processor_list.append(tgt_prscr)

        return source_processor_list, target_processor_list, multilingual_ids

    def _validate_encoder_decoder_hidden_size(self):
        """
        Validate encoder and decoder hidden sizes, and enforce same size.
        Can be overridden by child classes to support encoder/decoder different
        hidden_size.
        """
        if self.encoder.hidden_size != self.decoder.hidden_size:
            raise ValueError(
                f"Class does not support encoder.hidden_size ({self.encoder.hidden_size}) != decoder.hidden_size ({self.decoder.hidden_size}). Please use bottleneck architecture instead (i.e., model.encoder.arch = 'seq2seq' in conf/aayn_bottleneck.yaml)"
            )

    @classmethod
    def filter_predicted_ids(cls, ids, decoder_tokenizer):
        ids[ids >= decoder_tokenizer.vocab_size] = decoder_tokenizer.unk_id
        return ids

    def test_encoder_ids(self, ids, raise_error=False):
        invalid_ids = torch.logical_or((ids >= self.encoder_tokenizer.vocab_size).any(), (ids < 0).any(),)

        if raise_error and invalid_ids:
            raise ValueError("Encoder ids are out of range (tip: check encoder tokenizer)")

        return not invalid_ids

    def test_decoder_ids(self, ids, raise_error=False):
        invalid_ids = torch.logical_or((ids >= self.decoder_tokenizer.vocab_size).any(), (ids < 0).any(),)

        if raise_error and invalid_ids:
            raise ValueError("Decoder ids are out of range (tip: check decoder tokenizer)")

        return not invalid_ids

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        if self.validate_input_ids:
            # test src/tgt for id range (i.e., hellp in catching wrong tokenizer)
            self.test_encoder_ids(src, raise_error=True)
            self.test_decoder_ids(tgt, raise_error=True)

        src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask)
        tgt_hiddens = self.decoder(
            input_ids=tgt, decoder_mask=tgt_mask, encoder_embeddings=src_hiddens, encoder_mask=src_mask
        )
        log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        return log_probs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
        }

        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode, dataloader_idx=0):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)

        if self.multilingual:
            self.source_processor = self.source_processor_list[dataloader_idx]
            self.target_processor = self.target_processor_list[dataloader_idx]

        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask)
        eval_loss = self.eval_loss_fn(log_probs=log_probs, labels=labels)
        # this will run encoder twice -- TODO: potentially fix
        inputs, translations = self.batch_translate(src=src_ids, src_mask=src_mask)
        if dataloader_idx == 0:
            getattr(self, f'{mode}_loss')(loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1])
        else:
            getattr(self, f'{mode}_loss_{dataloader_idx}')(
                loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1]
            )
        np_tgt = tgt_ids.detach().cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        ground_truths = [self.target_processor.detokenize(tgt.split(' ')) for tgt in ground_truths]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        return {
            'inputs': inputs,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.eval_step(batch, batch_idx, 'test', dataloader_idx)
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(loss)
        else:
            self.test_step_outputs.append(loss)

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        loss = self.eval_step(batch, batch_idx, 'val', dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss)
        else:
            self.validation_step_outputs.append(loss)

        return loss

    def eval_epoch_end(self, outputs, mode, global_rank):
        # if user specifies one validation dataloader, then PTL reverts to giving a list of dictionary instead of a list of list of dictionary
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        loss_list = []
        sb_score_list = []
        for dataloader_idx, output in enumerate(outputs):
            if dataloader_idx == 0:
                eval_loss = getattr(self, f'{mode}_loss').compute()
            else:
                eval_loss = getattr(self, f'{mode}_loss_{dataloader_idx}').compute()

            inputs = list(itertools.chain(*[x['inputs'] for x in output]))
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))
            assert len(translations) == len(inputs)
            assert len(translations) == len(ground_truths)

            # Gather translations and ground truths from all workers
            tr_and_gt = [None for _ in range(self.world_size)]
            # we also need to drop pairs where ground truth is an empty string
            dist.all_gather_object(
                tr_and_gt, [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']
            )
            if global_rank == 0:
                _translations = []
                _ground_truths = []
                for rank in range(0, self.world_size):
                    _translations += [t for (t, g) in tr_and_gt[rank]]
                    _ground_truths += [g for (t, g) in tr_and_gt[rank]]

                if self.multilingual and isinstance(self.tgt_language, ListConfig):
                    tgt_language = self.tgt_language[dataloader_idx]
                else:
                    tgt_language = self.tgt_language
                if tgt_language in ['ja', 'ja-mecab']:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="ja-mecab")
                elif tgt_language in ['zh']:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="zh")
                else:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")

                # because the reduction op later is average (over word_size)
                sb_score = sacre_bleu.score * self.world_size

                dataset_name = "Validation" if mode == 'val' else "Test"
                logging.info(
                    f"Dataset name: {dataset_name}, Dataloader index: {dataloader_idx}, Set size: {len(translations)}"
                )
                logging.info(
                    f"Dataset name: {dataset_name}, Dataloader index: {dataloader_idx}, Val Loss = {eval_loss}"
                )
                logging.info(
                    f"Dataset name: {dataset_name}, Dataloader index: {dataloader_idx}, Sacre BLEU = {sb_score / self.world_size}"
                )
                logging.info(
                    f"Dataset name: {dataset_name}, Dataloader index: {dataloader_idx}, Translation Examples:"
                )
                for i in range(0, 3):
                    ind = random.randint(0, len(translations) - 1)
                    logging.info("    " + '\u0332'.join(f"Example {i}:"))
                    logging.info(f"    Input:        {inputs[ind]}")
                    logging.info(f"    Prediction:   {translations[ind]}")
                    logging.info(f"    Ground Truth: {ground_truths[ind]}")
            else:
                sb_score = 0.0

            loss_list.append(eval_loss.cpu().numpy())
            sb_score_list.append(sb_score)
            if dataloader_idx == 0:
                self.log(f"{mode}_loss", eval_loss, sync_dist=True)
                self.log(f"{mode}_sacreBLEU", sb_score, sync_dist=True)
                getattr(self, f'{mode}_loss').reset()
            else:
                self.log(f"{mode}_loss_dl_index_{dataloader_idx}", eval_loss, sync_dist=True)
                self.log(f"{mode}_sacreBLEU_dl_index_{dataloader_idx}", sb_score, sync_dist=True)
                getattr(self, f'{mode}_loss_{dataloader_idx}').reset()
            outputs[dataloader_idx].clear()  # free memory

        if len(loss_list) > 1:
            self.log(f"{mode}_loss_avg", np.mean(loss_list), sync_dist=True)
            self.log(f"{mode}_sacreBLEU_avg", np.mean(sb_score_list), sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.eval_epoch_end(self.validation_step_outputs, 'val', self.global_rank)

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.test_step_outputs, 'test', self.global_rank)

    @classmethod
    def setup_enc_dec_tokenizers(
        cls,
        encoder_tokenizer_library=None,
        encoder_tokenizer_model=None,
        encoder_bpe_dropout=0.0,
        encoder_model_name=None,
        encoder_r2l=False,
        encoder_tokenizer_vocab_file=None,
        decoder_tokenizer_library=None,
        decoder_tokenizer_model=None,
        decoder_bpe_dropout=0.0,
        decoder_model_name=None,
        decoder_r2l=False,
        encoder_sentencepiece_legacy=False,
        decoder_sentencepiece_legacy=False,
        special_tokens={},
    ):

        supported_tokenizers = ['yttm', 'huggingface', 'sentencepiece', 'megatron', 'byte-level']
        if (
            encoder_tokenizer_library not in supported_tokenizers
            or decoder_tokenizer_library not in supported_tokenizers
        ):
            raise NotImplementedError(f"Currently we only support tokenizers in {supported_tokenizers}.")

        encoder_tokenizer = get_nmt_tokenizer(
            library=encoder_tokenizer_library,
            tokenizer_model=encoder_tokenizer_model,
            bpe_dropout=encoder_bpe_dropout,
            model_name=encoder_model_name,
            vocab_file=encoder_tokenizer_vocab_file,
            special_tokens=special_tokens,
            use_fast=False,
            r2l=encoder_r2l,
            legacy=encoder_sentencepiece_legacy,
        )

        decoder_tokenizer = get_nmt_tokenizer(
            library=decoder_tokenizer_library,
            tokenizer_model=decoder_tokenizer_model,
            bpe_dropout=decoder_bpe_dropout,
            model_name=decoder_model_name,
            vocab_file=None,
            special_tokens=special_tokens,
            use_fast=False,
            r2l=decoder_r2l,
            legacy=decoder_sentencepiece_legacy,
        )

        # validate no token is negative for sentencepiece tokenizers
        for tok_name, tok_library, tok_model, legacy in [
            ("encoder_tokenizer", encoder_tokenizer_library, encoder_tokenizer, encoder_sentencepiece_legacy),
            ("decoder_tokenizer", decoder_tokenizer_library, decoder_tokenizer, decoder_sentencepiece_legacy),
        ]:
            if tok_library == 'sentencepiece':
                negative_tokens = []
                for n in ["eos_id", "bos_id", "unk_id", "pad_id"]:
                    v = getattr(tok_model.tokenizer, n)()
                    if v < 0:
                        negative_tokens.append(f"{n}={v}")
                if negative_tokens and not legacy:
                    raise ValueError(
                        f"{tok_name}=sentencepiece has invalid negative special tokens = {negative_tokens}"
                    )
                # If using the legacy sentencepiece tokenizer, we can add the missing tokens as "special" tokens.
                else:
                    # If using sentencepiece legacy, eos, bos and pad need to be set/added differently.
                    if legacy:
                        # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
                        if not hasattr(tok_model, 'pad_token'):
                            if hasattr(tok_model.tokenizer, 'pad_id') and tok_model.tokenizer.pad_id() > 0:
                                tok_model.pad_token = tok_model.tokenizer.id_to_piece(tok_model.tokenizer.pad_id())
                            else:
                                tok_model.add_special_tokens({'pad_token': '<pad>'})
                        else:
                            tok_model.add_special_tokens({'pad_token': '<pad>'})

                        if not hasattr(tok_model, 'bos_token'):
                            if hasattr(tok_model.tokenizer, 'bos_id') and tok_model.tokenizer.bos_id() > 0:
                                tok_model.bos_token = tok_model.tokenizer.id_to_piece(tok_model.tokenizer.bos_id())
                            else:
                                tok_model.add_special_tokens({'bos_token': '<bos>'})
                        else:
                            tok_model.add_special_tokens({'bos_token': '<s>'})

                        if not hasattr(tok_model, 'eos_token'):
                            if hasattr(tok_model.tokenizer, 'eos_id') and tok_model.tokenizer.eos_id() > 0:
                                tok_model.eos_token = tok_model.tokenizer.id_to_piece(tok_model.tokenizer.eos_id())
                            else:
                                tok_model.add_special_tokens({'eos_token': '<eos>'})
                        else:
                            tok_model.add_special_tokens({'eos_token': '</s>'})

        return encoder_tokenizer, decoder_tokenizer

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_ds = MTEncDecModel._setup_dataset_from_config(
            cfg=train_data_config,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            global_rank=self.global_rank,
            world_size=self.world_size,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
        )
        self._train_dl = MTEncDecModel._setup_dataloader_from_config(cfg=train_data_config, dataset=self._train_ds,)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'use_tarred_dataset' in train_data_config and train_data_config['use_tarred_dataset']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches * ceil(len(self._train_dl.dataset) / self.world_size)
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.setup_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.setup_test_data(test_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_ds = MTEncDecModel._setup_eval_dataset_from_config(
            cfg=val_data_config,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
        )
        self._validation_dl = MTEncDecModel._setup_eval_dataloader_from_config(
            cfg=val_data_config, datasets=self._validation_ds
        )

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'use_tarred_dataset' in val_data_config and val_data_config['use_tarred_dataset']:
            # We also need to check if limit_val_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # validation batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches * ceil(len(self._validation_dl.dataset) / self.world_size)
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "validation batches will be used. Please set the trainer and rebuild the dataset."
                )

        # instantiate Torchmetric for each val dataloader
        if self._validation_dl is not None:
            for dataloader_idx in range(len(self._validation_dl)):
                if dataloader_idx == 0:
                    setattr(
                        self, f'val_loss', GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )
                else:
                    setattr(
                        self,
                        f'val_loss_{dataloader_idx}',
                        GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_ds = MTEncDecModel._setup_eval_dataset_from_config(
            cfg=test_data_config,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
        )
        self._test_dl = MTEncDecModel._setup_eval_dataloader_from_config(cfg=test_data_config, datasets=self._test_ds)
        # instantiate Torchmetric for each test dataloader
        if self._test_dl is not None:
            for dataloader_idx in range(len(self._test_dl)):
                if dataloader_idx == 0:
                    setattr(
                        self, f'test_loss', GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )
                else:
                    setattr(
                        self,
                        f'test_loss_{dataloader_idx}',
                        GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )

    @classmethod
    def _setup_dataset_from_config(
        cls,
        cfg: DictConfig,
        encoder_tokenizer,
        decoder_tokenizer,
        global_rank,
        world_size,
        multilingual,
        multilingual_ids,
    ):
        if cfg.get("use_tarred_dataset", False) or cfg.get("dataset_type", "") == "tarred":
            if cfg.get("metadata_file") is None:
                raise FileNotFoundError("Trying to use tarred data set but could not find metadata path in config.")
            metadata_file_list = cfg.get('metadata_file')
            tar_files_list = cfg.get('tar_files', None)
            if isinstance(metadata_file_list, str):
                metadata_file_list = [metadata_file_list]
            if tar_files_list is not None and isinstance(tar_files_list, str):
                tar_files_list = [tar_files_list]
            if tar_files_list is not None and len(tar_files_list) != len(metadata_file_list):
                raise ValueError('The config must have the same number of tarfile paths and metadata file paths.')

            datasets = []
            for idx, metadata_file in enumerate(metadata_file_list):
                with open(metadata_file) as metadata_reader:
                    metadata = json.load(metadata_reader)
                if tar_files_list is None:
                    tar_files = metadata.get('tar_files')
                    if tar_files is not None:
                        # update absolute path of tar files based on metadata_file path
                        valid_tar_files = []
                        metadata_basedir = os.path.abspath(os.path.dirname(metadata_file))
                        updated_fn = 0
                        for fn in tar_files:
                            # if a file does not exist, look in metadata file directory
                            if os.path.exists(fn):
                                valid_fn = fn
                            else:
                                updated_fn += 1
                                valid_fn = os.path.join(metadata_basedir, os.path.basename(fn))
                                if not os.path.exists(valid_fn):
                                    raise RuntimeError(
                                        f"File in tarred dataset is missing from absolute and relative paths {fn}"
                                    )

                            valid_tar_files.append(valid_fn)

                        tar_files = valid_tar_files

                        logging.info(f'Updated the path of {updated_fn} tarred files')
                        logging.info(f'Loading from tarred dataset {tar_files}')
                else:
                    tar_files = tar_files_list[idx]
                    if metadata.get('tar_files') is not None:
                        logging.info(
                            f'Tar file paths found in both cfg and metadata using one in cfg by default - {tar_files}'
                        )

                dataset = TarredTranslationDataset(
                    text_tar_filepaths=tar_files,
                    metadata_path=metadata_file,
                    encoder_tokenizer=encoder_tokenizer,
                    decoder_tokenizer=decoder_tokenizer,
                    shuffle_n=cfg.get("tar_shuffle_n", 100),
                    shard_strategy=cfg.get("shard_strategy", "scatter"),
                    global_rank=global_rank,
                    world_size=world_size,
                    reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                    prepend_id=multilingual_ids[idx] if multilingual else None,
                )
                datasets.append(dataset)

            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    sampling_technique=cfg.get('concat_sampling_technique'),
                    sampling_temperature=cfg.get('concat_sampling_temperature'),
                    sampling_probabilities=cfg.get('concat_sampling_probabilities'),
                    global_rank=global_rank,
                    world_size=world_size,
                )
            else:
                dataset = datasets[0]
        else:
            src_file_list = cfg.src_file_name
            tgt_file_list = cfg.tgt_file_name
            if isinstance(src_file_list, str):
                src_file_list = [src_file_list]
            if isinstance(tgt_file_list, str):
                tgt_file_list = [tgt_file_list]

            if len(src_file_list) != len(tgt_file_list):
                raise ValueError('The same number of filepaths must be passed in for source and target.')

            datasets = []
            for idx, src_file in enumerate(src_file_list):
                dataset = TranslationDataset(
                    dataset_src=str(Path(src_file).expanduser()),
                    dataset_tgt=str(Path(tgt_file_list[idx]).expanduser()),
                    tokens_in_batch=cfg.tokens_in_batch,
                    clean=cfg.get("clean", False),
                    max_seq_length=cfg.get("max_seq_length", 512),
                    min_seq_length=cfg.get("min_seq_length", 1),
                    max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                    max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                    cache_ids=cfg.get("cache_ids", False),
                    cache_data_per_node=cfg.get("cache_data_per_node", False),
                    use_cache=cfg.get("use_cache", False),
                    reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                    prepend_id=multilingual_ids[idx] if multilingual else None,
                )
                dataset.batchify(encoder_tokenizer, decoder_tokenizer)
                datasets.append(dataset)

            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    shuffle=cfg.get('shuffle'),
                    sampling_technique=cfg.get('concat_sampling_technique'),
                    sampling_temperature=cfg.get('concat_sampling_temperature'),
                    sampling_probabilities=cfg.get('concat_sampling_probabilities'),
                    global_rank=global_rank,
                    world_size=world_size,
                )
            else:
                dataset = datasets[0]

        return dataset

    @classmethod
    def _setup_dataloader_from_config(cls, cfg, dataset):
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=None
            if (
                cfg.get("use_tarred_dataset", False)
                or cfg.get("dataset_type", "") == "tarred"
                or isinstance(dataset, ConcatDataset)
            )
            else sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    def replace_beam_with_sampling(self, topk=500):
        self.beam_search = TopKSequenceGenerator(
            embedding=self.decoder.embedding,
            decoder=self.decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.beam_search.max_seq_length,
            beam_size=topk,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
        )

    @classmethod
    def _setup_eval_dataset_from_config(
        cls,
        cfg: DictConfig,
        multilingual: bool,
        multilingual_ids,
        encoder_tokenizer,
        decoder_tokenizer,
        add_bos_eos_to_encoder=True,
    ):
        src_file_name = cfg.get('src_file_name')
        tgt_file_name = cfg.get('tgt_file_name')

        if src_file_name is None or tgt_file_name is None:
            raise ValueError(
                'Validation dataloader needs both cfg.src_file_name and cfg.tgt_file_name to not be None.'
            )
        else:
            # convert src_file_name and tgt_file_name to list of strings
            if isinstance(src_file_name, str):
                src_file_list = [src_file_name]
            elif isinstance(src_file_name, ListConfig):
                src_file_list = src_file_name
            else:
                raise ValueError("cfg.src_file_name must be string or list of strings")
            if isinstance(tgt_file_name, str):
                tgt_file_list = [tgt_file_name]
            elif isinstance(tgt_file_name, ListConfig):
                tgt_file_list = tgt_file_name
            else:
                raise ValueError("cfg.tgt_file_name must be string or list of strings")
        if len(src_file_list) != len(tgt_file_list):
            raise ValueError('The same number of filepaths must be passed in for source and target validation.')

        datasets = []
        prepend_idx = 0
        for idx, src_file in enumerate(src_file_list):
            if multilingual:
                prepend_idx = idx
            dataset = TranslationDataset(
                dataset_src=str(Path(src_file).expanduser()),
                dataset_tgt=str(Path(tgt_file_list[idx]).expanduser()),
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                cache_ids=cfg.get("cache_ids", False),
                cache_data_per_node=cfg.get("cache_data_per_node", False),
                use_cache=cfg.get("use_cache", False),
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                prepend_id=multilingual_ids[prepend_idx] if multilingual else None,
                add_bos_eos_to_encoder=add_bos_eos_to_encoder,
            )
            dataset.batchify(encoder_tokenizer, decoder_tokenizer)
            datasets.append(dataset)

        return datasets

    @classmethod
    def _setup_eval_dataloader_from_config(cls, cfg, datasets):
        dataloaders = []
        for dataset in datasets:
            if cfg.shuffle:
                sampler = pt_data.RandomSampler(dataset)
            else:
                sampler = pt_data.SequentialSampler(dataset)

            dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    sampler=None
                    if (cfg.get("use_tarred_dataset", False) or isinstance(datasets[0], ConcatDataset))
                    else sampler,
                    num_workers=cfg.get("num_workers", 2),
                    pin_memory=cfg.get("pin_memory", False),
                    drop_last=cfg.get("drop_last", False),
                )
            )

        return dataloaders

    @classmethod
    def setup_pre_and_post_processing_utils(
        cls, source_lang, target_lang, encoder_tokenizer_library, decoder_tokenizer_library
    ):
        """
        Creates source and target processor objects for input and output pre/post-processing.
        """
        source_processor, target_processor = None, None

        if encoder_tokenizer_library == 'byte-level':
            source_processor = ByteLevelProcessor()
        elif (source_lang == 'en' and target_lang == 'ja') or (source_lang == 'ja' and target_lang == 'en'):
            source_processor = EnJaProcessor(source_lang)
        elif source_lang == 'ja-mecab':
            source_processor = JaMecabProcessor()
        elif source_lang == 'zh':
            source_processor = ChineseProcessor()
        elif source_lang == 'hi':
            source_processor = IndicProcessor(source_lang)
        elif source_lang == 'ignore':
            source_processor = None
        elif source_lang is not None and source_lang not in ['ja', 'zh', 'hi']:
            source_processor = MosesProcessor(source_lang)

        if decoder_tokenizer_library == 'byte-level':
            target_processor = ByteLevelProcessor()
        elif (source_lang == 'en' and target_lang == 'ja') or (source_lang == 'ja' and target_lang == 'en'):
            target_processor = EnJaProcessor(target_lang)
        elif target_lang == 'ja-mecab':
            target_processor = JaMecabProcessor()
        elif target_lang == 'zh':
            target_processor = ChineseProcessor()
        elif target_lang == 'hi':
            target_processor = IndicProcessor(target_lang)
        elif target_lang == 'ignore':
            target_processor = None
        elif target_lang is not None and target_lang not in ['ja', 'zh', 'hi']:
            target_processor = MosesProcessor(target_lang)

        return source_processor, target_processor

    @classmethod
    def ids_to_postprocessed_text(cls, beam_ids, tokenizer, processor, filter_beam_ids=True):
        if filter_beam_ids:
            beam_ids = MTEncDecModel.filter_predicted_ids(beam_ids, decoder_tokenizer=tokenizer)
        translations = [tokenizer.ids_to_text(tr) for tr in beam_ids.cpu().numpy()]
        if processor is not None:
            translations = [processor.detokenize(translation.split(' ')) for translation in translations]
        return translations

    @torch.no_grad()
    def batch_translate(
        self, src: torch.LongTensor, src_mask: torch.LongTensor, return_beam_scores: bool = False, cache={}
    ):
        """
        Translates a minibatch of inputs from source language to target language.
        Args:
            src: minibatch of inputs in the src language (batch x seq_len)
            src_mask: mask tensor indicating elements to be ignored (batch x seq_len)
        Returns:
            translations: a list strings containing detokenized translations
            inputs: a list of string containing detokenized inputs
        """
        mode = self.training
        timer = cache.get("timer", None)
        try:
            self.eval()
            if timer is not None:
                timer.start("encoder")
            src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask)
            if timer is not None:
                timer.stop("encoder")
                timer.start("sampler")
            best_translations = self.beam_search(
                encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask, return_beam_scores=return_beam_scores
            )
            if timer is not None:
                timer.stop("sampler")
            if return_beam_scores:
                all_translations, scores, best_translations = best_translations
                scores = scores.view(-1)
                all_translations = MTEncDecModel.ids_to_postprocessed_text(
                    all_translations, self.decoder_tokenizer, self.target_processor, filter_beam_ids=True
                )

            best_translations = MTEncDecModel.ids_to_postprocessed_text(
                best_translations, self.decoder_tokenizer, self.target_processor, filter_beam_ids=True
            )
            inputs = MTEncDecModel.ids_to_postprocessed_text(
                src, self.encoder_tokenizer, self.source_processor, filter_beam_ids=False
            )

        finally:
            self.train(mode=mode)
        if return_beam_scores:
            return inputs, all_translations, scores.data.cpu().numpy().tolist(), best_translations

        return inputs, best_translations

    @classmethod
    def prepare_inference_batch(
        cls,
        text,
        prepend_ids=[],
        target=False,
        source_processor=None,
        target_processor=None,
        encoder_tokenizer=None,
        decoder_tokenizer=None,
        device=None,
    ):
        inputs = []
        processor = source_processor if not target else target_processor
        tokenizer = encoder_tokenizer if not target else decoder_tokenizer
        for txt in text:
            txt = txt.rstrip("\n")
            if processor is not None:
                txt = processor.normalize(txt)
                txt = processor.tokenize(txt)
            ids = tokenizer.text_to_ids(txt)
            ids = prepend_ids + [tokenizer.bos_id] + ids + [tokenizer.eos_id]
            inputs.append(ids)
        max_len = max(len(txt) for txt in inputs)
        src_ids_ = np.ones((len(inputs), max_len)) * tokenizer.pad_id
        for i, txt in enumerate(inputs):
            src_ids_[i][: len(txt)] = txt

        src_mask = torch.FloatTensor((src_ids_ != tokenizer.pad_id)).to(device)
        src = torch.LongTensor(src_ids_).to(device)

        return src, src_mask

    @torch.no_grad()
    def translate(
        self,
        text: List[str],
        source_lang: str = None,
        target_lang: str = None,
        return_beam_scores: bool = False,
        log_timing: bool = False,
    ) -> List[str]:
        """
        Translates list of sentences from source language to target language.
        Should be regular text, this method performs its own tokenization/de-tokenization
        Args:
            text: list of strings to translate
            source_lang: if not "ignore", corresponding MosesTokenizer and MosesPunctNormalizer will be run
            target_lang: if not "ignore", corresponding MosesDecokenizer will be run
            return_beam_scores: if True, returns a list of translations and their corresponding beam scores.
            log_timing: if True, prints timing information.
        Returns:
            list of translated strings
        """
        # __TODO__: This will reset both source and target processors even if you want to reset just one.
        if source_lang is not None or target_lang is not None:
            self.source_processor, self.target_processor = MTEncDecModel.setup_pre_and_post_processing_utils(
                source_lang, target_lang, self.encoder_tokenizer_library, self.decoder_tokenizer_library
            )

        mode = self.training
        prepend_ids = []
        if self.multilingual:
            if source_lang is None or target_lang is None:
                raise ValueError("Expect source_lang and target_lang to infer for multilingual model.")
            src_symbol = self.encoder_tokenizer.token_to_id('<' + source_lang + '>')
            tgt_symbol = self.encoder_tokenizer.token_to_id('<' + target_lang + '>')
            if src_symbol in self.multilingual_ids:
                prepend_ids = [src_symbol]
            elif tgt_symbol in self.multilingual_ids:
                prepend_ids = [tgt_symbol]

        if log_timing:
            timer = timers.NamedTimer()
        else:
            timer = None

        cache = {
            "timer": timer,
        }

        try:
            self.eval()
            src, src_mask = MTEncDecModel.prepare_inference_batch(
                text=text,
                prepend_ids=prepend_ids,
                target=False,
                source_processor=self.source_processor,
                target_processor=self.target_processor,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
                device=self.device,
            )
            if return_beam_scores:
                _, all_translations, scores, best_translations = self.batch_translate(
                    src, src_mask, return_beam_scores=True, cache=cache,
                )
                return_val = all_translations, scores, best_translations
            else:
                _, best_translations = self.batch_translate(src, src_mask, return_beam_scores=False, cache=cache)
                return_val = best_translations
        finally:
            self.train(mode=mode)

        if log_timing:
            timing = timer.export()
            timing["mean_src_length"] = src_mask.sum().cpu().item() / src_mask.shape[0]
            tgt, tgt_mask = self.prepare_inference_batch(
                text=best_translations,
                prepend_ids=prepend_ids,
                target=True,
                source_processor=self.source_processor,
                target_processor=self.target_processor,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
                device=self.device,
            )
            timing["mean_tgt_length"] = tgt_mask.sum().cpu().item() / tgt_mask.shape[0]

            if type(return_val) is tuple:
                return_val = return_val + (timing,)
            else:
                return_val = (return_val, timing)

        return return_val

    def itn_translate_tn(
        self,
        text: List[str],
        source_lang: str = None,
        target_lang: str = None,
        return_beam_scores: bool = False,
        log_timing: bool = False,
        inverse_normalizer=None,
        normalizer=None,
    ) -> List[str]:
        """
        Calls the translate() method with the option of running ITN (inverse text-normalization) on the input adn TN (text-normalization) on the output.
        Pipeline : ITN -> translate -> TN
        NOTE: ITN and TN objects must be initialized with the right languages.
        Args:
            text: list of strings to translate
            source_lang: if not "ignore", corresponding MosesTokenizer and MosesPunctNormalizer will be run
            target_lang: if not "ignore", corresponding MosesDecokenizer will be run
            return_beam_scores: if True, returns a list of translations and their corresponding beam scores.
            log_timing: if True, prints timing information.
            inverse_normalizer: instance of nemo_text_processing.inverse_text_normalization.inverse_normalize.InverseNormalizer
            normalizer: instance of nemo_text_processing.text_normalization.normalize.Normalizer
        Returns:
            list of translated strings
        """
        if inverse_normalizer is not None:
            text = [inverse_normalizer.normalize(example) for example in text]
        translations = self.translate(text, source_lang, target_lang, return_beam_scores, log_timing)
        if normalizer is not None:
            translations = [normalizer.normalize(example) for example in translations]
        return translations

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        return ['encoder', 'decoder']

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_de_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_de_transformer12x2/versions/1.0.0rc1/files/nmt_en_de_transformer12x2.nemo",
            description="En->De translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_de_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_de_en_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_de_en_transformer12x2/versions/1.0.0rc1/files/nmt_de_en_transformer12x2.nemo",
            description="De->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_de_en_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_es_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_es_transformer12x2/versions/1.0.0rc1/files/nmt_en_es_transformer12x2.nemo",
            description="En->Es translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_es_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_es_en_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_es_en_transformer12x2/versions/1.0.0rc1/files/nmt_es_en_transformer12x2.nemo",
            description="Es->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_es_en_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_fr_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_fr_transformer12x2/versions/1.0.0rc1/files/nmt_en_fr_transformer12x2.nemo",
            description="En->Fr translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_fr_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_fr_en_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_fr_en_transformer12x2/versions/1.0.0rc1/files/nmt_fr_en_transformer12x2.nemo",
            description="Fr->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_fr_en_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_ru_transformer6x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_ru_transformer6x6/versions/1.0.0rc1/files/nmt_en_ru_transformer6x6.nemo",
            description="En->Ru translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_ru_transformer6x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_ru_en_transformer6x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_ru_en_transformer6x6/versions/1.0.0rc1/files/nmt_ru_en_transformer6x6.nemo",
            description="Ru->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_ru_en_transformer6x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_zh_en_transformer6x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_zh_en_transformer6x6/versions/1.0.0rc1/files/nmt_zh_en_transformer6x6.nemo",
            description="Zh->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_zh_en_transformer6x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_zh_transformer6x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_zh_transformer6x6/versions/1.0.0rc1/files/nmt_en_zh_transformer6x6.nemo",
            description="En->Zh translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_zh_transformer6x6",
        )
        result.append(model)

        # English <-> Hindi models

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_hi_en_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_hi_en_transformer12x2/versions/v1.0.0/files/nmt_hi_en_transformer12x2.nemo",
            description="Hi->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_hi_en_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_hi_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_hi_transformer12x2/versions/v1.0.0/files/nmt_en_hi_transformer12x2.nemo",
            description="En->Hi translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_hi_transformer12x2",
        )
        result.append(model)

        # De/Fr/Es -> English models

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_deesfr_en_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_deesfr_en_transformer12x2/versions/1.2.0/files/mnmt_deesfr_en_transformer12x2.nemo",
            description="De/Es/Fr->En multilingual many-one translation model. The model has 12 encoder and 2 decoder layers with hidden dim 1,024. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_deesfr_en_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_deesfr_en_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_deesfr_en_transformer24x6/versions/1.2.0/files/mnmt_deesfr_en_transformer24x6.nemo",
            description="De/Es/Fr->En multilingual many-one translation model. The model has 24 encoder and 6 decoder layers with hidden dim 1,024. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_deesfr_en_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_deesfr_en_transformer6x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_deesfr_en_transformer6x6/versions/1.2.0/files/mnmt_deesfr_en_transformer6x6.nemo",
            description="De/Es/Fr->En multilingual many-one translation model. The model has 6 encoder and 6 decoder layers with hidden dim 1,024. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_deesfr_en_transformer6x6",
        )
        result.append(model)

        # English -> De/Fr/Es models

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_en_deesfr_transformer12x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_en_deesfr_transformer12x2/versions/1.2.0/files/mnmt_en_deesfr_transformer12x2.nemo",
            description="En->De/Es/Fr multilingual one-many translation model. The model has 12 encoder and 2 decoder layers with hidden dim 1,024. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_en_deesfr_transformer12x2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_en_deesfr_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_en_deesfr_transformer24x6/versions/1.2.0/files/mnmt_en_deesfr_transformer24x6.nemo",
            description="En->De/Es/Fr multilingual one-many translation model. The model has 24 encoder and 6 decoder layers with hidden dim 1,024. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_en_deesfr_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_en_deesfr_transformer6x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_en_deesfr_transformer6x6/versions/1.2.0/files/mnmt_en_deesfr_transformer6x6.nemo",
            description="En->De/Es/Fr multilingual one-many translation model. The model has 6 encoder and 6 decoder layers with hidden dim 1,024. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_en_deesfr_transformer6x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mnmt_en_deesfr_transformerbase",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mnmt_en_deesfr_transformerbase/versions/1.2.0/files/mnmt_en_deesfr_transformerbase.nemo",
            description="En->De/Es/Fr multilingual one-many translation model. The model has 6 encoder and 6 decoder layers with hidden dim 512. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:mnmt_en_deesfr_transformerbase",
        )
        result.append(model)

        # 24x6 models
        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_de_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_de_transformer24x6/versions/1.5/files/en_de_24x6.nemo",
            description="En->De translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_de_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_de_en_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_de_en_transformer24x6/versions/1.5/files/de_en_24x6.nemo",
            description="De->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_de_en_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_es_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_es_transformer24x6/versions/1.5/files/en_es_24x6.nemo",
            description="En->Es translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_es_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_es_en_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_es_en_transformer24x6/versions/1.5/files/es_en_24x6.nemo",
            description="Es->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_es_en_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_fr_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_fr_transformer24x6/versions/1.5/files/en_fr_24x6.nemo",
            description="En->Fr translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_fr_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_fr_en_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_fr_en_transformer24x6/versions/1.5/files/fr_en_24x6.nemo",
            description="Fr->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_fr_en_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_ru_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_ru_transformer24x6/versions/1.5/files/en_ru_24x6.nemo",
            description="En->Ru translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_ru_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_ru_en_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_ru_en_transformer24x6/versions/1.5/files/ru_en_24x6.nemo",
            description="Ru->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_ru_en_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_en_zh_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_zh_transformer24x6/versions/1.5/files/en_zh_24x6.nemo",
            description="En->Zh translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_zh_transformer24x6",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="nmt_zh_en_transformer24x6",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_zh_en_transformer24x6/versions/1.5/files/zh_en_24x6.nemo",
            description="Zh->En translation model. See details here: https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_zh_en_transformer24x6",
        )
        result.append(model)

        return result
