# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import enum
import itertools
import random
from typing import List, Optional

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper
from pytorch_lightning.trainer.trainer import Trainer
from sacrebleu import corpus_bleu

from nemo.collections.nlp.data.common.sequence_to_sequence_dataset import (
    BinarizedMemmapSequenceToSequenceDataset,
    TextMemmapSequenceToSequenceDataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.xlm_dataset import (
    BinarizedMemmapCrossLingualMLMAndTranslationDataset,
    TextMemmapCrossLingualMLMAndTranslationDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.modules.common.megatron.megatron_export import DecEmb, EncEmb, TokensHeadEmb
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.parts.nlp_overrides import GlobalBatchDataFetcher
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes import Exportable
from nemo.utils import AppState, logging, timers

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import (
        get_micro_batch_size,
        get_num_microbatches,
        reconfigure_num_microbatches_calculator,
    )

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator as reconfigure_num_microbatches_calculator,
    )
    from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches


__all__ = ["MegatronNMTModel"]


class MultilingualModelType(enum.Enum):
    one_to_many = 1
    many_to_one = 2
    many_to_many = 3


class MegatronNMTModel(MegatronLMEncoderDecoderModel, Exportable):
    """
    Megatron NMT training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        # All of the lines below need to be set when the parent class calls self._build_tokenizer()
        self.encoder_tokenizer_library = cfg.encoder_tokenizer.get('library', 'sentencepiece')
        self.decoder_tokenizer_library = cfg.decoder_tokenizer.get('library', 'sentencepiece')
        self.multilingual_lang_tokens = {}
        self.src_language = cfg.get("src_language", None)
        self.tgt_language = cfg.get("tgt_language", None)

        self.multilingual = cfg.get("multilingual", False)
        self.multilingual_ids = []

        self.validate_input_ids = cfg.get("validate_input_ids", True)
        self.objective = cfg.train_ds.get("objective", "nmt")

        if self.objective == 'nmt-xlm':
            if not self.multilingual:
                raise ValueError("nmt-xlm objective requires model.multilingual=True")

        if self.multilingual:
            self.multilingual_type = self._determine_multilingual_training_type()
            self._setup_multilingual_special_tokens()
        else:
            self.multilingual_type = None

        super().__init__(cfg, trainer=trainer)

    def _determine_multilingual_training_type(self):
        """Determines whether we are doing one-many, many-one, or many-many training based on the config."""
        if self.objective == 'nmt-xlm':
            return MultilingualModelType.many_to_many
        if isinstance(self.src_language, ListConfig) and isinstance(self.tgt_language, ListConfig):
            return MultilingualModelType.many_to_many
        elif isinstance(self.src_language, ListConfig):
            return MultilingualModelType.many_to_one
        elif isinstance(self.tgt_language, ListConfig):
            return MultilingualModelType.one_to_many
        else:
            raise ValueError(
                f"Invalid multilingual training config: {self.src_language}, {self.tgt_language}. Must have either src/tgt as a list of languages."
            )

    def _setup_multilingual_special_tokens(self):
        if self.multilingual_type == MultilingualModelType.many_to_many:
            if self.objective == 'nmt-xlm':
                unique_langs = set(self.src_language + self.tgt_language)
            else:
                # We don't take a set() for tgt_language here because the same lang can appear multiple times.
                unique_langs = set(self.tgt_language)
            for lng in unique_langs:
                self.multilingual_lang_tokens["<" + lng + ">"] = "<" + lng + ">"
        elif self.multilingual_type == MultilingualModelType.many_to_one:
            # Do nothing here since many -> one does not need special tokens for the target language.
            pass
        elif self.multilingual_type == MultilingualModelType.one_to_many:
            for lng in self.tgt_language:
                self.multilingual_lang_tokens["<" + lng + ">"] = "<" + lng + ">"
        else:
            raise ValueError(f"Invalid multilingual training type: {self.multilingual_type}")

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        if stage == 'predict':
            return

        # If the user wants to manually override train and validation dataloaders before calling `.fit()`
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self._cfg.train_ds)
        self.setup_validation_data(self._cfg.validation_ds)
        if hasattr(self._cfg, 'test_ds'):
            self.setup_test_data(self._cfg.test_ds)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            assert (
                self.cfg.share_token_embeddings
            ), "share_word_embedding must be True when using pipeline model parallel > 1"
            assert (
                self.cfg.share_decoder_tokens_head_embeddings
            ), "share_decoder_tokens_head_embeddings must be True when using pipeline model parallel > 1"
            self.enc_dec_model.sync_initial_word_embeddings()
            if (
                self.cfg.encoder.get('position_embedding_type') != 'relative'
                and self.cfg.decoder.get('position_embedding_type') != 'relative'
            ):
                self.enc_dec_model.sync_initial_position_embeddings()
            # Synchronize RPE embeddings across pipeline parallel ranks.
            else:
                if self.cfg.encoder.get('position_embedding_type', 'learned_absolute') == 'relative':
                    self.enc_dec_model.sync_initial_encoder_relative_position_embeddings()
                if self.cfg.decoder.get('position_embedding_type', 'learned_absolute') == 'relative':
                    self.enc_dec_model.sync_initial_decoder_relative_position_embeddings()
                if self.cfg.decoder.get(
                    'position_embedding_type', 'learned_absolute'
                ) == 'relative' and not self.cfg.decoder.get('relative_position_bias_self_attention_only', True):
                    self.enc_dec_model.sync_initial_decoder_cross_attention_relative_position_embeddings()

    def _build_tokenizer(self):
        # Instantiates tokenizers and register to be saved with NeMo Model archive
        # After this call, there will be self.encoder_tokenizer and self.decoder_tokenizer
        # Which can convert between tokens and token_ids for SRC and TGT languages correspondingly.
        encoder_tokenizer_model = self.register_artifact(
            "encoder_tokenizer.model", self._cfg.encoder_tokenizer.get('model')
        )
        decoder_tokenizer_model = self.register_artifact(
            "decoder_tokenizer.model", self._cfg.decoder_tokenizer.get('model')
        )

        self.encoder_tokenizer, self.decoder_tokenizer = MTEncDecModel.setup_enc_dec_tokenizers(
            encoder_tokenizer_library=self.encoder_tokenizer_library,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=(
                self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0)
                if self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0) is not None
                else 0.0
            ),
            encoder_model_name=self._cfg.encoder_tokenizer.get('type', None),
            encoder_r2l=self._cfg.encoder_tokenizer.get('r2l', False),
            decoder_tokenizer_library=self.decoder_tokenizer_library,
            encoder_tokenizer_vocab_file=self._cfg.encoder_tokenizer.get('vocab_file', None),
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=(
                self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0)
                if self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0) is not None
                else 0.0
            ),
            decoder_model_name=self._cfg.encoder_tokenizer.get('type', None),
            decoder_r2l=self._cfg.decoder_tokenizer.get('r2l', False),
            encoder_sentencepiece_legacy=self._cfg.encoder_tokenizer.get('sentencepiece_legacy', False),
            decoder_sentencepiece_legacy=self._cfg.decoder_tokenizer.get('sentencepiece_legacy', False),
        )

    def _build_vocab(self):
        if hasattr(self.cfg, "data") and self.cfg.train_ds.get('objective', 'nmt') != 'nmt-xlm':
            if hasattr(self.cfg.data, "dataset_type"):
                # This happens only when restoring a pre-trained model. We need to add all of the special tokens that were added while pre-training to avoid a checkpoint shape mismatch while restoring.
                MegatronT5Model.add_special_tokens_to_tokenizer(
                    tokenizer=self.encoder_tokenizer,
                    tokenizer_cfg=self.cfg.encoder_tokenizer,
                    dataset_type=self.cfg.data.dataset_type,
                )
                MegatronT5Model.add_special_tokens_to_tokenizer(
                    tokenizer=self.decoder_tokenizer,
                    tokenizer_cfg=self.cfg.decoder_tokenizer,
                    dataset_type=self.cfg.data.dataset_type,
                )

        if self.cfg.train_ds.get('objective', 'nmt') == 'nmt-xlm':
            if self.cfg.encoder_tokenizer.library != 'sentencepiece':
                raise ValueError(
                    f"NMT-XLM objective requires sentencepiece tokenizer, but got encoder tokenizer library : {self.cfg.encoder_tokenizer.library}"
                )
            if self.cfg.decoder_tokenizer.library != 'sentencepiece':
                raise ValueError(
                    f"NMT-XLM objective requires sentencepiece tokenizer, but got decoder tokenizer library : {self.cfg.decoder_tokenizer.library}"
                )
            MegatronT5Model.add_special_tokens_to_tokenizer(
                tokenizer=self.encoder_tokenizer,
                tokenizer_cfg=self.cfg.encoder_tokenizer,
                dataset_type='ul2',
            )
            MegatronT5Model.add_special_tokens_to_tokenizer(
                tokenizer=self.decoder_tokenizer,
                tokenizer_cfg=self.cfg.decoder_tokenizer,
                dataset_type='ul2',
            )

        # Set up pre and post processors as well.
        # NOTE: multilingual language tokens are set up after other special tokens such as eos, pad, sentinel tokens etc are added.
        if self.multilingual:
            (
                self.source_processor_list,
                self.target_processor_list,
                self.multilingual_lang_to_id,
            ) = MTEncDecModel.setup_multilingual_ids_and_processors(
                src_language=self.src_language,
                tgt_language=self.tgt_language,
                encoder_tokenizer=self.encoder_tokenizer,  # Multilingual training requires shared tokenizers.
                decoder_tokenizer=self.decoder_tokenizer,
                encoder_tokenizer_library=self.encoder_tokenizer_library,
                decoder_tokenizer_library=self.decoder_tokenizer_library,
            )
            self.multilingual_ids = list(self.multilingual_lang_to_id.values())
        else:
            # After this call, the model will have  self.source_processor and self.target_processor objects
            self.source_processor, self.target_processor = MTEncDecModel.setup_pre_and_post_processing_utils(
                self.src_language,
                self.tgt_language,
                self.encoder_tokenizer_library,
                self.decoder_tokenizer_library,
            )
            self.multilingual_ids = [None]

        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.encoder_tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        """
        Dataloader produces a global batch which is turned into a list of microbatches.
        The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
        batch = next(dataloader_iter)
        if isinstance(batch, tuple):
            batch = batch[0]
        if isinstance(batch, dict):
            # convert to list if not already converted.
            batch = self._process_batch(batch)

        # Get seq length of batch
        encoder_seq_length = batch[0].size(1)
        decoder_seq_length = batch[1].size(1)

        tensor_shape = [encoder_seq_length, get_micro_batch_size(), self.cfg.encoder.hidden_size]
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        return self._execute_fwd_bwd_function(
            data_iterator=data_iter,
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            decoder_seq_length=decoder_seq_length,
        )

    def eval_step(self, dataloader_iter):
        # Need to squeze dim 0 for old NMT datasets since things are pre-batched and we ask the dataloader for batch size 1.
        batch, _, dataloader_idx = next(dataloader_iter)
        batch = [x.squeeze(dim=0) if x.ndim == 3 else x for x in batch]
        batch = self.process_global_batch_for_text_translation_datasets(batch)

        # Eval step requires text datasets so we need to reconfigure MBS on each batch.
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=batch['text_enc'].size(0) * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=batch['text_enc'].size(0),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        # This returns the averaged loss across data-parallel groups.
        reduced_loss = self.fwd_bwd_step(itertools.chain([batch]), True)

        tokens_enc, labels, enc_mask = batch['text_enc'], batch['labels'], batch['enc_mask']

        predicted_tokens_ids, _ = self.decode(
            tokens_enc,
            enc_mask,
            tokens_enc.size(1)
            + self._cfg.max_generation_delta,  # Generate up to src-length + max generation delta. TODO: Implement better stopping when everything hits <EOS>.
            tokenizer=self.decoder_tokenizer,
        )

        if self.multilingual:
            source_processor = self.source_processor_list[dataloader_idx]
            target_processor = self.target_processor_list[dataloader_idx]
        else:
            source_processor = self.source_processor
            target_processor = self.target_processor

        # Post-process the translations and inputs to log.
        preds = self.postprocess_outputs(
            outputs=predicted_tokens_ids,
            tokenizer=self.decoder_tokenizer,
            processor=target_processor,
        )
        labels = self.postprocess_outputs(
            outputs=labels,
            tokenizer=self.decoder_tokenizer,
            processor=target_processor,
        )
        encoder_inputs = self.postprocess_outputs(
            outputs=tokens_enc,
            tokenizer=self.encoder_tokenizer,
            processor=source_processor,
        )

        loss_dict = {
            'inputs': encoder_inputs,
            'translations': preds,
            'ground_truths': labels,
        }
        if isinstance(reduced_loss, dict):
            loss_dict.update(reduced_loss)
        else:
            loss_dict['loss'] = reduced_loss

        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss_dict)
        else:
            self.validation_step_outputs.append(loss_dict)

        return loss_dict

    def postprocess_outputs(self, outputs, tokenizer, processor):
        # Convert ids to lists.
        outputs = outputs.cpu().numpy().tolist()

        # Filter out the special tokens and de-tokenize.
        results = []
        for item in outputs:
            if tokenizer.eos_id in item:
                idx = item.index(tokenizer.eos_id)
                item = item[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(tokenizer, 'special_token_to_id'):
                item = [id for id in item if id not in tokenizer.special_token_to_id.values()]

            item = tokenizer.ids_to_text(item)
            results.append(item)

        if processor is not None:
            results = [processor.detokenize(item.split(' ')) for item in results]

        return results

    def validation_step(self, dataloader_iter):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(dataloader_iter)

    def _setup_eval_dataloader_from_config(self, cfg: DictConfig, dataset):
        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        dataloaders = []
        for _dataset in dataset:
            sampler = torch.utils.data.distributed.DistributedSampler(
                _dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset=_dataset,
                    batch_size=1,
                    sampler=sampler,
                    num_workers=cfg.get("num_workers", 0),
                    pin_memory=cfg.get("pin_memory", False),
                    drop_last=cfg.get("drop_last", False),
                    shuffle=False,
                    persistent_workers=True if cfg.get("num_workers", 0) > 0 else False,
                )
            )

        return dataloaders

    def on_validation_epoch_end(self):
        return self.eval_epoch_end(self.validation_step_outputs, 'val')

    def on_test_epoch_end(self):
        return self.eval_epoch_end(self.test_step_outputs, 'test')

    def eval_epoch_end(self, outputs, mode):
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        loss_list = []
        bleu_score_list = []
        for dataloader_idx, output in enumerate(outputs):
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([x['loss'] for x in output]).mean()
            else:
                averaged_loss = torch.tensor(0.0).to(self.device)

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(averaged_loss, get_last_rank())

            # averaged_loss = average_losses_across_data_parallel_group([x['loss'] for x in output])
            inputs = list(itertools.chain(*[x['inputs'] for x in output]))
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))
            assert len(translations) == len(inputs)
            assert len(translations) == len(ground_truths)

            # Gather translations and ground truths from all workers
            tr_gt_inp = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            # we also need to drop pairs where ground truth is an empty string
            torch.distributed.all_gather_object(
                tr_gt_inp,
                [(t, g, i) for (t, g, i) in zip(translations, ground_truths, inputs)],
                group=parallel_state.get_data_parallel_group(),
            )
            # if parallel_state.get_data_parallel_rank() == 0:
            if self.global_rank == 0:
                _translations = []
                _ground_truths = []
                _inputs = []

                # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
                gt_inp_set = set()
                for rank in range(0, parallel_state.get_data_parallel_world_size()):
                    for t, g, i in tr_gt_inp[rank]:
                        if g + i not in gt_inp_set:
                            gt_inp_set.add(g + i)
                            _translations.append(t)
                            _ground_truths.append(g)
                            _inputs.append(i)

                if self.tgt_language in ['ja']:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="ja-mecab")
                elif self.tgt_language in ['zh']:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="zh")
                else:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")

                bleu_score = sacre_bleu.score

                dataset_name = "Validation" if mode == 'val' else "Test"
                logging.info(f"{dataset_name}, Dataloader index: {dataloader_idx}, Set size: {len(_translations)}")
                logging.info(f"{dataset_name}, Dataloader index: {dataloader_idx}, SacreBLEU = {bleu_score}")
                logging.info(f"{dataset_name}, Dataloader index: {dataloader_idx}, Translation Examples:")
                logging.info('============================================================')
                for example_idx in range(0, 3):
                    random_index = random.randint(0, len(_translations) - 1)
                    logging.info("    " + '\u0332'.join(f"Example {example_idx}:"))
                    logging.info(f"    Input:        {_inputs[random_index]}")
                    logging.info(f"    Prediction:   {_translations[random_index]}")
                    logging.info(f"    Ground Truth: {_ground_truths[random_index]}")
                    logging.info('============================================================')

            else:
                bleu_score = 0.0

            bleu_score = torch.FloatTensor([bleu_score]).to(self.device)
            # BLEU score is computed on global rank 0 only and then broadcasted to other ranks.
            torch.distributed.all_reduce(bleu_score, op=torch.distributed.ReduceOp.SUM)
            bleu_score = bleu_score.cpu().item()
            loss_list.append(averaged_loss.cpu().numpy())
            bleu_score_list.append(bleu_score)
            if dataloader_idx == 0:
                if self.multilingual:
                    self._log_multilingual_bleu_and_loss(dataloader_idx, bleu_score, averaged_loss, mode)
                else:
                    self.log(f'{mode}_sacreBLEU', bleu_score, batch_size=1)
                    self.log(f'{mode}_loss', averaged_loss, prog_bar=True, batch_size=1)
            else:
                if self.multilingual:
                    self._log_multilingual_bleu_and_loss(dataloader_idx, bleu_score, averaged_loss, mode)
                else:
                    self.log(f'{mode}_sacreBLEU_dl_index_{dataloader_idx}', bleu_score, batch_size=1)
                    self.log(f'{mode}_loss_dl_index_{dataloader_idx}', averaged_loss, prog_bar=False, batch_size=1)
            outputs[dataloader_idx].clear()  # free memory

        if len(loss_list) > 1:
            self.log(f"{mode}_loss_avg", np.mean(loss_list), sync_dist=True, batch_size=1)
            self.log(f"{mode}_sacreBLEU_avg", np.mean(bleu_score_list), batch_size=1)

        app_state = AppState()
        if hasattr(self, "_train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self._cfg.train_ds.global_batch_size,
                micro_batch_size=self._cfg.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

    def _log_multilingual_bleu_and_loss(self, dataloader_idx, bleu_score, loss, mode):
        """
        Function to log multilingual BLEU scores with the right source-target language string instead of just the dataloader idx.
        """
        # Check if one-many or many-one and log with lang ids instead of dataloader_idx
        if isinstance(self.src_language, ListConfig):
            translation_lang_string = f'{self.src_language[dataloader_idx]}-{self.tgt_language}'
        else:
            translation_lang_string = f'{self.src_language}-{self.tgt_language[dataloader_idx]}'

        self.log(f'{mode}_sacreBLEU_{translation_lang_string}', bleu_score, sync_dist=True, batch_size=1)
        self.log(f'{mode}_loss_{translation_lang_string}', loss, sync_dist=True, batch_size=1)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if hasattr(self, '_validation_ds'):
            self._validation_dl = self._setup_eval_dataloader_from_config(
                cfg=val_data_config, dataset=self._validation_ds
            )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if hasattr(self, '_test_ds'):
            self._test_dl = self._setup_eval_dataloader_from_config(cfg=test_data_config, dataset=self._test_ds)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self._setup_megatron_dataloader_from_config(
                cfg=train_data_config, dataset=self._train_ds, consumed_samples=consumed_samples
            )

    def _setup_megatron_dataloader_from_config(self, cfg, dataset, consumed_samples):
        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=cfg.micro_batch_size,
            global_batch_size=cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=True if cfg.num_workers > 0 else False,
        )

    def process_global_batch_for_text_translation_datasets(self, batch):
        """Override parent process_batch since TranslationDataset does not return dictionaries."""
        # Convert each microbatch into a dictionary.
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        batch = {
            'text_enc': src_ids,
            'text_dec': tgt_ids,
            'labels': labels,
            'enc_mask': src_mask.long(),  # super().process_batch() expects torch.int64
            'dec_mask': tgt_mask.long(),  # super().process_batch() expects torch.int64
            'loss_mask': tgt_mask.long(),  # super().process_batch() expects torch.int64
        }

        # Parent function will pad microbatches to the same length.
        return self._process_global_batch_without_megatron_batch_sampler([batch], tokenizer=self.encoder_tokenizer)

    def _build_eval_dataset(self, data_cfg):
        # Set up prepend IDs for validation datasets even if not multingual.
        if self._cfg.train_ds.get('objective', 'nmt') == 'nmt-xlm' or (
            self.multilingual and self.multilingual_type != MultilingualModelType.many_to_one
        ):
            multilingual_ids = [self.multilingual_lang_to_id[lang] for lang in self.cfg.tgt_language]
            dataset = MTEncDecModel._setup_eval_dataset_from_config(
                cfg=data_cfg,
                multilingual=True,
                multilingual_ids=multilingual_ids,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
                add_bos_eos_to_encoder=self._cfg.train_ds.get('objective', 'nmt')
                != 'nmt-xlm',  # nmt-xlm does not add bos/eos to encoder while training so make sure this happens for validation as well.
            )
        else:
            num_eval_datasets = len(data_cfg.src_file_name) if isinstance(data_cfg.src_file_name, ListConfig) else 1
            multilingual_ids = [None] * num_eval_datasets
            dataset = MTEncDecModel._setup_eval_dataset_from_config(
                cfg=data_cfg,
                multilingual=self.multilingual,
                multilingual_ids=multilingual_ids,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
            )
        return dataset

    def build_train_valid_test_datasets(self):
        """Builds the train, validation, and test datasets."""
        self._train_ds = self.build_memmap_dataset_from_config(self._cfg.train_ds)

        if self._cfg.validation_ds.get("dataset_type", "text") != "text":
            raise ValueError(f"Validation dataset type must be 'text', found {self._cfg.validation_ds.dataset_type}")

        self._validation_ds = self._build_eval_dataset(self._cfg.validation_ds)
        if hasattr(self._cfg, 'test_ds'):
            self._test_ds = self._build_eval_dataset(self._cfg.test_ds)

    def _instantiate_memmap_dataset(
        self, cfg, src_file, tgt_file, src_language, tgt_language, num_samples, prepend_id=None
    ):
        if cfg.dataset_type == 'bin_memmap':
            if cfg.get("objective", "nmt") == "nmt":
                dataset = BinarizedMemmapSequenceToSequenceDataset(
                    src_dataset_prefix=src_file,
                    tgt_dataset_prefix=tgt_file,
                    src_tokenizer=self.encoder_tokenizer,
                    tgt_tokenizer=self.decoder_tokenizer,
                    max_src_seq_length=cfg.max_seq_length,
                    max_tgt_seq_length=cfg.max_seq_length,
                    max_num_samples=num_samples[0],
                    seed=self._cfg.seed,
                    prepend_id=prepend_id,
                )
            elif cfg.get("objective", "nmt") == "nmt-xlm":
                # Pass sentinel tokens to the dataset after removing language tokens.
                additional_special_ids = self.encoder_tokenizer.additional_special_tokens_ids
                sentinel_tokens = [id for id in additional_special_ids if id not in self.multilingual_ids]
                dataset = BinarizedMemmapCrossLingualMLMAndTranslationDataset(
                    src_dataset_prefix=src_file,
                    tgt_dataset_prefix=tgt_file,
                    src_tokenizer=self.encoder_tokenizer,
                    tgt_tokenizer=self.decoder_tokenizer,
                    src_language=src_language,
                    tgt_language=tgt_language,
                    max_src_seq_length=cfg.max_seq_length // 2,
                    max_tgt_seq_length=cfg.max_seq_length // 2,
                    max_seq_length_dec=cfg.max_seq_length,
                    max_num_samples=num_samples[0],
                    sampling_ratios=cfg.sampling_ratios,
                    sentinel_tokens=sentinel_tokens,
                    seed=self._cfg.seed,
                )
        elif cfg.dataset_type == 'text_memmap':
            if cfg.get("objective", "nmt") == "nmt":
                dataset = TextMemmapSequenceToSequenceDataset(
                    src_file_name=src_file,
                    tgt_file_name=tgt_file,
                    src_tokenizer=self.encoder_tokenizer,
                    tgt_tokenizer=self.decoder_tokenizer,
                    max_src_seq_length=cfg.max_seq_length,
                    max_tgt_seq_length=cfg.max_seq_length,
                    max_num_samples=num_samples[0],
                    seed=self._cfg.seed,
                    prepend_id=prepend_id,
                )
            elif cfg.get("objective", "nmt") == "nmt-xlm":
                additional_special_ids = self.encoder_tokenizer.additional_special_tokens_ids
                sentinel_tokens = [id for id in additional_special_ids if id not in self.multilingual_ids]
                dataset = TextMemmapCrossLingualMLMAndTranslationDataset(
                    src_file_name=src_file,
                    tgt_file_name=tgt_file,
                    src_tokenizer=self.encoder_tokenizer,
                    tgt_tokenizer=self.decoder_tokenizer,
                    src_language=src_language,
                    tgt_language=tgt_language,
                    max_src_seq_length=cfg.max_seq_length // 2,
                    max_tgt_seq_length=cfg.max_seq_length // 2,
                    max_seq_length_dec=cfg.max_seq_length,
                    max_num_samples=num_samples[0],
                    sampling_ratios=cfg.sampling_ratios,
                    sentinel_tokens=sentinel_tokens,
                    seed=self._cfg.seed,
                )

        return dataset

    def build_memmap_dataset_from_config(self, cfg: DictConfig):
        """Builds a memmap dataset from a existing binary based o nthe provided config."""
        is_src_listconfig = isinstance(cfg.src_file_name, ListConfig)
        is_tgt_listconfig = isinstance(cfg.tgt_file_name, ListConfig)
        # If multilingual, make sure both source and target are list configs
        if self.multilingual:
            if not (is_src_listconfig and is_tgt_listconfig):
                raise ValueError(
                    f"Multilingual datasets must be configured with a ListConfig for both src_file_name and tgt_file_name"
                )
        if is_src_listconfig and not is_tgt_listconfig or is_tgt_listconfig and not is_src_listconfig:
            raise ValueError(
                f"Datasets must be configured with a ListConfig for both src_file_name and tgt_file_name or neither. Found only one of them as listconfig."
            )

        if is_src_listconfig and is_tgt_listconfig:
            if len(cfg.src_file_name) != len(cfg.tgt_file_name):
                raise ValueError(f"Datasets must have the same number of files in src_file_name and tgt_file_name")

            if cfg.concat_sampling_probabilities is None or not isinstance(
                cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    f"concat_sampling_probabilities must be a ListConfig with the same number of files in src_file_name and tgt_file_name, found {cfg.concat_sampling_probabilities}"
                )
            if len(cfg.concat_sampling_probabilities) != len(cfg.src_file_name):
                raise ValueError(
                    f"concat_sampling_probabilities must be of the same size as src_file_name and tgt_file_name. Provided size {len(cfg.concat_sampling_probabilities)}, number of datasets {len(cfg.src_file_name)}"
                )

            # Construct the data prefix list for `get_datasets_weights_and_num_samples()` that is of the format [weight1,file_name1,weight2,file_name2,...]
            data_prefix = []
            for weight, prefix in zip(cfg.concat_sampling_probabilities, cfg.src_file_name):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            num_train_samples = [self.trainer.max_steps * self._cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])

            datasets = []

            # For many -> one multilingual or bilingual models, we don't need to prepend a language token ID
            if not self.multilingual or self.multilingual_type == MultilingualModelType.many_to_one:
                multilingual_ids = [None] * len(cfg.src_file_name)
            # For one -> many and many -> many multilingual models, we need to prepend a language token ID
            else:
                multilingual_ids = [self.multilingual_lang_to_id[lang] for lang in self.cfg.tgt_language]

            for idx, (src_file, tgt_file, num_samples) in enumerate(
                zip(cfg.src_file_name, cfg.tgt_file_name, num_train_samples_per_dataset)
            ):
                dataset = self._instantiate_memmap_dataset(
                    cfg=cfg,
                    src_file=src_file,
                    tgt_file=tgt_file,
                    num_samples=num_samples,
                    prepend_id=multilingual_ids[idx],
                    src_language=(
                        self.src_language if not isinstance(self.src_language, ListConfig) else self.src_language[idx]
                    ),
                    tgt_language=(
                        self.tgt_language if not isinstance(self.tgt_language, ListConfig) else self.tgt_language[idx]
                    ),
                )
                datasets.append(dataset)
            dataset = BlendableDataset(
                datasets=datasets, weights=cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
        else:
            dataset = self._instantiate_memmap_dataset(
                cfg=cfg,
                src_file=cfg.src_file_name,
                tgt_file=cfg.tgt_file_name,
                num_samples=[self.trainer.max_steps * self._cfg.global_batch_size],
                src_language=self.src_language,
                tgt_language=self.tgt_language,
            )
        return dataset

    def list_available_models(self):
        pass

    def on_validation_epoch_start(self):
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=parallel_state.get_data_parallel_world_size(),
            micro_batch_size=1,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

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
        # NOTE: This will also set up appropriate source and target processors for a given src/tgt language for multilingual models instead of creating a list of them.
        if source_lang is not None or target_lang is not None:
            self.source_processor, self.target_processor = MTEncDecModel.setup_pre_and_post_processing_utils(
                source_lang, target_lang, self.encoder_tokenizer_library, self.decoder_tokenizer_library
            )

        mode = self.training
        prepend_ids = []
        if self.multilingual and self.multilingual_type != MultilingualModelType.many_to_one:
            if target_lang is None:
                raise ValueError("target_lang needs to be specified to run inference for multilingual model.")
            tgt_symbol = self.encoder_tokenizer.token_to_id('<' + target_lang + '>')
            if tgt_symbol in self.multilingual_ids:
                prepend_ids = [tgt_symbol]
            else:
                print("WARNING: Target language ID not found in multilingual model. Prepending nothing.")

        if log_timing:
            timer = timers.NamedTimer()
        else:
            timer = None

        cache = {
            "timer": timer,
        }

        try:
            self.eval()
            self.training = False
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
            with torch.inference_mode():
                predicted_tokens_ids, _ = self.decode(
                    src,
                    src_mask,
                    src.size(1)
                    + self._cfg.max_generation_delta,  # Generate up to src-length + max generation delta. TODO: Implement better stopping when everything hits <EOS>.
                    tokenizer=self.decoder_tokenizer,
                )
            best_translations = self.postprocess_outputs(
                outputs=predicted_tokens_ids, tokenizer=self.decoder_tokenizer, processor=self.target_processor
            )
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
        prepend_tgt_lang_id: bool = False,
    ) -> List[str]:
        """
        Calls the translate() method with the option of running ITN (inverse text-normalization) on the input and TN (text-normalization) on the output.
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
        translations = self.translate(
            text, source_lang, target_lang, return_beam_scores, log_timing, prepend_tgt_lang_id
        )
        if normalizer is not None:
            translations = [normalizer.normalize(example) for example in translations]
        return translations

    def on_test_start(self) -> None:
        self.trainer.test_loop._data_fetcher = GlobalBatchDataFetcher()

    @property
    def encoder(self):
        return EncEmb(
            self.enc_dec_model.encoder_embedding,
            self.enc_dec_model.enc_dec_model.encoder,
            self.enc_dec_model.encoder_relative_position_embedding,
            self.device,
        )

    @property
    def decoder(self):
        return DecEmb(
            self.enc_dec_model.decoder_embedding,
            self.enc_dec_model.enc_dec_model.decoder,
            self.enc_dec_model.decoder_relative_position_embedding,
            self.device,
        )

    @property
    def log_softmax(self):
        return TokensHeadEmb(self.enc_dec_model.decoder_embedding, self.enc_dec_model.tokens_head, self.device)

    @property
    def input_module(self):
        return self.encoder

    def list_export_subnets(self):
        return ['encoder', 'log_softmax', 'decoder']
