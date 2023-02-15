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

import json
import os
from collections import defaultdict
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset
from nemo.collections.nlp.data.text_normalization.decoder_dataset import (
    TarredTextNormalizationDecoderDataset,
    TextNormalizationDecoderDataset,
)
from nemo.collections.nlp.models.duplex_text_normalization.utils import get_formatted_string
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import ChannelType, LabelsType, LossType, MaskType, NeuralType
from nemo.utils import logging

try:
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


__all__ = ['DuplexDecoderModel']


class DuplexDecoderModel(NLPModel):
    """
    Transformer-based (duplex) decoder model for TN/ITN.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "decoder_input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "labels": NeuralType(('B', 'T'), LabelsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"loss": NeuralType((), LossType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.transformer)
        self.max_sequence_len = cfg.get('max_sequence_len', self.tokenizer.model_max_length)
        self.mode = cfg.get('mode', 'joint')

        self.transformer_name = cfg.transformer

        # Language
        self.lang = cfg.get('lang', None)

        # Covering Grammars
        self.cg_normalizer = None  # Default
        # We only support integrating with English TN covering grammars at the moment
        self.use_cg = cfg.get('use_cg', False) and self.lang == constants.ENGLISH
        if self.use_cg:
            self.setup_cgs(cfg)

        # setup processor for detokenization
        self.processor = MosesProcessor(lang_id=self.lang)

    # Setup covering grammars (if enabled)
    def setup_cgs(self, cfg: DictConfig):
        """
        Setup covering grammars (if enabled).
        :param cfg: Configs of the decoder model.
        """
        self.use_cg = True
        self.neural_confidence_threshold = cfg.get('neural_confidence_threshold', 0.99)
        self.n_tagged = cfg.get('n_tagged', 1)
        input_case = 'cased'  # input_case is cased by default
        if hasattr(self.tokenizer, 'do_lower_case') and self.tokenizer.do_lower_case:
            input_case = 'lower_cased'

        if PYNINI_AVAILABLE:
            self.cg_normalizer = NormalizerWithAudio(input_case=input_case, lang=self.lang)
        else:
            self.cg_normalizer = None
            logging.warning(
                "`nemo_text_processing` is not installed, see https://github.com/NVIDIA/NeMo-text-processing for details"
            )

    @typecheck()
    def forward(self, input_ids, decoder_input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # tarred dataset contains batches, and the first dimension of size 1 added by the DataLoader
        # (batch_size is set to 1) is redundant
        if batch['input_ids'].ndim == 3:
            batch = {k: v.squeeze(dim=0) for k, v in batch.items()}

        # Apply Transformer
        train_loss = self.forward(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch['decoder_input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': train_loss, 'lr': lr}

    # Validation and Testing
    def validation_step(self, batch, batch_idx, dataloader_idx=0, split="val"):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        # Apply Transformer
        val_loss = self.forward(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch['decoder_input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        labels_str = self.tokenizer.batch_decode(
            torch.ones_like(batch['labels']) * ((batch['labels'] == -100) * 100) + batch['labels'],
            skip_special_tokens=True,
        )
        generated_texts, _, _ = self._generate_predictions(
            input_ids=batch['input_ids'], model_max_len=self.max_sequence_len
        )
        results = defaultdict(int)
        for idx, class_id in enumerate(batch['semiotic_class_id']):
            direction = constants.TASK_ID_TO_MODE[batch['direction'][idx][0].item()]
            class_name = self._val_id_to_class[dataloader_idx][class_id[0].item()]

            pred_result = TextNormalizationTestDataset.is_same(
                generated_texts[idx], labels_str[idx], constants.DIRECTIONS_TO_MODE[direction]
            )

            results[f"correct_{class_name}_{direction}"] += torch.tensor(pred_result, dtype=torch.int).to(self.device)
            results[f"total_{class_name}_{direction}"] += torch.tensor(1).to(self.device)

        results[f"{split}_loss"] = val_loss
        return dict(results)

    def multi_validation_epoch_end(self, outputs: List, dataloader_idx=0, split="val"):
        """
        Called at the end of validation to aggregate outputs.

        Args:
            outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x[f'{split}_loss'] for x in outputs]).mean()

        # create a dictionary to store all the results
        results = {}
        directions = [constants.TN_MODE, constants.ITN_MODE] if self.mode == constants.JOINT_MODE else [self.mode]
        for class_name in self._val_class_to_id[dataloader_idx]:
            for direction in directions:
                results[f"correct_{class_name}_{direction}"] = 0
                results[f"total_{class_name}_{direction}"] = 0

        for key in results:
            count = [x[key] for x in outputs if key in x]
            count = torch.stack(count).sum() if len(count) > 0 else torch.tensor(0).to(self.device)
            results[key] = count

        all_results = defaultdict(list)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                for key, v in results.items():
                    all_results[key].append(torch.empty_like(v))
            for key, v in results.items():
                torch.distributed.all_gather(all_results[key], v)
        else:
            for key, v in results.items():
                all_results[key].append(v)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if split == "test":
                val_name = self._test_names[dataloader_idx].upper()
            else:
                val_name = self._validation_names[dataloader_idx].upper()
            final_results = defaultdict(int)
            for key, v in all_results.items():
                for _v in v:
                    final_results[key] += _v.item()

            accuracies = defaultdict(dict)
            for key, value in final_results.items():
                if "total_" in key:
                    _, class_name, mode = key.split('_')
                    correct = final_results[f"correct_{class_name}_{mode}"]
                    if value == 0:
                        accuracies[mode][class_name] = (0, correct, value)
                    else:
                        acc = round(correct / value * 100, 3)
                        accuracies[mode][class_name] = (acc, correct, value)

            for mode, values in accuracies.items():
                report = f"Accuracy {mode.upper()} task {val_name}:\n"
                report += '\n'.join(
                    [
                        get_formatted_string((class_name, f'{v[0]}% ({v[1]}/{v[2]})'), str_max_len=24)
                        for class_name, v in values.items()
                    ]
                )
                # calculate average across all classes
                all_total = 0
                all_correct = 0
                for _, class_values in values.items():
                    _, correct, total = class_values
                    all_correct += correct
                    all_total += total
                all_acc = round((all_correct / all_total) * 100, 3) if all_total > 0 else 0
                report += '\n' + get_formatted_string(
                    ('AVG', f'{all_acc}% ({all_correct}/{all_total})'), str_max_len=24
                )
                logging.info(report)
                accuracies[mode]['AVG'] = [all_acc]

        self.log(f'{split}_loss', avg_loss)
        if self.trainer.is_global_zero:
            for mode in accuracies:
                for class_name, values in accuracies[mode].items():
                    self.log(f'{val_name}_{mode.upper()}_acc_{class_name.upper()}', values[0], rank_zero_only=True)
        return {
            f'{split}_loss': avg_loss,
        }

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx, split="test")

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
        Called at the end of test to aggregate outputs.
        outputs: list of individual outputs of each test step.
        """
        return self.multi_validation_epoch_end(outputs, dataloader_idx, split="test")

    @torch.no_grad()
    def _generate_predictions(self, input_ids: torch.Tensor, model_max_len: int = 512):
        """
        Generates predictions
        """
        outputs = self.model.generate(
            input_ids, output_scores=True, return_dict_in_generate=True, max_length=model_max_len
        )

        generated_ids, sequence_toks_scores = outputs['sequences'], outputs['scores']
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts, generated_ids, sequence_toks_scores

    # Functions for inference
    @torch.no_grad()
    def _infer(
        self,
        sents: List[List[str]],
        nb_spans: List[int],
        span_starts: List[List[int]],
        span_ends: List[List[int]],
        inst_directions: List[str],
    ):
        """ Main function for Inference
        Args:
            sents: A list of inputs tokenized by a basic tokenizer.
            nb_spans: A list of ints where each int indicates the number of semiotic spans in each input.
            span_starts: A list of lists where each list contains the starting locations of semiotic spans in an input.
            span_ends: A list of lists where each list contains the ending locations of semiotic spans in an input.
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).

        Returns: A list of lists where each list contains the decoded spans for the corresponding input.
        """
        self.eval()

        if sum(nb_spans) == 0:
            return [[]] * len(sents)
        model, tokenizer = self.model, self.tokenizer
        ctx_size = constants.DECODE_CTX_SIZE
        extra_id_0 = constants.EXTRA_ID_0
        extra_id_1 = constants.EXTRA_ID_1

        """
        Build all_inputs - extracted spans to be transformed by the decoder model
        Inputs for TN direction have "0" prefix, while the backward, ITN direction, has prefix "1"
        "input_centers" - List[str] - ground-truth labels for the span
        """
        input_centers, input_dirs, all_inputs = [], [], []
        for ix, sent in enumerate(sents):
            cur_inputs = []
            for jx in range(nb_spans[ix]):
                cur_start = span_starts[ix][jx]
                cur_end = span_ends[ix][jx]
                ctx_left = sent[max(0, cur_start - ctx_size) : cur_start]
                ctx_right = sent[cur_end + 1 : cur_end + 1 + ctx_size]
                span_words = sent[cur_start : cur_end + 1]
                span_words_str = ' '.join(span_words)
                input_centers.append(span_words_str)
                input_dirs.append(inst_directions[ix])
                # Build cur_inputs
                if inst_directions[ix] == constants.INST_BACKWARD:
                    cur_inputs = [constants.ITN_PREFIX]
                if inst_directions[ix] == constants.INST_FORWARD:
                    cur_inputs = [constants.TN_PREFIX]
                cur_inputs += ctx_left
                cur_inputs += [extra_id_0] + span_words_str.split(' ') + [extra_id_1]
                cur_inputs += ctx_right
                all_inputs.append(' '.join(cur_inputs))

        # Apply the decoding model
        batch = tokenizer(all_inputs, padding=True, return_tensors='pt')
        input_ids = batch['input_ids'].to(self.device)

        generated_texts, generated_ids, sequence_toks_scores = self._generate_predictions(
            input_ids=input_ids, model_max_len=self.max_sequence_len
        )

        # Use covering grammars (if enabled)
        if self.use_cg:

            # Compute sequence probabilities
            sequence_probs = torch.ones(len(all_inputs)).to(self.device)
            for ix, cur_toks_scores in enumerate(sequence_toks_scores):
                cur_generated_ids = generated_ids[:, ix + 1].tolist()
                cur_toks_probs = torch.nn.functional.softmax(cur_toks_scores, dim=-1)
                # Compute selected_toks_probs
                selected_toks_probs = []
                for jx, _id in enumerate(cur_generated_ids):
                    if _id != self.tokenizer.pad_token_id:
                        selected_toks_probs.append(cur_toks_probs[jx, _id])
                    else:
                        selected_toks_probs.append(1)
                selected_toks_probs = torch.tensor(selected_toks_probs).to(self.device)
                sequence_probs *= selected_toks_probs

            # For TN cases where the neural model is not confident, use CGs
            neural_confidence_threshold = self.neural_confidence_threshold
            for ix, (_dir, _input, _prob) in enumerate(zip(input_dirs, input_centers, sequence_probs)):
                if _dir == constants.INST_FORWARD and _prob < neural_confidence_threshold:
                    try:
                        cg_outputs = self.cg_normalizer.normalize(text=_input, verbose=False, n_tagged=self.n_tagged)
                        generated_texts[ix] = list(cg_outputs)[0]
                    except:  # if there is any exception, fall back to the input
                        generated_texts[ix] = _input

        # Prepare final_texts
        final_texts, span_ctx = [], 0
        for nb_span in nb_spans:
            cur_texts = []
            for i in range(nb_span):
                cur_texts.append(generated_texts[span_ctx])
                span_ctx += 1
            final_texts.append(cur_texts)

        return final_texts

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self.train_dataset, self._train_dl = None, None
            return
        self.train_dataset, self._train_dl = self._setup_dataloader_from_config(
            cfg=train_data_config, data_split="train"
        )

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

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self.validation_dataset, self._validation_dl = None, None
            return
        self.validation_dataset, self._validation_dl = self._setup_dataloader_from_config(
            cfg=val_data_config, data_split="val"
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

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict] = None):
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds
        return super().setup_multiple_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict] = None):
        if test_data_config is None:
            test_data_config = self._cfg.test_ds
        return super().setup_multiple_test_data(test_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.data_path is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self.test_dataset, self._test_dl = None, None
            return
        self.test_dataset, self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, data_split="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, data_split: str):
        logging.info(f"Creating {data_split} dataset")

        shuffle = cfg["shuffle"]

        if cfg.get("use_tarred_dataset", False):
            logging.info('Tarred dataset')
            metadata_file = cfg["tar_metadata_file"]
            if metadata_file is None or not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Trying to use tarred dataset but could not find {metadata_file}.")

            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                num_batches = metadata["num_batches"]
                tar_files = os.path.join(os.path.dirname(metadata_file), metadata["text_tar_filepaths"])
            logging.info(f"Loading {tar_files}")

            dataset = TarredTextNormalizationDecoderDataset(
                text_tar_filepaths=tar_files,
                num_batches=num_batches,
                shuffle_n=cfg.get("tar_shuffle_n", 4 * cfg['batch_size']) if shuffle else 0,
                shard_strategy=cfg.get("shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )

            dl = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                sampler=None,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
        else:
            input_file = cfg.data_path
            if not os.path.exists(input_file):
                raise ValueError(f"{input_file} not found.")

            dataset = TextNormalizationDecoderDataset(
                input_file=input_file,
                tokenizer=self.tokenizer,
                tokenizer_name=self.transformer_name,
                mode=self.mode,
                max_len=self.max_sequence_len,
                decoder_data_augmentation=cfg.get('decoder_data_augmentation', False)
                if data_split == "train"
                else False,
                lang=self.lang,
                use_cache=cfg.get('use_cache', False),
                max_insts=cfg.get('max_insts', -1),
                do_tokenize=True,
            )

            # create and save class names to class_ids mapping for validation
            # (each validation set might have different classes)
            if data_split in ['val', 'test']:
                if not hasattr(self, "_val_class_to_id"):
                    self._val_class_to_id = []
                    self._val_id_to_class = []
                self._val_class_to_id.append(dataset.label_ids_semiotic)
                self._val_id_to_class.append({v: k for k, v in dataset.label_ids_semiotic.items()})

            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, label_pad_token_id=constants.LABEL_PAD_TOKEN_ID, padding=True
            )
            dl = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle,
                collate_fn=data_collator,
                num_workers=cfg.get("num_workers", 3),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )

        return dataset, dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="neural_text_normalization_t5",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/neural_text_normalization_t5/versions/1.5.0/files/neural_text_normalization_t5_decoder.nemo",
                description="Text Normalization model's decoder model.",
            )
        )
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="itn_en_t5",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/itn_en_t5/versions/1.11.0/files/itn_en_t5_decoder.nemo",
                description="English Inverse Text Normalization model's decoder model.",
            )
        )
        return result
