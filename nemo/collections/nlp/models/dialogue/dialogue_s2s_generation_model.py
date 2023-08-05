# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

import os
from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM

from nemo.collections.nlp.data.dialogue import DialogueSGDDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.mellon_qa_data_processor import DialogueMellonQADataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.ms_marco_data_processor import DialogueMSMarcoDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_s2s_generation_dataset import DialogueS2SGenerationDataset
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueGenerationMetrics
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True
except:
    HAVE_APEX = False


__all__ = ['DialogueS2SGenerationModel']


class DialogueS2SGenerationModel(NLPModel):
    def __init__(
        self, cfg: DictConfig, trainer: Trainer = None,
    ):

        self.cfg = cfg
        self.data_prepared = False
        self.epoch_number = 0
        if self.cfg.library == "huggingface":
            self.setup_tokenizer(cfg.tokenizer)
        elif self.cfg.library == "megatron":
            # supporting MegatronT5Model in precision = fp16
            t5_cfg = MegatronT5Model.restore_from(
                restore_path=cfg.language_model.lm_checkpoint, trainer=trainer, return_config=True
            )
            # Override the T5 configuration with the one from the config file.
            OmegaConf.set_struct(t5_cfg, True)
            with open_dict(t5_cfg):
                t5_cfg.masked_softmax_fusion = False
                t5_cfg.precision = 16
                t5_cfg.encoder_arch = 'transformer'
                t5_cfg.decoder_arch = 'transformer'

            language_model = MegatronT5Model.restore_from(
                restore_path=cfg.language_model.lm_checkpoint, trainer=trainer, override_config_path=t5_cfg
            )
            self.tokenizer = language_model.tokenizer

        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=True)

        if self.cfg.library == "huggingface":
            self.language_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.language_model.pretrained_model_name)
            self.language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))
            if self.cfg.language_model.lm_checkpoint:
                self.language_model.load_state_dict(torch.load(self.cfg.language_model.lm_checkpoint))
        elif self.cfg.library == "megatron":
            self.language_model = language_model

    def training_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        loss = self(input_ids, attn_masks, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.eval_step_helper(batch=batch)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        self.eval_epoch_end(self.validation_step_outputs, mode='val')
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.test_step_outputs, mode='test')
        self.test_step_outputs.clear()  # free memory

    def eval_epoch_end(self, outputs, mode='val'):

        generated_field = []
        ground_truth_field = []
        inputs = []
        loss = []

        for output in outputs:
            generated_field += output["generated_field"]
            ground_truth_field += output["ground_truth_field"]
            inputs += output["input"]
            loss.append(output["loss"].item())

        os.makedirs(self.cfg.dataset.dialogues_example_dir, exist_ok=True)
        filename = os.path.join(
            self.cfg.dataset.dialogues_example_dir, f"{mode}_predictions_epoch{self.epoch_number}.jsonl"
        )

        DialogueGenerationMetrics.save_predictions(
            filename, generated_field, ground_truth_field, inputs,
        )

        label_acc = np.mean([int(generated_field[i] == ground_truth_field[i]) for i in range(len(generated_field))])
        precision, recall, f1 = DialogueGenerationMetrics.get_f1(generated_field, ground_truth_field)
        bleu = DialogueGenerationMetrics.get_bleu(generated_field, ground_truth_field)
        avg_loss = np.mean(loss)
        ppl = np.exp(avg_loss)

        self.log('{}_accuracy'.format(mode), label_acc * 100)
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1', f1)
        self.log('bleu', bleu)
        self.log('{}_loss'.format(mode), avg_loss)
        self.log('{}_ppl'.format(mode), ppl)

        if mode == 'val':
            self.epoch_number += 1
            if self.cfg.save_model:
                filename = '{}/val_loss-{}-epoch-{}-answer-extender.bin'.format(
                    self.cfg.dataset.dialogues_example_dir, avg_loss, self.epoch_number
                )
                torch.save(self.language_model.state_dict(), filename)

    def test_step(self, batch, batch_idx):
        loss = self.eval_step_helper(batch=batch, mode='test')
        self.test_step_outputs.append(loss)
        return loss

    # for inference only
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # return self(batch)
        raise NotImplementedError()

    def forward(self, input_ids, attention_masks, labels):
        if self.cfg.library == "huggingface":
            output = self.language_model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            loss = output['loss']
        elif self.cfg.library == "megatron":

            labels = torch.where(labels != -100, labels, torch.zeros_like(labels))
            decoder_attn_masks = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))

            unmasked_unreduced_loss = self.language_model(
                input_ids, labels[:, :-1], attention_masks, decoder_attn_masks[:, :-1], lm_labels=labels[:, 1:]
            )
            loss = self.language_model.loss_func(decoder_attn_masks[:, 1:].contiguous(), unmasked_unreduced_loss)
        return loss

    def prepare_megatron_generation(self, labels, input_ids, template_length):
        """
        # adapted from MegatronGPTModel._bucketize_gpt_inference 
        """
        batch_size = labels.size(0)
        prompt_tags = [self.prompt_tags[0]] * batch_size if self.prompt_tags else None
        batch_tokens = input_ids.tolist()

        # unpad tokens
        lens = template_length
        indxs = [index for index in range(batch_size)]
        for lenn, index in zip(lens, indxs):
            batch_tokens[index] = batch_tokens[index][:lenn]

        # chunk tokens by same length
        pre_buckets, lens = [], list(set(lens.tolist()))
        for lenn in lens:
            pre_buckets.append([(tokens, index) for index, tokens in enumerate(batch_tokens) if len(tokens) == lenn])

        buckets, positions, bucket_prompt_tags = [], [], []

        # get buckets and prompts initial positions
        for bucket in pre_buckets:
            buckets.append(torch.tensor([item[0] for item in bucket]).to(device=labels.device))
            positions.append([item[1] for item in bucket])

            # bucket prompt tags identically to their corresponding examples
            if prompt_tags:
                bucket_prompt_tags.append([prompt_tags[item[1]] for item in bucket])

        # Flatten position list
        positions = [item for sublist in positions for item in sublist]

        # Flatten buckets and bucket_prompt_tags # temp fix for megatron complete issue. However, this is also slower than bucketized inference
        buckets = [item.unsqueeze(0) for sublist in buckets for item in sublist]
        bucket_prompt_tags = [[item] for sublist in bucket_prompt_tags for item in sublist]

        request = {"tokens": buckets, "prompt_tags": bucket_prompt_tags}

        return positions, request

    def post_process_megatron_generation(self, outputs):
        text_outputs = [output[0] for output in outputs]
        generated_tokens = self.tokenizer.tokenizer(text_outputs, padding=True, return_tensors="pt").data["input_ids"]
        return generated_tokens

    def generate_candidates(self, input_ids, attn_masks, labels):

        tokens_to_generate = self.cfg.tokens_to_generate
        if self.cfg.library == "huggingface":

            param_dict = {
                "input_ids": input_ids,
                "attention_mask": attn_masks,
                "max_length": tokens_to_generate,
            }
            generated_tokens = self.language_model.generate(**param_dict)

        elif self.cfg.library == 'megatron':
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
            )
            generated_tokens, _ = self.language_model.decode(input_ids, attn_masks, tokens_to_generate)

        generated_field = self.process_into_structured_fields(generated_tokens)
        ground_truth_field = self.process_into_structured_fields(labels)

        return generated_field, ground_truth_field

    def process_into_structured_fields(self, full_seq_ids, template_length=None):

        structured_field = []
        for i in range(full_seq_ids.size(0)):
            start_point = 0 if template_length is None else template_length[i].item()
            stop_point = full_seq_ids.size(1)

            for j in range(start_point, stop_point):
                if full_seq_ids.data[i, j] in [self.tokenizer.tokenizer.pad_token_id, -100] and j != 0:
                    stop_point = j
                    break
            token_ids = full_seq_ids[i, start_point:stop_point]
            one_generated_field = self.tokenizer.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            structured_field.append(one_generated_field)
        return structured_field

    def eval_step_helper(self, batch, mode='val'):

        input_ids, attn_masks, labels = batch

        loss = self(input_ids, attn_masks, labels)
        self.log("{}_loss".format(mode), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        generated_field, ground_truth_field = self.generate_candidates(input_ids, attn_masks, labels)

        return {
            'loss': loss,
            'input': self.tokenizer.tokenizer.batch_decode(input_ids, skip_special_tokens=True),
            'generated_field': generated_field,
            'ground_truth_field': ground_truth_field,
        }

    def prepare_data(self):
        """
        Preprocessed schema and dialogues and caches this
        """
        if self.data_prepared:
            return

        if self._cfg.dataset.task == "ms_marco":
            self.dialogues_processor = DialogueMSMarcoDataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset
            )
        elif self._cfg.dataset.task == "sgd_generation":
            self.dialogues_processor = DialogueSGDDataProcessor(
                data_dir=self._cfg.dataset.data_dir,
                dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
                tokenizer=self.tokenizer,
                cfg=self._cfg.dataset,
            )
        elif self._cfg.dataset.task == "mellon_qa":
            self.dialogues_processor = DialogueMellonQADataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset
            )
        else:
            raise ValueError("Only ms_marco, sgd_generation and mellon_qa supported for Dialogue GPT Generation Model")

        self.data_prepared = True

    def update_data_dirs(self, data_dir: str, dialogues_example_dir: str):
        """
        Update data directories

        Args:
            data_dir: path to data directory
            dialogues_example_dir: path to preprocessed dialogues example directory, if not exists will be created.
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} is not found")
        self._cfg.dataset.data_dir = data_dir
        self._cfg.dataset.dialogues_example_dir = dialogues_example_dir
        logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
        logging.info(f'Setting model.dataset.dialogues_example_dir to {dialogues_example_dir}.')

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, split=train_data_config.ds_item)

    def setup_multiple_validation_data(self, val_data_config: Optional[DictConfig] = None):
        return self.setup_validation_data(val_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split=val_data_config.ds_item)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.setup_test_data(test_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, split=test_data_config.ds_item)

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        dataset = DialogueS2SGenerationDataset(
            dataset_split=split,
            dialogues_processor=self.dialogues_processor,
            tokenizer=self.dialogues_processor._tokenizer,
            cfg=dataset_cfg,
        )

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result
