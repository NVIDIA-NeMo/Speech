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

import copy
import os
from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoModelWithLMHead

from nemo.collections.nlp.data.dialogue.data_processor.mellon_qa_data_processor import DialogueMellonQADataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.ms_marco_data_processor import DialogueMSMarcoDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_gpt_generation_dataset import DialogueGPTGenerationDataset
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueGenerationMetrics
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = ['DialogueGPTGenerationModel']

NUM_TASKS = 1  # focussing on intent currently 6  # number of multi-head tasks


class DialogueGPTGenerationModel(NLPModel):
    def __init__(
        self, cfg: DictConfig, trainer: Trainer = None,
    ):

        self.cfg = cfg
        self.data_prepared = False

        self.setup_tokenizer(cfg.tokenizer)
        self.tokenizer.tokenizer.pad_token = self.tokenizer.tokenizer.eos_token
        self.epoch_number = 0
        self.prompt_learning = self.cfg.prompt_learning
        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=True)

        if self.cfg.library == "huggingface":
            self.language_model = AutoModelWithLMHead.from_pretrained(cfg.language_model.pretrained_model_name)
            self.language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))
            if self.cfg.language_model.lm_checkpoint:
                self.language_model.load_state_dict(torch.load(self.cfg.language_model.lm_checkpoint))
        elif self.cfg.library == "megatron":
            if self.prompt_learning:
                # removing tokenizer cfg as this triggers tokenizer construction which is not helpful here as we have a separate tokenizer
                new_cfg = copy.copy(cfg)
                del new_cfg.tokenizer
                self.language_model = MegatronGPTPromptLearningModel(new_cfg, trainer)
            else:
                self.language_model = MegatronGPTModel.restore_from(cfg.language_model.lm_checkpoint, trainer=trainer)

    def training_step(self, batch, batch_idx):
        input_ids, attn_masks, labels, _, _ = batch

        loss = self(input_ids, attn_masks, labels, inference=False)
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

    def forward(self, input_ids, attention_mask, labels, inference=True):

        if self.cfg.library == "huggingface":
            output = self.language_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output['loss']

        elif self.cfg.library == "megatron":
            num_prompt_tokens = (
                len(self.language_model.pseudo_token_ids) if hasattr(self.language_model, 'pseudo_token_ids') else 0
            )

            position_ids = torch.arange(
                start=0, end=num_prompt_tokens + input_ids.size(1), dtype=torch.long, device=input_ids.device,
            )

            position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

            prompt_ids = torch.tensor([0] * input_ids.size(0)) if self.prompt_learning else None

            attn_mask_add_on = torch.ones((attention_mask.size(0), num_prompt_tokens), device=attention_mask.device)
            full_attention_mask = torch.cat([attn_mask_add_on, attention_mask], axis=-1)
            full_attention_mask_expand = torch.tril(
                full_attention_mask.unsqueeze(2).tile(full_attention_mask.size(1))
            ).unsqueeze(1)

            attn_mask = full_attention_mask_expand > 0

            prompt_token_labels = torch.full(
                size=(input_ids.size(0), num_prompt_tokens),
                fill_value=self.tokenizer.tokenizer.pad_token_id,
                dtype=torch.long,
            )

            if self.prompt_learning:
                prompt_token_labels.data = torch.LongTensor(
                    np.tile(np.array(self.language_model.pseudo_token_ids), (input_ids.size(0), 1))
                )

            prompt_token_labels = prompt_token_labels.to(input_ids.device)

            input_ids_new = torch.cat([torch.zeros_like(prompt_token_labels), input_ids], axis=1)
            make_up_last_column_input_ids = (
                torch.ones_like(input_ids_new[:, -1:]) * self.tokenizer.tokenizer.pad_token_id
            )
            left_shifted_input_ids = torch.cat([input_ids_new[:, 1:], make_up_last_column_input_ids], axis=-1)
            if self.prompt_learning:
                unmasked_unreduced_loss = self.language_model(
                    input_ids_new,
                    position_ids,
                    attn_mask,
                    labels=left_shifted_input_ids,
                    taskname_ids=prompt_ids,
                    inference=inference,
                )
            else:
                unmasked_unreduced_loss = self.language_model(
                    input_ids, position_ids, attn_mask, labels=left_shifted_input_ids
                )

            if isinstance(unmasked_unreduced_loss, tuple):
                unmasked_unreduced_loss = unmasked_unreduced_loss[0]

            labels = torch.cat([prompt_token_labels, labels], axis=1)
            make_up_last_column_labels = torch.ones_like(labels[:, -1:]) * self.tokenizer.tokenizer.pad_token_id
            new_labels = torch.cat([labels[:, 1:], make_up_last_column_labels], axis=-1)
            filler = torch.zeros_like(new_labels)
            labels_mask_0 = torch.where(new_labels != -100, new_labels, filler)
            labels_mask = labels_mask_0 > 0

            loss = self.mask_and_reduce_loss(labels_mask, unmasked_unreduced_loss)
        return loss

    def mask_and_reduce_loss(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss

    def setup(self, stage=None):
        super().setup(stage)
        if self.cfg.library == "megatron" and self.prompt_learning:
            self.language_model.init_new_prompts()

    def prepare_megatron_generation(self, labels, input_ids, template_length):
        """
        # adapted from MegatronGPTModel._bucketize_gpt_inference 
        """
        batch_size = labels.size(0)
        prompt_tags = [self.prompt_tags[0]] * batch_size if self.prompt_learning else None
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

    def generate_candidates(self, labels, template_length, input_ids, attn_masks):

        tokens_to_generate = self.cfg.tokens_to_generate
        if self.cfg.library == "huggingface":
            generated_tokens = []
            max_length = 0
            for i in range(input_ids.size(0)):
                param_dict = {
                    "input_ids": input_ids[i : i + 1, : template_length[i]],
                    "max_length": template_length[i] + tokens_to_generate,
                    "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
                }
                generated_tokens.append(self.language_model.generate(**param_dict))
                max_length = max(max_length, generated_tokens[-1].size(1))

            # pad each generated to ensure they are of same length in dim 1, therefore stack-able
            generated_tokens = [
                torch.cat(
                    [i, torch.ones((1, max_length - i.size(1))).to(i.device) * self.tokenizer.tokenizer.pad_token_id],
                    axis=-1,
                )
                for i in generated_tokens
            ]
            generated_tokens = torch.cat(generated_tokens, axis=0)

        elif self.cfg.library == "megatron":
            positions, request = self.prepare_megatron_generation(labels, input_ids, template_length)
            outputs = self.language_model.complete(request, positions, tokens_to_generate)
            generated_tokens = self.post_process_megatron_generation(outputs)

        generated_field = self.process_into_structured_fields(generated_tokens, template_length=template_length)

        ground_truth_field = self.process_into_structured_fields(labels, template_length=template_length)

        return generated_field, ground_truth_field

    def process_into_structured_fields(self, full_seq_ids, template_length=None):

        structured_field = []
        for i in range(full_seq_ids.size(0)):
            start_point = 0 if template_length is None else template_length[i].item()
            stop_point = full_seq_ids.size(1)

            for j in range(start_point, stop_point):
                if full_seq_ids.data[i, j] == self.tokenizer.tokenizer.pad_token_id:
                    stop_point = j
                    break
            one_generated_field = self.tokenizer.tokenizer.decode(full_seq_ids[i, start_point:stop_point]).strip()
            structured_field.append(one_generated_field)
        return structured_field

    def eval_step_helper(self, batch, mode='val'):

        input_ids, attn_masks, labels, template_length, utterance_length = batch

        loss = self(input_ids, attn_masks, labels)
        self.log("{}_loss".format(mode), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # autoregressively generate candidates (possibly with constraint)
        generated_field, ground_truth_field = self.generate_candidates(labels, template_length, input_ids, attn_masks)

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
        elif self._cfg.dataset.task == "mellon_qa":
            self.dialogues_processor = DialogueMellonQADataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset
            )
        else:
            raise ValueError("Only ms_marco and mellon_qa supported for Dialogue GPT Generation Model")

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

        dataset = DialogueGPTGenerationDataset(
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
