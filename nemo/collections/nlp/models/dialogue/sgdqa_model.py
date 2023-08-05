# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

import os
from typing import List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.dialogue import DialogueSGDBERTDataset, DialogueSGDDataProcessor
from nemo.collections.nlp.data.dialogue.sgd.evaluate import evaluate, get_in_domain_services
from nemo.collections.nlp.data.dialogue.sgd.prediction_utils import write_predictions_to_file
from nemo.collections.nlp.losses import SGDDialogueStateLoss
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules import SGDDecoder, SGDEncoder
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging

__all__ = ['SGDQAModel']


class SGDQAModel(NLPModel):
    """
    Dialogue State Tracking Model SGD-QA (https://arxiv.org/abs/2105.08049)

    The SGD-QA model is a fast multi-pass schema-guided state-tracking model, that is trained on the Google schema-guided state tracking dataset (https://arxiv.org/abs/1909.05855).
    The model takes dialogue as input and outputs the dialogue state, which includes slot-value pairs. 
    The model consists of two components: a neural natural language understanding model (NLU), and a rule-based state tracker.
    The NLU takes in a dialogue turn and different schema (entity) information options and outputs their match score. The state tracker takes the highest rated entities and composes
    the dialogue state across turns.
    """

    @property
    def output_module(self):
        return self.decoder

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.data_prepared = False
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = SGDEncoder(hidden_size=self.bert_model.config.hidden_size, dropout=self._cfg.encoder.dropout)
        self.decoder = SGDDecoder(embedding_dim=self.bert_model.config.hidden_size)
        self.loss = SGDDialogueStateLoss(reduction="mean")

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        token_embeddings = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(token_embeddings, tuple):
            token_embeddings = token_embeddings[0]

        encoded_utterance, token_embeddings = self.encoder(hidden_states=token_embeddings)
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_spans,
        ) = self.decoder(
            encoded_utterance=encoded_utterance, token_embeddings=token_embeddings, utterance_mask=attention_mask
        )
        return (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_spans,
        )

    def training_step(self, batch, batch_idx):
        (
            example_id_num,
            service_id,
            utterance_ids,
            token_type_ids,
            attention_mask,
            intent_status,
            requested_slot_status,
            categorical_slot_status,
            categorical_slot_value_status,
            noncategorical_slot_status,
            noncategorical_slot_value_start,
            noncategorical_slot_value_end,
            start_char_idx,
            end_char_idx,
            task_mask,
        ) = batch
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_spans,
        ) = self(input_ids=utterance_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.loss(
            logit_intent_status=logit_intent_status,
            intent_status=intent_status,
            logit_req_slot_status=logit_req_slot_status,
            requested_slot_status=requested_slot_status,
            logit_cat_slot_status=logit_cat_slot_status,
            categorical_slot_status=categorical_slot_status,
            logit_cat_slot_value_status=logit_cat_slot_value_status,
            categorical_slot_value_status=categorical_slot_value_status,
            logit_noncat_slot_status=logit_noncat_slot_status,
            noncategorical_slot_status=noncategorical_slot_status,
            logit_spans=logit_spans,
            noncategorical_slot_value_start=noncategorical_slot_value_start,
            noncategorical_slot_value_end=noncategorical_slot_value_end,
            task_mask=task_mask,
        )
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': loss,
            'lr': lr,
        }

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> dict:
        """
        Called at every validation step to aggregate and postprocess outputs on each GPU
        Args:
            batch: input batch at validation step
            batch_idx: batch index 
            dataloader_idx: dataloader index
        """
        loss, tensors = self.eval_step_helper(batch=batch)
        self.log(f'val_loss', loss)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append({f'val_loss': loss, f'tensors': tensors})
        else:
            self.validation_step_outputs.append({f'val_loss': loss, f'tensors': tensors})

        return {f'val_loss': loss, f'tensors': tensors}

    def test_step(self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> dict:
        """
        Called at every test step to aggregate and postprocess outputs on each GPU
        Args:
            batch: input batch at test step
            batch_idx: batch index 
            dataloader_idx: dataloader index
        """
        loss, tensors = self.eval_step_helper(batch=batch)
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append({f'test_loss': loss, f'tensors': tensors})
        else:
            self.test_step_outputs.append({f'test_loss': loss, f'tensors': tensors})

        return {f'test_loss': loss, f'tensors': tensors}

    def eval_step_helper(self, batch: List[torch.Tensor]):
        """
        Helper called at every validation/test step to aggregate and postprocess outputs on each GPU
        Args:
            batch: input batch at step
        Returns:
            loss: averaged batch loss
            tensors: collection of aggregated output tensors across all GPU workers
        """
        (
            example_id_num,
            service_id,
            utterance_ids,
            token_type_ids,
            attention_mask,
            intent_status,
            requested_slot_status,
            categorical_slot_status,
            categorical_slot_value_status,
            noncategorical_slot_status,
            noncategorical_slot_value_start,
            noncategorical_slot_value_end,
            start_char_idx,
            end_char_idx,
            task_mask,
        ) = batch
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_spans,
        ) = self(input_ids=utterance_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.loss(
            logit_intent_status=logit_intent_status,
            intent_status=intent_status,
            logit_req_slot_status=logit_req_slot_status,
            requested_slot_status=requested_slot_status,
            logit_cat_slot_status=logit_cat_slot_status,
            categorical_slot_status=categorical_slot_status,
            logit_cat_slot_value_status=logit_cat_slot_value_status,
            categorical_slot_value_status=categorical_slot_value_status,
            logit_noncat_slot_status=logit_noncat_slot_status,
            noncategorical_slot_status=noncategorical_slot_status,
            logit_spans=logit_spans,
            noncategorical_slot_value_start=noncategorical_slot_value_start,
            noncategorical_slot_value_end=noncategorical_slot_value_end,
            task_mask=task_mask,
        )

        all_example_id_num = []
        all_service_id = []
        all_logit_intent_status = []
        all_logit_req_slot_status = []
        all_logit_cat_slot_status = []
        all_logit_cat_slot_value_status = []
        all_logit_noncat_slot_status = []
        all_logit_spans = []
        all_start_char_idx = []
        all_end_char_idx = []

        if self.trainer.num_devices and self.trainer.world_size > 1:
            world_size = self.trainer.world_size
            for ind in range(world_size):
                all_example_id_num.append(torch.empty_like(example_id_num))
                all_service_id.append(torch.empty_like(service_id))
                all_logit_intent_status.append(torch.empty_like(logit_intent_status))
                all_logit_req_slot_status.append(torch.empty_like(logit_req_slot_status))
                all_logit_cat_slot_status.append(torch.empty_like(logit_cat_slot_status))
                all_logit_cat_slot_value_status.append(torch.empty_like(logit_cat_slot_value_status))
                all_logit_noncat_slot_status.append(torch.empty_like(logit_noncat_slot_status))
                all_logit_spans.append(torch.empty_like(logit_spans))
                all_start_char_idx.append(torch.empty_like(start_char_idx))
                all_end_char_idx.append(torch.empty_like(end_char_idx))

            torch.distributed.all_gather(all_example_id_num, example_id_num)
            torch.distributed.all_gather(all_service_id, service_id)
            torch.distributed.all_gather(all_logit_intent_status, logit_intent_status)
            torch.distributed.all_gather(all_logit_req_slot_status, logit_req_slot_status)
            torch.distributed.all_gather(all_logit_cat_slot_status, logit_cat_slot_status)
            torch.distributed.all_gather(all_logit_cat_slot_value_status, logit_cat_slot_value_status)
            torch.distributed.all_gather(all_logit_noncat_slot_status, logit_noncat_slot_status)
            torch.distributed.all_gather(all_logit_spans, logit_spans)
            torch.distributed.all_gather(all_start_char_idx, start_char_idx)
            torch.distributed.all_gather(all_end_char_idx, end_char_idx)
        else:
            all_example_id_num.append(example_id_num)
            all_service_id.append(service_id)
            all_logit_intent_status.append(logit_intent_status)
            all_logit_req_slot_status.append(logit_req_slot_status)
            all_logit_cat_slot_status.append(logit_cat_slot_status)
            all_logit_cat_slot_value_status.append(logit_cat_slot_value_status)
            all_logit_noncat_slot_status.append(logit_noncat_slot_status)
            all_logit_spans.append(logit_spans)
            all_start_char_idx.append(start_char_idx)
            all_end_char_idx.append(end_char_idx)

        # after this: all_x is list of tensors, of length world_size
        example_id_num = torch.cat(all_example_id_num)
        service_id = torch.cat(all_service_id)
        logit_intent_status = torch.cat(all_logit_intent_status)
        logit_req_slot_status = torch.cat(all_logit_req_slot_status)
        logit_cat_slot_status = torch.cat(all_logit_cat_slot_status)
        logit_cat_slot_value_status = torch.cat(all_logit_cat_slot_value_status)
        logit_noncat_slot_status = torch.cat(all_logit_noncat_slot_status)
        logit_spans = torch.cat(all_logit_spans)
        start_char_idx = torch.cat(all_start_char_idx)
        end_char_idx = torch.cat(all_end_char_idx)

        intent_status = torch.nn.Sigmoid()(logit_intent_status)

        # Scores are output for each requested slot.
        req_slot_status = torch.nn.Sigmoid()(logit_req_slot_status)

        # For categorical slots, the status of each slot and the predicted value are output.
        cat_slot_status_dist = torch.nn.Softmax(dim=-1)(logit_cat_slot_status)

        cat_slot_status = torch.argmax(logit_cat_slot_status, axis=-1)
        cat_slot_status_p = torch.max(cat_slot_status_dist, axis=-1)[0]
        cat_slot_value_status = torch.nn.Sigmoid()(logit_cat_slot_value_status)

        # For non-categorical slots, the status of each slot and the indices for spans are output.
        noncat_slot_status_dist = torch.nn.Softmax(dim=-1)(logit_noncat_slot_status)

        noncat_slot_status = torch.argmax(logit_noncat_slot_status, axis=-1)
        noncat_slot_status_p = torch.max(noncat_slot_status_dist, axis=-1)[0]

        softmax = torch.nn.Softmax(dim=1)

        scores = softmax(logit_spans)
        start_scores, end_scores = torch.unbind(scores, dim=-1)

        batch_size, max_num_tokens = end_scores.size()
        # Find the span with the maximum sum of scores for start and end indices.
        total_scores = torch.unsqueeze(start_scores, axis=2) + torch.unsqueeze(end_scores, axis=1)
        start_idx = torch.arange(max_num_tokens, device=total_scores.get_device()).view(1, -1, 1)
        end_idx = torch.arange(max_num_tokens, device=total_scores.get_device()).view(1, 1, -1)
        invalid_index_mask = (start_idx > end_idx).repeat(batch_size, 1, 1)
        total_scores = torch.where(
            invalid_index_mask,
            torch.zeros(total_scores.size(), device=total_scores.get_device(), dtype=total_scores.dtype),
            total_scores,
        )
        max_span_index = torch.argmax(total_scores.view(-1, max_num_tokens ** 2), axis=-1)
        max_span_p = torch.max(total_scores.view(-1, max_num_tokens ** 2), axis=-1)[0]

        span_start_index = torch.floor_divide(max_span_index, max_num_tokens)
        span_end_index = torch.fmod(max_span_index, max_num_tokens)

        tensors = {
            'example_id_num': example_id_num,
            'service_id': service_id,
            'intent_status': intent_status,
            'req_slot_status': req_slot_status,
            'cat_slot_status': cat_slot_status,
            'cat_slot_status_p': cat_slot_status_p,
            'cat_slot_value_status': cat_slot_value_status,
            'noncat_slot_status': noncat_slot_status,
            'noncat_slot_status_p': noncat_slot_status_p,
            'noncat_slot_p': max_span_p,
            'noncat_slot_start': span_start_index,
            'noncat_slot_end': span_end_index,
            'noncat_alignment_start': start_char_idx,
            'noncat_alignment_end': end_char_idx,
        }
        return loss, tensors

    def multi_validation_epoch_end(self, outputs: List[dict], dataloader_idx: int = 0):
        """
        Called at the end of validation to post process outputs into human readable format
        Args:
            outputs: list of individual outputs of each validation step
            dataloader_idx: dataloader index
        """
        avg_loss = torch.stack([x[f'val_loss'] for x in outputs]).mean()
        split = self._validation_names[dataloader_idx][:-1]
        dataloader = self._validation_dl[dataloader_idx]
        metrics = self.multi_eval_epoch_end_helper(outputs=outputs, split=split, dataloader=dataloader)

        for k, v in metrics.items():
            self.log(f'{split}_{k}', v, rank_zero_only=True)

        self.log(f'val_loss', avg_loss, prog_bar=True, rank_zero_only=True)

    def multi_test_epoch_end(self, outputs: List[dict], dataloader_idx: int = 0):
        """
        Called at the end of test to post process outputs into human readable format
        Args:
            outputs: list of individual outputs of each test step
            dataloader_idx: dataloader index
        """
        avg_loss = torch.stack([x[f'test_loss'] for x in outputs]).mean()
        split = self._test_names[dataloader_idx][:-1]
        dataloader = self._test_dl[dataloader_idx]
        metrics = self.multi_eval_epoch_end_helper(outputs=outputs, split=split, dataloader=dataloader)

        for k, v in metrics.items():
            self.log(f'{split}_{k}', v, rank_zero_only=True)

        self.log(f'test_loss', avg_loss, prog_bar=True, rank_zero_only=True)

    def multi_eval_epoch_end_helper(
        self, outputs: List[dict], split: str, dataloader: torch.utils.data.DataLoader
    ) -> dict:
        """
        Helper called at the end of evaluation to post process outputs into human readable format
        Args:
            outputs: list of individual outputs of each step
            split: data split
            dataloader: dataloader
        Returns:
            metrics: metrics collection
        """

        def get_str_example_id(split: str, ids_to_service_names_dict: dict, example_id_num: torch.Tensor) -> str:
            """
            Constructs string representation of example ID
            Args:
                split: evaluation data split
                ids_to_service_names_dict: id to service name mapping
                example_id_num: tensor example id
            """

            def format_turn_id(ex_id_num):
                dialog_id_1, dialog_id_2, turn_id, service_id, model_task_id, slot_intent_id, value_id = ex_id_num
                return "{}-{}_{:05d}-{:02d}-{}-{}-{}-{}".format(
                    split,
                    dialog_id_1,
                    dialog_id_2,
                    turn_id,
                    ids_to_service_names_dict[service_id],
                    model_task_id,
                    slot_intent_id,
                    value_id,
                )

            return list(map(format_turn_id, tensor2list(example_id_num)))

        def combine_predictions_in_example(predictions: dict, batch_size: int):
            '''
            Combines predicted values to a single example. 
            Args:
                predictions: predictions ordered by keys then batch
                batch_size: batch size
            Returns:
                examples_preds: predictions ordered by batch then key
            '''
            examples_preds = [{} for _ in range(batch_size)]
            for k, v in predictions.items():
                if k != 'example_id':
                    v = torch.chunk(v, batch_size)

                for i in range(batch_size):
                    if k == 'example_id':
                        examples_preds[i][k] = v[i]
                    else:
                        examples_preds[i][k] = v[i].view(-1)
            return examples_preds

        example_id_num = torch.cat([x[f'tensors']['example_id_num'] for x in outputs])
        service_id = torch.cat([x[f'tensors']['service_id'] for x in outputs])
        intent_status = torch.cat([x[f'tensors']['intent_status'] for x in outputs])
        req_slot_status = torch.cat([x[f'tensors']['req_slot_status'] for x in outputs])
        cat_slot_status = torch.cat([x[f'tensors']['cat_slot_status'] for x in outputs])
        cat_slot_status_p = torch.cat([x[f'tensors']['cat_slot_status_p'] for x in outputs])
        cat_slot_value_status = torch.cat([x[f'tensors']['cat_slot_value_status'] for x in outputs])
        noncat_slot_status = torch.cat([x[f'tensors']['noncat_slot_status'] for x in outputs])
        noncat_slot_status_p = torch.cat([x[f'tensors']['noncat_slot_status_p'] for x in outputs])
        noncat_slot_p = torch.cat([x[f'tensors']['noncat_slot_p'] for x in outputs])
        noncat_slot_start = torch.cat([x[f'tensors']['noncat_slot_start'] for x in outputs])
        noncat_slot_end = torch.cat([x[f'tensors']['noncat_slot_end'] for x in outputs])
        noncat_alignment_start = torch.cat([x[f'tensors']['noncat_alignment_start'] for x in outputs])
        noncat_alignment_end = torch.cat([x[f'tensors']['noncat_alignment_end'] for x in outputs])

        ids_to_service_names_dict = self.dialogues_processor.schemas._services_id_to_vocab
        example_id = get_str_example_id(dataloader.dataset, ids_to_service_names_dict, example_id_num)

        metrics = {}
        try:
            prediction_dir = self.trainer.log_dir if self.trainer.log_dir is not None else ""
        except:
            prediction_dir = ""

        if self.trainer.global_rank == 0:
            prediction_dir = os.path.join(
                prediction_dir, 'predictions', 'pred_res_{}_{}'.format(split, self._cfg.dataset.task_name)
            )
            os.makedirs(prediction_dir, exist_ok=True)

            input_json_files = DialogueSGDDataProcessor.get_dialogue_files(
                self._cfg.dataset.data_dir, split, self._cfg.dataset.task_name
            )

            predictions = {}
            predictions['example_id'] = example_id
            predictions['service_id'] = service_id
            predictions['intent_status'] = intent_status
            predictions['req_slot_status'] = req_slot_status
            predictions['cat_slot_status'] = cat_slot_status
            predictions['cat_slot_status_p'] = cat_slot_status_p
            predictions['cat_slot_value_status'] = cat_slot_value_status
            predictions['noncat_slot_status'] = noncat_slot_status
            predictions['noncat_slot_status_p'] = noncat_slot_status_p
            predictions['noncat_slot_p'] = noncat_slot_p
            predictions['noncat_slot_start'] = noncat_slot_start
            predictions['noncat_slot_end'] = noncat_slot_end
            predictions['noncat_alignment_start'] = noncat_alignment_start
            predictions['noncat_alignment_end'] = noncat_alignment_end

            in_domain_services = get_in_domain_services(
                os.path.join(self._cfg.dataset.data_dir, split, "schema.json"),
                self.dialogues_processor.get_seen_services("train"),
            )
            predictions = combine_predictions_in_example(predictions, service_id.shape[0])

            # write predictions to file in Dstc8/SGD format
            write_predictions_to_file(
                predictions,
                input_json_files,
                output_dir=prediction_dir,
                schemas=self.dialogues_processor.schemas,
                state_tracker=self._cfg.dataset.state_tracker,
                eval_debug=False,
                in_domain_services=in_domain_services,
            )
            metrics = evaluate(
                prediction_dir,
                self._cfg.dataset.data_dir,
                split,
                in_domain_services,
                joint_acc_across_turn=self._cfg.dataset.joint_acc_across_turn,
                use_fuzzy_match=self._cfg.dataset.use_fuzzy_match,
            )

        return metrics

    def prepare_data(self):
        """
        Preprocessed schema and dialogues and caches this
        """
        if self.data_prepared:
            return

        self.dialogues_processor = DialogueSGDDataProcessor(
            data_dir=self._cfg.dataset.data_dir,
            dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
            tokenizer=self.tokenizer,
            cfg=self._cfg.dataset,
        )

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

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split=val_data_config.ds_item)

    def setup_test_data(self, test_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, split=test_data_config.ds_item)

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        # dataset = SGDDataset(dataset_split=split, dialogues_processor=self.dialogues_processor)

        dataset = DialogueSGDBERTDataset(
            dataset_split=split,
            dialogues_processor=self.dialogues_processor,
            tokenizer=self.dialogues_processor._tokenizer,
            schemas=self.dialogues_processor.schemas,
            schema_config=self.dialogues_processor.schema_config,
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

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="sgdqa_bertbasecased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/sgdqa_bertbasecased/versions/1.0.0/files/sgdqa_bertbasecased.nemo",
                description="Dialogue State Tracking model finetuned from NeMo BERT Base Cased on Google SGD dataset which has a joint goal accuracy of 59.72% on dev set and 45.85% on test set.",
            )
        )
        return result
