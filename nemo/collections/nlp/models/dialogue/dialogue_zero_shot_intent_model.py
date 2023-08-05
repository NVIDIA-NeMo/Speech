# Copyright 2018 The HuggingFace Inc. team.
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

import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nemo.collections.nlp.data.dialogue import DialogueSGDDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.assistant_data_processor import DialogueAssistantDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.design_data_processor import DialogueDesignDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_zero_shot_intent_dataset import DialogueZeroShotIntentDataset
from nemo.collections.nlp.data.zero_shot_intent_recognition.zero_shot_intent_dataset import (
    ZeroShotIntentInferenceDataset,
    calc_class_weights_from_dataloader,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueGenerationMetrics
from nemo.collections.nlp.models import TextClassificationModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = ['DialogueZeroShotIntentModel']


class DialogueZeroShotIntentModel(TextClassificationModel):
    """TextClassificationModel to be trained on two- or three-class textual entailment data, to be used for zero shot intent recognition."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.cfg = cfg
        super().__init__(cfg=cfg, trainer=trainer)

        if self.cfg.library == 'megatron':
            # zero shot intent classification loading
            # cannot directly load as .nemo uses the pre-refactor model
            # therefore transfer its attributes over
            if self.cfg.original_nemo_checkpoint is not None:
                original_model = DialogueZeroShotIntentModel.restore_from(self.cfg.original_nemo_checkpoint)
                self.classifier = original_model.classifier
                self.bert_model = original_model.bert_model
                self.loss = original_model.loss
                self.classification_report = original_model.classification_report
        elif self.cfg.library == "huggingface":
            self.nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
            self.bert_model = self.nli_model.model
            self.classifier = self.nli_model.classification_head
            original_model = DialogueZeroShotIntentModel.restore_from(self.cfg.original_nemo_checkpoint)
            self.loss = original_model.loss
            self.classification_report = original_model.classification_report
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            self.tokenizer.max_seq_length = self.cfg.dataset.max_seq_length

    def _setup_dataloader_from_config(self, cfg: DictConfig, dataset_split) -> 'torch.utils.data.DataLoader':
        if self._cfg.dataset.task == "zero_shot":
            self.data_processor = DialogueAssistantDataProcessor(
                self.cfg.data_dir, self.tokenizer, cfg=self.cfg.dataset
            )
        elif self._cfg.dataset.task == "design":
            self.data_processor = DialogueDesignDataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset
            )
        elif self._cfg.dataset.task == 'sgd':
            self.data_processor = DialogueSGDDataProcessor(
                data_dir=self._cfg.dataset.data_dir,
                dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
                tokenizer=self.tokenizer,
                cfg=self._cfg.dataset,
            )
        else:
            raise ValueError("Only zero_shot, design and sgd supported for Zero Shot Intent Model")

        dataset = DialogueZeroShotIntentDataset(
            dataset_split,
            self.data_processor,
            self.tokenizer,
            self.cfg.dataset,  # this is the model.dataset cfg, which is diff from train_ds cfg etc
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.cfg.library == 'megatron':
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            logits = self.classifier(hidden_states=hidden_states)
        elif self.cfg.library == 'huggingface':
            output = self.nli_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output['logits']
        return logits

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config:
            logging.info(
                f"Dataloader config or file_name for the training set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(train_data_config, "train")

        # calculate the class weights to be used in the loss function
        if self.cfg.dataset.class_balancing == 'weighted_loss':
            self.class_weights = calc_class_weights_from_dataloader(
                self._train_dl, self.cfg.dataset.num_classes, self.cfg.dataset.data_dir
            )
        else:
            self.class_weights = None
        # we need to create/update the loss module by using the weights calculated from the training data
        self.create_loss_module()

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config:
            logging.info(
                f"Dataloader config or file_path for the validation data set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(val_data_config, "dev")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config:
            logging.info(
                f"Dataloader config or file_path for the test data set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(test_data_config, "test")

    def _setup_infer_dataloader(
        self,
        queries: List[str],
        candidate_labels: List[str],
        hypothesis_template=str,
        batch_size=1,
        max_seq_length: int = -1,
    ) -> 'torch.utils.data.DataLoader':
        """
        Setup method for inference data loader. Here the premise-hypothesis pairs are made from queries and candidate labels.

        Args:
            queries: the queries to classify
            candidate_labels: strings to be used as labels
            hypothesis_template: the template used to turn each label into an NLI-style hypothesis. Must include a {}
                or similar syntax for the candidate label to be inserted.
            batch_size: batch size to use during inference
            max_seq_length: maximum length of queries, default is -1 for no limit
        Returns:
            A pytorch DataLoader.
        """
        dataset = ZeroShotIntentInferenceDataset(
            queries=queries,
            candidate_labels=candidate_labels,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            hypothesis_template=hypothesis_template,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    def validation_step(self, batch, batch_idx, split='val'):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        preds = torch.argmax(logits, axis=-1)

        tp, fn, fp, _ = self.classification_report(preds, labels)

        loss = {
            'val_loss': val_loss,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'logits': logits,
            'input_ids': input_ids,
            'labels': labels,
        }
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self, split="val"):
        """
        Get metrics based on the candidate label with the highest predicted likelihood and the ground truth label for intent
        """
        output_logits = torch.cat([output['logits'] for output in self.validation_step_outputs], dim=0)
        output_input_ids = torch.cat([output['input_ids'] for output in self.validation_step_outputs], dim=0)
        output_labels = torch.cat([output['labels'] for output in self.validation_step_outputs], dim=0)

        if self.cfg.library == 'huggingface':
            entail_logits = output_logits[..., 2]
            decoded_input_ids = [self.tokenizer.decode(output_input_ids[i]) for i in range(len(output_input_ids))]
            utterance_candidate_pairs = [i.split(self.tokenizer.sep_token) for i in decoded_input_ids]
            utterances = [
                i[0].replace(self.tokenizer.bos_token, '').replace(self.tokenizer.eos_token, '')
                for i in utterance_candidate_pairs
            ]

        elif self.cfg.library == 'megatron':
            entail_logits = output_logits[..., 1]
            decoded_input_ids = [
                self.tokenizer.tokenizer.decode(output_input_ids[i]) for i in range(len(output_input_ids))
            ]
            utterance_candidate_pairs = [i.split(self.tokenizer.tokenizer.sep_token) for i in decoded_input_ids]
            utterances = [
                i[0].replace(self.tokenizer.tokenizer.bos_token, '').replace(self.tokenizer.tokenizer.eos_token, '')
                for i in utterance_candidate_pairs
            ]

        # account for uncased tokenization
        candidates = [
            i[1]
            .replace(self.cfg.dataset.prompt_template.lower(), '')
            .replace(self.cfg.dataset.prompt_template, '')
            .strip()
            for i in utterance_candidate_pairs
        ]
        utterance_to_idx = defaultdict(list)
        for idx, utterance in enumerate(utterances):
            utterance_to_idx[utterance].append(idx)

        predicted_labels = []
        ground_truth_labels = []
        utterances = []
        for utterance, idxs in utterance_to_idx.items():
            utterance_candidates = [candidates[idx] for idx in idxs]
            logits = [entail_logits[idx].item() for idx in idxs]
            labels = [output_labels[idx].item() for idx in idxs]
            correct_candidate = utterance_candidates[np.argmax(labels)]
            predicted_candidate = utterance_candidates[np.argmax(logits)]
            predicted_labels.append(predicted_candidate)
            ground_truth_labels.append(correct_candidate)
            utterances.append(utterance)

        os.makedirs(self.cfg.dataset.dialogues_example_dir, exist_ok=True)
        filename = os.path.join(self.cfg.dataset.dialogues_example_dir, "test_predictions.jsonl")

        DialogueGenerationMetrics.save_predictions(
            filename, predicted_labels, ground_truth_labels, utterances,
        )

        label_to_ids = {label: idx for idx, label in enumerate(list(set(predicted_labels + ground_truth_labels)))}
        self.classification_report = ClassificationReport(
            num_classes=len(label_to_ids), mode='micro', label_ids=label_to_ids, dist_sync_on_step=True
        ).to(output_logits[0].device)
        predicted_label_ids = torch.tensor([label_to_ids[label] for label in predicted_labels]).to(
            output_logits[0].device
        )
        ground_truth_label_ids = torch.tensor([label_to_ids[label] for label in ground_truth_labels]).to(
            output_logits[0].device
        )

        tp, fn, fp, _ = self.classification_report(predicted_label_ids, ground_truth_label_ids)
        precision, recall, f1, report = self.classification_report.compute()
        label_acc = np.mean([int(predicted_labels[i] == ground_truth_labels[i]) for i in range(len(predicted_labels))])

        avg_loss = torch.stack([x[f'val_loss'] for x in self.validation_step_outputs]).mean()

        logging.info(report)

        self.log('unified_precision', precision)
        self.log('unified_f1', f1)
        self.log('unified_recall', recall)
        self.log('unfied_accuracy', label_acc * 100)
        self.log('val_loss', avg_loss, prog_bar=True)

        self.validation_step_outputs.clear()  # free memory
        self.classification_report.reset()

    def predict(
        self,
        queries: Union[str, List[str]],
        candidate_labels: Union[str, List[str]],
        hypothesis_template='This example is {}.',
        batch_size=1,
        multi_label=True,
        entailment_idx=1,
        contradiction_idx=0,
    ) -> List[Dict]:

        """
        Given a list of queries and a list of candidate labels, return a ranked list of labels and scores for each query.

        Example usage:
            queries = ["I'd like a veggie burger, fries, and a coke", "Turn off the lights in the living room",]
            candidate_labels = ["Food order", "Change lighting"]
            model.predict(queries, candidate_labels)

        Example output:
            [{'sentence': "I'd like a veggie burger, fries, and a coke",
              'labels': ['Food order', 'Change lighting'],
              'scores': [0.8557153344154358, 0.12036784738302231]},
             {'sentence': 'Turn off the lights in the living room',
              'labels': ['Change lighting', 'Food order'],
              'scores': [0.8506497144699097, 0.06594637036323547]}]


        Args:
            queries: the query or list of queries to classify
            candidate_labels: string or list of strings to be used as labels
            hypothesis_template: the template used to turn each label into an NLI-style hypothesis. Must include a {}
            or similar syntax for the candidate label to be inserted.
            batch_size: the batch size to use for inference.
            multi_label: whether or not multiple candidate labels can be true. If False, the scores are normalized
            such that all class probabilities sum to 1. If True, the labels are
            considered independent and probabilities are normalized for each candidate by doing a softmax of
            the entailment score vs. the contradiction score.
            entailment_idx: the index of the "entailment" class in the trained model; models trained on MNLI
             using NeMo's glue_benchmark.py or zero_shot_intent_model.py use an index of 1 by default.
            contradiction_idx: the index of the "contradiction" class in the trained model; models trained on MNLI
             using NeMo's glue_benchmark.py or zero_shot_intent_model.py use an index of 0 by default.

        Returns:
            list of dictionaries; one dict per input query. Each dict has keys "sentence", "labels", "scores".
            labels and scores are parallel lists (with each score corresponding to the label at the same index),
                 sorted from highest to lowest score.

        """
        if not queries:
            raise ValueError("No queries were passed for classification!")
        if not candidate_labels:
            raise ValueError("No candidate labels were provided!")

        queries = [queries] if isinstance(queries, str) else queries
        candidate_labels = [candidate_labels] if isinstance(candidate_labels, str) else candidate_labels

        if len(candidate_labels) == 1:
            multi_label = True

        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            infer_datalayer = self._setup_infer_dataloader(
                queries,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                batch_size=batch_size,
                max_seq_length=self._cfg.dataset.max_seq_length,
            )

            all_batch_logits = []
            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, _ = batch

                logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )
                all_batch_logits.append(logits.detach().cpu().numpy())

            all_logits = np.concatenate(all_batch_logits)
            outputs = all_logits.reshape((len(queries), len(candidate_labels), -1))

            if not multi_label:
                # softmax the "entailment" logits over all candidate labels
                entail_logits = outputs[..., entailment_idx]
                scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
            else:
                # softmax over the entailment vs. contradiction dim for each label independently
                entail_contr_logits = outputs[..., [contradiction_idx, entailment_idx]]
                scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
                scores = scores[..., 1]

            result = []
            for i in range(len(queries)):
                sorted_idxs = list(reversed(scores[i].argsort()))
                result.append(
                    {
                        "sentence": queries[i],
                        "labels": [candidate_labels[j] for j in sorted_idxs],
                        "scores": scores[i][sorted_idxs].tolist(),
                    }
                )

        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return result

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained models which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="zeroshotintent_en_bert_base_uncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/zeroshotintent_en_bert_base_uncased/versions/1.4.1/files/zeroshotintent_en_bert_base_uncased.nemo",
                description="DialogueZeroShotIntentModel trained by fine tuning BERT-base-uncased on the MNLI (Multi-Genre Natural Language Inference) dataset, which achieves an accuracy of 84.9% and 84.8% on the matched and mismatched dev sets, respectively.",
            )
        )
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="zeroshotintent_en_megatron_uncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/zeroshotintent_en_megatron_uncased/versions/1.4.1/files/zeroshotintent_en_megatron_uncased.nemo",
                description="DialogueZeroShotIntentModel trained by fine tuning Megatron-BERT-345m=M-uncased on the MNLI (Multi-Genre Natural Language Inference) dataset, which achieves an accuracy of 90.0% and 89.9% on the matched and mismatched dev sets, respectively",
            )
        )
        return result
