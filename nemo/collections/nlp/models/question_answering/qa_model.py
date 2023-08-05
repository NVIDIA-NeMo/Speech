# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.cuda.amp import autocast

from nemo.collections.common.losses import SpanningLoss
from nemo.collections.nlp.data import SquadDataset
from nemo.collections.nlp.data.question_answering_squad.qa_squad_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
)
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging

__all__ = ['QAModel']


class QAModel(NLPModel):
    """
    BERT encoder with QA head training.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=cfg.token_classifier.num_classes,
            num_layers=cfg.token_classifier.num_layers,
            activation=cfg.token_classifier.activation,
            log_softmax=cfg.token_classifier.log_softmax,
            dropout=cfg.token_classifier.dropout,
            use_transformer_init=cfg.token_classifier.use_transformer_init,
        )

        self.loss = SpanningLoss()

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        with autocast():
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            logits = self.classifier(hidden_states=hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, _, _ = self.loss(logits=logits, start_positions=start_positions, end_positions=end_positions)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx):
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, start_logits, end_logits = self.loss(
            logits=logits, start_positions=start_positions, end_positions=end_positions
        )

        tensors = {
            'unique_ids': unique_ids,
            'start_logits': start_logits,
            'end_logits': end_logits,
        }
        loss = {f'{prefix}_loss': loss, f'{prefix}_tensors': tensors}
        self.validation_step_outputs.append(loss) if prefix == 'val' else self.test_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        if self.trainer.testing:
            prefix = 'test'
            outputs = self.test_step_outputs
        else:
            prefix = 'val'
            outputs = self.validation_step_outputs

        avg_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        unique_ids = torch.cat([x[f'{prefix}_tensors']['unique_ids'] for x in outputs])
        start_logits = torch.cat([x[f'{prefix}_tensors']['start_logits'] for x in outputs])
        end_logits = torch.cat([x[f'{prefix}_tensors']['end_logits'] for x in outputs])

        all_unique_ids = []
        all_start_logits = []
        all_end_logits = []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_unique_ids.append(torch.empty_like(unique_ids))
                all_start_logits.append(torch.empty_like(start_logits))
                all_end_logits.append(torch.empty_like(end_logits))
            torch.distributed.all_gather(all_unique_ids, unique_ids)
            torch.distributed.all_gather(all_start_logits, start_logits)
            torch.distributed.all_gather(all_end_logits, end_logits)
        else:
            all_unique_ids.append(unique_ids)
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

        exact_match, f1, all_predictions, all_nbest = -1, -1, [], []
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

            unique_ids = []
            start_logits = []
            end_logits = []
            for u in all_unique_ids:
                unique_ids.extend(tensor2list(u))
            for u in all_start_logits:
                start_logits.extend(tensor2list(u))
            for u in all_end_logits:
                end_logits.extend(tensor2list(u))

            eval_dataset = self._test_dl.dataset if self.trainer.testing else self._validation_dl.dataset
            exact_match, f1, all_predictions, all_nbest = eval_dataset.evaluate(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._cfg.dataset.n_best_size,
                max_answer_length=self._cfg.dataset.max_answer_length,
                version_2_with_negative=self._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._cfg.dataset.null_score_diff_threshold,
                do_lower_case=self._cfg.dataset.do_lower_case,
            )

        logging.info(f"{prefix} exact match {exact_match}")
        logging.info(f"{prefix} f1 {f1}")

        self.log(f'{prefix}_loss', avg_loss)
        self.log(f'{prefix}_exact_match', exact_match)
        self.log(f'{prefix}_f1', f1)
        self.validation_step_outputs.clear() if prefix == 'val' else self.test_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    @torch.no_grad()
    def inference(
        self,
        file: str,
        batch_size: int = 1,
        num_samples: int = -1,
        output_nbest_file: Optional[str] = None,
        output_prediction_file: Optional[str] = None,
    ):
        """
        Get prediction for unlabeled inference data

        Args:
            file: inference data
            batch_size: batch size to use during inference
            num_samples: number of samples to use of inference data. Default: -1 if all data should be used.
            output_nbest_file: optional output file for writing out nbest list
            output_prediction_file: optional output file for writing out predictions
            
        Returns:
            model predictions, model nbest list
        """
        # store predictions for all queries in a single list
        all_predictions = []
        all_nbest = []
        mode = self.training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # Switch model to evaluation mode
            self.eval()
            self.to(device)
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            dataloader_cfg = {
                "batch_size": batch_size,
                "file": file,
                "shuffle": False,
                "num_samples": num_samples,
                'num_workers': 2,
                'pin_memory': False,
                'drop_last': False,
            }
            dataloader_cfg = OmegaConf.create(dataloader_cfg)
            infer_datalayer = self._setup_dataloader_from_config(cfg=dataloader_cfg, mode=INFERENCE_MODE)

            all_logits = []
            all_unique_ids = []
            for i, batch in enumerate(infer_datalayer):
                input_ids, token_type_ids, attention_mask, unique_ids = batch
                logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=token_type_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                all_logits.append(logits)
                all_unique_ids.append(unique_ids)
            logits = torch.cat(all_logits)
            unique_ids = tensor2list(torch.cat(all_unique_ids))
            s, e = logits.split(dim=-1, split_size=1)
            start_logits = tensor2list(s.squeeze(-1))
            end_logits = tensor2list(e.squeeze(-1))
            (all_predictions, all_nbest, scores_diff) = infer_datalayer.dataset.get_predictions(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._cfg.dataset.n_best_size,
                max_answer_length=self._cfg.dataset.max_answer_length,
                version_2_with_negative=self._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._cfg.dataset.null_score_diff_threshold,
                do_lower_case=self._cfg.dataset.do_lower_case,
            )

            with open(file, 'r') as test_file_fp:
                test_data = json.load(test_file_fp)["data"]
                id_to_question_mapping = {}
                for title in test_data:
                    for par in title["paragraphs"]:
                        for question in par["qas"]:
                            id_to_question_mapping[question["id"]] = question["question"]

            for question_id in all_predictions:
                all_predictions[question_id] = (id_to_question_mapping[question_id], all_predictions[question_id])

            if output_nbest_file is not None:
                with open(output_nbest_file, "w") as writer:
                    writer.write(json.dumps(all_nbest, indent=4) + "\n")
            if output_prediction_file is not None:
                with open(output_prediction_file, "w") as writer:
                    writer.write(json.dumps(all_predictions, indent=4) + "\n")

        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)

        return all_predictions, all_nbest

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode=TRAINING_MODE)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode=EVALUATION_MODE)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.file is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode=EVALUATION_MODE)

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        dataset = SquadDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.file,
            keep_doc_spans='all',  # self._cfg.dataset.keep_doc_spans,
            doc_stride=self._cfg.dataset.doc_stride,
            max_query_length=self._cfg.dataset.max_query_length,
            max_seq_length=self._cfg.dataset.max_seq_length,
            version_2_with_negative=self._cfg.dataset.version_2_with_negative,
            num_samples=cfg.num_samples,
            mode=mode,
            use_cache=self._cfg.dataset.use_cache,
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
                pretrained_model_name="qa_squadv1.1_bertbase",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv1_1_bertbase/versions/1.0.0rc1/files/qa_squadv1.1_bertbase.nemo",
                description="Question answering model finetuned from NeMo BERT Base Uncased on SQuAD v1.1 dataset which obtains an exact match (EM) score of 82.78% and an F1 score of 89.97%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv2.0_bertbase",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv2_0_bertbase/versions/1.0.0rc1/files/qa_squadv2.0_bertbase.nemo",
                description="Question answering model finetuned from NeMo BERT Base Uncased on SQuAD v2.0 dataset which obtains an exact match (EM) score of 75.04% and an F1 score of 78.08%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv1_1_bertlarge",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv1_1_bertlarge/versions/1.0.0rc1/files/qa_squadv1.1_bertlarge.nemo",
                description="Question answering model finetuned from NeMo BERT Large Uncased on SQuAD v1.1 dataset which obtains an exact match (EM) score of 85.44% and an F1 score of 92.06%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv2.0_bertlarge",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv2_0_bertlarge/versions/1.0.0rc1/files/qa_squadv2.0_bertlarge.nemo",
                description="Question answering model finetuned from NeMo BERT Large Uncased on SQuAD v2.0 dataset which obtains an exact match (EM) score of 80.22% and an F1 score of 83.05%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv1_1_megatron_cased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv1_1_megatron_cased/versions/1.0.0rc1/files/qa_squadv1.1_megatron_cased.nemo",
                description="Question answering model finetuned from Megatron Cased on SQuAD v1.1 dataset which obtains an exact match (EM) score of 88.18% and an F1 score of 94.07%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv2.0_megatron_cased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv2_0_megatron_cased/versions/1.0.0rc1/files/qa_squadv2.0_megatron_cased.nemo",
                description="Question answering model finetuned from Megatron Cased on SQuAD v2.0 dataset which obtains an exact match (EM) score of 84.73% and an F1 score of 87.89%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv1.1_megatron_uncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv1_1_megatron_uncased/versions/1.0.0rc1/files/qa_squadv1.1_megatron_uncased.nemo",
                description="Question answering model finetuned from Megatron Unased on SQuAD v1.1 dataset which obtains an exact match (EM) score of 87.61% and an F1 score of 94.00%.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="qa_squadv2.0_megatron_uncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/qa_squadv2_0_megatron_uncased/versions/1.0.0rc1/files/qa_squadv2.0_megatron_uncased.nemo",
                description="Question answering model finetuned from Megatron Uncased on SQuAD v2.0 dataset which obtains an exact match (EM) score of 84.48% and an F1 score of 87.65%.",
            )
        )
        return result
