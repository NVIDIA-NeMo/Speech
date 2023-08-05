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

import collections
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers.models.bert.tokenization_bert import BasicTokenizer

from nemo.collections.common.losses import SpanningLoss
from nemo.collections.common.parts.utils import _compute_softmax
from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import QAProcessor
from nemo.collections.nlp.data.question_answering.dataset.qa_bert_dataset import BERTQADataset
from nemo.collections.nlp.metrics.qa_metrics import QAMetrics
from nemo.collections.nlp.models.question_answering.qa_base_model import BaseQAModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


class BERTQAModel(BaseQAModel):
    """ BERT model with a QA (token classification) head """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=False)
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

    def training_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, _, _ = self.loss(logits=logits, start_positions=start_positions, end_positions=end_positions)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('lr', lr, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx):
        prefix = "test" if self.trainer.testing else "val"

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
        if prefix == "val":
            self.validation_step_outputs.append(loss)
        else:
            self.test_step_outputs.append(loss)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        prefix = "test" if self.trainer.testing else "val"

        if prefix == 'val':
            avg_loss = torch.stack([x[f'{prefix}_loss'] for x in self.validation_step_outputs]).mean()

            unique_ids = torch.cat([x[f'{prefix}_tensors']['unique_ids'] for x in self.validation_step_outputs])
            start_logits = torch.cat([x[f'{prefix}_tensors']['start_logits'] for x in self.validation_step_outputs])
            end_logits = torch.cat([x[f'{prefix}_tensors']['end_logits'] for x in self.validation_step_outputs])
            self.validation_step_outputs.clear()  # free memory
        else:
            avg_loss = torch.stack([x[f'{prefix}_loss'] for x in self.test_step_outputs]).mean()

            unique_ids = torch.cat([x[f'{prefix}_tensors']['unique_ids'] for x in self.test_step_outputs])
            start_logits = torch.cat([x[f'{prefix}_tensors']['start_logits'] for x in self.test_step_outputs])
            end_logits = torch.cat([x[f'{prefix}_tensors']['end_logits'] for x in self.test_step_outputs])
            self.test_step_outputs.clear()  # free memory

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

        eval_results, all_predictions, all_nbest = {}, [], []
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
            eval_results, _, _ = self.evaluate(
                eval_dataset.features,
                eval_dataset.examples,
                eval_dataset.processor,
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._cfg.dataset.n_best_size,
                max_answer_length=self._cfg.dataset.max_answer_length,
                version_2_with_negative=self._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._cfg.dataset.null_score_diff_threshold,
                do_lower_case=self._cfg.dataset.do_lower_case,
            )

        self.log(f'{prefix}_loss', avg_loss)
        for eval_key in eval_results:
            logging.info(f"{prefix} {eval_key}: {eval_results[eval_key]}")
            self.log(f"{prefix}_{eval_key}", eval_results[eval_key])

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.cuda.amp.autocast():
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            logits = self.classifier(hidden_states=hidden_states)

        return logits

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
        device = "cuda" if isinstance(self.trainer.device_ids, list) else "cpu"

        try:
            # Switch model to evaluation mode
            self.eval()
            self.to(device)
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            infer_datalayer = self.setup_inference_data(
                file, batch_size=batch_size, num_samples=num_samples, num_workers=2,
            )

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
            (all_predictions, all_nbest, scores_diff) = self.get_predictions(
                infer_datalayer.dataset.features,
                infer_datalayer.dataset.examples,
                infer_datalayer.dataset.processor,
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._cfg.dataset.n_best_size,
                max_answer_length=self._cfg.dataset.max_answer_length,
                version_2_with_negative=self._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._cfg.dataset.null_score_diff_threshold,
                do_lower_case=self._cfg.dataset.do_lower_case,
            )

            if output_prediction_file:
                QAMetrics.dump_predicted_answers_to_file(
                    output_prediction_file, infer_datalayer.dataset.examples, all_predictions,
                )

            if output_nbest_file:
                QAMetrics.dump_nbest_predictions_to_file(
                    output_nbest_file,
                    infer_datalayer.dataset.examples,
                    all_nbest,
                    keys_to_dump=["text", "probability"],
                )

        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)

        return all_predictions, all_nbest

    def evaluate(
        self,
        features: List,
        examples: List,
        processor: object,
        unique_ids: List[str],
        start_logits: List[List[float]],
        end_logits: List[List[float]],
        n_best_size: int,
        max_answer_length: int,
        do_lower_case: bool,
        version_2_with_negative: bool,
        null_score_diff_threshold: float,
    ):
        (all_predictions, all_nbest_json, scores_diff_json) = self.get_predictions(
            features,
            examples,
            processor,
            unique_ids,
            start_logits,
            end_logits,
            n_best_size,
            max_answer_length,
            do_lower_case,
            version_2_with_negative,
            null_score_diff_threshold,
        )

        eval_results = QAMetrics.evaluate_predictions(examples, all_predictions)

        return eval_results, all_predictions, all_nbest_json

    def get_predictions(
        self,
        features: List,
        examples: List,
        processor: object,
        unique_ids: List[int],
        start_logits: List[List[float]],
        end_logits: List[List[float]],
        n_best_size: int,
        max_answer_length: int,
        do_lower_case: bool,
        version_2_with_negative: bool,
        null_score_diff_threshold: float,
    ):
        example_index_to_features = collections.defaultdict(list)

        unique_id_to_pos = {}
        for index, unique_id in enumerate(unique_ids):
            unique_id_to_pos[unique_id] = index

        for feature in features:
            example_index_to_features[feature.example_index].append(feature)

        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()
        for (example_index, example) in enumerate(examples):

            # finish this loop if we went through all batch examples
            if example_index >= len(unique_ids):
                break

            curr_features = example_index_to_features[example_index]

            doc_tokens, _, _, _, _ = BERTQADataset.get_doc_tokens_and_offset_from_context_id(
                example.context_id,
                example.start_position_character,
                example.is_impossible,
                example.answer_text,
                processor.doc_id_to_context_text,
            )
            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            # large and positive
            score_null = 1000000
            # the paragraph slice with min null score
            min_null_feature_index = 0
            # start logit at the slice with min null score
            null_start_logit = 0
            # end logit at the slice with min null score
            null_end_logit = 0
            for (feature_index, feature) in enumerate(curr_features):
                pos = unique_id_to_pos[feature.unique_id]
                start_indexes = self._get_best_indexes(start_logits[pos], n_best_size)
                end_indexes = self._get_best_indexes(end_logits[pos], n_best_size)
                # if we could have irrelevant answers,
                # get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = start_logits[pos][0] + end_logits[pos][0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = start_logits[pos][0]
                        null_end_logit = end_logits[pos][0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions,
                        # e.g., predict that the start of the span is in the
                        # question. We throw out all invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=start_logits[pos][start_index],
                                end_logit=end_logits[pos][end_index],
                            )
                        )

            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

            _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = curr_features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = self._get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

                # In very rare edge cases we could only
                # have single null pred. We just create a nonce prediction
                # in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["question"] = example.question_text
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = (
                    entry.start_logit
                    if (isinstance(entry.start_logit, float) or isinstance(entry.start_logit, int))
                    else list(entry.start_logit)
                )
                output["end_logit"] = (
                    entry.end_logit
                    if (isinstance(entry.end_logit, float) or isinstance(entry.end_logit, int))
                    else list(entry.end_logit)
                )
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score -
                # the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

        return all_predictions, all_nbest_json, scores_diff_json

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        processor = QAProcessor(cfg.file, mode)

        dataset = BERTQADataset(
            data_file=cfg.file,
            processor=processor,
            tokenizer=self.tokenizer,
            keep_doc_spans=self._cfg.dataset.keep_doc_spans,
            doc_stride=self._cfg.dataset.doc_stride,
            max_query_length=self._cfg.dataset.max_query_length,
            max_seq_length=self._cfg.dataset.max_seq_length,
            version_2_with_negative=self._cfg.dataset.version_2_with_negative,
            num_samples=cfg.num_samples,
            mode=mode,
            use_cache=self._cfg.dataset.use_cache,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        return data_loader

    def _get_best_indexes(self, logits, n_best_size):
        """ Get the n-best logits from a list """

        best_indices = np.argsort(logits)[::-1]

        return best_indices[:n_best_size]

    def _get_final_text(self, pred_text: str, orig_text: str, do_lower_case: bool, verbose_logging: bool = False):
        """
        Project the tokenized prediction back to the original text.
        When we created the data, we kept track of the alignment between original
        (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        now `orig_text` contains the span of our original text corresponding to
        the span that we predicted.

        However, `orig_text` may contain extra characters that we don't want in
        our prediction.

        For example, let's say:
        pred_text = steve smith
        orig_text = Steve Smith's

        We don't want to return `orig_text` because it contains the extra "'s".

        We don't want to return `pred_text` because it's already been normalized
        (the SQuAD eval script also does punctuation stripping/lower casing but
        our tokenizer does additional normalization like stripping accent
        characters).

        What we really want to return is "Steve Smith".

        Therefore, we have to apply a semi-complicated alignment heuristic
        between `pred_text` and `orig_text` to get a character-to-character
        alignment. This can fail in certain cases in which case we just return
        `orig_text`
        """

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return ns_text, ns_to_s_map

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logging.warning("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logging.warning(
                    "Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text,
                )
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logging.warning("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logging.warning("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position : (orig_end_position + 1)]

        return output_text

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
