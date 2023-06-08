# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from time import perf_counter
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.spellchecking_asr_customization import (
    SpellcheckingAsrCustomizationDataset,
    SpellcheckingAsrCustomizationTestDataset,
    TarredSpellcheckingAsrCustomizationDataset,
    bert_example,
)
from nemo.collections.nlp.data.text_normalization_as_tagging.utils import read_label_map
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ["SpellcheckingAsrCustomizationModel"]


@experimental
class SpellcheckingAsrCustomizationModel(NLPModel):
    """
    https://arxiv.org/abs/2306.02317
    BERT-based model for Spellchecking ASR Customization.
    It takes as input ASR hypothesis and candidate customization entries.
    It labels the hypothesis with correct entry index or 0.
    Example input:   [CLS] a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o [SEP] d i d i e r _ s a u m o n [SEP] a s t r o n o m i e [SEP] t r i s t a n _ g u i l l o t [SEP] ...
    Input segments:      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0     1 1 1 1 1 1 1 1 1 1 1 1 1 1     2 2 2 2 2 2 2 2 2 2 2     3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3     4      
    Example output:      0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 0     ...
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
        }

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)

        # Label map contains 11 labels: 0 for nothing, 1..10 for target candidate ids
        label_map_file = self.register_artifact("label_map", cfg.label_map, verify_src_exists=True)

        # Semiotic classes for this model consist only of classes CUSTOM(means fragment containing custom candidate) and PLAIN (any other single-character fragment)
        # They are used only during validation step, to calculate accuracy for CUSTOM and PLAIN classes separately
        semiotic_classes_file = self.register_artifact(
            "semiotic_classes", cfg.semiotic_classes, verify_src_exists=True
        )
        self.label_map = read_label_map(label_map_file)
        self.semiotic_classes = read_label_map(semiotic_classes_file)

        self.num_labels = len(self.label_map)
        self.num_semiotic_labels = len(self.semiotic_classes)
        self.id_2_tag = {tag_id: tag for tag, tag_id in self.label_map.items()}
        self.id_2_semiotic = {semiotic_id: semiotic for semiotic, semiotic_id in self.semiotic_classes.items()}
        self.max_sequence_len = cfg.get('max_sequence_len', self.tokenizer.tokenizer.model_max_length)

        # Setup to track metrics
        # We will have (len(self.semiotic_classes) + 1) labels.
        # Last one stands for WRONG (span in which the predicted tags don't match the labels)
        # This is needed to feed the sequence of classes to classification_report during validation
        label_ids = self.semiotic_classes.copy()
        label_ids["WRONG"] = len(self.semiotic_classes)
        self.tag_classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )

        self.hidden_size = cfg.hidden_size

        # hidden size is doubled because in forward we concatenate embeddings for characters and embeddings for subwords
        self.logits = TokenClassifier(
            self.hidden_size * 2, num_classes=self.num_labels, num_layers=1, log_softmax=False, dropout=0.1
        )

        self.loss_fn = CrossEntropyLoss(logits_ndim=3)

        self.builder = bert_example.BertExampleBuilder(
            self.label_map, self.semiotic_classes, self.tokenizer.tokenizer, self.max_sequence_len
        )

    @typecheck()
    def forward(
        self,
        input_ids,
        input_mask,
        segment_ids,
        input_ids_for_subwords,
        input_mask_for_subwords,
        segment_ids_for_subwords,
        character_pos_to_subword_pos,
    ):
        """
        Same BERT-based model is used to calculate embeddings for sequence of single characters and for sequence of subwords.
        Then we concatenate subword embeddings to each character corresponding to this subword.
        We return logits for each character x 11 labels: 0 - character doesn't belong to any candidate, 1..10 - character belongs to candidate with this id.

        # Arguments
            input_ids: token_ids for single characters; .shape = [batch_size, char_seq_len]; .dtype = int64
            input_mask: mask for input_ids(1 - real, 0 - padding); .shape = [batch_size, char_seq_len]; .dtype = int64
            segment_ids: segment types for input_ids (0 - ASR-hypothesis, 1..10 - candidate); .shape = [batch_size, char_seq_len]; .dtype = int64
            input_ids_for_subwords: token_ids for subwords; .shape = [batch_size, subword_seq_len]; .dtype = int64
            input_mask_for_subwords: mask for input_ids_for_subwords(1 - real, 0 - padding); .shape = [batch_size, subword_seq_len]; .dtype = int64
            segment_ids_for_subwords: segment types for input_ids_for_subwords (0 - ASR-hypothesis, 1..10 - candidate); .shape = [batch_size, subword_seq_len]; .dtype = int64
            character_pos_to_subword_pos: tensor mapping character position in the input sequence to subword position; .shape = [batch_size, char_seq_len]; .dtype = int64
        """

        # src_hiddens.shape = [batch_size, char_seq_len, bert_hidden_size]; .dtype=float32
        src_hiddens = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        # src_hiddens_for_subwords.shape = [batch_size, subword_seq_len, bert_hidden_size]; .dtype=float32
        src_hiddens_for_subwords = self.bert_model(
            input_ids=input_ids_for_subwords,
            token_type_ids=segment_ids_for_subwords,
            attention_mask=input_mask_for_subwords,
        )

        # Next three commands concatenate subword embeddings to each character embedding of the corresponding subword
        # index.shape = [batch_size, char_seq_len, bert_hidden_size]; .dtype=int64
        index = character_pos_to_subword_pos.unsqueeze(-1).expand((-1, -1, src_hiddens_for_subwords.shape[2]))
        # src_hiddens_2.shape = [batch_size, char_seq_len, bert_hidden_size]; .dtype=float32
        src_hiddens_2 = torch.gather(src_hiddens_for_subwords, 1, index)
        # src_hiddens.shape = [batch_size, char_seq_len, bert_hidden_size * 2]; .dtype=float32
        src_hiddens = torch.cat((src_hiddens, src_hiddens_2), 2)

        # logits.shape = [batch_size, char_seq_len, num_labels]; num_labels=11: ids from 0 to 10; .dtype=float32
        logits = self.logits(hidden_states=src_hiddens)
        return logits

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """

        (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            _,
        ) = batch
        logits = self.forward(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_ids_for_subwords=input_ids_for_subwords,
            input_mask_for_subwords=input_mask_for_subwords,
            segment_ids_for_subwords=segment_ids_for_subwords,
            character_pos_to_subword_pos=character_pos_to_subword_pos,
        )
        loss = self.loss_fn(logits=logits, labels=labels, loss_mask=labels_mask)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': loss, 'lr': lr}

    # Validation and Testing
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            spans,
        ) = batch
        logits = self.forward(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_ids_for_subwords=input_ids_for_subwords,
            input_mask_for_subwords=input_mask_for_subwords,
            segment_ids_for_subwords=segment_ids_for_subwords,
            character_pos_to_subword_pos=character_pos_to_subword_pos,
        )
        tag_preds = torch.argmax(logits, dim=2)

        # Update tag classification_report
        for input_mask_seq, segment_seq, prediction_seq, label_seq, span_seq in zip(
            input_mask.tolist(), segment_ids.tolist(), tag_preds.tolist(), labels.tolist(), spans.tolist()
        ):
            # Here we want to track whether the predicted output matches ground truth labels for each whole span.
            # We construct the special input for classification report, for example:
            #   span_labels = [PLAIN, PLAIN, PLAIN, PLAIN, CUSTOM, CUSTOM]
            #   span_predictions = [PLAIN, WRONG, PLAIN, PLAIN, WRONG, CUSTOM]
            # Note that the number of PLAIN and CUSTOM occurrences in the report is not comparable,
            #   because PLAIN is for characters, and CUSTOM is for phrases.
            span_labels = []
            span_predictions = []
            plain_cid = self.semiotic_classes["PLAIN"]
            wrong_cid = self.tag_classification_report.num_classes - 1

            # First we loop through all predictions for input characters with label=0, they are regarded as separate spans with PLAIN class.
            # It either stays as PLAIN if the model prediction is 0, or turns to WRONG.
            for i in range(len(segment_seq)):
                if input_mask_seq[i] == 0:
                    continue
                if segment_seq[i] > 0:  # token does not belong to ASR-hypothesis => it's over
                    break
                if label_seq[i] == 0:
                    span_labels.append(plain_cid)
                    if prediction_seq[i] == 0:
                        span_predictions.append(plain_cid)
                    else:
                        span_predictions.append(wrong_cid)
                # if label_seq[i] != 0 then it belongs to CUSTOM span and will be handled later

            # Second we loop through spans tensor which contains only spans for CUSTOM class.
            # It stays as CUSTOM if all predictions for the whole span are equal to the labels, otherwise it turns to WRONG.
            for cid, start, end in span_seq:
                if cid == -1:
                    break
                span_labels.append(cid)
                if prediction_seq[start:end] == label_seq[start:end]:
                    span_predictions.append(cid)
                else:
                    span_predictions.append(wrong_cid)

            if len(span_labels) != len(span_predictions):
                raise ValueError(
                    "Length mismatch: len(span_labels)="
                    + str(len(span_labels))
                    + "; len(span_predictions)="
                    + str(len(span_predictions))
                )
            self.tag_classification_report(
                torch.tensor(span_predictions).to(self.device), torch.tensor(span_labels).to(self.device)
            )

        val_loss = self.loss_fn(logits=logits, labels=labels, loss_mask=labels_mask)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Calculate metrics and classification report
        # Note that in our task recall = accuracy, and the recall column is the per class accuracy
        _, tag_accuracy, _, tag_report = self.tag_classification_report.compute()

        logging.info("Total tag accuracy: " + str(tag_accuracy))
        logging.info(tag_report)

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('tag accuracy', tag_accuracy)

        self.tag_classification_report.reset()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs)

    # Functions for inference

    @torch.no_grad()
    def infer(self, dataloader_cfg: DictConfig, input_name: str, output_name: str) -> None:
        """ Main function for Inference

        Args:
            dataloader_cfg: config for dataloader
            input_name: Input file with tab-separated text records. Each record consists of 2 items:
                - ASR hypothesis
                - candidate phrases separated by semicolon
            output_name: Output file with tab-separated text records. Each record consists of 2 items:
                - ASR hypothesis
                - candidate phrases separated by semicolon
                - list of possible replacements with probabilities (start, pos, candidate_id, prob), separated by semicolon
                - list of labels, predicted for each letter (for debug purposes)

        Returns: None
        """
        mode = self.training
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Switch model to evaluation mode
            self.eval()
            self.to(device)
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, input_name)

            all_tag_preds = (
                []
            )  # list(size=number of sentences) of lists(size=number of letters) of tag predictions (best candidate_id for each letter)
            all_possible_replacements = (
                []
            )  # list(size=number of sentences) of lists(size=number of potential replacements) of tuples(start, pos, candidate_id, prob)
            for batch in iter(infer_datalayer):
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    input_ids_for_subwords,
                    input_mask_for_subwords,
                    segment_ids_for_subwords,
                    character_pos_to_subword_pos,
                    fragment_indices,
                ) = batch

                # tag_logits.shape = [batch_size, char_seq_len, num_labels]; num_labels=11: ids from 0 to 10; .dtype=float32
                tag_logits = self.forward(
                    input_ids=input_ids.to(self.device),
                    input_mask=input_mask.to(self.device),
                    segment_ids=segment_ids.to(self.device),
                    input_ids_for_subwords=input_ids_for_subwords.to(self.device),
                    input_mask_for_subwords=input_mask_for_subwords.to(self.device),
                    segment_ids_for_subwords=segment_ids_for_subwords.to(self.device),
                    character_pos_to_subword_pos=character_pos_to_subword_pos.to(self.device),
                )

                # fragment_indices.shape=[batsh_size, num_fragments, 3], where last dimension is [start, end, label], where label is candidate id from 1 to 10
                # Next we want to convert predictions for separate letters to probabilities for each whole fragment from fragment_indices.
                # To achieve this we first sum the letter logits in each fragment and divide by its length.
                # (We use .cumsum and then difference between end and start to get sum per fragment).
                # Then we convert logits to probs with softmax and for each fragment extract only the prob for given label.
                # Finally we get a list of tuples (start, end, label, prob)
                indices_len = fragment_indices.shape[1]
                # this padding adds a row of zeros (size=num_labels) as first element of sequence in second dimension. This is needed for cumsum operations.
                padded_logits = torch.nn.functional.pad(tag_logits, pad=(0, 0, 1, 0))
                (
                    batch_size,
                    seq_len,
                    num_labels,
                ) = padded_logits.shape  # seq_len is +1 compared to that of tag_logits, because of padding
                # cumsum.shape=[batch_size, seq_len, num_labels]
                cumsum = padded_logits.cumsum(dim=1)
                # the size -1 is inferred from other dimensions. We get rid of batch dimension.
                cumsum_view = cumsum.view(-1, num_labels)
                word_index = (
                    torch.ones((batch_size, indices_len), dtype=torch.long)
                    * torch.arange(batch_size).reshape((-1, 1))
                    * seq_len
                ).view(-1)
                lower_index = (fragment_indices[..., 0]).view(-1) + word_index
                higher_index = (fragment_indices[..., 1]).view(-1) + word_index
                d_index = (higher_index - lower_index).reshape((-1, 1)).to(self.device)  # word lengths
                dlog = cumsum_view[higher_index, :] - cumsum_view[lower_index, :]  # sum of logits
                # word_logits.shape=[batch_size, indices_len, num_labels]
                word_logits = (dlog / d_index.float()).view(batch_size, indices_len, num_labels)
                # convert logits to probs, same shape
                word_probs = torch.nn.functional.softmax(word_logits, dim=-1).to(self.device)
                # candidate_index.shape=[batch_size, indices_len]
                candidate_index = fragment_indices[:, :, 2].to(self.device)
                # candidate_probs.shape=[batch_size, indices_len]
                candidate_probs = torch.take_along_dim(word_probs, candidate_index.unsqueeze(2), dim=-1).squeeze(2)
                for i in range(batch_size):
                    possible_replacements = []
                    for j in range(indices_len):
                        start, end, candidate_id = (
                            int(fragment_indices[i][j][0]),
                            int(fragment_indices[i][j][1]),
                            int(fragment_indices[i][j][2]),
                        )
                        if candidate_id == 0:  # this is padding
                            continue
                        prob = round(float(candidate_probs[i][j]), 5)
                        if prob < 0.01:
                            continue
                        # -1 because in the output file we will not have a [CLS] token
                        possible_replacements.append(
                            str(start - 1) + " " + str(end - 1) + " " + str(candidate_id) + " " + str(prob)
                        )
                    all_possible_replacements.append(possible_replacements)

                # torch.argmax(tag_logits, dim=-1) gives a tensor of best predicted labels with shape [batch_size, char_seq_len], .dtype = int64
                # character_preds is list of lists of predicted labels
                character_preds = tensor2list(torch.argmax(tag_logits, dim=-1))
                all_tag_preds.extend(character_preds)

            if len(all_possible_replacements) != len(all_tag_preds) or len(all_possible_replacements) != len(
                infer_datalayer.dataset.examples
            ):
                raise IndexError(
                    "number of sentences mismatch: len(all_possible_replacements)="
                    + str(len(all_possible_replacements))
                    + "; len(all_tag_preds)="
                    + str(len(all_tag_preds))
                    + "; len(infer_datalayer.dataset.examples)="
                    + str(len(infer_datalayer.dataset.examples))
                )
            # save results to file
            with open(output_name, "w", encoding="utf-8") as out:
                for i in range(len(infer_datalayer.dataset.examples)):
                    hyp, ref = infer_datalayer.dataset.hyps_refs[i]
                    num_letters = hyp.count(" ") + 1
                    tag_pred_str = " ".join(list(map(str, all_tag_preds[i][1 : (num_letters + 1)])))
                    possible_replacements_str = ";".join(all_possible_replacements[i])
                    out.write(hyp + "\t" + ref + "\t" + possible_replacements_str + "\t" + tag_pred_str + "\n")

        except Exception as e:
            raise ValueError("Error processing file " + input_name)

        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, data_split="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, data_split="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.data_path is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, data_split="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, data_split: str):
        start_time = perf_counter()
        logging.info(f'Creating {data_split} dataset')
        if cfg.get("use_tarred_dataset", False):
            dataset = TarredSpellcheckingAsrCustomizationDataset(
                cfg.data_path,
                shuffle_n=cfg.get("tar_shuffle_n", 100),
                global_rank=self.global_rank,
                world_size=self.world_size,
                pad_token_id=self.builder._pad_id,
            )
        else:
            input_file = cfg.data_path
            dataset = SpellcheckingAsrCustomizationDataset(input_file=input_file, example_builder=self.builder)
        dl = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_fn
        )
        running_time = perf_counter() - start_time
        logging.info(f'Took {running_time} seconds')
        return dl

    def _setup_infer_dataloader(self, cfg: DictConfig, input_name: str) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            input_name: path to input file. 
        Returns:
            A pytorch DataLoader.
        """
        dataset = SpellcheckingAsrCustomizationTestDataset(input_name, example_builder=self.builder)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None
