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
import os
from typing import List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.collections.tts.g2p.data.heteronym_classification import HeteronymClassificationDataset
from nemo.collections.tts.g2p.utils import get_heteronym_spans, get_wordid_to_phonemes, read_wordids
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    from nemo.collections.nlp.models.nlp_model import NLPModel

    NLP_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    NLP_AVAILABLE = False

__all__ = ['HeteronymClassificationModel']


class HeteronymClassificationModel(NLPModel):
    """
    This is a classification model that selects the best heteronym option out of possible dictionary entries.
    Supports only heteronyms, no OOV.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.max_seq_length = cfg.max_seq_length
        self.wordids = self.register_artifact("wordids", cfg.wordids)
        self.heteronym_dict, self.wordid_to_idx = read_wordids(self.wordids)
        self.idx_to_wordid = {v: k for k, v in self.wordid_to_idx.items()}
        self.supported_heteronyms = list(self.heteronym_dict.keys())

        if cfg.class_labels.class_labels_file is None:
            label_ids_file = "/tmp/label_ids.csv"
            with open(label_ids_file, 'w') as f:
                for idx in range(len(self.idx_to_wordid)):
                    f.write(self.idx_to_wordid[idx] + "\n")
            self.register_artifact("class_labels.class_labels_file", label_ids_file)

        super().__init__(cfg=cfg, trainer=trainer)
        self.lang = self._cfg.get('lang', None)
        num_classes = len(self.wordid_to_idx)
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            num_layers=self._cfg.head.num_fc_layers,
            activation=self._cfg.head.activation,
            log_softmax=False,
            dropout=self._cfg.head.fc_dropout,
            use_transformer_init=self._cfg.head.use_transformer_init,
        )

        # Loss Functions
        self.loss = CrossEntropyLoss(logits_ndim=3)

        # setup to track metrics
        self.classification_report = ClassificationReport(
            num_classes=num_classes, mode='macro', dist_sync_on_step=True, label_ids=self.wordid_to_idx
        )

        # used for inference to convert predicted wordids to phonemes
        self.wordid_to_phonemes_file = None
        self.wordid_to_phonemes = None

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        logits = self.classifier(hidden_states=hidden_states)
        return logits

    def make_step(self, batch):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=torch.zeros_like(batch["input_ids"]),
        )

        if "targets" in batch:
            loss = self.loss(logits=logits, labels=batch["targets"])
        else:
            # skip loss calculation for inference
            loss = None
        return loss, logits

        # Training

    def training_step(self, batch, batch_idx):
        """
		Lightning calls this inside the training loop with the data from the training dataloader
		passed in as `batch`.
		"""

        loss, logits = self.make_step(batch)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    # Validation and Testing
    def validation_step(self, batch, batch_idx, split="val"):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        val_loss, logits = self.make_step(batch)
        subtokens_mask = batch["subtokens_mask"]
        targets = batch["targets"]
        targets = targets[targets != -100]

        self.log(f"{split}_loss", val_loss)
        tag_preds = torch.argmax(logits, axis=-1)[subtokens_mask > 0]
        tp, fn, fp, _ = self.classification_report(tag_preds, targets)
        loss = {f'{split}_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

        if split == 'val':
            self.validation_step_outputs.append(loss)
        elif split == 'test':
            self.test_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        split = "test" if self.trainer.testing else "val"
        if split == 'val':
            avg_loss = torch.stack([x[f'{split}_loss'] for x in self.validation_step_outputs]).mean()
        elif split == 'test':
            avg_loss = torch.stack([x[f'{split}_loss'] for x in self.test_step_outputs]).mean()

        # calculate metrics and classification report
        precision, recall, f1, report = self.classification_report.compute()

        # remove examples with support=0
        report = "\n".join(
            [
                x
                for x in report.split("\n")
                if not x.endswith("          0") and "100.00     100.00     100.00" not in x
            ]
        )
        logging.info(f"{split}_report: {report}")
        logging.info(f"{split}_f1: {f1:.2f}%")
        self.log(f"{split}_loss", avg_loss, prog_bar=True)
        self.log(f"{split}_precision", precision)
        self.log(f"{split}_f1", f1)
        self.log(f"{split}_recall", recall)

        f1_macro = report[report.index("macro") :].split("\n")[0].replace("macro avg", "").strip().split()[-2]
        f1_micro = report[report.index("micro") :].split("\n")[0].replace("micro avg", "").strip().split()[-2]
        self.log(f"{split}_f1_macro", torch.Tensor([float(f1_macro)]))
        self.log(f"{split}_f1_micro", torch.Tensor([float(f1_micro)]))

        self.classification_report.reset()

        if split == 'val':
            self.validation_step_outputs.clear()  # free memory
        elif split == 'test':
            self.test_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        """
        Called at the end of test to aggregate outputs.

        Args:
            outputs: list of individual outputs of each test step.
        """
        return self.on_validation_epoch_end()

    def set_wordid_to_phonemes(self, wordid_to_phonemes_file: str):
        if wordid_to_phonemes_file is None or not os.path.exists(wordid_to_phonemes_file):
            logging.warning(f"{wordid_to_phonemes_file} not found, skip setting wordid_to_phonemes.")
        else:
            self.wordid_to_phonemes_file = wordid_to_phonemes_file
            self.wordid_to_phonemes = get_wordid_to_phonemes(self.wordid_to_phonemes_file)
            logging.info(f"Wordid to phonemes file is set to {wordid_to_phonemes_file}")

    # Functions for inference
    def _process_sentence(self, text: str, start_end: List[List[int]], predictions: List[str]):
        text_with_heteronym_replaced = ""
        last_idx = 0
        for heteronym_idx, cur_start_end in enumerate(start_end):
            cur_start, cur_end = cur_start_end
            cur_pred = predictions[heteronym_idx]

            if self.wordid_to_phonemes is None or cur_pred not in self.wordid_to_phonemes:
                cur_pred = f"[{cur_pred}]"
            else:
                cur_pred = self.wordid_to_phonemes[cur_pred]
                # to use mixed grapheme format as an input for a TTS model, we need to have vertical bars around phonemes
                cur_pred = "".join([f"|{p}|" for p in cur_pred])

            text_with_heteronym_replaced += text[last_idx:cur_start] + cur_pred
            last_idx = cur_end
        if last_idx < len(text):
            text_with_heteronym_replaced += text[last_idx:]
        return text_with_heteronym_replaced

    @torch.no_grad()
    def disambiguate(
        self,
        sentences: List[str],
        batch_size: int = 4,
        num_workers: int = 0,
        wordid_to_phonemes_file: Optional[str] = None,
    ):
        """
        Replaces heteronyms, supported by the model, with the phoneme form (if wordid_to_phonemes_file)
        or with predicted wordids.

        Args:
            sentences: Sentences to use for inference
            batch_size: batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            num_workers: number of workers for DataLoader
            wordid_to_phonemes_file: (Optional) file with mapping between wordid predicted by the model to phonemes

        Returns:
            preds: model predictions
            output: sentences with heteronym replaced with phonemes (if wordid_to_phonemes_file specified)
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        batch_size = min(batch_size, len(sentences))

        start_end, heteronyms = get_heteronym_spans(sentences, self.heteronym_dict)
        if len(sentences) != len(start_end) != len(heteronyms):
            raise ValueError(
                f"Number of sentences should match the lengths of provided start-end indices, {len(sentences)} != {len(start_end)}"
            )

        tmp_manifest = "/tmp/manifest.json"
        with open(tmp_manifest, "w") as f:
            for cur_sentence, cur_start_ends, cur_heteronyms in zip(sentences, start_end, heteronyms):
                item = {"text_graphemes": cur_sentence, "start_end": cur_start_ends, "heteronym_span": cur_heteronyms}
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        all_preds = self._disambiguate(manifest=tmp_manifest, batch_size=batch_size, num_workers=num_workers,)

        if wordid_to_phonemes_file is not None:
            self.set_wordid_to_phonemes(wordid_to_phonemes_file)

        output = []
        for sent_idx, sent_start_end in enumerate(start_end):
            output.append(
                self._process_sentence(
                    text=sentences[sent_idx], start_end=sent_start_end, predictions=all_preds[sent_idx]
                ),
            )

        return all_preds, output

    @torch.no_grad()
    def _disambiguate(self, manifest: str, batch_size: int, num_workers: int = 0, grapheme_field="text_graphemes"):
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Switch model to evaluation mode
            self.eval()
            self.to(device)
            infer_datalayer = self._setup_infer_dataloader(
                manifest, grapheme_field=grapheme_field, batch_size=batch_size, num_workers=num_workers
            )

            for batch in infer_datalayer:
                subtokens_mask = batch["subtokens_mask"]
                batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                _, logits = self.make_step(batch)

                preds = tensor2list(torch.argmax(logits, axis=-1)[subtokens_mask > 0])
                # preds are flatten for all the samples, we need to separate predictions per sample
                preds_num = [len([p_ for p_ in p if p_ == 1]) for p in tensor2list(subtokens_mask)]

                last_idx = 0
                for num in preds_num:
                    preds_ = preds[last_idx : last_idx + num]
                    preds_ = [self.idx_to_wordid[p] for p in preds_]
                    all_preds.append(preds_)
                    last_idx += num
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    @torch.no_grad()
    def disambiguate_manifest(
        self,
        manifest,
        output_manifest: str,
        grapheme_field: str = "text_graphemes",
        batch_size: int = 4,
        num_workers: int = 0,
        wordid_to_phonemes_file: Optional[str] = None,
    ):
        all_preds = self._disambiguate(
            manifest=manifest, batch_size=batch_size, num_workers=num_workers, grapheme_field=grapheme_field
        )

        self.set_wordid_to_phonemes(wordid_to_phonemes_file)

        with open(manifest, "r", encoding="utf-8") as f_in, open(output_manifest, "w", encoding="utf-8") as f_preds:
            for idx, line in enumerate(f_in):
                line = json.loads(line)
                start_end = line["start_end"]
                if len(start_end) > 0 and isinstance(start_end[0], int):
                    start_end = [start_end]

                text_with_heteronym_replaced = self._process_sentence(
                    text=line[grapheme_field], start_end=start_end, predictions=all_preds[idx]
                )

                line["pred_text"] = text_with_heteronym_replaced
                line["pred_wordid"] = all_preds[idx]
                f_preds.write(json.dumps(line, ensure_ascii=False) + '\n')

        logging.info(f"Predictions save at {output_manifest}")
        return all_preds

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or train_data_config.dataset.manifest is None:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, data_split="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or val_data_config.dataset.manifest is None:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, data_split="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.dataset.manifest is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, data_split="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, data_split: str):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {data_split}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {data_split}")

        dataset = instantiate(
            cfg.dataset,
            manifest=cfg.dataset.manifest,
            grapheme_field=cfg.dataset.grapheme_field,
            tokenizer=self.tokenizer,
            wordid_to_idx=self.wordid_to_idx,
            heteronym_dict=self.heteronym_dict,
            max_seq_len=self.max_seq_length,
            with_labels=True,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def _setup_infer_dataloader(
        self, manifest: str, grapheme_field: str, batch_size: int, num_workers: int
    ) -> 'torch.utils.data.DataLoader':

        dataset = HeteronymClassificationDataset(
            manifest=manifest,
            grapheme_field=grapheme_field,
            tokenizer=self.tokenizer,
            wordid_to_idx=self.wordid_to_idx,
            heteronym_dict=self.heteronym_dict,
            max_seq_len=self.tokenizer.tokenizer.model_max_length,
            with_labels=False,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        return []
