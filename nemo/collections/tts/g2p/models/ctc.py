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

import os
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo.collections.tts.g2p.data.ctc import CTCG2PBPEDataset
from nemo.collections.tts.models.base import G2PModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    from nemo.collections.asr.losses.ctc import CTCLoss
    from nemo.collections.asr.metrics.wer_bpe import WERBPE, CTCBPEDecoding, CTCBPEDecodingConfig
    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.collections.asr.parts.mixins import ASRBPEMixin

    ASR_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    ASR_AVAILABLE = False


__all__ = ['CTCG2PModel']


@dataclass
class CTCG2PConfig:
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class CTCG2PModel(G2PModel, ASRBPEMixin):
    """
    CTC-based grapheme-to-phoneme model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        self.mode = cfg.model_name.lower()

        self.supported_modes = ["byt5", "conformer_bpe"]
        if self.mode not in self.supported_modes:
            raise ValueError(f"{self.mode} is not supported, choose from {self.supported_modes}")

        # Setup phoneme tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Setup grapheme tokenizer
        self.tokenizer_grapheme = self.setup_grapheme_tokenizer(cfg)

        # Initialize vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()
        cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))
        self.vocabulary = cfg.decoder.vocabulary
        self.labels_tkn2id = {l: i for i, l in enumerate(self.vocabulary)}
        self.labels_id2tkn = {i: l for i, l in enumerate(self.vocabulary)}

        super().__init__(cfg, trainer)

        self._setup_encoder()
        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

        self._wer = WERBPE(decoding=self.decoding, use_cer=False, log_prediction=False, dist_sync_on_step=True,)
        self._per = WERBPE(decoding=self.decoding, use_cer=True, log_prediction=False, dist_sync_on_step=True,)

    def setup_grapheme_tokenizer(self, cfg):
        """ Initialized grapheme tokenizer """

        if self.mode == "byt5":
            # Load appropriate tokenizer from HuggingFace
            grapheme_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_grapheme.pretrained)
            self.max_source_len = cfg.get("max_source_len", grapheme_tokenizer.model_max_length)
            self.max_target_len = cfg.get("max_target_len", grapheme_tokenizer.model_max_length)

            # TODO store byt5 vocab file
        elif self.mode == "conformer_bpe":
            grapheme_unk_token = (
                cfg.tokenizer_grapheme.unk_token if cfg.tokenizer_grapheme.unk_token is not None else ""
            )
            chars = string.ascii_lowercase + grapheme_unk_token + " " + "'"

            if not cfg.tokenizer_grapheme.do_lower:
                chars += string.ascii_uppercase

            if cfg.tokenizer_grapheme.add_punctuation:
                punctuation_marks = string.punctuation.replace('"', "").replace("\\", "").replace("'", "")
                chars += punctuation_marks

            vocab_file = "/tmp/char_vocab.txt"
            with open(vocab_file, "w") as f:
                [f.write(f'"{ch}"\n') for ch in chars]
                f.write('"\\""\n')  # add " to the vocab

            self.register_artifact("tokenizer_grapheme.vocab_file", vocab_file)
            grapheme_tokenizer = instantiate(cfg.tokenizer_grapheme.dataset, vocab_file=vocab_file)
            self.max_source_len = cfg.get("max_source_len", 512)
            self.max_target_len = cfg.get("max_target_len", 512)
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")
        return grapheme_tokenizer

    def _setup_encoder(self):
        if self.mode == "byt5":
            config = AutoConfig.from_pretrained(self._cfg.tokenizer_grapheme.pretrained)
            if self._cfg.encoder.dropout is not None:
                config.dropout_rate = self._cfg.encoder.dropout
                print(f"\nDROPOUT: {config.dropout_rate}")
            self.encoder = AutoModel.from_pretrained(self._cfg.encoder.transformer, config=config).encoder
            # add encoder hidden dim size to the config
            if self.cfg.decoder.feat_in is None:
                self._cfg.decoder.feat_in = self.encoder.config.d_model
        elif self.mode == "conformer_bpe":
            self.embedding = torch.nn.Embedding(
                embedding_dim=self._cfg.embedding.d_model, num_embeddings=self.tokenizer.vocab_size, padding_idx=0
            )
            self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)
            with open_dict(self._cfg):
                if "feat_in" not in self._cfg.decoder or (
                    not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
                ):
                    self._cfg.decoder.feat_in = self.encoder._feat_out
                if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                    raise ValueError("param feat_in of the decoder's config is not set!")
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")

    # @typecheck()
    def forward(self, input_ids, attention_mask, input_len):
        if self.mode == "byt5":
            encoded_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoded_len = input_len
            # encoded_input = [B, seq_len, hid_dim]
            # swap seq_len and hid_dim dimensions to get [B, hid_dim, seq_len]
            encoded_input = encoded_input.transpose(1, 2)
        elif self.mode == "conformer_bpe":
            input_embedding = self.embedding(input_ids)
            input_embedding = input_embedding.transpose(1, 2)
            encoded_input, encoded_len = self.encoder(audio_signal=input_embedding, length=input_len)
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")

        log_probs = self.decoder(encoder_output=encoded_input)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, greedy_predictions, encoded_len

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )

        loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx, dataloader_idx=0, split="val"):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, greedy_predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )
        val_loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )

        self._wer.update(
            predictions=log_probs, targets=targets, target_lengths=target_lengths, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()

        self._per.update(
            predictions=log_probs, targets=targets, target_lengths=target_lengths, predictions_lengths=encoded_len
        )
        per, per_num, per_denom = self._per.compute()
        self._per.reset()

        self.log(f"{split}_loss", val_loss)
        loss = {
            f"{split}_loss": val_loss,
            f"{split}_wer_num": wer_num,
            f"{split}_wer_denom": wer_denom,
            f"{split}_wer": wer,
            f"{split}_per_num": per_num,
            f"{split}_per_denom": per_denom,
            f"{split}_per": per,
        }

        if split == 'val':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(loss)
            else:
                self.validation_step_outputs.append(loss)
        elif split == 'test':
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(loss)
            else:
                self.test_step_outputs.append(loss)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx, split="test")

    def multi_validation_epoch_end(self, outputs, dataloader_idx=0, split="val"):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        avg_loss = torch.stack([x[f"{split}_loss"] for x in outputs]).mean()
        self.log(f"{split}_loss", avg_loss, prog_bar=True)

        wer_num = torch.stack([x[f"{split}_wer_num"] for x in outputs]).sum()
        wer_denom = torch.stack([x[f"{split}_wer_denom"] for x in outputs]).sum()
        wer = wer_num / wer_denom

        per_num = torch.stack([x[f"{split}_per_num"] for x in outputs]).sum()
        per_denom = torch.stack([x[f"{split}_per_denom"] for x in outputs]).sum()
        per = per_num / per_denom

        if split == "test":
            dataloader_name = self._test_names[dataloader_idx].upper()
        else:
            dataloader_name = self._validation_names[dataloader_idx].upper()

        self.log(f"{split}_wer", wer)
        self.log(f"{split}_per", per)

        self.log(f"{split}_per", per)
        # to save all PER values for each dataset in WANDB
        self.log(f"{split}_per_{dataloader_name}", per)

        logging.info(f"PER: {per * 100}% {dataloader_name}")
        logging.info(f"WER: {wer * 100}% {dataloader_name}")

    def multi_test_epoch_end(self, outputs, dataloader_idx=0):
        self.multi_validation_epoch_end(outputs, dataloader_idx, split="test")

    def _setup_infer_dataloader(self, cfg: DictConfig) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Returns:
            A pytorch DataLoader.
        """
        dataset = CTCG2PBPEDataset(
            manifest_filepath=cfg.manifest_filepath,
            grapheme_field=cfg.grapheme_field,
            tokenizer_graphemes=self.tokenizer_grapheme,
            tokenizer_phonemes=self.tokenizer,
            do_lower=self._cfg.tokenizer_grapheme.do_lower,
            labels=self.vocabulary,
            max_source_len=self._cfg.max_source_len,
            with_labels=False,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    @torch.no_grad()
    def _infer(self, config: DictConfig,) -> List[int]:
        """
        Runs model inference.

        Args:
            Config: configuration file to set up DataLoader
        Returns:
            all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            infer_datalayer = self._setup_infer_dataloader(config)

            for batch in infer_datalayer:
                input_ids, attention_mask, input_len = batch
                log_probs, greedy_predictions, encoded_len = self.forward(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask if attention_mask is None else attention_mask.to(device),
                    input_len=input_len.to(device),
                )

                preds_str, _ = self.decoding.ctc_decoder_predictions_tensor(
                    log_probs, decoder_lengths=encoded_len, return_hypotheses=False
                )
                all_preds.extend(preds_str)

                del greedy_predictions
                del log_probs
                del batch
                del input_len
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg: DictConfig, name: str):
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        if not os.path.exists(cfg.manifest_filepath):
            raise ValueError(f"{cfg.dataset.manifest_filepath} not found")

        dataset = instantiate(
            cfg.dataset,
            manifest_filepath=cfg.manifest_filepath,
            phoneme_field=cfg.dataset.phoneme_field,
            grapheme_field=cfg.dataset.grapheme_field,
            tokenizer_graphemes=self.tokenizer_grapheme,
            do_lower=self._cfg.tokenizer_grapheme.do_lower,
            tokenizer_phonemes=self.tokenizer,
            labels=self.vocabulary,
            max_source_len=self.max_source_len,
            with_labels=True,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg: DictConfig):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict] = None):
        if not val_data_config or val_data_config.manifest_filepath is None:
            self._validation_dl = None
            return
        super().setup_multiple_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict] = None):
        if not test_data_config or test_data_config.manifest_filepath is None:
            self._test_dl = None
            return
        super().setup_multiple_test_data(test_data_config)

    def setup_validation_data(self, cfg: Optional[DictConfig]):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="val")

    def setup_test_data(self, cfg: Optional[DictConfig]):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
