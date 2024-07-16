# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import json
import os
import tempfile
from math import ceil
from typing import Any, Dict, List, Optional, Union

import editdistance
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics.text import SacreBLEUScore
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.modules.transformer import (
    BeamSearchSequenceGenerator,
    TransformerEncoder,
    get_nemo_transformer,
)
from nemo.collections.asr.parts.mixins import ASRBPEMixin, ASRTranscriptionMixin, TranscribeConfig
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import (
    AudioSignal,
    ChannelType,
    LabelsType,
    LengthsType,
    LogprobsType,
    MaskType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['EncDecTransfModelBPE']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class EncDecTransfModelBPE(ASRModel, ExportableEncDecModel, ASRBPEMixin, ASRTranscriptionMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup audio preprocessor
        self.preprocessor = EncDecTransfModelBPE.from_config_dict(self.cfg.preprocessor)

        # Setup audio encoder
        self.encoder = EncDecTransfModelBPE.from_config_dict(self.cfg.encoder)

        # Add projection layer if encoder and decoder differ in hidden size
        if self.cfg.encoder['d_model'] != self.cfg.transf_decoder['hidden_size']:
            self.adapter = torch.nn.Linear(self.cfg.encoder['d_model'], self.cfg.transf_decoder['hidden_size'])
        else:
            self.adapter = torch.nn.Identity()

        transf_encoder_cfg_dict = OmegaConf.to_container(cfg.get('transf_encoder'))

        # Whether to add Transformer Encoder block between Conformer and Transformer Decoder
        self.use_transf_encoder = False
        if transf_encoder_cfg_dict['num_layers'] > 0:
            self.use_transf_encoder = True

            self.transf_encoder = TransformerEncoder(
                num_layers=transf_encoder_cfg_dict['num_layers'],
                hidden_size=transf_encoder_cfg_dict['hidden_size'],
                inner_size=transf_encoder_cfg_dict['inner_size'],
                mask_future=False,
                num_attention_heads=transf_encoder_cfg_dict['num_attention_heads'],
                attn_score_dropout=transf_encoder_cfg_dict['attn_score_dropout'],
                attn_layer_dropout=transf_encoder_cfg_dict['attn_layer_dropout'],
                ffn_dropout=transf_encoder_cfg_dict['ffn_dropout'],
                pre_ln=transf_encoder_cfg_dict.get('pre_ln', True),
                pre_ln_final_layer_norm=transf_encoder_cfg_dict.get('pre_ln_final_layer_norm', True),
            )
            std_init_range = 1 / transf_encoder_cfg_dict['hidden_size'] ** 0.5
            self.transf_encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        transf_decoder_cfg_dict = OmegaConf.to_container(cfg.get('transf_decoder'))

        # Transformer decoder
        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)
        transf_decoder_cfg_dict['vocab_size'] = vocab_size
        library = transf_decoder_cfg_dict.pop('library', 'nemo')
        if library != 'nemo':
            raise ValueError(f"Currently only 'nemo' library is supported for Transformer decoder. Got {library}")
        model_name = transf_decoder_cfg_dict.pop('model_name', None)
        pretrained = transf_decoder_cfg_dict.pop('pretrained', False)
        self.transf_decoder = get_nemo_transformer(
            model_name=model_name,
            pretrained=pretrained,
            config_dict=transf_decoder_cfg_dict,
            encoder=False,
            pre_ln_final_layer_norm=transf_decoder_cfg_dict.get("pre_ln_final_layer_norm", False),
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self.transf_decoder.hidden_size,
            num_classes=vocab_size,
            activation=self.cfg.head.activation,
            log_softmax=self.cfg.head.log_softmax,
            dropout=self.cfg.head.dropout,
            use_transformer_init=self.cfg.head.use_transformer_init,
            num_layers=self.cfg.head.num_layers,
        )
        self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight
        std_init_range = 1 / self.transf_decoder.hidden_size**0.5
        self.transf_decoder.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Beam Search decoding
        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.transf_decoder.embedding,
            decoder=self.transf_decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.transf_decoder.max_sequence_length,
            beam_size=self.cfg.beam_search.beam_size,
            bos=self.tokenizer.bos_id,
            pad=self.tokenizer.pad_id,
            eos=self.tokenizer.eos_id,
            len_pen=self.cfg.beam_search.len_pen,
            max_delta_length=self.cfg.beam_search.max_generation_delta,
        )

        # Define autoregressive CE loss
        self.transf_loss = SmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, label_smoothing=self.cfg.label_smoothing
        )

        if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
            self.spec_augmentation = EncDecTransfModelBPE.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[List[str], DataLoader],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> Union[List[str], List[Hypothesis]]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
        Args:
            audio: (a list) of paths to audio files.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
        )

    def _update_default_values(self, config: DictConfig):
        if self.training:  # don't do anything for training
            return config
        with open_dict(config):
            for k, v in self.cfg.train_ds.items():
                if k not in config:
                    config[k] = v
        return config

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            config = self._update_default_values(config)
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=self.tokenizer,
                ),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        shuffle = config['shuffle']
        if config.get('is_tarred', False):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):

        # create audio-only data loader
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the
        # dataloader is the total number of samples rather than the number of batches,
        # and this messes up the tqdm progress bar. So we set the number of steps manually
        # (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane,
            # i.e. <= # training batches, and don't change it. Otherwise, adjust
            # batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.
        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.
        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "transcript": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "transf_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "encoder_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.adapter(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask

    def compute_audio_loss(self, batch):

        if batch is None:
            return 0

        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        return transf_loss

    # PTL-specific methods
    def training_step(self, batch, batch_nb):

        audio_loss = self.compute_audio_loss(batch)

        tensorboard_logs = {
            'train_loss': audio_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
        }

        return {'loss': audio_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        else:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )

        beam_hypotheses = self.beam_search(
            encoder_hidden_states=enc_states, encoder_input_mask=enc_mask, return_beam_scores=False
        )
        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        translations = [self.tokenizer.ids_to_text(sent) for sent in beam_hypotheses.detach().cpu().tolist()]

        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])

        output_dict = {f'{eval_mode}_loss': transf_loss, 'translations': translations, 'ground_truths': ground_truths}

        self.validation_step_outputs.append(output_dict)

        return output_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, eval_mode="test")

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0, eval_mode: str = "val"):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for output in outputs:
            eval_loss = getattr(self, 'val_loss').compute()
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))

            # Gather translations and ground truths from all workers
            tr_and_gt = [None for _ in range(self.world_size)]
            # we also need to drop pairs where ground truth is an empty string
            if self.world_size > 1:
                dist.all_gather_object(
                    tr_and_gt, [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']
                )
            else:
                tr_and_gt[0] = [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']

            if self.global_rank == 0:
                _translations = []
                _ground_truths = []
                for rank in range(0, self.world_size):
                    _translations += [t for (t, g) in tr_and_gt[rank]]
                    _ground_truths += [g for (t, g) in tr_and_gt[rank]]

                sacre_bleu = SacreBLEUScore()(_translations, [[x] for x in _ground_truths]).item()
                sb_score = sacre_bleu * self.world_size

                wer_scores, wer_words = 0, 0
                for h, r in zip(_translations, _ground_truths):
                    wer_words += len(r.split())
                    wer_scores += editdistance.eval(h.split(), r.split())
                wer_score = 1.0 * wer_scores * self.world_size / wer_words

            else:
                sb_score = 0.0
                wer_score = 0.0

            self.log(f"{eval_mode}_loss", eval_loss, sync_dist=True)
            self.log(f"{eval_mode}_sacreBLEU", sb_score, sync_dist=True)
            self.log(f"{eval_mode}_WER", wer_score, sync_dist=True)
            self.val_loss.reset()

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_validation_epoch_end(outputs, dataloader_idx, eval_mode="test")

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.
        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    """ Transcription related methods """

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        # Freeze the encoder and decoder modules
        self.transf_decoder.freeze()

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch[0], input_signal_length=batch[1]
        )
        output = dict(log_probs=log_probs, encoded_len=encoded_len, enc_states=enc_states, enc_mask=enc_mask)
        return output

    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> List[str]:
        log_probs = outputs.pop('log_probs')
        encoded_len = outputs.pop('encoded_len')
        enc_states = outputs.pop('enc_states')
        enc_mask = outputs.pop('enc_mask')

        # TODO(@AlexGrinch): add support for returning logprobs from return_hypotheses=True
        del log_probs

        beam_hypotheses = (
            # TODO(@titu1994): maybe set return_beam_scores to True if theres no perf difference
            self.beam_search(encoder_hidden_states=enc_states, encoder_input_mask=enc_mask, return_beam_scores=False)
            .detach()
            .cpu()
            .numpy()
        )

        beam_hypotheses_out = [self.tokenizer.ids_to_text(hyp) for hyp in beam_hypotheses]
        del enc_states, enc_mask, encoded_len

        if trcfg.return_hypotheses:
            # TODO: add support for returning logprobs from return_hypotheses=True @AlexGrinch
            # dump log probs per file
            # for idx in range(logits.shape[0]):
            #     current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
            hypotheses = []
            for idx, hyp in enumerate(beam_hypotheses):
                hypotheses.append(
                    Hypothesis(
                        score=0.0,
                        y_sequence=beam_hypotheses[idx],
                        text=beam_hypotheses_out[idx],
                        length=len(beam_hypotheses[idx]),
                    )
                )

            # Replace output with Hypothesis list
            beam_hypotheses_out = hypotheses

        del beam_hypotheses

        return beam_hypotheses_out

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        # Unfreeze the encoder and decoder modules
        self.transf_decoder.unfreeze(partial=True)
