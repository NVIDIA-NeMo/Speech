# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.data.audio_to_text_lhotse_prompt import LhotseSpeechToTextBpeDatasetWithPrompt
from nemo.collections.asr.metrics.bleu import BLEU
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.mixins import ASRTranscriptionMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils


@dataclass
class HybridRNNTCTCPromptTranscribeConfig(TranscribeConfig):
    """
    Configuration for Hybrid RNNT-CTC BPE Model with Prompt Transcription
    """

    target_lang: str = "en-US"
    prompt_field: str = "lang"


class EncDecHybridRNNTCTCBPEModelWithPrompt(EncDecHybridRNNTCTCBPEModel, ASRTranscriptionMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss, subword tokenization, and prompt conditioning."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Tokenizer is necessary for this model
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        # setup auxiliary CTC decoder
        if 'aux_ctc' not in cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )

        with open_dict(cfg):
            if self.tokenizer_type == "agg":
                cfg.aux_ctc.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.aux_ctc.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

            # Setup prompt settings - default to 128 prompts if not specified
            cfg.num_prompts = cfg.model_defaults.get('num_prompts', 128)

            # Make sure prompt_dictionary exists
            if 'prompt_dictionary' not in cfg.model_defaults:
                raise ValueError("No prompt_dictionary found in config.")

            # Set subsampling_factor in a place accessible to the class
            self.subsampling_factor = cfg.get('subsampling_factor', 8)

        if cfg.aux_ctc.decoder["num_classes"] < 1:
            logging.info(
                "\nReplacing placholder number of classes ({}) with actual number of classes - {}".format(
                    cfg.aux_ctc.decoder["num_classes"], len(vocabulary)
                )
            )
            cfg.aux_ctc.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize concat flag
        self.concat = False

        if self.cfg.model_defaults.get('initialize_prompt_feature', False):
            self.initialize_prompt_feature()

    def initialize_prompt_feature(self):
        """Initialize model components for prompt feature via concatenation."""
        logging.info("Model with prompt feature has been initialized")

        # Enable concatenation mode
        self.concat = True
        self.num_prompts = self.cfg.get('num_prompts', 128)

        # Setup projection layers
        proj_in_size = self.num_prompts + self._cfg.model_defaults.enc_hidden
        proj_out_size = self._cfg.model_defaults.enc_hidden

        self.prompt_kernel = torch.nn.Sequential(
            torch.nn.Linear(proj_in_size, proj_out_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_out_size * 2, proj_out_size),
        )

        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup bleu object
        self._bleu = BLEU(decoding=self.decoding, tokenize=self.cfg.get('bleu_tokenizer', "13a"), log_prediction=True)

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup CTC decoding
        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg
        self.ctc_decoding = CTCBPEDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer)

        # Setup CTC WER
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            if config.get('initialize_prompt_feature', True):
                dataset = LhotseSpeechToTextBpeDatasetWithPrompt(tokenizer=self.tokenizer, cfg=config)
                logging.info("Setting up Lhotse dataset with prompt support")
            else:
                dataset = LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer)
                logging.info("Setting up Lhotse dataset without prompt support")
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=dataset,
                tokenizer=self.tokenizer,
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

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

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
            target_lang: (str) target language ID for transcription

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        # Get target language from config
        target_lang = config.get('target_lang', 'en-US')

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': config.get('use_lhotse', True),
            'use_bucketing': False,
            'drop_last': False,
            'prompt_field': config.get('prompt_field', 'target_lang'),
            'initialize_prompt_feature': True,
            'prompt_dictionary': self.cfg.model_defaults.get('prompt_dictionary'),
            'num_prompts': self.cfg.model_defaults.get('num_prompts', 128),
            'subsampling_factor': self.cfg.get('subsampling_factor', 8),
            'default_lang': target_lang,
            'window_stride': self.cfg.preprocessor.get('window_stride', 0.01),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def _transcribe_forward(self, batch: tuple[torch.Tensor, ...], trcfg: HybridRNNTCTCPromptTranscribeConfig) -> dict:
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
                Expected structure: (audio, audio_lens, tokens, token_lens, prompt_targets)
                For transcription, we may only have (audio, audio_lens) or (audio, audio_lens, ..., prompt_targets)
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """
        # Handling DataLoader batch - should be a tuple of tensors
        # Expected structure: (audio, audio_lens, tokens, token_lens, prompt_targets)
        # For transcription, we may only have (audio, audio_lens) or (audio, audio_lens, ..., prompt_targets)
        audio, audio_lens = batch[0], batch[1]
        if len(batch) >= 5:
            # Prompt provided by the dataloader (one-hot vectors)
            prompt = batch[4]  # This should be the prompt_targets from dataset
        else:
            # Prompt to be built dynamically.
            prompt = None

        batch_size = audio.shape[0]

        if prompt is None:
            # The dataloader provided only audio + audio_lens, so we need to construct
            # the prompt as one-hot vectors dynamically using TranscribeConfig.
            target_lang = trcfg.target_lang

            # Get prompt dictionary and num_prompts from model config
            prompt_dict = self.cfg.model_defaults.get('prompt_dictionary')
            num_prompts = self.cfg.model_defaults.get('num_prompts', 128)

            if not prompt_dict:
                raise ValueError("Prompt dictionary is empty. Cannot create dynamic prompts.")

            # Get the prompt index for the target language
            if target_lang not in prompt_dict:
                available_keys = list(prompt_dict.keys())
                raise ValueError(
                    f"Unknown target language: '{target_lang}'. Available languages: {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}"
                )

            prompt_id = prompt_dict[target_lang]

            # Preprocess audio to get the actual feature dimensions (like streaming does)
            processed_signal, processed_signal_length = self.preprocessor(input_signal=audio, length=audio_lens)

            # Calculate exact hidden length using the same approach as streaming
            time_length = processed_signal.shape[2]  # Feature time dimension
            subsampling_factor = self.cfg.get('subsampling_factor', 8)
            hidden_length = math.ceil(time_length / subsampling_factor)

            # Create one-hot prompt tensor: (batch_size, time_steps, num_prompts)
            prompt = torch.zeros(batch_size, hidden_length, num_prompts, dtype=torch.float32, device=audio.device)
            prompt[:, :, prompt_id] = 1.0  # Set the target language prompt to 1

            # Now call forward with preprocessed signal and prompt
            encoded, encoded_len = self.forward(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length, prompt=prompt
            )
        else:
            # Prompt was provided, use normal forward path
            encoded, encoded_len = self.forward(input_signal=audio, input_signal_length=audio_lens, prompt=prompt)

        # Prepare output dictionary based on decoder type
        if self.cur_decoder == "rnnt":
            # RNNT Path - just use encoded outputs directly
            output = dict(encoded=encoded, encoded_len=encoded_len)
        else:
            # CTC Path - compute logits from encoder output
            logits = self.ctc_decoder(encoder_output=encoded)
            output = dict(logits=logits, encoded_len=encoded_len)
            del encoded

        return output

    @torch.no_grad()
    def transcribe(
        self,
        audio: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        override_config: Optional[HybridRNNTCTCPromptTranscribeConfig] = None,
        **prompt,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray audio array.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            partial_hypothesis: (Optional[List['Hypothesis']]) partial hypotheses for streaming
            num_workers: (int) number of workers for DataLoader
            channel_selector: (Optional[ChannelSelectorType]) select a single channel or a subset of channels from
                multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set
                to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            timestamps: (Optional[bool]) timestamps will be returned if set to True as part of hypothesis object
            override_config: (Optional[HybridRNNTCTCPromptTranscribeConfig]) override transcription config pre-defined by the user.
            **prompt: Optional input to construct the prompts for the model. Accepted formats include:
                target_lang: (str) target language ID for transcription (e.g., "en-US", "de-DE")
                prompt_field: (str) field name to use for prompt extraction from manifest
                Additional prompt parameters can be passed and will be forwarded to the transcription config.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        if self.cur_decoder not in ["ctc", "rnnt"]:
            raise ValueError(
                f"{self.cur_decoder} is not supported for cur_decoder. Supported values are ['ctc', 'rnnt']"
            )

        if timestamps is not None:
            if self.cur_decoder not in ["ctc", "rnnt"]:
                raise ValueError(
                    f"{self.cur_decoder} is not supported for cur_decoder. Supported values are ['ctc', 'rnnt']"
                )
            decoding_cfg = self.cfg.aux_ctc.decoding if self.cur_decoder == "ctc" else self.cfg.decoding
            if timestamps or (override_config is not None and override_config.timestamps):
                logging.info(
                    "Timestamps requested, setting decoding timestamps to True. Capture them in Hypothesis object, \
                        with output[idx].timestep['word'/'segment'/'char']"
                )
                return_hypotheses = True
                with open_dict(decoding_cfg):
                    decoding_cfg.compute_timestamps = True
                    decoding_cfg.preserve_alignments = True
            else:
                with open_dict(decoding_cfg):
                    decoding_cfg.compute_timestamps = False
                    decoding_cfg.preserve_alignments = False
            self.change_decoding_strategy(decoding_cfg, decoder_type=self.cur_decoder, verbose=False)

        # Create transcription config if not provided
        if override_config is None:
            # Extract target_lang from prompt or use default
            target_lang = prompt.get('target_lang', 'en-US')
            prompt_field = prompt.get('prompt_field', 'target_lang')

            trcfg = HybridRNNTCTCPromptTranscribeConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                timestamps=timestamps,
                target_lang=target_lang,
                prompt_field=prompt_field,
            )

        else:
            if not isinstance(override_config, HybridRNNTCTCPromptTranscribeConfig):
                raise ValueError(
                    f"override_config must be of type {HybridRNNTCTCPromptTranscribeConfig}, "
                    f"but got {type(override_config)}"
                )
            trcfg = override_config

        # Call parent class transcribe method with proper parameters
        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            partial_hypothesis=partial_hypothesis,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=trcfg,
        )

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
            "prompt": NeuralType(('B', 'T', 'D'), LabelsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        prompt=None,
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

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
            prompt: Tensor that represents the prompt embeddings,
                of shape (B, T, D) where D is the number of supported prompts.
                Used for prompt-conditioned encoding via concatenation with acoustic features.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = torch.transpose(encoded, 1, 2)  # B * D * T -> B * T * D

        if self.concat:
            if prompt.shape[1] > encoded.shape[1]:
                prompt = prompt[:, : encoded.shape[1], :]
            out_dtype = encoded.dtype  # this is dtype, which the decoder previously got from encoder

            # Concatenate encoded states with prompt
            concat_enc_states = torch.cat([encoded, prompt], dim=-1)

            # Apply joint projection
            encoded = self.prompt_kernel(concat_enc_states).to(
                out_dtype
            )  # cast: unexpectedly without cast dtype is different from out_dtype

        encoded = torch.transpose(encoded, 1, 2)  # B * T * D -> B * D * T
        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, prompt = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, prompt=prompt)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, prompt = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, prompt=prompt)
        del signal

        if self.cur_decoder == 'rnnt':
            best_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
            )
        else:
            logits = self.ctc_decoder(encoder_output=encoded)
            best_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=logits,
                decoder_lengths=encoded_len,
                return_hypotheses=False,
            )

        batch_size = signal_len.shape[0]
        sample_id = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size).cpu().detach().numpy()

        return list(zip(sample_id, best_hyp))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, prompt = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, prompt=prompt)
        del signal

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )
            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value

        # CTC WER calculation
        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        # BLEU score calculation
        self.bleu.update(
            predictions=encoded, predictions_lengths=encoded_len, targets=transcript, targets_lengths=transcript_len
        )
        bleu_metrics = self.bleu.compute(return_all_metrics=True, prefix="val_")
        tensorboard_logs.update(
            {
                'val_bleu_num': bleu_metrics['val_bleu_num'],
                'val_bleu_denom': bleu_metrics['val_bleu_denom'],
                'val_bleu_pred_len': bleu_metrics['val_bleu_pred_len'],
                'val_bleu_target_len': bleu_metrics['val_bleu_target_len'],
                'val_bleu': bleu_metrics['val_bleu'],
            }
        )
        self.bleu.reset()

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Inter-CTC losses and additional logging
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        # Calculate validation loss if required
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}

        # Calculate WER
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}

        # Calculate CTC WER if applicable
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        # Calculate BLEU score
        bleu_num = torch.stack([x['val_bleu_num'] for x in outputs]).sum(dim=0)
        bleu_denom = torch.stack([x['val_bleu_denom'] for x in outputs]).sum(dim=0)
        bleu_pred_len = torch.stack([x['val_bleu_pred_len'] for x in outputs]).sum(dim=0)
        bleu_target_len = torch.stack([x['val_bleu_target_len'] for x in outputs]).sum(dim=0)

        val_bleu = self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)
        tensorboard_logs['val_bleu'] = val_bleu

        # Finalize and log metrics
        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")

        return metrics

    @property
    def bleu(self):
        return self._bleu

    @bleu.setter
    def bleu(self, bleu):
        self._bleu = bleu

    @classmethod
    def get_transcribe_config(cls) -> HybridRNNTCTCPromptTranscribeConfig:
        """
        Get the default transcribe config for this model.
        """
        return HybridRNNTCTCPromptTranscribeConfig()

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        """
        Sets up the training data loader via a Dict-like object.
        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompt.LhotseSpeechToTextBpeDatasetWithPrompt`
        """
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
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompt.LhotseSpeechToTextBpeDatasetWithPrompt`
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
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompt.LhotseSpeechToTextBpeDatasetWithPrompt`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return None
