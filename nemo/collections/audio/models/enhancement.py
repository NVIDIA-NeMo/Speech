# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

import einops
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.audio.models.audio_to_audio import AudioToAudioModel
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, LossType, NeuralType
from nemo.utils import logging

__all__ = [
    'EncMaskDecAudioToAudioModel',
    'ScoreBasedGenerativeAudioToAudioModel',
    'PredictiveAudioToAudioModel',
    'SchroedingerBridgeAudioToAudioModel',
]


class EncMaskDecAudioToAudioModel(AudioToAudioModel):
    """Class for encoder-mask-decoder audio processing models.

    The model consists of the following blocks:
        - encoder: transforms input multi-channel audio signal into an encoded representation (analysis transform)
        - mask_estimator: estimates a mask used by signal processor
        - mask_processor: mask-based signal processor, combines the encoded input and the estimated mask
        - decoder: transforms processor output into the time domain (synthesis transform)
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.sample_rate = self._cfg.sample_rate

        # Setup processing modules
        self.encoder = EncMaskDecAudioToAudioModel.from_config_dict(self._cfg.encoder)
        self.mask_estimator = EncMaskDecAudioToAudioModel.from_config_dict(self._cfg.mask_estimator)
        self.mask_processor = EncMaskDecAudioToAudioModel.from_config_dict(self._cfg.mask_processor)
        self.decoder = EncMaskDecAudioToAudioModel.from_config_dict(self._cfg.decoder)

        if 'mixture_consistency' in self._cfg:
            logging.debug('Using mixture consistency')
            self.mixture_consistency = EncMaskDecAudioToAudioModel.from_config_dict(self._cfg.mixture_consistency)
        else:
            logging.debug('Mixture consistency not used')
            self.mixture_consistency = None

        # Setup augmentation
        if hasattr(self.cfg, 'channel_augment') and self.cfg.channel_augment is not None:
            logging.debug('Using channel augmentation')
            self.channel_augmentation = EncMaskDecAudioToAudioModel.from_config_dict(self.cfg.channel_augment)
        else:
            logging.debug('Channel augmentation not used')
            self.channel_augmentation = None

        # Setup optional Optimization flags
        self.setup_optimization_flags()

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "input_signal": NeuralType(
                ('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)
            ),  # multi-channel format, channel dimension can be 1 for single-channel audio
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "output_signal": NeuralType(
                ('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)
            ),  # multi-channel format, channel dimension can be 1 for single-channel audio
            "output_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input_signal, input_length=None):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T] or [B, T, C]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.

        Returns:
            Output signal `output` in the time domain and the length of the output signal `output_length`.
        """
        batch_length = input_signal.size(-1)

        # Encoder
        encoded, encoded_length = self.encoder(input=input_signal, input_length=input_length)

        # Mask estimator
        mask, _ = self.mask_estimator(input=encoded, input_length=encoded_length)

        # Mask-based processor in the encoded domain
        processed, processed_length = self.mask_processor(input=encoded, input_length=encoded_length, mask=mask)

        # Mixture consistency
        if self.mixture_consistency is not None:
            processed = self.mixture_consistency(mixture=encoded, estimate=processed)

        # Decoder
        processed, processed_length = self.decoder(input=processed, input_length=processed_length)

        # Trim or pad the estimated signal to match input length
        processed = self.match_batch_length(input=processed, batch_length=batch_length)
        return processed, processed_length

    # PTL-specific methods
    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Apply channel augmentation
        if self.training and self.channel_augmentation is not None:
            input_signal = self.channel_augmentation(input=input_signal)

        # Process input
        processed_signal, _ = self.forward(input_signal=input_signal, input_length=input_length)

        # Calculate the loss
        loss = self.loss(estimate=processed_signal, target=target_signal, input_length=input_length)

        # Logs
        self.log('train_loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Return loss
        return loss

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Process input
        processed_signal, _ = self.forward(input_signal=input_signal, input_length=input_length)

        # Calculate the loss
        loss = self.loss(estimate=processed_signal, target=target_signal, input_length=input_length)

        # Update metrics
        if hasattr(self, 'metrics') and tag in self.metrics:
            # Update metrics for this (tag, dataloader_idx)
            for name, metric in self.metrics[tag][dataloader_idx].items():
                metric.update(preds=processed_signal, target=target_signal, input_length=input_length)

        # Log global step
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Return loss
        return {f'{tag}_loss': loss}

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results


class PredictiveAudioToAudioModel(AudioToAudioModel):
    """This models aims to directly estimate the coefficients
    in the encoded domain by applying a neural model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.sample_rate = self._cfg.sample_rate

        # Setup processing modules
        self.encoder = self.from_config_dict(self._cfg.encoder)
        self.decoder = self.from_config_dict(self._cfg.decoder)

        # Neural estimator
        self.estimator = self.from_config_dict(self._cfg.estimator)

        # Normalization
        self.normalize_input = self._cfg.get('normalize_input', False)

        # Term added to the denominator to improve numerical stability
        self.eps = self._cfg.get('eps', 1e-8)

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tnormalize_input: %s', self.normalize_input)
        logging.debug('\teps:             %s', self.eps)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)),
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "output_signal": NeuralType(('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)),
            "output_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input_signal, input_length=None):
        """Forward pass of the model.

        Args:
            input_signal: time-domain signal
            input_length: valid length of each example in the batch

        Returns:
            Output signal `output` in the time domain and the length of the output signal `output_length`.
        """
        batch_length = input_signal.size(-1)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)

        # Encoder
        encoded, encoded_length = self.encoder(input=input_signal, input_length=input_length)

        # Backbone
        estimated, estimated_length = self.estimator(input=encoded, input_length=encoded_length)

        # Decoder
        output, output_length = self.decoder(input=estimated, input_length=estimated_length)

        if self.normalize_input:
            # rescale to the original scale
            output = output * norm_scale

        # Trim or pad the estimated signal to match input length
        output = self.match_batch_length(input=output, batch_length=batch_length)
        return output, output_length

    # PTL-specific methods
    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Estimate the signal
        output_signal, _ = self.forward(input_signal=input_signal, input_length=input_length)

        # Calculate the loss
        loss = self.loss(estimate=output_signal, target=target_signal, input_length=input_length)

        # Logs
        self.log('train_loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return loss

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Estimate the signal
        output_signal, _ = self.forward(input_signal=input_signal, input_length=input_length)

        # Prepare output
        loss = self.loss(estimate=output_signal, target=target_signal, input_length=input_length)

        # Update metrics
        if hasattr(self, 'metrics') and tag in self.metrics:
            # Update metrics for this (tag, dataloader_idx)
            for name, metric in self.metrics[tag][dataloader_idx].items():
                metric.update(preds=output_signal, target=target_signal, input_length=input_length)

        # Log global step
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return {f'{tag}_loss': loss}


class ScoreBasedGenerativeAudioToAudioModel(AudioToAudioModel):
    """This models is using a score-based diffusion process to generate
    an encoded representation of the enhanced signal.

    The model consists of the following blocks:
        - encoder: transforms input multi-channel audio signal into an encoded representation (analysis transform)
        - estimator: neural model, estimates a score for the diffusion process
        - sde: stochastic differential equation (SDE) defining the forward and reverse diffusion process
        - sampler: sampler for the reverse diffusion process, estimates coefficients of the target signal
        - decoder: transforms sampler output into the time domain (synthesis transform)
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.sample_rate = self._cfg.sample_rate

        # Setup processing modules
        self.encoder = self.from_config_dict(self._cfg.encoder)
        self.decoder = self.from_config_dict(self._cfg.decoder)

        # Neural score estimator
        self.estimator = self.from_config_dict(self._cfg.estimator)

        # SDE
        self.sde = self.from_config_dict(self._cfg.sde)

        # Sampler
        if 'sde' in self._cfg.sampler:
            raise ValueError('SDE should be defined in the model config, not in the sampler config')
        if 'score_estimator' in self._cfg.sampler:
            raise ValueError('Score estimator should be defined in the model config, not in the sampler config')

        self.sampler = hydra.utils.instantiate(self._cfg.sampler, sde=self.sde, score_estimator=self.estimator)

        # Normalization
        self.normalize_input = self._cfg.get('normalize_input', False)

        # Metric evaluation
        self.max_utts_evaluation_metrics = self._cfg.get('max_utts_evaluation_metrics')

        if self.max_utts_evaluation_metrics is not None:
            logging.warning(
                'Metrics will be evaluated on first %d examples of the evaluation datasets.',
                self.max_utts_evaluation_metrics,
            )

        # Term added to the denominator to improve numerical stability
        self.eps = self._cfg.get('eps', 1e-8)

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tnormalize_input: %s', self.normalize_input)
        logging.debug('\teps:             %s', self.eps)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)),
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "output_signal": NeuralType(('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)),
            "output_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    @torch.inference_mode()
    def forward(self, input_signal, input_length=None):
        """Forward pass of the model.

        Forward pass of the model aplies the following steps:
            - encoder to obtain the encoded representation of the input signal
            - sampler to generate the estimated coefficients of the target signal
            - decoder to transform the sampler output into the time domain

        Args:
            input_signal: Tensor that represents a batch of time-domain audio signals,
                of shape [B, C, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, contains the individual lengths of the audio sequences.

        Returns:
            Output `output_signal` in the time domain and the length of the output signal `output_length`.
        """
        batch_length = input_signal.size(-1)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)

        # Encoder
        encoded, encoded_length = self.encoder(input=input_signal, input_length=input_length)

        # Sampler
        generated, generated_length = self.sampler(
            prior_mean=encoded, score_condition=encoded, state_length=encoded_length
        )

        # Decoder
        output, output_length = self.decoder(input=generated, input_length=generated_length)

        if self.normalize_input:
            # rescale to the original scale
            output = output * norm_scale

        # Trim or pad the estimated signal to match input length
        output = self.match_batch_length(input=output, batch_length=batch_length)
        return output, output_length

    @typecheck(
        input_types={
            "target_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "loss": NeuralType(None, LossType()),
        },
    )
    def _step(self, target_signal, input_signal, input_length=None):
        """Randomly generate a time step for each example in the batch, estimate
        the score and calculate the loss value.

        Note that this step does not include sampler.
        """
        batch_size = target_signal.size(0)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)
            # scale the target signal
            target_signal = target_signal / (norm_scale + self.eps)

        # Apply encoder to both target and the input
        input_enc, input_enc_len = self.encoder(input=input_signal, input_length=input_length)
        target_enc, _ = self.encoder(input=target_signal, input_length=input_length)

        # Generate random time steps
        sde_time = self.sde.generate_time(size=batch_size, device=input_enc.device)

        # Get the mean and the variance of the perturbation kernel
        pk_mean, pk_std = self.sde.perturb_kernel_params(state=target_enc, prior_mean=input_enc, time=sde_time)

        # Generate a random sample from a standard normal distribution
        z_norm = torch.randn_like(input_enc)

        # Prepare perturbed data
        perturbed_enc = pk_mean + pk_std * z_norm

        # Score is conditioned on the perturbed data and the input
        estimator_input = torch.cat([perturbed_enc, input_enc], dim=-3)

        # Estimate the score using the neural estimator
        # SDE time is used to inform the estimator about the current time step
        # Note:
        # - some implementations use `score = -self._raw_dnn_output(x, t, y)`
        # - this seems to be unimportant, and is an artifact of transfering code from the original Song's repo
        score_est, score_len = self.estimator(input=estimator_input, input_length=input_enc_len, condition=sde_time)

        # Score loss weighting as in Section 4.2 in http://arxiv.org/abs/1907.05600
        score_est = score_est * pk_std
        score_ref = -z_norm

        # Score matching loss on the normalized scores
        loss = self.loss(estimate=score_est, target=score_ref, input_length=score_len)

        return loss

    # PTL-specific methods
    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Calculate the loss
        loss = self._step(target_signal=target_signal, input_signal=input_signal, input_length=input_length)

        # Logs
        self.log('train_loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return loss

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Calculate loss
        loss = self._step(target_signal=target_signal, input_signal=input_signal, input_length=input_length)

        # Update metrics
        update_metrics = False
        if self.max_utts_evaluation_metrics is None:
            # Always update if max is not configured
            update_metrics = True
            # Number of examples to process
            num_examples = input_signal.size(0)  # batch size
        else:
            # Check how many examples have been used for metric calculation
            first_metric_name = next(iter(self.metrics[tag][dataloader_idx]))
            num_examples_evaluated = self.metrics[tag][dataloader_idx][first_metric_name].num_examples
            # Update metrics if some examples were not processed
            update_metrics = num_examples_evaluated < self.max_utts_evaluation_metrics
            # Number of examples to process
            num_examples = min(self.max_utts_evaluation_metrics - num_examples_evaluated, input_signal.size(0))

        if update_metrics:
            # Generate output signal
            output_signal, _ = self.forward(
                input_signal=input_signal[:num_examples, ...], input_length=input_length[:num_examples]
            )

            # Update metrics
            if hasattr(self, 'metrics') and tag in self.metrics:
                # Update metrics for this (tag, dataloader_idx)
                for name, metric in self.metrics[tag][dataloader_idx].items():
                    metric.update(
                        preds=output_signal,
                        target=target_signal[:num_examples, ...],
                        input_length=input_length[:num_examples],
                    )

        # Log global step
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return {f'{tag}_loss': loss}


class SchroedingerBridgeAudioToAudioModel(AudioToAudioModel):
    """This models is using a Schrödinger Bridge process to generate
    an encoded representation of the enhanced signal.

    The model consists of the following blocks:
        - encoder: transforms input audio signal into an encoded representation (analysis transform)
        - estimator: neural model, estimates the coefficients for the SB process
        - noise_schedule: defines the path between the clean and noisy signals
        - sampler: sampler for the reverse process, estimates coefficients of the target signal
        - decoder: transforms sampler output into the time domain (synthesis transform)

    References:
        Schrödinger Bridge for Generative Speech Enhancement, https://arxiv.org/abs/2407.16074
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.sample_rate = self._cfg.sample_rate

        # Setup processing modules
        self.encoder = self.from_config_dict(self._cfg.encoder)
        self.decoder = self.from_config_dict(self._cfg.decoder)

        # Neural estimator
        self.estimator = self.from_config_dict(self._cfg.estimator)
        self.estimator_output = self._cfg.estimator_output

        # Noise schedule
        self.noise_schedule = self.from_config_dict(self._cfg.noise_schedule)

        # Sampler
        self.sampler = hydra.utils.instantiate(
            self._cfg.sampler,
            noise_schedule=self.noise_schedule,
            estimator=self.estimator,
            estimator_output=self.estimator_output,
        )

        # Normalization
        self.normalize_input = self._cfg.get('normalize_input', False)

        # Metric evaluation
        self.max_utts_evaluation_metrics = self._cfg.get('max_utts_evaluation_metrics')

        if self.max_utts_evaluation_metrics is not None:
            logging.warning(
                'Metrics will be evaluated on first %d examples of the evaluation datasets.',
                self.max_utts_evaluation_metrics,
            )

        # Loss in the encoded domain
        if 'loss_encoded' in self._cfg:
            self.loss_encoded = self.from_config_dict(self._cfg.loss_encoded)
            self.loss_encoded_weight = self._cfg.get('loss_encoded_weight', 1.0)
        else:
            self.loss_encoded = None
            self.loss_encoded_weight = 0.0

        # Loss in the time domain
        if 'loss_time' in self._cfg:
            self.loss_time = self.from_config_dict(self._cfg.loss_time)
            self.loss_time_weight = self._cfg.get('loss_time_weight', 1.0)
        else:
            self.loss_time = None
            self.loss_time_weight = 0.0

        if self.loss is not None and (self.loss_encoded is not None or self.loss_time is not None):
            raise ValueError('Either ``loss`` or ``loss_encoded`` and ``loss_time`` should be defined, not both.')

        # Term added to the denominator to improve numerical stability
        self.eps = self._cfg.get('eps', 1e-8)

        # Setup optional optimization flags
        self.setup_optimization_flags()

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\testimator_output:    %s', self.estimator_output)
        logging.debug('\tnormalize_input:     %s', self.normalize_input)
        logging.debug('\tloss:                %s', self.loss)
        logging.debug('\tloss_encoded:        %s', self.loss_encoded)
        logging.debug('\tloss_encoded_weight: %s', self.loss_encoded_weight)
        logging.debug('\tloss_time:           %s', self.loss_time)
        logging.debug('\tloss_time_weight:    %s', self.loss_time_weight)
        logging.debug('\teps:                 %s', self.eps)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        # time-domain input
        return {
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)),
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        # time-domain output
        return {
            "output_signal": NeuralType(('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)),
            "output_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    @torch.inference_mode()
    def forward(self, input_signal, input_length=None):
        """Forward pass of the model.

        Forward pass of the model consists of the following steps
            - encoder to obtain the encoded representation of the input signal
            - sampler to generate the estimated coefficients of the target signal
            - decoder to transform the estimated output into the time domain

        Args:
            input_signal: Tensor that represents a batch of time-domain audio signals,
                of shape [B, C, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, contains the individual lengths of the audio sequences.

        Returns:
            Output `output_signal` in the time domain and the length of the output signal `output_length`.
        """
        batch_length = input_signal.size(-1)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)

        # Encoder
        encoded, encoded_length = self.encoder(input=input_signal, input_length=input_length)

        # Sampler
        generated, generated_length = self.sampler(
            prior_mean=encoded, estimator_condition=encoded, state_length=encoded_length
        )

        # Decoder
        output, output_length = self.decoder(input=generated, input_length=generated_length)

        if self.normalize_input:
            # rescale to the original scale
            output = output * norm_scale

        # Trim or pad the estimated signal to match input length
        output = self.match_batch_length(input=output, batch_length=batch_length)

        return output, output_length

    @typecheck(
        input_types={
            "target_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "loss": NeuralType(None, LossType()),
            "loss_encoded": NeuralType(None, LossType()),
            "loss_time": NeuralType(None, LossType()),
        },
    )
    def _step(self, target_signal, input_signal, input_length=None):
        """Randomly generate time step for each example in the batch, run neural estimator
        to estimate the target and calculate the loss.
        """
        batch_size = target_signal.size(0)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)
            # scale the target signal
            target_signal = target_signal / (norm_scale + self.eps)

        # Apply encoder to both target and the input
        # For example, if the encoder is STFT, then _enc is the complex-valued STFT of the corresponding signal
        input_enc, input_enc_len = self.encoder(input=input_signal, input_length=input_length)
        target_enc, _ = self.encoder(input=target_signal, input_length=input_length)

        # Generate random time steps
        process_time = self.noise_schedule.generate_time(size=batch_size, device=input_enc.device)

        # Prepare necessary info from the noise schedule
        alpha_t, alpha_bar_t, alpha_t_max = self.noise_schedule.get_alphas(time=process_time)
        sigma_t, sigma_bar_t, sigma_t_max = self.noise_schedule.get_sigmas(time=process_time)

        # Marginal distribution
        weight_target = alpha_t * sigma_bar_t**2 / (sigma_t_max**2 + self.eps)
        weight_input = alpha_bar_t * sigma_t**2 / (sigma_t_max**2 + self.eps)
        # view weights as [B, C, D, T]
        weight_target = weight_target.view(-1, 1, 1, 1)
        weight_input = weight_input.view(-1, 1, 1, 1)
        # mean
        mean_x = weight_target * target_enc + weight_input * input_enc
        # standard deviation
        std_x = alpha_t * sigma_bar_t * sigma_t / (sigma_t_max + self.eps)
        # view as [B, C, D, T]
        std_x = std_x.view(-1, 1, 1, 1)

        # Generate a random sample from a standard normal distribution
        z_norm = torch.randn_like(input_enc)

        # Generate a random sample from the marginal distribution
        x_t = mean_x + std_x * z_norm

        # Estimator is conditioned on the generated sample and the original input (prior)
        estimator_input = torch.cat([x_t, input_enc], dim=-3)

        # Neural estimator
        # Estimator input is the same data type as the encoder output
        # For example, if the encoder is STFT, then the estimator input and output are complex-valued coefficients
        estimate, estimate_len = self.estimator(
            input=estimator_input, input_length=input_enc_len, condition=process_time
        )

        # Prepare output target and calculate loss
        if self.estimator_output == 'data_prediction':
            if self.loss is not None:
                # Single loss in the encoded domain
                loss = self.loss(estimate=estimate, target=target_enc, input_length=estimate_len)
                loss_encoded = loss_time = None
            else:
                # Weighted loss between encoded and time domain
                loss = 0.0

                # Loss in the encoded domain
                if self.loss_encoded is not None:
                    # Loss between the estimate and the target in the encoded domain
                    loss_encoded = self.loss_encoded(estimate=estimate, target=target_enc, input_length=estimate_len)
                    # Weighting
                    loss += self.loss_encoded_weight * loss_encoded
                else:
                    loss_encoded = None

                # Loss in the time domain
                if self.loss_time is not None:
                    # Convert the estimate to the time domain
                    with typecheck.disable_checks():
                        # Note: stimate is FloatType, decoder requires SpectrogramType
                        estimate_signal, _ = self.decoder(input=estimate, input_length=estimate_len)

                    # Match estimate length
                    batch_length = input_signal.size(-1)
                    estimate_signal = self.match_batch_length(input=estimate_signal, batch_length=batch_length)

                    # Loss between the estimate and the target in the time domain
                    loss_time = self.loss_time(
                        estimate=estimate_signal, target=target_signal, input_length=input_length
                    )
                    # Weighting
                    loss += self.loss_time_weight * loss_time
                else:
                    loss_time = None
        else:
            raise NotImplementedError(f'Output type {self.estimator_output} is not implemented')

        return loss, loss_encoded, loss_time

    # PTL-specific methods
    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Calculate the loss
        loss, loss_encoded, loss_time = self._step(
            target_signal=target_signal, input_signal=input_signal, input_length=input_length
        )

        # Logs
        self.log('train_loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        if loss_encoded is not None:
            self.log('train_loss_encoded', loss_encoded)

        if loss_time is not None:
            self.log('train_loss_time', loss_time)

        return loss

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Calculate loss
        loss, *_ = self._step(target_signal=target_signal, input_signal=input_signal, input_length=input_length)

        # Update metrics
        update_metrics = False
        if self.max_utts_evaluation_metrics is None:
            # Always update if max is not configured
            update_metrics = True
            # Number of examples to process
            num_examples = input_signal.size(0)  # batch size
        else:
            # Check how many examples have been used for metric calculation
            first_metric_name = next(iter(self.metrics[tag][dataloader_idx]))
            num_examples_evaluated = self.metrics[tag][dataloader_idx][first_metric_name].num_examples
            # Update metrics if some examples were not processed
            update_metrics = num_examples_evaluated < self.max_utts_evaluation_metrics
            # Number of examples to process
            num_examples = min(self.max_utts_evaluation_metrics - num_examples_evaluated, input_signal.size(0))

        if update_metrics:
            # Generate output signal
            output_signal, _ = self.forward(
                input_signal=input_signal[:num_examples, ...], input_length=input_length[:num_examples]
            )

            # Update metrics
            if hasattr(self, 'metrics') and tag in self.metrics:
                # Update metrics for this (tag, dataloader_idx)
                for name, metric in self.metrics[tag][dataloader_idx].items():
                    metric.update(
                        preds=output_signal,
                        target=target_signal[:num_examples, ...],
                        input_length=input_length[:num_examples],
                    )

        # Log global step
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return {f'{tag}_loss': loss}
