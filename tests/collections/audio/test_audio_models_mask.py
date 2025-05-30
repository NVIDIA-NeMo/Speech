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

import importlib
import itertools
import json

import einops
import lhotse
import lightning.pytorch as pl
import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import DictConfig

from nemo.collections.audio.models import EncMaskDecAudioToAudioModel

try:
    importlib.import_module('torchaudio')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


@pytest.fixture(params=["nemo_manifest", "lhotse_cuts"])
def mock_dataset_config(tmp_path, request):
    num_files = 8
    num_samples = 16000

    for i in range(num_files):
        data = np.random.randn(num_samples, 1)
        sf.write(tmp_path / f"audio_{i}.wav", data, 16000)

    if request.param == "lhotse_cuts":
        with lhotse.CutSet.open_writer(tmp_path / "cuts.jsonl") as writer:
            for i in range(num_files):
                recording = lhotse.Recording.from_file(tmp_path / f"audio_{i}.wav")
                cut = lhotse.MonoCut(
                    id=f"audio_{i}",
                    start=0,
                    channel=0,
                    duration=num_samples / 16000,
                    recording=recording,
                    custom={"target_recording": recording},
                )
                writer.write(cut)

            return {
                'cuts_path': str(tmp_path / "cuts.jsonl"),
                'use_lhotse': True,
                'batch_size': 2,
                'num_workers': 1,
            }
    elif request.param == "nemo_manifest":
        with (tmp_path / "small_manifest.jsonl").open("w") as f:
            for i in range(num_files):
                entry = {
                    "noisy_filepath": str(tmp_path / f"audio_{i}.wav"),
                    "clean_filepath": str(tmp_path / f"audio_{i}.wav"),
                    "duration": num_samples / 16000,
                    "offset": 0,
                }
                f.write(f"{json.dumps(entry)}\n")
        return {
            'manifest_filepath': str(tmp_path / "small_manifest.jsonl"),
            'input_key': 'noisy_filepath',
            'target_key': 'clean_filepath',
            'use_lhotse': False,
            'batch_size': 2,
            'num_workers': 1,
        }
    else:
        raise NotImplementedError(f"Dataset type {request.param} not implemented")


@pytest.fixture()
def mask_model_rnn_params():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 512,
        'hop_length': 256,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
    }
    mask_estimator = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskEstimatorRNN',
        'num_outputs': model['num_outputs'],
        'num_subbands': encoder['fft_length'] // 2 + 1,
        'num_features': 256,
        'num_layers': 3,
        'bidirectional': True,
    }
    mask_processor = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskReferenceChannel',
        'ref_channel': 0,
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.SDRLoss',
        'scale_invariant': True,
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'mask_estimator': DictConfig(mask_estimator),
            'mask_processor': DictConfig(mask_processor),
            'loss': DictConfig(loss),
            'optim': {
                'name': 'adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    return model_config


@pytest.fixture()
def mask_model_rnn(mask_model_rnn_params):
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = EncMaskDecAudioToAudioModel(cfg=mask_model_rnn_params)
    return model


@pytest.fixture()
def mask_model_rnn_with_trainer_and_mock_dataset(mask_model_rnn_params, mock_dataset_config):
    # Add train and validation dataset configs
    mask_model_rnn_params["train_ds"] = {**mock_dataset_config, "shuffle": True}
    mask_model_rnn_params["validation_ds"] = {**mock_dataset_config, "shuffle": False}

    # Trainer config
    trainer_cfg = {
        "max_epochs": -1,
        "max_steps": 8,
        "logger": False,
        "use_distributed_sampler": False,
        "val_check_interval": 2,
        "limit_train_batches": 4,
        "accelerator": "cpu",
        "enable_checkpointing": False,
    }
    mask_model_rnn_params["trainer"] = trainer_cfg

    trainer = pl.Trainer(**trainer_cfg)

    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = EncMaskDecAudioToAudioModel(cfg=mask_model_rnn_params, trainer=trainer)

    return model, trainer


@pytest.fixture()
def mask_model_flexarray():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 512,
        'hop_length': 256,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
    }
    mask_estimator = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskEstimatorFlexChannels',
        'num_outputs': model['num_outputs'],
        'num_subbands': encoder['fft_length'] // 2 + 1,
        'num_blocks': 3,
        'channel_reduction_position': 3,
        'channel_reduction_type': 'average',
        'channel_block_type': 'transform_average_concatenate',
        'temporal_block_type': 'conformer_encoder',
        'temporal_block_num_layers': 5,
        'temporal_block_num_heads': 4,
        'temporal_block_dimension': 128,
        'mag_reduction': None,
        'mag_normalization': 'mean_var',
        'use_ipd': True,
        'ipd_normalization': 'mean',
    }
    mask_processor = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskReferenceChannel',
        'ref_channel': 0,
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.SDRLoss',
        'scale_invariant': True,
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'mask_estimator': DictConfig(mask_estimator),
            'mask_processor': DictConfig(mask_processor),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = EncMaskDecAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def bf_model_flexarray(mask_model_flexarray):

    model_config = mask_model_flexarray.to_config_dict()
    # Switch processor to beamformer
    model_config['mask_processor'] = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskBasedBeamformer',
        'filter_type': 'pmwf',
        'filter_beta': 0.0,
        'filter_rank': 'one',
        'ref_channel': 'max_snr',
        'ref_hard': 1,
        'ref_hard_use_grad': False,
        'ref_subband_weighting': False,
        'num_subbands': model_config['mask_estimator']['num_subbands'],
    }

    model = EncMaskDecAudioToAudioModel(cfg=model_config)

    return model


class TestMaskModelRNN:
    """Test masking model with RNN mask estimator."""

    @pytest.mark.unit
    def test_constructor(self, mask_model_rnn):
        """Test that the model can be constructed from a config dict."""
        model = mask_model_rnn.train()
        confdict = model.to_config_dict()
        instance2 = EncMaskDecAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, EncMaskDecAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, mask_model_rnn, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = mask_model_rnn.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            output_length_list = []
            for i in range(input_signal.size(0)):
                output, output_length = model.forward(
                    input_signal=input_signal[i : i + 1], input_length=input_signal_length[i : i + 1]
                )
                output_list.append(output)
                output_length_list.append(output_length)
            output_instance = torch.cat(output_list, 0)
            output_length_instance = torch.cat(output_length_list, 0)

            # batch size batch_size
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol

    def test_training_step(self, mask_model_rnn_with_trainer_and_mock_dataset):
        model, _ = mask_model_rnn_with_trainer_and_mock_dataset
        model = model.train()

        for batch in itertools.islice(model._train_dl, 2):
            # start boilerplate from EncMaskDecAudioToAudioModel.training_step
            if isinstance(batch, dict):
                # lhotse batches are dictionaries
                input_signal = batch["input_signal"]
                input_length = batch["input_length"]
                target_signal = batch.get("target_signal", input_signal)
            else:
                input_signal, input_length, target_signal, _ = batch

            if input_signal.ndim == 2:
                input_signal = einops.rearrange(input_signal, "B T -> B 1 T")
            if target_signal.ndim == 2:
                target_signal = einops.rearrange(target_signal, "B T -> B 1 T")
            # end boilerplate

            output_signal, _ = model.forward(input_signal=input_signal, input_length=input_length)
            loss = model.loss(estimate=output_signal, target=target_signal, input_length=input_length)
            loss.backward()

    def test_model_training(self, mask_model_rnn_with_trainer_and_mock_dataset):
        """
        Test that the model can be trained for a few steps. An evaluation step is also expected.
        """
        model, trainer = mask_model_rnn_with_trainer_and_mock_dataset
        model = model.train()
        trainer.fit(model)


class TestMaskModelFlexArray:
    """Test masking model with channel-flexible mask estimator."""

    @pytest.mark.unit
    def test_constructor(self, mask_model_flexarray):
        """Test that the model can be constructed from a config dict."""
        model = mask_model_flexarray.train()
        confdict = model.to_config_dict()
        instance2 = EncMaskDecAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, EncMaskDecAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, num_channels, sample_len",
        [
            (4, 1, 4),  # 1-channel, Example 1
            (2, 1, 8),  # 1-channel, Example 2
            (1, 1, 10),  # 1-channel, Example 3
            (4, 3, 4),  # 3-channel, Example 1
            (2, 3, 8),  # 3-channel, Example 2
            (1, 3, 10),  # 3-channel, Example 3
        ],
    )
    def test_forward_infer(self, mask_model_flexarray, batch_size, num_channels, sample_len):
        """Test that the model can run forward inference."""
        model = mask_model_flexarray.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, num_channels, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            output_length_list = []
            for i in range(input_signal.size(0)):
                output, output_length = model.forward(
                    input_signal=input_signal[i : i + 1], input_length=input_signal_length[i : i + 1]
                )
                output_list.append(output)
                output_length_list.append(output_length)
            output_instance = torch.cat(output_list, 0)
            output_length_instance = torch.cat(output_length_list, 0)

            # batch size batch_size
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol


class TestBFModelFlexArray:
    """Test beamforming model with channel-flexible mask estimator."""

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    def test_constructor(self, bf_model_flexarray):
        """Test that the model can be constructed from a config dict."""
        model = bf_model_flexarray.train()
        confdict = model.to_config_dict()
        instance2 = EncMaskDecAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, EncMaskDecAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize(
        "batch_size, num_channels, sample_len",
        [
            (4, 1, 4),  # 1-channel, Example 1
            (2, 1, 8),  # 1-channel, Example 2
            (1, 1, 10),  # 1-channel, Example 3
            (4, 3, 4),  # 3-channel, Example 1
            (2, 3, 8),  # 3-channel, Example 2
            (1, 3, 10),  # 3-channel, Example 3
        ],
    )
    def test_forward_infer(self, bf_model_flexarray, batch_size, num_channels, sample_len):
        """Test that the model can run forward inference."""
        model = bf_model_flexarray.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, num_channels, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            output_length_list = []
            for i in range(input_signal.size(0)):
                output, output_length = model.forward(
                    input_signal=input_signal[i : i + 1], input_length=input_signal_length[i : i + 1]
                )
                output_list.append(output)
                output_length_list.append(output_length)
            output_instance = torch.cat(output_list, 0)
            output_length_instance = torch.cat(output_length_list, 0)

            # batch size batch_size
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol
