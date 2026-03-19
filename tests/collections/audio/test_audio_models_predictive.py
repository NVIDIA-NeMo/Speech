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

from nemo.collections.audio.models.enhancement import PredictiveAudioToAudioModel


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
def predictive_model_ncsn():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.ncsnpp.SpectrogramNoiseConditionalScoreNetworkPlusPlus',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'channels': [8, 8, 8, 8, 8],
        'num_res_blocks': 3,  # increased number of res blocks
        'pad_time_to': 64,  # pad to 64 frames for the time dimension
        'pad_dimension_to': 0,  # no padding in the frequency dimension
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.audio.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def predictive_model_conformer():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.conformer.SpectrogramConformer',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'feat_in': 256,  # input feature dimension = number of subbands
        'n_layers': 8,  # number of layers in the model
        'd_model': 64,  # the hidden size of the model
        'subsampling_factor': 1,  # subsampling factor for the model
        'self_attention_model': 'rel_pos',
        'n_heads': 8,  # number of heads for the model
        # streaming-related arguments
        # - this is a non-streaming config
        'conv_context_size': None,
        'conv_norm_type': 'layer_norm',
        'causal_downsampling': False,
        'att_context_size': [-1, -1],
        'att_context_style': 'regular',
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.audio.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def predictive_model_streaming_conformer():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.conformer.SpectrogramConformer',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'feat_in': 256,  # input feature dimension = number of subbands
        'n_layers': 8,  # number of layers in the model
        'd_model': 64,  # the hidden size of the model
        'subsampling_factor': 1,  # subsampling factor for the model
        'self_attention_model': 'rel_pos',
        'n_heads': 8,  # number of heads for the model
        # streaming-related arguments
        # - streaming config with causal convolutions and limited attention context
        'conv_context_size': 'causal',
        'conv_norm_type': 'layer_norm',
        'causal_downsampling': True,
        'att_context_size': [102, 16],
        'att_context_style': 'chunked_limited',
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.audio.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def predictive_model_transformer_unet_params_base():
    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.transformerunet.SpectrogramTransformerUNet',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'freq_dim': 256,  # input feature dimension = number of subbands
        'depth': 8,  # number of layers in the model
        'dim': 64,  # the hidden size of the model
        'heads': 8,  # number of heads for the model
        'adaptive_rmsnorm': False,  # should be false for predictive model
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.audio.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
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
def predictive_model_conformer_unet():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.conformer_unet.SpectrogramConformerUNet',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'feat_in': 256,  # input feature dimension = number of subbands
        'n_layers': 8,  # number of layers in the model
        'd_model': 64,  # the hidden size of the model
        'subsampling_factor': 1,  # subsampling factor for the model
        'self_attention_model': 'rel_pos',
        'n_heads': 8,  # number of heads for the model
        # streaming-related arguments
        # - this is a non-streaming config
        'conv_context_size': None,
        'conv_norm_type': 'layer_norm',
        'causal_downsampling': False,
        'att_context_size': [-1, -1],
        'att_context_style': 'regular',
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.audio.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def predictive_model_streaming_conformer_unet():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.conformer_unet.SpectrogramConformerUNet',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'feat_in': 256,  # input feature dimension = number of subbands
        'n_layers': 8,  # number of layers in the model
        'd_model': 64,  # the hidden size of the model
        'subsampling_factor': 1,  # subsampling factor for the model
        'self_attention_model': 'rel_pos',
        'n_heads': 8,  # number of heads for the model
        # streaming-related arguments
        # - streaming config with causal convolutions and limited attention context
        'conv_context_size': 'causal',
        'conv_norm_type': 'layer_norm',
        'causal_downsampling': True,
        'att_context_size': [102, 16],
        'att_context_style': 'chunked_limited',
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.audio.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture
def predictive_model_transformer_unet_params(predictive_model_transformer_unet_params_base, request):
    overrides = getattr(request, "param", {})

    for section, values in overrides.items():
        if section in predictive_model_transformer_unet_params_base and isinstance(
            predictive_model_transformer_unet_params_base[section], DictConfig
        ):
            for k, v in values.items():
                predictive_model_transformer_unet_params_base[section][k] = v
        else:
            predictive_model_transformer_unet_params_base[section] = values
    return predictive_model_transformer_unet_params_base


@pytest.fixture()
def predictive_model_transformer_unet(predictive_model_transformer_unet_params):
    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=predictive_model_transformer_unet_params)
    return model


@pytest.fixture()
def predictive_model_transformer_unet_with_trainer_and_mock_dataset(
    predictive_model_transformer_unet_params, mock_dataset_config
):
    # Add train and validation dataset configs
    predictive_model_transformer_unet_params['train_ds'] = {**mock_dataset_config, 'shuffle': True}
    predictive_model_transformer_unet_params['validation_ds'] = {**mock_dataset_config, 'shuffle': False}

    # Trainer config
    trainer_cfg = {
        'max_epochs': -1,
        'max_steps': 8,
        'logger': False,
        'use_distributed_sampler': False,
        'val_check_interval': 2,
        'limit_train_batches': 4,
        'accelerator': 'cpu',
        'enable_checkpointing': False,
    }
    predictive_model_transformer_unet_params['trainer'] = trainer_cfg

    trainer = pl.Trainer(**trainer_cfg)

    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = PredictiveAudioToAudioModel(cfg=predictive_model_transformer_unet_params, trainer=trainer)
    return model, trainer


class TestPredictiveModelNCSN:
    """Test predictive model with NCSN estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_ncsn):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_ncsn.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_ncsn, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_ncsn.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

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


class TestPredictiveModelConformer:
    """Test predictive model with conformer estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_conformer):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_conformer.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_conformer, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_conformer.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

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


class TestPredictiveModelStreamingConformer:
    """Test predictive model with streaming conformer estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_streaming_conformer):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_streaming_conformer.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_streaming_conformer, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_streaming_conformer.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

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


class TestPredictiveModelTransformerUNet:
    """Test predictive model with transformer_unet estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_transformer_unet):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_transformer_unet.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_transformer_unet, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_transformer_unet.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

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

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),
        ],
    )
    @pytest.mark.parametrize(
        "predictive_model_transformer_unet_params", [{"estimator": {"adaptive_rmsnorm": True}}], indirect=True
    )
    def test_adaptive_rms_ebabled_fails(self, predictive_model_transformer_unet, batch_size, sample_len):
        """Test that the predictive model raises TypeError when adaptive RMS turned on"""
        model = predictive_model_transformer_unet.eval()

        confdict = model.to_config_dict()

        sampling_rate = confdict['sample_rate']

        rng = torch.Generator()
        rng.manual_seed(0)

        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        with pytest.raises(TypeError):
            # fail because of adaptive RMS turned on for predictive model
            with torch.no_grad():
                _, _ = model.forward(input_signal=input_signal, input_length=input_signal_length)

    def test_training_step(self, predictive_model_transformer_unet_with_trainer_and_mock_dataset):
        model, _ = predictive_model_transformer_unet_with_trainer_and_mock_dataset
        model = model.train()

        for batch in itertools.islice(model._train_dl, 2):
            if isinstance(batch, dict):
                input_signal = batch['input_signal']
                input_length = batch['input_length']
                target_signal = batch.get('target_signal', input_signal)
            else:
                input_signal, input_length, target_signal, _ = batch
            if input_signal.ndim == 2:
                input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
            if target_signal.ndim == 2:
                target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')
            output_signal, _ = model.forward(input_signal=input_signal, input_length=input_length)
            loss = model.loss(estimate=output_signal, target=target_signal, input_length=input_length)
            loss.backward()

    def test_model_training(self, predictive_model_transformer_unet_with_trainer_and_mock_dataset):
        """
        Test that the model can be trained for a few steps. An evaluation step is also expected.
        """
        model, trainer = predictive_model_transformer_unet_with_trainer_and_mock_dataset
        model = model.train()
        trainer.fit(model)


class TestPredictiveModelConformerUNet:
    """Test predictive model with conformer U-Net estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_conformer_unet):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_conformer_unet.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_conformer_unet, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_conformer_unet.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

        with torch.no_grad():
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

            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol


class TestPredictiveModelStreamingConformerUNet:
    """Test predictive model with streaming conformer U-Net estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_streaming_conformer_unet):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_streaming_conformer_unet.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_streaming_conformer_unet, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_streaming_conformer_unet.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

        with torch.no_grad():
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

            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol
