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

from nemo.collections.audio.models.enhancement import FlowMatchingAudioToAudioModel


def convert_to_dictconfig(d):
    """Recursively convert dictionary to DictConfig."""
    if isinstance(d, dict):
        return DictConfig({k: convert_to_dictconfig(v) for k, v in d.items()})
    return d


flow_matching_base_config_params = list(itertools.product([True, False], ["conditional_vector_field", "data"]))
flow_matching_base_config_ids = [
    f"{ssl}__{target}"
    for ssl, target in itertools.product(
        ["ssl_pretrain_masking", "no_ssl_pretrain_masking"], ["conditional_vector_field", "data"]
    )
]


@pytest.fixture(params=flow_matching_base_config_params, ids=flow_matching_base_config_ids)
def flow_matching_base_config(request):
    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 2,
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
    flow = {
        '_target_': 'nemo.collections.audio.parts.submodules.flow.OptimalTransportFlow',
        'time_min': 1e-8,
        'time_max': 1.0,
        'sigma_start': 1.0,
        'sigma_end': 1e-4,
    }
    sampler = {
        '_target_': 'nemo.collections.audio.parts.submodules.flow.ConditionalFlowMatchingEulerSampler',
        'num_steps': 2,
        'time_min': flow['time_min'],
        'time_max': flow['time_max'],
    }

    loss = {'_target_': 'nemo.collections.audio.losses.audio.MSELoss', 'ndim': 4}

    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.transformerunet.SpectrogramTransformerUNet',
        'in_channels': 2,
        'out_channels': 1,
        'freq_dim': 256,
        'dim': 32,
        'depth': 2,
        'heads': 2,
        'ff_mult': 2,
        'ff_dropout': 0.1,
        'attn_dropout': 0.0,
        'max_positions': 6000,
        'time_hidden_dim': 1024,
        'conv_pos_embed_kernel_size': 3,
    }
    trainer = {
        'max_epochs': -1,
        'max_steps': 8,
        'logger': False,
        'use_distributed_sampler': False,
        'val_check_interval': 2,
        'limit_train_batches': 4,
        'accelerator': 'cpu',
        'enable_checkpointing': False,
    }

    enable_ssl_pretrain_masking, estimator_target = request.param
    if enable_ssl_pretrain_masking:
        ssl_pretrain_masking = {
            '_target_': 'nemo.collections.audio.modules.ssl_pretrain_masking.SSLPretrainWithMaskedPatch',
            'patch_size': 10,
            'mask_fraction': 0.7,
        }
    else:
        ssl_pretrain_masking = None
    model['estimator_target'] = estimator_target
    sampler['estimator_target'] = estimator_target

    metrics = {
        'val': {
            'sisdr': {
                '_target_': 'torchmetrics.audio.ScaleInvariantSignalDistortionRatio',
            },
        },
    }

    model_base_config = {
        **model,
        'metrics': metrics,
        'p_cond': 1.0,
        'encoder': encoder,
        'decoder': decoder,
        'flow': flow,
        'sampler': sampler,
        'loss': loss,
        'estimator': estimator,
        'optim': {
            'name': 'adam',
            'lr': 0.001,
            'betas': (0.9, 0.98),
        },
        'trainer': trainer,
        'ssl_pretrain_masking': ssl_pretrain_masking,
    }
    return model_base_config


def test_flow_matching_model_init(flow_matching_base_config):
    flow_matching_config = convert_to_dictconfig(flow_matching_base_config)
    model = FlowMatchingAudioToAudioModel(cfg=flow_matching_config)
    assert isinstance(model, FlowMatchingAudioToAudioModel)


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
def flow_matching_model(flow_matching_base_config, request):
    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        return FlowMatchingAudioToAudioModel(cfg=convert_to_dictconfig(flow_matching_base_config))


@pytest.fixture()
def flow_matching_model_with_trainer_and_mock_dataset(flow_matching_base_config, mock_dataset_config):
    flow_matching_base_config['train_ds'] = {
        **mock_dataset_config,
        'shuffle': True,
    }
    flow_matching_base_config['validation_ds'] = {
        **mock_dataset_config,
        'shuffle': False,
    }
    flow_matching_config = convert_to_dictconfig(flow_matching_base_config)

    trainer = pl.Trainer(**flow_matching_config.trainer)

    # deterministic model init
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        model = FlowMatchingAudioToAudioModel(cfg=flow_matching_config, trainer=trainer)
    return model, trainer


@pytest.mark.parametrize("p_cond", [0, 0.9, 1.0])
@pytest.mark.parametrize("eval", [True, False])
@pytest.mark.parametrize(
    "batch_size, sample_len",
    [
        (4, 4),
        (2, 8),
        (1, 10),
    ],
)
def test_flow_matching_model_forward(flow_matching_model, batch_size, sample_len, eval, p_cond):
    model = flow_matching_model.eval()
    model.p_cond = p_cond

    confdict = model.to_config_dict()
    sampling_rate = confdict['sample_rate']
    rng = torch.Generator()
    rng.manual_seed(0)
    input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
    input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.long)

    with torch.no_grad():
        if eval:
            output_batch, output_length_batch = model.forward_eval(
                input_signal=input_signal, input_length=input_signal_length
            )
        else:
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

    assert input_signal.shape == output_batch.shape, "Input and output batch shapes must match"
    assert input_signal_length.shape == output_length_batch.shape, "Input and output length shapes must match"
    assert torch.all(input_signal_length == output_length_batch), "Input and output lengths must match"


def test_flow_matching_model_step(flow_matching_model_with_trainer_and_mock_dataset):
    model, _ = flow_matching_model_with_trainer_and_mock_dataset
    model = model.train()

    for batch in itertools.islice(model._train_dl, 2):
        # start boilerplate from FlowMatchingAudioToAudioModel.training_step
        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch.get('target_signal', input_signal)
        else:
            input_signal, input_length, target_signal, _ = batch
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')
        # end boilerplate

        loss = model._step(target_signal=target_signal, input_signal=input_signal, input_length=input_length)
        loss.backward()


def test_flow_matching_model_training(flow_matching_model_with_trainer_and_mock_dataset):
    """
    Test that the model can be trained for a few steps. An evaluation step is also expected.
    """
    model, trainer = flow_matching_model_with_trainer_and_mock_dataset
    model = model.train()

    trainer.fit(model)
