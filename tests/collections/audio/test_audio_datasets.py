# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch.cuda
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.audio.data import audio_to_audio_dataset
from nemo.collections.audio.data.audio_to_audio import (
    ASRAudioProcessor,
    AudioToTargetDataset,
    AudioToTargetWithEmbeddingDataset,
    AudioToTargetWithReferenceDataset,
    _audio_collate_fn,
)
from nemo.collections.audio.data.audio_to_audio_lhotse import (
    LhotseAudioToTargetDataset,
    convert_manifest_nemo_to_lhotse,
)
from nemo.collections.audio.parts.utils.audio import get_segment_start
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config


class TestAudioDatasets:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2])
    @pytest.mark.parametrize('num_targets', [1, 3])
    def test_list_to_multichannel(self, num_channels, num_targets):
        """Test conversion of a list of arrays into"""
        random_seed = 42
        num_samples = 1000

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Multi-channel signal
        golden_target = _rng.normal(size=(num_channels * num_targets, num_samples))

        # Create a list of num_targets signals with num_channels channels
        target_list = [golden_target[n * num_channels : (n + 1) * num_channels, :] for n in range(num_targets)]

        # Check the original signal is not modified
        assert (ASRAudioProcessor.list_to_multichannel(golden_target) == golden_target).all()
        # Check the list is converted back to the original signal
        assert (ASRAudioProcessor.list_to_multichannel(target_list) == golden_target).all()

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2])
    def test_processor_process_audio(self, num_channels):
        """Test signal normalization in process_audio."""
        num_samples = 1000
        num_examples = 30

        signals = ['input_signal', 'target_signal', 'reference_signal']

        for normalization_signal in [None] + signals:
            # Create processor
            processor = ASRAudioProcessor(
                sample_rate=16000, random_offset=False, normalization_signal=normalization_signal
            )

            # Generate random signals
            for n in range(num_examples):
                example = {signal: torch.randn(num_channels, num_samples) for signal in signals}
                processed_example = processor.process_audio(example)

                # Expected scale
                if normalization_signal:
                    scale = 1.0 / (example[normalization_signal].abs().max() + processor.eps)
                else:
                    scale = 1.0

                # Make sure all signals are scaled as expected
                for signal in signals:
                    assert torch.allclose(
                        processed_example[signal], example[signal] * scale
                    ), f'Failed example {n} signal {signal}'

    @pytest.mark.unit
    def test_audio_collate_fn(self):
        """Test `_audio_collate_fn`"""
        batch_size = 16
        random_seed = 42
        atol = 1e-5

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        signal_to_channels = {
            'input_signal': 2,
            'target_signal': 1,
            'reference_signal': 1,
        }

        signal_to_length = {
            'input_signal': _rng.integers(low=5, high=25, size=batch_size),
            'target_signal': _rng.integers(low=5, high=25, size=batch_size),
            'reference_signal': _rng.integers(low=5, high=25, size=batch_size),
        }

        # Generate batch
        batch = []
        for n in range(batch_size):
            item = dict()
            for signal, num_channels in signal_to_channels.items():
                random_signal = _rng.normal(size=(num_channels, signal_to_length[signal][n]))
                random_signal = np.squeeze(random_signal)  # get rid of channel dimention for single-channel
                item[signal] = torch.tensor(random_signal)
            batch.append(item)

        # Run UUT
        batched = _audio_collate_fn(batch)

        batched_signals = {
            'input_signal': batched[0].cpu().detach().numpy(),
            'target_signal': batched[2].cpu().detach().numpy(),
            'reference_signal': batched[4].cpu().detach().numpy(),
        }

        batched_lengths = {
            'input_signal': batched[1].cpu().detach().numpy(),
            'target_signal': batched[3].cpu().detach().numpy(),
            'reference_signal': batched[5].cpu().detach().numpy(),
        }

        # Check outputs
        for signal, b_signal in batched_signals.items():
            for n in range(batch_size):
                # Check length
                uut_length = batched_lengths[signal][n]
                golden_length = signal_to_length[signal][n]
                assert (
                    uut_length == golden_length
                ), f'Example {n} signal {signal} length mismatch: batched ({uut_length}) != golden ({golden_length})'

                uut_signal = b_signal[n][:uut_length, ...]
                golden_signal = batch[n][signal][:uut_length, ...].cpu().detach().numpy()
                assert np.allclose(
                    uut_signal, golden_signal, atol=atol
                ), f'Example {n} signal {signal} value mismatch.'

    @pytest.mark.unit
    def test_audio_to_target_dataset(self):
        """Test AudioWithTargetDataset in different configurations.

        Test below cover the following:
        1) no constraints
        2) filtering based on signal duration
        3) use with channel selector
        4) use with fixed audio duration and random subsegments
        5) collate a batch of items

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': 'path/to/path_to_target.wav',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    # filenames
                    signal_filename = f'{signal}_{n:02d}.wav'

                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_dataset(config)

            # Prepare lhotse manifest
            cuts_path = manifest_filepath.replace('.json', '_cuts.jsonl')
            convert_manifest_nemo_to_lhotse(
                input_manifest=manifest_filepath,
                output_manifest=cuts_path,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
            )

            # Prepare lhotse dataset
            config_lhotse = {
                'cuts_path': cuts_path,
                'use_lhotse': True,
                'sample_rate': sample_rate,
                'batch_size': 1,
            }
            dl_lhotse = get_lhotse_dataloader_from_config(
                OmegaConf.create(config_lhotse), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
            )
            dataset_lhotse = [item for item in dl_lhotse]

            # Test number of channels
            for signal in data:
                assert data_num_channels[signal] == dataset.num_channels(
                    signal
                ), f'Num channels not correct for signal {signal}'
                assert data_num_channels[signal] == dataset_factory.num_channels(
                    signal
                ), f'Num channels not correct for signal {signal}'

            # Test returned examples
            for n in range(num_examples):
                for signal in data:
                    golden_signal = data[signal][n]

                    for use_lhotse in [False, True]:
                        item_signal = (
                            dataset_lhotse[n][signal].squeeze(0) if use_lhotse else dataset.__getitem__(n)[signal]
                        )
                        item_factory_signal = dataset_factory.__getitem__(n)[signal]

                        assert (
                            item_signal.shape == golden_signal.shape
                        ), f'Test 1, use_lhotse={use_lhotse}: Signal {signal} item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                        assert np.allclose(
                            item_signal, golden_signal, atol=atol
                        ), f'Test 1, use_lhotse={use_lhotse}: Failed for example {n}, signal {signal} (random seed {random_seed})'

                        assert np.allclose(
                            item_factory_signal, golden_signal, atol=atol
                        ), f'Test 1, use_lhotse={use_lhotse}: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2
            # - Filtering based on signal duration
            min_duration = 3.5
            max_duration = 7.5

            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                min_duration=min_duration,
                max_duration=max_duration,
                sample_rate=sample_rate,
            )

            # Prepare lhotse dataset
            config_lhotse = {
                'cuts_path': cuts_path,
                'use_lhotse': True,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'sample_rate': sample_rate,
                'batch_size': 1,
            }
            dl_lhotse = get_lhotse_dataloader_from_config(
                OmegaConf.create(config_lhotse), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
            )
            dataset_lhotse = [item for item in dl_lhotse]

            filtered_examples = [n for n, val in enumerate(data_duration) if min_duration <= val <= max_duration]

            for n in range(len(dataset)):
                for use_lhotse in [False, True]:
                    for signal in data:
                        item_signal = (
                            dataset_lhotse[n][signal].squeeze(0) if use_lhotse else dataset.__getitem__(n)[signal]
                        )
                        golden_signal = data[signal][filtered_examples[n]]
                        assert (
                            item_signal.shape == golden_signal.shape
                        ), f'Test 2, use_lhotse={use_lhotse}: Signal {signal} item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'

                        assert np.allclose(
                            item_signal, golden_signal, atol=atol
                        ), f'Test 2, use_lhotse={use_lhotse}: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 3
            # - Use channel selector
            channel_selector = {
                'input_signal': [0, 2],
                'target_signal': 1,
            }

            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                input_channel_selector=channel_selector['input_signal'],
                target_channel_selector=channel_selector['target_signal'],
                sample_rate=sample_rate,
            )

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                for signal in data:
                    cs = channel_selector[signal]
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n][cs, ...]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 3: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 4
            # - Use fixed duration (random segment selection)
            audio_duration = 4.0
            audio_duration_samples = int(np.floor(audio_duration * sample_rate))

            filtered_examples = [n for n, val in enumerate(data_duration) if val >= audio_duration]

            for random_offset in [True, False]:
                # Test subsegments with the default fixed offset and a random offset

                dataset = AudioToTargetDataset(
                    manifest_filepath=manifest_filepath,
                    input_key=data_key['input_signal'],
                    target_key=data_key['target_signal'],
                    sample_rate=sample_rate,
                    min_duration=audio_duration,
                    audio_duration=audio_duration,
                    random_offset=random_offset,  # random offset when selecting subsegment
                )

                # Prepare lhotse dataset
                config_lhotse = {
                    'cuts_path': cuts_path,
                    'use_lhotse': True,
                    'min_duration': audio_duration,
                    'truncate_duration': audio_duration,
                    'truncate_offset_type': 'random' if random_offset else 'start',
                    'sample_rate': sample_rate,
                    'batch_size': 1,
                }
                dl_lhotse = get_lhotse_dataloader_from_config(
                    OmegaConf.create(config_lhotse), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
                )
                dataset_lhotse = [item for item in dl_lhotse]

                for n in range(len(dataset)):
                    for use_lhotse in [False, True]:
                        item = dataset_lhotse[n] if use_lhotse else dataset.__getitem__(n)
                        golden_start = golden_end = None
                        for signal in data:
                            item_signal = item[signal].squeeze(0) if use_lhotse else item[signal]
                            full_golden_signal = data[signal][filtered_examples[n]]

                            # Find random segment using correlation on the first channel
                            # of the first signal, and then use it fixed for other signals
                            if golden_start is None:
                                golden_start = get_segment_start(
                                    signal=full_golden_signal[0, :], segment=item_signal[0, :]
                                )
                                if not random_offset:
                                    assert (
                                        golden_start == 0
                                    ), f'Test 4, use_lhotse={use_lhotse}: Expecting the signal to start at 0 when random_offset is False'

                                golden_end = golden_start + audio_duration_samples
                            golden_signal = full_golden_signal[..., golden_start:golden_end]

                            # Test length is correct
                            assert (
                                item_signal.shape[-1] == audio_duration_samples
                            ), f'Test 4, use_lhotse={use_lhotse}: Signal length ({item_signal.shape[-1]}) not matching the expected length ({audio_duration_samples})'

                            assert (
                                item_signal.shape == golden_signal.shape
                            ), f'Test 4, use_lhotse={use_lhotse}: Signal {signal} item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                            # Test signal values
                            assert np.allclose(
                                item_signal, golden_signal, atol=atol
                            ), f'Test 4, use_lhotse={use_lhotse}: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 5:
            # - Test collate_fn
            batch_size = 16

            for use_lhotse in [False, True]:
                if use_lhotse:
                    # Get batch from lhotse dataloader
                    config_lhotse['batch_size'] = batch_size
                    dl_lhotse = get_lhotse_dataloader_from_config(
                        OmegaConf.create(config_lhotse),
                        global_rank=0,
                        world_size=1,
                        dataset=LhotseAudioToTargetDataset(),
                    )
                    batched = next(iter(dl_lhotse))
                else:
                    # Get examples from dataset and collate into a batch
                    batch = [dataset.__getitem__(n) for n in range(batch_size)]
                    batched = dataset.collate_fn(batch)

                # Test all shapes and lengths
                for n, signal in enumerate(data.keys()):
                    length = signal.replace('_signal', '_length')

                    if isinstance(batched, dict):
                        signal_shape = batched[signal].shape
                        signal_len = batched[length]
                    else:
                        signal_shape = batched[2 * n].shape
                        signal_len = batched[2 * n + 1]

                    assert signal_shape == (
                        batch_size,
                        data_num_channels[signal],
                        audio_duration_samples,
                    ), f'Test 5, use_lhotse={use_lhotse}: Unexpected signal {signal} shape {signal_shape}'
                    assert (
                        len(signal_len) == batch_size
                    ), f'Test 5, use_lhotse={use_lhotse}: Unexpected length of signal_len ({len(signal_len)})'
                    assert all(
                        signal_len == audio_duration_samples
                    ), f'Test 5, use_lhotse={use_lhotse}: Unexpected signal_len {signal_len}'

    @pytest.mark.unit
    def test_audio_to_target_dataset_with_target_list(self):
        """Test AudioWithTargetDataset when the input manifest has a list
        of audio files in the target key.

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': ['path/to/path_to_target_ch0.wav', 'path/to/path_to_target_ch1.wav'],
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    if signal == 'target_signal':
                        # Save targets as individual files
                        signal_filename = []
                        for ch in range(data_num_channels[signal]):
                            # add current filename
                            signal_filename.append(f'{signal}_{n:02d}_ch_{ch}.wav')
                            # write audio file
                            sf.write(
                                os.path.join(test_dir, signal_filename[-1]),
                                data[signal][n][ch, :],
                                sample_rate,
                                'float',
                            )
                    else:
                        # single file
                        signal_filename = f'{signal}_{n:02d}.wav'

                        # write audio files
                        sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                sample_rate=sample_rate,
            )

            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_dataset(config)

            # Prepare lhotse manifest
            cuts_path = manifest_filepath.replace('.json', '_cuts.jsonl')
            convert_manifest_nemo_to_lhotse(
                input_manifest=manifest_filepath,
                output_manifest=cuts_path,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
            )

            # Prepare lhotse dataset
            config_lhotse = {
                'cuts_path': cuts_path,
                'use_lhotse': True,
                'sample_rate': sample_rate,
                'batch_size': 1,
            }
            dl_lhotse = get_lhotse_dataloader_from_config(
                OmegaConf.create(config_lhotse), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
            )
            dataset_lhotse = [item for item in dl_lhotse]

            for n in range(num_examples):
                for use_lhotse in [False, True]:
                    item = dataset_lhotse[n] if use_lhotse else dataset.__getitem__(n)
                    item_factory = dataset_factory.__getitem__(n)
                    for signal in data:
                        item_signal = item[signal].squeeze(0) if use_lhotse else item[signal]
                        golden_signal = data[signal][n]
                        assert (
                            item_signal.shape == golden_signal.shape
                        ), f'Test 1, use_lhotse={use_lhotse}: Signal {signal} item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                        assert np.allclose(
                            item_signal, golden_signal, atol=atol
                        ), f'Test 1, use_lhotse={use_lhotse}: Failed for example {n}, signal {signal} (random seed {random_seed})'

                        assert np.allclose(
                            item_factory[signal], golden_signal, atol=atol
                        ), f'Test 1, use_lhotse={use_lhotse}: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2
            # Set target as the first channel of input_filepath and all files listed in target_filepath.
            # In this case, the target will have 3 channels.
            # Note: this is currently not supported by lhotse, so we only test the default dataset here.
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=[data_key['input_signal'], data_key['target_signal']],
                target_channel_selector=0,
                sample_rate=sample_rate,
            )

            for n in range(num_examples):
                item = dataset.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    if signal == 'target_signal':
                        # add the first channel of the input
                        golden_signal = np.concatenate([data['input_signal'][n][0:1, ...], golden_signal], axis=0)
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 2: Failed for example {n}, signal {signal} (random seed {random_seed})'

    @pytest.mark.unit
    def test_audio_to_target_dataset_for_inference(self):
        """Test AudioWithTargetDataset when target_key is
        not set, i.e., it is `None`. This is the case, e.g., when
        running inference, and a target is not available.

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:
            # Build metadata for manifest
            metadata = []
            for n in range(num_examples):
                meta = dict()
                for signal in data:
                    # filenames
                    signal_filename = f'{signal}_{n:02d}.wav'
                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')
                    # update metadata
                    meta[data_key[signal]] = signal_filename
                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=None,  # target_signal will be empty
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': None,
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_dataset(config)

            # Prepare lhotse manifest
            cuts_path = manifest_filepath.replace('.json', '_cuts.jsonl')
            convert_manifest_nemo_to_lhotse(
                input_manifest=manifest_filepath,
                output_manifest=cuts_path,
                input_key=data_key['input_signal'],
                target_key=None,
            )

            # Prepare lhotse dataset
            config_lhotse = {
                'cuts_path': cuts_path,
                'use_lhotse': True,
                'sample_rate': sample_rate,
                'batch_size': 1,
            }
            dl_lhotse = get_lhotse_dataloader_from_config(
                OmegaConf.create(config_lhotse), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
            )
            dataset_lhotse = [item for item in dl_lhotse]

            for n in range(num_examples):

                for label in ['original', 'factory', 'lhotse']:

                    if label == 'original':
                        item = dataset.__getitem__(n)
                    elif label == 'factory':
                        item = dataset_factory.__getitem__(n)
                    elif label == 'lhotse':
                        item = dataset_lhotse[n]
                    else:
                        raise ValueError(f'Unknown label {label}')

                    # Check target is None
                    if 'target_signal' in item:
                        assert item['target_signal'].numel() == 0, f'{label}: target_signal is expected to be empty.'

                    # Check valid signals
                    for signal in data:

                        item_signal = item[signal].squeeze(0) if label == 'lhotse' else item[signal]
                        golden_signal = data[signal][n]
                        assert (
                            item_signal.shape == golden_signal.shape
                        ), f'{label} -- Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                        assert np.allclose(
                            item_signal, golden_signal, atol=atol
                        ), f'{label} -- Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

    @pytest.mark.unit
    def test_audio_to_target_with_reference_dataset(self):
        """Test AudioWithTargetWithReferenceDataset in different configurations.

        1) reference synchronized with input and target
        2) reference not synchronized

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': 'path/to/path_to_target.wav',
            'reference_filepath': 'path/to/path_to_reference.wav',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
            'reference_signal': 1,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
            'reference_signal': 'reference_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    # filenames
                    signal_filename = f'{signal}_{n:02d}.wav'

                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            # - Reference is not synchronized with input and target, so whole reference signal will be loaded
            dataset = AudioToTargetWithReferenceDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                reference_key=data_key['reference_signal'],
                reference_is_synchronized=False,
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'reference_key': data_key['reference_signal'],
                'reference_is_synchronized': False,
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_with_reference_dataset(config)

            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2
            # - Use fixed duration (random segment selection)
            # - Reference is synchronized with input and target, so the same segment of reference signal will be loaded
            audio_duration = 4.0
            audio_duration_samples = int(np.floor(audio_duration * sample_rate))
            dataset = AudioToTargetWithReferenceDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                reference_key=data_key['reference_signal'],
                reference_is_synchronized=True,
                sample_rate=sample_rate,
                min_duration=audio_duration,
                audio_duration=audio_duration,
                random_offset=True,
            )

            filtered_examples = [n for n, val in enumerate(data_duration) if val >= audio_duration]

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                golden_start = golden_end = None
                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    full_golden_signal = data[signal][filtered_examples[n]]

                    # Find random segment using correlation on the first channel
                    # of the first signal, and then use it fixed for other signals
                    if golden_start is None:
                        golden_start = get_segment_start(signal=full_golden_signal[0, :], segment=item_signal[0, :])
                        golden_end = golden_start + audio_duration_samples
                    golden_signal = full_golden_signal[..., golden_start:golden_end]

                    # Test length is correct
                    assert (
                        item_signal.shape[-1] == audio_duration_samples
                    ), f'Test 2: Signal {signal} length ({item_signal.shape[-1]}) not matching the expected length ({audio_duration_samples})'

                    # Test signal values
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 2: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 3
            # - Use fixed duration (random segment selection)
            # - Reference is not synchronized with input and target, so whole reference signal will be loaded
            audio_duration = 4.0
            audio_duration_samples = int(np.floor(audio_duration * sample_rate))
            dataset = AudioToTargetWithReferenceDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                reference_key=data_key['reference_signal'],
                reference_is_synchronized=False,
                sample_rate=sample_rate,
                min_duration=audio_duration,
                audio_duration=audio_duration,
                random_offset=True,
            )

            filtered_examples = [n for n, val in enumerate(data_duration) if val >= audio_duration]

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                golden_start = golden_end = None
                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    full_golden_signal = data[signal][filtered_examples[n]]

                    if signal == 'reference_signal':
                        # Complete signal is loaded for reference
                        golden_signal = full_golden_signal
                    else:
                        # Find random segment using correlation on the first channel
                        # of the first signal, and then use it fixed for other signals
                        if golden_start is None:
                            golden_start = get_segment_start(
                                signal=full_golden_signal[0, :], segment=item_signal[0, :]
                            )
                            golden_end = golden_start + audio_duration_samples
                        golden_signal = full_golden_signal[..., golden_start:golden_end]

                        # Test length is correct
                        assert (
                            item_signal.shape[-1] == audio_duration_samples
                        ), f'Test 3: Signal {signal} length ({item_signal.shape[-1]}) not matching the expected length ({audio_duration_samples})'
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    # Test signal values
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 3: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 4:
            # - Test collate_fn
            batch_size = 16
            batch = [dataset.__getitem__(n) for n in range(batch_size)]
            _ = dataset.collate_fn(batch)

    @pytest.mark.unit
    def test_audio_to_target_with_embedding_dataset(self):
        """Test AudioWithTargetWithEmbeddingDataset.

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': 'path/to/path_to_target.wav',
            'embedding_filepath': 'path/to/path_to_embedding.npy',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
            'embedding_vector': 1,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        embedding_length = 64  # 64-dimensional embedding vector
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
            'embedding_vector': 'embedding_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                data_length = embedding_length if signal == 'embedding_vector' else data_duration_samples[n]

                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_length))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_length))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    if signal == 'embedding_vector':
                        signal_filename = f'{signal}_{n:02d}.npy'
                        np.save(os.path.join(test_dir, signal_filename), data[signal][n])

                    else:
                        # filenames
                        signal_filename = f'{signal}_{n:02d}.wav'

                        # write audio files
                        sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetWithEmbeddingDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                embedding_key=data_key['embedding_vector'],
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'embedding_key': data_key['embedding_vector'],
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_with_embedding_dataset(config)

            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2:
            # - Test collate_fn
            batch_size = 16
            batch = [dataset.__getitem__(n) for n in range(batch_size)]
            _ = dataset.collate_fn(batch)
