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

"""
Script to compute metrics for a given audio-to-audio model for a given manifest file for some dataset.
The manifest file must include path to input audio and path to target (ground truth) audio.

Note: This scripts depends on the `process_audio.py` script, and therefore both scripts should be
located in the same directory during execution.

# Arguments

<< All arguments of `process_audio.py` are inherited by this script, so please refer to `process_audio.py`
for full list of arguments >>

    dataset_manifest: Required - path to dataset JSON manifest file (in NeMo format)
    output_dir: Optional - output directory where the processed audio will be saved
    metrics: Optional - list of metrics to evaluate. Defaults to [sdr,estoi]
    sample_rate: Optional - sample rate for loaded audio. Defaults to 16kHz.
    only_score_manifest: Optional - If set, processing will be skipped and it is assumed the processed audio is available in dataset_manifest

# Usage

## To score a dataset with a manifest file that contains the input audio which needs to be processed and target audio

python audio_to_audio_eval.py \
    model_path=null \
    pretrained_model=null \
    dataset_manifest=<Mandatory: path to a dataset manifest file> \
    output_dir=<Optional: Directory where processed audio will be saved> \
    processed_channel_selector=<Optional: list of channels to select from the processed audio file> \
    target_key=<Optional: key for the target audio in the dataset manifest. Default: target_audio_filepath> \
    target_channel_selector=<Optional: list of channels to select from the target audio file> \
    metrics=<Optional: list of metrics to evaluate. Defaults to [sdr,estoi]>
    batch_size=32 \
    amp=True

## To score a manifest file which has been previously processed and contains both processed audio and target audio

python audio_to_audio_eval.py \
    dataset_manifest=<Mandatory: path to a dataset manifest file> \
    processed_key=<Optional: key for the target audio in the dataset manifest. Default: processed_audio_filepath>
    processed_channel_selector=<Optional: list of channels to select from the processed audio file> \
    target_key=<Optional: key for the target audio in the dataset manifest. Default: target_audio_filepath> \
    target_channel_selector=<Optional: list of channels to select from the target audio file> \
    metrics=<Optional: list of metrics to evaluate. Defaults to [sdr,estoi]>
    batch_size=32 \
    amp=True
"""
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional

import process_audio
import torch
from omegaconf import OmegaConf, open_dict
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

from nemo.collections.asr.data import audio_to_audio_dataset
from nemo.collections.asr.data.audio_to_audio_lhotse import LhotseAudioToTargetDataset
from nemo.collections.asr.metrics.audio import AudioMetricWrapper
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing import manifest
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class AudioEvaluationConfig(process_audio.ProcessConfig):
    # Processed audio config
    processed_channel_selector: Optional[List] = None
    processed_key: str = 'processed_audio_filepath'

    # Target audio configs
    target_dataset_dir: Optional[str] = None  # If not provided, defaults to dirname(cfg.dataset_manifest)
    target_channel_selector: Optional[List] = None
    target_key: str = 'target_audio_filepath'

    # Sample rate for audio evaluation
    sample_rate: int = 16000

    # Score an existing manifest without running processing
    only_score_manifest: bool = False

    # Metrics to calculate
    metrics: List[str] = field(default_factory=lambda: ['sdr', 'estoi'])

    # Return metric values for each example
    return_values_per_example: bool = False


def get_evaluation_dataloader(config):
    """Prepare a dataloader for evaluation.
    """
    if config.get("use_lhotse", False):
        return get_lhotse_dataloader_from_config(
            config, global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
        )

    dataset = audio_to_audio_dataset.get_audio_to_target_dataset(config=config)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset.collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=False,
        num_workers=config.get('num_workers', min(config['batch_size'], os.cpu_count() - 1)),
        pin_memory=True,
    )


def get_metrics(cfg: AudioEvaluationConfig):
    """Prepare a dictionary with metrics.
    """
    available_metrics = ['sdr', 'sisdr', 'stoi', 'estoi', 'pesq']

    metrics = dict()
    for name in sorted(set(cfg.metrics)):
        name = name.lower()
        if name == 'sdr':
            metric = AudioMetricWrapper(metric=SignalDistortionRatio())
        elif name == 'sisdr':
            metric = AudioMetricWrapper(metric=ScaleInvariantSignalDistortionRatio())
        elif name == 'stoi':
            metric = AudioMetricWrapper(metric=ShortTimeObjectiveIntelligibility(fs=cfg.sample_rate, extended=False))
        elif name == 'estoi':
            metric = AudioMetricWrapper(metric=ShortTimeObjectiveIntelligibility(fs=cfg.sample_rate, extended=True))
        elif name == 'pesq':
            metric = AudioMetricWrapper(metric=PerceptualEvaluationSpeechQuality(fs=cfg.sample_rate, mode='wb'))
        else:
            raise ValueError(f'Unexpected metric: {name}. Currently available metrics: {available_metrics}')

        metrics[name] = metric

    return metrics


@hydra_runner(config_name="AudioEvaluationConfig", schema=AudioEvaluationConfig)
def main(cfg: AudioEvaluationConfig):
    torch.set_grad_enabled(False)

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.audio_dir is not None:
        raise RuntimeError(
            "Evaluation script requires ground truth audio to be passed via a manifest file. "
            "If manifest file is available, submit it via `dataset_manifest` argument."
        )

    if not os.path.exists(cfg.dataset_manifest):
        raise FileNotFoundError(f'The dataset manifest file could not be found at path : {cfg.dataset_manifest}')

    if cfg.target_dataset_dir is None:
        # Assume the target data is available in the same directory as the input data
        cfg.target_dataset_dir = os.path.dirname(cfg.dataset_manifest)
    elif not os.path.isdir(cfg.target_dataset_dir):
        raise FileNotFoundError(f'Target dataset dir could not be found at path : {cfg.target_dataset_dir}')

    # Setup metrics
    metrics = get_metrics(cfg)

    if cfg.return_values_per_example and cfg.batch_size > 1:
        raise ValueError('return_example_values is only supported for batch_size=1.')

    # Processing
    if not cfg.only_score_manifest:
        # Process audio using the configured model and save in the output directory
        process_cfg = process_audio.main(cfg)  # type: ProcessConfig

        # Release GPU memory if it was used during transcription
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info('Finished processing audio.')
    else:
        # Score the input manifest, no need to run a model
        cfg.output_filename = cfg.dataset_manifest
        process_cfg = cfg

    # Evaluation
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prepare a temporary manifest with processed audio and target
        temporary_manifest_filepath = os.path.join(tmp_dir, 'manifest.json')

        num_files = 0

        with open(process_cfg.output_filename, 'r') as f_processed, open(
            temporary_manifest_filepath, 'w', encoding='utf-8'
        ) as f_tmp:
            for line_processed in f_processed:
                data_processed = json.loads(line_processed)

                if cfg.processed_key not in data_processed:
                    raise ValueError(
                        f'Processed key {cfg.processed_key} not found in manifest: {process_cfg.output_filename}.'
                    )

                if cfg.target_key not in data_processed:
                    raise ValueError(
                        f'Target key {cfg.target_key} not found in manifest: {process_cfg.output_filename}.'
                    )

                item = {
                    'processed': manifest.get_full_path(
                        audio_file=data_processed[cfg.processed_key], manifest_file=process_cfg.output_filename
                    ),
                    'target': manifest.get_full_path(
                        audio_file=data_processed[cfg.target_key], data_dir=cfg.target_dataset_dir
                    ),
                    'duration': data_processed.get('duration'),
                }

                # Double-check files exist
                for key in ['processed', 'target']:
                    if not os.path.isfile(item[key]):
                        raise ValueError(f'File for key "{key}" not found at: {item[key]}.\nCurrent item: {item}')

                # Warn if we're comparing the same files
                if item['target'] == item['processed']:
                    logging.warning('Using the same file as processed and target: %s', item['target'])

                # Write the entry in the temporary manifest file
                f_tmp.write(json.dumps(item) + '\n')

                num_files += 1

                if cfg.max_utts is not None and num_files >= cfg.max_utts:
                    logging.info('Reached max_utts: %s', cfg.max_utts)
                    break

        # Prepare dataloader
        config = {
            'manifest_filepath': temporary_manifest_filepath,
            'sample_rate': cfg.sample_rate,
            'input_key': 'processed',
            'input_channel_selector': cfg.processed_channel_selector,
            'target_key': 'target',
            'target_channel_selector': cfg.target_channel_selector,
            'batch_size': min(cfg.batch_size, num_files),
            'num_workers': cfg.num_workers,
        }
        temporary_dataloader = get_evaluation_dataloader(config)

        metrics_value_per_example = defaultdict(list)

        # Calculate metrics
        for eval_batch in tqdm(temporary_dataloader, desc='Evaluating'):
            processed_signal, processed_length, target_signal, target_length = eval_batch

            if not torch.equal(processed_length, target_length):
                raise RuntimeError(f'Length mismatch.')

            for name, metric in metrics.items():
                value = metric(preds=processed_signal, target=target_signal, input_length=target_length)
                if cfg.return_values_per_example:
                    metrics_value_per_example[name].append(value.item())

    # Convert to a dictionary with name: value
    metrics_value = {name: metric.compute().item() for name, metric in metrics.items()}

    logging.info('Finished running evaluation.')

    # Show results
    logging.info('Summary\n')
    logging.info('Data')
    logging.info('\tmanifest:           %s', cfg.output_filename)
    logging.info('\ttarget_dataset_dir: %s', cfg.target_dataset_dir)
    logging.info('\tnum_files:          %s', num_files)
    logging.info('Metrics')
    for name, value in metrics_value.items():
        logging.info('\t%10s: \t%6.2f', name, value)

    # Inject the metric name and score into the config, and return the entire config
    with open_dict(cfg):
        cfg.metrics_value = metrics_value
        cfg.metrics_value_per_example = dict(metrics_value_per_example)

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
