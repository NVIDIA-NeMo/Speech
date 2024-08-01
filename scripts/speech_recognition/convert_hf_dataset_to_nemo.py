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

"""
Python wrapper over HuggingFace Datasets to create preprocessed NeMo ASR Datasets.

List of HuggingFace datasets : https://huggingface.co/datasets
(Please filter by task: automatic-speech-recognition)

# Setup
After installation of huggingface datasets (pip install datasets), some datasets might require authentication
- for example Mozilla Common Voice. You should go to the above link, register as a user and generate an API key.

## Authenticated Setup Steps

Website steps:
- Visit https://huggingface.co/settings/profile
- Visit "Access Tokens" on list of items.
- Create new token - provide a name for the token and "read" access is sufficient.
  - PRESERVE THAT TOKEN API KEY. You can copy that key for next step.
- Visit the HuggingFace Dataset page for Mozilla Common Voice
  - There should be a section that asks you for your approval.
  - Make sure you are logged in and then read that agreement.
  - If and only if you agree to the text, then accept the terms.

Code steps:
- Now on your machine, run `huggingface-cli login`
- Paste your preserved HF TOKEN API KEY (from above).

Now you should be logged in. When running the script, dont forget to set `use_auth_token=True` !

# Usage
The script supports two modes, but the offline mode is the preferred mechanism. The drawback of the offline mode
is that it requires 3 copies of the dataset to exist simultanously -

1) The .arrow files for HF cache
2) The extracted dataset in HF cache
3) The preprocessed audio files preserved in the output_dir provided in the script.

Due to this, make sure your HDD is large enough to store the processed dataset !

## Usage - Offline Mode

python convert_hf_dataset_to_nemo.py \
    output_dir=<Path to some storage drive that will hold preprocessed audio files> \
    path=<`path` argument in HF datasets, cannot be null> \
    name=<`name` argument in HF datasets, can be null> \
    split=<`split` argument in HF datasets, can be null> \
    use_auth_token=<Can be `True` or `False` depending on whether the dataset requires authentication>

This will create an output directory of multiple sub-folders containing the preprocessed .wav files,
along with a nemo compatible JSON manifest file.

NOTE:
    The JSON manifest itself is not preprocessed ! You should perform text normalization, and cleanup
    inconsistent text by using NeMo Text Normalization tool and Speech Data Explorer toolkit !

## Usage - Streaming Mode

NOTE:
    This mode is not well supported. It trades of speed for storage by only having one copy of the dataset in
    output_dir, however the speed of processing is around 10x slower than offline mode. Some datasets (such as MCV)
    fail to run entirely.

    DO NOT USE if you have sufficient disk space.

python convert_hf_dataset_to_nemo.py \
    ... all the arguments from above \
    streaming=True

"""

import json
import os
import traceback
from dataclasses import dataclass, is_dataclass
from typing import Optional

import hydra
import librosa
import soundfile
import tqdm
from datasets import Audio, Dataset, IterableDataset, load_dataset
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class HFDatasetConversionConfig:
    # Nemo Dataset info
    output_dir: str  # path to output directory where the files will be saved

    # HF Dataset info
    path: str  # HF dataset path
    name: Optional[str] = None  # name of the dataset subset
    split: Optional[str] = None  # split of the dataset subset
    use_auth_token: bool = False  # whether authentication token should be passed or not (Required for MCV)

    # NeMo dataset conversion
    sampling_rate: int = 16000
    streaming: bool = False  # Whether to use Streaming dataset API. [NOT RECOMMENDED]
    num_proc: int = -1
    ensure_ascii: bool = True  # When saving the JSON entry, whether to ensure ascii.

    # Placeholders. Generated internally.
    resolved_output_dir: str = ''
    split_output_dir: Optional[str] = None

    hydra: HydraConf = HydraConf(run=RunDir(dir="."))


def prepare_output_dirs(cfg: HFDatasetConversionConfig):
    """
    Prepare output directories and subfolders as needed.
    Also prepare the arguments of the config with these directories.
    """
    output_dir = os.path.abspath(cfg.output_dir)
    output_dir = os.path.join(output_dir, cfg.path)

    if cfg.name is not None:
        output_dir = os.path.join(output_dir, cfg.name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cfg.resolved_output_dir = output_dir
    cfg.split_output_dir = None


def infer_dataset_segments(batch):
    """
    Helper method to run in batch mode over a mapped Dataset.

    Infers the path of the subdirectories for the dataset, removing {extracted/HASH}.

    Returns:
        A cleaned list of path segments
    """
    segments = []
    segment, path = os.path.split(batch['audio']['path'])
    segments.insert(0, path)
    while segment not in ('', os.path.sep):
        segment, path = os.path.split(segment)
        segments.insert(0, path)

    if 'extracted' in segments:
        index_of_basedir = segments.index("extracted")
        segments = segments[(index_of_basedir + 1 + 1) :]  # skip .../extracted/{hash}/

    return segments


def prepare_audio_filepath(audio_filepath):
    """
    Helper method to run in batch mode over a mapped Dataset.

    Prepares the audio filepath and its subdirectories. Remaps the extension to .wav file.

    Args:
        audio_filepath: String path to the audio file.

    Returns:
        Cleaned filepath renamed to be a wav file.
    """
    audio_basefilepath = os.path.split(audio_filepath)[0]
    if not os.path.exists(audio_basefilepath):
        os.makedirs(audio_basefilepath, exist_ok=True)

    # Remove temporary fmt file
    if os.path.exists(audio_filepath):
        os.remove(audio_filepath)

    # replace any ext with .wav
    audio_filepath, ext = os.path.splitext(audio_filepath)
    audio_filepath = audio_filepath + '.wav'

    # Remove previous run file
    if os.path.exists(audio_filepath):
        os.remove(audio_filepath)
    return audio_filepath


def build_map_dataset_to_nemo_func(cfg: HFDatasetConversionConfig, basedir):
    """
    Helper method to run in batch mode over a mapped Dataset.

    Creates a function that can be passed to Dataset.map() containing the config and basedir.
    Useful to map a HF dataset to NeMo compatible format in an efficient way for offline processing.

    Returns:
        A function pointer which can be used for Dataset.map()
    """

    def map_dataset_to_nemo(batch):
        # Write audio file to correct path
        if cfg.streaming:
            batch['audio_filepath'] = batch['audio']['path'].split("::")[0].replace("zip://", "")
        else:
            segments = infer_dataset_segments(batch)
            audio_filepath = os.path.join(*segments)
            batch['audio_filepath'] = audio_filepath

        batch['audio_filepath'] = os.path.abspath(os.path.join(basedir, batch['audio_filepath']))
        audio_filepath = batch['audio_filepath']
        audio_filepath = prepare_audio_filepath(audio_filepath)
        batch['audio_filepath'] = audio_filepath  # update filepath with prepared path

        soundfile.write(audio_filepath, batch['audio']['array'], samplerate=cfg.sampling_rate, format='wav')

        batch['duration'] = librosa.get_duration(y=batch['audio']['array'], sr=batch['audio']['sampling_rate'])
        return batch

    return map_dataset_to_nemo


def convert_offline_dataset_to_nemo(
    dataset: Dataset,
    cfg: HFDatasetConversionConfig,
    basedir: str,
    manifest_filepath: str,
):
    """
    Converts a HF dataset to a audio-preprocessed Nemo dataset in Offline mode.
    Also writes out a nemo compatible manifest file.

    Args:
        dataset: Iterable HF Dataset.
        cfg: HFDatasetConvertionConfig.
        basedir: Base output directory.
        manifest_filepath: Filepath of manifest.
    """
    num_proc = cfg.num_proc
    if num_proc < 0:
        num_proc = max(1, os.cpu_count() // 2)

    dataset = dataset.map(build_map_dataset_to_nemo_func(cfg, basedir), num_proc=num_proc)
    ds_iter = iter(dataset)

    with open(manifest_filepath, 'w') as manifest_f:
        for idx, sample in enumerate(
            tqdm.tqdm(
                ds_iter, desc=f'Processing {cfg.path} (split : {cfg.split}):', total=len(dataset), unit=' samples'
            )
        ):
            # remove large components from sample
            del sample['audio']
            if 'file' in sample:
                del sample['file']
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=cfg.ensure_ascii)}\n")


def convert_streaming_dataset_to_nemo(
    dataset: IterableDataset, cfg: HFDatasetConversionConfig, basedir: str, manifest_filepath: str
):
    """
    Converts a HF dataset to a audio-preprocessed Nemo dataset in Streaming mode.
    Also writes out a nemo compatible manifest file.

    Args:
        dataset: Iterable HF Dataset.
        cfg: HFDatasetConvertionConfig.
        basedir: Base output directory.
        manifest_filepath: Filepath of manifest.
    """
    # Disable until fix https://github.com/huggingface/datasets/pull/3556 is merged
    # dataset = dataset.map(build_map_dataset_to_nemo_func(cfg, basedir))

    ds_iter = iter(dataset)

    with open(manifest_filepath, 'w') as manifest_f:
        for idx, sample in enumerate(
            tqdm.tqdm(ds_iter, desc=f'Processing {cfg.path} (split: {cfg.split}):', unit=' samples')
        ):

            audio_filepath = sample['audio']['path'].split("::")[0].replace("zip://", "")
            audio_filepath = os.path.abspath(os.path.join(basedir, audio_filepath))
            audio_filepath = prepare_audio_filepath(audio_filepath)

            soundfile.write(audio_filepath, sample['audio']['array'], samplerate=cfg.sampling_rate, format='wav')

            manifest_line = {
                'audio_filepath': audio_filepath,
                'text': sample['text'],
                'duration': librosa.get_duration(sample['audio']['array'], sr=cfg.sampling_rate),
            }

            # remove large components from sample
            del sample['audio']
            del sample['text']
            if 'file' in sample:
                del sample['file']

            manifest_line.update(sample)

            manifest_f.write(f"{json.dumps(sample, ensure_ascii=cfg.ensure_ascii)}\n")


def process_dataset(dataset: IterableDataset, cfg: HFDatasetConversionConfig):
    """
    Top level method that processes a given IterableDataset to Nemo compatible dataset.
    It also writes out a nemo compatible manifest file.

    Args:
        dataset: HF Dataset.
        cfg: HFDatasetConvertionConfig
    """
    dataset = dataset.cast_column("audio", Audio(cfg.sampling_rate, mono=True))

    # for Common Voice, "sentence" is used instead of "text" to store the transcript.
    if 'sentence' in dataset.features:
        dataset = dataset.rename_column("sentence", "text")

    if cfg.split_output_dir is None:
        basedir = cfg.resolved_output_dir
        manifest_filename = f"{cfg.path.replace('/', '_')}_manifest.json"
    else:
        basedir = cfg.split_output_dir
        split = os.path.split(cfg.split_output_dir)[-1]
        manifest_filename = f"{split}_{cfg.path.replace('/', '_')}_manifest.json"

        if not os.path.exists(cfg.split_output_dir):
            os.makedirs(cfg.split_output_dir, exist_ok=True)

        cfg.split = split

    manifest_filepath = os.path.abspath(os.path.join(basedir, manifest_filename))

    if cfg.streaming:
        convert_streaming_dataset_to_nemo(dataset, cfg, basedir=basedir, manifest_filepath=manifest_filepath)
    else:
        convert_offline_dataset_to_nemo(dataset, cfg, basedir=basedir, manifest_filepath=manifest_filepath)

    print()
    print("Dataset conversion finished !")


@hydra.main(config_name='hfds_config', config_path=None)
def main(cfg: HFDatasetConversionConfig):
    # Convert dataclass to omegaconf
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    # Prepare output subdirs
    prepare_output_dirs(cfg)

    # Load dataset in offline/streaming mode
    dataset = None
    try:
        dataset = load_dataset(
            path=cfg.path,
            name=cfg.name,
            split=cfg.split,
            cache_dir=None,
            streaming=cfg.streaming,
            token=cfg.use_auth_token,
            trust_remote_code=True,
        )

    except Exception as e:
        print(
            "HuggingFace datasets failed due to some reason (stack trace below). \nFor certain datasets (eg: MCV), "
            "it may be necessary to login to the huggingface-cli (via `huggingface-cli login`).\n"
            "Once logged in, you need to set `use_auth_token=True` when calling this script.\n\n"
            "Traceback error for reference :\n"
        )
        print(traceback.format_exc())
        exit(1)

    # Multiple datasets were provided at once, process them one by one into subdirs.
    if isinstance(dataset, dict):
        print()
        print("Multiple splits found for dataset", cfg.path, ":", list(dataset.keys()))

        keys = list(dataset.keys())
        for key in keys:
            ds_split = dataset[key]
            print(f"Processing split {key} for dataset {cfg.path}")

            cfg.split_output_dir = os.path.join(cfg.resolved_output_dir, key)
            process_dataset(ds_split, cfg)

            del dataset[key], ds_split

        # reset the split output directory
        cfg.split_output_dir = None

    else:
        # Single dataset was found, process into resolved directory.
        print("Single split found for dataset", cfg.path, "| Split chosen =", cfg.split)

        if cfg.split is not None:
            cfg.split_output_dir = os.path.join(cfg.resolved_output_dir, cfg.split)

        process_dataset(dataset, cfg)


# Register the dataclass as a valid config
ConfigStore.instance().store(name='hfds_config', node=HFDatasetConversionConfig)

if __name__ == '__main__':
    main()
