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

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import List

from omegaconf import ListConfig
from tqdm import tqdm
from transcribe_speech import TranscriptionConfig as SingleTranscribeConfig
from transcribe_speech import main as single_transcribe_main

from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo formats
  compute_langs: Bool to request language ID information (if the model supports it)
  timestamps: Bool to request greedy time stamp information (if the model supports it) by default None 

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word, segment])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word, segment])

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference
  presort_manifest: sorts the provided manifest by audio length for faster inference (default: True)

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  allow_mps: Bool to allow using MPS (Apple Silicon M-series GPU) device if available
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allows repeated transcriptions to overwrite previous results.

  ctc_decoding: Decoding sub-config for CTC. Refer to documentation for specific values.
  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

  calculate_wer: Bool to decide whether to calculate wer/cer at end of this script
  clean_groundtruth_text: Bool to clean groundtruth text
  langid: Str used for convert_num_to_words during groundtruth cleaning
  use_cer: Bool to use Character Error Rate (CER)  or Word Error Rate (WER)

  calculate_rtfx: Bool to calculate the RTFx throughput to transcribe the input dataset.

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

CUDA_VISIBLE_DEVICES=1 python transcribe_speech_distributed.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    clean_groundtruth_text=True \
    langid='en' \
    batch_size=32 \
    timestamps=False \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix="<remove or use another model name for output filename>" \
    split_size=10000 \
    num_nodes=1 \
    node_idx=0 \
    num_gpus_per_node=1 \
    gpu_idx=0
"""


@dataclass
class ModelChangeConfig:
    """
    Sub-config for changes specific to the Conformer Encoder
    """

    conformer: ConformerChangeConfig = field(default_factory=ConformerChangeConfig)


@dataclass
class TranscriptionConfig(SingleTranscribeConfig):
    """
    Transcription Configuration for audio to text transcription.
    """

    # General configs
    pattern: str = "*.json"
    output_dir: str = "transcribe_output/"

    # Distributed config
    num_nodes: int = 1
    node_idx: int = 0
    num_gpus_per_node: int = 1
    gpu_idx: int = 0
    bind_gpu_to_cuda: bool = False

    # handle long manifest
    split_size: int = -1  # -1 means no split


def get_unfinished_manifest(manifest_list: List[Path], output_dir: Path):
    unfinished = []
    for manifest_file in manifest_list:
        output_manifest_file = output_dir / manifest_file.name
        if not output_manifest_file.exists():
            unfinished.append(manifest_file)
    return sorted(unfinished)


def get_manifest_for_current_rank(
    manifest_list: List[Path], gpu_id: int = 0, num_gpu: int = 1, node_idx: int = 0, num_node: int = 1
):
    node_manifest_list = []
    assert num_node > 0, f"num_node ({num_node}) must be greater than 0"
    assert num_gpu > 0, f"num_gpu ({num_gpu}) must be greater than 0"
    assert 0 <= gpu_id < num_gpu, f"gpu_id ({gpu_id}) must be in range [0, {num_gpu})"
    assert 0 <= node_idx < num_node, f"node_idx ({node_idx}) must be in range [0, {num_node})"
    for i, manifest_file in enumerate(manifest_list):
        if (i + node_idx) % num_node == 0:
            node_manifest_list.append(manifest_file)

    gpu_manifest_list = []
    for i, manifest_file in enumerate(node_manifest_list):
        if (i + gpu_id) % num_gpu == 0:
            gpu_manifest_list.append(manifest_file)
    return gpu_manifest_list


def maybe_split_manifest(manifest_list: List[Path], cfg: TranscriptionConfig) -> List[Path]:
    if cfg.split_size is None or cfg.split_size <= 0:
        return manifest_list

    all_sharded_manifest_files = []
    sharded_manifest_dir = Path(cfg.output_dir) / "sharded_manifest_todo"
    sharded_manifest_dir.mkdir(parents=True, exist_ok=True)

    sharded_manifest_done_dir = Path(cfg.output_dir) / "sharded_manifest_done"
    sharded_manifest_done_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = sharded_manifest_done_dir

    logging.info(f"Splitting {len(manifest_list)} manifest files by every {cfg.split_size} samples.")
    for manifest_file in tqdm(manifest_list, total=len(manifest_list), desc="Splitting manifest files"):
        manifest = read_manifest(manifest_file)

        num_chunks = ceil(len(manifest) / cfg.split_size)
        for i in range(num_chunks):
            chunk_manifest = manifest[i * cfg.split_size : (i + 1) * cfg.split_size]
            sharded_manifest_file = sharded_manifest_dir / f"{manifest_file.stem}--tpart_{i}.json"
            write_manifest(sharded_manifest_file, chunk_manifest)
            all_sharded_manifest_files.append(sharded_manifest_file)

    return all_sharded_manifest_files


def maybe_merge_manifest(cfg: TranscriptionConfig):
    if cfg.split_size is None or cfg.split_size <= 0:
        return

    # only merge manifest on the first GPU of the first node
    if not cfg.gpu_idx == 0 and cfg.node_idx == 0:
        return

    sharded_manifest_dir = Path(cfg.output_dir)
    sharded_manifests = list(sharded_manifest_dir.glob("*--tpart_*.json"))
    if not sharded_manifests:
        logging.info(f"No sharded manifest files found in {sharded_manifest_dir}")
        return

    logging.info(f"Merging {len(sharded_manifests)} sharded manifest files.")
    manifest_dict = defaultdict(list)
    for sharded_manifest in sharded_manifests:
        data_name = sharded_manifest.stem.split("--tpart_")[0]
        manifest_dict[data_name].append(sharded_manifest)

    output_dir = Path(cfg.output_dir).parent
    for data_name, sharded_manifest_list in tqdm(
        manifest_dict.items(), total=len(manifest_dict), desc="Merging manifest files"
    ):
        merged_manifest = []
        for sharded_manifest in sharded_manifest_list:
            manifest = read_manifest(sharded_manifest)
            merged_manifest.extend(manifest)
        output_manifest = output_dir / f"{data_name}.json"
        write_manifest(output_manifest, merged_manifest)
    logging.info(f"Merged manifest files saved to {output_dir}")


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def run_distributed_transcribe(cfg: TranscriptionConfig):

    logging.info(f"Running distributed transcription with config: {cfg}")

    if isinstance(cfg.dataset_manifest, str) and "," in cfg.dataset_manifest:
        manifest_list = cfg.dataset_manifest.split(",")
    elif isinstance(cfg.dataset_manifest, (ListConfig, list)):
        manifest_list = cfg.dataset_manifest
    else:
        input_manifest = Path(cfg.dataset_manifest)
        if input_manifest.is_dir():
            manifest_list = list(input_manifest.glob(cfg.pattern))
        elif input_manifest.is_file():
            manifest_list = [input_manifest]
        else:
            raise ValueError(f"Invalid manifest file or directory: {input_manifest}")

    if not manifest_list:
        raise ValueError(f"No manifest files found matching pattern: {cfg.pattern} in {input_manifest}")

    manifest_list = maybe_split_manifest(manifest_list, cfg)
    original_manifest_list = list(manifest_list)
    logging.info(f"Found {len(manifest_list)} manifest files.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unfinished_manifest = get_unfinished_manifest(manifest_list, output_dir=output_dir)
    if not unfinished_manifest:
        maybe_merge_manifest(cfg)
        logging.info("All manifest files have been processed. Exiting.")
        return
    logging.info(f"Found {len(unfinished_manifest)} unfinished manifest files.")

    manifest_list = get_manifest_for_current_rank(
        unfinished_manifest,
        gpu_id=cfg.gpu_idx,
        num_gpu=cfg.num_gpus_per_node,
        node_idx=cfg.node_idx,
        num_node=cfg.num_nodes,
    )
    if not manifest_list:
        logging.info(f"No manifest files found for GPU {cfg.gpu_idx} on node {cfg.node_idx}. Exiting.")
        return

    logging.info(f"Processing {len(manifest_list)} manifest files with GPU {cfg.gpu_idx} on node {cfg.node_idx}.")

    cfg.cuda = cfg.gpu_idx if cfg.bind_gpu_to_cuda else None
    for manifest_file in tqdm(manifest_list):
        logging.info(f"Processing {manifest_file}...")
        output_filename = output_dir / Path(manifest_file).name
        curr_cfg = deepcopy(cfg)
        curr_cfg.dataset_manifest = str(manifest_file)
        curr_cfg.output_filename = str(output_filename)

        single_transcribe_main(curr_cfg)

    # check if all manifest files have been processed
    unfinished_manifest = get_unfinished_manifest(original_manifest_list, output_dir=output_dir)
    if not unfinished_manifest:
        maybe_merge_manifest(cfg)
        logging.info("All manifest files have been processed. Exiting.")
        return


if __name__ == '__main__':
    run_distributed_transcribe()  # noqa pylint: disable=no-value-for-parameter
