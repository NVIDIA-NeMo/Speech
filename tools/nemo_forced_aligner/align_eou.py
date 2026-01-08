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

import copy
import json
import math
import os
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from string import punctuation
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from utils.data_prep import (
    add_t_start_end_to_utt_obj,
    get_batch_starts_ends,
    get_batch_variables,
    get_manifest_lines_batch,
    is_entry_in_all_lines,
    is_entry_in_any_lines,
)
from utils.make_ass_files import make_ass_files
from utils.make_ctm_files import make_ctm_files
from utils.make_output_manifest import write_manifest_out_line
from utils.viterbi_decoding import viterbi_decoding

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Align the utterances in manifest_filepath. 
Results are saved in ctm files in output_dir as well as json manifest in output_manifest_filepath.
If no output_manifest_filepath is specified, it will save the results in the same parent directory as 
the input manifest_filepath.

Arguments:
    pretrained_name: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded
        from NGC and used for generating the log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
    model_path: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the
        log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
        Note: if a model_path is provided, it will override the pretrained_name.
    manifest_filepath: filepath to the manifest of the data you want to align,
        containing 'audio_filepath' and 'text' fields.
    output_dir: the folder where output CTM files and new JSON manifest will be saved.
    output_manifest_filepath: Optional[str] = None  # output of manfiest with sou_time and eou_time
    manifest_pattern: Optional[str] = None  # pattern used in Path.glob() for finding manifests

    align_using_pred_text: if True, will transcribe the audio using the specified model and then use that transcription 
        as the reference text for the forced alignment. 
    transcribe_device: None, or a string specifying the device that will be used for generating log-probs (i.e. "transcribing").
        The string needs to be in a format recognized by torch.device(). If None, NFA will set it to 'cuda' if it is available 
        (otherwise will set it to 'cpu').
    viterbi_device: None, or string specifying the device that will be used for doing Viterbi decoding. 
        The string needs to be in a format recognized by torch.device(). If None, NFA will set it to 'cuda' if it is available 
        (otherwise will set it to 'cpu').
    batch_size: int specifying batch size that will be used for generating log-probs and doing Viterbi decoding.
    use_local_attention: boolean flag specifying whether to try to use local attention for the ASR Model (will only
        work if the ASR Model is a Conformer model). If local attention is used, we will set the local attention context 
        size to [64,64].
    additional_segment_grouping_separator: an optional string used to separate the text into smaller segments. 
        If this is not specified, then the whole text will be treated as a single segment. 
    remove_blank_tokens_from_ctm:  a boolean denoting whether to remove <blank> tokens from token-level output CTMs. 
    audio_filepath_parts_in_utt_id: int specifying how many of the 'parts' of the audio_filepath
        we will use (starting from the final part of the audio_filepath) to determine the 
        utt_id that will be used in the CTM files. Note also that any spaces that are present in the audio_filepath 
        will be replaced with dashes, so as not to change the number of space-separated elements in the 
        CTM files.
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 1 => utt_id will be "e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 2 => utt_id will be "d_e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 3 => utt_id will be "c_d_e1"
    use_buffered_infer: False, if set True, using streaming to do get the logits for alignment
                        This flag is useful when aligning large audio file.
                        However, currently the chunk streaming inference does not support batch inference,
                        which means even you set batch_size > 1, it will only infer one by one instead of doing
                        the whole batch inference together.
    chunk_len_in_secs: float chunk length in seconds
    total_buffer_in_secs: float  Length of buffer (chunk + left and right padding) in seconds
    chunk_batch_size: int batch size for buffered chunk inference,
                      which will cut one audio into segments and do inference on chunk_batch_size segments at a time

    simulate_cache_aware_streaming: False, if set True, using cache aware streaming to do get the logits for alignment

    save_output_file_formats: List of strings specifying what type of output files to save (default: ["ctm", "ass"])
    ctm_file_config: CTMFileConfig to specify the configuration of the output CTM files
    ass_file_config: ASSFileConfig to specify the configuration of the output ASS files
"""


@dataclass
class CTMFileConfig:
    remove_blank_tokens: bool = False
    # minimum duration (in seconds) for timestamps in the CTM.If any line in the CTM has a
    # duration lower than this, it will be enlarged from the middle outwards until it
    # meets the minimum_timestamp_duration, or reaches the beginning or end of the audio file.
    # Note that this may cause timestamps to overlap.
    minimum_timestamp_duration: float = 0


@dataclass
class ASSFileConfig:
    fontsize: int = 20
    vertical_alignment: str = "center"
    # if resegment_text_to_fill_space is True, the ASS files will use new segments
    # such that each segment will not take up more than (approximately) max_lines_per_segment
    # when the ASS file is applied to a video
    resegment_text_to_fill_space: bool = False
    max_lines_per_segment: int = 2
    text_already_spoken_rgb: List[int] = field(default_factory=lambda: [49, 46, 61])  # dark gray
    text_being_spoken_rgb: List[int] = field(default_factory=lambda: [57, 171, 9])  # dark green
    text_not_yet_spoken_rgb: List[int] = field(default_factory=lambda: [194, 193, 199])  # light gray


@dataclass
class AlignmentConfig:
    # Required configs
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    manifest_filepath: Optional[str] = None  # path to manifest file or directory
    output_dir: Optional[str] = '.tmp'  # set it to .tmp and will be removed after alignment
    output_manifest_filepath: Optional[str] = None  # output of manfiest with sou_time and eou_time
    manifest_pattern: Optional[str] = None  # pattern used in Path.glob() for finding manifests

    # General configs
    align_using_pred_text: bool = False
    transcribe_device: Optional[str] = None
    viterbi_device: Optional[str] = None
    batch_size: int = 1
    use_local_attention: bool = True
    additional_segment_grouping_separator: Optional[str] = None
    audio_filepath_parts_in_utt_id: int = 4

    # Buffered chunked streaming configs
    use_buffered_chunked_streaming: bool = False
    chunk_len_in_secs: float = 1.6
    total_buffer_in_secs: float = 4.0
    chunk_batch_size: int = 32

    # Cache aware streaming configs
    simulate_cache_aware_streaming: Optional[bool] = False

    # Output file configs
    save_output_file_formats: List[str] = field(default_factory=lambda: ["ctm", "ass"])
    ctm_file_config: CTMFileConfig = field(default_factory=lambda: CTMFileConfig())
    ass_file_config: ASSFileConfig = field(default_factory=lambda: ASSFileConfig())

    # remove tmp dir after alignment
    remove_tmp_dir: bool = False
    clean_text: bool = True

    # For multi-node multi-gpu processing
    num_nodes: int = 1  # total num of nodes/machines
    num_gpus: int = 1  # num of GPUs per node/machine
    node_idx: int = 0  # current node index
    gpu_idx: int = 0  # current GPU index


def unicode_to_ascii(text: str) -> str:
    """
    Converts text with accented or special Latin characters (e.g., ó, ñ, ū, ō)
    into their closest ASCII equivalents.
    """
    # Normalize the string to NFKD to separate base characters from diacritics
    normalized = unicodedata.normalize('NFKD', text)

    # Encode to ASCII bytes, ignoring characters that can't be converted
    ascii_bytes = normalized.encode('ascii', 'ignore')

    # Decode back to string
    ascii_text = ascii_bytes.decode('ascii')

    return ascii_text


def drop_pnc(text):
    """
    Clean the text by removing invalid characters and converting to lowercase.

    :param text: Input text.
    :return: Cleaned text.
    """
    valid_chars = "abcdefghijklmnopqrstuvwxyz'"
    text = text.lower()
    text = unicode_to_ascii(text)
    text = text.replace(":", " ")
    text = ''.join([c for c in text if c in valid_chars or c.isspace() or c == "'"])
    return " ".join(text.split()).strip()


def clean_text(manifest: List[dict]):
    punctuations = punctuation.replace("'", "")
    # replace_with_space = [char for char in '/?*\",.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    replace_with_apos = [char for char in '‘’ʻ‘’‘']

    for i in range(len(manifest)):
        text = manifest[i]["text"].strip().lower()  # type: str
        text = text.translate(str.maketrans("", "", punctuations))
        text = drop_pnc(text)
        for c in replace_with_blank:
            text = text.replace(c, "")
        for c in replace_with_apos:
            text = text.replace(c, "'")
        manifest[i]["text"] = text
    return manifest


def get_manifests_for_this_rank(manifest_list, num_nodes, num_gpus, node_idx, gpu_idx):
    """
    Get the manifest files for this rank.
    """
    if len(manifest_list) == 0:
        return manifest_list

    assert num_nodes > 0, "num_nodes must be greater than 0"
    assert num_gpus > 0, "num_gpus must be greater than 0"
    assert 0 <= node_idx < num_nodes, f"node_idx {node_idx} must be between 0 and {num_nodes - 1}"
    assert 0 <= gpu_idx < num_gpus, f"gpu_idx {gpu_idx} must be between 0 and {num_gpus - 1}"

    manifests_this_node = []
    for i, manifest_file in enumerate(manifest_list):
        if num_nodes > 1:
            if i % num_nodes == node_idx:
                manifests_this_node.append(manifest_file)
        else:
            manifests_this_node.append(manifest_file)

    manifests_this_gpu = []
    for i, manifest_file in enumerate(manifests_this_node):
        if num_gpus > 1:
            if i % num_gpus == gpu_idx:
                manifests_this_gpu.append(manifest_file)
        else:
            manifests_this_gpu.append(manifest_file)
    return manifests_this_gpu


@hydra_runner(config_name="AlignmentConfig", schema=AlignmentConfig)
def main(cfg: AlignmentConfig):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    # Validate config
    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None")

    if cfg.model_path is not None and cfg.pretrained_name is not None:
        raise ValueError("One of cfg.model_path and cfg.pretrained_name must be None")

    if cfg.manifest_filepath is None:
        raise ValueError("cfg.manifest_filepath must be specified")

    if cfg.output_dir is None and not cfg.remove_tmp_dir:
        raise ValueError("cfg.output_dir must be specified if cfg.remove_tmp_dir is False")

    if cfg.batch_size < 1:
        raise ValueError("cfg.batch_size cannot be zero or a negative number")

    if cfg.additional_segment_grouping_separator == "" or cfg.additional_segment_grouping_separator == " ":
        raise ValueError("cfg.additional_grouping_separator cannot be empty string or space character")

    if cfg.ctm_file_config.minimum_timestamp_duration < 0:
        raise ValueError("cfg.minimum_timestamp_duration cannot be a negative number")

    if cfg.ass_file_config.vertical_alignment not in ["top", "center", "bottom"]:
        raise ValueError("cfg.ass_file_config.vertical_alignment must be one of 'top', 'center' or 'bottom'")

    for rgb_list in [
        cfg.ass_file_config.text_already_spoken_rgb,
        cfg.ass_file_config.text_already_spoken_rgb,
        cfg.ass_file_config.text_already_spoken_rgb,
    ]:
        if len(rgb_list) != 3:
            raise ValueError(
                "cfg.ass_file_config.text_already_spoken_rgb,"
                " cfg.ass_file_config.text_being_spoken_rgb,"
                " and cfg.ass_file_config.text_already_spoken_rgb all need to contain"
                " exactly 3 elements."
            )

    # init devices
    if cfg.transcribe_device is None:
        transcribe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        transcribe_device = torch.device(cfg.transcribe_device)
    logging.info(f"Device to be used for transcription step (`transcribe_device`) is {transcribe_device}")

    if cfg.viterbi_device is None:
        viterbi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        viterbi_device = torch.device(cfg.viterbi_device)
    logging.info(f"Device to be used for viterbi step (`viterbi_device`) is {viterbi_device}")

    if transcribe_device.type == 'cuda' or viterbi_device.type == 'cuda':
        logging.warning(
            'One or both of transcribe_device and viterbi_device are GPUs. If you run into OOM errors '
            'it may help to change both devices to be the CPU.'
        )

    # load model
    model, _ = setup_model(cfg, transcribe_device)
    model.eval()

    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoder_type="ctc")

    if cfg.use_local_attention:
        logging.info(
            "Flag use_local_attention is set to True => will try to use local attention for model if it allows it"
        )
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])

    if not (isinstance(model, EncDecCTCModel) or isinstance(model, EncDecHybridRNNTCTCModel)):
        raise NotImplementedError(
            f"Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
            " Currently only instances of these models are supported"
        )

    if cfg.ctm_file_config.minimum_timestamp_duration > 0:
        logging.warning(
            f"cfg.ctm_file_config.minimum_timestamp_duration has been set to {cfg.ctm_file_config.minimum_timestamp_duration} seconds. "
            "This may cause the alignments for some tokens/words/additional segments to be overlapping."
        )

    buffered_chunk_params = {}
    if cfg.use_buffered_chunked_streaming:
        model_cfg = copy.deepcopy(model._cfg)

        OmegaConf.set_struct(model_cfg.preprocessor, False)
        # some changes for streaming scenario
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0

        if model_cfg.preprocessor.normalize != "per_feature":
            logging.error(
                "Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently"
            )
        # Disable config overwriting
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        feature_stride = model_cfg.preprocessor['window_stride']
        model_stride_in_secs = feature_stride * cfg.model_downsample_factor
        total_buffer = cfg.total_buffer_in_secs
        chunk_len = float(cfg.chunk_len_in_secs)
        tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
        mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
        logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

        model = FrameBatchASR(
            asr_model=model,
            frame_len=chunk_len,
            total_buffer=cfg.total_buffer_in_secs,
            batch_size=cfg.chunk_batch_size,
        )
        buffered_chunk_params = {
            "delay": mid_delay,
            "model_stride_in_secs": model_stride_in_secs,
            "tokens_per_chunk": tokens_per_chunk,
        }

    if Path(cfg.manifest_filepath).is_file():
        manifest_list = [cfg.manifest_filepath]
    elif Path(cfg.manifest_filepath).is_dir():
        if cfg.manifest_pattern is not None:
            manifest_list = list(Path(cfg.manifest_filepath).glob(cfg.manifest_pattern))
        else:
            manifest_list = list(Path(cfg.manifest_filepath).glob("*.json"))
    else:
        raise ValueError(
            f"cfg.manifest_filepath is not a valid file or directory. "
            f"Please check the path: {cfg.manifest_filepath}"
        )

    origin_output_manifest_filepath = cfg.output_manifest_filepath

    manifest_list = get_manifests_for_this_rank(manifest_list, cfg.num_nodes, cfg.num_gpus, cfg.node_idx, cfg.gpu_idx)
    logging.info(f"Found {len(manifest_list)} manifest files to process.")
    # process each manifest file
    for manifest_filepath in manifest_list:
        logging.info(f"Processing manifest file: {manifest_filepath}")
        cfg.manifest_filepath = str(manifest_filepath)

        if origin_output_manifest_filepath is None:
            manifest_stem = Path(manifest_filepath).stem
            cfg.output_manifest_filepath = str(Path(manifest_filepath).parent / f"{manifest_stem}-aligned.json")
        elif len(manifest_list) > 1 and origin_output_manifest_filepath is not None:
            raise ValueError(
                "cfg.output_manifest_filepath must be None when processing multiple manifest files. "
                "Please set it to None."
            )

        if not cfg.remove_tmp_dir and len(manifest_list) > 1:
            # if keep alignment files, then we need to set output_dir to be different for each manifest
            cfg.output_dir = str(Path(manifest_filepath).parent / f"{Path(manifest_filepath).stem}_alignment")

        process_single_manifest(cfg, model, buffered_chunk_params, viterbi_device)
        logging.info(f"Output manifest saved to: {cfg.output_manifest_filepath}")

    logging.info("All manifest files processed successfully.")


def process_single_manifest(cfg: AlignmentConfig, model, buffered_chunk_params, viterbi_device):
    # Validate manifest contents
    if not is_entry_in_all_lines(cfg.manifest_filepath, "audio_filepath"):
        raise RuntimeError(
            "At least one line in cfg.manifest_filepath does not contain an 'audio_filepath' entry. "
            "All lines must contain an 'audio_filepath' entry."
        )

    if cfg.align_using_pred_text:
        if is_entry_in_any_lines(cfg.manifest_filepath, "pred_text"):
            raise RuntimeError(
                "Cannot specify cfg.align_using_pred_text=True when the manifest at cfg.manifest_filepath "
                "contains 'pred_text' entries. This is because the audio will be transcribed and may produce "
                "a different 'pred_text'. This may cause confusion."
            )
    else:
        if not is_entry_in_all_lines(cfg.manifest_filepath, "text"):
            raise RuntimeError(
                "At least one line in cfg.manifest_filepath does not contain a 'text' entry. "
                "NFA requires all lines to contain a 'text' entry when cfg.align_using_pred_text=False."
            )

    # get start and end line IDs of batches
    starts, ends = get_batch_starts_ends(cfg.manifest_filepath, cfg.batch_size)

    # init output_timestep_duration = None and we will calculate and update it during the first batch
    output_timestep_duration = None

    if cfg.remove_tmp_dir and cfg.output_dir is None:
        cfg.output_dir = f"alignment-{uuid.uuid4()}"

    # init f_manifest_out
    os.makedirs(cfg.output_dir, exist_ok=True)
    tgt_manifest_name = str(Path(cfg.manifest_filepath).stem) + "_with_output_file_paths.json"
    tgt_manifest_filepath = str(Path(cfg.output_dir) / tgt_manifest_name)
    f_manifest_out = open(tgt_manifest_filepath, 'w')

    # get alignment and save in CTM batch-by-batch
    for start, end in zip(starts, ends):
        manifest_lines_batch = get_manifest_lines_batch(cfg.manifest_filepath, start, end)

        if cfg.clean_text:
            manifest_lines_batch = clean_text(manifest_lines_batch)
        (
            log_probs_batch,
            y_batch,
            T_batch,
            U_batch,
            utt_obj_batch,
            output_timestep_duration,
        ) = get_batch_variables(
            manifest_lines_batch,
            model,
            cfg.additional_segment_grouping_separator,
            cfg.align_using_pred_text,
            cfg.audio_filepath_parts_in_utt_id,
            output_timestep_duration,
            cfg.simulate_cache_aware_streaming,
            cfg.use_buffered_chunked_streaming,
            buffered_chunk_params,
        )

        alignments_batch = viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device)

        for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):

            utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration)

            if "ctm" in cfg.save_output_file_formats:
                utt_obj = make_ctm_files(
                    utt_obj,
                    cfg.output_dir,
                    cfg.ctm_file_config,
                )

            if "ass" in cfg.save_output_file_formats:
                utt_obj = make_ass_files(utt_obj, cfg.output_dir, cfg.ass_file_config)

            write_manifest_out_line(
                f_manifest_out,
                utt_obj,
            )

    f_manifest_out.close()

    # adding eou processing here
    input_manifest_lines = [json.loads(line) for line in open(cfg.manifest_filepath)]
    output_manifest_lines = []
    with open(tgt_manifest_filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            item = json.loads(line)
            assert os.path.basename(input_manifest_lines[i]['audio_filepath']) == os.path.basename(
                item['audio_filepath']
            )

            if 'segments_level_ctm_filepath' not in item:
                print(
                    f"`segments_level_ctm_filepath` not found for {input_manifest_lines[i]['audio_filepath']}, skipping"
                )
                continue

            # get sou/eou time
            lines = [line.split() for line in open(item['segments_level_ctm_filepath'])]
            start_time = min([float(line[2]) for line in lines])
            end_time = max([float(line[2]) + float(line[3]) for line in lines])
            input_manifest_lines[i]['sou_time'] = start_time
            input_manifest_lines[i]['eou_time'] = end_time
            output_manifest_lines.append(input_manifest_lines[i])

    with open(cfg.output_manifest_filepath, 'w') as f:
        for item in output_manifest_lines:
            f.write(json.dumps(item) + '\n')

    if cfg.remove_tmp_dir:  # safely removing tmp dir after alignment
        for file_or_folder in [
            tgt_manifest_filepath,
            os.path.join(cfg.output_dir, 'ctm'),
            os.path.join(cfg.output_dir, 'ass'),
        ]:
            if os.path.exists(file_or_folder):
                if os.path.isfile(file_or_folder):
                    os.remove(file_or_folder)
                else:
                    shutil.rmtree(file_or_folder)
        if os.path.exists(cfg.output_dir) and len(os.listdir(cfg.output_dir)) == 0:
            shutil.rmtree(cfg.output_dir)

    return None


if __name__ == "__main__":
    main()
