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

import json
import os
from collections import Counter
from collections import OrderedDict as od
from pathlib import Path
from typing import Dict, List, Union

import librosa
import numpy as np

from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_subsegments,
    get_uniqname_from_filepath,
    rttm_to_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
)
from nemo.utils import logging
from nemo.utils.data_utils import DataStoreObject


def get_rounded_str_float(num: float, output_precision: int, min_precision=1, max_precision=3) -> str:
    """
    Get a string of a float number with rounded precision.

    Args:
        num (float): float number to round
        output_precision (int): precision of the output floating point number
        min_precision (int, optional): Minimum precision of the output floating point number. Defaults to 1.
        max_precision (int, optional): Maximum precision of the output floating point number. Defaults to 3.

    Returns:
        (str): Return a string of a float number with rounded precision.
    """
    output_precision = min(max_precision, max(min_precision, output_precision))
    return f"{num:.{output_precision}f}"


def get_ctm_line(
    source: str,
    channel: int,
    start_time: float,
    duration: float,
    token: str,
    conf: float,
    type_of_token: str,
    speaker: str,
    NA_token: str = 'NA',
    UNK: str = 'unknown',
    default_channel: str = '1',
    output_precision: int = 2,
) -> str:
    """
    Get a line in Conversation Time Mark (CTM) format. Following CTM format appeared in `Rich Transcription Meeting Eval Plan: RT09` document.
    
    CTM Format: 
        <SOURCE><SP><CHANNEL><SP><BEG-TIME><SP><DURATION><SP><TOKEN><SP><CONF><SP><TYPE><SP><SPEAKER><NEWLINE>
    
    Reference: 
        https://web.archive.org/web/20170119114252/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    Args:
        source (str): <SOURCE> is name of the source file, session name or utterance ID
        channel (int): <CHANNEL> is channel number defaults to 1
        start_time (float): <BEG_TIME> is the begin time of the word, which we refer to as `start_time` in NeMo.
        duration (float): <DURATION> is duration of the word
        token (str): <TOKEN> Token or word for the current entry
        conf (float): <CONF> is a floating point number between 0 (no confidence) and 1 (certainty). A value of “NA” is used (in CTM format data) 
                      when no confidence is computed and in the reference data. 
        type_of_token (str): <TYPE> is the token type. The legal values of <TYPE> are “lex”, “frag”, “fp”, “un-lex”, “for-lex”, “non-lex”, “misc”, or “noscore”
        speaker (str): <SPEAKER> is a string identifier for the speaker who uttered the token. This should be “null” for non-speech tokens and “unknown” when
                       the speaker has not been determined. 
        NA_token (str, optional): A token for  . Defaults to '<NA>'.
        output_precision (int, optional): The precision of the output floating point number. Defaults to 3.

    Returns:
        str: Return a line in CTM format filled with the given information.
    """
    VALID_TOKEN_TYPES = ["lex", "frag", "fp", "un-lex", "for-lex", "non-lex", "misc", "noscore"]

    if type(start_time) == str and start_time.replace('.', '', 1).isdigit():
        start_time = float(start_time)
    elif type(start_time) != float:
        raise ValueError(f"`start_time` must be a float or str containing float, but got {type(start_time)}")

    if type(duration) == str and duration.replace('.', '', 1).isdigit():
        duration = float(duration)
    elif type(duration) != float:
        raise ValueError(f"`duration` must be a float or str containing float, but got {type(duration)}")

    if type(conf) == str and conf.replace('.', '', 1).isdigit():
        conf = float(conf)
    elif conf is None:
        conf = NA_token
    elif type(conf) != float:
        raise ValueError(f"`conf` must be a float or str containing float, but got {type(conf)}")

    if channel is not None and type(channel) != int:
        channel = str(channel)
    if conf is not None and type(conf) == float and not (0 <= conf <= 1):
        raise ValueError(f"`conf` must be between 0 and 1, but got {conf}")
    if type_of_token is not None and type(type_of_token) != str:
        raise ValueError(f"`type` must be a string, but got {type(type_of_token)} type {type_of_token}")
    if type_of_token is not None and type_of_token not in VALID_TOKEN_TYPES:
        raise ValueError(f"`type` must be one of {VALID_TOKEN_TYPES}, but got {type_of_token} type {type_of_token}")
    if speaker is not None and type(speaker) != str:
        raise ValueError(f"`speaker` must be a string, but got {type(speaker)}")

    channel = default_channel if channel is None else channel
    conf = NA_token if conf is None else conf
    speaker = NA_token if speaker is None else speaker
    type_of_token = UNK if type_of_token is None else type_of_token
    start_time = get_rounded_str_float(start_time, output_precision)
    duration = get_rounded_str_float(duration, output_precision)
    conf = get_rounded_str_float(conf, output_precision) if conf != NA_token else conf
    return f"{source} {channel} {start_time} {duration} {token} {conf} {type_of_token} {speaker}\n"


def rreplace(s: str, old: str, new: str) -> str:
    """
    Replace end of string.

    Args:
        s (str): string to operate on
        old (str): ending of string to replace
        new (str): replacement for ending of string
    Returns:
        new.join(li) (string): new string with end replaced
    """
    li = s.rsplit(old, 1)
    return new.join(li)


def get_uniq_id_with_period(path: str) -> str:
    """
    Get uniq_id from path string with period in it.

    Args:
        path (str): path to audio file
    Returns:
        uniq_id (str): unique speaker ID
    """
    split_path = os.path.basename(path).split('.')[:-1]
    uniq_id = '.'.join(split_path) if len(split_path) > 1 else split_path[0]
    return uniq_id


def get_subsegment_dict(subsegments_manifest_file: str, window: float, shift: float, deci: int) -> Dict[str, dict]:
    """
    Get subsegment dictionary from manifest file.

    Args:
        subsegments_manifest_file (str): Path to subsegment manifest file
        window (float): Window length for segmentation
        shift (float): Shift length for segmentation
        deci (int): Rounding number of decimal places
    Returns:
        _subsegment_dict (dict): Subsegment dictionary
    """
    _subsegment_dict = {}
    with open(subsegments_manifest_file, 'r') as subsegments_manifest:
        segments = subsegments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            if dic['uniq_id'] is not None:
                uniq_id = dic['uniq_id']
            else:
                uniq_id = get_uniq_id_with_period(audio)
            if uniq_id not in _subsegment_dict:
                _subsegment_dict[uniq_id] = {'ts': [], 'json_dic': []}
            for subsegment in subsegments:
                start, dur = subsegment
            _subsegment_dict[uniq_id]['ts'].append([round(start, deci), round(start + dur, deci)])
            _subsegment_dict[uniq_id]['json_dic'].append(dic)
    return _subsegment_dict


def get_input_manifest_dict(input_manifest_path: str) -> Dict[str, dict]:
    """
    Get dictionary from manifest file.

    Args:
        input_manifest_path (str): Path to manifest file
    Returns:
        input_manifest_dict (dict): Dictionary from manifest file
    """
    input_manifest_dict = {}
    with open(input_manifest_path, 'r') as input_manifest_fp:
        json_lines = input_manifest_fp.readlines()
        for json_line in json_lines:
            dic = json.loads(json_line)
            dic["text"] = "-"
            uniq_id = get_uniqname_from_filepath(dic["audio_filepath"])
            input_manifest_dict[uniq_id] = dic
    return input_manifest_dict


def write_truncated_subsegments(
    input_manifest_dict: Dict[str, dict],
    _subsegment_dict: Dict[str, dict],
    output_manifest_path: str,
    step_count: int,
    deci: int,
):
    """
    Write subsegments to manifest filepath.

    Args:
        input_manifest_dict (dict): Input manifest dictionary
        _subsegment_dict (dict): Input subsegment dictionary
        output_manifest_path (str): Path to output manifest file
        step_count (int): Number of the unit segments you want to create per utterance
        deci (int): Rounding number of decimal places
    """
    with open(output_manifest_path, 'w') as output_manifest_fp:
        for uniq_id, subseg_dict in _subsegment_dict.items():
            subseg_array = np.array(subseg_dict['ts'])
            subseg_array_idx = np.argsort(subseg_array, axis=0)
            chunked_set_count = subseg_array_idx.shape[0] // step_count

            for idx in range(chunked_set_count - 1):
                chunk_index_stt = subseg_array_idx[:, 0][idx * step_count]
                chunk_index_end = subseg_array_idx[:, 1][(idx + 1) * step_count]
                offset_sec = subseg_array[chunk_index_stt, 0]
                end_sec = subseg_array[chunk_index_end, 1]
                dur = round(end_sec - offset_sec, deci)
                meta = input_manifest_dict[uniq_id]
                meta['offset'] = offset_sec
                meta['duration'] = dur
                json.dump(meta, output_manifest_fp)
                output_manifest_fp.write("\n")


def write_file(name: str, lines: List[dict], idx: int):
    """
    Write json lines to file.

    Args:
        name (str): Output file path
        lines (list): List of json lines
        idx (int): Indices to dump to the file
    """
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')


def read_file(pathlist: str) -> List[str]:
    """
    Read list of lines from target file.

    Args:
        pathlist (str): Input file path
    Returns:
        sorted(pathlist) (list): List of lines
    """
    with open(pathlist, 'r') as f:
        pathlist = f.readlines()
    return sorted(pathlist)


def get_dict_from_wavlist(pathlist: List[str]) -> Dict[str, str]:
    """
    Read dictionaries from list of lines

    Args:
        pathlist (list): List of file paths
    Returns:
        path_dict (dict): Dictionary containing dictionaries read from files
    """
    path_dict = od()
    pathlist = sorted(pathlist)
    for line_path in pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        path_dict[uniq_id] = line_path
    return path_dict


def get_dict_from_list(data_pathlist: List[str], uniqids: List[str]) -> Dict[str, str]:
    """
    Create dictionaries from list of lines

    Args:
        data_pathlist (list): List of file paths
        uniqids (list): List of file IDs
    Returns:
        path_dict (dict): Dictionary containing file paths
    """
    path_dict = {}
    for line_path in data_pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        if uniq_id in uniqids:
            path_dict[uniq_id] = line_path
        else:
            raise ValueError(f'uniq id {uniq_id} is not in wav filelist')
    return path_dict


def get_path_dict(data_path: str, uniqids: List[str], len_wavs: int = None) -> Dict[str, str]:
    """
    Create dictionary from list of lines (using the get_dict_from_list function)

    Args:
        data_path (str): Path to file containing list of files
        uniqids (list): List of file IDs
        len_wavs (int): Length of file list
    Returns:
        data_pathdict (dict): Dictionary containing file paths
    """
    if data_path is not None:
        data_pathlist = read_file(data_path)
        if len_wavs is not None:
            assert len(data_pathlist) == len_wavs
            data_pathdict = get_dict_from_list(data_pathlist, uniqids)
    elif len_wavs is not None:
        data_pathdict = {uniq_id: None for uniq_id in uniqids}
    return data_pathdict


def create_segment_manifest(
    input_manifest_path: str, output_manifest_path: str, window: float, shift: float, step_count: int, deci: int
):
    """
    Create segmented manifest file from base manifest file

    Args:
        input_manifest_path (str): Path to input manifest file
        output_manifest_path (str): Path to output manifest file
        window (float): Window length for segmentation
        shift (float): Shift length for segmentation
        step_count (int): Number of the unit segments you want to create per utterance
        deci (int): Rounding number of decimal places
    """
    if '.json' not in input_manifest_path:
        raise ValueError("input_manifest_path file should be .json file format")
    if output_manifest_path and '.json' not in output_manifest_path:
        raise ValueError("output_manifest_path file should be .json file format")
    elif not output_manifest_path:
        output_manifest_path = rreplace(input_manifest_path, '.json', f'_{step_count}seg.json')

    input_manifest_dict = get_input_manifest_dict(input_manifest_path)
    segment_manifest_path = rreplace(input_manifest_path, '.json', '_seg.json')
    subsegment_manifest_path = rreplace(input_manifest_path, '.json', '_subseg.json')
    min_subsegment_duration = 0.05
    step_count = int(step_count)

    AUDIO_RTTM_MAP = audio_rttm_map(input_manifest_path)
    segments_manifest_file = write_rttm2manifest(AUDIO_RTTM_MAP, segment_manifest_path, deci)
    subsegments_manifest_file = subsegment_manifest_path
    segments_manifest_to_subsegments_manifest(
        segments_manifest_file, subsegments_manifest_file, window, shift, min_subsegment_duration,
    )
    subsegments_dict = get_subsegment_dict(subsegments_manifest_file, window, shift, deci)
    write_truncated_subsegments(input_manifest_dict, subsegments_dict, output_manifest_path, step_count, deci)
    os.remove(segment_manifest_path)
    os.remove(subsegment_manifest_path)


def create_manifest(
    wav_path: str,
    manifest_filepath: str,
    text_path: str = None,
    rttm_path: str = None,
    uem_path: str = None,
    ctm_path: str = None,
    add_duration: bool = False,
):
    """
    Create base manifest file

    Args:
        wav_path (str): Path to list of wav files
        manifest_filepath (str): Path to output manifest file
        text_path (str): Path to list of text files
        rttm_path (str): Path to list of rttm files
        uem_path (str): Path to list of uem files
        ctm_path (str): Path to list of ctm files
        add_duration (bool): Whether to add durations to the manifest file
    """
    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)
    wav_pathlist = read_file(wav_path)
    wav_pathdict = get_dict_from_wavlist(wav_pathlist)
    len_wavs = len(wav_pathlist)
    uniqids = sorted(wav_pathdict.keys())

    text_pathdict = get_path_dict(text_path, uniqids, len_wavs)
    rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs)
    uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs)
    ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs)

    lines = []
    for uid in uniqids:
        wav, text, rttm, uem, ctm = (
            wav_pathdict[uid],
            text_pathdict[uid],
            rttm_pathdict[uid],
            uem_pathdict[uid],
            ctm_pathdict[uid],
        )

        audio_line = wav.strip()
        if rttm is not None:
            rttm = rttm.strip()
            labels = rttm_to_labels(rttm)
            num_speakers = Counter([l.split()[-1] for l in labels]).keys().__len__()
        else:
            num_speakers = None

        if uem is not None:
            uem = uem.strip()

        if text is not None:
            with open(text.strip()) as f:
                text = f.readlines()[0].strip()
        else:
            text = "-"

        if ctm is not None:
            ctm = ctm.strip()

        duration = None
        if add_duration:
            y, sr = librosa.load(audio_line, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
        meta = [
            {
                "audio_filepath": audio_line,
                "offset": 0,
                "duration": duration,
                "label": "infer",
                "text": text,
                "num_speakers": num_speakers,
                "rttm_filepath": rttm,
                "uem_filepath": uem,
                "ctm_filepath": ctm,
            }
        ]
        lines.extend(meta)

    write_file(manifest_filepath, lines, range(len(lines)))


def read_manifest(manifest: Union[Path, str]) -> List[dict]:
    """
    Read manifest file

    Args:
        manifest (str or Path): Path to manifest file
    Returns:
        data (list): List of JSON items
    """
    manifest = DataStoreObject(str(manifest))

    data = []
    try:
        f = open(manifest.get(), 'r', encoding='utf-8')
    except:
        raise Exception(f"Manifest file could not be opened: {manifest}")

    errors = []
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            errors.append(line)
            continue
        data.append(item)
    f.close()
    if errors:
        logging.error(f"{len(errors)} Errors encountered while reading manifest file: {manifest}")
        for error in errors:
            logging.error(f"-- Failed to parse line: `{error}`")
        raise RuntimeError(f"Errors encountered while reading manifest file: {manifest}")
    return data


def write_manifest(output_path: Union[Path, str], target_manifest: List[dict], ensure_ascii: bool = True):
    """
    Write to manifest file

    Args:
        output_path (str or Path): Path to output manifest file
        target_manifest (list): List of manifest file entries
        ensure_ascii (bool): default is True, meaning the output is guaranteed to have all incoming non-ASCII characters escaped. If ensure_ascii is false, these characters will be output as-is.
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')


def write_ctm(output_path: str, target_ctm: Dict[str, dict]):
    """
    Write ctm entries from diarization session to a .ctm file.

    Args:
        output_path (str): target file path
        target_ctm (dict): list of ctm entries
    """
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            outfile.write(tgt)


def write_text(output_path: str, target_ctm: Dict[str, dict]):
    """
    Write text from diarization session to a .txt file

    Args:
        output_path (str): target file path
        target_ctm (dict): list of ctm entries
    """
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            word = tgt.split(' ')[4]
            outfile.write(word + ' ')
        outfile.write('\n')
