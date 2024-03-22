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

import os

import soundfile as sf
from utils.constants import BLANK_TOKEN, SPACE_TOKEN
from utils.data_prep import Segment, Word
from nemo.collections.asr.parts.utils.manifest_utils import get_ctm_line


def make_ctm_files(
    utt_obj, output_dir_root, ctm_file_config,
):
    """
    Function to save CTM files for all the utterances in the incoming batch.
    """

    # don't try to make files if utt_obj.segments_and_tokens is empty, which will happen
    # in the case of the ground truth text being empty or the number of tokens being too large vs audio duration
    if not utt_obj.segments_and_tokens:
        return utt_obj

    # get audio file duration if we will need it later
    if ctm_file_config.minimum_timestamp_duration > 0:
        with sf.SoundFile(utt_obj.audio_filepath) as f:
            audio_file_duration = f.frames / f.samplerate
    else:
        audio_file_duration = None

    utt_obj = make_ctm("tokens", utt_obj, output_dir_root, audio_file_duration, ctm_file_config,)
    utt_obj = make_ctm("words", utt_obj, output_dir_root, audio_file_duration, ctm_file_config,)
    utt_obj = make_ctm("segments", utt_obj, output_dir_root, audio_file_duration, ctm_file_config,)

    return utt_obj


def make_ctm(
    alignment_level, utt_obj, output_dir_root, audio_file_duration, ctm_file_config,
):
    output_dir = os.path.join(output_dir_root, "ctm", alignment_level)
    os.makedirs(output_dir, exist_ok=True)

    boundary_info_utt = []
    for segment_or_token in utt_obj.segments_and_tokens:
        if type(segment_or_token) is Segment:
            segment = segment_or_token
            if alignment_level == "segments":
                boundary_info_utt.append(segment)

            for word_or_token in segment.words_and_tokens:
                if type(word_or_token) is Word:
                    word = word_or_token
                    if alignment_level == "words":
                        boundary_info_utt.append(word)

                    for token in word.tokens:
                        if alignment_level == "tokens":
                            boundary_info_utt.append(token)

                else:
                    token = word_or_token
                    if alignment_level == "tokens":
                        boundary_info_utt.append(token)

        else:
            token = segment_or_token
            if alignment_level == "tokens":
                boundary_info_utt.append(token)

    with open(os.path.join(output_dir, f"{utt_obj.utt_id}.ctm"), "w") as f_ctm:
        for boundary_info_ in boundary_info_utt:  # loop over every token/word/segment

            # skip if t_start = t_end = negative number because we used it as a marker to skip some blank tokens
            if not (boundary_info_.t_start < 0 or boundary_info_.t_end < 0):
                text = boundary_info_.text
                start_time = boundary_info_.t_start
                end_time = boundary_info_.t_end

                if (
                    ctm_file_config.minimum_timestamp_duration > 0
                    and ctm_file_config.minimum_timestamp_duration > end_time - start_time
                ):
                    # make the predicted duration of the token/word/segment longer, growing it outwards equal
                    # amounts from the predicted center of the token/word/segment
                    token_mid_point = (start_time + end_time) / 2
                    start_time = max(token_mid_point - ctm_file_config.minimum_timestamp_duration / 2, 0.0)
                    end_time = min(
                        token_mid_point + ctm_file_config.minimum_timestamp_duration / 2, audio_file_duration
                    )

                if not (
                    text == BLANK_TOKEN and ctm_file_config.remove_blank_tokens
                ):  # don't save blanks if we don't want to
                    # replace any spaces with <space> so we dont introduce extra space characters to our CTM files
                    text = text.replace(" ", SPACE_TOKEN)

                    ctm_line = get_ctm_line(
                        source=utt_obj.utt_id,
                        channel=1,
                        start_time=start_time,
                        duration=end_time - start_time,
                        token=text,
                        conf=None,
                        type_of_token='lex',
                        speaker=None,
                    )
                    f_ctm.write(ctm_line)

    utt_obj.saved_output_files[f"{alignment_level}_level_ctm_filepath"] = os.path.join(
        output_dir, f"{utt_obj.utt_id}.ctm"
    )

    return utt_obj
