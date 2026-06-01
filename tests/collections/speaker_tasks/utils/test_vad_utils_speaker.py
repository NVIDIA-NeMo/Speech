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

import numpy as np
import pytest
from lhotse import SupervisionSegment

from nemo.collections.asr.parts.utils.vad_utils import (
    align_labels_to_frames,
    convert_labels_to_speech_segments,
    frame_vad_construct_supervisions_per_file,
    get_frame_labels,
    get_nonspeech_segments,
    load_speech_overlap_segments_from_rttm,
    load_speech_segments_from_rttm,
    read_rttm_as_supervisions,
)


def get_simple_rttm_without_overlap(rttm_file="test1.rttm"):
    line = "SPEAKER <NA> 1 0 2 <NA> <NA> speech <NA> <NA>\n"
    speech_segments = [[0.0, 2.0]]
    with open(rttm_file, "w") as f:
        f.write(line)
    return rttm_file, speech_segments


def get_simple_rttm_with_overlap(rttm_file="test2.rttm"):
    speech_segments = [[0.0, 3.0]]
    overlap_segments = [[1.0, 2.0]]
    with open(rttm_file, "w") as f:
        f.write("SPEAKER <NA> 1 0 2 <NA> <NA> speech <NA> <NA>\n")
        f.write("SPEAKER <NA> 1 1 2 <NA> <NA> speech <NA> <NA>\n")
    return rttm_file, speech_segments, overlap_segments


def get_simple_rttm_with_silence(rttm_file="test3.rttm"):
    line = "SPEAKER <NA> 1 1 2 <NA> <NA> speech <NA> <NA>\n"
    speech_segments = [[1.0, 2.0]]
    silence_segments = [[0.0, 1.0]]
    with open(rttm_file, "w") as f:
        f.write(line)
    return rttm_file, speech_segments, silence_segments


class TestVADUtils:
    @pytest.mark.parametrize(["logits_len", "labels_len"], [(20, 10), (20, 11), (20, 9), (10, 21), (10, 19)])
    @pytest.mark.unit
    def test_align_label_logits(self, logits_len, labels_len):
        logits = np.arange(logits_len).tolist()
        labels = np.arange(labels_len).tolist()
        labels_new = align_labels_to_frames(probs=logits, labels=labels)

        assert len(labels_new) == len(logits)

    @pytest.mark.unit
    def test_load_speech_segments_from_rttm(self, test_data_dir):
        rttm_file, speech_segments = get_simple_rttm_without_overlap(test_data_dir + "/test1.rttm")
        speech_segments_new = load_speech_segments_from_rttm(rttm_file)
        assert speech_segments_new == speech_segments

    @pytest.mark.unit
    def test_load_speech_overlap_segments_from_rttm(self, test_data_dir):
        rttm_file, speech_segments, overlap_segments = get_simple_rttm_with_overlap(test_data_dir + "/test2.rttm")
        speech_segments_new, overlap_segments_new = load_speech_overlap_segments_from_rttm(rttm_file)
        assert speech_segments_new == speech_segments
        assert overlap_segments_new == overlap_segments

    @pytest.mark.unit
    def test_get_nonspeech_segments(self, test_data_dir):
        rttm_file, speech_segments, silence_segments = get_simple_rttm_with_silence(test_data_dir + "/test3.rttm")
        speech_segments_new = load_speech_segments_from_rttm(rttm_file)
        silence_segments_new = get_nonspeech_segments(speech_segments_new)
        assert silence_segments_new == silence_segments

    @pytest.mark.unit
    def test_get_frame_labels(self, test_data_dir):
        rttm_file, speech_segments = get_simple_rttm_without_overlap(test_data_dir + "/test4.rttm")
        speech_segments_new = load_speech_segments_from_rttm(rttm_file)
        frame_labels = get_frame_labels(speech_segments_new, 0.02, 0.0, 3.0, as_str=False)
        assert frame_labels[0] == 1
        assert len(frame_labels) == 150

    @pytest.mark.unit
    def test_convert_labels_to_speech_segments(self, test_data_dir):
        rttm_file, speech_segments = get_simple_rttm_without_overlap(test_data_dir + "/test5.rttm")
        speech_segments_new = load_speech_segments_from_rttm(rttm_file)
        frame_labels = get_frame_labels(speech_segments_new, 0.02, 0.0, 3.0, as_str=False)
        speech_segments_new = convert_labels_to_speech_segments(frame_labels, 0.02)
        assert speech_segments_new == speech_segments

    @pytest.mark.unit
    def test_read_rttm_as_supervisions(self, test_data_dir):
        rttm_file, speech_segments = get_simple_rttm_without_overlap(test_data_dir + "/test6.rttm")
        annotation = read_rttm_as_supervisions(rttm_file)
        assert _annotation_equals(annotation, [(0.0, 2.0, 'speech')])

    @pytest.mark.unit
    def test_frame_vad_construct_supervisions_per_file(self, test_data_dir):
        rttm_file, speech_segments = get_simple_rttm_without_overlap(test_data_dir + "/test7.rttm")
        # test for rttm input
        ref, hyp = frame_vad_construct_supervisions_per_file(rttm_file, rttm_file)
        expected = [(0.0, 2.0, 'speech')]
        assert _annotation_equals(ref, expected)
        assert _annotation_equals(hyp, expected)

        # test for list input
        speech_segments = load_speech_segments_from_rttm(rttm_file)
        frame_labels = get_frame_labels(speech_segments, 0.02, 0.0, 3.0, as_str=False)
        speech_segments_new = convert_labels_to_speech_segments(frame_labels, 0.02)
        assert speech_segments_new == speech_segments
        ref, hyp = frame_vad_construct_supervisions_per_file(frame_labels, frame_labels, 0.02)
        assert _annotation_equals(ref, expected)
        assert _annotation_equals(hyp, expected)


def _annotation_equals(annotation, expected_segments, *, atol=1e-6):
    """Compare a list of :class:`lhotse.SupervisionSegment` to expected ``(start, end, speaker)`` tuples."""
    assert isinstance(annotation, list)
    assert all(isinstance(s, SupervisionSegment) for s in annotation)
    if len(annotation) != len(expected_segments):
        return False
    for seg, (exp_start, exp_end, exp_spk) in zip(annotation, expected_segments):
        if abs(float(seg.start) - exp_start) > atol:
            return False
        if abs(float(seg.end) - exp_end) > atol:
            return False
        if seg.speaker != exp_spk:
            return False
    return True
