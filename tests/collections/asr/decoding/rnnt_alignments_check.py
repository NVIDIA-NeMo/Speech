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


# NOTE: the file name does not contain "test" on purpose to avoid executing
#       these tests outside of the CI machines environment, where test data is
#       stored

import os
import pytest
from examples.asr.transcribe_speech import TranscriptionConfig
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.transcribe_utils import prepare_audio_data, setup_model

TEST_DATA_PATH = "/home/TestData/an4_dataset/an4_val.json"
PRETRAINED_MODEL_NAME = "stt_en_conformer_transducer_small"


def get_rnnt_alignments(strategy: str):
    cfg = OmegaConf.structured(TranscriptionConfig(pretrained_name=PRETRAINED_MODEL_NAME))
    cfg.rnnt_decoding.confidence_cfg.preserve_frame_confidence = True
    cfg.rnnt_decoding.preserve_alignments = True
    cfg.rnnt_decoding.strategy = strategy
    cfg.dataset_manifest = TEST_DATA_PATH
    filepaths = prepare_audio_data(cfg)[0][:10]  # selecting 10 files only

    model = setup_model(cfg, map_location="cuda")[0]
    model.change_decoding_strategy(cfg.rnnt_decoding)

    transcriptions = model.transcribe(
        paths2audio_files=filepaths,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        return_hypotheses=True,
        channel_selector=cfg.channel_selector,
    )[0]

    for transcription in transcriptions:
        for align_elem, frame_confidence in zip(transcription.alignments, transcription.frame_confidence):
            assert len(align_elem) == len(frame_confidence)  # frame confidences have to match alignments
            assert len(align_elem) > 0  # no empty alignments
            for idx, pred in enumerate(align_elem):
                if idx < len(align_elem) - 1:
                    assert pred[1].item() != model.decoder.blank_idx  # all except last have to be non-blank
                else:
                    assert pred[1].item() == model.decoder.blank_idx  # last one has to be blank
    return transcriptions


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    """Overriding global fixture to make sure it's not applied for this test.

    Otherwise, there will be errors in the CI in github.
    """
    return


# TODO: add the same tests for multi-blank RNNT decoding
@pytest.mark.skipif(not os.path.exists('/home/TestData'), reason='Not a Jenkins machine')
def test_rnnt_alignments():
    # using greedy as baseline and comparing all other configurations to it
    ref_transcriptions = get_rnnt_alignments("greedy")
    transcriptions = get_rnnt_alignments("greedy_batch")
    # comparing that label sequence in alignments is exactly the same
    # we can't compare logits as well, because they are expected to be
    # slightly different in batched and single-sample mode
    assert len(ref_transcriptions) == len(transcriptions)
    for ref_transcription, transcription in zip(ref_transcriptions, transcriptions):
        for ref_align_elem, align_elem in zip(ref_transcription.alignments, transcription.alignments):
            assert len(ref_align_elem) == len(align_elem)
            for ref_pred, pred in zip(ref_align_elem, align_elem):
                assert ref_pred[1].item() == pred[1].item()
