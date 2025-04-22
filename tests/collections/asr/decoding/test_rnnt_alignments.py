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


from pathlib import Path
from typing import Union

import pytest
import torch.cuda
from examples.asr.transcribe_speech import TranscriptionConfig
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import prepare_audio_data

DEVICES = []

if torch.cuda.is_available():
    DEVICES.append('cuda')


@pytest.fixture(scope="module")
def stt_en_conformer_transducer_small_model():
    model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_small", map_location="cpu")
    return model


def get_rnnt_alignments(
    strategy: str,
    manifest_path: Union[Path, str],
    model: EncDecRNNTBPEModel,
    loop_labels: bool = True,
    use_cuda_graph_decoder=False,
    device="cuda",
) -> list[Hypothesis]:
    cfg = OmegaConf.structured(TranscriptionConfig())
    cfg.rnnt_decoding.confidence_cfg.preserve_frame_confidence = True
    cfg.rnnt_decoding.preserve_alignments = True
    cfg.rnnt_decoding.strategy = strategy
    if cfg.rnnt_decoding.strategy == "greedy_batch":
        cfg.rnnt_decoding.greedy.loop_labels = loop_labels
        cfg.rnnt_decoding.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
    cfg.dataset_manifest = str(manifest_path)
    filepaths = prepare_audio_data(cfg)[0][:8]  # selecting 8 files only
    # NB: 9th file has the same transcription but a bit different alignment for batched/non-batched decoding

    model = model.to(device)
    model.change_decoding_strategy(cfg.rnnt_decoding)

    transcriptions: list[Hypothesis] = model.transcribe(
        audio=filepaths,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        return_hypotheses=True,
        channel_selector=cfg.channel_selector,
    )

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
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loop_labels", [True, False])
@pytest.mark.parametrize("use_cuda_graph_decoder", [True, False])
@pytest.mark.with_downloads
def test_rnnt_alignments(
    loop_labels: bool,
    use_cuda_graph_decoder: bool,
    device: str,
    an4_val_manifest_corrected,
    stt_en_conformer_transducer_small_model,
):
    if use_cuda_graph_decoder and device != "cuda":
        pytest.skip("CUDA decoder works only with CUDA")
    if not loop_labels and use_cuda_graph_decoder:
        pytest.skip("Frame-Looping algorithm with CUDA graphs does not yet support alignments")
    # using greedy as baseline and comparing all other configurations to it
    ref_transcriptions = get_rnnt_alignments(
        "greedy",
        manifest_path=an4_val_manifest_corrected,
        model=stt_en_conformer_transducer_small_model,
        device=device,
    )
    transcriptions = get_rnnt_alignments(
        "greedy_batch",
        loop_labels=loop_labels,
        use_cuda_graph_decoder=use_cuda_graph_decoder,
        manifest_path=an4_val_manifest_corrected,
        model=stt_en_conformer_transducer_small_model,
        device=device,
    )
    # comparing that label sequence in alignments is exactly the same
    # we can't compare logits as well, because they are expected to be
    # slightly different in batched and single-sample mode
    assert len(ref_transcriptions) == len(transcriptions)
    for ref_transcription, transcription in zip(ref_transcriptions, transcriptions):
        for ref_align_elem, align_elem in zip(ref_transcription.alignments, transcription.alignments):
            assert len(ref_align_elem) == len(align_elem)
            for ref_pred, pred in zip(ref_align_elem, align_elem):
                assert ref_pred[1].item() == pred[1].item()
