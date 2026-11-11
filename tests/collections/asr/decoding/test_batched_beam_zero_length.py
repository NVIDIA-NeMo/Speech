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

"""Batched beam search must not decode utterances whose encoder output is empty.

Streaming inference feeds mixed batches in which an utterance that has already been
consumed is passed with ``encoded_lengths == 0`` while the rest of the batch keeps
decoding (see ``examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py``).
Such a row must stay inactive for the whole call.
"""

import pytest
import torch

from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamBatchedRNNTInfer

VOCABULARY = ["a", "b", "c", "d", "e", "f"]
ENCODER_DIM = 8
PRED_HIDDEN = 8


@pytest.fixture(scope="module")
def rnnt_decoder_joint():
    torch.manual_seed(1234)
    decoder = RNNTDecoder({'pred_hidden': PRED_HIDDEN, 'pred_rnn_layers': 1}, len(VOCABULARY) + 1)
    joint = RNNTJoint(
        {
            'encoder_hidden': ENCODER_DIM,
            'pred_hidden': PRED_HIDDEN,
            'joint_hidden': 8,
            'activation': 'relu',
        },
        len(VOCABULARY) + 1,
        vocabulary=VOCABULARY,
    )
    return decoder.eval(), joint.eval()


def _beam_search(decoder, joint, beam_size):
    return BeamBatchedRNNTInfer(
        decoder,
        joint,
        blank_index=len(VOCABULARY),
        beam_size=beam_size,
        search_type="malsd_batch",
        score_norm=True,
        max_symbols_per_step=5,
        allow_cuda_graphs=False,
        return_best_hypothesis=True,
    )


def _decode(decoder, joint, encoder_output, encoded_lengths, beam_size):
    beam_search = _beam_search(decoder, joint, beam_size)
    with torch.no_grad():
        result = beam_search(encoder_output=encoder_output, encoded_lengths=encoded_lengths)
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.unit
@pytest.mark.parametrize("beam_size", [2, 4])
def test_malsd_batch_zero_length_utterance_decodes_nothing(rnnt_decoder_joint, beam_size):
    """A row with ``encoded_lengths == 0`` must produce an empty hypothesis."""
    decoder, joint = rnnt_decoder_joint
    torch.manual_seed(0)
    encoder_output = torch.randn(2, ENCODER_DIM, 9) * 3.0
    encoded_lengths = torch.tensor([9, 0], dtype=torch.int32)

    hypotheses = _decode(decoder, joint, encoder_output, encoded_lengths, beam_size)

    assert len(hypotheses[1].y_sequence) == 0


@pytest.mark.unit
@pytest.mark.parametrize("beam_size", [2, 4])
def test_malsd_batch_finished_stream_gains_no_labels(rnnt_decoder_joint, beam_size):
    """A stream that ends mid-batch must not keep decoding while the batch continues."""
    decoder, joint = rnnt_decoder_joint
    torch.manual_seed(0)
    encoder_output = torch.randn(2, 12, ENCODER_DIM) * 3.0
    chunk_size, finished_after = 4, 1
    computer = _beam_search(decoder, joint, beam_size).decoding_computer

    state, decoded_lengths = None, None
    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, encoder_output.shape[1], chunk_size)):
            length = min(chunk_size, encoder_output.shape[1] - start)
            out_len = torch.tensor([length, 0 if chunk_idx > finished_after else length])
            hyps, state = computer(
                x=encoder_output[:, start : start + chunk_size], out_len=out_len, prev_batched_state=state
            )
            if chunk_idx == finished_after:
                decoded_lengths = hyps.current_lengths_nb[1].clone()
            elif chunk_idx > finished_after:
                assert torch.equal(hyps.current_lengths_nb[1], decoded_lengths)
            hyps.flatten_()
