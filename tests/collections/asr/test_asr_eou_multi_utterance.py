# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
Multi-utterance decoding with ASR-EOU models (models with ``<EOU>``/``<EOB>`` tokens).

These models are trained on single-utterance samples and stop emitting tokens after the first ``<EOU>`` unless
decoding restarts after it. The tests use a tiny randomly-initialized model whose decoding algorithm is replaced
with a simulation of this behavior: it emits ``<word> <EOU>`` for the next utterance and nothing after that.
"""

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.eou_utils import (
    get_utterance_end_token_ids,
    reset_hypothesis_decoder_state_,
    truncate_hypothesis_after_first_utterance_end_,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model

WORD_ID = 5
SUBSAMPLING_FACTOR = 4


@pytest.fixture(scope="module")
def eou_tokenizer_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("eou_tokenizer")
    text_file = tmp_path / "text.txt"
    text_file.write_text("\n".join(["go no stop yes one two three four five six seven eight nine ten"] * 10))
    create_spt_model(
        str(text_file),
        vocab_size=40,
        sample_size=-1,
        do_lower_case=True,
        tokenizer_type="bpe",
        output_dir=str(tmp_path),
        user_defined_symbols=["<EOU>", "<EOB>"],
    )
    return str(tmp_path)


def _make_model(tokenizer_dir: str) -> EncDecRNNTBPEModel:
    cfg = DictConfig(
        {
            'sample_rate': 16000,
            'model_defaults': {'enc_hidden': 16, 'pred_hidden': 16},
            'preprocessor': {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'sample_rate': 16000,
                'features': 64,
                'normalize': 'NA',
                'dither': 0.0,
                'pad_to': 0,
            },
            'encoder': {
                '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                'feat_in': 64,
                'n_layers': 1,
                'd_model': 16,
                'subsampling': 'striding',
                'subsampling_factor': SUBSAMPLING_FACTOR,
                'subsampling_conv_channels': 8,
                'causal_downsampling': True,
                'ff_expansion_factor': 2,
                'self_attention_model': 'rel_pos',
                'n_heads': 2,
                'att_context_size': [16, 1],
                'att_context_style': 'chunked_limited',
                'conv_kernel_size': 9,
                'conv_context_size': 'causal',
                'dropout': 0.0,
                'dropout_pre_encoder': 0.0,
                'dropout_emb': 0.0,
                'dropout_att': 0.0,
            },
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                'prednet': {'pred_hidden': 16, 'pred_rnn_layers': 1},
            },
            'joint': {
                '_target_': 'nemo.collections.asr.modules.RNNTJoint',
                'jointnet': {'joint_hidden': 16, 'activation': 'relu', 'encoder_hidden': 16, 'pred_hidden': 16},
            },
            'tokenizer': {'dir': tokenizer_dir, 'type': 'bpe'},
            'decoding': {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}},
            'loss': {'loss_name': 'default'},
        }
    )
    model = EncDecRNNTBPEModel(cfg=cfg)
    model.eval()
    return model


@pytest.fixture()
def eou_model(eou_tokenizer_dir):
    return _make_model(eou_tokenizer_dir)


class _SingleUtteranceDecodingSim:
    """
    Simulates the decoding of an ASR-EOU model trained on single-utterance samples.

    The utterances start at given (absolute) encoder frames. A hypothesis that has not ended yet emits
    ``<word> <EOU>`` at the next utterance frame; after ``<EOU>``,
    it emits nothing (unless its decoder state is reset).
    """

    def __init__(self, utterance_frames: list[list[int]], eou_id: int, total_frames: list[int] | None = None):
        self.utterance_frames = utterance_frames
        self.eou_id = eou_id
        self.total_frames = total_frames  # offline: segments are suffixes of the full audio
        self.consumed_frames = [0] * len(utterance_frames)  # streaming: frames decoded in previous steps

    def __call__(self, encoder_output, encoded_lengths, partial_hypotheses=None):
        hypotheses = []
        for b, num_frames in enumerate(encoded_lengths.tolist()):
            if self.total_frames is not None:
                offset = self.total_frames[b] - num_frames
            else:
                offset = self.consumed_frames[b]
                self.consumed_frames[b] += num_frames
            prev = partial_hypotheses[b] if partial_hypotheses is not None else None
            ended = prev is not None and prev.dec_state == "ended"
            tokens, timestamps, dec_state = [], [], "ended" if ended else None
            if not ended:
                for u in self.utterance_frames[b]:
                    if offset <= u < offset + num_frames:
                        tokens, timestamps, dec_state = [WORD_ID, self.eou_id], [u, u], "ended"
                        break
            if self.total_frames is not None:
                timestamps = [t - offset for t in timestamps]  # offline: frame indices within the segment
            hyp = Hypothesis(
                score=0.0,
                y_sequence=torch.tensor(tokens, dtype=torch.long),
                timestamp=torch.tensor(timestamps, dtype=torch.long),
                dec_state=dec_state,
            )
            if prev is not None:
                prev.merge_(hyp)
                hyp = prev
            hypotheses.append(hyp)
        return (hypotheses,)


@pytest.mark.unit
def test_get_utterance_end_token_ids(eou_model):
    eou_ids = get_utterance_end_token_ids(eou_model.tokenizer)
    assert eou_model.tokenizer.ids_to_tokens(eou_ids) == ["<EOU>", "<EOB>"]
    assert get_utterance_end_token_ids(None) == []


@pytest.mark.unit
def test_transcribe_continues_after_eou(eou_model):
    """transcribe() decodes every utterance of a multi-utterance audio, not only the first one."""
    eou_id = get_utterance_end_token_ids(eou_model.tokenizer)[0]
    audio = torch.randn(16000 * 3) * 0.1
    with torch.no_grad():
        processed, processed_len = eou_model.preprocessor(
            input_signal=audio[None], length=torch.tensor([audio.shape[0]])
        )
        _, encoded_len = eou_model.forward(processed_signal=processed, processed_signal_length=processed_len)
    total_frames = int(encoded_len[0])
    utterance_frames = [3, 20, 40, total_frames - 4]
    eou_model.decoding.decoding = _SingleUtteranceDecodingSim(
        [utterance_frames], eou_id=eou_id, total_frames=[total_frames]
    )

    hyp = eou_model.transcribe([audio.numpy()], batch_size=1, return_hypotheses=True)[0]

    assert hyp.y_sequence.tolist() == [WORD_ID, eou_id] * len(utterance_frames)
    assert hyp.timestamp.tolist() == [t for u in utterance_frames for t in (u, u)]
    assert hyp.text.count("<EOU>") == len(utterance_frames)


@pytest.mark.unit
def test_transcribe_without_eou_tokens_is_unchanged(eou_model):
    """Models without utterance-end tokens use the regular decoding path."""
    eou_model.tokenizer.token_to_id = lambda token: 0  # no utterance-end tokens in the vocabulary
    calls = []
    original = eou_model._decode_with_utterance_end_restart
    eou_model._decode_with_utterance_end_restart = lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)
    eou_model.transcribe([torch.randn(16000).numpy() * 0.1], batch_size=1)
    assert calls == []


@pytest.mark.unit
def test_cache_aware_streaming_continues_after_eou(eou_model):
    """conformer_stream_step restarts decoding and resets the encoder cache only for the streams that emit <EOU>."""
    eou_id = get_utterance_end_token_ids(eou_model.tokenizer)[0]
    eou_model.encoder.setup_streaming_params()
    cfg = eou_model.encoder.streaming_cfg
    utterance_frames = [[3, 20, 40], [70]]  # the second stream emits <EOU> only at the end
    eou_model.decoding.decoding = _SingleUtteranceDecodingSim(utterance_frames, eou_id=eou_id)

    num_feature_frames = 80 * SUBSAMPLING_FACTOR
    features = torch.randn(2, 64, num_feature_frames)
    cache_last_channel, cache_last_time, cache_last_channel_len = eou_model.encoder.get_initial_cache_state(
        batch_size=2
    )
    hyps = None
    chunk_size, shift_size, pre_encode_cache_size = cfg.chunk_size[1], cfg.shift_size[1], cfg.pre_encode_cache_size[1]
    start, step = 0, 0
    eou_steps = set()
    with torch.inference_mode():
        while start < num_feature_frames:
            first = step == 0
            size = cfg.chunk_size[0] if first else chunk_size
            cache_start = max(0, start - pre_encode_cache_size) if not first else start
            chunk = features[:, :, cache_start : start + size]
            num_eou_before = [
                0 if hyps is None else hyp.y_sequence.tolist().count(eou_id) for hyp in hyps or [None] * 2
            ]
            (_, _, cache_last_channel, cache_last_time, cache_last_channel_len, hyps) = (
                eou_model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=torch.tensor([chunk.shape[-1]] * 2),
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=start + size >= num_feature_frames,
                    previous_hypotheses=hyps,
                    drop_extra_pre_encoded=0 if first else cfg.drop_extra_pre_encoded,
                )
            )
            if hyps[0].y_sequence.tolist().count(eou_id) > num_eou_before[0]:
                eou_steps.add(step)
                assert cache_last_channel_len[0].item() == 0
                assert cache_last_channel[:, 0].abs().sum().item() == 0
                assert cache_last_time[:, 0].abs().sum().item() == 0
                assert cache_last_channel_len[1].item() > 0
            start += cfg.chunk_size[0] if first else shift_size
            step += 1

    assert len(eou_steps) == 3
    assert hyps[0].y_sequence.tolist() == [WORD_ID, eou_id] * 3
    assert hyps[0].timestamp.tolist() == [t for u in utterance_frames[0] for t in (u, u)]
    assert hyps[1].y_sequence.tolist() == [WORD_ID, eou_id]


@pytest.mark.unit
def test_reset_hypothesis_decoder_state_keeps_time_offset(eou_model):
    """Resetting a label-looping decoder state restores the start-of-sequence state but keeps the frame offset."""
    computer = eou_model.decoding.decoding.decoding_computer
    with torch.inference_mode():
        encoded = torch.randn(1, 16, 10)
        hyp = eou_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, torch.tensor([10]), return_hypotheses=True, partial_hypotheses=[None]
        )[0]
        start_item = computer.split_batched_state(computer._get_batched_decoding_state_after_sos("cpu", 1))[0]
        hyp.dec_state.label.fill_(WORD_ID)
        reset_hypothesis_decoder_state_(hyp, decoding_computer=computer)

    assert hyp.dec_state.decoded_length.item() == 10
    assert hyp.dec_state.label.item() == start_item.label.item()
    torch.testing.assert_close(hyp.dec_state.predictor_output, start_item.predictor_output)
    for state, start_state in zip(hyp.dec_state.predictor_state, start_item.predictor_state):
        torch.testing.assert_close(state, start_state)


@pytest.mark.unit
def test_truncate_hypothesis_after_first_utterance_end():
    hyp = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([5, 1, 6, 1]),
        timestamp=torch.tensor([2, 3, 7, 8]),
        alignments=[[i] for i in range(10)],
    )
    assert truncate_hypothesis_after_first_utterance_end_(hyp, token_ids=[1], num_frames=10) == 3
    assert hyp.y_sequence.tolist() == [5, 1]
    assert hyp.timestamp.tolist() == [2, 3]
    assert len(hyp.alignments) == 4

    # an utterance-end token at the last frame does not start a new segment
    hyp = Hypothesis(score=0.0, y_sequence=torch.tensor([5, 1]), timestamp=torch.tensor([2, 9]))
    assert truncate_hypothesis_after_first_utterance_end_(hyp, token_ids=[1], num_frames=10) is None
    assert hyp.y_sequence.tolist() == [5, 1]
