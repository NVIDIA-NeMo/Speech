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
Multi-utterance decoding with end-of-utterance models and ``decoding.restart_after_tokens``.

These models are trained on single-utterance samples and stop emitting tokens after the first ``<EOU>`` unless
decoding restarts after it. The tests use a tiny randomly-initialized model whose decoding algorithm is replaced
with a simulation of this behavior: it emits ``<word> <EOU>`` for the next utterance and nothing after that.
"""

import dataclasses

import pytest
import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.eou_utils import (
    append_hypothesis_segment_,
    get_restart_token_ids,
    reset_hypothesis_decoder_state_,
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
            'decoding': {
                'strategy': 'greedy_batch',
                'greedy': {'max_symbols': 5},
                'restart_after_tokens': ['<EOU>', '<EOB>'],
            },
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
    Simulates the greedy decoding of an end-of-utterance model trained on single-utterance samples.

    Decoder states are real label-looping state items: ``decoded_length`` gives the absolute frame offset, and the last
    label is ``<EOU>`` once the utterance has ended. A hypothesis that has not ended emits ``<word> <EOU>`` on the next
    utterance frame (absolute encoder frame); after ``<EOU>``, it emits nothing unless its decoder state is reset.
    Utterance frames are looked up by batch index, so only the first stream may restart within a chunk
    (single-stream re-decoding uses batch index 0).
    """

    def __init__(self, decoding_computer, utterance_frames: list[list[int]], eou_id: int):
        self.decoding_computer = decoding_computer
        self.utterance_frames = utterance_frames
        self.eou_id = eou_id

    def __call__(self, encoder_output, encoded_lengths, partial_hypotheses=None):
        hypotheses = []
        for b, num_frames in enumerate(encoded_lengths.tolist()):
            prev = partial_hypotheses[b] if partial_hypotheses is not None else None
            state = prev.dec_state if prev is not None and prev.dec_state is not None else None
            if state is None:
                state = self.decoding_computer._get_decoding_state_item_after_sos(device=encoder_output.device)
            start = int(state.decoded_length)
            tokens, timestamps, label = [], [], int(state.label)
            if label != self.eou_id:
                for u in self.utterance_frames[b]:
                    if start <= u < start + num_frames:
                        tokens, timestamps, label = [WORD_ID, self.eou_id], [u, u], self.eou_id
                        break
            new_state = dataclasses.replace(
                state,
                label=torch.tensor(label),
                decoded_length=torch.tensor(start + num_frames),
            )
            hyp = Hypothesis(
                score=-float(len(tokens)),
                y_sequence=torch.tensor(tokens, dtype=torch.long),
                timestamp=torch.tensor(timestamps, dtype=torch.long),
                dec_state=new_state,
                length=num_frames,
            )
            if prev is not None:
                prev.merge_(hyp)
                hyp = prev
            hypotheses.append(hyp)
        return (hypotheses,)


def _simulate(model, utterance_frames: list[list[int]]) -> int:
    eou_id = get_restart_token_ids(model)[0]
    model.decoding.decoding = _SingleUtteranceDecodingSim(
        model.decoding.decoding.decoding_computer, utterance_frames, eou_id=eou_id
    )
    return eou_id


@pytest.mark.unit
def test_get_restart_token_ids(eou_model):
    assert eou_model.tokenizer.ids_to_tokens(get_restart_token_ids(eou_model)) == ["<EOU>", "<EOB>"]
    with open_dict(eou_model.decoding.cfg):
        eou_model.decoding.cfg.restart_after_tokens = ["<XYZ>"]
    with pytest.raises(ValueError):
        get_restart_token_ids(eou_model)
    with open_dict(eou_model.decoding.cfg):
        eou_model.decoding.cfg.restart_after_tokens = ["<EOU>"]
        eou_model.decoding.cfg.strategy = "beam"
    with pytest.raises(NotImplementedError):
        get_restart_token_ids(eou_model)
    with open_dict(eou_model.decoding.cfg):
        eou_model.decoding.cfg.restart_after_tokens = None
    assert get_restart_token_ids(eou_model) == []


@pytest.mark.unit
def test_transcribe_continues_after_eou(eou_model):
    """transcribe() decodes every utterance of a multi-utterance audio, not only the first one."""
    utterance_frames = [3, 20, 21, 40, 60]  # 20 and 21 fall into the same streaming chunk
    eou_id = _simulate(eou_model, [utterance_frames])
    audio = torch.randn(16000 * 3) * 0.1

    hyp = eou_model.transcribe([audio.numpy()], batch_size=1, return_hypotheses=True)[0]

    assert hyp.y_sequence.tolist() == [WORD_ID, eou_id] * len(utterance_frames)
    assert hyp.timestamp.tolist() == [t for u in utterance_frames for t in (u, u)]
    assert hyp.text.count("<EOU>") == len(utterance_frames)
    # segments are disjoint: decoded frames are counted once, and scores of re-decoded chunks are not duplicated
    assert hyp.score == -2.0 * len(utterance_frames)
    assert hyp.length == hyp.dec_state.decoded_length.item()


@pytest.mark.unit
def test_transcribe_encodes_audio_once(eou_model):
    """Offline restarts stream over the features, so the encoded features grow linearly with the audio."""
    _simulate(eou_model, [list(range(2, 1000, 4))])  # an utterance in every other chunk
    encoded_features = []
    stream_step = eou_model.encoder.cache_aware_stream_step

    def counting_stream_step(**kwargs):
        encoded_features.append(int(kwargs['processed_signal_length'][0]))
        return stream_step(**kwargs)

    eou_model.encoder.cache_aware_stream_step = counting_stream_step
    for seconds in (2, 4):
        encoded_features.clear()
        eou_model.transcribe([(torch.randn(16000 * seconds) * 0.1).numpy()], batch_size=1)
        num_features = seconds * 100
        # chunk features + pre-encode cache per chunk, with at most one re-encoded chunk tail per restart
        assert sum(encoded_features) < 3 * num_features


@pytest.mark.unit
def test_transcribe_without_restart_tokens_is_unchanged(eou_model):
    """Without ``restart_after_tokens``, the regular single-pass decoding is used."""
    eou_id = _simulate(eou_model, [[3, 20]])
    with open_dict(eou_model.decoding.cfg):
        eou_model.decoding.cfg.restart_after_tokens = None
    hyp = eou_model.transcribe([(torch.randn(16000 * 3) * 0.1).numpy()], batch_size=1, return_hypotheses=True)[0]
    assert hyp.y_sequence.tolist() == [WORD_ID, eou_id]


@pytest.mark.unit
def test_stream_step_restarts_within_chunk(eou_model):
    """Several utterances ending inside a single (final) chunk are all decoded, with absolute timestamps."""
    utterance_frames = [1, 3, 7]
    eou_id = _simulate(eou_model, [utterance_frames])
    eou_model.encoder.setup_streaming_params()
    features = torch.randn(1, 64, 10 * SUBSAMPLING_FACTOR)
    num_frames = 10 + 1  # causal subsampling: floor(num_features / subsampling_factor) + 1
    cache = eou_model.encoder.get_initial_cache_state(batch_size=1)
    with torch.inference_mode():
        *_, cache_len, hyps = eou_model.conformer_stream_step(
            processed_signal=features,
            processed_signal_length=torch.tensor([features.shape[-1]]),
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=True,
            drop_extra_pre_encoded=0,
        )
    assert hyps[0].y_sequence.tolist() == [WORD_ID, eou_id] * 3
    assert hyps[0].timestamp.tolist() == [t for u in utterance_frames for t in (u, u)]
    assert hyps[0].dec_state.decoded_length.item() == num_frames
    assert cache_len[0].item() == num_frames - (
        utterance_frames[-1] + 1
    )  # encoder context restarted after the last <EOU>


@pytest.mark.unit
def test_cache_aware_streaming_continues_after_eou(eou_model):
    """conformer_stream_step restarts decoding and resets the encoder cache only for the streams that emit <EOU>."""
    utterance_frames = [[3, 20, 40], []]  # the second stream never emits <EOU>
    eou_id = _simulate(eou_model, utterance_frames)
    eou_model.encoder.setup_streaming_params()
    cfg = eou_model.encoder.streaming_cfg

    num_feature_frames = 80 * SUBSAMPLING_FACTOR
    features = torch.randn(2, 64, num_feature_frames)
    cache_last_channel, cache_last_time, cache_last_channel_len = eou_model.encoder.get_initial_cache_state(
        batch_size=2
    )
    hyps = None
    start, step = 0, 0
    with torch.inference_mode():
        while start < num_feature_frames:
            first = step == 0
            size = cfg.chunk_size[0] if first else cfg.chunk_size[1]
            chunk = features[:, :, (start if first else max(0, start - cfg.pre_encode_cache_size[1])) : start + size]
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
                # the encoder context of the first stream restarted within this chunk, the other stream is unaffected
                assert cache_last_channel_len[0].item() < cache_last_channel_len[1].item()
            start += size if first else cfg.shift_size[1]
            step += 1

    assert hyps[0].y_sequence.tolist() == [WORD_ID, eou_id] * 3
    assert hyps[0].timestamp.tolist() == [t for u in utterance_frames[0] for t in (u, u)]
    assert hyps[1].y_sequence.tolist() == []
    assert cache_last_channel_len[1].item() == cfg.last_channel_cache_size


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
def test_append_hypothesis_segment():
    hyp = Hypothesis(score=-1.0, y_sequence=[5, 1], timestamp=[1, 2], token_duration=[0, 1], length=3)
    segment = Hypothesis(score=-2.0, y_sequence=[6], timestamp=[0], token_duration=[2], length=4)
    append_hypothesis_segment_(hyp, segment, frame_offset=3)
    assert hyp.y_sequence == [5, 1, 6]
    assert hyp.timestamp == [1, 2, 3]
    assert hyp.token_duration == [0, 1, 2]
    assert hyp.score == -3.0
    assert hyp.length == 7
