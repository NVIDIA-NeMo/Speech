# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.tts.modules.attention_alignment import (
    StreamingTimestampSession,
    resolve_timestamp_config,
    save_token_timestamps,
)
from nemo.collections.tts.modules.magpietts_inference import utils
from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
    _timestamp_input_ids,
)

pytestmark = pytest.mark.unit


def test_recording_defaults_are_disabled_and_configuration_rejects_guidance_options():
    config = resolve_timestamp_config()
    assert not config['enabled'] and config['layer'] == 'middle'
    assert 'lookahead_frames' not in config
    for option in ('lookahead_frames', 'prior_epsilon', 'enabled_guidance'):
        with pytest.raises(ValueError, match='Unknown'):
            resolve_timestamp_config({option: 1})


def test_first_audio_frame_has_live_events_and_confidence_uses_the_full_key_distribution():
    session = _session()
    session.register_text(0, 1, 17)
    _observe(session, [[0.7, 0.2, 0.1]], frame=10)
    events = session.drain_events()
    assert events and events[0]['event'] == 'start'
    assert events[0]['start'] == 0.0 and events[0]['emitted_at'] == pytest.approx(0.08)
    assert events[0]['confidence'] == pytest.approx(0.2)
    assert not events[0]['forced']
    assert session.snapshots()[0]['timestamps'][0]['status'] == 'active'
    assert session.drain_events() == []


def test_each_audio_step_has_instantaneous_frame_confidence_and_independent_retained_history():
    session = _session()
    session.register_text(0, 1, 17)
    _observe(session, [[0.7, 0.2, 0.1]], frame=10)
    first = session.drain_frames()
    assert len(first) == 1
    assert first[0]['frame_index'] == first[0]['turn_frame_index'] == 0
    assert first[0]['batch_index'] == first[0]['turn_index'] == 0
    assert first[0]['codec_frame_start'] == 0 and first[0]['codec_frame_end'] == 2
    assert first[0]['valid_codec_frames'] == 2
    assert first[0]['start'] == 0.0 and first[0]['end'] == pytest.approx(0.08)
    assert first[0]['emitted_at'] == pytest.approx(0.08)
    assert first[0]['confidence'] == pytest.approx(0.2)
    assert first[0]['attention_confidence'] == pytest.approx(0.2)
    assert first[0]['token_index'] == 0 and first[0]['token_id'] == 17
    assert first[0]['text_key_position'] == first[0]['assignments'][0]['text_key_position'] == 1
    assert session.drain_events() and session.drain_frames() == []
    first[0]['confidence'] = -1  # A consumer cannot mutate the retained trace.
    first[0]['assignments'][0]['confidence'] = -1
    _observe(session, [[0.2, 0.7, 0.1]], frame=12)
    second = session.drain_frames()[0]
    assert second['frame_index'] == second['turn_frame_index'] == 1
    assert second['codec_frame_start'] == 2 and second['codec_frame_end'] == 4
    assert second['confidence'] == pytest.approx(0.7)
    trace = session.snapshots(finalize=True)[0]
    assert trace['codec_frame_seconds'] == pytest.approx(0.04)
    assert trace['frame_step_seconds'] == pytest.approx(0.08)
    assert trace['frame_count'] == 2 and trace['frame_history_retained']
    assert [frame['confidence'] for frame in trace['frames']] == pytest.approx([0.2, 0.7])
    assert trace['frames'][0]['assignments'][0]['confidence'] == pytest.approx(0.2)
    trace['frames'][0]['token_id'] = -1
    assert session.snapshots()[0]['frames'][0]['token_id'] == 17
    assert session.drain_frames() == []


def test_frames_remain_explicit_when_attention_is_missing_or_gated_out():
    session = StreamingTimestampSession(1, 0.04, 2, {'enabled': True, 'min_confidence': 0.5})
    session.register_text(0, 0, 17)
    session.observe_audio([True], frame_start=10)
    _observe(session, [[0.2, 0.8]], frame=12)
    frames = session.drain_frames()
    assert [frame['frame_index'] for frame in frames] == [0, 1]
    assert [frame['start'] for frame in frames] == pytest.approx([0.0, 0.08])
    assert [frame['end'] for frame in frames] == pytest.approx([0.08, 0.16])
    assert all(frame['status'] == 'unaligned' and frame['token_index'] is None for frame in frames)
    assert all(frame['text_key_position'] is None for frame in frames)
    assert all(frame['confidence'] is None for frame in frames)
    assert session.snapshots()[0]['frame_count'] == 2


def test_frame_history_can_be_disabled_while_frames_are_drained_per_batch():
    session = StreamingTimestampSession(2, 0.04, 2, {'enabled': True, 'retain_frame_history': False})
    session.register_text(0, 0, 17)
    session.register_text(1, 0, 18)
    _observe(session, [[1.0], [1.0]], frame=4)
    assert [frame['token_id'] for frame in session.drain_frames([1])] == [18]
    assert [frame['token_id'] for frame in session.drain_frames()] == [17]
    assert session.drain_frames() == []
    for trace in session.snapshots(finalize=True):
        assert trace['frames'] == [] and trace['frame_count'] == 1 and not trace['frame_history_retained']
    with pytest.raises(ValueError, match='retain_frame_history'):
        resolve_timestamp_config({'retain_frame_history': 1})
    with pytest.raises(ValueError, match='batch index'):
        session.drain_frames([2])


@pytest.mark.parametrize('frame,count', [(-1, 2), (12, -1), (12, 3), (11, 2), (12.0, 2), (12, 1.5)])
def test_invalid_audio_interval_preserves_the_previous_clock_and_remains_usable(frame, count):
    session = _session()
    session.register_text(0, 0, 17)
    _observe(session, [[1.0]], frame=10)
    before = session.snapshots()
    with pytest.raises(ValueError):
        session.observe_row(0, frame, count)
    assert session.snapshots() == before
    _observe(session, [[1.0]], frame=12)
    assert session.snapshots(finalize=True)[0]['audio_duration_seconds'] == pytest.approx(0.16)


@pytest.mark.parametrize('needed,counts', [([True], [2, 2]), ([True, True], [2]), ([True, True], [2, 3])])
def test_invalid_batch_audio_metadata_is_rejected_before_any_row_is_updated(needed, counts):
    session = _session(batch_size=2)
    with pytest.raises(ValueError):
        session.observe_audio(needed, frame_start=10, valid_frames=counts)
    assert session.drain_frames() == []
    assert all(trace['audio_duration_seconds'] == 0.0 for trace in session.snapshots())
    session.observe_audio([True, True], frame_start=10, valid_frames=[2, 2])
    assert [frame['frame_index'] for frame in session.drain_frames()] == [0, 0]


@pytest.mark.parametrize('lengths', [[], [float('nan')], [float('inf')], [-0.1]])
def test_invalid_snapshot_lengths_do_not_finalize_or_corrupt_the_stream(lengths):
    session = _session()
    session.register_text(0, 0, 17)
    _observe(session, [[1.0]], frame=10)
    with pytest.raises(ValueError):
        session.snapshots(finalize=True, audio_lengths=lengths)
    _observe(session, [[1.0]], frame=12)
    assert session.snapshots(finalize=True)[0]['audio_duration_seconds'] == pytest.approx(0.16)


def test_turn_index_accessor_reads_metadata_without_copying_timestamp_history(monkeypatch):
    session = _session(batch_size=2)
    session.register_text(0, 0, 17)
    session.reset_turn(0)
    monkeypatch.setattr(session, 'snapshots', lambda **kwargs: pytest.fail('Timestamp history was copied'))
    indices = session.get_turn_indices()
    assert indices == [1, 0]
    indices[0] = -1
    assert session.get_turn_indices() == [1, 0]


def test_cfg_unconditional_attention_and_future_keys_do_not_leak_into_timestamps():
    session = _session()
    session.register_text(0, 0, 17)
    session.register_text(0, 2, 18)
    query = torch.ones(2, 1, 1, 1)
    key = torch.tensor([[[[0.0], [-20.0], [100.0]]], [[[0.0], [100.0], [0.0]]]])
    before_query, before_key = query.clone(), key.clone()
    session.observe_attention(query, key, cache_position=torch.tensor([0]))
    session.observe_audio([True], frame_start=0)
    records = session.snapshots()[0]['timestamps']
    assert records[0]['confidence'] == pytest.approx(1.0)
    assert records[1]['start'] is None
    assert all(event['batch_index'] == 0 for event in session.drain_events())
    assert torch.equal(query, before_query) and torch.equal(key, before_key)


def test_forced_progress_is_published_in_the_same_frame_with_zero_confidence():
    session = _session()
    for index in range(3):
        session.register_text(0, index, index + 20)
    _observe(session, [[0.9, 0.09, 0.01]], frame=0)
    session.drain_events()
    _observe(session, [[0.2, 0.001, 0.799]], frame=2)
    events = session.drain_events()
    forced = [event for event in events if event['forced']]
    assert {event['token_index'] for event in forced} == {1, 2}
    assert all(event['confidence'] == 0.0 and event['emitted_at'] == pytest.approx(0.16) for event in forced)
    records = session.snapshots()[0]['timestamps']
    assert records[1]['start'] == pytest.approx(0.08) and records[1]['end'] == pytest.approx(0.12)
    assert records[2]['start'] == pytest.approx(0.12) and records[2]['end'] == pytest.approx(0.16)
    assert records[1]['attention_confidence'] > 0
    frames = session.drain_frames()
    assert len(frames) == 2 and frames[1]['token_index'] == 2
    assert frames[1]['forced'] and frames[1]['confidence'] == 0.0
    assert frames[1]['attention_confidence'] == pytest.approx(0.799)
    assert [assignment['token_index'] for assignment in frames[1]['assignments']] == [1, 2]
    assert all(assignment['confidence'] == 0.0 for assignment in frames[1]['assignments'])
    assert all(assignment['emitted_at'] == pytest.approx(0.16) for assignment in frames[1]['assignments'])


def test_turn_reset_preserves_occurrences_and_the_global_audio_clock():
    session = _session()
    session.register_text(0, 0, 17)
    _observe(session, [[1.0]], frame=10)
    first = session.snapshots()[0]['timestamps'][0]
    session.reset_turn(0)
    session.reset_turn(0)
    session.register_text(0, 5, 17)
    _observe(session, [[0.1, 0.0, 0.0, 0.0, 0.0, 0.9]], frame=18)
    records = session.snapshots(finalize=True)[0]['timestamps']
    assert len(records) == 2
    assert [record['turn_index'] for record in records] == [0, 1]
    assert records[0]['start'] == first['start'] and records[0]['end'] == first['end']
    assert records[1]['start'] == pytest.approx(0.32)
    assert records[1]['end'] == pytest.approx(0.40)
    frames = session.drain_frames()
    assert [frame['frame_index'] for frame in frames] == [0, 1]
    assert [frame['turn_frame_index'] for frame in frames] == [0, 0]
    assert [frame['turn_index'] for frame in frames] == [0, 1]
    assert [frame['codec_frame_start'] for frame in frames] == [0, 8]
    assert [frame['codec_frame_end'] for frame in frames] == [2, 10]
    assert [frame['text_key_position'] for frame in frames] == [0, 5]


def test_partial_audio_eos_does_not_add_a_frame_or_fabricate_a_trailing_word():
    session = _session()
    session.register_text(0, 0, 17)
    session.register_text(0, 1, 18)
    _observe(session, [[0.99, 0.01]], frame=4, count=1)
    snapshot = session.snapshots(finalize=True, audio_lengths=[0.04])[0]
    assert snapshot['audio_duration_seconds'] == pytest.approx(0.04)
    assert snapshot['timestamps'][0]['end'] == pytest.approx(0.04)
    assert snapshot['timestamps'][1]['start'] is None
    assert snapshot['frame_count'] == 1 and len(snapshot['frames']) == 1
    assert snapshot['frames'][0]['codec_frame_end'] == 1
    assert snapshot['frames'][0]['valid_codec_frames'] == 1
    assert snapshot['frames'][0]['end'] == pytest.approx(0.04)


def test_padding_context_and_zero_frame_eos_do_not_generate_token_intervals():
    session = _session(batch_size=2)
    session.register_text(0, 0, 17)
    session.register_text(1, 0, 18)
    _observe(session, [[1.0], [1.0]], frame=0, needs_audio=[False, True], count=0)
    assert session.drain_events() == []
    assert session.drain_frames() == []
    assert all(token['start'] is None for trace in session.snapshots() for token in trace['timestamps'])


def test_cli_flag_enables_only_timestamp_recording_and_preserves_ordinary_identifier():
    parser = argparse.ArgumentParser()
    utils._add_easy_magpie_args(parser)
    assert not parser.parse_args([]).save_timestamps
    assert parser.parse_args(['--save_timestamps']).save_timestamps
    assert not parser.parse_args(['--no-save_timestamps']).save_timestamps
    with pytest.raises(SystemExit):
        parser.parse_args(['--timestamp_lookahead_frames', '1'])
    ordinary = EasyMagpieInferenceConfig().build_identifier()
    assert EasyMagpieInferenceConfig(save_timestamps=True).build_identifier() == ordinary + '_Timestamps'


def test_runner_enables_the_read_only_timestamp_api_and_leaves_normal_loading_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(EasyMagpieInferenceRunner, '_configure_tokenizer', lambda self: None)
    model = SimpleNamespace(configure_timestamps=lambda **kwargs: calls.append(kwargs))
    EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig(save_timestamps=True))
    assert calls == [{'enabled': True}]
    calls.clear()
    EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig())
    assert calls == [{'enabled': False}]
    with pytest.raises(ValueError, match='V2'):
        EasyMagpieInferenceRunner(SimpleNamespace(), EasyMagpieInferenceConfig(save_timestamps=True))


def test_requested_ids_keep_repetitions_and_drop_only_timeline_controls():
    model = SimpleNamespace(bos_id=90, eos_id=91, pad_id=92)
    assert _timestamp_input_ids(model, torch.tensor([90, 7, 7, 92, 8, 91, 100, -1])) == [7, 7, 8]


def test_response_text_warmup_registers_target_keys_without_labeling_user_profile_audio():
    from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import (
        _attention_peaks,
        _init_stream,
        _model,
    )

    model = _model(enabled=True)
    state = _init_stream(model)
    state = model.streaming_prefill(state, torch.tensor([[20, 21], [22, 23]]))
    model.reset_timestamp_turn(state)
    with _attention_peaks([0]):
        for token in [7, 8, 9, model.pad_id, model.pad_id]:
            state, audio, _ = model.streaming_step(
                state, torch.full((2,), token), prefill_like_step=True, timestamp_text_warmup=True
            )
            assert audio is None
        trace = model.get_timestamps(state)[0]
        assert trace['turns'][-1]['source_token_ids'] == [7, 8, 9]
        assert all(record['start'] is None for record in trace['timestamps'])
        assert model.drain_timestamp_events(state) == []
        state, _, _ = model.streaming_step(state, torch.full((2,), model.eos_id))
        events = model.drain_timestamp_events(state)
        assert events and events[0]['token_id'] == 7
    assert all(record['token_id'] not in (20, 21, 22, 23) for record in model.get_timestamps(state)[0]['timestamps'])


def test_external_singleturn_runner_saves_timestamp_sidecar_and_manifest(tmp_path, monkeypatch):
    session = _session()
    session.register_text(0, 0, 17)
    _observe(session, [[1.0]], frame=0)
    output = SimpleNamespace(
        predicted_audio=torch.zeros(1, 8),
        predicted_audio_lens=torch.tensor([8]),
        predicted_codes=torch.zeros(1, 1, 2),
        predicted_codes_lens=torch.tensor([2]),
        rtf_metrics={},
        timestamps=session.snapshots(finalize=True),
    )
    model = SimpleNamespace(
        configure_timestamps=lambda **kwargs: None,
        infer_batch=lambda *args, **kwargs: output,
        tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda token_id: 'hello'),
        bos_id=90,
        eos_id=91,
        pad_id=92,
        sample_rate=100,
        output_sample_rate=100,
    )
    batch = {'text': torch.tensor([[17, 91]]), 'text_lens': torch.tensor([2])}

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

        def collate_fn(self, values):
            return batch

    monkeypatch.setattr(EasyMagpieInferenceRunner, '_configure_tokenizer', lambda self: None)
    monkeypatch.setattr(EasyMagpieInferenceRunner, '_batch_to_cuda', staticmethod(lambda values: values))
    runner = EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig(save_timestamps=True))
    runner._run_decoder_only_inference(
        Dataset(),
        str(tmp_path),
        [{'text': 'hello', 'audio_filepath': 'reference.wav'}],
        str(tmp_path),
        save_context_audio=False,
        save_predicted_codes=False,
    )
    destination = tmp_path / 'predicted_audio_0.timestamps.json'
    payload = json.loads(destination.read_text())
    assert payload['expected_input_token_ids'] == [17]
    assert payload['timestamps'][0]['token'] == 'hello'
    assert payload['timestamps'][0]['confidence'] == pytest.approx(1.0)
    assert payload['timestamps'][0]['forced'] is False
    record = json.loads(Path(runner.timestamp_manifest_path).read_text())
    assert record['timestamps_filepath'] == str(destination)
    assert record['predicted_audio_filepath'] == str(tmp_path / 'predicted_audio_0.wav')


def test_multiturn_rank_merge_preserves_timestamp_sidecars_and_rebases_audio_paths(tmp_path):
    session = _session()
    session.register_text(0, 0, 17)
    _observe(session, [[1.0]], frame=0)
    for rank in range(2):
        directory = tmp_path / f'rank_{rank:04d}'
        directory.mkdir()
        for name in ('predicted_audio_0.wav', 'target_audio_0.wav', 'context_audio_0.wav'):
            (directory / name).write_bytes(b'audio')
        sidecar = save_token_timestamps(directory / 'predicted_audio_0.wav', session.snapshots(finalize=True)[0])
        row = dict(
            audio_filepath='target_audio_0.wav',
            context_audio_filepath='context_audio_0.wav',
            text='hello',
            timestamps_filepath=Path(sidecar).name,
        )
        (directory / f'multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl').write_text(json.dumps(row) + '\n')
    manifest = Path(utils._merge_multiturn_rank_outputs(str(tmp_path), 2, save_predicted_codes=False))
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for index, record in enumerate(records):
        sidecar = tmp_path / f'predicted_audio_{index}.timestamps.json'
        assert record['timestamps_filepath'] == str(sidecar)
        assert json.loads(sidecar.read_text())['predicted_audio_filepath'] == str(
            tmp_path / f'predicted_audio_{index}.wav'
        )


def _session(batch_size=1):
    return StreamingTimestampSession(batch_size, 0.04, 2, {'enabled': True, 'confirmation_frames': 1})


def _observe(session, probabilities, frame, count=2, needs_audio=None):
    logits = torch.tensor(probabilities).clamp(min=1e-30).log()
    query = torch.ones(logits.size(0), 1, 1, 1)
    key = logits[:, None, :, None]
    session.observe_attention(query, key, cache_position=torch.tensor([key.size(2) - 1]))
    session.observe_audio(
        [True] * session.batch_size if needs_audio is None else needs_audio,
        frame_start=frame,
        valid_frames=[count] * session.batch_size,
    )
