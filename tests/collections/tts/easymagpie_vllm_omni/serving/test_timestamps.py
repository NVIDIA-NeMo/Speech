# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.timestamps import (
    StreamingAlignmentDecoder,
    bundle_timestamp_runtime,
    load_timestamp_runtime,
    pack_alignment_packet,
    unpack_alignment_packets,
)
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer

pytestmark = pytest.mark.unit


class CharacterTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **_kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, **_kwargs):
        return ''.join(chr(i) for i in ids)


@pytest.fixture
def model(tmp_path):
    repo = Path(__file__).resolve().parents[5]
    native = repo / 'nemo/collections/tts/modules/attention_alignment.py'
    config = bundle_timestamp_runtime(tmp_path, native)
    config.update(frame_stacking_factor=1, timestamp_config={'confirmation_frames': 1})
    (tmp_path / 'config.json').write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def tokenizer():
    return EasyMagpieTextTokenizer(CharacterTokenizer(), text_vocab_size=131072)


def observation(sequence, probabilities, registrations=(), start=None, count=1):
    return {
        'sequence': sequence,
        'codec_frame_seconds': 0.08,
        'codec_frame_start': sequence if start is None else start,
        'valid_codec_frames': count,
        'registrations': list(registrations),
        'token_indices': list(range(len(probabilities))),
        'probabilities': list(probabilities),
        'attention_block': 12,
    }


def registration(index, token_id, position=None):
    return {
        'token_index': index,
        'input_token_index': index,
        'token_id': token_id,
        'text_key_position': index if position is None else position,
    }


def test_streaming_delta_is_available_before_final_audio(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'ab')
    delta = decoder.update(observation(0, [0.8], [registration(0, ord('a'))]))
    assert delta['frames'][0]['token_id'] == ord('a')
    assert delta['token_timestamps']['0']['frame_step'] == 0
    assert delta['character_timestamps']['0']['character'] == 'a'
    assert decoder.tracker._duration is None
    assert '1' not in delta['token_timestamps']


def test_missing_occurrences_keep_null_times(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'abc')
    decoder.update(observation(0, [0.9], [registration(0, ord('a'))]))
    final = decoder.finalize(0.08)
    assert list(final['token_timestamps']) == ['0', '1', '2']
    assert final['token_timestamps']['1']['token_id'] == ord('b')
    assert final['token_timestamps']['1']['time_seconds'] is None
    assert final['character_timestamps']['2']['time_seconds'] is None
    assert final['words'][0]['partially_aligned']


def test_repeated_ids_are_distinct_input_occurrences(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'aa')
    decoder.update(observation(0, [0.9, 0.01], [registration(0, 97), registration(1, 97)]))
    decoder.update(observation(1, [0.9, 0.01]))
    decoder.update(observation(2, [0.01, 0.9]))
    decoder.update(observation(3, [0.01, 0.9]))
    final = decoder.finalize(0.32)
    assert final['token_timestamps']['0']['token_id'] == final['token_timestamps']['1']['token_id'] == 97
    assert final['token_timestamps']['0']['start'] < final['token_timestamps']['1']['start']
    assert final['token_timestamps']['0']['end'] <= final['token_timestamps']['1']['end']


def test_probabilities_are_not_renormalized_to_text_mass(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'a')
    decoder.update(observation(0, [0.1], [registration(0, 97)]))
    final = decoder.finalize(0.08)
    assert final['frames'][0]['text_attention_mass'] == pytest.approx(0.1)
    assert final['token_timestamps']['0']['confidence'] == pytest.approx(0.1)


def test_native_streaming_session_parity_with_delays_and_partial_visibility(model, tokenizer):
    runtime = load_timestamp_runtime(model)
    native = runtime.StreamingTimestampSession(1, 0.08, 1, {'confirmation_frames': 1})
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'ab')
    keys = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.1, 0.1], [0.0, 2.0]])
    native.register_text(0, 1, ord('a'), 'a')
    for step, q in enumerate([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]):
        visible = 2 if step < 2 else 4
        additions = [registration(0, 97, 1)] if step == 0 else []
        if step == 2:
            native.register_text(0, 3, ord('b'), 'b')
            additions = [registration(1, 98, 3)]
        query = torch.tensor(q).reshape(1, 1, 1, 2)
        key = keys[:visible].reshape(1, 1, visible, 2)
        native.begin_step()
        native.observe_attention(query, key, cache_position=torch.tensor([visible - 1]))
        native.observe_row(0, 5 + step, 1)  # First five acoustic positions belong to warmup.
        full = torch.softmax(torch.tensor(q) @ keys[:visible].T / math.sqrt(2), dim=-1)
        subset = full[[1] if step < 2 else [1, 3]].tolist()
        decoder.update(observation(step, subset, additions))
    final = decoder.finalize(0.32)
    expected = native.snapshots(finalize=True)[0]
    for record in expected['timestamps']:
        actual = final['token_timestamps'][str(record['token_index'])]
        assert actual['token_id'] == record['token_id']
        assert actual['start'] == pytest.approx(record['start'])
        assert actual['end'] == pytest.approx(record['end'])
        assert actual['confidence'] == pytest.approx(record['confidence'])
    assert [f['token_id'] for f in final['frames']] == [f['token_id'] for f in expected['frames']]
    assert [f['text_key_position'] for f in final['frames']] == [f['text_key_position'] for f in expected['frames']]


def test_packet_roundtrip_and_cumulative_duplicate_detection(model, tokenizer):
    packet = observation(0, [0.8], [registration(0, 97)])
    value = pack_alignment_packet(packet)
    output = SimpleNamespace(multimodal_output={'alignment': [value]})
    assert unpack_alignment_packets(output) == [packet]
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'a')
    assert decoder.update(packet) is not None
    assert decoder.update(packet) is None
    assert len(decoder.frames) == 1
    with pytest.raises(ValueError, match='lost or reordered'):
        decoder.update(observation(2, [0.8]))


def test_corrupt_exported_runtime_is_rejected(model):
    runtime = model / 'attention_timestamp_runtime.py'
    runtime.write_text(runtime.read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='SHA256'):
        load_timestamp_runtime(model)


def test_truncated_wire_packet_is_rejected():
    value = pack_alignment_packet({'text': 'Megan 🌟'})
    with pytest.raises(ValueError, match='Truncated'):
        unpack_alignment_packets(SimpleNamespace(multimodal_output={'alignment': value[:-1]}))


def test_appended_chunks_preserve_character_offsets_and_whitespace(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer)
    decoder.append_text('a ')
    decoder.append_text('é', expected_ids=[ord('é')])
    assert decoder.input_ids == [97, 32, 233]
    assert decoder.offsets == [(0, 1), (1, 2), (2, 3)]
    with pytest.raises(ValueError, match='actual supplied'):
        decoder.append_text('x', expected_ids=[12])


def test_actual_exported_tokenizer_offsets_roundtrip():
    model = os.environ.get('EASYMAGPIE_TEST_MODEL')
    if not model:
        pytest.skip('Set EASYMAGPIE_TEST_MODEL for the real converted-tokenizer regression test')
    tokenizer = EasyMagpieTextTokenizer.from_pretrained(model)
    for text in ['  Hello, hello!  ', 'Meet at 10:30; $42.50.', 'café 中文 🌟', 'go <bop>lɛft<eop> now']:
        ids, offsets = tokenizer.encode_with_offsets(text)
        assert ids == tokenizer.encode(text)
        assert len(ids) == len(offsets)
        assert all(0 <= a <= b <= len(text) for a, b in offsets)


def test_shared_byte_offsets_do_not_drop_token_occurrences(model):
    class ByteTokenizer:
        base_tokenizer = CharacterTokenizer()

        def encode_with_offsets(self, text):
            assert text == 'é'
            return [7, 8], [(0, 1), (0, 1)]

    decoder = StreamingAlignmentDecoder(model, ByteTokenizer(), 'é')
    decoder.update(observation(0, [0.9, 0.01], [registration(0, 7), registration(1, 8)]))
    decoder.update(observation(1, [0.01, 0.9]))
    final = decoder.finalize(0.16)
    assert final['character_timestamps']['0']['token_indices'] == [0, 1]
    assert final['character_timestamps']['0']['token_ids'] == [7, 8]
    assert len(final['token_timestamps']) == 2


def test_audio_clock_cannot_be_rewritten_after_streaming(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'a')
    decoder.update(observation(0, [0.9], [registration(0, 97)]))
    with pytest.raises(ValueError, match='shorter than the observed'):
        decoder.finalize(0.01)


def test_transport_alignment_payload_is_a_drained_client_tensor():
    from easymagpie_vllm_omni.timestamp_observer import patch_alignment_transport
    from vllm_omni.engine.output_modality import DRAINABLE_MODALITIES
    from vllm_omni.utils.mm_outputs import partition_flat_payload, to_payload_element

    patch_alignment_transport()
    a, b = pack_alignment_packet({'request': 'a'}), pack_alignment_packet({'request': 'b'})
    chosen = to_payload_element([a, b], idx=1, start=9, end=10, seq_len=10)
    inter_stage, client = partition_flat_payload({'alignment': chosen})
    assert not inter_stage and torch.equal(client['alignment'], b)
    assert 'alignment' in DRAINABLE_MODALITIES


@pytest.mark.asyncio
@pytest.mark.parametrize('debug', [False, True])
async def test_aligned_http_preserves_sampling_and_streams_audio_with_final_maps(model, tokenizer, debug):
    from easymagpie_vllm_omni.serving_timestamps import create_aligned_speech
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    seen = {}

    class Adapter:
        def validate(self, request):
            assert request.extra_params['return_timestamps']

        def _model_path(self):
            return model

        def _model_tokenizer(self):
            return tokenizer

        async def build(self, request, params, inline):
            seen['request'] = request
            seen['params'] = params
            return SimpleNamespace(prompt={'additional_information': dict(request.extra_params)})

    class Engine:
        default_sampling_params_list = [SamplingParams(temperature=0.0, max_tokens=64)]

        async def generate(self, **kwargs):
            seen['generate'] = kwargs
            packet = observation(0, [0.9], [registration(0, ord('a'))])
            yield SimpleNamespace(
                multimodal_output={
                    'alignment': pack_alignment_packet(packet),
                    'audio': torch.ones(2560) * 0.1,
                    'sr': 32000,
                }
            )

        async def abort(self, request_id):
            seen['abort'] = request_id

    service = SimpleNamespace(_tts_model_type='easymagpie', _get_tts_adapter=lambda: Adapter(), engine_client=Engine())

    async def connected():
        return False

    raw = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(openai_serving_speech=service)), is_disconnected=connected
    )
    extra = {'temperature': 0.7, 'top_k': 80}
    if debug:
        extra['timestamp_debug'] = True
    request = OpenAICreateSpeechRequest(input='ab', voice='eng', max_new_tokens=32, extra_params=extra)
    response = await create_aligned_speech(request, raw)
    events = [json.loads(line) async for line in response.body_iterator]
    types = [event['type'] for event in events]
    assert types == ['speech.start', 'alignment.delta', 'audio.chunk', 'alignment.done', 'speech.done']
    assert seen['request'].extra_params == dict(extra, return_timestamps=True)
    assert ('attention_observation' in events[1]) == debug
    assert seen['params'][0].max_tokens == 32
    assert seen['params'][0].output_kind == RequestOutputKind.DELTA
    assert Engine.default_sampling_params_list[0].max_tokens == 64
    assert events[2]['sample_offset'] == 0 and events[2]['sample_rate'] == 32000
    assert events[3]['audio_samples'] == 2560 and events[3]['input_text'] == 'ab'
    assert events[3]['token_timestamps']['1']['start'] is None
    assert 'abort' not in seen


@pytest.mark.asyncio
async def test_aligned_http_rejects_speed_adjustment_before_generation():
    from easymagpie_vllm_omni.serving_timestamps import create_aligned_speech
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    service = SimpleNamespace(_tts_model_type='easymagpie')
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(openai_serving_speech=service)))
    response = await create_aligned_speech(OpenAICreateSpeechRequest(input='test', speed=1.5), raw)
    assert response.status_code == 400
    assert 'speed=1.0' in json.loads(response.body)['error']


def test_character_updates_visit_only_changed_token_characters(model, tokenizer):
    decoder = StreamingAlignmentDecoder(model, tokenizer, 'ab' * 2000)
    decoder.update(observation(0, [0.9], [registration(0, 97)]))
    assert list(decoder._characters({0})) == ['0']


def test_selected_gqa_head_observation_keeps_attention_output_and_rng_unchanged(model):
    from easymagpie_vllm_omni.timestamp_observer import V2TimestampObserver

    class AttentionOperation(torch.nn.Module):
        def forward(self, query, key, value):
            return query + 1

    class NemotronHAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads, self.num_kv_heads, self.head_dim = 6, 2, 4
            self.scaling = 0.5
            self.attn = AttentionOperation()

    class Layer(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.mixer = NemotronHAttention() if attention else torch.nn.Identity()

    backbone = torch.nn.Module()
    backbone.layers = torch.nn.ModuleList([Layer(False), Layer(True), Layer(False), Layer(True), Layer(True)])
    config = json.loads((model / 'config.json').read_text())
    config['timestamp_config']['head'] = 4
    (model / 'config.json').write_text(json.dumps(config))
    (model / 'codec_native').mkdir()
    (model / 'codec_native/config.json').write_text(json.dumps({'frame_stacking_factor': 1}))
    fake = SimpleNamespace(
        arch=SimpleNamespace(
            acoustic_predictor_type='acoustic_codes_predictor', frame_stacking_factor=1, num_audio_codebooks=2
        ),
        backbone=backbone,
        model_path=str(model),
        text_eos_id=131073,
        speech_delay=1,
        _combined_embeddings=torch.zeros(16, 24),
        _out_codes=torch.full((16, 2), 3),
        audio_eos_id=4097,
        _forward_tensors=lambda *args, **kwargs: None,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(max_model_len=16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
    )
    observer = V2TimestampObserver(fake, vllm_config)
    assert observer.block_index == 3 and observer.kv_head == 1
    info = {
        'request_id': 'a',
        'return_timestamps': True,
        'speaker_id': 'eng',
        '_omni_is_prefill': True,
        '_omni_num_computed_tokens': 0,
        '_omni_prompt_len': 2,
        'text_prefill_num': 1,
        'prefill_text_tokens': [97],
        'text_tokens': [97, 98],
    }
    observer.register_query(torch.zeros(2, dtype=torch.long), info, {}, 0)
    query = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 100
    key = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 100
    before = torch.random.get_rng_state()
    output = backbone.layers[3].mixer.attn(query, key, key)
    observer.finish_forward()
    assert torch.equal(output, query + 1) and torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(observer._key[:2], key.reshape(2, 2, 4)[:, 1])
    info.update(_omni_is_prefill=False, _omni_num_computed_tokens=2, decode_offset=1)
    observer.register_query(torch.zeros(1, dtype=torch.long), info, {}, 0)
    backbone.layers[3].mixer.attn(query[:1], key[:1], key[:1])
    observer.finish_forward()
    packet = unpack_alignment_packets(SimpleNamespace(multimodal_output={'alignment': observer.output_packets[0]}))[0]
    assert [r['token_id'] for r in packet['registrations']] == [97, 98]
    expected_q = query[:1].reshape(1, 6, 4)[:, 4]
    expected_k = torch.cat([key.reshape(2, 2, 4)[:, 1], key[:1].reshape(1, 2, 4)[:, 1]])
    expected = torch.softmax((expected_q @ expected_k.T).reshape(-1) * 0.5, dim=-1)[[1, 2]]
    assert packet['probabilities'] == pytest.approx(expected.tolist())
    assert torch.equal(before, torch.random.get_rng_state())
    # The async scheduler may have one more forward queued after sampled EOS.
    # Keep a closed tombstone until finished_req_ids, rather than restarting a
    # request at a nonzero cache offset or emitting a second terminal packet.
    info.update(_omni_num_computed_tokens=3, decode_offset=2)
    observer.register_query(torch.zeros(1, dtype=torch.long), info, {}, 0)
    fake._out_codes[0].fill_(fake.audio_eos_id)
    backbone.layers[3].mixer.attn(query[:1], key[:1], key[:1])
    observer.finish_forward()
    terminal = unpack_alignment_packets(SimpleNamespace(multimodal_output={'alignment': observer.output_packets[0]}))[
        0
    ]
    assert terminal['audio_eos'] and terminal['valid_codec_frames'] == 0
    assert observer._states['a'].closed
    info.update(_omni_num_computed_tokens=4, decode_offset=3)
    observer.register_query(torch.zeros(1, dtype=torch.long), info, {}, 0)
    backbone.layers[3].mixer.attn(query[:1], key[:1], key[:1])
    observer.finish_forward()
    assert observer.output_packets[0].numel() == 0
    observer.discard_requests(['a'])
    assert not observer._states


@pytest.mark.asyncio
@pytest.mark.parametrize('disconnected', [False, True])
async def test_aligned_http_closes_generator_and_aborts_failed_or_disconnected_requests(
    model, tokenizer, disconnected
):
    from easymagpie_vllm_omni.serving_timestamps import create_aligned_speech
    from vllm.sampling_params import SamplingParams
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    seen = {}

    class Adapter:
        def validate(self, request):
            return None

        def _model_path(self):
            return model

        def _model_tokenizer(self):
            return tokenizer

        async def build(self, *_args):
            return SimpleNamespace(prompt={})

    class Engine:
        default_sampling_params_list = [SamplingParams(max_tokens=64)]

        async def generate(self, **_kwargs):
            try:
                if not disconnected:
                    raise RuntimeError('synthetic generation failure')
                yield SimpleNamespace(multimodal_output={})
            finally:
                seen['generator_closed'] = True

        async def abort(self, request_id):
            seen['abort'] = request_id
            # Cleanup failure must not mask the original stream error.
            raise RuntimeError('synthetic cleanup failure')

    async def is_disconnected():
        return disconnected

    service = SimpleNamespace(_tts_model_type='easymagpie', _get_tts_adapter=lambda: Adapter(), engine_client=Engine())
    raw = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(openai_serving_speech=service)), is_disconnected=is_disconnected
    )
    response = await create_aligned_speech(OpenAICreateSpeechRequest(input='a', voice='eng'), raw)
    events = [json.loads(line) async for line in response.body_iterator]
    assert seen['generator_closed'] and seen['abort'].startswith('speech-aligned-')
    assert events[0]['type'] == 'speech.start'
    assert all(event['type'] != 'speech.done' for event in events)
    if disconnected:
        assert len(events) == 1
    else:
        assert events[-1]['type'] == 'error' and events[-1]['message'] == 'synthetic generation failure'
