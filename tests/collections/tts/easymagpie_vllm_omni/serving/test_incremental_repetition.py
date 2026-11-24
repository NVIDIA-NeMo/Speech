# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for incremental BPE, explicit flush and timestamp input offsets."""
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer

pytestmark = pytest.mark.unit

TEXT = "Apple was founded in 1976 by Steve Jobs. It's known for the iPhone and Apple Music."


@pytest.fixture(scope='module')
def base_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [TEXT, 'Thanks for calling, we will call shortly.', 'Hello.\nNext line!\r\nOne  two\tthree.', 'café 中文 🌟'],
        trainers.BpeTrainer(
            vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[UNK]']
        ),
    )

    class BaseTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return tokenizer.encode(text, add_special_tokens=False).ids

        def __call__(self, text, **_kwargs):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            return {'input_ids': encoded.ids, 'offset_mapping': encoded.offsets}

        def decode(self, ids, **_kwargs):
            return tokenizer.decode(ids)

        def get_vocab(self):
            return tokenizer.get_vocab()

    return BaseTokenizer()


def tokenizer_with_phonemes(base, enabled):
    class Phonemes:
        def get_vocab(self):
            return {f'p{i}': i for i in range(5)}

        def encode(self, text):
            positions = [(index, char) for index, char in enumerate(text) if not char.isspace()]
            return SimpleNamespace(
                ids=[ord(char) - ord('a') for _, char in positions],
                offsets=[(index, index + 1) for index, _ in positions],
            )

    offset = len(base.get_vocab())
    return EasyMagpieTextTokenizer(
        base,
        text_vocab_size=offset + (8 if enabled else 0),
        phoneme_tokenizer=Phonemes() if enabled else None,
        text_phoneme_token_offset=offset if enabled else None,
        text_phoneme_vocab_size=8 if enabled else 0,
    )


@pytest.mark.parametrize('phonemes', [False, True])
@pytest.mark.parametrize('chunking', ['characters', 'words', 'subwords'])
def test_tokenization_is_chunk_invariant(base_tokenizer, phonemes, chunking):
    tokenizer = tokenizer_with_phonemes(base_tokenizer, phonemes)
    chunks = {
        'characters': list(TEXT),
        'words': re.findall(r'\S+\s*', TEXT),
        'subwords': [TEXT[:3], TEXT[3:17], TEXT[17:43], TEXT[43:]],
    }[chunking]
    encoder = tokenizer.incremental_encoder()
    actual = [token for chunk in chunks for token in encoder.push(chunk)] + encoder.finish()
    assert actual == tokenizer.encode(TEXT)


@pytest.mark.parametrize('phonemes', [False, True])
@pytest.mark.parametrize('text', [TEXT, 'café 中文 🌟', 'go <bop> ab <eop> now'])
def test_emitted_text_preserves_actual_ids_and_offsets(base_tokenizer, phonemes, text):
    tokenizer = tokenizer_with_phonemes(base_tokenizer, phonemes)
    encoder = tokenizer.incremental_encoder()
    committed, actual = [], []
    for chunk in [*text, None]:
        ids = encoder.finish() if chunk is None else encoder.push(chunk)
        emitted = encoder.emitted_text
        committed.append(emitted)
        actual.extend(ids)
        expected, offsets = tokenizer.encode_with_offsets(emitted)
        assert ids == expected
        assert len(ids) == len(offsets)
    assert ''.join(committed) == text
    assert actual == tokenizer.encode(text)


@pytest.mark.parametrize('phonemes', [False, True])
def test_flush_keeps_trailing_whitespace_for_next_sentence(base_tokenizer, phonemes):
    tokenizer = tokenizer_with_phonemes(base_tokenizer, phonemes)
    encoder = tokenizer.incremental_encoder()
    actual = []
    emitted = []
    for sentence in re.findall(r'.*?[.!?](?:\s+|$)', TEXT):
        for word in re.findall(r'\S+\s*', sentence):
            actual.extend(encoder.push(word))
            emitted.append(encoder.emitted_text)
        actual.extend(encoder.flush())
        emitted.append(encoder.emitted_text)
        assert encoder.flush() == []
        assert encoder.emitted_text == ''
    actual.extend(encoder.finish())
    emitted.append(encoder.emitted_text)
    assert ''.join(emitted) == TEXT
    assert actual == tokenizer.encode(TEXT)


@pytest.mark.asyncio
@pytest.mark.parametrize('timestamps', [False, True])
@pytest.mark.parametrize('mode', ['word_flush', 'single', 'characters', 'raw_tokens'])
async def test_handler_preserves_text_ids_with_timestamps_and_flush(
    monkeypatch, tmp_path, base_tokenizer, timestamps, mode
):
    from easymagpie_vllm_omni import serving_stream
    from easymagpie_vllm_omni import timestamps as timestamp_module
    from vllm.sampling_params import SamplingParams

    tokenizer = tokenizer_with_phonemes(base_tokenizer, True)
    native = Path(__file__).resolve().parents[5] / 'nemo/collections/tts/modules/attention_alignment.py'
    config = timestamp_module.bundle_timestamp_runtime(tmp_path, native)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    decoder = timestamp_module.StreamingAlignmentDecoder(tmp_path, tokenizer)
    monkeypatch.setattr(timestamp_module, 'StreamingAlignmentDecoder', lambda *_args, **_kwargs: decoder)
    if mode == 'single':
        messages = [{'type': 'input.text', 'text': TEXT}]
    elif mode == 'characters':
        messages = [{'type': 'input.text', 'text': char} for char in TEXT]
    elif mode == 'raw_tokens':
        messages = [
            {'type': 'input.text', 'text': 'Apple'},
            {'type': 'input.tokens', 'tokens': tokenizer.encode(' Music.')},
        ]
    else:
        messages = []
        for sentence in re.findall(r'.*?[.!?](?:\s+|$)', TEXT):
            messages.extend({'type': 'input.text', 'text': word} for word in re.findall(r'\S+\s*', sentence))
            messages.append({'type': 'input.flush'})

    class WebSocket:
        def __init__(self):
            self.received = iter(
                [
                    {
                        'type': 'session.config',
                        'voice': 'eng',
                        'stream_audio': True,
                        'response_format': 'pcm',
                        'word_timestamps': timestamps,
                    },
                    *messages,
                    {'type': 'input.done'},
                ]
            )
            self.sent = []

        async def accept(self):
            pass

        async def receive_text(self):
            return json.dumps(next(self.received))

        async def send_json(self, value):
            self.sent.append(value)

        async def send_bytes(self, _value):
            pass

    supplied = []

    class Engine:
        default_sampling_params_list = [SamplingParams(max_tokens=2048), SamplingParams(max_tokens=2048)]

        def generate(self, *, prompt, **_kwargs):
            async def outputs():
                async for chunk in prompt:
                    supplied.extend(chunk.prompt['additional_information'].get('text_token', []))
                    yield SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[1], finish_reason='length')])

            return outputs()

        async def abort(self, _request_id):
            pass

    class Adapter:
        def build_streaming_spec(self, _request):
            return SimpleNamespace(
                tokenizer=tokenizer,
                prefill_prompt={'prompt_token_ids': [0], 'additional_information': {}},
                text_eos_id=999999,
                sample_rate=32000,
            )

        def timestamps_enabled(self):
            return True

        def _model_path(self):
            return tmp_path

    class Service:
        _tts_model_type = 'easymagpie'
        engine_client = Engine()

        def _get_tts_adapter(self):
            return Adapter()

        async def _generate_pcm_chunks(self, generator, _request_id):
            async for _output in generator:
                yield b'\x01\x02'

    async def pcm(generator, _sample_rate):
        async for _output in generator:
            yield b'\x01\x02'

    monkeypatch.setattr(serving_stream, '_alignment_pcm_deltas', pcm)
    socket_ = WebSocket()
    await asyncio.wait_for(
        serving_stream.EasyMagpieStreamingSpeechHandler(Service()).handle_session(socket_), timeout=5
    )
    errors = [value for value in socket_.sent if value['type'] == 'error']
    assert not errors, errors
    expected_text = 'Apple Music.' if mode == 'raw_tokens' else TEXT
    assert supplied == tokenizer.encode(expected_text) + [999999]
    assert socket_.sent[-1]['type'] == 'session.done'
    if timestamps:
        final = next(value for value in socket_.sent if value['type'] == 'alignment.done')
        assert final['input_text'] == expected_text
        assert decoder.input_ids == tokenizer.encode(expected_text)
        assert len(final['character_timestamps']) == len(expected_text)
        assert decoder.offsets == tokenizer.encode_with_offsets(expected_text)[1]
