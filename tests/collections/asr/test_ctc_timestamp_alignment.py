# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import itertools
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    MultiSpeakerSOTWordTimestampAligner,
    _banded_ctc_search,
    _character_token_ids,
    _refine_word_timestamps,
    _timestamp_units,
    align_prepared_batch,
)
from tests.collections.asr.ctc_timestamp_test_utils import search_streams


@pytest.mark.unit
@pytest.mark.parametrize(
    'text,expected',
    [
        ('你好。', ['你', '好']),
        ('こんにちは', list('こんにちは')),
        ('한국어 단어', ['한국어', '단어']),
        ('हिन्दी भाषा', ['हिन्दी', 'भाषा']),
        ('أَهْلًا بالعالم', ['أَهْلًا', 'بالعالم']),
    ],
)
def test_auto_timestamp_units_without_dataset_language(text, expected):
    words = MultiSpeakerSOTWordTimestampAligner.parse_sot_words(text)
    assert [w['word'] for w in _timestamp_units(words, 'auto')] == expected


@pytest.mark.unit
def test_inference_cleanup_keeps_acoustic_tags():
    text = '<s> <spk:0> <sty:flat> 你好 <ev:laughter> </s>'
    assert (
        MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript(
            text, ['<s>', '</s>', '<spk:0>', '<sty:flat>', '<ev:laughter>']
        )
        == '<spk:0> <sty:flat> 你好 <ev:laughter>'
    )


@pytest.mark.unit
def test_normal_deferred_generation_automatically_selects_units_per_record(monkeypatch):
    import inspect

    from nemo.collections.asr.modules.parallel_expert_encoder import CTCTimestampInputs, ParallelExpertEncoder
    from nemo.collections.speechlm2.parts import ctc_timestamp_utils

    class Tokenizer:
        tokenizer = SimpleNamespace(
            piece_to_id=lambda text: {'你': 1, '好': 2}[text],
            unk_id=lambda: 0,
            decode=lambda ids: ''.join({1: '你', 2: '好'}[i] for i in ids),
        )

        def text_to_ids(self, text):
            return [1]

    class Head(torch.nn.Module):
        num_classes_with_blank = 4

        def forward(self, encoder_output, encoded_lengths):
            logits = torch.full((encoder_output.shape[0], encoder_output.shape[-1], 4), -10.0)
            logits[:, :, 3] = 0
            logits[:, 2, 1] = 10
            logits[:, 4, 2] = 10
            return torch.log_softmax(logits, -1)

    aligner = MultiSpeakerSOTWordTimestampAligner(tokenizer=Tokenizer(), ctc_decoder=Head(), blank_id=3)
    monkeypatch.setattr(ctc_timestamp_utils, 'get_ctc_timestamp_aligner', lambda *args: aligner)
    inputs = CTCTimestampInputs(torch.zeros(3, 1, 8), torch.full((3,), 8), torch.ones(3, 8, 1), torch.full((3,), 8))
    result = inspect.unwrap(ParallelExpertEncoder).generate_ctc_timestamps(
        SimpleNamespace(ctc_timestamp_model_path='unused'), inputs, ['你好', 'hello', '한국어'], [0.64] * 3
    )
    assert [[w['word'] for w in r['speaker_word_timestamps'][0]] for r in result] == [
        ['你', '好'],
        ['hello'],
        ['한국어'],
    ]
    assert all(r['word_timestamp_refinement']['method'] == 'diarization_boundary_refinement' for r in result)
    assert aligner.timestamp_language is None


@pytest.mark.unit
@pytest.mark.parametrize('language', ['zh', 'ja'])
def test_timestamp_character_units_are_instance_local(language):
    class Processor:
        def piece_to_id(self, text):
            return {'你': 1, '好': 2}[text]

        def unk_id(self):
            return 0

        def decode(self, ids):
            return ''.join({1: '你', 2: '好'}[i] for i in ids)

    aligner = MultiSpeakerSOTWordTimestampAligner(
        tokenizer=SimpleNamespace(tokenizer=Processor()), blank_id=3, timestamp_language=language
    )
    prepared = aligner.prepare_alignment(
        torch.log_softmax(torch.zeros(1, 8, 4), -1), None, ['<spk:0> 你好。'], audio_durations=[0.64]
    )
    assert [w['word'] for w in prepared['records'][0]['words']] == ['你', '好']
    assert [w['token_ids'] for w in prepared['records'][0]['words']] == [[1], [2]]
    assert prepared['config']['timestamp_language'] == language
    assert MultiSpeakerSOTWordTimestampAligner.parse_sot_words('你好。')[0]['word'] == '你好。'
    assert align_prepared_batch(prepared) == aligner.align_prepared(prepared)


@pytest.mark.unit
@pytest.mark.parametrize('language,text', [('hi', 'हिन्दी भाषा'), ('ko', '한국어 단어'), ('ar', 'أَهْلًا بالعالم')])
def test_timestamp_word_languages_preserve_acoustic_spelling(language, text):
    class Tokenizer:
        def text_to_ids(self, text):
            return [1] * len(text.split())

    aligner = MultiSpeakerSOTWordTimestampAligner(tokenizer=Tokenizer(), blank_id=2, timestamp_language=language)
    prepared = aligner.prepare_alignment(torch.zeros(1, 8, 3), None, [text], audio_durations=[0.64])
    assert [w['word'] for w in prepared['records'][0]['words']] == text.split()


@pytest.mark.unit
def test_timestamp_units_preserve_tags_and_turns():
    words = MultiSpeakerSOTWordTimestampAligner.parse_sot_words('<spk:0> 你好 <ev:laughter> <sty:flat> <spk:1> 好')
    units = _timestamp_units(words, 'zh')
    assert [w['word'] for w in units] == ['你', '好', '<ev:laughter>', '<sty:flat>', '好']
    assert [w['word_index'] for w in units] == list(range(5))
    assert [w['speaker_tag'] for w in units] == [0, 0, 0, 0, 1]
    assert units[-1]['turn_index'] != units[0]['turn_index']


@pytest.mark.unit
@pytest.mark.parametrize('language', ['th', 'auto'])
def test_thai_units_use_newmm_without_crossing_turns(language):
    pytest.importorskip('pythainlp')
    words = MultiSpeakerSOTWordTimestampAligner.parse_sot_words('<spk:0> ฉัน รักคุณ <ev:laughter> <spk:1> ฉันรักคุณ')
    units = _timestamp_units(words, language)
    assert [w['word'] for w in units] == ['ฉัน', 'รัก', 'คุณ', '<ev:laughter>', 'ฉัน', 'รัก', 'คุณ']
    assert [w['speaker_tag'] for w in units] == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.unit
def test_character_tokenization_rejects_unknown_characters():
    processor = SimpleNamespace(
        piece_to_id=lambda _: 0, unk_id=lambda: 0, encode=lambda _: [0], id_to_piece=lambda _: '?'
    )
    with pytest.raises(ValueError, match='Unrepresentable character'):
        _character_token_ids(processor, '你', 3)


@pytest.mark.unit
@pytest.mark.parametrize("target", [[1, 2], [1, 1]])
@pytest.mark.parametrize("use_prior", [False, True])
def test_banded_ctc_full_band_matches_exhaustive_paths(target, use_prior):
    rng = np.random.default_rng(71)
    labels = np.array([0, target[0], 0, target[1], 0])
    for frames in range(3, 7):
        logp = rng.normal(size=(frames, 3)).astype(np.float32)
        prior = rng.uniform(-2, 0, frames).astype(np.float32) if use_prior else None
        path, score, _ = _banded_ctc_search(logp, labels, np.arange(frames), prior)
        best = -np.inf
        for states in itertools.product(range(5), repeat=frames):
            if states[0] not in (0, 1) or states[-1] not in (3, 4):
                continue
            if any(
                b - a not in (0, 1, 2) or (b - a == 2 and (labels[b] == 0 or labels[a] == labels[b]))
                for a, b in zip(states, states[1:])
            ):
                continue
            value = sum(
                float(logp[t, labels[s]]) + (float(prior[t]) if prior is not None and labels[s] else 0)
                for t, s in enumerate(states)
            )
            best = max(best, value)
        assert score == pytest.approx(best, abs=1e-5)
        assert path[0] in (0, 1) and path[-1] in (3, 4)


@pytest.mark.unit
def test_banded_ctc_refinement_caps_words_without_mutating_input():
    result = {
        "ctc_frame_seconds": 0.08,
        "speaker_tag_to_sortformer_column": {"0": 2},
        "speaker_word_timestamps": {
            "0": [{"word": str(i), "speaker": 0, "start": s, "end": s + 0.08} for i, s in enumerate([0.4, 0.8, 1.8])]
        },
        "diarization_timestamps": [{"speaker": 2, "start": 0.2, "end": 2.5}],
    }
    saved = copy.deepcopy(result)
    output = _refine_word_timestamps(result)
    assert result == saved
    assert [(w['start'], w['end']) for w in output['speaker_word_timestamps']['0']] == [
        (0.2, 0.8),
        (0.8, 1.76),
        (1.8, 2.5),
    ]
    assert output['word_timestamp_refinement']['onsets_moved'] == 1
    result['speaker_tag_to_sortformer_column'] = {}
    assert _refine_word_timestamps(result)['word_timestamp_refinement']['unmatched_words'] == 3


@pytest.mark.unit
def test_banded_ctc_serialized_worker_config_and_empty_support():
    class Tokenizer:
        def text_to_ids(self, text):
            return [1]

    aligner = MultiSpeakerSOTWordTimestampAligner(tokenizer=Tokenizer(), blank_id=2, max_word_frames=6)
    prepared = aligner.prepare_alignment(
        torch.log_softmax(torch.zeros(1, 8, 3), -1), None, ["hello"], ctc_lengths=[8], audio_durations=[0.64]
    )
    saved = copy.deepcopy(prepared)
    expected = aligner.align_prepared(prepared)
    assert align_prepared_batch(prepared) == expected
    assert expected[0]['word_timestamp_refinement']['max_frames'] == 6
    assert torch.equal(saved['records'][0]['ctc_log_probs'], prepared['records'][0]['ctc_log_probs'])
    path, score, starts = _banded_ctc_search(np.zeros((0, 3)), np.array([0, 1, 0]), np.array([], int), None)
    assert not len(path) and not len(starts) and score == -np.inf


@pytest.mark.unit
def test_banded_ctc_anchor_band_and_greedy_cache_agree():
    # Enough states/frames to exercise a real band, not the short-clip full-width path.
    labels = np.zeros(1201, dtype=np.int64)
    labels[1::2] = np.arange(1, 601)
    positions = np.arange(1201)
    logp = np.full((1201, 601), -20, dtype=np.float32)
    logp[positions, labels] = 0
    uncached = _banded_ctc_search(logp, labels, positions, None)
    cached = _banded_ctc_search(logp, labels, positions, None, logp.argmax(axis=1))
    assert np.array_equal(uncached[0], cached[0])
    assert uncached[1] == cached[1] == 0
    assert np.array_equal(uncached[2], cached[2])
    assert uncached[2][-1] > 0
    assert np.all(np.isin(np.diff(uncached[2]), [0, 1, 2]))


@pytest.mark.unit
@pytest.mark.parametrize("prior_dtype", [None, np.float32, np.float64])
@pytest.mark.parametrize("frames,tokens", [(40, 6), (1500, 600)])  # full band; a band narrower than the target
def test_compiled_banded_forward_matches_the_numpy_loop(prior_dtype, frames, tokens):
    from nemo.collections.speechlm2.parts import ctc_timestamp_utils as utils

    if not utils.NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed")
    rng = np.random.default_rng(frames + tokens)
    columns = 12
    logp = np.log(rng.dirichlet(np.ones(columns), size=frames)).astype(np.float32)
    # Ties between predecessors exercise the stay, step, skip preference; repeated tokens disable skips.
    logp[rng.random(logp.shape) < 0.2] = -2.0
    target = rng.integers(1, columns, size=tokens)
    target[1::7] = target[::7][: len(target[1::7])]
    labels = np.zeros(2 * tokens + 1, dtype=np.int64)
    labels[1::2] = target
    positions = np.sort(rng.choice(frames + 50, size=frames, replace=False))
    logp = np.concatenate([logp, np.log(rng.dirichlet(np.ones(columns), size=50)).astype(np.float32)])
    prior = None if prior_dtype is None else np.log(rng.uniform(0.01, 1.0, size=len(logp))).astype(prior_dtype)
    states = len(labels)
    half = states if frames < 1000 else min(states, int(np.ceil(0.10 * states + 100)))
    width = min(states, 2 * half + 1)
    starts = utils._ctc_anchor_band_starts(logp[positions].argmax(axis=1), target, states, half, width)

    compiled = utils._ctc_banded_forward(logp, labels, positions, prior, starts, width)
    loop = utils._ctc_banded_forward_numpy(logp, labels, positions, prior, starts, width)

    assert compiled[0].dtype == loop[0].dtype == np.float32
    assert np.array_equal(compiled[0], loop[0])
    assert np.array_equal(compiled[1], loop[1])


@pytest.mark.unit
def test_deferred_columns_stay_float32_while_another_thread_changes_the_default_dtype():
    # A vLLM server prepares on a thread of its own while the speech encoder, on the engine thread, sets the
    # process-wide default dtype to bfloat16 around its streaming steps.
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _read_deferred_ctc_columns

    class Decoder(torch.nn.Module):
        num_classes_with_blank = 8
        upsampling_factor = 1

        def forward(self, states, encoded_lengths):
            return states.transpose(1, 2).log_softmax(dim=-1)

    class Tokenizer:
        def text_to_ids(self, text):
            return []

    aligner = MultiSpeakerSOTWordTimestampAligner(ctc_decoder=Decoder(), tokenizer=Tokenizer(), blank_id=7)
    inputs = SimpleNamespace(asr_encoded=torch.randn(2, 8, 5), asr_encoded_lengths=torch.tensor([5, 3]))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        kept = _read_deferred_ctc_columns(aligner, inputs, 8, [[7, 1, 2], [7, 3]], [5, 3])
    finally:
        torch.set_default_dtype(previous)

    assert [record.dtype for record in kept] == [torch.float32, torch.float32]
    assert [tuple(record.shape) for record in kept] == [(5, 3), (3, 2)]


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deferred_head_never_decodes_a_batch_of_one_on_the_gpu():
    # torch.compile specializes the decoder's attention for a batch of one; a lone record is decoded twice instead, so
    # the CTC head never adds that variant to the compiled function it shares with the speech encoder.
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _read_deferred_ctc_columns

    class Decoder(torch.nn.Module):
        num_classes_with_blank = 8
        upsampling_factor = 1

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.batch_sizes = []

        def forward(self, states, encoded_lengths):
            self.batch_sizes.append(states.shape[0])
            return states.transpose(1, 2).log_softmax(dim=-1)

    class Tokenizer:
        def text_to_ids(self, text):
            return []

    decoder = Decoder().cuda()
    aligner = MultiSpeakerSOTWordTimestampAligner(ctc_decoder=decoder, tokenizer=Tokenizer(), blank_id=7)
    states = torch.randn(1, 8, 5, device="cuda")
    inputs = SimpleNamespace(asr_encoded=states, asr_encoded_lengths=torch.tensor([5], device="cuda"))

    (kept,) = _read_deferred_ctc_columns(aligner, inputs, 8, [[7, 1, 2]], [5])

    assert decoder.batch_sizes == [2]
    expected = states.transpose(1, 2).log_softmax(dim=-1)[0][:, [7, 1, 2]].cpu()
    assert torch.equal(kept, expected)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deferred_columns_reach_the_host_under_a_cuda_default_device():
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _read_deferred_ctc_columns

    class Decoder(torch.nn.Module):
        num_classes_with_blank = 8
        upsampling_factor = 1

        def forward(self, states, encoded_lengths):
            return states.transpose(1, 2).log_softmax(dim=-1)

    class Tokenizer:
        def text_to_ids(self, text):
            return []

    aligner = MultiSpeakerSOTWordTimestampAligner(ctc_decoder=Decoder(), tokenizer=Tokenizer(), blank_id=7)
    states = torch.randn(2, 8, 5, device="cuda")
    inputs = SimpleNamespace(asr_encoded=states, asr_encoded_lengths=torch.tensor([5, 3], device="cuda"))
    with torch.device("cuda"):
        kept = _read_deferred_ctc_columns(aligner, inputs, 8, [[7, 1, 2], [7, 3]], [5, 3])

    assert [record.device.type for record in kept] == ["cpu", "cpu"]
    expected = states.transpose(1, 2).log_softmax(dim=-1).cpu()
    assert torch.equal(kept[0], expected[0][:, [7, 1, 2]]) and torch.equal(kept[1], expected[1][:3, [7, 3]])


@pytest.mark.unit
@pytest.mark.parametrize("use_transformer", [True, False])
@pytest.mark.parametrize("upsampling_factor", [1, 2])
def test_split_transformer_ctc_head_matches_its_whole_forward(use_transformer, upsampling_factor):
    # The deferred head runs TransformerCTCDecoder.trunk on each window and classify on slices of its frames.
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import TransformerCTCDecoder

    class Tokenizer:
        def text_to_ids(self, text):
            return [{"a": 2, "b": 7}[word] for word in text.split()]

    torch.manual_seed(5)
    decoder = TransformerCTCDecoder(
        feat_in=32,
        num_classes=39,
        use_transformer=use_transformer,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        sync_max_audio_length=False,
        upsampling_factor=upsampling_factor,
    ).eval()
    # In float64, the two paths' rounding differences vanish when the kept columns are cast to float32.
    decoder.double()
    aligner = MultiSpeakerSOTWordTimestampAligner(
        ctc_decoder=decoder,
        tokenizer=Tokenizer(),
        blank_id=20,  # A blank other than the last column.
        online_inference_length=3,
        chunk_left_context=1,
        chunk_right_context=2,
    )
    inputs = SimpleNamespace(
        asr_encoded=torch.randn(3, 32, 7, dtype=torch.float64),
        asr_encoded_lengths=torch.tensor([7, 5, 2]),
        sortformer_sigmoids=torch.rand(3, 7, 2),
        sortformer_lengths=torch.tensor([7, 5, 2]),
        diarization_labels=None,
        diarization_lengths=None,
        diarization_frame_seconds=0.01,
    )
    transcripts = ["<spk:0> a a <spk:1> b", "b a", ""]
    durations = [0.7, 0.5, 0.2]
    with torch.no_grad():
        chunks = []
        for start in range(0, 7, 3):
            end = min(start + 3, 7)
            left, right = max(start - 1, 0), min(end + 2, 7)
            lengths = (inputs.asr_encoded_lengths - left).clamp(min=0, max=right - left)
            output = decoder(inputs.asr_encoded[:, :, left:right], lengths)
            chunks.append(output[:, (start - left) * upsampling_factor : (end - left) * upsampling_factor])
        expected = aligner.prepare_alignment(
            torch.cat(chunks, dim=1),
            inputs.sortformer_sigmoids,
            transcripts,
            ctc_lengths=inputs.asr_encoded_lengths * upsampling_factor,
            sortformer_lengths=inputs.sortformer_lengths,
            audio_durations=durations,
        )
        # Small enough that each window's frames reach classify in slices of two frames.
        aligner._CTC_WINDOW_MAX_ELEMENTS = 80
        actual = aligner.prepare_from_inputs(inputs, transcripts, durations)

    for record, reference in zip(actual["records"], expected["records"]):
        assert record.keys() == reference.keys()
        for key in record:
            if isinstance(record[key], torch.Tensor):
                torch.testing.assert_close(record[key], reference[key], rtol=0, atol=1e-6)
            else:
                assert record[key] == reference[key]
        assert record["ctc_log_probs"].dtype == torch.float32 and record["ctc_log_probs"].device.type == "cpu"
    assert aligner.align_prepared(actual) == aligner.align_prepared(expected)


@pytest.mark.unit
def test_banded_ctc_cannot_start_inside_an_infeasible_long_target():
    labels = np.zeros(10001, dtype=np.int64)
    labels[1::2] = 1
    path, score, _ = _banded_ctc_search(np.zeros((1000, 2), np.float32), labels, np.arange(1000), None)
    assert score == -np.inf
    assert np.all(path == -1)


@pytest.mark.unit
@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_banded_ctc_rejects_invalid_word_cap(cap):
    with pytest.raises(ValueError, match="max_word_frames"):
        MultiSpeakerSOTWordTimestampAligner(max_word_frames=cap)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["piece—just", "piece — just", "—piece——just—"])
def test_em_dash_separates_alignment_words(text):
    transcript = f"<spk:1> <sty:stt:flat> {text} <ev:laughter> <sty:end:flat>"
    words = MultiSpeakerSOTWordTimestampAligner.parse_sot_words(transcript)
    assert [w["word"] for w in words] == [
        "<sty:stt:flat>",
        "piece",
        "just",
        "<ev:laughter>",
        "<sty:end:flat>",
    ]
    assert all(w["speaker_tag"] == 1 for w in words)
    assert [w["word_index"] for w in words] == list(range(5))
    assert [w["word"] for w in MultiSpeakerSOTWordTimestampAligner.parse_sot_words("don't well-known")] == [
        "don't",
        "well-known",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("core_length", [0, 3, 7])
@pytest.mark.parametrize("speaker_weight", [0.0, 0.25])
@pytest.mark.parametrize("window_budget", [16 * 1024 * 1024, 768])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("upsampling_factor", [1, 2])
def test_deferred_windows_release_full_vocabulary_and_preserve_alignment(
    core_length, speaker_weight, window_budget, dtype, upsampling_factor
):
    class Decoder(torch.nn.Module):
        num_classes_with_blank = 128

        def __init__(self):
            super().__init__()
            self.outputs = []
            self.check_lifetime = False
            self.shapes = []
            self.upsampling_factor = upsampling_factor

        def forward(self, states, encoded_lengths):
            if self.check_lifetime:
                assert all(ref() is None for ref in self.outputs), "Previous full-vocabulary window is still live"
            self.shapes.append(states.shape)
            # Depend on context as well as the frame, so incorrect window slicing changes the oracle.
            logits = states.transpose(1, 2) + states.mean(dim=-1).unsqueeze(1)
            output = logits.repeat_interleave(upsampling_factor, dim=1).log_softmax(dim=-1)
            self.outputs.append(weakref.ref(output))
            return output

    class Tokenizer:
        def text_to_ids(self, text):
            return [{"a": 2, "b": 7}[word] for word in text.split()]

    torch.manual_seed(23)
    decoder = Decoder()
    aligner = MultiSpeakerSOTWordTimestampAligner(
        ctc_decoder=decoder,
        tokenizer=Tokenizer(),
        blank_id=126,  # Also exercise a blank other than the last vocabulary column.
        online_inference_length=core_length,
        chunk_left_context=1,
        chunk_right_context=2,
        speaker_logprob_weight=speaker_weight,
    )
    inputs = SimpleNamespace(
        asr_encoded=torch.randn(3, 128, 7).to(dtype),
        asr_encoded_lengths=torch.tensor([7, 5, 2]),
        sortformer_sigmoids=torch.rand(3, 7, 2),
        sortformer_lengths=torch.tensor([7, 5, 2]),
        diarization_labels=None,
        diarization_lengths=None,
        diarization_frame_seconds=0.01,
    )
    transcripts = ["<spk:0> a a <spk:1> b", "b a", ""]
    durations = [0.7, 0.5, 0.2]
    # Former full-vocabulary window/cat implementation, independent of the new iterator.
    chunks = []
    for start in range(0, 7, core_length or 7):
        end = min(start + (core_length or 7), 7)
        left, right = max(start - 1, 0), min(end + 2, 7)
        lengths = (inputs.asr_encoded_lengths - left).clamp(min=0, max=right - left)
        output = decoder(inputs.asr_encoded[:, :, left:right], lengths)
        chunks.append(output[:, (start - left) * upsampling_factor : (end - left) * upsampling_factor])
    dense = torch.cat(chunks, dim=1)
    del chunks, output
    expected = aligner.prepare_alignment(
        dense,
        inputs.sortformer_sigmoids,
        transcripts,
        ctc_lengths=inputs.asr_encoded_lengths * upsampling_factor,
        sortformer_lengths=inputs.sortformer_lengths,
        audio_durations=durations,
    )
    del dense
    decoder.outputs.clear()
    decoder.shapes.clear()
    decoder.check_lifetime = True
    aligner._CTC_WINDOW_MAX_ELEMENTS = window_budget
    actual = aligner.prepare_from_inputs(inputs, transcripts, durations)
    assert all(ref() is None for ref in decoder.outputs)
    assert all(
        batch * frames * upsampling_factor * classes <= window_budget or batch == 1
        for batch, classes, frames in decoder.shapes
    )
    assert actual["config"]["max_word_frames"] == 12 * upsampling_factor
    for record, reference in zip(actual["records"], expected["records"]):
        assert record.keys() == reference.keys()
        for key in record:
            if isinstance(record[key], torch.Tensor):
                torch.testing.assert_close(record[key], reference[key], rtol=0, atol=0)
            else:
                assert record[key] == reference[key]
        assert record["ctc_log_probs"].device.type == "cpu"
        assert record["ctc_log_probs"].shape[1] <= 3
    assert aligner.align_prepared(actual) == aligner.align_prepared(expected)


@pytest.mark.unit
def test_word_timestamp_alignment_defaults_to_speaker_prior():
    assert MultiSpeakerSOTWordTimestampAligner().speaker_logprob_weight == 1.0


@pytest.mark.unit
def test_legacy_alignment_backend_is_not_selectable():
    assert "alignment_backend" not in inspect.signature(MultiSpeakerSOTWordTimestampAligner).parameters


@pytest.mark.unit
@pytest.mark.parametrize("weight", [None, 0.0, 0.25, 1.0])
def test_default_pipeline_applies_configured_prior_and_always_refines(monkeypatch, weight):
    from nemo.collections.speechlm2.parts import ctc_timestamp_utils as utils

    class Tokenizer:
        def text_to_ids(self, text):
            return [1]

    aligner = MultiSpeakerSOTWordTimestampAligner(tokenizer=Tokenizer(), blank_id=2)
    logp = torch.log_softmax(torch.zeros(1, 8, 3), -1)
    prepared = aligner.prepare_alignment(
        logp,
        torch.full((1, 8, 1), 0.8),
        ["hello"],
        audio_durations=[0.64],
        speaker_logprob_weight=weight,
        diarization_labels=torch.ones(1, 1, 64, dtype=torch.bool),
    )
    priors = []
    search = utils._banded_ctc_search

    def capture(*args):
        priors.append(args[3].copy())
        return search(*args)

    monkeypatch.setattr(utils, '_banded_ctc_search', capture)
    result = aligner.align_prepared(prepared)[0]
    assert len(priors) == 1
    np.testing.assert_allclose(priors[0], (1.0 if weight is None else weight) * np.log(0.8), atol=1e-7)
    assert result['alignment_mode'] == 'banded'
    assert result['alignment_diagnostics']['one_pass'] is True
    assert result['word_timestamp_refinement']['method'] == 'diarization_boundary_refinement'
    assert result['word_timestamp_refinement']['max_frames'] == 12
    assert 'alignment_backend' not in prepared['config']


def _dense_reference(
    log_probs,
    labels,
    state_lengths,
    blank_id,
    state_speaker_columns,
    speaker_probs,
    speaker_logprob_weight,
    epsilon=1.0e-6,
):
    """Former dense implementation retained only as an exactness oracle."""
    num_streams, max_states = labels.shape
    num_frames = log_probs.shape[0]
    state_mask = torch.arange(max_states).unsqueeze(0) < state_lengths.unsqueeze(1)
    emissions = log_probs[:, labels].permute(1, 0, 2).contiguous()
    emissions.masked_fill_(~state_mask.unsqueeze(1), -float("inf"))
    token_states = state_mask & (state_speaker_columns >= 0)
    if speaker_probs is not None and speaker_logprob_weight > 0 and token_states.any():
        columns = state_speaker_columns.clamp_min(0)
        activity = speaker_probs[:, columns].permute(1, 0, 2)
        emissions += torch.where(
            token_states.unsqueeze(1),
            speaker_logprob_weight * torch.log(activity.clamp_min(epsilon)),
            torch.zeros_like(activity),
        )

    previous = torch.full((num_streams, max_states), -float("inf"))
    previous[:, 0] = emissions[:, 0, 0]
    previous[:, 1] = emissions[:, 0, 1]
    backpointers = torch.full((num_frames, num_streams, max_states), -1, dtype=torch.long)
    states = torch.arange(max_states).unsqueeze(0).expand(num_streams, -1)
    for frame in range(1, num_frames):
        best = previous
        previous_states = states
        advance = torch.full_like(previous, -float("inf"))
        advance[:, 1:] = previous[:, :-1]
        take = advance > best
        best = torch.where(take, advance, best)
        previous_states = torch.where(take, states - 1, previous_states)
        skip = torch.full_like(previous, -float("inf"))
        can_skip = state_mask[:, 2:] & (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
        skip[:, 2:] = torch.where(can_skip, previous[:, :-2], skip[:, 2:])
        take = skip > best
        best = torch.where(take, skip, best)
        previous_states = torch.where(take, states - 2, previous_states)
        previous = best + emissions[:, frame]
        previous.masked_fill_(~state_mask, -float("inf"))
        backpointers[frame] = previous_states

    last_blank = state_lengths - 1
    last_token = state_lengths - 2
    blank_scores = previous.gather(1, last_blank[:, None]).squeeze(1)
    token_scores = previous.gather(1, last_token[:, None]).squeeze(1)
    final_states = torch.where(token_scores > blank_scores, last_token, last_blank)
    final_scores = torch.maximum(token_scores, blank_scores)
    paths = torch.empty((num_streams, num_frames), dtype=torch.long)
    current = final_states
    stream_indices = torch.arange(num_streams)
    for frame in range(num_frames - 1, -1, -1):
        paths[:, frame] = current
        if frame:
            current = backpointers[frame, stream_indices, current]
    return list(paths), [float(score) for score in final_scores]


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 7, 19])
@pytest.mark.parametrize("speaker_weight", [0.0, 0.25])
def test_full_band_dp_matches_dense_reference(seed, speaker_weight):
    torch.manual_seed(seed)
    blank_id = 5
    labels = torch.tensor([[5, 0, 5, 1, 5, 1, 5], [5, 2, 5, 3, 5, 5, 5]])
    state_lengths = torch.tensor([7, 5])
    columns = torch.tensor([[-1, 0, -1, 0, -1, 0, -1], [-1, 1, -1, 1, -1, -1, -1]])
    log_probs = torch.log_softmax(torch.randn(15, blank_id + 1), dim=-1)
    speaker_probs = torch.sigmoid(torch.randn(15, 2))
    args = (
        log_probs,
        labels,
        state_lengths,
        blank_id,
        columns,
        speaker_probs,
        speaker_weight,
    )

    expected_paths, expected_scores = _dense_reference(*args)
    actual_paths, actual_scores = search_streams(*args)

    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_paths, expected_paths))
    # NumPy adds the prior after the previous score; the oracle combines emissions first.
    assert actual_scores == pytest.approx(expected_scores, abs=1.0e-5)


@pytest.mark.unit
def test_banded_backpointers_use_one_byte_per_visited_cell():
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _ctc_banded_forward

    _, pointers = _ctc_banded_forward(
        np.zeros((20, 2), np.float32),
        np.array([0, 1, 0]),
        np.arange(20),
        None,
        np.zeros(20, dtype=np.int64),
        3,
    )
    assert pointers.dtype == np.uint8
    assert pointers.nbytes == 20 * 3


@pytest.mark.unit
def test_padded_record_batch_matches_independent_dp_and_backtraces():
    torch.manual_seed(23)
    blank_id = 4
    labels = torch.tensor([[4, 0, 4, 1, 4], [4, 2, 4, 3, 4]])
    state_lengths = torch.tensor([5, 5])
    columns = torch.tensor([[-1, 0, -1, 0, -1], [-1, 1, -1, 1, -1]])
    log_probs = torch.log_softmax(torch.randn(2, 9, blank_id + 1), dim=-1)
    speaker_probs = torch.sigmoid(torch.randn(2, 9, 2))
    frame_lengths = torch.tensor([9, 7])

    batched_paths, batched_scores = search_streams(
        log_probs,
        labels,
        state_lengths,
        blank_id,
        columns,
        speaker_probs,
        0.25,
        frame_lengths=frame_lengths,
    )

    for index, frame_length in enumerate(frame_lengths.tolist()):
        paths, scores = search_streams(
            log_probs[index, :frame_length],
            labels[index : index + 1],
            state_lengths[index : index + 1],
            blank_id,
            columns[index : index + 1],
            speaker_probs[index, :frame_length],
            0.25,
        )
        assert torch.equal(batched_paths[index], paths[0])
        assert batched_scores[index] == pytest.approx(scores[0], abs=1.0e-6)


@pytest.mark.unit
def test_diarization_timestamps_are_contiguous_10ms_activity_segments():
    labels = torch.tensor(
        [
            [True, True, False, False, True, False],
            [False, True, True, True, False, False],
            [False, False, False, False, False, False],
        ]
    )

    segments = MultiSpeakerSOTWordTimestampAligner.diarization_segments(
        labels,
        frame_seconds=0.01,
        time_offset=1.0,
        audio_duration=0.055,
    )

    assert segments == [
        {"speaker": 0, "start": 1.0, "end": 1.02},
        {"speaker": 1, "start": 1.01, "end": 1.04},
        {"speaker": 0, "start": 1.04, "end": 1.05},
    ]


@pytest.mark.unit
def test_public_word_timestamps_only_expose_word_speaker_start_and_end():
    rows = [
        {
            "word": "hello",
            "speaker_tag": 2,
            "word_index": 7,
            "turn_index": 3,
            "start": 0.16,
            "end": 0.32,
            "start_frame": 2,
            "end_frame": 3,
            "ctc_confidence": 0.9,
            "sortformer_column": 1,
            "speaker_confidence": 0.8,
            "speaker_activity_start": 0.16,
            "speaker_activity_end": 0.32,
        }
    ]

    timestamps = MultiSpeakerSOTWordTimestampAligner._public_word_timestamps(rows)

    assert timestamps == {2: [{"word": "hello", "speaker": 2, "start": 0.16, "end": 0.32}]}
