# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.mixins.mixins import PromptStreamingMixin
from nemo.collections.asr.parts.mixins.transcription import ASRTranscriptionMixin, TranscriptionMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTDecoding
from nemo.collections.asr.parts.utils.longform_inference import (
    ChunkedAudioDataset,
    ChunkingConfig,
    collate_audio_chunks,
    make_chunk_plan,
    merge_rnnt_chunks,
    select_chunk_frames,
    select_chunk_words,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


@pytest.mark.unit
@pytest.mark.parametrize('duration', [1, 15999, 16000, 480001, 16000 * 600])
def test_chunk_plan_owns_every_sample_once(duration):
    plan = make_chunk_plan(duration, 16000, ChunkingConfig(), frame_samples=1280)
    assert plan[0].keep_start == 0
    assert plan[-1].keep_end == duration
    assert all(a.keep_end == b.keep_start for a, b in zip(plan, plan[1:]))
    for chunk in plan:
        assert 0 <= chunk.read_start <= chunk.keep_start < chunk.keep_end <= chunk.read_end <= duration
        assert chunk.read_start % 1280 == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    'kwargs',
    [
        {'chunk_size': 0},
        {'chunk_size': -1},
        {'context_size': -1},
        {'chunk_size': float('nan')},
        {'context_size': float('inf')},
    ],
)
def test_invalid_chunk_configuration(kwargs):
    with pytest.raises(ValueError):
        make_chunk_plan(100, 10, ChunkingConfig(**kwargs))


@pytest.mark.unit
def test_chunk_frames_remove_context_before_ctc_collapse():
    plan = make_chunk_plan(8, 1, ChunkingConfig(chunk_size=4, context_size=1))
    # Global CTC path: a a blank b | b blank b blank. The b at the join must collapse once.
    path = torch.tensor([1, 1, 0, 2, 2, 0, 2, 0])
    selected = [select_chunk_frames(path[c.read_start : c.read_end], c, 1) for c in plan]
    result = torch.cat(selected)
    assert torch.equal(result, path)
    collapsed = [v for i, v in enumerate(result.tolist()) if v and (i == 0 or v != result[i - 1])]
    assert collapsed == [1, 2, 2]


@pytest.mark.unit
def test_word_selection_keeps_real_repetitions_and_exact_start_times():
    plan = make_chunk_plan(600, 10, ChunkingConfig(chunk_size=30, context_size=2))
    first = [
        {'word': 'yes', 'start': 29.2, 'end': 29.4},
        {'word': 'yes', 'start': 29.5, 'end': 29.7},
        {'word': 'later', 'start': 30.2, 'end': 30.4},
    ]
    second = [{'word': 'yes', 'start': 1.5, 'end': 1.7}, {'word': 'later', 'start': 2.2, 'end': 2.4}]
    words = select_chunk_words(first, plan[0], 10) + select_chunk_words(second, plan[1], 10)
    assert [w['word'] for w in words] == ['yes', 'yes', 'later']
    assert words[1]['start'] == 29.5 and words[1]['end'] == 29.7
    assert words[2]['start'] == pytest.approx(30.2)
    assert first[1]['start'] == 29.5  # Inputs must not be changed.


@pytest.mark.unit
def test_rnnt_merge_preserves_tokens_confidence_and_absolute_offsets():
    plan = make_chunk_plan(1280 * 10, 16000, ChunkingConfig(chunk_size=0.4, context_size=0.16), 1280)
    parts = []
    for chunk in plan:
        frames = math.ceil((chunk.keep_end - chunk.keep_start) / 1280)
        parts.append(
            Hypothesis(
                score=-0.5,
                y_sequence=torch.tensor([1, 1]),
                timestamp=[0, frames - 1],
                token_duration=torch.tensor([1, 1]),
                alignments=[['frame']] * frames,
                frame_confidence=[[0.8]] * frames,
                non_blank_step_confidence_precomputed=[0.8, 0.9],
            )
        )
    merged = merge_rnnt_chunks(parts, plan, 1280)
    assert merged.y_sequence.tolist() == [1, 1, 1, 1]
    assert merged.timestamp == [0, 4, 5, 9]
    assert len(merged.alignments) == len(merged.frame_confidence) == 10
    assert merged.non_blank_step_confidence_precomputed == [0.8, 0.9, 0.8, 0.9]
    assert merged.token_duration.tolist() == [1, 1, 1, 1]
    assert merged.score == -1.0


@pytest.mark.unit
def test_chunk_plan_has_no_cumulative_timestamp_drift():
    plan = make_chunk_plan(16000 * 30000, 16000, ChunkingConfig(chunk_size=299, context_size=1), 1280)
    assert plan[100].keep_start == 100 * plan[0].keep_end
    assert plan[100].keep_start % 1280 == 0
    assert plan[100].read_start % 1280 == 0


@pytest.mark.unit
def test_word_ownership_uses_half_open_intervals():
    plan = make_chunk_plan(60, 1, ChunkingConfig(chunk_size=30, context_size=2))
    assert select_chunk_words([{'word': 'boundary', 'start': 29, 'end': 31}], plan[0], 1) == []
    words = select_chunk_words([{'word': 'boundary', 'start': 1, 'end': 3}], plan[1], 1)
    assert len(words) == 1 and words[0]['start'] == 29


class _LabelCTCModel(TranscriptionMixin, torch.nn.Module):
    """Real mixin and CTC decoder; deterministic acoustic output needs no checkpoint."""

    _transcribe_chunk_validate = EncDecCTCModel._transcribe_chunk_validate
    _transcribe_chunk_ctc_decoding = EncDecCTCModel._transcribe_chunk_ctc_decoding
    _transcribe_chunk_ctc_decoder = EncDecCTCModel._transcribe_chunk_ctc_decoder
    _transcribe_chunk_outputs = EncDecCTCModel._transcribe_chunk_outputs
    _transcribe_chunk_finalize = EncDecCTCModel._transcribe_chunk_finalize
    _transcribe_input_manifest_processing = ASRTranscriptionMixin._transcribe_input_manifest_processing

    def __init__(self, confidence=False, strategy='greedy'):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cfg = OmegaConf.create({'sample_rate': 10, 'preprocessor': {'window_stride': 0.1}})
        self.encoder = torch.nn.Identity()
        self.encoder.subsampling_factor = 1
        self.decoder = SimpleNamespace(upsampling_factor=1)
        self.decoding = CTCDecoding(
            OmegaConf.create(
                {
                    'strategy': strategy,
                    'greedy': {'preserve_frame_confidence': confidence},
                    'confidence_cfg': {
                        'preserve_frame_confidence': confidence,
                        'preserve_token_confidence': confidence,
                        'preserve_word_confidence': confidence,
                    },
                }
            ),
            vocabulary=['a', 'b', ' '],
        )

    def _transcribe_forward(self, batch, trcfg):
        labels = batch[0].long()
        logits = torch.nn.functional.one_hot(labels, num_classes=4).float() * 10
        factor = self.decoder.upsampling_factor
        return {'logits': logits.repeat_interleave(factor, dim=1).log_softmax(-1), 'logits_len': batch[1] * factor}

    def _setup_transcribe_dataloader(self, config):
        raise AssertionError('Planned chunking must use the common chunk dataset.')

    def _transcribe_output_processing(self, outputs, trcfg):
        raise AssertionError('Planned chunking must use the model output selection hook.')


@pytest.mark.unit
@pytest.mark.parametrize('batch_size', [1, 2, 5])
@pytest.mark.parametrize('strategy', ['greedy', 'greedy_batch'])
def test_transcribe_chunked_ctc_across_batches(batch_size, strategy):
    model = _LabelCTCModel(strategy=strategy)
    # CTC blank is 3. A repeated b across the join collapses; b separated by blank survives.
    audio = torch.tensor([0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0])
    result = model.transcribe(
        [audio, audio], batch_size=batch_size, verbose=False, chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1)
    )
    assert len(result) == 2
    assert [h.text for h in result] == ['abb a', 'abb a']
    assert model.training


@pytest.mark.unit
def test_transcribe_ctc_confidence_matches_words_after_chunk_selection():
    model = _LabelCTCModel(confidence=True)
    audio = torch.tensor([0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0])
    result = model.transcribe(
        audio,
        return_hypotheses=True,
        timestamps=True,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.text == 'abb a'
    assert len(result.word_confidence) == len(result.text.split())
    assert len(result.frame_confidence) == len(audio)
    assert result.y_sequence.shape == (len(audio), 4)
    from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs

    # Compare against the normal CTC frame timeline, including its word-boundary convention.
    outputs = model._transcribe_forward((audio.unsqueeze(0), torch.tensor([len(audio)])), None)
    baseline = model.decoding.ctc_decoder_predictions_tensor(
        outputs['logits'], outputs['logits_len'], return_hypotheses=True
    )[0]
    baseline = process_timestamp_outputs(baseline, 1, 0.1)[0]
    assert result.timestamp == baseline.timestamp
    assert result.token_confidence == baseline.token_confidence
    assert result.word_confidence == baseline.word_confidence


@pytest.mark.unit
def test_character_ctc_word_confidence_decodes_token_ids():
    model = _LabelCTCModel(confidence=True)
    labels = torch.tensor([[0, 3, 2, 1, 3]])
    logits = (torch.nn.functional.one_hot(labels, 4).float() * 10).log_softmax(-1)
    result = model.decoding.ctc_decoder_predictions_tensor(logits, torch.tensor([5]), return_hypotheses=True)[0]
    assert result.text == 'a b'
    assert len(result.word_confidence) == 2


@pytest.mark.unit
def test_ctc_upsampling_keeps_frame_count_and_timestamp_rate():
    from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs

    model = _LabelCTCModel(confidence=True)
    model.decoder.upsampling_factor = 2
    audio = torch.tensor([0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0])
    result = model.transcribe(
        audio,
        return_hypotheses=True,
        timestamps=True,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    outputs = model._transcribe_forward((audio.unsqueeze(0), torch.tensor([len(audio)])), None)
    baseline = model.decoding.ctc_decoder_predictions_tensor(
        outputs['logits'], outputs['logits_len'], return_hypotheses=True
    )[0]
    baseline = process_timestamp_outputs(baseline, 1, 0.05)[0]
    assert result.text == baseline.text
    assert result.y_sequence.shape == (len(audio) * 2, 4)
    assert result.timestamp == baseline.timestamp


@pytest.mark.unit
def test_cli_chunking_mapping_accepts_hydra_override():
    from examples.asr.transcribe_speech import TranscriptionConfig

    config = OmegaConf.structured(TranscriptionConfig)
    assert not config.chunking
    OmegaConf.set_struct(config, True)
    OmegaConf.update(config, 'chunking', {'chunk_size': 30.0, 'context_size': 2.0}, merge=True, force_add=True)
    assert config.chunking.chunk_size == 30.0
    assert config.chunking.context_size == 2.0


@pytest.mark.unit
def test_restored_conformer_checkpoint_chunking_with_files_and_tensors(tmp_path):
    import soundfile as sf

    config = OmegaConf.create(
        {
            'sample_rate': 16000,
            'preprocessor': {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'sample_rate': 16000,
                'features': 8,
                'n_fft': 256,
                'window_size': 0.016,
                'window_stride': 0.01,
                'dither': 0.0,
                'pad_to': 0,
            },
            'encoder': {
                '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                'feat_in': 8,
                'n_layers': 1,
                'd_model': 8,
                'n_heads': 2,
                'subsampling': 'striding',
                'subsampling_factor': 4,
                'subsampling_conv_channels': 8,
                'ff_expansion_factor': 2,
                'conv_kernel_size': 3,
                'dropout': 0.0,
                'dropout_pre_encoder': 0.0,
                'dropout_att': 0.0,
            },
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': 8,
                'num_classes': 3,
                'vocabulary': ['a', 'b', ' '],
            },
            'decoding': {'strategy': 'greedy_batch'},
        }
    )
    model = EncDecCTCModel(config)
    checkpoint = tmp_path / 'tiny.nemo'
    model.save_to(str(checkpoint))
    restored = EncDecCTCModel.restore_from(str(checkpoint), map_location='cpu')
    waveform = torch.randn(32163, generator=torch.Generator().manual_seed(21)) * 0.1
    sf.write(tmp_path / 'audio.wav', waveform.numpy(), 16000, subtype='FLOAT')
    chunking = ChunkingConfig(chunk_size=0.64, context_size=0.16)
    tensor = restored.transcribe(
        waveform, chunking=chunking, return_hypotheses=True, timestamps=True, batch_size=1, verbose=False
    )[0]
    file = restored.transcribe(
        str(tmp_path / 'audio.wav'),
        chunking=chunking,
        return_hypotheses=True,
        timestamps=True,
        batch_size=2,
        verbose=False,
    )[0]
    assert tensor.y_sequence.shape == file.y_sequence.shape == (math.ceil(len(waveform) / 640), 4)
    assert tensor.text == file.text
    assert torch.allclose(tensor.y_sequence, file.y_sequence, atol=1e-5)
    assert tensor.timestamp == file.timestamp


@pytest.mark.unit
def test_chunked_file_tensor_and_offset_manifest_match(tmp_path):
    import json

    import soundfile as sf

    model = _LabelCTCModel()
    audio = torch.tensor([0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0])
    sf.write(tmp_path / 'audio.wav', audio.numpy(), 10, subtype='FLOAT')
    manifest = tmp_path / 'input.jsonl'
    manifest.write_text(json.dumps({'audio_filepath': 'audio.wav', 'offset': 0.3, 'duration': 0.8, 'text': ''}) + '\n')
    config = ChunkingConfig(chunk_size=0.4, context_size=0.1)
    file_result = model.transcribe([str(tmp_path / 'audio.wav')] * 2, chunking=config, verbose=False)
    tensor_result = model.transcribe([audio] * 2, chunking=config, verbose=False)
    assert [h.text for h in file_result] == [h.text for h in tensor_result]
    offset_result = model.transcribe(str(manifest), chunking=config, verbose=False)
    assert offset_result[0].text == model.transcribe(audio[3:], chunking=config, verbose=False)[0].text


@pytest.mark.unit
@pytest.mark.parametrize('workers', [0, 2])
def test_lazy_chunk_dataset_preserves_order_with_workers(workers):
    sources = [torch.arange(13), torch.arange(7)]
    ds = ChunkedAudioDataset(sources, 1, ChunkingConfig(chunk_size=4, context_size=1), frame_samples=1, batch_size=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=workers, collate_fn=collate_audio_chunks)
    order = [(idx, chunk.keep_start) for _, _, ids, chunks, _ in loader for idx, chunk in zip(ids, chunks)]
    assert order == [(0, 0), (0, 4), (0, 8), (0, 12), (1, 0), (1, 4)]


@pytest.mark.unit
def test_chunking_failure_restores_model_and_caller_config():
    model = _LabelCTCModel()
    config = ASRTranscriptionMixin.get_transcribe_config()
    config.chunking = ChunkingConfig(chunk_size=-1)
    config.verbose = False
    with pytest.raises(ValueError, match='chunk_size'):
        model.transcribe(torch.ones(10), override_config=config)
    assert model.training
    assert config.chunking.chunk_size == -1


@pytest.mark.unit
def test_chunking_rejects_external_loader_and_unsupported_decoder():
    model = _LabelCTCModel()
    loader = torch.utils.data.DataLoader([torch.ones(10)])
    with pytest.raises(ValueError, match='external DataLoader'):
        model.transcribe(loader, chunking=ChunkingConfig())
    model.decoding.cfg.strategy = 'beam'
    with pytest.raises(ValueError, match='greedy'):
        model.transcribe(torch.ones(10), chunking=ChunkingConfig())


class _TinyBPETokenizer:
    vocab = ['▁a', 'b', ' ']
    tokenizer = SimpleNamespace(vocab_size=3)

    def ids_to_tokens(self, ids):
        return [self.vocab[int(i)] for i in ids]

    def tokens_to_text(self, tokens):
        return ''.join(tokens).replace('▁', ' ').strip()

    def ids_to_text(self, ids):
        return self.tokens_to_text(self.ids_to_tokens(ids))


class _ConstantRNNTModel(_LabelCTCModel):
    _transcribe_chunk_validate = EncDecRNNTModel._transcribe_chunk_validate
    _transcribe_chunk_decoding = EncDecRNNTModel._transcribe_chunk_decoding
    _transcribe_chunk_outputs = EncDecRNNTModel._transcribe_chunk_outputs
    _transcribe_chunk_finalize = EncDecRNNTModel._transcribe_chunk_finalize

    def __init__(self, model_type='rnnt', strategy='greedy', duration=1):
        super().__init__()
        decoder = RNNTDecoder({'pred_hidden': 4, 'pred_rnn_layers': 1}, vocab_size=3)
        joint = RNNTJoint(
            {'encoder_hidden': 1, 'pred_hidden': 4, 'joint_hidden': 4, 'activation': 'relu'},
            num_classes=3,
            num_extra_outputs=3 if model_type == 'tdt' else 0,
        )
        with torch.no_grad():
            for parameter in joint.parameters():
                parameter.zero_()
            joint.joint_net[-1].bias[0] = 10
            if model_type == 'tdt':
                joint.joint_net[-1].bias[5 if duration == 1 else 6] = 10
        self.decoder, self.joint = decoder, joint
        decoding_class = RNNTBPEDecoding if model_type == 'tdt' else RNNTDecoding
        kwargs = {'tokenizer': _TinyBPETokenizer()} if model_type == 'tdt' else {'vocabulary': ['a', 'b', ' ']}
        self.decoding = decoding_class(
            OmegaConf.create(
                {
                    'strategy': strategy,
                    'model_type': model_type,
                    'durations': [0, 1, 3] if model_type == 'tdt' else [],
                    'compute_timestamps': False,
                    'greedy': {'max_symbols_per_step': 1, 'preserve_frame_confidence': True},
                    'confidence_cfg': {
                        'preserve_frame_confidence': True,
                        'preserve_token_confidence': True,
                        'preserve_word_confidence': True,
                    },
                }
            ),
            decoder,
            joint,
            **kwargs,
        )

    def _transcribe_forward(self, batch, trcfg):
        return {'encoded': batch[0].unsqueeze(1), 'encoded_len': batch[1]}


@pytest.mark.unit
@pytest.mark.parametrize('timestamps', [False, True])
@pytest.mark.parametrize('strategy', ['greedy', 'greedy_batch'])
def test_real_rnnt_decoder_selects_owned_frames_and_rebuilds_metadata(timestamps, strategy):
    model = _ConstantRNNTModel(strategy=strategy)
    result = model.transcribe(
        torch.ones(11),
        return_hypotheses=True,
        timestamps=timestamps,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.text == 'a' * 11
    assert len(result.y_sequence) == len(result.token_confidence) == 11
    assert len(result.word_confidence) == 1
    if timestamps:
        assert result.timestamp['timestep'] == list(range(11))
        assert result.timestamp['word'][0]['end'] == pytest.approx(1.1)
    else:
        assert result.timestamp == list(range(11))


@pytest.mark.unit
@pytest.mark.parametrize('strategy', ['greedy', 'greedy_batch'])
def test_tdt_chunk_timestamps_collect_durations_without_changing_model_config(strategy):
    model = _ConstantRNNTModel(model_type='tdt', strategy=strategy)
    original_decoding = model.decoding
    assert not model.decoding.tdt_include_token_duration
    result = model.transcribe(
        torch.ones(11),
        return_hypotheses=True,
        timestamps=True,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.text.split() == ['a'] * 11
    assert len(result.token_duration) == len(result.y_sequence) == len(result.token_confidence) == 11
    assert result.timestamp['word'][-1]['end'] == pytest.approx(1.1)
    assert model.decoding is original_decoding
    assert not model.decoding.compute_timestamps


@pytest.mark.unit
@pytest.mark.parametrize('strategy', ['greedy', 'greedy_batch'])
def test_tdt_duration_stays_inside_owned_region(strategy):
    model = _ConstantRNNTModel(model_type='tdt', strategy=strategy, duration=3)
    result = model.transcribe(
        torch.ones(11),
        return_hypotheses=True,
        timestamps=True,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.timestamp['timestep'] == [0, 3, 4, 7, 8]
    assert list(result.token_duration) == [3, 1, 3, 1, 3]
    if result.frame_confidence is not None:
        assert len(result.frame_confidence) == 11
    words = result.timestamp['word']
    assert all(a['end'] <= b['start'] for a, b in zip(words, words[1:]))


class _AlignedAEDModel(_LabelCTCModel):
    _transcribe_chunk_validate = EncDecMultiTaskModel._transcribe_chunk_validate
    _transcribe_chunk_outputs = EncDecMultiTaskModel._transcribe_chunk_outputs
    _transcribe_chunk_finalize = EncDecMultiTaskModel._transcribe_chunk_finalize

    def __init__(self):
        super().__init__()
        self.timestamps_asr_model = torch.nn.Identity()
        self.prompt = SimpleNamespace(
            get_default_dialog_slots=lambda: [{'role': 'user', 'slots': {'source_lang': 'de', 'target_lang': 'de'}}]
        )
        self.tokenizer = SimpleNamespace(text_to_ids=lambda text: [ord(c) for c in text])

    def _transcribe_forward(self, batch, trcfg):
        return {'batch': batch}

    def _transcribe_output_processing(self, outputs, trcfg):
        assert trcfg.timestamps and not trcfg.enable_chunking
        return [
            Hypothesis(
                score=-1,
                y_sequence=torch.tensor([1]),
                text=' '.join(['ja'] * int(length)),
                timestamp={'word': [{'word': 'ja', 'start': i / 10, 'end': (i + 1) / 10} for i in range(int(length))]},
            )
            for length in outputs['batch'][1]
        ]


@pytest.mark.unit
def test_aed_adapter_keeps_repeated_words_with_original_duration():
    from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig

    model = _AlignedAEDModel()
    config = MultiTaskTranscriptionConfig(
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
        timestamps=True,
        verbose=False,
        batch_size=2,
        num_workers=0,
    )
    result = model.transcribe(torch.ones(11), override_config=config)[0]
    assert result.text.split() == ['ja'] * 11
    assert [w['start'] for w in result.timestamp['word']] == pytest.approx([i / 10 for i in range(11)])
    assert all(w['end'] - w['start'] == pytest.approx(0.1) for w in result.timestamp['word'])
    assert config.enable_chunking  # The new path must not change the caller's legacy setting.
    assert model.training


@pytest.mark.unit
def test_aed_adapter_rejects_unaligned_or_translated_text():
    model = _AlignedAEDModel()
    model.timestamps_asr_model = None
    with pytest.raises(ValueError, match='aligner'):
        model.transcribe(torch.ones(11), chunking=ChunkingConfig())


@pytest.mark.unit
def test_generator_close_restores_state_and_emits_complete_utterances():
    model = _LabelCTCModel()
    config = ASRTranscriptionMixin.get_transcribe_config()
    config.chunking = ChunkingConfig(chunk_size=0.4)
    config.verbose = False
    config.num_workers = 0
    generator = model.transcribe_generator([torch.zeros(11), torch.zeros(11)], override_config=config)
    assert next(generator)[0].text == 'a'
    assert not model.training
    generator.close()
    assert model.training


@pytest.mark.unit
def test_empty_manifest_raises_and_restores_state(tmp_path):
    manifest = tmp_path / 'empty.jsonl'
    manifest.write_text('')
    model = _LabelCTCModel()
    with pytest.raises(ValueError, match='no audio entries'):
        model.transcribe(str(manifest), chunking=ChunkingConfig(), verbose=False)
    assert model.training


class _HybridCTCModel(_LabelCTCModel):
    _transcribe_chunk_ctc_decoding = EncDecHybridRNNTCTCModel._transcribe_chunk_ctc_decoding
    _transcribe_chunk_ctc_decoder = EncDecHybridRNNTCTCModel._transcribe_chunk_ctc_decoder
    _transcribe_chunk_validate = EncDecHybridRNNTCTCModel._transcribe_chunk_validate
    _transcribe_chunk_outputs = EncDecHybridRNNTCTCModel._transcribe_chunk_outputs
    _transcribe_chunk_finalize = EncDecHybridRNNTCTCModel._transcribe_chunk_finalize

    def __init__(self):
        super().__init__()
        self.cur_decoder = 'ctc'
        self.ctc_decoder = self.decoder
        self.ctc_decoding = self.decoding
        self.decoding = SimpleNamespace(cfg=OmegaConf.create({'strategy': 'beam'}))


@pytest.mark.unit
def test_hybrid_chunking_uses_selected_ctc_head():
    model = _HybridCTCModel()
    result = model.transcribe(
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0]),
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )
    assert result[0].text == 'abb'


@pytest.mark.unit
def test_prompt_conditioned_chunk_batch_preserves_per_utterance_language():
    model = SimpleNamespace(resolve_prompt_language=lambda lang: {None: 0, 'de': 1, 'en': 2}[lang])
    cuts = [SimpleNamespace(supervisions=[SimpleNamespace(language=lang)]) for lang in ['de', 'en']]
    config = SimpleNamespace(target_lang=None)
    batch = PromptStreamingMixin._transcribe_chunk_batch(
        model, torch.zeros(2, 10), torch.tensor([10, 10]), cuts, config
    )
    assert batch[4].tolist() == [1, 2]
    config.target_lang = 'en'
    batch = PromptStreamingMixin._transcribe_chunk_batch(
        model, torch.zeros(2, 10), torch.tensor([10, 10]), cuts, config
    )
    assert batch[4].tolist() == [2, 2]


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required.')
@pytest.mark.parametrize('kind', ['ctc', 'rnnt', 'tdt'])
def test_chunk_selection_and_metadata_on_cuda(kind):
    if kind == 'ctc':
        model = _LabelCTCModel(confidence=True, strategy='greedy_batch')
        audio = torch.tensor([0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0])
        expected_words = ['abb', 'a']
    else:
        model = _ConstantRNNTModel(model_type=kind, strategy='greedy_batch')
        audio = torch.ones(11)
        expected_words = ['a' * 11] if kind == 'rnnt' else ['a'] * 11
    model = model.to('cuda')
    result = model.transcribe(
        audio,
        return_hypotheses=True,
        timestamps=True,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.text.split() == expected_words
    assert len(result.word_confidence) == len(expected_words)
    assert result.y_sequence.device.type == 'cpu'
