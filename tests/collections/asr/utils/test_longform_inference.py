# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.asr.parts.utils.longform_inference import (
    ChunkBatches,
    ChunkedAudioDataset,
    ChunkingConfig,
    collate_audio_chunks,
    make_chunk_plan,
    select_chunk_frames,
    select_chunk_words,
    transfer_word_times,
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
def test_last_chunk_accepts_encoder_dropping_final_partial_hop():
    # 25 frames plus 1 sample: the feature extractor drops the final partial hop.
    last = make_chunk_plan(1280 * 25 + 1, 16000, ChunkingConfig(chunk_size=0.8, context_size=0.16), 1280)[-1]
    assert last.is_last and select_chunk_frames(torch.arange(7), last, 1280).tolist() == [2, 3, 4, 5, 6]
    with pytest.raises(RuntimeError, match='requires 8'):
        select_chunk_frames(torch.arange(6), last, 1280)
    inner = make_chunk_plan(1280 * 25, 16000, ChunkingConfig(chunk_size=0.8, context_size=0.16), 1280)[0]
    with pytest.raises(RuntimeError, match='requires 10'):
        select_chunk_frames(torch.arange(9), inner, 1280)


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
def test_ctc_fusion_model_keeps_log_probs():
    model = _LabelCTCModel(strategy='greedy_batch')
    model.decoding.decoding.fusion_models = [object()]
    chunk = make_chunk_plan(4, 10, ChunkingConfig(chunk_size=1.0, context_size=0.0))[0]
    outputs = {'logits': torch.zeros(1, 4, 4), 'logits_len': torch.tensor([4])}
    [part] = model._transcribe_chunk_outputs(outputs, [chunk], 1, ASRTranscriptionMixin.get_transcribe_config())
    assert part.shape == (4, 4)


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


def _tiny_conformer_config(subsampling='striding', factor=4):
    return {
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
            'subsampling': subsampling,
            'subsampling_factor': factor,
            'subsampling_conv_channels': 8,
            'ff_expansion_factor': 2,
            'conv_kernel_size': 3,
            'dropout': 0.0,
            'dropout_pre_encoder': 0.0,
            'dropout_att': 0.0,
        },
    }


@pytest.mark.unit
# 640 * 50 + r samples. A remainder below one 160-sample hop yields no final feature frame.
@pytest.mark.parametrize('num_samples', [32163, 32001, 32159])
def test_restored_conformer_checkpoint_chunking_with_files_and_tensors(tmp_path, num_samples):
    import soundfile as sf

    config = OmegaConf.create(
        {
            **_tiny_conformer_config(),
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
    waveform = torch.randn(num_samples, generator=torch.Generator().manual_seed(21)) * 0.1
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
    with torch.inference_mode():
        _, num_frames = restored.forward(input_signal=waveform[None], input_signal_length=torch.tensor([num_samples]))[
            :2
        ]
    # Owned frames tile the encoder output of the whole source: none missing, none repeated.
    assert tensor.y_sequence.shape == file.y_sequence.shape == (int(num_frames[0]), 4)
    assert tensor.text == file.text
    assert torch.allclose(tensor.y_sequence, file.y_sequence, atol=1e-5)
    assert tensor.timestamp == file.timestamp


@pytest.mark.unit
# 1280 * 25 + r samples. A remainder below one 160-sample hop yields no final feature frame.
@pytest.mark.parametrize('num_samples', [32001, 32159, 32160])
def test_transducer_chunking_accepts_source_end_off_the_frame_grid(num_samples):
    # Character vocabulary, so RNNT only: NeMo's character TDT decoding cannot build timestamps.
    model_type, vocabulary, durations = 'rnnt', ['a', 'b', ' '], []
    config = OmegaConf.create(
        {
            **_tiny_conformer_config('dw_striding', 8),
            'labels': vocabulary,
            'model_defaults': {'enc_hidden': 8, 'pred_hidden': 8, 'tdt_durations': durations},
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                'vocab_size': 3,
                'prednet': {'pred_hidden': 8, 'pred_rnn_layers': 1},
            },
            'joint': {
                '_target_': 'nemo.collections.asr.modules.RNNTJoint',
                'num_classes': 3,
                'num_extra_outputs': len(durations),
                'vocabulary': vocabulary,
                'jointnet': {'joint_hidden': 8, 'activation': 'relu', 'encoder_hidden': 8, 'pred_hidden': 8},
            },
            'decoding': {
                'strategy': 'greedy_batch',
                'greedy': {'max_symbols': 10},
                'model_type': model_type,
                'durations': durations,
            },
            'loss': (
                {'loss_name': 'tdt', 'tdt_kwargs': {'durations': durations, 'sigma': 0.02}}
                if model_type == 'tdt'
                else {'loss_name': 'default'}
            ),
        }
    )
    torch.manual_seed(0)
    model = EncDecRNNTModel(config).eval()
    waveform = torch.randn(num_samples, generator=torch.Generator().manual_seed(5)) * 0.1
    result = model.transcribe(
        waveform,
        chunking=ChunkingConfig(chunk_size=0.8, context_size=0.16),
        timestamps=True,
        batch_size=2,
        verbose=False,
    )[0]
    words = result.timestamp['word']
    assert result.text.split() == [w['word'] for w in words]
    assert all(0 <= w['start'] <= w['end'] <= num_samples / 16000 + 0.08 for w in words)


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
    plan = ChunkBatches(sources, 1, ChunkingConfig(chunk_size=4, context_size=1), frame_samples=1, batch_size=2)
    loader = torch.utils.data.DataLoader(
        ChunkedAudioDataset(sources, 1), batch_sampler=plan, num_workers=workers, collate_fn=collate_audio_chunks
    )
    order = [(idx, chunk.keep_start) for _, _, ids, chunks, _ in loader for idx, chunk in zip(ids, chunks)]
    assert order == [(0, 0), (0, 4), (0, 8), (0, 12), (1, 0), (1, 4)]


@pytest.mark.unit
@pytest.mark.parametrize('workers', [0, 2, 4])
def test_manifest_entries_are_parsed_once_whatever_the_worker_count(tmp_path, monkeypatch, workers):
    import json

    import soundfile as sf

    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    audio = torch.tensor([0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0])
    sf.write(tmp_path / 'a.wav', audio.numpy(), 10, subtype='FLOAT')
    sf.write(tmp_path / 'b.wav', audio[:4].numpy(), 10, subtype='FLOAT')
    manifest = tmp_path / 'input.jsonl'
    # No sample-rate field, so every parse reads the file header; a.wav appears twice.
    entries = [('a.wav', 1.1), ('b.wav', 0.4), ('a.wav', 1.1)]
    manifest.write_text(
        ''.join(json.dumps({'audio_filepath': f, 'duration': d, 'text': ''}) + '\n' for f, d in entries)
    )
    parses = []
    create = LazyNeMoIterator._create_recording

    def counted(self, *args, **kwargs):
        assert torch.utils.data.get_worker_info() is None, 'workers must not parse the manifest'
        parses.append(1)
        return create(self, *args, **kwargs)

    monkeypatch.setattr(LazyNeMoIterator, '_create_recording', counted)
    model = _LabelCTCModel()
    # 3 + 1 + 3 chunks in batches of 2 leave a partial final batch.
    results = model.transcribe(
        str(manifest), chunking=ChunkingConfig(0.4, 0.1), batch_size=2, num_workers=workers, verbose=False
    )
    assert len(parses) == len(entries)
    expected = model.transcribe([audio, audio[:4], audio], chunking=ChunkingConfig(0.4, 0.1), verbose=False)
    assert [h.text for h in results] == [h.text for h in expected]


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

    def text_to_ids(self, text):
        return [0 for _ in text.split()]


class _ConstantRNNTModel(_LabelCTCModel):
    _transcribe_chunk_validate = EncDecRNNTModel._transcribe_chunk_validate
    _transcribe_chunk_decoding = EncDecRNNTModel._transcribe_chunk_decoding
    _transcribe_chunk_outputs = EncDecRNNTModel._transcribe_chunk_outputs
    _transcribe_chunk_finalize = EncDecRNNTModel._transcribe_chunk_finalize

    def __init__(self, model_type='rnnt', strategy='greedy', duration=1, timestamps=False):
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
        self.decoding = RNNTBPEDecoding(
            OmegaConf.create(
                {
                    'strategy': strategy,
                    'model_type': model_type,
                    'durations': [0, 1, 3] if model_type == 'tdt' else [],
                    # EncDecRNNTModel.transcribe() sets this when timestamps are requested.
                    'compute_timestamps': timestamps,
                    # Non-batched TDT greedy confidence reads the alignments.
                    'preserve_alignments': model_type == 'tdt' and strategy == 'greedy',
                    'greedy': {'max_symbols_per_step': 1, 'preserve_frame_confidence': True},
                    'beam': {'beam_size': 2},
                    'confidence_cfg': {
                        'preserve_frame_confidence': True,
                        'preserve_token_confidence': True,
                        'preserve_word_confidence': True,
                    },
                }
            ),
            decoder,
            joint,
            tokenizer=_TinyBPETokenizer(),
        )

    def _transcribe_forward(self, batch, trcfg):
        return {'encoded': batch[0].unsqueeze(1), 'encoded_len': batch[1]}


@pytest.mark.unit
@pytest.mark.parametrize('model_type', ['rnnt', 'tdt'])
@pytest.mark.parametrize('timestamps', [False, True])
@pytest.mark.parametrize('strategy', ['greedy', 'greedy_batch'])
def test_rnnt_chunks_keep_each_word_once_at_its_time(model_type, timestamps, strategy):
    # One word per frame; every read window is decoded, and only owned words are kept.
    model = _ConstantRNNTModel(model_type=model_type, strategy=strategy)
    result = model.transcribe(
        torch.ones(11),
        return_hypotheses=True,
        timestamps=timestamps,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.text.split() == ['a'] * 11
    assert len(result.y_sequence) == 11
    if timestamps:
        assert [w['start'] for w in result.timestamp['word']] == pytest.approx([k / 10 for k in range(11)])
        assert [w['start_offset'] for w in result.timestamp['word']] == list(range(11))
    assert not model.decoding.compute_timestamps  # Word times come from a call-local decoding.


@pytest.mark.unit
def test_rnnt_chunking_rejects_n_best():
    model = _ConstantRNNTModel(strategy='greedy_batch')
    model.decoding.cfg = OmegaConf.create({'strategy': 'beam', 'beam': {'return_best_hypothesis': False}})
    with pytest.raises(ValueError, match='one best hypothesis'):
        model.transcribe(torch.ones(11), chunking=ChunkingConfig())


class _FramewiseRNNTModel(_ConstantRNNTModel):
    """Random prediction and joint networks over a context-free encoder."""

    def __init__(self, model_type, strategy):
        super().__init__(model_type=model_type, strategy=strategy)
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in [*self.decoder.parameters(), *self.joint.parameters()]:
                parameter.copy_(torch.randn(parameter.shape, generator=generator))


@pytest.mark.unit
@pytest.mark.parametrize('model_type', ['rnnt', 'tdt'])
@pytest.mark.parametrize('strategy', ['greedy_batch', 'malsd_batch'])
@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required.')],
        ),
    ],
)
def test_rnnt_chunking_is_batch_invariant_and_one_window_matches_plain_decoding(model_type, strategy, device):
    model = _FramewiseRNNTModel(model_type, strategy).to(device)
    generator = torch.Generator().manual_seed(3)
    sources = [torch.randn(n, generator=generator) * 4 for n in (23, 9, 4, 17)]
    chunked = [
        [h.text for h in model.transcribe(sources, batch_size=b, verbose=False, chunking=ChunkingConfig(0.4, 0.2))]
        for b in (1, 3, 8)
    ]
    assert chunked[0] == chunked[1] == chunked[2]
    # A window that covers the whole source is one plain decode of that source.
    whole = model.transcribe(sources, batch_size=8, verbose=False, chunking=ChunkingConfig(10.0, 0.0))
    num_words = 0
    for source, hyp in zip(sources, whole):
        encoded = source.to(device)[None, None]
        reference = model.decoding.rnnt_decoder_predictions_tensor(
            encoded, torch.tensor([len(source)], device=device), return_hypotheses=True
        )[0]
        assert hyp.text.split() == reference.text.split()
        num_words += len(reference.text.split())
    assert num_words >= 3  # The comparison needs real emissions.


class _AlignedAEDModel(_LabelCTCModel):
    _transcribe_chunk_validate = EncDecMultiTaskModel._transcribe_chunk_validate
    _transcribe_chunk_batch = EncDecMultiTaskModel._transcribe_chunk_batch
    _transcribe_chunk_outputs = EncDecMultiTaskModel._transcribe_chunk_outputs
    _transcribe_chunk_finalize = EncDecMultiTaskModel._transcribe_chunk_finalize

    def __init__(self):
        super().__init__()
        self.timestamps_asr_model = torch.nn.Identity()
        self.prompt = SimpleNamespace(
            get_default_dialog_slots=lambda: [{'role': 'user', 'slots': {'source_lang': 'de', 'target_lang': 'de'}}]
        )
        self.tokenizer = SimpleNamespace(text_to_ids=lambda text: [ord(c) for c in text])

        self.keep_ctc_words = lambda words: words

    @staticmethod
    def _words(audio, length):
        # One word per sample: 'ja' for 1, otherwise a word naming the sample value.
        return ['ja' if int(x) == 1 else f'w{int(x)}' for x in audio[: int(length)]]

    def _transcribe_forward(self, batch, trcfg):
        return {'batch': batch}

    def _transcribe_output_processing(self, outputs, trcfg):
        assert not trcfg.timestamps and not trcfg.enable_chunking
        audio, lengths = outputs['batch'][:2]
        return [
            Hypothesis(score=-1, y_sequence=None, text=' '.join(self._words(a, n))) for a, n in zip(audio, lengths)
        ]

    def _transcribe_chunk_ctc_words(self, audio, lengths):
        timed = [
            [{'word': w, 'start': k / 10, 'end': (k + 1) / 10} for k, w in enumerate(self._words(a, n))]
            for a, n in zip(audio, lengths)
        ]
        return [self.keep_ctc_words(words) for words in timed]


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
def test_aed_adapter_keeps_words_the_ctc_model_misses():
    # Fast speech: the CTC model times only every third word; the AED words still all survive once.
    model = _AlignedAEDModel()
    model.keep_ctc_words = lambda words: words[::3]
    audio = torch.arange(2, 25).float()
    result = model.transcribe(audio, override_config=_aed_config())[0]
    assert result.text.split() == [f'w{i}' for i in range(2, 25)]


@pytest.mark.unit
def test_aed_without_ctc_model_stitches_overlapping_windows():
    model = _AlignedAEDModel()
    model.timestamps_asr_model = None
    audio = torch.arange(2, 25).float()
    result = model.transcribe(audio, override_config=_aed_config())[0]
    assert result.text.split() == [f'w{i}' for i in range(2, 25)]
    with pytest.raises(ValueError, match='CTC timestamp model'):
        model.transcribe(audio, override_config=_aed_config(timestamps=True))


def _aed_config(**kwargs):
    from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig

    return MultiTaskTranscriptionConfig(
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1), verbose=False, batch_size=2, **kwargs
    )


@pytest.mark.unit
def test_transfer_word_times_matches_and_fills_gaps():
    timed = [{'word': 'hello', 'start': 0.0, 'end': 1.0}, {'word': 'world', 'start': 2.0, 'end': 3.0}]
    words = transfer_word_times(['Hello,', 'big', 'world', 'again'], timed, duration=5.0)
    assert [(w['start'], w['end']) for w in words] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    assert [(w['start'], w['end']) for w in transfer_word_times(['a', 'b'], [], duration=4.0)] == [(0, 2), (2, 4)]


@pytest.mark.unit
@pytest.mark.parametrize(
    'decoding, rejected',
    [
        ({'strategy': 'beam', 'beam': {'beam_size': 2, 'return_best_hypothesis': False}}, True),
        ({'strategy': 'greedy', 'greedy': {'temperature': 1.0, 'n_samples': 2}}, True),
        ({'strategy': 'greedy_batch', 'greedy': {'temperature': 1.0, 'n_samples': 2}}, True),
        ({'strategy': 'greedy', 'greedy': {'temperature': 1.0, 'n_samples': 1}}, False),
        ({'strategy': 'greedy_batch', 'greedy': {'temperature': None, 'n_samples': 2}}, False),
        ({'strategy': 'beam', 'beam': {'beam_size': 2, 'return_best_hypothesis': True}}, False),
    ],
)
def test_aed_adapter_rejects_multiple_outputs_per_chunk(decoding, rejected):
    model = _AlignedAEDModel()
    model.decoding.cfg = OmegaConf.create(decoding)
    forward = model._transcribe_forward
    calls = []
    model._transcribe_forward = lambda *args: calls.append(1) or forward(*args)
    if rejected:
        with pytest.raises(ValueError, match='one best hypothesis'):
            model.transcribe(torch.ones(11), override_config=_aed_config())
        assert not calls  # Rejected before any forward pass.
    else:
        assert model.transcribe(torch.ones(11), override_config=_aed_config())[0].text.split() == ['ja'] * 11


@pytest.mark.unit
def test_aed_adapter_accepts_wrapped_and_unset_prompt_languages():
    from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig

    config = MultiTaskTranscriptionConfig(
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
        prompt={'source_lang': '<|de|>', 'target_lang': None},
        verbose=False,
    )
    result = _AlignedAEDModel().transcribe(torch.ones(11), override_config=config)[0]
    assert result.text.split() == ['ja'] * 11


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
        expected_words = ['a'] * 11
    model = model.to('cuda')
    result = model.transcribe(
        audio,
        return_hypotheses=True,
        timestamps=True,
        verbose=False,
        chunking=ChunkingConfig(chunk_size=0.4, context_size=0.1),
    )[0]
    assert result.text.split() == expected_words
    if kind == 'ctc':  # Transducer chunks are assembled from words, without token confidence.
        assert len(result.word_confidence) == len(expected_words)
    assert result.y_sequence.device.type == 'cpu'
