# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planned chunking through small saved and restored transducer, hybrid and AED checkpoints."""

import json
import os

import pytest
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import (
    EncDecCTCModelBPE,
    EncDecHybridRNNTCTCBPEModel,
    EncDecMultiTaskModel,
    EncDecRNNTBPEModel,
)
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig
from nemo.collections.asr.parts.utils.longform_inference import ChunkingConfig

CHUNKING = ChunkingConfig(chunk_size=0.8, context_size=0.16)
WORD = 'yes'  # A whole-word token of the CI tokenizer; deterministic heads emit it.


def _preprocessor():
    return {
        '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'sample_rate': 16000,
        'window_stride': 0.01,
        'features': 16,
        'dither': 0.0,
    }


def _conformer(d_model=16):
    return {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
        'feat_in': 16,
        'n_layers': 1,
        'd_model': d_model,
        'n_heads': 2,
        'subsampling': 'striding',
        'subsampling_factor': 4,
        'subsampling_conv_channels': 8,
        'ff_expansion_factor': 2,
        'conv_kernel_size': 3,
        'dropout': 0.0,
        'dropout_pre_encoder': 0.0,
        'dropout_att': 0.0,
    }


def _transducer_config(test_data_dir, model_type, hybrid=False):
    durations = [0, 1, 2, 3, 4] if model_type == 'tdt' else []
    config = {
        'sample_rate': 16000,
        'preprocessor': _preprocessor(),
        'model_defaults': {'enc_hidden': 16, 'pred_hidden': 16, 'tdt_durations': durations},
        'encoder': _conformer(),
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
            'prednet': {'pred_hidden': 16, 'pred_rnn_layers': 1},
        },
        'joint': {
            '_target_': 'nemo.collections.asr.modules.RNNTJoint',
            'num_extra_outputs': len(durations),
            'jointnet': {'joint_hidden': 16, 'activation': 'relu'},
        },
        'tokenizer': {'dir': os.path.join(test_data_dir, 'asr', 'tokenizers', 'an4_wpe_128'), 'type': 'wpe'},
        'decoding': {
            'strategy': 'greedy_batch',
            'greedy': {'max_symbols': 1},
            'model_type': model_type,
            'durations': durations,
        },
        'loss': (
            {'loss_name': 'tdt', 'tdt_kwargs': {'durations': durations, 'sigma': 0.02}}
            if model_type == 'tdt'
            else {'loss_name': 'default'}
        ),
    }
    if hybrid:
        config['aux_ctc'] = {
            'ctc_loss_weight': 0.3,
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': 16,
                'num_classes': -2,
                'vocabulary': None,
            },
            'decoding': DictConfig(CTCBPEDecodingConfig),
        }
    return DictConfig(config)


def _emit_one_word_per_frame(model):
    """Make the joint always emit WORD for one frame, so emissions are dense and deterministic."""
    word = model.tokenizer.text_to_ids(WORD)
    assert len(word) == 1
    vocab = model.tokenizer.vocab_size
    with torch.no_grad():
        layer = model.joint.joint_net[-1]
        layer.weight.zero_()
        layer.bias.zero_()
        layer.bias[word[0]] = 10.0
        if layer.bias.numel() > vocab + 1:  # TDT durations follow the blank: choose duration 1.
            layer.bias[vocab + 2] = 10.0
        if hasattr(model, 'ctc_decoder'):
            conv = model.ctc_decoder.decoder_layers[0]
            conv.weight.zero_()
            conv.bias.zero_()
            conv.bias[word[0]] = 10.0


def _restore(model, tmp_path):
    path = str(tmp_path / f'{type(model).__name__}.nemo')
    model.save_to(path)
    return type(model).restore_from(path, map_location='cpu').eval()


def _sources(tmp_path):
    generator = torch.Generator().manual_seed(0)
    tensors = [torch.randn(n, generator=generator) * 0.1 for n in (32000, 52800)]  # 2.0 s and 3.3 s
    files = []
    for i, tensor in enumerate(tensors):
        files.append(str(tmp_path / f'source{i}.wav'))
        sf.write(files[-1], tensor.numpy(), 16000, subtype='FLOAT')
    return tensors, files


def _encoder_frames(model, source):
    with torch.inference_mode():
        return int(model.forward(input_signal=source[None], input_signal_length=torch.tensor([len(source)]))[1][0])


@pytest.mark.unit
@pytest.mark.parametrize('model_type', ['rnnt', 'tdt'])
def test_restored_transducer_chunks_sources_in_order(test_data_dir, tmp_path, model_type):
    model = EncDecRNNTBPEModel(cfg=_transducer_config(test_data_dir, model_type)).eval()
    _emit_one_word_per_frame(model)
    model = _restore(model, tmp_path)
    tensors, files = _sources(tmp_path)

    results = model.transcribe(tensors, batch_size=4, chunking=CHUNKING, timestamps=True, verbose=False)
    assert len(results) == 2
    for source, hyp in zip(tensors, results):
        words = hyp.timestamp['word']
        # One word per encoder frame: every frame is owned by exactly one window.
        assert abs(len(words) - _encoder_frames(model, source)) <= 1
        assert hyp.text.split() == [WORD] * len(words)
        starts = [w['start'] for w in words]
        assert starts == sorted(starts) and 0 <= starts[0] and starts[-1] <= len(source) / 16000
        assert hyp.score == 0.0 and hyp.token_confidence is None and hyp.alignments is None
    assert [h.text for h in model.transcribe(tensors, batch_size=1, chunking=CHUNKING, verbose=False)] == [
        h.text for h in results
    ]
    assert [h.text for h in model.transcribe(files, batch_size=3, chunking=CHUNKING, verbose=False)] == [
        h.text for h in results
    ]


@pytest.mark.unit
def test_restored_hybrid_chunks_with_both_heads(test_data_dir, tmp_path):
    model = EncDecHybridRNNTCTCBPEModel(cfg=_transducer_config(test_data_dir, 'rnnt', hybrid=True)).eval()
    _emit_one_word_per_frame(model)
    model = _restore(model, tmp_path)
    tensors, _ = _sources(tmp_path)

    model.change_decoding_strategy(decoder_type='rnnt', verbose=False)
    rnnt = model.transcribe(tensors, batch_size=4, chunking=CHUNKING, verbose=False)
    assert [len(h.text.split()) for h in rnnt] == pytest.approx([_encoder_frames(model, t) for t in tensors], abs=1)

    model.change_decoding_strategy(decoder_type='ctc', verbose=False)
    ctc = model.transcribe(tensors, batch_size=4, chunking=CHUNKING, verbose=False)
    plain = model.transcribe(tensors, batch_size=2, verbose=False)
    # Owned CTC frames are joined before one collapse, so chunking keeps the plain result here.
    assert [h.text for h in ctc] == [h.text for h in plain] == [WORD, WORD]


@pytest.mark.unit
@pytest.mark.parametrize('head', ['rnnt', 'hybrid-rnnt', 'hybrid-ctc'])
@pytest.mark.parametrize('timestamps', [True, False])
def test_restored_transducer_chunking_leaves_model_decoding_unchanged(test_data_dir, tmp_path, head, timestamps):
    if head == 'rnnt':
        model = EncDecRNNTBPEModel(cfg=_transducer_config(test_data_dir, 'rnnt')).eval()
    else:
        model = EncDecHybridRNNTCTCBPEModel(cfg=_transducer_config(test_data_dir, 'rnnt', hybrid=True)).eval()
        model.change_decoding_strategy(decoder_type=head.split('-')[1], verbose=False)
    model = _restore(model, tmp_path)
    tensors, _ = _sources(tmp_path)
    decoding, ctc_decoding = model.decoding, getattr(model, 'ctc_decoding', None)
    settings = OmegaConf.to_container(model.cfg)
    results = model.transcribe(tensors, chunking=CHUNKING, timestamps=timestamps, verbose=False)
    assert len(results) == 2 and bool(results[0].timestamp) is timestamps
    assert model.decoding is decoding and getattr(model, 'ctc_decoding', None) is ctc_decoding
    assert OmegaConf.to_container(model.cfg) == settings


@pytest.mark.unit
def test_transducer_with_aggregate_tokenizer_rejects_chunking(test_data_dir, tmp_path):
    config = _transducer_config(test_data_dir, 'rnnt')
    wpe = {'dir': config.tokenizer.dir, 'type': 'wpe'}
    config.tokenizer = {'dir': None, 'type': 'agg', 'langs': {'en': wpe, 'de': wpe}}
    model = EncDecRNNTBPEModel(cfg=config).eval()
    tensors, _ = _sources(tmp_path)
    assert len(model.transcribe(tensors, verbose=False)) == 2  # Plain transcription supports it.
    with pytest.raises(ValueError, match='aggregate tokenizer'):
        model.transcribe(tensors, chunking=CHUNKING, verbose=False)


def _aed_model(test_data_dir):
    tokenizers = os.path.join(test_data_dir, 'asr', 'tokenizers')
    config = DictConfig(
        {
            'prompt_format': 'canary',
            'prompt_defaults': [
                {'role': 'user', 'slots': {'source_lang': 'en', 'target_lang': 'en', 'task': 'asr', 'pnc': 'yes'}}
            ],
            'sample_rate': 16000,
            'preprocessor': _preprocessor(),
            'model_defaults': {'asr_enc_hidden': 16, 'lm_enc_hidden': 16, 'lm_dec_hidden': 16},
            'encoder': _conformer(),
            'transf_decoder': {
                '_target_': 'nemo.collections.asr.modules.transformer.get_nemo_transformer',
                'model_name': None,
                'pretrained': False,
                'encoder': None,
                'pre_ln_final_layer_norm': True,
                'config_dict': {
                    'max_sequence_length': 64,
                    'num_token_types': 0,
                    'hidden_size': 16,
                    'inner_size': 32,
                    'num_layers': 1,
                    'num_attention_heads': 2,
                    'pre_ln': True,
                    'vocab_size': None,
                },
            },
            'head': {
                '_target_': 'nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier',
                'num_layers': 1,
                'activation': 'relu',
                'log_softmax': True,
                'hidden_size': 16,
                'num_classes': None,
            },
            'tokenizer': {
                'dir': None,
                'type': 'agg',
                'langs': {
                    'spl_tokens': {'dir': os.path.join(tokenizers, 'canary'), 'type': 'bpe'},
                    'en': {'dir': os.path.join(tokenizers, 'an4_wpe_128'), 'type': 'wpe'},
                    'de': {'dir': os.path.join(tokenizers, 'an4_wpe_128'), 'type': 'wpe'},
                },
                'custom_tokenizer': {
                    '_target_': 'nemo.collections.common.tokenizers.canary_tokenizer.CanaryTokenizer',
                    'tokenizers': None,
                },
            },
            'decoding': {'strategy': 'beam', 'beam': {'beam_size': 1}},
            'loss': {
                '_target_': 'nemo.collections.common.losses.smoothed_cross_entropy.SmoothedCrossEntropyLoss',
                'label_smoothing': 0.0,
            },
        }
    )
    torch.manual_seed(0)
    return EncDecMultiTaskModel(cfg=config).eval()


def _ctc_timestamp_model(test_data_dir):
    config = DictConfig(
        {
            'sample_rate': 16000,
            'preprocessor': _preprocessor(),
            'encoder': _conformer(),
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': 16,
                'num_classes': -1,
                'vocabulary': None,
            },
            'tokenizer': {'dir': os.path.join(test_data_dir, 'asr', 'tokenizers', 'an4_wpe_128'), 'type': 'wpe'},
            'decoding': {'strategy': 'greedy_batch'},
        }
    )
    torch.manual_seed(1)
    return EncDecCTCModelBPE(cfg=config).eval()


@pytest.mark.unit
def test_restored_aed_chunks_with_ctc_word_timing(test_data_dir, tmp_path):
    model = _restore(_aed_model(test_data_dir), tmp_path)
    object.__setattr__(model, 'timestamps_asr_model', _ctc_timestamp_model(test_data_dir))
    tensors, files = _sources(tmp_path)

    results = model.transcribe(tensors, batch_size=4, chunking=CHUNKING, timestamps=True, verbose=False)
    assert len(results) == 2 and all(h.text for h in results)  # The checks below need real emissions.
    for source, hyp in zip(tensors, results):
        assert hyp.y_sequence.tolist() == model.tokenizer.text_to_ids(hyp.text, 'en')
        assert all(0 <= w['start'] <= w['end'] <= len(source) / 16000 + 0.2 for w in hyp.timestamp['word'])
        assert hyp.score == 0.0 and hyp.token_confidence is None

    # Bare and wrapped language names select the same German token IDs.
    for lang in ('de', '<|de|>'):
        german = model.transcribe(tensors, chunking=CHUNKING, source_lang=lang, target_lang=lang, verbose=False)
        assert all(h.text for h in german)
        assert [h.y_sequence.tolist() for h in german] == [model.tokenizer.text_to_ids(h.text, 'de') for h in german]
    with pytest.raises(ValueError, match='not speech translation'):
        model.transcribe(tensors, chunking=CHUNKING, source_lang='en', target_lang='de', verbose=False)

    # A manifest builds each file's prompt from its own entry; final IDs use that file's language.
    manifest = tmp_path / 'aed_manifest.json'
    entries = [
        {'audio_filepath': path, 'duration': len(source) / 16000, 'answer': '', 'task': 'asr', 'pnc': 'yes'}
        | {'source_lang': lang, 'target_lang': lang}
        for path, source, lang in zip(files, tensors, ('en', 'de'))
    ]
    manifest.write_text(''.join(json.dumps(entry) + '\n' for entry in entries))
    per_file = model.transcribe(str(manifest), batch_size=4, chunking=CHUNKING, verbose=False)
    assert len(per_file) == 2 and all(h.text for h in per_file)
    for hyp, lang in zip(per_file, ('en', 'de')):
        assert hyp.y_sequence.tolist() == model.tokenizer.text_to_ids(hyp.text, lang)


@pytest.mark.unit
def test_restored_aed_without_timestamp_model_stitches_windows(test_data_dir, tmp_path):
    model = _restore(_aed_model(test_data_dir), tmp_path)
    object.__setattr__(model, 'timestamps_asr_model', None)
    tensors, _ = _sources(tmp_path)
    results = model.transcribe(tensors, batch_size=4, chunking=CHUNKING, verbose=False)
    assert len(results) == 2 and all(h.text for h in results)
    assert all(h.y_sequence.tolist() == model.tokenizer.text_to_ids(h.text, 'en') for h in results)
    # An explicit text-only request needs no timestamp model.
    text_only = model.transcribe(tensors, batch_size=4, chunking=CHUNKING, timestamps=False, verbose=False)
    assert len(text_only) == 2 and all(h.text for h in text_only)
    with pytest.raises(ValueError, match='CTC timestamp model'):
        model.transcribe(tensors, chunking=CHUNKING, timestamps=True, verbose=False)


@pytest.mark.unit
def test_restored_aed_chunks_a_selected_channel_of_a_stereo_file(test_data_dir, tmp_path):
    model = _restore(_aed_model(test_data_dir), tmp_path)
    object.__setattr__(model, 'timestamps_asr_model', None)
    tensors, _ = _sources(tmp_path)
    other = torch.randn(len(tensors[1]), generator=torch.Generator().manual_seed(9)) * 0.1
    stereo, mono = str(tmp_path / 'stereo.wav'), str(tmp_path / 'mono.wav')
    sf.write(stereo, torch.stack([other, tensors[1]], dim=1).numpy(), 16000, subtype='FLOAT')
    sf.write(mono, tensors[1].numpy(), 16000, subtype='FLOAT')
    selected = model.transcribe(stereo, channel_selector=1, chunking=CHUNKING, verbose=False)
    reference = model.transcribe(mono, chunking=CHUNKING, verbose=False)
    assert selected[0].text and [h.text for h in selected] == [h.text for h in reference]
