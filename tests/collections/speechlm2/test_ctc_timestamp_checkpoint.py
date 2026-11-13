# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict native persistence of the frozen public CTC timestamp bundle."""
import json

import pytest
import sentencepiece as spm
import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from nemo.collections.speechlm2.parts.ctc_timestamp_checkpoint import CTCTimestampCheckpoint, read_ctc_timestamp_config
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import CTC_TIMESTAMP_ARTIFACT_FORMAT, TransformerCTCDecoder
from nemo.collections.speechlm2.parts.weight_loading import checkpoint_load_state, load_nemo_rl_checkpoint


# The released timestamp head adds a Transformer bridge, a residual scale buffer, and an integer
# ``upsample_factor`` buffer (stored as I64) on top of the plain projection.
_DECODER_CONFIGS = {
    'plain': {'feat_in': 8, 'use_transformer': False},
    'upsampled_transformer': {
        'feat_in': 32,
        'use_transformer': True,
        'n_heads': 2,
        'n_layers': 2,
        'residual': True,
        'upsampling_factor': 4,
        'upsampling_mode': 'repeat_phase',
    },
}


@pytest.fixture(params=sorted(_DECODER_CONFIGS))
def bundle(request, tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('hello world\nhello speech\nworld speech\n')
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / 'tokenizer'),
        vocab_size=16,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    proto = (tmp_path / 'tokenizer.model').read_bytes()
    processor = spm.SentencePieceProcessor(model_proto=proto)
    config = {
        **_DECODER_CONFIGS[request.param],
        'num_classes': processor.vocab_size(),
        'vocabulary': [processor.id_to_piece(i) for i in range(processor.vocab_size())],
    }
    decoder = TransformerCTCDecoder(**config)
    state = {'ctc_timestamp.decoder.' + name: value for name, value in decoder.state_dict().items()}
    state['ctc_timestamp.tokenizer.model_proto'] = torch.tensor(list(proto), dtype=torch.uint8)
    state['core.weight'] = torch.randn(2, 2, dtype=torch.bfloat16)
    metadata = {
        'ctc_timestamp_format': CTC_TIMESTAMP_ARTIFACT_FORMAT,
        'ctc_timestamp_decoder_config': json.dumps(config),
        'ctc_timestamp_tokenizer_config': json.dumps({'legacy': False}),
    }
    path = tmp_path / 'model.safetensors'
    save_file(state, path, metadata=metadata)
    return path, state, metadata


def make_model(config):
    model = torch.nn.Module()
    model.core = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.ctc_timestamp = CTCTimestampCheckpoint(config)
    return model


def test_bundle_native_resume_preserves_all_tensors_and_metadata(bundle, tmp_path):
    path, state, metadata = bundle
    config = read_ctc_timestamp_config(path)
    # Exercise configuration serialization used to reconstruct native export.
    config = OmegaConf.to_container(OmegaConf.create(json.loads(json.dumps(config))))
    model = make_model(config)
    model.load_state_dict(state, strict=True)
    model.ctc_timestamp.validate_tokenizer()
    model.train()
    assert not model.ctc_timestamp.training
    assert all(not p.requires_grad for p in model.ctc_timestamp.parameters())
    model.core.weight.sum().backward()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    dcp.save(model.state_dict(), checkpoint_id=tmp_path / 'native')
    resumed = make_model(config)
    load_nemo_rl_checkpoint(resumed, str(tmp_path / 'native'))
    exported = tmp_path / 'export.safetensors'
    save_file(resumed.state_dict(), exported, metadata=resumed.ctc_timestamp.artifact_metadata)
    with safe_open(exported, framework='pt') as saved:
        assert saved.metadata() == metadata
    for name, tensor in load_file(exported).items():
        if name.startswith('ctc_timestamp.'):
            assert tensor.dtype == state[name].dtype
            assert torch.equal(tensor, state[name].cpu())


@pytest.mark.parametrize('defect', ['core_missing', 'ctc_missing', 'ctc_unknown', 'shape'])
def test_strict_coverage_rejects_core_and_ctc_defects(bundle, defect):
    path, state, _ = bundle
    model = make_model(read_ctc_timestamp_config(path))
    shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    if defect == 'core_missing':
        del shapes['core.weight']
    elif defect == 'ctc_missing':
        del shapes['ctc_timestamp.tokenizer.model_proto']
    elif defect == 'ctc_unknown':
        shapes['ctc_timestamp.unknown'] = (2,)
    else:
        shapes['ctc_timestamp.decoder.decoder_layers.0.weight'] = (1,)
    with pytest.raises(ValueError):
        checkpoint_load_state(model, shapes)


def test_bundle_requires_recognized_metadata(bundle, tmp_path):
    _, state, _ = bundle
    path = tmp_path / 'unknown.safetensors'
    save_file(state, path)
    with pytest.raises(ValueError, match='metadata'):
        read_ctc_timestamp_config(path)


@pytest.mark.parametrize('bundle', ['upsampled_transformer'], indirect=True)
def test_integer_buffer_keeps_its_dtype(bundle):
    path, _, _ = bundle
    ctc_config = read_ctc_timestamp_config(path)
    assert ctc_config['tensor_dtypes']['decoder.upsample_factor'] == 'int64'
    module = CTCTimestampCheckpoint(ctc_config)
    assert module.decoder.upsample_factor.dtype == torch.int64

    # Integer storage for a floating-point tensor (or vice versa) is never silently cast.
    for name, dtype in (('decoder.upsample_factor', 'float32'), ('decoder.residual_scale', 'int64')):
        bad = json.loads(json.dumps(ctc_config))
        bad['tensor_dtypes'][name] = dtype
        with pytest.raises(ValueError, match=name):
            CTCTimestampCheckpoint(bad)


def test_unsupported_safetensors_dtype_raises_value_error(bundle, tmp_path):
    _, state, metadata = bundle
    state = dict(state)
    state['ctc_timestamp.decoder.decoder_layers.0.bias'] = state['ctc_timestamp.decoder.decoder_layers.0.bias'].to(
        torch.float8_e4m3fn
    )
    path = tmp_path / 'fp8.safetensors'
    save_file(state, path, metadata=metadata)
    with pytest.raises(ValueError, match='F8_E4M3'):
        read_ctc_timestamp_config(path)


def test_no_ctc_header_preserves_ordinary_checkpoint(tmp_path):
    path = tmp_path / 'ordinary.safetensors'
    save_file({'weight': torch.ones(2)}, path)
    assert read_ctc_timestamp_config(path) is None


def test_hub_mixin_roundtrip_keeps_ctc_metadata_and_uint8_bytes(bundle, tmp_path):
    from types import SimpleNamespace

    from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin

    class TinyModel(torch.nn.Module, HFHubMixin):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.perception = SimpleNamespace(encoder=SimpleNamespace(supports_ctc_timestamp_inputs=True))
            self.core = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            self.ctc_timestamp = CTCTimestampCheckpoint(cfg['ctc_timestamp_artifact'])

    path, state, metadata = bundle
    model = TinyModel({'ctc_timestamp_artifact': read_ctc_timestamp_config(path)})
    model.load_state_dict(state)
    output = tmp_path / 'hub'
    model.save_pretrained(output)
    restored = TinyModel.from_pretrained(str(output), strict=True)
    with safe_open(output / 'model.safetensors', framework='pt') as saved:
        assert saved.metadata() == metadata
    for name, tensor in restored.state_dict().items():
        assert tensor.dtype == state[name].dtype
        assert torch.equal(tensor, state[name])


def test_lightning_export_cast_keeps_ctc_storage(bundle):
    from examples.speechlm2.to_hf import cast_model_for_export

    path, state, _ = bundle
    model = make_model(read_ctc_timestamp_config(path))
    model.load_state_dict(state)
    model = cast_model_for_export(model, 'float16')
    assert model.core.weight.dtype == torch.float16
    assert all(not p.requires_grad for p in model.ctc_timestamp.parameters())
    for name, tensor in model.state_dict().items():
        if name.startswith('ctc_timestamp.'):
            assert tensor.dtype == state[name].dtype
            assert torch.equal(tensor, state[name])


def test_public_export_config_reconstructs_seed_artifact(bundle, tmp_path):
    from examples.speechlm2.prepare_rl_export_config import prepare_config

    path, _, metadata = bundle
    (tmp_path / 'config.json').write_text(json.dumps({'pretrained_llm': '/obsolete', 'tokenizer_path': '/obsolete'}))
    (tmp_path / 'llm_backbone').mkdir()
    (tmp_path / 'llm_backbone' / 'config.json').write_text('{}')
    config = prepare_config(tmp_path, 8, 8)
    assert config['model']['pretrained_llm'] == str(tmp_path / 'llm_backbone')
    assert config['model']['tokenizer_path'] == str(tmp_path)
    assert config['model']['ctc_timestamp_artifact'] == read_ctc_timestamp_config(path)
    assert config['model']['ctc_timestamp_artifact']['metadata'] == metadata
    assert config['trainer']['strategy']['ep_size'] == 8
    with pytest.raises(ValueError, match='divisible'):
        prepare_config(tmp_path, 8, 3)
