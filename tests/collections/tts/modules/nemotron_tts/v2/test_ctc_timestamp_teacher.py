# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.nemotron_tts.v2 import nemotron_asr_aligner as module
from nemo.core.classes import ModelPT
from tests.collections.tts.modules.nemotron_tts.v2.test_nemotron_asr_aligner import _teacher

pytestmark = pytest.mark.unit


def test_student_nemo_save_restore_preserves_tokenizer_and_head_without_teacher(tmp_path, monkeypatch):
    import sentencepiece as spm
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(['one two three four five', 'a multilingual sentence']),
        model_writer=writer,
        model_type='char',
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    proto = writer.getvalue()
    classes = spm.SentencePieceProcessor(model_proto=proto).vocab_size() + 1
    bundle = tmp_path / 'model.safetensors'
    save_file(
        {'ctc_timestamp.tokenizer.model_proto': torch.tensor(list(proto), dtype=torch.uint8)},
        str(bundle),
        metadata={
            'ctc_timestamp_format': 'nemo_ctc_timestamp_artifact_v1',
            'ctc_timestamp_decoder_config': json.dumps(dict(num_classes=classes - 1, upsampling_factor=2)),
            'ctc_timestamp_tokenizer_config': json.dumps(dict(legacy=False)),
        },
    )
    # Keep NeMo's target allowlist enabled while exposing only this fixture class.
    monkeypatch.setattr(module, '_TokenizerArtifactModel', _TokenizerArtifactModel, raising=False)
    student = _TokenizerArtifactModel(
        cfg=OmegaConf.create(
            dict(
                target=module.__name__ + '._TokenizerArtifactModel',
                ctc_timestamp_head=dict(aligner=dict(snapshot_path=str(tmp_path))),
            )
        )
    )
    expected_ids = student.ctc_timestamp_tokenizer.text_to_ids('one two')
    expected_weights = {name: value.detach().clone() for name, value in student.state_dict().items()}
    checkpoint = tmp_path / 'student.nemo'
    student.save_to(str(checkpoint))
    with tarfile.open(checkpoint) as archive:
        config_member = next(member for member in archive.getmembers() if member.name.endswith('model_config.yaml'))
        saved_config = OmegaConf.load(io.StringIO(archive.extractfile(config_member).read().decode()))
        artifact = saved_config.ctc_timestamp_head.tokenizer_model
        assert artifact.startswith('nemo:')
        artifact_name = artifact[len('nemo:') :]
        model_member = next(member for member in archive.getmembers() if member.name.endswith(artifact_name))
        assert archive.extractfile(model_member).read() == proto
    bundle.unlink()
    student._ctc_tokenizer_directory.cleanup()
    monkeypatch.setattr(
        module, '_load_teacher_components', Mock(side_effect=AssertionError('Unexpected teacher load'))
    )
    monkeypatch.setattr(module, '_load_selected_weights', Mock(side_effect=AssertionError('Unexpected weight read')))
    restored = _TokenizerArtifactModel.restore_from(str(checkpoint), map_location='cpu')
    assert restored.ctc_timestamp_tokenizer.text_to_ids('one two') == expected_ids
    assert restored.ctc_timestamp_head.vocab_size == classes
    assert restored.ctc_timestamp_head.blank_id == classes - 1
    assert restored.ctc_timestamp_head_config['frame_seconds'] == 0.04
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
    # Registered archive artifacts remain saveable after the restoration workspace is removed.
    second_checkpoint = tmp_path / 'student_resaved.nemo'
    restored.save_to(str(second_checkpoint))
    second = _TokenizerArtifactModel.restore_from(str(second_checkpoint), map_location='cpu')
    assert second.ctc_timestamp_tokenizer.text_to_ids('one two') == expected_ids


def test_tokenizer_metadata_reads_only_sentencepiece_and_roundtrips_without_teacher(tmp_path, monkeypatch):
    import sentencepiece as spm
    from safetensors.torch import save_file

    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(['one two three four five', 'a multilingual sentence']),
        model_writer=writer,
        model_type='char',
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    proto = writer.getvalue()
    processor = spm.SentencePieceProcessor(model_proto=proto)
    classes = processor.vocab_size() + 1
    bundle = tmp_path / 'model.safetensors'
    save_file(
        {
            'ctc_timestamp.tokenizer.model_proto': torch.tensor(list(proto), dtype=torch.uint8),
            'llm.unused': torch.ones(1),
        },
        str(bundle),
        metadata={
            'ctc_timestamp_format': 'nemo_ctc_timestamp_artifact_v1',
            'ctc_timestamp_decoder_config': json.dumps(dict(num_classes=classes - 1, upsampling_factor=2)),
            'ctc_timestamp_tokenizer_config': json.dumps(dict(legacy=False)),
        },
    )
    import safetensors

    original_open = safetensors.safe_open
    reads = []

    class Handle:
        def __enter__(self):
            self.handle = original_open(str(bundle), framework='pt', device='cpu').__enter__()
            return self

        def __exit__(self, *args):
            self.handle.__exit__(*args)

        def metadata(self):
            return self.handle.metadata()

        def get_tensor(self, name):
            reads.append(name)
            assert name == 'ctc_timestamp.tokenizer.model_proto'
            return self.handle.get_tensor(name)

    monkeypatch.setattr(safetensors, 'safe_open', lambda *args, **kwargs: Handle())
    tokenizer, metadata = module.load_ctc_timestamp_tokenizer(dict(aligner=dict(snapshot_path=str(tmp_path))))
    assert reads == ['ctc_timestamp.tokenizer.model_proto']
    assert metadata['num_classes'] == classes
    assert metadata['blank_id'] == classes - 1
    assert metadata['frame_seconds'] == 0.04 and metadata['upsampling_factor'] == 2
    assert metadata['model_proto'] == proto
    model_path = tmp_path / 'student_tokenizer.model'
    model_path.write_bytes(metadata.pop('model_proto'))
    bundle.unlink()
    restored, saved_metadata = module.load_ctc_timestamp_tokenizer(dict(metadata, tokenizer_model=str(model_path)))
    assert restored.text_to_ids('one two') == tokenizer.text_to_ids('one two')
    assert saved_metadata['model_proto'] == proto


@pytest.mark.parametrize('changes', [dict(upsampling_factor=1), dict(blank_id=2), dict(num_classes=99)])
def test_tokenizer_metadata_rejects_incompatible_dimensions(monkeypatch, changes):
    from nemo.collections.common.tokenizers import sentencepiece_tokenizer

    tokenizer = SimpleNamespace(vocab_size=4, tokenizer=SimpleNamespace(serialized_model_proto=lambda: b'model'))
    monkeypatch.setattr(sentencepiece_tokenizer, 'SentencePieceTokenizer', lambda **kwargs: tokenizer)
    config = dict(tokenizer_model='unused', num_classes=5, blank_id=4, upsampling_factor=2)
    with pytest.raises(ValueError):
        module.load_ctc_timestamp_tokenizer({**config, **changes})


def test_ctc_targets_keep_explicit_blank_and_tail_with_two_ordered_subframes(monkeypatch):
    teacher = _ctc_teacher(monkeypatch)
    audio = torch.arange(8).float().unsqueeze(0)
    with torch.inference_mode():
        records = teacher(
            audio,
            torch.tensor([8]),
            16000,
            ['one two'],
            ['en'],
            return_ctc=True,
            align_words=False,
            distillation_topk=1,
        )
    record = records[0]
    assert record['error'] is None and record['words'] == []
    targets = record['ctc']
    assert targets['token_ids'].tolist() == [0, 1]
    assert targets['frame_seconds'] == 0.04
    assert targets['num_classes'] == 5 and targets['blank_id'] == 4
    assert targets['indices'].shape == (16, 2)
    assert (targets['indices'][:, -1] == 4).all()
    inputs = module._timestamp_inputs(audio.unsqueeze(1), torch.tensor([8]))
    dense = teacher._get_aligner('whitespace').ctc_log_probs_from_inputs(inputs)[0].exp()
    expected = dense.gather(-1, targets['indices'])
    torch.testing.assert_close(targets['probabilities'], expected)
    torch.testing.assert_close(targets['tail_probabilities'], 1 - expected.sum(-1))
    assert not torch.equal(targets['indices'][0], targets['indices'][1])
    for value in targets.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == 'cpu' and not value.requires_grad and not value.is_inference()


def test_ctc_only_skips_word_preparation_and_dynamic_programming(monkeypatch):
    teacher = _ctc_teacher(monkeypatch)
    for aligner in [teacher._get_aligner('whitespace')]:
        aligner.prepare_from_inputs = Mock(side_effect=AssertionError('Unexpected word preparation'))
        aligner.align_prepared = Mock(side_effect=AssertionError('Unexpected word DP'))
    with torch.no_grad():
        records = teacher(
            torch.ones(1, 8), torch.tensor([8]), 16000, ['one'], ['en'], return_ctc=True, align_words=False
        )
    assert records[0]['error'] is None
    assert records[0]['ctc']['indices'].shape == (16, 5)
    torch.testing.assert_close(records[0]['ctc']['tail_probabilities'], torch.zeros(16), atol=1e-6, rtol=0)


def test_default_binary_path_never_requests_distillation(monkeypatch):
    teacher, _, _ = _teacher(monkeypatch)
    monkeypatch.setattr(
        module, '_ctc_distillation_targets', Mock(side_effect=AssertionError('Unexpected CTC targets'))
    )
    result = teacher(torch.ones(1, 8), torch.tensor([8]), 16000, ['unchanged'])
    assert result[0]['error'] is None and 'ctc' not in result[0]


def test_ctc_microbatch_order_padding_and_transcript_errors_are_independent(monkeypatch):
    teacher = _ctc_teacher(monkeypatch, max_batch_size=2)
    records = teacher(
        torch.ones(4, 8),
        torch.tensor([8, 5, 8, 0]),
        16000,
        ['one', 'two', 'one ' * 20, 'empty'],
        ['en'] * 4,
        return_ctc=True,
        align_words=False,
        distillation_topk=2,
    )
    assert records[0]['ctc']['token_ids'].tolist() == [0]
    assert records[1]['ctc']['token_ids'].tolist() == [1]
    assert records[0]['ctc']['indices'].shape[0] == 16
    assert records[1]['ctc']['indices'].shape[0] == 10
    assert 'ctc' not in records[2] and 'too long' in records[2]['error']
    assert records[3]['error'] == 'invalid_audio_length'


@pytest.mark.parametrize('topk', [0, -1, True, 1.5])
def test_ctc_teacher_rejects_invalid_sparse_budget(monkeypatch, topk):
    teacher = _ctc_teacher(monkeypatch)
    with pytest.raises(ValueError, match='distillation_topk'):
        teacher(
            torch.ones(1, 8),
            torch.tensor([8]),
            16000,
            ['one'],
            return_ctc=True,
            align_words=False,
            distillation_topk=topk,
        )


def _ctc_teacher(monkeypatch, **config):
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import MultiSpeakerSOTWordTimestampAligner

    teacher, _, _ = _teacher(monkeypatch, **config)
    teacher.ctc_decoder = _Decoder().eval().requires_grad_(False)
    vocabulary = {'one': 0, 'two': 1, 'three': 2, 'four': 3}
    teacher.tokenizer = SimpleNamespace(text_to_ids=lambda text: [vocabulary[word] for word in text.split()])
    teacher._alignment_options.update(online_inference_length=2, chunk_left_context=0, chunk_right_context=0)
    monkeypatch.setattr(
        module,
        '_make_aligner',
        lambda decoder, tokenizer, options: MultiSpeakerSOTWordTimestampAligner(
            ctc_decoder=decoder, tokenizer=tokenizer, **options
        ),
    )
    return teacher


class _Decoder(nn.Module):
    num_classes_with_blank = 5
    upsampling_factor = 2

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, states, encoded_lengths):
        values = states[:, 0].repeat_interleave(2, dim=1).float()
        phase = torch.arange(values.size(1), device=values.device) % 2
        logits = torch.stack(
            (values + 4 * phase, values + 4 * (1 - phase), -values, values / 5, values * 0 - 8), dim=-1
        )
        return logits.log_softmax(dim=-1)


class _TokenizerArtifactModel(ModelPT):
    """Exercise the production artifact initializer without loading a TTS backbone."""

    def __init__(self, cfg, trainer=None):
        from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBase

        super().__init__(cfg=cfg, trainer=trainer)
        self.decoder = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
        self.frame_stacking_factor = 1
        self.codec_model_samples_per_frame = 1280
        self.sample_rate = 16000
        self.ctc_timestamp_head_config = dict(cfg.ctc_timestamp_head)
        NemotronTTSBase._init_ctc_timestamp_head(self)

    @classmethod
    def list_available_models(cls):
        return []

    def setup_training_data(self, train_data_config):
        pass

    def setup_validation_data(self, val_data_config):
        pass
