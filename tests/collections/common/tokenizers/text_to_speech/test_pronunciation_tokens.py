# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import pickle

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPABPETokenizer

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on('CPU')]


@pytest.fixture
def shared_bpe(tmp_path):
    pieces = ['<pad>', '<unk>'] + sorted(ByteLevel.alphabet()) + ['ab', 'Ã©']
    backend = Tokenizer(BPE(vocab={piece: i for i, piece in enumerate(pieces)}, merges=[('a', 'b'), ('Ã', '©')]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = ByteLevelDecoder()
    backend.add_special_tokens([AddedToken('<pad>', special=True), AddedToken('<unk>', special=True)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token='<pad>', unk_token='<unk>', split_special_tokens=True
    )
    fast.save_pretrained(tmp_path)
    return tmp_path


def _write_sidecar(folder, **changes):
    data = dict(
        format_version=1,
        tokenizer_sha256=hashlib.sha256((folder / 'tokenizer.json').read_bytes()).hexdigest(),
        tokens=sorted(ByteLevel.alphabet()) + ['ab'],
    )
    data.update(changes)
    (folder / 'pronunciation_tokens.json').write_text(json.dumps(data))
    return data


def test_missing_sidecar_preserves_legacy_ids(shared_bpe):
    backend = Tokenizer.from_file(str(shared_bpe / 'tokenizer.json'))
    tokenizer = IPABPETokenizer(str(shared_bpe))
    assert not tokenizer.has_pronunciation_tokens
    assert tokenizer.encode('abé') == backend.encode('abé').ids
    assert tokenizer.bos == backend.get_vocab_size()
    assert tokenizer.vocab_size == backend.get_vocab_size() + 3
    assert tokenizer.pad == 0


@pytest.mark.parametrize('use_directory', [False, True])
def test_sidecar_compacts_output_only(shared_bpe, use_directory):
    data = _write_sidecar(shared_bpe)
    path = shared_bpe if use_directory else shared_bpe / 'tokenizer.json'
    tokenizer = IPABPETokenizer(str(path))
    content = len(data['tokens'])
    assert tokenizer.has_pronunciation_tokens
    assert tokenizer.vocab_size == content + 4
    assert [tokenizer.pad, tokenizer.bos, tokenizer.eos, tokenizer.unk_token_id] == list(range(content, content + 4))
    assert tokenizer.encode('ab') == [data['tokens'].index('ab')]
    text = AutoTokenizer.from_pretrained(shared_bpe)
    assert len(text) == len(tokenizer._tokenizer.shared_vocab)
    assert len(text) != tokenizer.vocab_size


@pytest.mark.parametrize('text', ['abé', '0カ1', 'বাংলা தமிழ் 中文 한국어 🙂', '<pad> <unk>', '', ' a\n b '])
def test_compact_fallback_roundtrips_and_token_api(shared_bpe, text):
    _write_sidecar(shared_bpe)
    tokenizer = IPABPETokenizer(str(shared_bpe))
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.text_to_ids(text) == ids
    assert tokenizer.tokens_to_ids(tokenizer.text_to_tokens(text)) == ids
    assert tokenizer.ids_to_tokens(ids) == tokenizer.text_to_tokens(text)
    assert tokenizer.tokens_to_text(tokenizer.text_to_tokens(text)) == text
    assert tokenizer.unk_token_id not in ids
    assert tokenizer.pad not in ids


def test_unlisted_merged_piece_falls_back_to_bytes(shared_bpe):
    _write_sidecar(shared_bpe)
    tokenizer = IPABPETokenizer(str(shared_bpe))
    assert len(tokenizer._tokenizer.base_tokenizer.encode('é').ids) == 1
    assert len(tokenizer.encode('é')) == 2
    assert tokenizer.decode([tokenizer.bos] + tokenizer.encode('abé') + [tokenizer.eos, tokenizer.pad]) == 'abé'


def test_spawn_serialization_preserves_literal_markers(shared_bpe):
    _write_sidecar(shared_bpe)
    tokenizer = pickle.loads(pickle.dumps(IPABPETokenizer(str(shared_bpe))))
    text = '<pad> <unk> abé'
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.pad not in tokenizer.encode(text)


@pytest.mark.parametrize(
    'changes,message',
    [
        ({'format_version': 2}, 'expected version 1'),
        ({'tokenizer_sha256': 'bad'}, 'does not match'),
        ({'tokens': ['a', 'a']}, 'duplicate'),
        ({'tokens': ['not-a-shared-piece']}, 'absent'),
        ({'tokens': ['a']}, '256 byte'),
        ({'tokens': ['<pad>']}, 'reserved'),
        ({'tokens': [1]}, 'token strings'),
    ],
)
def test_invalid_sidecar_fails_actionably(shared_bpe, changes, message):
    _write_sidecar(shared_bpe, **changes)
    with pytest.raises(ValueError, match=message):
        IPABPETokenizer(str(shared_bpe))
