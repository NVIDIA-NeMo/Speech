# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact pronunciation views over an unchanged byte-level BPE tokenizer."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel


class PronunciationTokenAdapter:
    """Remap a shared BPE's pronunciation pieces into a compact output vocabulary.

    The sidecar lists content pieces in compact-ID order. PAD follows those
    pieces; the enclosing ``IPABPETokenizer`` adds its normal BOS/EOS/UNK IDs.
    Unlisted BPE pieces fall back to the existing byte tokens without retraining
    or changing the shared input tokenizer.
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_file: str, pronunciation_file: str):
        with open(pronunciation_file, encoding='utf-8') as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get('format_version') != 1:
            raise ValueError(f"Unsupported pronunciation_tokens format in {pronunciation_file}; expected version 1")
        expected_hash = hashlib.sha256(Path(tokenizer_file).read_bytes()).hexdigest()
        if data.get('tokenizer_sha256') != expected_hash:
            raise ValueError(f"pronunciation_tokens tokenizer_sha256 does not match {tokenizer_file}")
        pieces = data.get('tokens')
        if (
            not isinstance(pieces, list)
            or not pieces
            or any(not isinstance(piece, str) or not piece for piece in pieces)
        ):
            raise ValueError("pronunciation_tokens.tokens must be a non-empty list of non-empty token strings")
        if len(set(pieces)) != len(pieces):
            raise ValueError("pronunciation_tokens.tokens contains duplicate pieces")
        if set(pieces) & {'<pad>', '<unk>', '<sp_bos>', '<sp_eos>', '<sp_unk>'}:
            raise ValueError("pronunciation_tokens.tokens must not include reserved padding or channel special tokens")
        state = json.loads(tokenizer.to_str())
        if state['model']['type'] != 'BPE' or (state.get('decoder') or {}).get('type') != 'ByteLevel':
            raise ValueError("pronunciation_tokens requires a byte-level BPE tokenizer")
        self.shared_vocab = tokenizer.get_vocab()
        missing = set(pieces) - self.shared_vocab.keys()
        if missing:
            raise ValueError(
                f"pronunciation_tokens contains pieces absent from the shared tokenizer: {sorted(missing)[:5]}"
            )
        if not set(ByteLevel.alphabet()) <= set(pieces):
            raise ValueError("pronunciation_tokens must retain all 256 byte-alphabet pieces for lossless fallback")
        if '<pad>' not in self.shared_vocab:
            raise ValueError("The shared tokenizer must define <pad> for the pronunciation channel")

        self.base_tokenizer = tokenizer
        self.base_tokenizer.encode_special_tokens = True
        self._tokens = {piece: index for index, piece in enumerate(pieces)}
        self._tokens['<pad>'] = len(pieces)
        self.compact_to_shared = [self.shared_vocab[piece] for piece in self._tokens]
        self.shared_to_compact = {shared: compact for compact, shared in enumerate(self.compact_to_shared)}
        self._shared_pieces = {index: piece for piece, index in self.shared_vocab.items()}
        self._compact_pieces = list(self._tokens)

    def get_vocab(self):
        """Return content and PAD IDs; channel BOS/EOS/UNK are added by the caller."""
        return dict(self._tokens)

    def __setstate__(self, state):
        # Tokenizers' pickle state omits this runtime flag. Dataloader spawn
        # must not turn literal <pad> text into an actual padding token.
        self.__dict__.update(state)
        self.base_tokenizer.encode_special_tokens = True

    def encode(self, text: str):
        """Return compact IDs, splitting unlisted shared pieces into byte tokens."""
        result = []
        for shared in self.base_tokenizer.encode(text, add_special_tokens=False).ids:
            compact = self.shared_to_compact.get(shared)
            if compact is not None:
                result.append(compact)
            else:
                result.extend(self._tokens[byte] for byte in self._shared_pieces[shared])
        return _PronunciationEncoding(result, [self._compact_pieces[index] for index in result])

    def decode(self, tokens):
        """Decode compact IDs, omitting the enclosing tokenizer's channel specials."""
        shared = [self.compact_to_shared[index] for index in tokens if 0 <= index < len(self.compact_to_shared)]
        return self.base_tokenizer.decode(shared)


@dataclass
class _PronunciationEncoding:
    ids: list[int]
    tokens: list[str]
