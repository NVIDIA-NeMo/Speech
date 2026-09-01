# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json
import logging
import math
import os
import random
import tarfile
from collections import deque
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import Iterator, Literal, Optional, Sequence, Union

import numpy as np
import torch
from lhotse import AudioSource, CutSet, Recording
from lhotse.audio import AudioLoadingError
from lhotse.custom import CustomFieldMixin
from lhotse.cut import Cut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import read_audio_from_cuts
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.input_strategies import _get_executor
from lhotse.serialization import load_jsonl, open_best
from lhotse.shar import AudioTarWriter, JsonlShardWriter
from lhotse.utils import Pathlike, compute_num_samples, fastcopy, is_valid_url

from nemo.collections.common.data.lhotse._compat import (
    IteratorNode,
    PartitionedIndexedIterator,
    attach_graph_origin,
    normalize_graph_token,
)
from nemo.collections.common.data.lhotse.audio_path_resolver import AudioPathPrefixMap
from nemo.collections.common.data.lhotse.indexed_adapters import (
    IndexedTarMemberReader,
    IndexedTarSampleBundleReader,
    IndexedTarSampleReader,
    PackedTarMemberReader,
    PackedTarSampleBundleReader,
    _split_json_audio_pair,
    wds_v2_index_path,
)
from nemo.collections.common.data.lhotse.nemo_adapters import (
    expand_sharded_filepaths,
    manifest_entry_is_explicitly_skipped,
)
from nemo.collections.common.data.lhotse.wds_catalog import discover_webdataset_shards
from nemo.collections.common.data.prompt_fn import apply_prompt_format_fn, registered_prompt_format_fn
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper

"""
Formattable: mixin class with data fields for prompt formatter outputs and method for
applying prompt formatters to derived data types.
"""


class Formattable:
    def __init__(self):
        self.input_ids: np.ndarray | torch.Tensor | None = None
        self.context_ids: np.ndarray | torch.Tensor | None = None
        self.answer_ids: np.ndarray | torch.Tensor | None = None
        self.mask: np.ndarray | torch.Tensor | None = None

    @property
    def input_length(self) -> int | None:
        if self.context_ids is None:
            return None
        return self.context_ids.shape[0]

    @property
    def output_length(self) -> int | None:
        if self.answer_ids is None:
            return None
        return self.answer_ids.shape[0]

    @property
    def total_length(self) -> int | None:
        if self.input_ids is None:
            return None
        return self.input_ids.shape[0]

    def apply_prompt_format(self, prompt) -> "Formattable":
        ans = apply_prompt_format_fn(self, prompt)
        self.input_ids = ans["input_ids"]
        self.context_ids = ans["context_ids"]
        self.answer_ids = ans.get("answer_ids")
        self.mask = ans.get("mask")
        return self


"""
TextExample: data types, file parser, default prompt formatting logic.
"""


@dataclass
class TextExample(Formattable, CustomFieldMixin):
    """
    Represents a single text example. Useful e.g. for language modeling.
    """

    text: str
    language: str | None = None
    tokens: Optional[np.ndarray] = None
    custom: dict = None

    def tokenize(self, tokenizer: TokenizerWrapper) -> "TextExample":
        self.tokens = np.asarray(tokenizer(self.text, self.language))
        return self


@dataclass
class LhotseTextAdapter:
    """
    ``LhotseTextAdapter`` is used to read a text file and wrap
    each line into a ``TextExample``.
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)

    def __iter__(self) -> Iterator[TextExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            with open(path) as f:
                for line in f:
                    yield TextExample(line, language=self.language)


@dataclass
class LhotseTextJsonlAdapter(IteratorNode):
    """
    ``LhotseTextJsonlAdapter`` is used to read a JSONL file and wrap
    the text field of each line into a ``TextExample``.

    Set ``indexed=True`` to enable O(1) random access plus graph-token
    checkpointing (requires uncompressed ``.jsonl`` paths).
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    text_field: str = "text"
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    indexed: bool = False
    indexes_root: Optional[Pathlike] = None

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)
        self._readers: list = []
        self._cum_lens: list[int] = []
        self._iter_state = PartitionedIndexedIterator()
        if self.indexed:
            from lhotse.indexing import IndexedJsonlReader, index_file_path

            for p in self.paths:
                self._readers.append(IndexedJsonlReader(p, index_path=index_file_path(p, self.indexes_root)))
            cum = 0
            self._cum_lens.append(cum)
            for r in self._readers:
                cum += len(r)
                self._cum_lens.append(cum)

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def __len__(self) -> int:
        if not self.indexed:
            raise TypeError("LhotseTextJsonlAdapter has unknown length unless constructed with indexed=True.")
        return self._cum_lens[-1] if self._cum_lens else 0

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._cum_lens[-1]
        for s in range(len(self._readers)):
            if idx < self._cum_lens[s + 1]:
                return s, idx - self._cum_lens[s]
        raise IndexError(idx)

    def _data_to_example(self, data: dict) -> TextExample | None:
        if self.text_field not in data:
            return None
        return TextExample(data[self.text_field], language=self.language)

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError("LhotseTextJsonlAdapter only supports __getitem__ when indexed=True.")
        idx = int(normalize_graph_token(token))
        shard_idx, local_idx = self._resolve(idx)
        ex = self._data_to_example(self._readers[shard_idx][local_idx])
        if ex is None:
            raise IndexError(
                f"Index {idx} in {self.paths[shard_idx]} has no '{self.text_field}' field; "
                f"cannot satisfy random-access __getitem__."
            )
        return attach_graph_origin(ex, idx)

    def state_dict(self) -> dict:
        return self._iter_state.state_dict() if self.indexed else {}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)

    def __iter__(self) -> Iterator[TextExample]:
        if self.indexed:
            yield from self._iter_indexed()
        else:
            yield from self._iter_streaming()

    def _iter_indexed(self) -> Iterator[TextExample]:
        total = self._cum_lens[-1] if self._cum_lens else 0
        for global_idx in self._iter_state.iterate(total):
            shard_idx, local_idx = self._resolve(global_idx)
            ex = self._data_to_example(self._readers[shard_idx][local_idx])
            if ex is None:
                continue
            attach_graph_origin(ex, global_idx)
            yield ex

    def _iter_streaming(self) -> Iterator[TextExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                if self.text_field not in data:
                    continue
                yield TextExample(data[self.text_field], language=self.language)


@registered_prompt_format_fn(TextExample)
def default_text_example_prompt_format_fn(example: TextExample, prompt):
    # It doesn't really make sense to prompt format a single line text example,
    # but we implement some default logic for the sake of completeness.
    # The default logic here is to treat the whole example as an assistant turn,
    # so that the mask is all set to true for the training loss.
    return prompt.encode_dialog(
        [
            {"role": prompt.OUTPUT_ROLE, "slots": {"message": example.text}},
        ]
    )


"""
SourceTargetTextExample: data types, file parser, default prompt formatting logic.
"""


@dataclass
class SourceTargetTextExample(Formattable, CustomFieldMixin):
    """
    Represents a pair of text examples. Useful e.g. for sequence-to-sequence tasks.
    Supports a ``question`` field, used as the prompt for LLM.
    """

    source: TextExample
    target: TextExample
    question: TextExample | None = None
    custom: dict = None

    def tokenize(self, tokenizer: TokenizerWrapper) -> "SourceTargetTextExample":
        self.source = self.source.tokenize(tokenizer)
        self.target = self.target.tokenize(tokenizer)
        if self.question is not None:
            self.question = self.question.tokenize(tokenizer)
        return self


@dataclass
class LhotseTextPairAdapter:
    """
    ``LhotseTextAdapter`` is used to read a tuple of N text files
    (e.g., a pair of files with translations in different languages)
    and wrap them in a ``TextExample`` object to enable dataloading
    with Lhotse together with training examples in audio modality.

    Provide ``questions_path`` to enable randomly sampling lines with questions.
    """

    source_paths: Union[Pathlike, list[Pathlike]]
    target_paths: Union[Pathlike, list[Pathlike]]
    source_language: str | None = None
    target_language: str | None = None
    questions_path: Pathlike = None
    questions_language: str = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        ASSERT_MSG = "Both source and target must be a single path or lists of paths"
        if isinstance(self.source_paths, (str, Path)):
            assert isinstance(self.target_paths, (str, Path)), ASSERT_MSG
        else:
            assert isinstance(self.source_paths, list) and isinstance(self.target_paths, list), ASSERT_MSG
            assert len(self.source_paths) == len(
                self.target_paths
            ), f"Source ({len(self.source_paths)}) and target ({len(self.target_paths)}) path lists must have the same number of items."
        self.source_paths = expand_sharded_filepaths(self.source_paths)
        self.target_paths = expand_sharded_filepaths(self.target_paths)

    def __iter__(self) -> Iterator[SourceTargetTextExample]:
        seed = resolve_seed(self.shard_seed)
        rng = random.Random(seed)
        paths = list(zip(self.source_paths, self.target_paths))
        if self.shuffle_shards:
            rng.shuffle(paths)
        questions = None
        if self.questions_path is not None:
            with open(self.questions_path) as f:
                questions = [q.strip() for q in f]
        for source_path, target_path in paths:
            with open(source_path) as fs, open(target_path) as ft:
                for ls, lt in zip(fs, ft):
                    yield SourceTargetTextExample(
                        source=TextExample(ls.strip(), language=self.source_language),
                        target=TextExample(lt.strip(), language=self.target_language),
                        question=(
                            TextExample(rng.choice(questions), language=self.questions_language)
                            if questions is not None
                            else None
                        ),
                    )


@registered_prompt_format_fn(SourceTargetTextExample)
def default_src_tgt_prompt_format_fn(example: SourceTargetTextExample, prompt):
    if example.question is not None:
        ctx = f"{example.question.text} {example.source.text}"
    else:
        ctx = example.source.text
    return prompt.encode_dialog(
        [
            {"role": "user", "slots": {"message": ctx}},
            {"role": prompt.OUTPUT_ROLE, "slots": {"message": example.target.text}},
        ]
    )


"""
NeMoSFTExample: data types, file parser, default prompt formatting logic.
"""


@dataclass
class NeMoSFTExample(Formattable, CustomFieldMixin):
    data: dict
    language: str | None = None
    metadata: dict | None = None
    custom: dict = None


@registered_prompt_format_fn(NeMoSFTExample)
def default_sft_prompt_format_fn(example: NeMoSFTExample, prompt):
    if "system" in example.data and example.data["system"]:
        raise RuntimeError(
            f"Default prompt format for NeMoSFTExample doesn't support 'system' prompt. "
            f"Please specialize the prompt_format_fn for PromptFormatter of type {prompt}"
        )
    return prompt.encode_dialog(
        [
            {"role": "user" if turn["from"] == "User" else prompt.OUTPUT_ROLE, "slots": {"message": turn["value"]}}
            for turn in example.data["conversations"]
        ]
    )


@dataclass
class NeMoSFTJsonlAdapter(IteratorNode):
    """
    ``NeMoSFTJsonlAdapter`` is used to read a NeMo LM SFT Chat JSONL file and yield objects of type
    ``NeMoSFTExample`` that can be sampled with Lhotse.

    We expect the following schema (contained in a single line per example)::

        {
            "conversations": [
                {
                    "value": str,
                    "from": "User" | "Assistant",
                    "canonical_form": str,
                    "label": str | null
                },
                ...
            ],
            "mask": "User" | "Assistant",
            "system": str,
            "dataset": str,
            "category": str,
        }

    Set ``indexed=True`` to enable O(1) random access plus graph-token
    checkpointing (requires uncompressed ``.jsonl`` paths).
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    indexed: bool = False
    indexes_root: Optional[Pathlike] = None

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)
        self._readers: list = []
        self._cum_lens: list[int] = []
        self._iter_state = PartitionedIndexedIterator()
        if self.indexed:
            from lhotse.indexing import IndexedJsonlReader, index_file_path

            for p in self.paths:
                self._readers.append(IndexedJsonlReader(p, index_path=index_file_path(p, self.indexes_root)))
            cum = 0
            self._cum_lens.append(cum)
            for r in self._readers:
                cum += len(r)
                self._cum_lens.append(cum)

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def __len__(self) -> int:
        if not self.indexed:
            raise TypeError("NeMoSFTJsonlAdapter has unknown length unless constructed with indexed=True.")
        return self._cum_lens[-1] if self._cum_lens else 0

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._cum_lens[-1]
        for s in range(len(self._readers)):
            if idx < self._cum_lens[s + 1]:
                return s, idx - self._cum_lens[s]
        raise IndexError(idx)

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError("NeMoSFTJsonlAdapter only supports __getitem__ when indexed=True.")
        idx = int(normalize_graph_token(token))
        shard_idx, local_idx = self._resolve(idx)
        ex = NeMoSFTExample(self._readers[shard_idx][local_idx], language=self.language)
        return attach_graph_origin(ex, idx)

    def state_dict(self) -> dict:
        return self._iter_state.state_dict() if self.indexed else {}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)

    def __iter__(self) -> Iterator[NeMoSFTExample]:
        if self.indexed:
            yield from self._iter_indexed()
        else:
            yield from self._iter_streaming()

    def _iter_indexed(self) -> Iterator[NeMoSFTExample]:
        total = self._cum_lens[-1] if self._cum_lens else 0
        for global_idx in self._iter_state.iterate(total):
            shard_idx, local_idx = self._resolve(global_idx)
            ex = NeMoSFTExample(self._readers[shard_idx][local_idx], language=self.language)
            attach_graph_origin(ex, global_idx)
            yield ex

    def _iter_streaming(self) -> Iterator[NeMoSFTExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                yield NeMoSFTExample(data, language=self.language)


_MATERIALIZED_SFT_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass
class MaterializedSFTMessagesExample(Formattable, CustomFieldMixin):
    """A packed SFT row whose message contents already contain the wire prompt.

    ``messages`` are kept as one example even when the row contains multiple
    conversations.  Each message's ``content`` is tokenized verbatim; its
    top-level ``role`` controls the loss mask.  In particular, ChatML-looking
    text nested inside ``content`` is data, not another role annotation.
    """

    id: str
    messages: list[dict]
    validate_chunk_tokenization: bool = True
    custom: dict = None

    def tokenize(self, tokenizer: TokenizerWrapper) -> "MaterializedSFTMessagesExample":
        encoded = _encode_materialized_sft_messages(
            self,
            lambda content: list(tokenizer(content)),
        )
        _apply_materialized_sft_encoding(self, encoded)
        return self


def _encode_materialized_sft_messages(example: MaterializedSFTMessagesExample, encode) -> dict[str, torch.Tensor]:
    messages = example.messages
    if not isinstance(messages, list) or not messages:
        size = len(messages) if hasattr(messages, "__len__") else None
        raise ValueError(
            f"Materialized SFT example id={example.id!r} must contain a non-empty messages list; "
            f"got type={type(messages).__name__} size={size}"
        )

    contents = []
    chunk_token_ids = []
    chunk_mask_values = []
    for index, message in enumerate(example.messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Materialized SFT example id={example.id!r} message {index} must be a mapping, "
                f"got {type(message).__name__}"
            )
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"Materialized SFT example id={example.id!r} message {index} must have exactly "
                f"the keys role/content, got {sorted(message)}"
            )
        role = message["role"]
        content = message["content"]
        if role not in _MATERIALIZED_SFT_ROLES:
            raise ValueError(
                f"Materialized SFT example id={example.id!r} message {index} has unsupported role={role!r}"
            )
        if not isinstance(content, str):
            raise ValueError(
                f"Materialized SFT example id={example.id!r} message {index} content must be a string, "
                f"got {type(content).__name__}"
            )
        token_ids = list(encode(content))
        contents.append(content)
        chunk_token_ids.append(token_ids)
        chunk_mask_values.append(role == "assistant")

    input_ids = [token for chunk in chunk_token_ids for token in chunk]
    if example.validate_chunk_tokenization:
        concatenated_ids = list(encode("".join(contents)))
        if concatenated_ids != input_ids:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(zip(concatenated_ids, input_ids))
                    if actual != expected
                ),
                min(len(concatenated_ids), len(input_ids)),
            )
            raise ValueError(
                "Materialized SFT chunk tokenization is not concatenation-stable for "
                f"example id={example.id!r}: concatenated_tokens={len(concatenated_ids)} "
                f"chunk_tokens={len(input_ids)} first_mismatch={mismatch}. Refusing to alter "
                "the production identity-preformatted token stream."
            )

    mask = [
        is_assistant
        for is_assistant, token_ids in zip(chunk_mask_values, chunk_token_ids)
        for _ in token_ids
    ]
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    return {
        "input_ids": input_tensor,
        # Packed rows can contain many disjoint assistant spans, so there is no
        # single prefix/answer split.  SALM trains from input_ids + mask; these
        # fields retain useful length semantics for generic Lhotse filters.
        "context_ids": input_tensor,
        "answer_ids": input_tensor[mask_tensor],
        "mask": mask_tensor,
    }


def _apply_materialized_sft_encoding(example: MaterializedSFTMessagesExample, encoded: dict) -> None:
    for key, value in encoded.items():
        setattr(example, key, value)


@registered_prompt_format_fn(MaterializedSFTMessagesExample)
def materialized_sft_messages_prompt_format_fn(example: MaterializedSFTMessagesExample, prompt, **kwargs):
    """Bypass chat templating while retaining the recipe tokenizer.

    The source has already been rendered with the upstream SFT chat template.
    Applying ``prompt.encode_dialog`` here would double-materialize it.
    """

    return _encode_materialized_sft_messages(example, prompt._apply_tokenizer)


def _transform_materialized_sft_messages(
    data: dict,
    sample_id: str,
    *,
    validate_chunk_tokenization: bool,
) -> MaterializedSFTMessagesExample:
    if not isinstance(data, dict):
        raise ValueError(f"Materialized SFT sample id={sample_id} must be a mapping")
    if set(data) != {"messages"}:
        raise ValueError(
            f"Materialized SFT sample id={sample_id} must have exactly the top-level key 'messages', "
            f"got {sorted(data)}"
        )
    return MaterializedSFTMessagesExample(
        id=sample_id,
        messages=data["messages"],
        validate_chunk_tokenization=validate_chunk_tokenization,
    )


def _normalize_nemotron_text_sender(sender: str, sample_id: str) -> str:
    role = str(sender).lower()
    if role in ("user", "human"):
        return "user"
    if role in ("assistant", "gpt", "model", "bot"):
        return "assistant"
    if role == "system":
        return "system"
    if role == "tool":
        return "tool"
    raise ValueError(f"Unsupported sender={sender!r} in Nemotron text conversation sample id={sample_id}")


def _flatten_nemotron_text_fragments(fragments: list, sample_id: str) -> str:
    values = []
    for fragment in fragments:
        if isinstance(fragment, str):
            values.append(fragment)
            continue
        if not isinstance(fragment, dict):
            raise ValueError(
                f"Unsupported fragment type={type(fragment).__name__} in Nemotron text conversation sample id={sample_id}"
            )
        fragment_type = fragment.get("t")
        if fragment_type not in (None, "text"):
            raise ValueError(
                f"Unsupported fragment t={fragment_type!r} in Nemotron text conversation sample id={sample_id}"
            )
        values.append(str(fragment.get("value", "")))
    return "".join(values)


def _transform_nemotron_text_conversation(data: dict, sample_id: str) -> "NeMoMultimodalConversation":
    conversation = data.get("conversation")
    if not isinstance(conversation, list):
        raise ValueError(f"Nemotron text conversation sample id={sample_id} has no list-valued 'conversation' field")

    turns = []
    for turn in conversation:
        if not isinstance(turn, dict):
            raise ValueError(
                f"Unsupported turn type={type(turn).__name__} in Nemotron text conversation sample id={sample_id}"
            )
        role = _normalize_nemotron_text_sender(turn.get("sender"), sample_id)
        value = _flatten_nemotron_text_fragments(turn.get("fragments", []), sample_id)
        turns.append(TextTurn(value=value, role=role))
    return NeMoMultimodalConversation(
        id=str(data.get("id") or sample_id),
        turns=turns,
        custom=data.get("custom"),
    )


@dataclass
class NemotronTextConversationAdapter(IteratorNode):
    """
    Read Nemotron/Energon text-only conversation data.

    Supported inputs are JSONL files and materialized tar directories whose JSON
    rows contain ``conversation`` turns with ``sender`` and ``fragments`` fields.
    """

    paths: Union[Pathlike, list[Pathlike]]
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    indexed: bool = False
    indexes_root: Optional[Pathlike] = None
    index_pack: Optional[Pathlike] = None
    index_pack_max_open_files: int = 32

    def __post_init__(self):
        raw_paths = self.paths
        self._readers: list = []
        self._reader_kinds: list[str] = []
        self._source_paths: list[str] = []
        self._cum_lens: list[int] = []
        self._iter_state = PartitionedIndexedIterator()
        if self.indexed and self.index_pack is not None:
            self._init_packed(raw_paths)
            return
        paths = [raw_paths] if isinstance(raw_paths, (str, Path)) else list(raw_paths)
        self.paths = [str(p) for raw in paths for p in expand_sharded_filepaths(str(raw))]
        if self.indexed:
            self._init_indexed()

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def _init_packed(self, source_spec) -> None:
        from lhotse.index_pack import index_pack_collection_key, open_index_pack
        from lhotse.packed_lazy import LazyPackedManifestIterator

        pack = open_index_pack(self.index_pack)
        collection_specs = (
            ("packed_jsonl", "jsonl"),
            ("packed_tar", "nemo_tar"),
        )
        found = []
        for reader_kind, pack_kind in collection_specs:
            key = index_pack_collection_key("paths", pack_kind, source_spec)
            try:
                collection = pack.collection(key)
            except KeyError:
                continue
            if reader_kind == "packed_jsonl":
                reader = LazyPackedManifestIterator(
                    pack,
                    key,
                    decode=dict,
                    max_open_files=self.index_pack_max_open_files,
                )
            else:
                reader = PackedTarMemberReader(collection, max_open_files=self.index_pack_max_open_files)
            self._readers.append(reader)
            self._reader_kinds.append(reader_kind)
            self._source_paths.append(str(pack.path))
            found.append(pack_kind)
        if not found:
            raise KeyError(f"Index pack {pack.path} has no paths collection for {source_spec!r}")
        if len(found) != 1:
            raise ValueError(
                "A packed Nemotron text source must be homogeneous; "
                f"found collections for {found}. Split mixed JSONL/tar paths "
                "into separate dataset entries."
            )
        self._finish_indexed_init()
        self._packed_indexed = True

    def _init_indexed(self) -> None:
        from lhotse.indexing import IndexedJsonlReader, index_file_path

        for p in self.paths:
            path = Path(p)
            if path.is_dir():
                tar_paths = sorted(path.rglob("*.tar"))
                if not tar_paths:
                    raise FileNotFoundError(f"No .tar files found under Nemotron text conversation directory: {path}")
                for tar_path in tar_paths:
                    self._add_indexed_tar_reader(str(tar_path), index_file_path(str(tar_path), self.indexes_root))
            elif path.suffix == ".tar":
                self._add_indexed_tar_reader(p, index_file_path(p, self.indexes_root))
            else:
                self._readers.append(IndexedJsonlReader(p, index_path=index_file_path(p, self.indexes_root)))
                self._reader_kinds.append("jsonl")
                self._source_paths.append(p)
        self._finish_indexed_init()

    def _finish_indexed_init(self) -> None:
        cum = 0
        self._cum_lens.append(cum)
        for reader in self._readers:
            cum += len(reader)
            self._cum_lens.append(cum)

    def _add_indexed_tar_reader(self, tar_path: str, idx_path: Pathlike) -> None:
        self._readers.append(IndexedTarMemberReader(tar_path, idx_path=idx_path))
        self._reader_kinds.append("tar")
        self._source_paths.append(tar_path)

    def __len__(self) -> int:
        if not self.indexed:
            raise TypeError("NemotronTextConversationAdapter has unknown length unless constructed with indexed=True.")
        return self._cum_lens[-1] if self._cum_lens else 0

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._cum_lens[-1]
        for shard_idx in range(len(self._readers)):
            if idx < self._cum_lens[shard_idx + 1]:
                return shard_idx, idx - self._cum_lens[shard_idx]
        raise IndexError(idx)

    def _data_to_conversation(
        self, data: dict, source_path: Union[str, Path], local_idx: int
    ) -> "NeMoMultimodalConversation":
        sample_id = f"{Path(source_path).stem}-{local_idx:012d}"
        return _transform_nemotron_text_conversation(data, sample_id)

    def _reader_item_to_conversation(self, shard_idx: int, local_idx: int) -> "NeMoMultimodalConversation":
        reader = self._readers[shard_idx]
        reader_kind = self._reader_kinds[shard_idx]
        source_path = self._source_paths[shard_idx]
        if reader_kind == "packed_tar":
            item, location = reader.read_with_location(local_idx)
            source_path = location.path
        else:
            item = reader[local_idx]
        if reader_kind == "packed_jsonl":
            location = reader.collection.locate(local_idx)
            return self._data_to_conversation(item, location.path, location.local_index)
        if reader_kind in ("tar", "packed_tar"):
            name, payload = item
            if not name.endswith(".json"):
                raise RuntimeError(
                    f"Index {local_idx} in {source_path} points to non-JSON tar member {name!r}; "
                    "Nemotron text conversation tar shards are expected to contain JSON samples."
                )
            return _transform_nemotron_text_conversation(json.loads(payload), Path(name).stem)
        return self._data_to_conversation(item, source_path, local_idx)

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError("NemotronTextConversationAdapter only supports __getitem__ when indexed=True.")
        idx = int(normalize_graph_token(token))
        shard_idx, local_idx = self._resolve(idx)
        conversation = self._reader_item_to_conversation(shard_idx, local_idx)
        return attach_graph_origin(conversation, idx)

    def state_dict(self) -> dict:
        return self._iter_state.state_dict() if self.indexed else {}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)

    def __iter__(self) -> Iterator["NeMoMultimodalConversation"]:
        if self.indexed:
            yield from self._iter_indexed()
            return
        yield from self._iter_streaming()

    def _iter_indexed(self) -> Iterator["NeMoMultimodalConversation"]:
        total = self._cum_lens[-1] if self._cum_lens else 0
        for global_idx in self._iter_state.iterate(total):
            shard_idx, local_idx = self._resolve(global_idx)
            conversation = self._reader_item_to_conversation(shard_idx, local_idx)
            attach_graph_origin(conversation, global_idx)
            yield conversation

    def _iter_streaming(self) -> Iterator["NeMoMultimodalConversation"]:
        paths = list(self.paths)
        if self.shuffle_shards:
            random.Random(resolve_seed(self.shard_seed)).shuffle(paths)
        for path in paths:
            yield from self._iter_path(Path(path))

    def _iter_path(self, path: Path) -> Iterator["NeMoMultimodalConversation"]:
        if path.is_dir():
            tar_paths = sorted(path.rglob("*.tar"))
            if not tar_paths:
                raise FileNotFoundError(f"No .tar files found under Nemotron text conversation directory: {path}")
            for tar_path in tar_paths:
                yield from self._iter_tar(tar_path)
        elif path.suffix == ".tar":
            yield from self._iter_tar(path)
        else:
            yield from self._iter_jsonl(path)

    def _iter_jsonl(self, path: Path) -> Iterator["NeMoMultimodalConversation"]:
        for idx, data in enumerate(load_jsonl(path)):
            sample_id = f"{path.stem}-{idx:012d}"
            yield _transform_nemotron_text_conversation(data, sample_id)

    def _iter_tar(self, path: Path) -> Iterator["NeMoMultimodalConversation"]:
        with tarfile.open(path, "r:*") as tar:
            for info in tar:
                if not info.isfile() or not info.name.endswith(".json"):
                    continue
                data = json.load(tar.extractfile(info))
                sample_id = Path(info.name).stem
                yield _transform_nemotron_text_conversation(data, sample_id)


@dataclass
class MaterializedSFTMessagesAdapter(NemotronTextConversationAdapter):
    """Read identity-preformatted, conversation-packed SFT JSONL rows.

    The indexed implementation intentionally reuses the proven JSONL/idxpack
    mechanics of :class:`NemotronTextConversationAdapter`.  Tar inputs are
    rejected because the production contract is line-addressable JSONL.
    """

    validate_chunk_tokenization: bool = True

    def __post_init__(self):
        super().__post_init__()
        unsupported = [kind for kind in self._reader_kinds if kind not in ("jsonl", "packed_jsonl")]
        if unsupported:
            raise ValueError(
                "MaterializedSFTMessagesAdapter supports JSONL paths only; "
                f"found reader kinds {unsupported}"
            )

    def _data_to_conversation(
        self, data: dict, source_path: Union[str, Path], local_idx: int
    ) -> MaterializedSFTMessagesExample:
        sample_id = f"{Path(source_path).stem}-{local_idx:012d}"
        return _transform_materialized_sft_messages(
            data,
            sample_id,
            validate_chunk_tokenization=self.validate_chunk_tokenization,
        )

    def _reader_item_to_conversation(self, shard_idx: int, local_idx: int) -> MaterializedSFTMessagesExample:
        reader = self._readers[shard_idx]
        reader_kind = self._reader_kinds[shard_idx]
        source_path = self._source_paths[shard_idx]
        if reader_kind == "packed_jsonl":
            item = reader[local_idx]
            location = reader.collection.locate(local_idx)
            return self._data_to_conversation(item, location.path, location.local_index)
        if reader_kind != "jsonl":
            raise RuntimeError(
                "MaterializedSFTMessagesAdapter supports JSONL paths only; "
                f"found reader kind {reader_kind!r} for {source_path}"
            )
        return self._data_to_conversation(reader[local_idx], source_path, local_idx)

    def _iter_path(self, path: Path) -> Iterator[MaterializedSFTMessagesExample]:
        if path.is_dir() or path.suffix == ".tar":
            raise ValueError(f"MaterializedSFTMessagesAdapter supports JSONL paths only, got {path}")
        yield from self._iter_jsonl(path)

    def _iter_jsonl(self, path: Path) -> Iterator[MaterializedSFTMessagesExample]:
        for idx, data in enumerate(load_jsonl(path)):
            sample_id = f"{path.stem}-{idx:012d}"
            yield _transform_materialized_sft_messages(
                data,
                sample_id,
                validate_chunk_tokenization=self.validate_chunk_tokenization,
            )


"""
NeMoMultimodalConversation: data types, file parser, default prompt formatting logic.
"""


@dataclass
class TextTurn:
    value: str
    role: str

    def to_dict(self):
        return {"type": "text", "from": self.role.title(), "value": self.value}


@dataclass
class AudioTurn:
    cut: Cut
    role: str
    audio_locator_tag: str
    text: str | None = None

    def to_dict(self):
        assert self.cut.has_recording and self.cut.recording.sources[0].type not in {
            "shar",
            "memory",
        }, "Cannot serialize AudioTurn to dict because it doesn't reference an audio file (the audio is stored in memory)."
        return {
            "type": "audio",
            "from": self.role.title(),
            "duration": self.cut.duration,
            "offset": self.cut.start,
            "value": self.cut.recording.sources[0].source,
            "text": self.text,
        }


@dataclass
class NeMoMultimodalConversation(Formattable, CustomFieldMixin):
    id: str
    turns: list[TextTurn | AudioTurn]
    token_equivalent_duration: float = None
    custom: dict = None

    @property
    def input_length(self) -> int | None:
        if self.context_ids is None:
            return None
        extra = _compute_num_audio_tokens(self, "context")
        return self.context_ids.shape[0] + extra

    @property
    def output_length(self) -> int | None:
        if self.answer_ids is None:
            return None
        extra = _compute_num_audio_tokens(self, "answer")
        return self.answer_ids.shape[0] + extra

    @property
    def total_length(self) -> int | None:
        if self.input_ids is None:
            return None
        extra = _compute_num_audio_tokens(self, "all")
        return self.input_ids.shape[0] + extra

    @property
    def has_audio_turns(self) -> bool:
        return any(isinstance(t, AudioTurn) for t in self.turns)

    @property
    def has_text_turns(self) -> bool:
        return any(isinstance(t, TextTurn) for t in self.turns)

    @property
    def is_text_only(self) -> bool:
        return all(isinstance(t, TextTurn) for t in self.turns)

    def to_dict(self):
        return {
            "id": self.id,
            "conversations": [t.to_dict() for t in self.turns],
            "custom": self.custom,
        }

    def list_cuts(self) -> list[Cut]:
        return [turn.cut for turn in self.turns if isinstance(turn, AudioTurn)]


def collate_conversation_audio_fault_tolerant(
    conversations: Sequence[Formattable],
    load_audio: AudioSamples,
) -> tuple[torch.Tensor, torch.Tensor, CutSet]:
    """
    Loads and collates audio data from a sequence of ``NeMoMultimodalConversation`` objects,
    preserving the order of conversations and turns.

    Audio is loaded via the provided ``AudioSamples`` (fault-tolerant and
    MultiCut-to-mono aware; optionally backed by AIStore GetBatch when
    constructed with ``use_batch_loader=True``) — one batched call per minibatch.

    Fault tolerance drops every conversation that has at least one audio turn
    whose cut failed to load (matching the legacy semantics).

    Cut ids are assumed unique within a minibatch (upheld by ``_make_cut_id``
    offset-suffixing in the adapters).

    Algorithm (four phases):

    1. **Flatten** — walk every conversation, collect each audio turn's cut into
       ``flat_cuts``, and record the per-conversation cut-id list in
       ``conv_to_cut_ids`` so we can regroup later. Empty ``flat_cuts`` (text-only
       batch) takes the early return.

    2. **Batched load** — a single ``AudioSamples`` call over the flat ``CutSet``
       returns ``audios``, ``audio_lens``, and the ``surviving`` subset that
       decoded successfully. ``survivor_rows`` maps each surviving cut id to its
       row index in ``audios``.

    3. **Regroup** — keep a conversation iff *all* its cut ids are in
       ``survivor_rows`` (legacy semantics: one failed turn invalidates the whole
       conversation). For survivors, append matching row indices to ``keep_rows``
       in conversation-then-turn order — so ``audios[keep_rows]`` aligns with the
       flattened turn order of ``CutSet(ok).list_cuts()``.

    4. **Return** — index ``audios`` / ``audio_lens`` by ``keep_rows`` and wrap
       the surviving conversations in a CutSet. If every conversation failed,
       returns empty tensors and an empty CutSet.

    Returns a tuple of:

    * ``audio`` tensor fp32 (B, T)

    * ``audio_lens`` tensor int64 (B)

    * ``conversations`` CutSet of NeMoMultimodalConversations that were successfully loaded.
    """
    # Phase 1: flatten — per-conv cut-id lists let us regroup after the batched load.
    flat_cuts, conv_to_cut_ids = _flatten_conversation_audio_cuts(conversations)

    if not flat_cuts:
        # Text-only batch: nothing to load, but pass conversations through unchanged.
        return torch.tensor([]), torch.tensor([]), CutSet(list(conversations))

    # Phase 2: batched load — one fault-tolerant AudioSamples call for the whole minibatch.
    # ``surviving`` is a subset (in arbitrary order) of cuts that decoded successfully.
    audios, audio_lens, surviving = load_audio(CutSet(flat_cuts))
    survivor_rows = {c.id: i for i, c in enumerate(surviving)}

    # Phase 3: regroup — keep a conversation only if every one of its turns survived.
    # ``keep_rows`` indexes ``audios`` in conversation-then-turn order.
    keep_rows: list[int] = []
    ok = []
    for conversation, ids in zip(conversations, conv_to_cut_ids):
        if all(cid in survivor_rows for cid in ids):
            keep_rows.extend(survivor_rows[cid] for cid in ids)
            ok.append(conversation)
        else:
            logging.warning(f"Skipping conversation because it failed to load audio: {conversation.id=}")

    if not ok:
        ids = [c.id for c in conversations]
        logging.warning(f"An entire batch of conversations failed to load audios. Conversations ids: {ids}")
        return torch.tensor([]), torch.tensor([]), CutSet()

    # Phase 4: return — re-order audio rows to match ``ok`` conversation/turn order.
    return audios[keep_rows], audio_lens[keep_rows], CutSet(ok)


def collate_conversation_audio_packed_fault_tolerant(
    conversations: Sequence[Formattable],
    load_audio: AudioSamples,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CutSet]:
    """Load conversation audio into one contiguous, unpadded waveform buffer.

    This preserves the conversation-atomic fault tolerance and conversation/turn
    ordering of :func:`collate_conversation_audio_fault_tolerant`, but bypasses
    :class:`AudioSamples`' padded collation. AIStore GetBatch is still invoked once
    when configured.

    Returns:
        A tuple of packed fp32 samples `(sum(audio_lens),)`, int64 cumulative sample
        offsets `(B_audio + 1,)`, int64 audio lengths `(B_audio,)`, and the
        successfully loaded conversations.
    """
    flat_cuts, conversation_cut_ids = _flatten_conversation_audio_cuts(conversations)
    if not flat_cuts:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            CutSet(list(conversations)),
        )

    audio_rows, surviving = _read_unpadded_audio_rows(CutSet(flat_cuts), load_audio)
    survivor_rows = {cut.id: row for row, cut in enumerate(surviving)}
    keep_rows, ok = _conversation_audio_survivor_rows(conversations, conversation_cut_ids, survivor_rows)
    if not ok:
        ids = [conversation.id for conversation in conversations]
        logging.warning(f"An entire batch of conversations failed to load audios. Conversations ids: {ids}")
        return (
            torch.empty(0, dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            CutSet(),
        )

    ordered_rows = [audio_rows[row] for row in keep_rows]
    if not ordered_rows:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            CutSet(ok),
        )
    audio_lens = torch.tensor([row.numel() for row in ordered_rows], dtype=torch.long)
    audio_cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.long), audio_lens.cumsum(dim=0, dtype=torch.long)])
    return torch.cat(ordered_rows), audio_cu_seqlens, audio_lens, CutSet(ok)


def _compute_num_audio_tokens(
    example: NeMoMultimodalConversation,
    mode: Literal["context", "answer", "all"],
    audio_token_estimator=None,
) -> int:
    if not example.has_audio_turns:
        return 0
    if audio_token_estimator is None:
        assert example.token_equivalent_duration is not None, (
            "Cannot compute the length of a NeMoMultimodalConversation: "
            "token_equivalent_duration must be set in order to estimate the number of tokens equivalent to audio "
            "turns. Did you forget to set token_equivalent_duration option in your dataloading config? "
            "Tip: generally it should be set to frame_shift * total_subsampling_factor of your audio encoder model."
        )
    if mode == "context":
        turns = example.turns[:-1]
    elif mode == "answer":
        turns = example.turns[-1:]
    elif mode == "all":
        turns = example.turns
    else:
        raise RuntimeError(f"invalid mode for number of audio token computation: {mode}")
    return sum(
        # Subtract one for each audio locator tag: its token is replaced by the
        # encoder frames rather than retained alongside them.
        (
            audio_token_estimator.estimate_cut(turn.cut)
            if audio_token_estimator is not None
            else math.ceil(turn.cut.duration / example.token_equivalent_duration)
        )
        - 1
        for turn in turns
        if isinstance(turn, AudioTurn)
    )


def measure_formattable_length(
    example: Formattable,
    mode: Literal["input", "output", "total"],
    *,
    audio_token_estimator=None,
) -> int | None:
    """Measure a prompt-formatted example, optionally with exact audio lengths.

    The legacy ``Formattable`` properties continue to use
    ``token_equivalent_duration``. Sampling constraints and token filters call
    this helper so an ``audio_token_estimator`` can replace the duration
    approximation without storing model-specific configuration on every data
    example.
    """
    if not isinstance(example, NeMoMultimodalConversation) or audio_token_estimator is None:
        return {
            "input": example.input_length,
            "output": example.output_length,
            "total": example.total_length,
        }[mode]

    ids = {
        "input": example.context_ids,
        "output": example.answer_ids,
        "total": example.input_ids,
    }[mode]
    if ids is None:
        return None
    audio_mode = {"input": "context", "output": "answer", "total": "all"}[mode]
    return ids.shape[0] + _compute_num_audio_tokens(
        example,
        audio_mode,
        audio_token_estimator=audio_token_estimator,
    )


@registered_prompt_format_fn(NeMoMultimodalConversation)
def default_multimodal_conversation_prompt_format_fn(example: NeMoMultimodalConversation, prompt, **prompt_kwargs):
    # Collapse consecutive same-role turns into single turn for proper prompt formatting.
    turns = groupby(
        [
            {
                "role": turn.role,
                "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
            }
            for turn in example.turns
        ],
        key=lambda turn: turn["role"],
    )
    turns = [(k, list(v)) for k, v in turns]
    turns = [
        {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
        for role, turn_grp in turns
    ]
    return prompt.encode_dialog(turns, **prompt_kwargs)


def _make_archive_member_cut(
    tar_path: str,
    audio_filename: str,
    duration: float,
    offset: float = 0.0,
    sampling_rate: int = 16000,
    source_type: str | None = None,
) -> Cut:
    """
    Build a Cut backed by an archive-member ``AudioSource`` (no tar file opened).

    Used for archive-backed paths in the multimodal conversation adapters. Audio
    is fetched lazily by Lhotse from either a local archive member or a URL-backed
    archive member.

    Unlike the richer helper in ``nemo_adapters.py``, this one does not attach
    supervisions, custom fields, or manifest/tar origin — the multimodal conversation
    adapters attach their own turn-level metadata downstream and re-id the cut via
    ``_make_cut_id``.
    """
    audio_path = f"{tar_path.rstrip('/')}/{audio_filename.lstrip('/')}"
    source_type = source_type or ("url" if is_valid_url(tar_path) else "file")
    recording_duration = offset + duration if offset > 0 else duration
    recording = Recording(
        id=audio_filename,
        sources=[AudioSource(type=source_type, channels=[0], source=audio_path)],
        sampling_rate=sampling_rate,
        num_samples=compute_num_samples(recording_duration, sampling_rate),
        duration=recording_duration,
    )
    cut = recording.to_cut()
    if offset > 0:
        cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
        cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
    return cut


_ARCHIVE_MEMBER_EXTENSIONS = (".tar.gz", ".tar", ".tgz")


def _split_archive_member_reference(audio_path: str) -> tuple[str, str] | None:
    for ext in _ARCHIVE_MEMBER_EXTENSIONS:
        marker = f"{ext}/"
        marker_index = audio_path.find(marker)
        if marker_index < 0:
            continue
        archive_end = marker_index + len(ext)
        member = audio_path[archive_end + 1 :]
        if member:
            return audio_path[:archive_end], member
    return None


@dataclass
class NeMoMultimodalConversationJsonlAdapter(IteratorNode):
    """
    ``NeMoMultimodalConversationJsonlAdapter`` is used to read a NeMo multimodal conversation JSONL
    and yield objects of type ``NeMoMultimodalConversation`` that can be sampled with Lhotse.

    We expect the following schema (contained in a single line per example)::

        {
            "id": str,
            "conversations": [
                {
                    "value": str,  # text message or path to audio
                    "from": "User" | "Assistant",
                    "type": "text" | "audio",
                    "duration": float,  # only for audio
                },
                ...
            ],
        }

    Set ``indexed=True`` to enable O(1) random access plus graph-token
    checkpointing. Indexed mode requires uncompressed JSONL manifests; for the
    tarred path it additionally requires uncompressed tar shards (the canonical
    ``.idx`` sidecars are built lazily on first construction).
    """

    manifest_filepath: str | list[str]
    audio_locator_tag: str
    tarred_audio_filepaths: str | list[str] = None
    token_equivalent_duration: float = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    system_prompt: str | None = None
    context: str | None = None
    slice_length: int | None = None
    indexed: bool = False
    indexes_root: Optional[Pathlike] = None
    index_pack: Optional[Pathlike] = None
    index_pack_max_open_files: int = 32
    skip_missing_manifest_entries: bool = False
    fault_tolerant_audio_loading: bool = True

    def __post_init__(self):
        raw_manifest_filepath = self.manifest_filepath
        if self.tarred_audio_filepaths is not None:
            self.tarred_audio_filepaths = expand_sharded_filepaths(self.tarred_audio_filepaths)
        self.epoch = 0
        self._cuts_readers: list = []
        self._tar_readers: list = []
        self._cum_lens: list[int] = []
        self._total_len = 0
        self._iter_state = PartitionedIndexedIterator()
        if self.indexed and self.index_pack is not None:
            self._init_packed(raw_manifest_filepath)
            return
        self.manifest_filepath = expand_sharded_filepaths(raw_manifest_filepath)
        if self.tarred_audio_filepaths is not None:
            assert len(self.manifest_filepath) == len(
                self.tarred_audio_filepaths
            ), f"{len(self.manifest_filepath)} != {len(self.tarred_audio_filepaths)}"
        if self.indexed:
            self._init_indexed()

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def _init_indexed(self) -> None:
        from lhotse.indexing import IndexedJsonlReader, index_file_path

        if self.slice_length is not None:
            raise ValueError("NeMoMultimodalConversationJsonlAdapter(indexed=True) does not support slice_length.")
        for p in self.manifest_filepath:
            self._cuts_readers.append(IndexedJsonlReader(p, index_path=index_file_path(p, self.indexes_root)))
        if self.tarred_audio_filepaths is not None:
            from nemo.collections.common.data.lhotse.indexed_adapters import IndexedTarMemberReader

            for p in self.tarred_audio_filepaths:
                self._tar_readers.append(IndexedTarMemberReader(p, idx_path=index_file_path(p, self.indexes_root)))
        cum = 0
        self._cum_lens.append(cum)
        for r in self._cuts_readers:
            cum += len(r)
            self._cum_lens.append(cum)
        self._total_len = cum

    def _init_packed(self, source_spec) -> None:
        from lhotse.index_pack import index_pack_collection_key, open_index_pack
        from lhotse.packed_lazy import LazyPackedManifestIterator

        if self.slice_length is not None:
            raise ValueError("NeMoMultimodalConversationJsonlAdapter(indexed=True) does not support slice_length.")
        if self.tarred_audio_filepaths is not None:
            raise ValueError(
                "Packed multimodal_conversation currently supports JSONL manifests with "
                "direct/remote audio paths, not paired audio tar files."
            )
        pack = open_index_pack(self.index_pack)
        key = index_pack_collection_key("manifest", "jsonl", source_spec)
        self._packed_source = LazyPackedManifestIterator(
            pack,
            key,
            decode=dict,
            max_open_files=self.index_pack_max_open_files,
        )
        self._cuts_readers = [self._packed_source]
        self._cum_lens = [0, len(self._packed_source)]
        self._total_len = len(self._packed_source)

    def __len__(self) -> int:
        if self.indexed:
            return self._total_len
        raise TypeError(
            "NeMoMultimodalConversationJsonlAdapter has unknown length unless constructed with indexed=True."
        )

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._total_len
        for s in range(len(self._cuts_readers)):
            if idx < self._cum_lens[s + 1]:
                return s, idx - self._cum_lens[s]
        raise IndexError(idx)

    def state_dict(self) -> dict:
        return {**self._iter_state.state_dict(), "epoch": self.epoch} if self.indexed else {}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)
        self.epoch = sd.get("epoch", 0)

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError(
                "NeMoMultimodalConversationJsonlAdapter only supports __getitem__ when indexed=True."
            )
        idx = int(normalize_graph_token(token))
        shard_idx, local_idx = self._resolve(idx)
        if hasattr(self, "_packed_source"):
            data, location = self._packed_source.read_with_location(local_idx)
            convo = self._build_conversation_local(data, manifest_path=location.path)
        else:
            data = self._cuts_readers[shard_idx][local_idx]
            if self._tar_readers:
                convo = self._build_conversation_tarred(
                    data,
                    tar_reader=self._tar_readers[shard_idx],
                    tar_path=self.tarred_audio_filepaths[shard_idx],
                )
            else:
                convo = self._build_conversation_local(data, manifest_path=self._cuts_readers[shard_idx].path)
        if convo is None:
            raise IndexError(
                f"Conversation at index {idx} (shard {shard_idx}, local {local_idx}) "
                f"could not be built; cannot satisfy random-access __getitem__."
            )
        return attach_graph_origin(convo, idx)

    def _build_conversation_local(self, data: dict, manifest_path: str) -> NeMoMultimodalConversation | None:
        if self._should_skip(data):
            return None
        try:
            turns = []
            for turn in data["conversations"]:
                if turn["type"] == "text":
                    turns.append(
                        TextTurn(value=turn["value"], role=turn["from"].lower())
                    )
                    continue
                cut = self._build_direct_audio_cut(
                    turn=turn, manifest_path=manifest_path
                )
                cut = cut.with_id(self._make_cut_id(cut, turn))
                turns.append(
                    AudioTurn(
                        cut=cut,
                        text=cut.supervisions[0].text if cut.supervisions else None,
                        role=turn["from"].lower(),
                        audio_locator_tag=self.audio_locator_tag,
                    )
                )
        except (AudioLoadingError, OSError) as ex:
            message = (
                f"Failed to load multimodal conversation {data.get('id')!r} "
                f"from manifest {manifest_path!r}."
            )
            if self.fault_tolerant_audio_loading:
                logging.warning(
                    f"Skipping conversation because fault_tolerant_audio_loading=true: {message} {ex}"
                )
                return None
            raise RuntimeError(message) from ex
        if self.context is not None and turns[0].role == "user" and isinstance(turns[0], AudioTurn):
            turns = [TextTurn(role="user", value=self.context)] + turns
        if self.system_prompt is not None and turns[0].role != "system":
            turns = [TextTurn(role="system", value=self.system_prompt)] + turns
        return NeMoMultimodalConversation(
            id=data["id"],
            turns=turns,
            token_equivalent_duration=self.token_equivalent_duration,
            custom=data.get("custom"),
        )

    def _build_direct_audio_cut(self, turn: dict, manifest_path: str) -> Cut:
        audio_path = get_full_path(turn["value"], manifest_path)
        archive_ref = _split_archive_member_reference(str(audio_path))
        if archive_ref is not None:
            duration = turn.get("duration")
            if duration is None:
                raise ValueError(
                    "Archive-member multimodal conversation audio requires a 'duration' field: " f"{turn['value']!r}"
                )
            tar_path, audio_filename = archive_ref
            return _make_archive_member_cut(
                tar_path=tar_path,
                audio_filename=audio_filename,
                duration=duration,
                offset=turn.get("offset", 0.0),
                sampling_rate=turn.get("sampling_rate", 16000),
            )
        try:
            recording = Recording.from_file(audio_path)
        except (AudioLoadingError, OSError, RuntimeError, ValueError) as ex:
            raise AudioLoadingError(
                f"Failed to read multimodal conversation audio {str(audio_path)!r}."
            ) from ex
        return recording.to_cut().truncate(
            offset=turn.get("offset", 0.0), duration=turn.get("duration")
        )

    def _build_conversation_tarred(self, data: dict, tar_reader, tar_path: str) -> NeMoMultimodalConversation | None:
        import io as _io

        import soundfile as _sf
        from lhotse import AudioSource as _AudioSource
        from lhotse import Recording as _Recording

        if self._should_skip(data):
            return None
        cuts: list = []
        for turn in data["conversations"]:
            if turn["type"] != "audio":
                continue
            try:
                audio_bytes = tar_reader.get(turn["value"])
                meta = _sf.info(_io.BytesIO(audio_bytes))
            except (EOFError, KeyError, OSError, RuntimeError, ValueError, tarfile.TarError) as ex:
                message = (
                    f"Failed to load multimodal audio member {turn['value']!r} "
                    f"from tar {tar_path!r}."
                )
                if self.fault_tolerant_audio_loading:
                    logging.warning(
                        f"Skipping conversation because fault_tolerant_audio_loading=true: {message} {ex}"
                    )
                    return None
                raise RuntimeError(message) from ex
            recording = _Recording(
                id=turn["value"],
                sources=[_AudioSource(type="memory", channels=list(range(meta.channels)), source=audio_bytes)],
                sampling_rate=int(meta.samplerate),
                num_samples=meta.frames,
                duration=meta.duration,
            )
            cut = recording.to_cut().truncate(offset=turn.get("offset", 0.0), duration=turn.get("duration"))
            cut = cut.with_id(self._make_cut_id(cut, turn))
            cuts.append(cut)
        cuts = deque(cuts)
        turns = [
            (
                TextTurn(
                    value=turn["value"],
                    role=turn["from"].lower(),
                )
                if turn["type"] == "text"
                else AudioTurn(
                    cut=(c := cuts.popleft()),
                    text=c.supervisions[0].text if c.supervisions else None,
                    role=turn["from"].lower(),
                    audio_locator_tag=self.audio_locator_tag,
                )
            )
            for turn in data["conversations"]
        ]
        if self.context is not None and turns[0].role == "user" and isinstance(turns[0], AudioTurn):
            turns = [TextTurn(role="user", value=self.context)] + turns
        if self.system_prompt is not None and turns[0].role != "system":
            turns = [TextTurn(role="system", value=self.system_prompt)] + turns
        return NeMoMultimodalConversation(
            id=data["id"],
            turns=turns,
            token_equivalent_duration=self.token_equivalent_duration,
            custom=data.get("custom"),
        )

    def __iter__(self) -> Iterator[NeMoMultimodalConversation]:
        if self.indexed:
            yield from self._iter_indexed()
            return
        if self.tarred_audio_filepaths is not None:
            yield from self._iter_tar()
        else:
            yield from self._iter_jsonl()

    def _iter_indexed(self) -> Iterator[NeMoMultimodalConversation]:
        for global_idx in self._iter_state.iterate(self._total_len):
            shard_idx, local_idx = self._resolve(global_idx)
            if hasattr(self, "_packed_source"):
                data, location = self._packed_source.read_with_location(local_idx)
                convo = self._build_conversation_local(data, manifest_path=location.path)
            else:
                data = self._cuts_readers[shard_idx][local_idx]
                if self._tar_readers:
                    convo = self._build_conversation_tarred(
                        data,
                        tar_reader=self._tar_readers[shard_idx],
                        tar_path=self.tarred_audio_filepaths[shard_idx],
                    )
                else:
                    convo = self._build_conversation_local(data, manifest_path=self._cuts_readers[shard_idx].path)
            if convo is None:
                continue
            attach_graph_origin(convo, global_idx)
            yield convo
        self.epoch += 1

    def _should_skip(self, example: dict) -> bool:
        return manifest_entry_is_explicitly_skipped(example)

    def _get_rng(self) -> random.Random:
        seed = resolve_seed(self.shard_seed) + self.epoch
        return random.Random(seed)

    def _make_cut_id(self, cut, turn) -> str:
        offset = turn.get('offset') if turn.get('offset') else cut.start
        duration = turn.get('duration') if turn.get('duration') else cut.duration
        if offset > 0.0:
            return f"{Path(turn['value']).stem}_{offset:.3f}_{duration:.3f}"
        return Path(turn['value']).stem

    def _iter_tar(self):
        # In GetBatch mode we do not open the tar; the manifest's audio path is trusted to match
        # the tar layout, mirroring LazyNeMoTarredIterator._iter_batch_for_ais_get_batch.
        use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"
        paths = list(zip(self.manifest_filepath, self.tarred_audio_filepaths))
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(paths)
        for jsonl_path, tar_path in paths:
            jsonl = load_jsonl(jsonl_path)
            if self.slice_length is not None:
                jsonl = list(jsonl)
            tar = None if use_ais_get_batch else iter(TarIterator(tar_path))
            slice_offset = (
                rng.randint(0, len(jsonl) - self.slice_length)
                if self.slice_length is not None and self.slice_length < len(jsonl)
                else -1
            )
            cntr = 0
            for idx, data in enumerate(jsonl):
                audio_turns = [t for t in data["conversations"] if t["type"] == "audio"]
                cuts = []
                try:
                    for turn in audio_turns:
                        if use_ais_get_batch:
                            cut = _make_archive_member_cut(
                                tar_path=str(tar_path),
                                audio_filename=turn['value'],
                                duration=turn.get('duration'),
                                offset=turn.get('offset', 0.0),
                                sampling_rate=turn.get('sampling_rate', 16000),
                                source_type='url',
                            )
                            cut = cut.with_id(self._make_cut_id(cut, turn))
                        else:
                            recording, _ = _next_matching_paired_audio(
                                tar,
                                turn['value'],
                                manifest_path=jsonl_path,
                                tar_path=tar_path,
                                skip_missing_manifest_entries=self.skip_missing_manifest_entries,
                            )
                            cut = recording.to_cut().truncate(
                                offset=turn.get("offset", 0.0), duration=turn.get("duration")
                            )
                            cut = cut.with_id(self._make_cut_id(cut, turn))
                        cuts.append(cut)
                except (
                    AudioLoadingError,
                    OSError,
                    RuntimeError,
                    StopIteration,
                    tarfile.TarError,
                ) as ex:
                    message = (
                        f"Failed to load multimodal tar shard {tar_path!r} "
                        f"for manifest {jsonl_path!r}."
                    )
                    if self.fault_tolerant_audio_loading:
                        logging.warning(
                            f"Skipping tar because fault_tolerant_audio_loading=true: {message} {ex}"
                        )
                        break
                    raise RuntimeError(message) from ex
                if self._should_skip(data):
                    continue  # Skip only after tar has been iterated, otherwise there will be data mismatch
                if idx < slice_offset:
                    continue
                elif cntr == self.slice_length:
                    break
                cuts = deque(cuts)
                turns = [
                    (
                        TextTurn(
                            value=turn["value"],
                            role=turn["from"].lower(),
                        )
                        if turn["type"] == "text"
                        else AudioTurn(
                            cut=(c := cuts.popleft()),
                            text=c.supervisions[0].text if c.supervisions else None,
                            role=turn["from"].lower(),
                            audio_locator_tag=self.audio_locator_tag,
                        )
                    )
                    for turn in data["conversations"]
                ]
                if self.context is not None and turns[0].role == "user" and isinstance(turns[0], AudioTurn):
                    turns = [TextTurn(role="user", value=self.context)] + turns
                if self.system_prompt is not None and turns[0].role != "system":
                    turns = [TextTurn(role="system", value=self.system_prompt)] + turns
                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=turns,
                    token_equivalent_duration=self.token_equivalent_duration,
                    custom=data.get("custom"),
                )
                cntr += 1
            else:
                if tar is not None:
                    _validate_no_trailing_paired_audio(
                        tar,
                        manifest_path=jsonl_path,
                        tar_path=tar_path,
                        skip_missing_manifest_entries=self.skip_missing_manifest_entries,
                        fault_tolerant_audio_loading=self.fault_tolerant_audio_loading,
                    )

        self.epoch += 1

    def _iter_jsonl(self):
        paths = self.manifest_filepath
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(paths)
        for path in paths:
            jsonl_iter = load_jsonl(path)
            if self.shuffle_shards:
                jsonl_iter = list(jsonl_iter)
                rng.shuffle(jsonl_iter)
            for data in jsonl_iter:
                convo = self._build_conversation_local(data, manifest_path=path)
                if convo is None:
                    continue
                yield convo

        self.epoch += 1


@dataclass
class NeMoMultimodalConversationShareGPTJsonlAdapter(IteratorNode):
    """
    ``NeMoMultimodalConversationShareGPTJsonlAdapter`` is used to read a ShareGPT format multimodal
    conversation JSONL and yield objects of type ``NeMoMultimodalConversation`` that can be sampled with Lhotse.

    We expect the following ShareGPT schema (contained in a single line per example)::

        {
            "id": str,  # not optional, but we fall back to "missing-example-id" if absent (see data.get("id", ...) below)
            "sound": str,  # path to audio file
            "conversations": [
                {
                    "value": str,  # text message, may contain <sound> or <speech> placeholder
                    "from": "human" | "gpt",
                },
                ...
            ],
            "ori_sound": str,  # optional original sound path
        }

    Audio placeholders (<sound>, <speech>) in conversation text will be replaced with the audio from the "sound" field.
    By default, both <sound> and <speech> placeholders are supported.
    """

    manifest_filepath: str | list[str]
    audio_locator_tag: str
    audio_placeholders: Union[str, list[str]] = None
    tarred_audio_filepaths: str | list[str] = None
    tar_lookup_mode: str | None = None
    tar_routing_filepath: Pathlike | None = None
    audio_root: str | None = None
    audio_path_prefix_map: dict[str, str] | None = None
    token_equivalent_duration: float = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    slice_length: int | None = None
    indexed: bool = False
    indexes_root: Optional[Pathlike] = None
    index_pack: Optional[Pathlike] = None
    index_pack_max_open_files: int = 32
    skip_missing_manifest_entries: bool = False
    fault_tolerant_audio_loading: bool = True
    excluded_manifest_lines: Sequence[int] | None = None
    excluded_manifest_lines_sha256: str | None = None
    approved_exclusion_audit_sha256: str | None = None

    def __post_init__(self):
        raw_manifest_filepath = self.manifest_filepath
        raw_tarred_audio_filepaths = self.tarred_audio_filepaths
        if self.tar_lookup_mode not in (None, "paired", "collection"):
            raise ValueError(
                f"Unsupported tar_lookup_mode={self.tar_lookup_mode!r}; expected 'paired' or 'collection'."
            )
        if self.tar_lookup_mode is None and self.tarred_audio_filepaths is not None:
            self.tar_lookup_mode = "paired"
        if self.tar_lookup_mode == "collection":
            if not self.indexed:
                raise ValueError("tar_lookup_mode='collection' requires indexed=true.")
            if self.tarred_audio_filepaths is None:
                raise ValueError("tar_lookup_mode='collection' requires tarred_audio_filepaths.")
            if self.tar_routing_filepath is None:
                raise ValueError("tar_lookup_mode='collection' requires tar_routing_filepath.")
            route_path = Path(self.tar_routing_filepath)
            if not route_path.is_absolute():
                if self.index_pack is not None:
                    route_path = Path(self.index_pack).parent / route_path
                elif self.indexes_root is not None:
                    route_path = Path(self.indexes_root) / route_path
                else:
                    raise ValueError("Relative tar_routing_filepath requires index_pack or indexes_root.")
            self.tar_routing_filepath = route_path
        self.audio_placeholders = _normalize_audio_placeholders(self.audio_placeholders)
        self._audio_path_prefix_mapper = AudioPathPrefixMap(self.audio_path_prefix_map)
        self.epoch = 0
        self._cuts_readers: list = []
        self._tar_readers: list = []
        self._cum_lens: list[int] = []
        self._total_len = 0
        self._iter_state = PartitionedIndexedIterator()
        if self.indexed and self.index_pack is not None:
            self._init_packed(raw_manifest_filepath, raw_tarred_audio_filepaths)
            self._configure_manifest_exclusions()
            return
        self.manifest_filepath = expand_sharded_filepaths(raw_manifest_filepath)
        if self.tarred_audio_filepaths is not None:
            self.tarred_audio_filepaths = expand_sharded_filepaths(self.tarred_audio_filepaths)
            if self.tar_lookup_mode == "paired":
                assert len(self.manifest_filepath) == len(
                    self.tarred_audio_filepaths
                ), f"{len(self.manifest_filepath)} != {len(self.tarred_audio_filepaths)}"
        if self.indexed:
            self._init_indexed(raw_manifest_filepath)
        self._configure_manifest_exclusions()

    @staticmethod
    def _manifest_line_set_sha256(lines: Sequence[int]) -> str:
        payload = json.dumps(list(lines), separators=(",", ":")) + "\n"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _configure_manifest_exclusions(self) -> None:
        """Remove approved rows from the logical indexed sample domain.

        The configuration uses one-based JSONL line numbers for direct auditability.
        We map logical graph tokens to the remaining physical rows before any audio
        I/O, so every rank and a restored iterator observe the same sample domain.
        """

        raw_lines = [] if self.excluded_manifest_lines is None else list(self.excluded_manifest_lines)
        if raw_lines and not self.indexed:
            raise ValueError("excluded_manifest_lines requires indexed=true.")
        if any(isinstance(line, bool) or not isinstance(line, int) for line in raw_lines):
            raise TypeError("excluded_manifest_lines must contain one-based integer line numbers.")
        lines = tuple(sorted(raw_lines))
        if len(lines) != len(set(lines)):
            raise ValueError("excluded_manifest_lines must not contain duplicates.")
        if any(line < 1 for line in lines):
            raise ValueError("excluded_manifest_lines must contain positive one-based line numbers.")
        source_total_len = self._total_len
        if lines and lines[-1] > source_total_len:
            raise ValueError(
                f"excluded_manifest_lines contains line {lines[-1]}, but the indexed source has "
                f"only {source_total_len} rows."
            )
        digest = self._manifest_line_set_sha256(lines)
        if self.excluded_manifest_lines_sha256 is not None:
            expected = str(self.excluded_manifest_lines_sha256).lower()
            if expected != digest:
                raise ValueError(
                    "excluded_manifest_lines_sha256 does not match the canonical line set: "
                    f"expected={expected}, actual={digest}."
                )
        if self.approved_exclusion_audit_sha256 is not None:
            audit_digest = str(self.approved_exclusion_audit_sha256).lower()
            if len(audit_digest) != 64 or any(ch not in "0123456789abcdef" for ch in audit_digest):
                raise ValueError("approved_exclusion_audit_sha256 must be a lowercase SHA-256 digest.")
            self.approved_exclusion_audit_sha256 = audit_digest
        elif lines:
            raise ValueError("excluded_manifest_lines requires approved_exclusion_audit_sha256 provenance.")
        self._source_total_len = source_total_len
        self._excluded_manifest_lines = lines
        self._excluded_physical_indexes = tuple(line - 1 for line in lines)
        self._excluded_manifest_lines_sha256 = digest
        self._total_len = source_total_len - len(lines)
        if lines:
            logging.warning(
                "Applying approved deterministic ShareGPT manifest exclusions before audio I/O: "
                f"excluded_rows={len(lines)} source_rows={source_total_len} "
                f"effective_rows={self._total_len} line_set_sha256={digest} "
                f"audit_sha256={self.approved_exclusion_audit_sha256}."
            )

    def _logical_to_physical_index(self, logical_idx: int) -> int:
        if logical_idx < 0:
            logical_idx += self._total_len
        if logical_idx < 0 or logical_idx >= self._total_len:
            raise IndexError(logical_idx)
        physical_idx = logical_idx
        for excluded_idx in self._excluded_physical_indexes:
            if excluded_idx > physical_idx:
                break
            physical_idx += 1
        return physical_idx

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def _init_indexed(self, source_spec) -> None:
        from lhotse.indexing import IndexedJsonlReader, index_file_path

        if self.slice_length is not None:
            raise ValueError(
                "NeMoMultimodalConversationShareGPTJsonlAdapter(indexed=True) does not support slice_length."
            )
        for p in self.manifest_filepath:
            self._cuts_readers.append(IndexedJsonlReader(p, index_path=index_file_path(p, self.indexes_root)))
        if self.tarred_audio_filepaths is not None:
            if self.tar_lookup_mode == "collection":
                self._collection_tar_readers = [
                    IndexedTarMemberReader(
                        p,
                        idx_path=index_file_path(p, self.indexes_root),
                        auto_create_index=False,
                    )
                    for p in self.tarred_audio_filepaths
                ]
            else:
                for p in self.tarred_audio_filepaths:
                    self._tar_readers.append(IndexedTarMemberReader(p, idx_path=index_file_path(p, self.indexes_root)))
        cum = 0
        self._cum_lens.append(cum)
        for r in self._cuts_readers:
            cum += len(r)
            self._cum_lens.append(cum)
        self._total_len = cum
        if self.tar_lookup_mode == "collection":
            self._init_collection_route(source_spec)

    def _init_packed(self, source_spec, tar_source_spec) -> None:
        from lhotse.index_pack import index_pack_collection_key, open_index_pack
        from lhotse.packed_lazy import LazyPackedManifestIterator

        if self.slice_length is not None:
            raise ValueError(
                "NeMoMultimodalConversationShareGPTJsonlAdapter(indexed=True) " "does not support slice_length."
            )
        if self.tarred_audio_filepaths is not None and self.tar_lookup_mode != "collection":
            raise ValueError(
                "Packed ShareGPT currently supports JSONL manifests with "
                "direct/remote audio paths or tar_lookup_mode='collection', not paired audio tar files."
            )
        pack = open_index_pack(self.index_pack)
        key = index_pack_collection_key("manifest", "jsonl", source_spec)
        self._packed_source = LazyPackedManifestIterator(
            pack,
            key,
            decode=dict,
            max_open_files=self.index_pack_max_open_files,
        )
        self._cuts_readers = [self._packed_source]
        self._cum_lens = [0, len(self._packed_source)]
        self._total_len = len(self._packed_source)
        if self.tar_lookup_mode == "collection":
            tar_key = index_pack_collection_key("tar_collection", "nemo_tar", tar_source_spec)
            tar_collection = pack.collection(tar_key)
            if not tar_collection.offsets_required:
                raise ValueError("ShareGPT collection routing requires an offset-bearing tar collection.")
            self._packed_collection_tar_reader = PackedTarMemberReader(
                tar_collection, max_open_files=self.index_pack_max_open_files
            )
            self._collection_tar_readers = self._packed_collection_tar_reader
            self._init_collection_route(source_spec)

    def _init_collection_route(self, manifest_source_spec) -> None:
        from nemo.collections.common.data.lhotse.sharegpt_tar_routing import (
            ShareGptTarRoutingIndex,
            canonical_audio_prefix_map_digest,
            ordered_manifest_source_identity_digest,
            ordered_manifest_spec_path_digest,
            ordered_tar_catalog_digest,
            validate_sharegpt_tar_routing_index,
        )

        if hasattr(self, "_packed_source"):
            manifest_collection = self._packed_source.collection
            manifest_paths = [
                manifest_collection.path_for_shard(shard_index)
                for shard_index in range(manifest_collection.sequence_count)
            ]
            tar_collection = self._packed_collection_tar_reader.collection
            with ShareGptTarRoutingIndex(self.tar_routing_filepath) as routing:
                tar_count = routing.header.tar_shard_count
            if tar_collection.sequence_count != tar_count:
                raise ValueError(
                    f"ShareGPT tar collection has {tar_collection.sequence_count} shards but "
                    f"the routing index declares {tar_count}."
                )
        else:
            manifest_paths = list(self.manifest_filepath)
            tar_count = len(self._collection_tar_readers)
        tar_paths = (
            [
                self._packed_collection_tar_reader.collection.path_for_shard(shard_index)
                for shard_index in range(tar_count)
            ]
            if hasattr(self, "_packed_source")
            else list(self.tarred_audio_filepaths)
        )
        raw_specs = (
            [manifest_source_spec]
            if isinstance(manifest_source_spec, (str, os.PathLike))
            else list(manifest_source_spec)
        )
        manifest_specs = [os.fspath(spec) for spec in raw_specs for _ in expand_sharded_filepaths(spec)]
        if len(manifest_specs) != len(manifest_paths):
            raise ValueError(
                f"ShareGPT manifest source spec expands to {len(manifest_specs)} shards but "
                f"the indexed collection contains {len(manifest_paths)}."
            )
        validate_sharegpt_tar_routing_index(
            self.tar_routing_filepath,
            expected_manifest_row_count=self._total_len,
            expected_manifest_spec_path_digest=ordered_manifest_spec_path_digest(manifest_paths, manifest_specs),
            expected_manifest_source_identity_digest=ordered_manifest_source_identity_digest(manifest_paths),
            expected_tar_shard_count=tar_count,
            expected_tar_catalog_digest=ordered_tar_catalog_digest(tar_paths),
            expected_audio_prefix_map_digest=canonical_audio_prefix_map_digest(self._audio_path_prefix_mapper.mapping),
            offset_bearing_tar_collections=True,
        )
        self._collection_route = ShareGptTarRoutingIndex(self.tar_routing_filepath)

    def __len__(self) -> int:
        if self.indexed:
            return self._total_len
        raise TypeError(
            "NeMoMultimodalConversationShareGPTJsonlAdapter has unknown length unless constructed with indexed=True."
        )

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._total_len
        for s in range(len(self._cuts_readers)):
            if idx < self._cum_lens[s + 1]:
                return s, idx - self._cum_lens[s]
        raise IndexError(idx)

    def state_dict(self) -> dict:
        if not self.indexed:
            return {}
        state = {**self._iter_state.state_dict(), "epoch": self.epoch}
        if self._excluded_manifest_lines:
            state.update(
                {
                    "excluded_manifest_lines_sha256": self._excluded_manifest_lines_sha256,
                    "approved_exclusion_audit_sha256": self.approved_exclusion_audit_sha256,
                }
            )
        return state

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        restored_digest = sd.get("excluded_manifest_lines_sha256")
        if self._excluded_manifest_lines and restored_digest != self._excluded_manifest_lines_sha256:
            raise ValueError(
                "ShareGPT manifest exclusion set changed across resume: "
                f"checkpoint={restored_digest!r}, current={self._excluded_manifest_lines_sha256!r}."
            )
        restored_audit = sd.get("approved_exclusion_audit_sha256")
        if self._excluded_manifest_lines and restored_audit != self.approved_exclusion_audit_sha256:
            raise ValueError(
                "ShareGPT exclusion audit changed across resume: "
                f"checkpoint={restored_audit!r}, current={self.approved_exclusion_audit_sha256!r}."
            )
        if not self._excluded_manifest_lines and (restored_digest is not None or restored_audit is not None):
            raise ValueError("Checkpoint contains ShareGPT exclusions but the current config does not.")
        self._iter_state.load_state_dict(sd)
        self.epoch = sd.get("epoch", 0)

    def _build_one(
        self,
        data: dict,
        shard_idx: int,
        manifest_path: str | None = None,
        route_row_idx: int | None = None,
    ) -> NeMoMultimodalConversation | None:
        try:
            conversations = _ShareGPTConversationParser(self.audio_placeholders, data).transform()
            if self.tar_lookup_mode == "collection":
                if route_row_idx is None:
                    raise RuntimeError("Collection-routed ShareGPT access requires a global manifest row index.")
                routes = self._collection_route.routes_for_row(route_row_idx)
                used_route_indexes: set[int] = set()

                def resolve_collection_cut(turn):
                    route_index = int(turn.get("audio_path_index", 0))
                    if route_index >= len(routes):
                        raise ValueError(
                            f"ShareGPT route row {route_row_idx} has {len(routes)} records but audio path index "
                            f"{route_index} was requested."
                        )
                    used_route_indexes.add(route_index)
                    return self._resolve_cut_from_collection(turn, routes[route_index])

                turns = _ShareGPTConversationParser.create_turns(
                    self.audio_locator_tag,
                    conversations,
                    resolve_collection_cut,
                )
                if used_route_indexes != set(range(len(routes))):
                    raise ValueError(
                        f"ShareGPT route row {route_row_idx} contains unused records: "
                        f"used={sorted(used_route_indexes)}, total={len(routes)}."
                    )
                return NeMoMultimodalConversation(
                    id=data.get("id", "missing-example-id"),
                    turns=turns,
                    token_equivalent_duration=self.token_equivalent_duration,
                )
            if self._tar_readers:
                tar_reader = self._tar_readers[shard_idx]
                tar_path = self.tarred_audio_filepaths[shard_idx]
                return NeMoMultimodalConversation(
                    id=data.get("id", "missing-example-id"),
                    turns=_ShareGPTConversationParser.create_turns(
                        self.audio_locator_tag,
                        conversations,
                        lambda t: self._resolve_cut_from_indexed_tar(t, tar_reader, tar_path),
                    ),
                    token_equivalent_duration=self.token_equivalent_duration,
                )
            if manifest_path is None:
                manifest_path = self._cuts_readers[shard_idx].path
            return NeMoMultimodalConversation(
                id=data.get("id", "missing-example-id"),
                turns=_ShareGPTConversationParser.create_turns(
                    self.audio_locator_tag,
                    conversations,
                    lambda t, _p=manifest_path: self._resolve_cut_from_path(t, _p),
                ),
                token_equivalent_duration=self.token_equivalent_duration,
            )
        except _SHAREGPT_AUDIO_LOADING_ERRORS as e:
            if not self.fault_tolerant_audio_loading:
                raise
            logging.warning(
                "Skipping ShareGPT sample due to audio loading failure: "
                f"sample_id={data.get('id', 'missing-example-id')!r} shard_idx={shard_idx} "
                f"error={type(e).__name__}: {e}"
            )
            return None

    def _resolve_cut_from_collection(self, turn, route):
        if hasattr(self, "_packed_collection_tar_reader"):
            (member_name, audio_bytes), location = self._packed_collection_tar_reader.read_shard_with_location(
                route.tar_shard_index, route.tar_member_local_index
            )
            source_range_bytes = location.end - location.start
            source_read_key = f"{location.path}@{location.start}:{location.end}"
        else:
            reader = self._collection_tar_readers[route.tar_shard_index]
            member_name, audio_bytes = reader[route.tar_member_local_index]
            source_start = int(reader.offsets[route.tar_member_local_index])
            source_end = int(reader.offsets[route.tar_member_local_index + 1])
            source_range_bytes = source_end - source_start
            source_read_key = f"{reader.data_path}@{source_start}:{source_end}"
        requested = os.fspath(
            _ShareGPTConversationParser.expect_one_audio_path(
                turn["value"], sample_id=turn.get("id", "?"), context="collection audio turn value"
            )
        )
        if requested != member_name and PurePosixPath(requested).name != PurePosixPath(member_name).name:
            raise ValueError(
                f"ShareGPT tar route resolved requested audio {requested!r} to mismatched member {member_name!r}."
            )
        recording = Recording.from_bytes(audio_bytes, recording_id=PurePosixPath(member_name).stem)
        cut = fastcopy(
            recording.to_cut(),
            custom={
                "_source_codec": PurePosixPath(member_name).suffix.lower().removeprefix(".") or None,
                "_source_range_bytes": source_range_bytes,
                "_source_read_key": source_read_key,
            },
        )
        cut = cut.truncate(offset=turn.get("offset", 0.0), duration=turn.get("duration"))
        return cut.with_id(self._make_cut_id(cut, {**turn, "value": member_name}))

    def _resolve_cut_from_indexed_tar(self, turn, tar_reader, tar_path):
        import io as _io

        import soundfile as _sf
        from lhotse import AudioSource as _AudioSource
        from lhotse import Recording as _Recording

        audio_path = os.fspath(
            _ShareGPTConversationParser.expect_one_audio_path(
                turn["value"], sample_id=turn.get("id", "?"), context="audio turn value"
            )
        )
        turn_for_id = {**turn, "value": audio_path}
        try:
            audio_bytes = tar_reader.get(audio_path)
        except KeyError as ex:
            raise _ShareGPTMissingTarMemberError(
                f"ShareGPT manifest references missing member {audio_path!r} in paired tar {tar_path!r}."
            ) from ex
        try:
            meta = _sf.info(_io.BytesIO(audio_bytes))
        except (AudioLoadingError, EOFError, OSError, RuntimeError, ValueError) as ex:
            raise _ShareGPTAudioDecodeError(
                f"Failed to decode ShareGPT member {audio_path!r} from paired tar {tar_path!r}."
            ) from ex
        recording = _Recording(
            id=audio_path,
            sources=[_AudioSource(type="memory", channels=list(range(meta.channels)), source=audio_bytes)],
            sampling_rate=int(meta.samplerate),
            num_samples=meta.frames,
            duration=meta.duration,
        )
        cut = recording.to_cut().truncate(offset=turn.get("offset", 0.0), duration=turn.get("duration"))
        return cut.with_id(self._make_cut_id(cut, turn_for_id))

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError(
                "NeMoMultimodalConversationShareGPTJsonlAdapter only supports __getitem__ when indexed=True."
            )
        logical_idx = int(normalize_graph_token(token))
        physical_idx = self._logical_to_physical_index(logical_idx)
        shard_idx, local_idx = self._resolve(physical_idx)
        if hasattr(self, "_packed_source"):
            data, location = self._packed_source.read_with_location(local_idx)
            convo = self._build_one(data, shard_idx, location.path, route_row_idx=physical_idx)
        else:
            data = self._cuts_readers[shard_idx][local_idx]
            convo = self._build_one(data, shard_idx, route_row_idx=physical_idx)
        if convo is None:
            raise IndexError(
                f"ShareGPT sample at logical index {logical_idx} (physical index {physical_idx}) is not decodable; "
                "cannot satisfy random-access __getitem__."
            )
        return attach_graph_origin(convo, logical_idx)

    def __iter__(self) -> Iterator[NeMoMultimodalConversation]:
        if self.indexed:
            yield from self._iter_indexed_node()
            return
        if self.tarred_audio_filepaths is not None:
            yield from self._iter_tar()
        else:
            yield from self._iter_jsonl()

    def _iter_indexed_node(self) -> Iterator[NeMoMultimodalConversation]:
        for logical_idx in self._iter_state.iterate(self._total_len):
            physical_idx = self._logical_to_physical_index(logical_idx)
            shard_idx, local_idx = self._resolve(physical_idx)
            if hasattr(self, "_packed_source"):
                data, location = self._packed_source.read_with_location(local_idx)
                convo = self._build_one(data, shard_idx, location.path, route_row_idx=physical_idx)
            else:
                data = self._cuts_readers[shard_idx][local_idx]
                convo = self._build_one(data, shard_idx, route_row_idx=physical_idx)
            if convo is None:
                continue
            attach_graph_origin(convo, logical_idx)
            yield convo
        self.epoch += 1

    def _get_rng(self) -> random.Random:
        return random.Random(resolve_seed(self.shard_seed) + self.epoch)

    def _make_cut_id(self, cut, turn) -> str:
        offset = turn.get('offset') if turn.get('offset') else cut.start
        duration = turn.get('duration') if turn.get('duration') else cut.duration
        if offset > 0.0:
            return f"{Path(turn['value']).stem}_{offset:.3f}_{duration:.3f}"
        return Path(turn['value']).stem

    def _resolve_cut_from_path(self, turn, manifest_path):
        audio_path = os.fspath(
            _ShareGPTConversationParser.expect_one_audio_path(
                turn["value"], sample_id=turn.get("id", "?"), context="audio turn value"
            )
        )
        audio_path = self._audio_path_prefix_mapper.resolve(audio_path)
        turn_for_id = {**turn, "value": audio_path}
        if is_valid_url(audio_path):
            data = open_best(audio_path, "rb").read()
            cut = Recording.from_bytes(data, recording_id=audio_path).to_cut()
        elif self.audio_root is not None:
            cut = Recording.from_file(get_full_path(audio_path, data_dir=self.audio_root)).to_cut()
        else:
            cut = Recording.from_file(get_full_path(audio_path, manifest_path)).to_cut()
        return cut.truncate(offset=turn["offset"], duration=turn["duration"]).with_id(
            self._make_cut_id(cut, turn_for_id)
        )

    def _iter_tar(self):
        # See NeMoMultimodalConversationJsonlAdapter._iter_tar for GetBatch-mode rationale.
        use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"
        paths = list(zip(self.manifest_filepath, self.tarred_audio_filepaths))
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(paths)
        for jsonl_path, tar_path in paths:
            jsonl = load_jsonl(jsonl_path)
            if self.slice_length is not None:
                jsonl = list(jsonl)
            tar = None if use_ais_get_batch else iter(TarIterator(tar_path))
            slice_offset = (
                rng.randint(0, len(jsonl) - self.slice_length)
                if self.slice_length is not None and self.slice_length < len(jsonl)
                else -1
            )
            cntr = 0
            for idx, data in enumerate(jsonl):
                conversations = _ShareGPTConversationParser(self.audio_placeholders, data).transform()
                audio_turns = [t for t in conversations if t["type"] == "audio"]
                cuts = []
                skip_remaining_shard = False
                for turn in audio_turns:
                    if use_ais_get_batch:
                        cut = _make_archive_member_cut(
                            tar_path=str(tar_path),
                            audio_filename=turn['value'],
                            duration=turn.get('duration'),
                            offset=turn.get('offset', 0.0),
                            sampling_rate=turn.get('sampling_rate', 16000),
                            source_type='url',
                        )
                        cut = cut.with_id(self._make_cut_id(cut, turn))
                    else:
                        try:
                            recording, _ = _next_matching_paired_audio(
                                tar,
                                turn['value'],
                                manifest_path=jsonl_path,
                                tar_path=tar_path,
                                skip_missing_manifest_entries=self.skip_missing_manifest_entries,
                            )
                        except (AudioLoadingError, OSError, RuntimeError, StopIteration) as ex:
                            message = (
                                f"Failed to load paired ShareGPT tar {tar_path!r} "
                                f"for manifest {jsonl_path!r}."
                            )
                            if not self.fault_tolerant_audio_loading:
                                raise RuntimeError(message) from ex
                            logging.warning(
                                "Skipping the remainder of paired ShareGPT shard because "
                                f"fault_tolerant_audio_loading=true: {message} {ex}"
                            )
                            skip_remaining_shard = True
                            break
                        cut = recording.to_cut().truncate(
                            offset=turn.get("offset", 0.0), duration=turn.get("duration")
                        )
                        cut = cut.with_id(self._make_cut_id(cut, turn))
                    turn["duration"] = cut.duration
                    turn["offset"] = cut.start
                    cuts.append(cut)
                if skip_remaining_shard:
                    break
                cuts = deque(cuts)

                if idx < slice_offset:
                    continue
                elif cntr == self.slice_length:
                    break

                yield NeMoMultimodalConversation(
                    id=data.get("id", "missing-example-id"),
                    turns=_ShareGPTConversationParser.create_turns(
                        self.audio_locator_tag, conversations, lambda t: cuts.popleft()
                    ),
                    token_equivalent_duration=self.token_equivalent_duration,
                )
                cntr += 1
            else:
                if tar is not None:
                    _validate_no_trailing_paired_audio(
                        tar,
                        manifest_path=jsonl_path,
                        tar_path=tar_path,
                        skip_missing_manifest_entries=self.skip_missing_manifest_entries,
                        fault_tolerant_audio_loading=self.fault_tolerant_audio_loading,
                    )

        self.epoch += 1

    def _iter_jsonl(self):
        paths = self.manifest_filepath
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(paths)
        for path in paths:
            jsonl_iter = load_jsonl(path)
            if self.shuffle_shards:
                jsonl_iter = list(jsonl_iter)
                rng.shuffle(jsonl_iter)
            for data in jsonl_iter:
                try:
                    conversations = _ShareGPTConversationParser(self.audio_placeholders, data).transform()
                    yield NeMoMultimodalConversation(
                        id=data.get("id", "missing-example-id"),
                        turns=_ShareGPTConversationParser.create_turns(
                            self.audio_locator_tag,
                            conversations,
                            lambda t, _p=path: self._resolve_cut_from_path(t, _p),
                        ),
                        token_equivalent_duration=self.token_equivalent_duration,
                    )
                except _SHAREGPT_AUDIO_LOADING_ERRORS as e:
                    if not self.fault_tolerant_audio_loading:
                        raise
                    logging.warning(
                        "Skipping ShareGPT sample due to audio loading failure: "
                        f"sample_id={data.get('id', 'missing-example-id')!r} manifest_path={path} "
                        f"error={type(e).__name__}: {e}"
                    )
        self.epoch += 1


@dataclass
class NeMoMultimodalConversationShareGPTWebdatasetAdapter(IteratorNode):
    """
    ``NeMoMultimodalConversationShareGPTWebdatasetAdapter`` reads ShareGPT format multimodal
    conversations from WebDataset tar archives and yields ``NeMoMultimodalConversation`` objects.

    Expected directory layout::

        data_dir/
          wids-meta.json                          # shard list metadata
          0/sharded_manifests/
            shard-0.tar      shard-0.tar.idx      # tar + optional index
            ...

    Each tar archive contains paired files per sample (same basename)::

        0.json   0.wav
        1.json   1.wav
        ...

    The ``.json`` files follow the ShareGPT schema (same as
    ``NeMoMultimodalConversationShareGPTJsonlAdapter``), and the ``.wav``
    (or other audio format) files contain the audio referenced via
    placeholders in conversation turns.

    When ``.tar.idx`` index files are present and ``shuffle_shards=True``,
    samples are read in random-access order without loading entire shards
    into memory.
    """

    data_dir: str
    audio_locator_tag: str
    audio_placeholders: Union[str, list[str]] = None
    token_equivalent_duration: float = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    indexed: bool = False
    indexes_root: Optional[Pathlike] = None
    wds_sample_index_version: int = 1
    index_pack: Optional[Pathlike] = None
    index_pack_max_open_files: int = 32
    skip_missing_manifest_entries: bool = False
    fault_tolerant_audio_loading: bool = True

    def __post_init__(self):
        if self.wds_sample_index_version not in (1, 2):
            raise ValueError(f"Unsupported wds_sample_index_version={self.wds_sample_index_version}; expected 1 or 2.")
        if self.wds_sample_index_version == 2 and not self.indexed:
            raise ValueError("wds_sample_index_version=2 requires indexed=true; sequential fallback is unsupported.")
        if self.index_pack is not None and not self.indexed:
            raise ValueError("Declaring index_pack requires indexed=true.")
        self.audio_placeholders = _normalize_audio_placeholders(self.audio_placeholders)
        self.epoch = 0
        self._tar_readers: list = []
        self._cum_lens: list[int] = []
        self._total_len = 0
        self._iter_state = PartitionedIndexedIterator()
        if self.index_pack is not None:
            self._init_packed(self.data_dir)
            return

        self._shard_paths = discover_webdataset_shards(
            self.data_dir,
            require_catalog=self.wds_sample_index_version == 2,
        )
        if self.indexed:
            self._init_indexed()

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def _init_indexed(self) -> None:
        from lhotse.indexing import index_file_path

        for p in self._shard_paths:
            if self.wds_sample_index_version == 2:
                self._tar_readers.append(
                    IndexedTarSampleBundleReader(p, idx_path=wds_v2_index_path(p, self.indexes_root))
                )
            else:
                self._tar_readers.append(IndexedTarSampleReader(p, idx_path=index_file_path(p, self.indexes_root)))
        cum = 0
        self._cum_lens.append(cum)
        for r in self._tar_readers:
            cum += len(r)
            self._cum_lens.append(cum)
        self._total_len = cum

    def _init_packed(self, source_spec) -> None:
        from lhotse.index_pack import index_pack_collection_key, open_index_pack

        if self.wds_sample_index_version != 2:
            raise ValueError("Packed ShareGPT WebDataset requires wds_sample_index_version=2.")
        pack = open_index_pack(self.index_pack)
        key = index_pack_collection_key("wds_tar", "wds_tar_v2", source_spec)
        reader = PackedTarSampleBundleReader(pack.collection(key), max_open_files=self.index_pack_max_open_files)
        self._packed_source = reader
        self._tar_readers = [reader]
        self._cum_lens = [0, len(reader)]
        self._total_len = len(reader)

    def __len__(self) -> int:
        if self.indexed:
            return self._total_len
        raise TypeError(
            "NeMoMultimodalConversationShareGPTWebdatasetAdapter has unknown length unless constructed with indexed=True."
        )

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._total_len
        for s in range(len(self._tar_readers)):
            if idx < self._cum_lens[s + 1]:
                return s, idx - self._cum_lens[s]
        raise IndexError(idx)

    def state_dict(self) -> dict:
        return {**self._iter_state.state_dict(), "epoch": self.epoch} if self.indexed else {}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)
        self.epoch = sd.get("epoch", 0)

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError(
                "NeMoMultimodalConversationShareGPTWebdatasetAdapter only supports __getitem__ when indexed=True."
            )
        idx = int(normalize_graph_token(token))
        shard_idx, local_idx = self._resolve(idx)
        sample = self._tar_readers[shard_idx][local_idx]
        convo = self._build_wds_sample(sample, sample_idx=idx)
        if convo is None:
            raise IndexError(
                f"ShareGPT WebDataset sample at index {idx} is not decodable; "
                "cannot satisfy random-access __getitem__."
            )
        return attach_graph_origin(convo, idx)

    def __iter__(self) -> Iterator[NeMoMultimodalConversation]:
        if self.indexed:
            yield from self._iter_indexed_node()
            return
        yield from self._iter_sequential()

    def _iter_indexed_node(self) -> Iterator[NeMoMultimodalConversation]:
        for global_idx in self._iter_state.iterate(self._total_len):
            shard_idx, local_idx = self._resolve(global_idx)
            sample = self._tar_readers[shard_idx][local_idx]
            convo = self._build_wds_sample(sample, sample_idx=global_idx)
            if convo is None:
                continue
            attach_graph_origin(convo, global_idx)
            yield convo
        self.epoch += 1

    def _get_rng(self) -> random.Random:
        return random.Random(resolve_seed(self.shard_seed) + self.epoch)

    def _build_wds_sample(self, sample, *, sample_idx: int | None = None):
        try:
            if self.wds_sample_index_version == 2:
                return self._yield_from_sample_bundle(sample)
            return self._yield_from_sample(*sample)
        except _SHAREGPT_AUDIO_LOADING_ERRORS as ex:
            if not self.fault_tolerant_audio_loading:
                raise
            logging.warning(
                "Skipping ShareGPT WebDataset sample due to audio loading failure: "
                f"sample_idx={sample_idx!r} error={type(ex).__name__}: {ex}"
            )
            return None

    def _yield_from_sample(self, json_data, audio_bytes, audio_name):
        sample_id = Path(audio_name).stem
        conversations = _ShareGPTConversationParser(self.audio_placeholders, json_data, audio_name).transform()
        recording = _sharegpt_recording_from_bytes(audio_bytes, recording_id=sample_id)
        base_cut = recording.to_cut()
        return NeMoMultimodalConversation(
            id=json_data.get("id", sample_id),
            turns=_ShareGPTConversationParser.create_turns(
                self.audio_locator_tag,
                conversations,
                lambda t: base_cut.truncate(offset=t.get("offset", 0.0), duration=t.get("duration")),
            ),
            token_equivalent_duration=self.token_equivalent_duration,
        )

    def _yield_from_sample_bundle(self, bundle):
        member_names = [member.name for member in bundle.audio_members]
        if len(member_names) != len(set(member_names)):
            raise ValueError(f"WDS sample {bundle.sample_key!r} contains duplicate audio member names.")
        members_by_name = {member.name: member for member in bundle.audio_members}
        members_by_basename: dict[str, list] = {}
        for member in bundle.audio_members:
            members_by_basename.setdefault(PurePosixPath(member.name).name, []).append(member)

        fallback = bundle.audio_members[0].name if len(bundle.audio_members) == 1 else None
        conversations = _ShareGPTConversationParser(
            self.audio_placeholders, bundle.json_data, audio_path_fallback=fallback
        ).transform()
        resolved_member_indexes: dict[str, int] = {}
        decoded_cuts = {}

        def resolve_cut(turn):
            requested = os.fspath(
                _ShareGPTConversationParser.expect_one_audio_path(
                    turn["value"], sample_id=bundle.sample_key, context="WDS audio turn value"
                )
            )
            member = members_by_name.get(requested)
            if member is None:
                basename = PurePosixPath(requested).name
                candidates = members_by_basename.get(basename, [])
                if not candidates:
                    if len(bundle.audio_members) == 1:
                        # Legacy conventional WDS often preserves the original
                        # source path in JSON while renaming the sole payload to
                        # ``<sample-key>.<codec>`` inside the tar.  With exactly
                        # one payload this binding is deterministic; multi-audio
                        # samples remain strict below.
                        (member,) = bundle.audio_members
                    else:
                        raise _ShareGPTMissingAudioMemberError(
                            f"WDS sample {bundle.sample_key!r} is missing audio member {requested!r}; "
                            f"available members are {sorted(member_names)!r}."
                        )
                elif len(candidates) > 1:
                    raise ValueError(
                        f"WDS sample {bundle.sample_key!r} has ambiguous audio member basename {basename!r}: "
                        f"{sorted(candidate.name for candidate in candidates)!r}."
                    )
                else:
                    (member,) = candidates

            audio_path_index = int(turn.get("audio_path_index", 0))
            previous_index = resolved_member_indexes.get(member.name)
            if previous_index is not None and previous_index != audio_path_index:
                raise ValueError(
                    f"WDS sample {bundle.sample_key!r} resolves multiple declared audio paths to duplicate "
                    f"member {member.name!r}."
                )
            resolved_member_indexes[member.name] = audio_path_index
            cut = decoded_cuts.get(member.name)
            if cut is None:
                recording_id = f"{bundle.sample_key}-{PurePosixPath(member.name).stem}"
                cut = _sharegpt_recording_from_bytes(member.data, recording_id=recording_id).to_cut()
                custom = {"_source_codec": PurePosixPath(member.name).suffix.lower().removeprefix(".") or None}
                if bundle.source_range_bytes is not None:
                    custom["_source_range_bytes"] = bundle.source_range_bytes
                    custom["_source_read_key"] = f"{bundle.source_path or '<unknown>'}#{bundle.sample_key}"
                cut = fastcopy(cut, custom=custom)
                decoded_cuts[member.name] = cut
            return cut.truncate(offset=turn.get("offset", 0.0), duration=turn.get("duration"))

        return NeMoMultimodalConversation(
            id=bundle.json_data.get("id", bundle.sample_key),
            turns=_ShareGPTConversationParser.create_turns(
                self.audio_locator_tag,
                conversations,
                resolve_cut,
            ),
            token_equivalent_duration=self.token_equivalent_duration,
        )

    def _iter_sequential(self):
        shard_paths = list(self._shard_paths)
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(shard_paths)
        for tar_path in shard_paths:
            with tarfile.open(tar_path, 'r:') as tar:
                members = (m for m in tar if m.isreg())
                for info_a, info_b in zip(members, members):
                    json_data, audio_bytes, audio_name = _split_json_audio_pair(
                        info_a.name,
                        tar.extractfile(info_a).read(),
                        info_b.name,
                        tar.extractfile(info_b).read(),
                    )
                    conversation = self._build_wds_sample(
                        (json_data, audio_bytes, audio_name), sample_idx=None
                    )
                    if conversation is not None:
                        yield conversation
        self.epoch += 1


class TarIterator:
    """
    Copy of lhotse.shar.readers.tar.TarIterator, modified to read both Lhotse-Shar style audio tar files
    and NeMo style audio tar files.
    """

    def __init__(self, source: Pathlike) -> None:
        self.source = source

    def __iter__(self):
        from lhotse.serialization import decode_json_line, deserialize_item, open_best
        from lhotse.shar.utils import fill_shar_placeholder

        with tarfile.open(fileobj=open_best(self.source, mode="rb"), mode="r|*") as tar:
            for (data, data_path), (meta, meta_path) in _iterate_tarfile_pairwise(tar):
                if meta_path is not None and meta_path.suffix == ".json":  # lhotse-shar tar format
                    if meta is not None:
                        meta = deserialize_item(decode_json_line(meta.decode("utf-8")))
                        fill_shar_placeholder(manifest=meta, data=data, tarpath=data_path)
                    yield meta, data_path
                else:  # nemo tar format
                    yield Recording.from_bytes(data, recording_id=data_path.stem), data_path
                    if meta is not None:  # the second item is also a recording despite the name
                        yield Recording.from_bytes(meta, recording_id=meta_path.stem), meta_path


def _iterate_tarfile_pairwise(
    tar_file: tarfile.TarFile,
):
    from lhotse.shar.readers.tar import parse_tarinfo

    result = []
    for tarinfo in tar_file:
        if len(result) == 2:
            yield tuple(result)
            result = []
        result.append(parse_tarinfo(tarinfo, tar_file))

    if len(result) == 2:
        yield tuple(result)

    if len(result) == 1:
        yield result[0], (None, None)


class NeMoMultimodalConversationTarWriter:
    def __init__(self, output_dir: str, shard_size: int = 100):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self._reset()
        self._setup_writers()

    def write(self, example: NeMoMultimodalConversation):
        self._maybe_increment_shard()
        serialized = example.to_dict()

        def change_audio_path(id, offset: float, duration: float):
            offset = f"{offset:.3f}" if offset > 0 else None
            new_path = f"{id}_{offset}_{duration:.3f}" if offset else id
            return new_path

        for turn in serialized["conversations"]:
            if turn["type"] == "audio":
                turn["value"] = Path(
                    change_audio_path(Path(turn['value']).stem, turn["offset"], turn["duration"]) + ".flac"
                ).name
                turn.pop(
                    "offset"
                )  # cut.load_audio() will load the segment based on the offset, so the new turn will start at offset=0
        self.manifest_writer.write(serialized)
        for cut in example.list_cuts():
            assert (
                cut.has_recording
            ), f"Cannot serialize multimodal conversation with cuts that have no recordings. We got: {cut}"
            self.tar_writer.write(
                change_audio_path(cut.recording.id, cut.start, cut.duration),
                cut.load_audio(),
                cut.sampling_rate,
                cut.recording,
            )
        self.item_cntr += 1

    def close(self):
        self.manifest_writer.close()
        self.tar_writer.close()

    def __enter__(self):
        self._reset()
        self.manifest_writer.__enter__()
        self.tar_writer.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _maybe_increment_shard(self):
        if self.item_cntr > 0 and self.item_cntr % self.shard_size == 0:
            self.item_cntr = 0
            self.shard_idx += 1
            self._setup_writers()

    def _reset(self):
        self.item_cntr = 0
        self.shard_idx = 0

    def _setup_writers(self):
        if not is_valid_url(self.output_dir):  # skip dir creation for URLs
            Path(self.output_dir).mkdir(exist_ok=True)
        self.manifest_writer = JsonlShardWriter(f"{self.output_dir}/manifest_{self.shard_idx}.jsonl", shard_size=None)
        self.tar_writer = AudioTarWriter(f"{self.output_dir}/audio_{self.shard_idx}.tar", shard_size=None)


class _ShareGPTAudioDecodeError(AudioLoadingError):
    """An audio payload was present but could not be decoded."""


def _next_matching_paired_audio(
    tar,
    expected_member: str,
    *,
    manifest_path: Pathlike,
    tar_path: Pathlike,
    skip_missing_manifest_entries: bool,
):
    """Read the expected tar member, optionally discarding members absent from JSONL."""
    while True:
        recording, audio_path = next(tar)
        actual_member = str(audio_path)
        if actual_member == expected_member:
            return recording, audio_path
        message = (
            "Paired tar member has no corresponding JSONL entry: "
            f"manifest={str(manifest_path)!r} tar={str(tar_path)!r} "
            f"expected={expected_member!r} actual={actual_member!r}."
        )
        if not skip_missing_manifest_entries:
            raise ValueError(message)
        logging.warning(
            "Skipping tar member because skip_missing_manifest_entries=true: "
            f"{message}"
        )


def _validate_no_trailing_paired_audio(
    tar,
    *,
    manifest_path: Pathlike,
    tar_path: Pathlike,
    skip_missing_manifest_entries: bool,
    fault_tolerant_audio_loading: bool,
) -> None:
    """Reject or explicitly tolerate a tar member left after JSONL exhaustion."""
    try:
        _, trailing_audio_path = next(tar)
    except StopIteration:
        return
    except (AudioLoadingError, EOFError, OSError, RuntimeError, ValueError, tarfile.TarError) as ex:
        message = (
            "Paired tar has unreadable trailing data after manifest exhaustion: "
            f"tar={str(tar_path)!r} manifest={str(manifest_path)!r}."
        )
        if fault_tolerant_audio_loading:
            logging.warning(
                "Skipping unreadable trailing audio because fault_tolerant_audio_loading=true: "
                f"{message} {ex}"
            )
            return
        raise AudioLoadingError(message) from ex

    message = (
        "Paired tar contains a member with no corresponding JSONL entry after "
        f"manifest exhaustion: tar={str(tar_path)!r} manifest={str(manifest_path)!r} "
        f"member={str(trailing_audio_path)!r}."
    )
    if skip_missing_manifest_entries:
        logging.warning(
            "Skipping trailing tar member because skip_missing_manifest_entries=true: "
            f"{message}"
        )
        return
    raise ValueError(message)


class _ShareGPTMissingTarMemberError(KeyError):
    """A paired ShareGPT manifest referenced an absent tar member."""


class _ShareGPTMissingAudioMemberError(ValueError):
    """A WDS sample did not contain the audio member named by its JSON."""


_SHAREGPT_AUDIO_LOADING_ERRORS = (
    AudioLoadingError,
    OSError,
    _ShareGPTMissingTarMemberError,
    _ShareGPTMissingAudioMemberError,
)


def _sharegpt_recording_from_bytes(data: bytes, *, recording_id: str) -> Recording:
    """Decode one embedded payload without reclassifying structural errors."""
    try:
        return Recording.from_bytes(data, recording_id=recording_id)
    except (AudioLoadingError, EOFError, OSError, RuntimeError, ValueError) as ex:
        raise _ShareGPTAudioDecodeError(
            f"Failed to decode ShareGPT audio payload for recording {recording_id!r}."
        ) from ex


def _normalize_audio_placeholders(val: Union[str, list[str], None]) -> list[str]:
    if val is None:
        return ["<sound>", "<speech>"]
    return [val] if isinstance(val, str) else list(val)


class _ShareGPTConversationParser:
    """Normalize ShareGPT multimodal records for the conversation adapters.

    ShareGPT audio examples are intentionally loose: audio paths may be stored
    in ``sound``, ``speech``, or ``ori_sound``, may be scalar or list-valued, and placement
    in the text is expressed with placeholders such as ``<sound>``. This class
    owns those conventions and emits the flat internal turn dictionaries shared
    by the JSONL and WebDataset adapters.
    """

    def __init__(self, placeholders: list[str], data: dict, audio_path_fallback: str | None = None) -> None:
        self.placeholders = placeholders
        self.data = data
        self.sample_id = data.get("id", "?")
        audio_path_value = audio_path_fallback
        audio_path_field = "fallback"
        for candidate in ("sound", "speech", "ori_sound"):
            candidate_value = data.get(candidate)
            if candidate_value not in (None, "", []):
                audio_path_value = candidate_value
                audio_path_field = candidate
                break
        self.audio_paths = self.normalize_audio_paths(
            audio_path_value, sample_id=self.sample_id, field_name=audio_path_field
        )

    def transform(self) -> list[dict]:
        """Convert one raw ShareGPT sample into text/audio turn dictionaries.

        User/human placeholders consume audio. Assistant turns are preserved as
        text so literal tokens such as an HTML ``<audio>`` tag are not mistaken
        for data references.
        """
        conversations = []
        placeholder_count = self._placeholder_count()
        if len(self.audio_paths) > 1 and placeholder_count > 1 and len(self.audio_paths) != placeholder_count:
            raise ValueError(
                f"ShareGPT sample id={self.sample_id} has {len(self.audio_paths)} audio paths but "
                f"{placeholder_count} audio placeholders. Use one path for all placeholders, one path per "
                f"placeholder, or a single placeholder for all paths."
            )

        audio_idx = 0
        for turn in self.data["conversations"]:
            role = self.role(turn)
            remaining = turn["value"]
            if not self.turn_can_consume_audio(turn):
                conversations.append({"type": "text", "from": role.title(), "value": remaining.strip()})
                continue

            found_any = False
            while True:
                idx, found = self.find_next_audio_placeholder(remaining, self.placeholders)
                if found is None:
                    if remaining.strip() or not found_any:
                        conversations.append({"type": "text", "from": role.title(), "value": remaining.strip()})
                    break

                found_any = True
                prefix = remaining[:idx]
                if prefix.strip():
                    conversations.append({"type": "text", "from": role.title(), "value": prefix.strip()})
                if not self.audio_paths:
                    raise ValueError(
                        f"Conversation turn contains audio placeholder '{found}' but no audio path was found in "
                        f"'sound', 'speech', 'ori_sound' fields or fallback for sample id={self.sample_id}"
                    )

                if len(self.audio_paths) > 1 and placeholder_count == 1:
                    path_indexes = range(len(self.audio_paths))
                elif len(self.audio_paths) > 1:
                    path_indexes = [audio_idx]
                    audio_idx += 1
                else:
                    path_indexes = [0]

                for path_idx in path_indexes:
                    audio_turn = {
                        "type": "audio",
                        "from": role.title(),
                        "value": self.audio_paths[path_idx],
                        "audio_path_index": path_idx,
                        "duration": self.audio_turn_field(turn, "duration", path_idx, self.sample_id),
                        "offset": self.audio_turn_field(turn, "offset", path_idx, self.sample_id, default=0.0),
                    }
                    if "sampling_rate" in turn:
                        audio_turn["sampling_rate"] = self.audio_turn_field(
                            turn, "sampling_rate", path_idx, self.sample_id
                        )
                    conversations.append(audio_turn)
                remaining = remaining[idx + len(found) :]
        return conversations

    def _placeholder_count(self) -> int:
        return sum(
            self.count_audio_placeholders(turn["value"], self.placeholders)
            for turn in self.data["conversations"]
            if self.turn_can_consume_audio(turn)
        )

    @staticmethod
    def create_turns(audio_locator_tag: str, conversations: list[dict], resolve_cut) -> list:
        """Build ``TextTurn`` / ``AudioTurn`` objects using ``resolve_cut(turn_dict)`` for audio."""
        turns = []
        for turn in conversations:
            if turn["type"] == "text":
                turns.append(TextTurn(value=turn["value"], role=turn["from"].lower()))
            else:
                cut = resolve_cut(turn)
                turns.append(
                    AudioTurn(
                        cut=cut,
                        text=cut.supervisions[0].text if cut.supervisions else None,
                        role=turn["from"].lower(),
                        audio_locator_tag=audio_locator_tag,
                    )
                )
        return turns

    @classmethod
    def expect_one_audio_path(cls, value, sample_id: str, context: str) -> Pathlike:
        paths = cls.normalize_audio_paths(value, sample_id=sample_id, field_name=context)
        if len(paths) != 1:
            raise ValueError(
                f"ShareGPT sample id={sample_id} resolved one audio turn to {len(paths)} audio paths. "
                f"Multiple paths must be expanded into separate audio turns before loading."
            )
        return paths[0]

    @staticmethod
    def normalize_audio_paths(value, sample_id: str, field_name: str) -> list[Pathlike]:
        if value is None or value == "":
            return []
        if isinstance(value, (str, os.PathLike)):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            paths = list(value)
            for idx, path in enumerate(paths):
                if not isinstance(path, (str, os.PathLike)):
                    raise ValueError(
                        f"ShareGPT sample id={sample_id} has unsupported {field_name}[{idx}]={path!r}; "
                        f"expected a string or os.PathLike audio path."
                    )
            return paths
        raise ValueError(
            f"ShareGPT sample id={sample_id} has unsupported {field_name}={value!r}; "
            f"expected a string, os.PathLike, or a list of audio paths."
        )

    @staticmethod
    def find_next_audio_placeholder(text: str, placeholders: list[str]) -> tuple[int, str] | tuple[None, None]:
        matches = [(idx, placeholder) for placeholder in placeholders if (idx := text.find(placeholder)) >= 0]
        if not matches:
            return None, None
        return min(matches, key=lambda item: item[0])

    @classmethod
    def count_audio_placeholders(cls, text: str, placeholders: list[str]) -> int:
        count = 0
        remaining = text
        while True:
            idx, placeholder = cls.find_next_audio_placeholder(remaining, placeholders)
            if placeholder is None:
                return count
            count += 1
            remaining = remaining[idx + len(placeholder) :]

    @staticmethod
    def role(turn: dict) -> str:
        return "user" if turn["from"].lower() in ("human", "user") else "assistant"

    @classmethod
    def turn_can_consume_audio(cls, turn: dict) -> bool:
        return cls.role(turn) == "user"

    @staticmethod
    def audio_turn_field(turn: dict, field_name: str, audio_idx: int, sample_id: str, default=None):
        value = turn.get(field_name, default)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = list(value)
            if len(values) == 1:
                return values[0]
            if audio_idx < len(values):
                return values[audio_idx]
            raise ValueError(
                f"ShareGPT sample id={sample_id} has {len(values)} values for turn field {field_name!r}, "
                f"but audio path index {audio_idx} was requested."
            )
        return value


def _flatten_conversation_audio_cuts(conversations):
    flat_cuts = []
    conversation_cut_ids = []
    for conversation in conversations:
        if isinstance(conversation, NeMoMultimodalConversation):
            cuts = conversation.list_cuts()
        elif isinstance(conversation, Formattable):
            # Prompt-formatted text-only examples share the SALM batching path
            # but intentionally carry no audio turns.
            cuts = []
        else:
            raise TypeError(f"Expected NeMoMultimodalConversation, got {type(conversation).__name__}.")
        flat_cuts.extend(cuts)
        conversation_cut_ids.append([cut.id for cut in cuts])
    return flat_cuts, conversation_cut_ids


def _conversation_audio_survivor_rows(conversations, conversation_cut_ids, survivor_rows):
    keep_rows = []
    ok = []
    for conversation, cut_ids in zip(conversations, conversation_cut_ids):
        if all(cut_id in survivor_rows for cut_id in cut_ids):
            keep_rows.extend(survivor_rows[cut_id] for cut_id in cut_ids)
            ok.append(conversation)
        else:
            logging.warning(f"Skipping conversation because it failed to load audio: {conversation.id=}")
    return keep_rows, ok


def _read_unpadded_audio_rows(cuts: CutSet, load_audio: AudioSamples):
    if load_audio.use_batch_loader and load_audio.ais_batch_loader is not None:
        cuts = load_audio.ais_batch_loader(cuts)
    executor = _get_executor(load_audio.num_workers, executor_type=load_audio._executor_type)
    audios, surviving = read_audio_from_cuts(
        cuts,
        executor=executor,
        suppress_errors=load_audio.fault_tolerant,
    )
    rows = [torch.as_tensor(audio) for audio in audios]
    mono_downmix = load_audio.mono_downmix
    if mono_downmix is None:
        mono_downmix = not rows or not all(row.ndim == 2 for row in rows)
    if not mono_downmix:
        raise ValueError("Packed conversation audio requires mono_downmix=True.")
    rows = [row.mean(dim=0) if row.ndim == 2 else row for row in rows]
    if any(row.ndim != 1 for row in rows):
        shapes = [tuple(row.shape) for row in rows]
        raise ValueError(f"Packed conversation audio requires one-dimensional waveforms, got {shapes}.")
    return [row.to(torch.float32) for row in rows], surviving
