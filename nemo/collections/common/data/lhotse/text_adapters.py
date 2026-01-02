# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import math
import random
import tarfile
from collections import deque
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence, Union

import numpy as np
import torch
from lhotse import CutSet, Recording
from lhotse.audio import AudioLoadingError
from lhotse.custom import CustomFieldMixin
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_matrices, collate_vectors
from lhotse.dataset.dataloading import resolve_seed
from lhotse.serialization import load_jsonl
from lhotse.shar import AudioTarWriter, JsonlShardWriter
from lhotse.utils import Pathlike, is_valid_url

from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths
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
class NeMoSFTJsonlAdapter:
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
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)

    def __iter__(self) -> Iterator[NeMoSFTExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                yield NeMoSFTExample(data, language=self.language)


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
    conversations: Sequence[NeMoMultimodalConversation],
) -> tuple[torch.Tensor, torch.Tensor, CutSet]:
    """
    Loads and collates audio data from a sequence of ``NeMoMultimodalConversation`` objects,
    preserving the order of conversations and turns.

    Fault tolerance skips over the conversations for which at least one audio turn failed to load
    due to ``lhotse.utils.AudioLoadingError``. This typically indicates corrupted data.

    Returns a tuple of:

    * ``audio`` tensor fp32 (B, T) or (B, C, T) if multi-channel

    * ``audio_lens`` tensor int64 (B)

    * ``conversations`` CutSet of NeMoMultimodalConversations that were successfully loaded.
    """

    audios = []
    all_cuts = []
    ok = []
    for conversation in conversations:
        assert isinstance(conversation, NeMoMultimodalConversation)
        try:
            conv_audios = []
            conv_cuts = []
            for cut in conversation.list_cuts():
                conv_audios.append(torch.as_tensor(cut.load_audio()).squeeze())
                conv_cuts.append(cut)
        except AudioLoadingError:
            continue
        else:
            audios.extend(conv_audios)
            all_cuts.extend(conv_cuts)
            ok.append(conversation)

    if not ok:
        ids = [c.id for c in conversations]
        logging.warning(f"An entire batch of conversations failed to load audios. Conversations ids: {ids}")
        return torch.tensor([]), torch.tensor([]), CutSet()

    audio_lens = torch.tensor([c.num_samples for c in all_cuts], dtype=torch.int64)
    if len(audios[0].shape) == 1:
        audios = collate_vectors(audios, padding_value=0.0)
    else:
        audios = collate_matrices([a.transpose(0, 1) for a in audios], padding_value=0.0).transpose(1, 2)

    return audios, audio_lens, CutSet(ok)


def _compute_num_audio_tokens(example: NeMoMultimodalConversation, mode: Literal["context", "answer", "all"]) -> int:
    if not example.has_audio_turns:
        return 0
    assert example.token_equivalent_duration is not None, (
        "Cannot compute the length of a NeMoMultimodalConversation: "
        "token_equivalent_duration must be set in order to estimate the number of tokens equivalent to audio turns. "
        "Did you forget to set token_equivalent_duration option in your dataloading config? "
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
        [
            # subtract 1 for each audio locator tag as its token will be replaced
            math.ceil(turn.cut.duration / example.token_equivalent_duration) - 1
            for turn in turns
            if isinstance(turn, AudioTurn)
        ]
    )


@registered_prompt_format_fn(NeMoMultimodalConversation)
def default_multimodal_conversation_prompt_format_fn(example: NeMoMultimodalConversation, prompt):
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
    return prompt.encode_dialog(turns)


@dataclass
class NeMoMultimodalConversationJsonlAdapter:
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
    """

    manifest_filepath: str | list[str]
    audio_locator_tag: str
    tarred_audio_filepaths: str | list[str] = None
    token_equivalent_duration: float = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    system_prompt: str | None = None
    slice_length: int | None = None

    def __post_init__(self):
        self.manifest_filepath = expand_sharded_filepaths(self.manifest_filepath)
        if self.tarred_audio_filepaths is not None:
            self.tarred_audio_filepaths = expand_sharded_filepaths(self.tarred_audio_filepaths)
            assert len(self.manifest_filepath) == len(
                self.tarred_audio_filepaths
            ), f"{len(self.manifest_filepath)} != {len(self.tarred_audio_filepaths)}"
        self.epoch = 0

    def __iter__(self) -> Iterator[NeMoMultimodalConversation]:
        if self.tarred_audio_filepaths is not None:
            yield from self._iter_tar()
        else:
            yield from self._iter_jsonl()

    def _should_skip(self, example: dict) -> bool:
        custom = example.get("custom")
        if custom is None:
            return False
        return bool(custom.get("_skipme", False))

    def _get_rng(self) -> random.Random:
        seed = resolve_seed(self.shard_seed) + self.epoch
        return random.Random(seed)

    def _iter_tar(self):
        paths = list(zip(self.manifest_filepath, self.tarred_audio_filepaths))
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(paths)
        for jsonl_path, tar_path in paths:
            jsonl = load_jsonl(jsonl_path)
            if self.slice_length is not None:
                jsonl = list(jsonl)
            tar = iter(TarIterator(tar_path))
            slice_offset = (
                rng.randint(0, len(jsonl) - self.slice_length)
                if self.slice_length is not None and self.slice_length < len(jsonl)
                else -1
            )
            cntr = 0
            for idx, data in enumerate(jsonl):
                audio_turns = [t for t in data["conversations"] if t["type"] == "audio"]
                cuts = []
                for turn in audio_turns:
                    recording, audio_path = next(tar)
                    audio_path = str(audio_path)
                    cut = recording.to_cut()
                    assert audio_path == turn['value'], (
                        f"Mismatch between JSONL and tar. JSONL defines audio path={turn['value']} but we got "
                        f"the following from tar {audio_path=}.\nBad inputs in: {jsonl_path=} {tar_path=}"
                    )
                    cuts.append(cut)
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
                if self.system_prompt is not None and turns[0].role != "system":
                    turns = [TextTurn(role="system", value=self.system_prompt)] + turns
                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=turns,
                    token_equivalent_duration=self.token_equivalent_duration,
                    custom=data.get("custom"),
                )
                cntr += 1

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
                if self._should_skip(data):
                    continue
                turns = [
                    (
                        TextTurn(
                            value=turn["value"],
                            role=turn["from"].lower(),
                        )
                        if turn["type"] == "text"
                        else AudioTurn(
                            cut=(cut := Recording.from_file(get_full_path(turn["value"], path)).to_cut()),
                            text=cut.supervisions[0].text if cut.supervisions else None,
                            role=turn["from"].lower(),
                            audio_locator_tag=self.audio_locator_tag,
                        )
                    )
                    for turn in data["conversations"]
                ]
                if self.system_prompt is not None and turns[0].role != "system":
                    turns = [TextTurn(role="system", value=self.system_prompt)] + turns
                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=turns,
                    token_equivalent_duration=self.token_equivalent_duration,
                    custom=data.get("custom"),
                )

        self.epoch += 1


@dataclass
class NeMoMultimodalConversationShareGPTJsonlAdapter:
    """
    ``NeMoMultimodalConversationShareGPTJsonlAdapter`` is used to read a ShareGPT format multimodal
    conversation JSONL and yield objects of type ``NeMoMultimodalConversation`` that can be sampled with Lhotse.

    We expect the following ShareGPT schema (contained in a single line per example)::

        {
            "id": str,
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
    token_equivalent_duration: float = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"
    slice_length: int | None = None

    def __post_init__(self):
        self.manifest_filepath = expand_sharded_filepaths(self.manifest_filepath)
        if self.tarred_audio_filepaths is not None:
            self.tarred_audio_filepaths = expand_sharded_filepaths(self.tarred_audio_filepaths)
            assert len(self.manifest_filepath) == len(
                self.tarred_audio_filepaths
            ), f"{len(self.manifest_filepath)} != {len(self.tarred_audio_filepaths)}"

        # Handle audio placeholders - default to both <sound> and <speech>
        if self.audio_placeholders is None:
            self.audio_placeholders = ["<sound>", "<speech>"]
        elif isinstance(self.audio_placeholders, str):
            self.audio_placeholders = [self.audio_placeholders]
        self.epoch = 0

    def __iter__(self) -> Iterator[NeMoMultimodalConversation]:
        if self.tarred_audio_filepaths is not None:
            yield from self._iter_tar()
        else:
            yield from self._iter_jsonl()

    def _get_rng(self) -> random.Random:
        seed = resolve_seed(self.shard_seed) + self.epoch
        return random.Random(seed)

    def _iter_tar(self):
        paths = list(zip(self.manifest_filepath, self.tarred_audio_filepaths))
        rng = self._get_rng()
        if self.shuffle_shards:
            rng.shuffle(paths)
        for jsonl_path, tar_path in paths:
            jsonl = load_jsonl(jsonl_path)
            if self.slice_length is not None:
                jsonl = list(jsonl)
            tar = iter(TarIterator(tar_path))
            slice_offset = (
                rng.randint(0, len(jsonl) - self.slice_length)
                if self.slice_length is not None and self.slice_length < len(jsonl)
                else -1
            )
            cntr = 0
            for idx, data in enumerate(jsonl):
                # Transform ShareGPT format to standard format
                conversations = self._transform_sharegpt_conversations(data)

                # Extract audio data from tar if needed
                audio_turns = [t for t in conversations if t["type"] == "audio"]
                cuts = []
                for turn in audio_turns:
                    recording, audio_path = next(tar)
                    audio_path = str(audio_path)
                    cut = recording.to_cut()
                    assert (
                        audio_path == turn['value']
                    ), f"Mismatch between JSONL and tar. JSONL defines audio path={turn['value']} but we got the following from tar {audio_path=}"
                    # Update the duration in the turn data with actual audio duration
                    turn["duration"] = cut.duration
                    cuts.append(cut)
                cuts = deque(cuts)

                if idx < slice_offset:
                    continue
                elif cntr == self.slice_length:
                    break

                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=self._create_turns(conversations, cuts, jsonl_path),
                    token_equivalent_duration=self.token_equivalent_duration,
                )
                cntr += 1

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
                # Transform ShareGPT format to standard format
                conversations = self._transform_sharegpt_conversations(data)

                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=self._create_turns(conversations, None, path),
                    token_equivalent_duration=self.token_equivalent_duration,
                )

        self.epoch += 1

    def _transform_sharegpt_conversations(self, data: dict) -> list[dict]:
        """
        Transform ShareGPT format conversations to standard format.
        Detects audio placeholders (<sound>, <speech>) and creates appropriate audio/text turns.
        """
        conversations = []
        audio_path = data.get("sound") or data.get("ori_sound")

        for turn in data["conversations"]:
            # Map ShareGPT roles to standard roles
            role = "user" if turn["from"].lower() == "human" else "assistant"

            # Check if this turn contains any audio placeholder
            found_placeholder = None
            for placeholder in self.audio_placeholders:
                if placeholder in turn["value"]:
                    found_placeholder = placeholder
                    break

            if found_placeholder:
                # Split text around audio placeholder
                parts = turn["value"].split(found_placeholder)

                # Add text before audio (if any)
                if parts[0].strip():
                    conversations.append({"type": "text", "from": role.title(), "value": parts[0].strip()})

                # Add audio turn
                if audio_path:
                    conversations.append(
                        {
                            "type": "audio",
                            "from": role.title(),
                            "value": audio_path,
                            "duration": 0.0,  # Will be set when loading actual audio
                        }
                    )

                # Add text after audio (if any)
                if len(parts) > 1 and parts[1].strip():
                    conversations.append({"type": "text", "from": role.title(), "value": parts[1].strip()})
            else:
                # Regular text turn
                conversations.append({"type": "text", "from": role.title(), "value": turn["value"]})

        return conversations

    def _create_turns(
        self, conversations: list[dict], cuts: deque = None, manifest_path: str = None
    ) -> list[Union[TextTurn, AudioTurn]]:
        """Create TextTurn and AudioTurn objects from conversation data."""
        turns = []

        for turn in conversations:
            if turn["type"] == "text":
                turns.append(TextTurn(value=turn["value"], role=turn["from"].lower()))
            else:  # audio turn
                if cuts is not None:
                    # Using tarred audio
                    cut = cuts.popleft()
                else:
                    # Load audio from file path
                    cut = Recording.from_file(get_full_path(turn["value"], manifest_path)).to_cut()

                turns.append(
                    AudioTurn(
                        cut=cut,
                        text=cut.supervisions[0].text if cut.supervisions else None,
                        role=turn["from"].lower(),
                        audio_locator_tag=self.audio_locator_tag,
                    )
                )

        return turns


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
        for turn in serialized["conversations"]:
            if turn["type"] == "audio":
                turn["value"] = Path(turn["value"]).with_suffix(".flac").name
        self.manifest_writer.write(serialized)
        for cut in example.list_cuts():
            assert (
                cut.has_recording
            ), f"Cannot serialize multimodal conversation with cuts that have no recordings. We got: {cut}"
            self.tar_writer.write(cut.recording.id, cut.load_audio(), cut.sampling_rate, cut.recording)
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
