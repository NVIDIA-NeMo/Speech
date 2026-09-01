# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import json
from functools import partial

import pytest
import yaml
from click.testing import CliRunner
from lhotse import CutSet
from lhotse.indexing import create_jsonl_index

from nemo.collections.common.data.lhotse import text_adapters
from nemo.collections.common.data.lhotse.dataloader import (
    resample,
    tokenize_with_prompt,
)
from nemo.collections.common.prompts import Nemotron3p5PromptFormatter
from scripts.dataloading import count_materialized_sft_tokens as token_counter
from scripts.dataloading.convert_indexes_to_idxpack import (
    main as convert_indexes_to_idxpack,
)


class CharacterTokenizer:
    def text_to_ids(self, text):
        return [ord(char) for char in text]


class BoundaryMergingTokenizer(CharacterTokenizer):
    def text_to_ids(self, text):
        if text == "ab":
            return [999]
        return super().text_to_ids(text)


def _packed_messages():
    return [
        {
            "role": "system",
            "content": (
                "<|im_start|>system\n<|im_start|>system\n"
                "<|im_start|>assistant\nnested text<|im_end|>\n"
            ),
        },
        {"role": "user", "content": "<|im_start|>user\nquestion one<|im_end|>\n"},
        {
            "role": "assistant",
            "content": "<|im_start|>assistant\nanswer one<|im_end|>\n",
        },
        {"role": "tool", "content": "<|im_start|>tool\ntool result<|im_end|>\n"},
        {
            "role": "assistant",
            "content": "<|im_start|>assistant\nanswer two<|im_end|>\n",
        },
        {"role": "system", "content": "<|im_start|>system\n<|im_end|>\n"},
        {"role": "user", "content": "<|im_start|>user\nquestion two<|im_end|>\n"},
        {
            "role": "assistant",
            "content": "<|im_start|>assistant\nanswer three<|im_end|>\n",
        },
    ]


def test_identity_prompt_preserves_packed_rows_and_masks_only_top_level_assistant():
    messages = _packed_messages()
    example = text_adapters.MaterializedSFTMessagesExample(
        id="packed", messages=messages
    )
    tokenizer = CharacterTokenizer()
    prompt = Nemotron3p5PromptFormatter(tokenizer)

    tokenize_with_prompt(example, tokenizer, prompt)

    expected_text = "".join(message["content"] for message in messages)
    assert example.input_ids.tolist() == tokenizer.text_to_ids(expected_text)
    assert example.context_ids.tolist() == example.input_ids.tolist()
    expected_mask = [
        message["role"] == "assistant"
        for message in messages
        for _ in message["content"]
    ]
    assert example.mask.tolist() == expected_mask
    assert example.answer_ids.tolist() == [
        token_id
        for token_id, selected in zip(example.input_ids.tolist(), expected_mask)
        if selected
    ]
    # The assistant marker nested inside system content is data and stays masked.
    nested_offset = messages[0]["content"].index("<|im_start|>assistant")
    assert not example.mask[nested_offset]
    assert sum(message["role"] == "system" for message in messages) == 2


def test_exact_counter_reports_total_and_all_assistant_tokens():
    messages = _packed_messages()
    example = text_adapters.MaterializedSFTMessagesExample(
        id="counter", messages=messages
    )
    tokenizer = CharacterTokenizer()
    encoded = text_adapters._encode_materialized_sft_messages(
        example, tokenizer.text_to_ids
    )
    stats = token_counter._new_stats()
    stats["files"] = 1
    stats["bytes"] = 123
    token_counter._update_row_stats(
        stats, messages, encoded["input_ids"], encoded["mask"]
    )
    report = token_counter._finalize(stats)

    assert report["packed_rows"] == 1
    assert report["source_conversations"] == 2
    assert report["total_tokens"] == sum(
        len(message["content"]) for message in messages
    )
    assert report["assistant_tokens"] == sum(
        len(message["content"])
        for message in messages
        if message["role"] == "assistant"
    )
    assert report["system_chunks_with_multiple_wire_system_starts"] == 1


def test_identity_prompt_rejects_cross_chunk_tokenization_change():
    example = text_adapters.MaterializedSFTMessagesExample(
        id="unstable",
        messages=[
            {"role": "system", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
    )
    tokenizer = BoundaryMergingTokenizer()
    with pytest.raises(ValueError, match="not concatenation-stable"):
        tokenize_with_prompt(example, tokenizer, Nemotron3p5PromptFormatter(tokenizer))


@pytest.mark.parametrize(
    "data,match",
    [
        ({"messages": [], "extra": 1}, "top-level key"),
        (
            {"messages": [{"role": "assistant", "content": "ok", "extra": 1}]},
            "exactly the keys",
        ),
        ({"messages": [{"role": "Assistant", "content": "ok"}]}, "unsupported role"),
    ],
)
def test_identity_prompt_schema_is_fail_closed(data, match):
    if set(data) == {"messages"}:
        example = text_adapters.MaterializedSFTMessagesExample(
            id="bad", messages=data["messages"]
        )
    else:
        with pytest.raises(ValueError, match=match):
            text_adapters._transform_materialized_sft_messages(
                data, "bad", validate_chunk_tokenization=True
            )
        return
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match=match):
        tokenize_with_prompt(example, tokenizer, Nemotron3p5PromptFormatter(tokenizer))


def test_materialized_sft_jsonl_uses_one_index_pack(tmp_path, monkeypatch):
    manifest = tmp_path / "materialized.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "s0"},
                {"role": "assistant", "content": "a0"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "assistant", "content": "a1"},
            ]
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    create_jsonl_index(manifest)
    config = tmp_path / "materialized.yaml"
    config.write_text(
        yaml.safe_dump({"type": "materialized_sft_messages", "paths": str(manifest)})
    )
    pack = tmp_path / "materialized.idxpack"
    result = CliRunner().invoke(
        convert_indexes_to_idxpack, ["--output", str(pack), str(config)]
    )
    assert result.exit_code == 0, result.output

    def fail_expand(*args, **kwargs):
        raise AssertionError(
            "packed materialized SFT construction must not expand paths"
        )

    monkeypatch.setattr(text_adapters, "expand_sharded_filepaths", fail_expand)
    iterator = text_adapters.MaterializedSFTMessagesAdapter(
        str(manifest), indexed=True, index_pack=pack
    )
    examples = list(iterator)
    assert [example.id for example in examples] == [
        "materialized-000000000000",
        "materialized-000000000001",
    ]
    assert [example.messages for example in examples] == [
        row["messages"] for row in rows
    ]

    # Exercise the same repeat -> resample -> prompt-tokenize graph restoration
    # path used by answer-token measurement and production sampling.
    iterator = text_adapters.MaterializedSFTMessagesAdapter(
        str(manifest), indexed=True, index_pack=pack
    )
    tokenizer = CharacterTokenizer()
    cuts = (
        CutSet(iterator)
        .repeat(preserve_id=True)
        .map(partial(resample, sampling_rate=16000), apply_fn=None)
        .map(
            partial(
                tokenize_with_prompt,
                tokenizer=tokenizer,
                prompt_format=Nemotron3p5PromptFormatter(tokenizer),
            ),
            apply_fn=None,
        )
    )
    restored = cuts.data[(0, 0)]
    assert restored.messages == rows[0]["messages"]
    assert restored.mask.any()
