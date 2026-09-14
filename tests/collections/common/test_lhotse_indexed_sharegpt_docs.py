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

import dataclasses
import json
import textwrap
from pathlib import Path

import yaml
from click.testing import CliRunner
from scripts.dataloading.build_indexes import main as build_indexes_cli
from scripts.dataloading.convert_indexes_to_idxpack import main as convert_indexes_cli
from scripts.dataloading.validate_dataloader import cli as validate_dataloader_cli

from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversationShareGPTJsonlAdapter,
    NeMoMultimodalConversationShareGPTWebdatasetAdapter,
    _ShareGPTConversationParser,
)

DOC_PATH = Path(__file__).parents[3] / "docs/source/speechlm2/datasets.rst"
SECTION_TITLE = "Indexed ShareGPT audio formats"


def _section_text() -> str:
    text = DOC_PATH.read_text()
    start = text.index(f"{SECTION_TITLE}\n{'-' * len(SECTION_TITLE)}")
    return text[start:]


def _code_blocks(language: str) -> list[str]:
    lines = _section_text().splitlines()
    marker = f".. code-block:: {language}"
    blocks = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != marker:
            idx += 1
            continue
        idx += 1
        while idx < len(lines) and (not lines[idx].strip() or lines[idx].startswith("   :")):
            idx += 1
        block = []
        while idx < len(lines) and (not lines[idx] or lines[idx].startswith("    ")):
            block.append(lines[idx][4:] if lines[idx].startswith("    ") else "")
            idx += 1
        blocks.append(textwrap.dedent("\n".join(block)).strip())
    return blocks


def _named_yaml_examples() -> dict[str, list[dict]]:
    examples = {}
    for block in _code_blocks("yaml"):
        first_line = block.splitlines()[0] if block else ""
        prefix = "# indexed-sharegpt-example: "
        if first_line.startswith(prefix):
            examples[first_line.removeprefix(prefix)] = yaml.safe_load(block)
    return examples


def _json_examples() -> dict[str, dict]:
    examples = {}
    for block in _code_blocks("json"):
        value = json.loads(block)
        examples[value["id"]] = value
    return examples


def test_all_indexed_sharegpt_success_formats_are_complete_parseable_yaml():
    examples = _named_yaml_examples()
    assert set(examples) == {
        "loose-relative-scalar",
        "loose-remote-list",
        "loose-absolute-prefix-map",
        "aligned-jsonl-tar",
        "unordered-tar-collection",
        "conventional-wds-v2",
        "variable-wds-v2",
        "text-only",
    }

    adapter_fields = {
        "share_gpt": {field.name for field in dataclasses.fields(NeMoMultimodalConversationShareGPTJsonlAdapter)},
        "share_gpt_webdataset": {
            field.name for field in dataclasses.fields(NeMoMultimodalConversationShareGPTWebdatasetAdapter)
        },
    }
    generic_fields = {"type", "force_finite"}
    renamed_fields = {"shuffle": "shuffle_shards"}
    for name, config in examples.items():
        assert isinstance(config, list) and len(config) == 1, name
        (entry,) = config
        assert entry["indexed"] is True, name
        assert entry["audio_locator_tag"] == "<|audio|>", name
        assert entry["audio_placeholders"] == ["<sound>", "<speech>"], name
        assert entry["force_finite"] is False, name
        assert entry["indexes_root"].startswith("/index-mirror/"), name
        for field_name in entry:
            public_name = renamed_fields.get(field_name, field_name)
            assert public_name in adapter_fields[entry["type"]] or field_name in generic_fields, (name, field_name)

    assert "index_pack" not in examples["aligned-jsonl-tar"][0]
    for name in set(examples) - {"aligned-jsonl-tar"}:
        assert examples[name][0]["index_pack"].endswith(".idxpack"), name

    collection = examples["unordered-tar-collection"][0]
    assert collection["tar_lookup_mode"] == "collection"
    assert collection["tar_routing_filepath"] == "multi-clip.sgroute"
    wds_examples = (examples["conventional-wds-v2"][0], examples["variable-wds-v2"][0])
    assert all(entry["wds_sample_index_version"] == 2 for entry in wds_examples)


def test_documented_json_rows_exercise_real_sharegpt_parser_contract():
    examples = _json_examples()
    required_ids = {
        "loose-relative-scalar",
        "loose-remote-list",
        "loose-absolute-prefix-map",
        "aligned-jsonl-tar",
        "unordered-tar-collection",
        "conventional-wds-v2",
        "variable-wds-v2",
        "text-only",
        "alias-precedence",
        "alias-speech",
        "alias-ori-sound",
    }
    assert required_ids <= set(examples)

    def audio_values(sample_id: str) -> list[str]:
        turns = _ShareGPTConversationParser(["<sound>", "<speech>"], examples[sample_id]).transform()
        return [turn["value"] for turn in turns if turn["type"] == "audio"]

    assert audio_values("loose-relative-scalar") == ["audio/request.wav"]
    assert audio_values("loose-remote-list") == [
        "s3://example-audio/clips/left.flac",
        "ais://example-audio/clips/right.flac",
    ]
    assert audio_values("loose-absolute-prefix-map") == ["/lustre/source/audio/request.wav"]
    assert audio_values("unordered-tar-collection") == [
        "clips/left.flac",
        "clips/right.flac",
    ]
    assert audio_values("variable-wds-v2") == ["nested/1.flac", "nested/1.1.flac"]
    assert audio_values("text-only") == []
    assert audio_values("alias-precedence") == ["chosen-by-sound.wav"]
    assert audio_values("alias-speech") == ["chosen-by-speech.wav"]
    assert audio_values("alias-ori-sound") == ["chosen-by-ori.wav"]


def test_documented_index_commands_match_current_click_flags():
    section = _section_text()
    assert "--kind wds_tar_v2" in section
    assert "--kind jsonl --kind nemo_tar --kind sharegpt_route" in section
    assert "role=manifest, kind=jsonl" in section
    assert "role=tar_collection, kind=nemo_tar" in section
    assert "role=wds_tar, kind=wds_tar_v2" in section
    assert "--pack-with-tar-offsets" in section
    assert "--mode full --no-metadata-only" in section

    runner = CliRunner()
    for command, expected_options in (
        (
            build_indexes_cli,
            ("--indexes-root", "--kind", "wds_tar_v2", "sharegpt_route"),
        ),
        (
            convert_indexes_cli,
            ("--indexes-root", "--output", "--native-tar-paths-only", "--overwrite"),
        ),
        (
            validate_dataloader_cli,
            ("--mode", "--no-metadata-only", "--phase", "--checkpoint-at"),
        ),
    ):
        result = runner.invoke(command, ["--help"])
        assert result.exit_code == 0, result.output
        for option in expected_options:
            assert option in result.output


def test_docs_state_fail_closed_and_portability_boundaries():
    section = _section_text()
    required_phrases = (
        "no recursive WDS scan",
        "offset-bearing",
        "full validator",
        "not cluster-path portable",
        "WDS v2",
        "exactly one JSON member",
        "non-contiguous sample-key reuse",
        "ambiguous basename",
        "sequential fallback is not allowed",
    )
    for phrase in required_phrases:
        assert phrase in section
