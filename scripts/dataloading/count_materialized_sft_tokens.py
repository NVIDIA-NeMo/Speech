#!/usr/bin/env python3
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
"""Exactly count tokens in identity-preformatted, conversation-packed SFT JSONL.

The command uses the same NeMo tokenizer API and the same fail-closed
concat-vs-chunk validation as the Lhotse ``materialized_sft_messages`` adapter.
It reports total and assistant/loss-bearing token mass per component and in
aggregate.  It also counts packed source conversations (top-level system
chunks) and potential duplicated/nested system-wire artifacts without
modifying the data.
"""

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

from nemo.collections.common.data.lhotse.text_adapters import (
    MaterializedSFTMessagesExample,
    _encode_materialized_sft_messages,
)
from nemo.collections.common.tokenizers import AutoTokenizer


def _name_value(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
    name, raw = value.split("=", 1)
    if not name or not raw:
        raise ValueError(f"{option} expects non-empty NAME=VALUE, got {value!r}")
    return name, raw


def _resolve_components(
    specs: list[str], expected_specs: list[str]
) -> dict[str, list[Path]]:
    components: dict[str, list[Path]] = {}
    for spec in specs:
        name, pattern = _name_value(spec, "--component")
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        if not matches:
            raise FileNotFoundError(
                f"Component {name!r} pattern matched no files: {pattern}"
            )
        components.setdefault(name, []).extend(matches)

    expected = {}
    for spec in expected_specs:
        name, raw_count = _name_value(spec, "--expected-files")
        expected[name] = int(raw_count)
    unknown = set(expected) - set(components)
    if unknown:
        raise ValueError(
            f"--expected-files names have no --component: {sorted(unknown)}"
        )
    for name, expected_count in expected.items():
        actual = len(components[name])
        if actual != expected_count:
            raise ValueError(
                f"Component {name!r} expected {expected_count} files, found {actual}"
            )
    return components


def _new_stats() -> dict:
    return {
        "files": 0,
        "bytes": 0,
        "packed_rows": 0,
        "source_conversations": 0,
        "messages": 0,
        "role_messages": Counter(),
        "total_tokens": 0,
        "assistant_tokens": 0,
        "min_row_tokens": None,
        "max_row_tokens": 0,
        "system_chunks_with_multiple_wire_system_starts": 0,
        "system_chunks_with_multiple_empty_wire_systems": 0,
    }


def _update_row_stats(stats: dict, messages: list[dict], input_ids, mask) -> None:
    row_tokens = len(input_ids)
    stats["packed_rows"] += 1
    stats["messages"] += len(messages)
    stats["total_tokens"] += row_tokens
    stats["assistant_tokens"] += int(mask.sum())
    stats["min_row_tokens"] = (
        row_tokens
        if stats["min_row_tokens"] is None
        else min(stats["min_row_tokens"], row_tokens)
    )
    stats["max_row_tokens"] = max(stats["max_row_tokens"], row_tokens)
    for message in messages:
        role = message["role"]
        content = message["content"]
        stats["role_messages"][role] += 1
        if role == "system":
            stats["source_conversations"] += 1
            if content.count("<|im_start|>system\n") > 1:
                stats["system_chunks_with_multiple_wire_system_starts"] += 1
            if content.count("<|im_start|>system\n<|im_end|>\n") > 1:
                stats["system_chunks_with_multiple_empty_wire_systems"] += 1


def _finalize(stats: dict) -> dict:
    stats = dict(stats)
    stats["role_messages"] = dict(sorted(stats["role_messages"].items()))
    stats["mean_row_tokens"] = (
        stats["total_tokens"] / stats["packed_rows"] if stats["packed_rows"] else 0.0
    )
    stats["assistant_token_fraction"] = (
        stats["assistant_tokens"] / stats["total_tokens"]
        if stats["total_tokens"]
        else 0.0
    )
    return stats


def _merge(target: dict, source: dict) -> None:
    for key in (
        "files",
        "bytes",
        "packed_rows",
        "source_conversations",
        "messages",
        "total_tokens",
        "assistant_tokens",
        "system_chunks_with_multiple_wire_system_starts",
        "system_chunks_with_multiple_empty_wire_systems",
    ):
        target[key] += source[key]
    target["role_messages"].update(source["role_messages"])
    if source["min_row_tokens"] is not None:
        target["min_row_tokens"] = (
            source["min_row_tokens"]
            if target["min_row_tokens"] is None
            else min(target["min_row_tokens"], source["min_row_tokens"])
        )
    target["max_row_tokens"] = max(target["max_row_tokens"], source["max_row_tokens"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF model or local tokenizer directory used by training.",
    )
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help=(
            "Component name and local JSONL glob; repeat for multiple "
            "patterns/components."
        ),
    )
    parser.add_argument(
        "--expected-files",
        action="append",
        default=[],
        metavar="NAME=N",
        help="Fail unless a component resolves to exactly N files.",
    )
    parser.add_argument(
        "--output", required=True, help="Atomic JSON report destination."
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--additional-special-token",
        action="append",
        default=[],
        help="Match recipe-added special tokens (for SpeechLM, pass <|audio|>).",
    )
    parser.add_argument(
        "--no-validate-chunk-tokenization",
        action="store_true",
        help="Skip concat-vs-chunk parity validation (not recommended for acceptance).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = _resolve_components(args.component, args.expected_files)
    for component_name, paths in components.items():
        print(
            json.dumps(
                {
                    "event": "resolved_component",
                    "component": component_name,
                    "files": len(paths),
                    "bytes": sum(path.stat().st_size for path in paths),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    tokenizer = AutoTokenizer(
        pretrained_model_name=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        additional_special_tokens=args.additional_special_token,
        include_special_tokens=False,
    )
    encode = tokenizer.text_to_ids
    print(
        json.dumps(
            {"event": "token_census_begin", "components": sorted(components)},
            sort_keys=True,
        ),
        flush=True,
    )

    component_reports = {}
    aggregate = _new_stats()
    for component_name, paths in components.items():
        stats = _new_stats()
        stats["files"] = len(paths)
        for path in paths:
            stats["bytes"] += path.stat().st_size
            print(
                json.dumps(
                    {
                        "event": "count_file_begin",
                        "component": component_name,
                        "path": str(path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            with path.open("r", encoding="utf-8") as stream:
                for line_index, line in enumerate(stream):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON at {path}:{line_index + 1}: {error}"
                        ) from error
                    if set(data) != {"messages"}:
                        raise ValueError(
                            f"{path}:{line_index + 1} must have exactly top-level "
                            "key 'messages'; "
                            f"got {sorted(data)}"
                        )
                    example = MaterializedSFTMessagesExample(
                        id=f"{path.name}:{line_index}",
                        messages=data["messages"],
                        validate_chunk_tokenization=(
                            not args.no_validate_chunk_tokenization
                        ),
                    )
                    encoded = _encode_materialized_sft_messages(example, encode)
                    _update_row_stats(
                        stats, example.messages, encoded["input_ids"], encoded["mask"]
                    )
        component_reports[component_name] = _finalize(stats)
        _merge(aggregate, stats)

    report = {
        "schema_version": 1,
        "metric": (
            "exact identity-preformatted tokens; assistant tokens selected "
            "by top-level role"
        ),
        "tokenizer": args.tokenizer,
        "additional_special_tokens": args.additional_special_token,
        "chunk_tokenization_validated": not args.no_validate_chunk_tokenization,
        "components": component_reports,
        "total": _finalize(aggregate),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
