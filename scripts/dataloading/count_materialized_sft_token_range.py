#!/usr/bin/env python3
"""Exactly count one newline-aligned byte range of a materialized SFT JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nemo.collections.common.data.lhotse.text_adapters import (
    MaterializedSFTMessagesExample,
    _encode_materialized_sft_messages,
)
from nemo.collections.common.tokenizers import AutoTokenizer

from count_materialized_sft_tokens import _finalize, _new_stats, _update_row_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--byte-start", required=True, type=int)
    parser.add_argument("--byte-end", required=True, type=int)
    parser.add_argument("--range-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--additional-special-token", action="append", default=[])
    return parser.parse_args()


def validate_range(path: Path, start: int, end: int) -> int:
    size = path.stat().st_size
    if not 0 <= start < end <= size:
        raise ValueError(f"Invalid byte range [{start}, {end}) for {path} size={size}")
    with path.open("rb") as stream:
        if start:
            stream.seek(start - 1)
            if stream.read(1) != b"\n":
                raise ValueError(f"Range start {start} is not newline-aligned")
        if end < size:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                raise ValueError(f"Range end {end} is not newline-aligned")
    return size


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    size = validate_range(path, args.byte_start, args.byte_end)
    tokenizer = AutoTokenizer(
        pretrained_model_name=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        additional_special_tokens=args.additional_special_token,
        include_special_tokens=False,
    )
    encode = tokenizer.text_to_ids
    stats = _new_stats()
    stats["bytes"] = args.byte_end - args.byte_start
    with path.open("rb") as stream:
        stream.seek(args.byte_start)
        row_index = 0
        while stream.tell() < args.byte_end:
            offset = stream.tell()
            raw = stream.readline()
            if not raw or stream.tell() > args.byte_end:
                raise ValueError(
                    f"Range crossed its boundary at {path}:{offset} end={args.byte_end}"
                )
            try:
                data = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSON at {path} byte {offset}: {error}") from error
            if set(data) != {"messages"}:
                raise ValueError(
                    f"{path} byte {offset} must have exactly top-level key 'messages'; "
                    f"got {sorted(data)}"
                )
            example = MaterializedSFTMessagesExample(
                id=f"{path.name}:{args.range_index}:{row_index}",
                messages=data["messages"],
                validate_chunk_tokenization=True,
            )
            encoded = _encode_materialized_sft_messages(example, encode)
            _update_row_stats(stats, example.messages, encoded["input_ids"], encoded["mask"])
            row_index += 1
        if stream.tell() != args.byte_end:
            raise ValueError(
                f"Range ended at {stream.tell()} instead of exact boundary {args.byte_end}"
            )
    report = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "additional_special_tokens": args.additional_special_token,
        "chunk_tokenization_validated": True,
        "path": str(path),
        "source_bytes": size,
        "byte_start": args.byte_start,
        "byte_end": args.byte_end,
        "range_index": args.range_index,
        "stats": _finalize(stats),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"event": "range_complete", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
