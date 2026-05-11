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
#
# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 (MIT-licensed):
#   src/eva/utils/hash_utils.py
# Verbatim port of the hashing/diff utilities. The wrapping eva metrics
# framework (TaskCompletion, MetricScore, MetricContext) is intentionally
# not ported — NeMo has its own scoring shape (`metrics.json`).

"""Utilities for computing hashes and structured diffs of scenario DB states.

Used by the eva-style "DB-state hash matching" scoring mode (see
``evaluation/README.md`` "eva_airline domain notes"): hash the post-run DB and
the expected DB; if they differ, ``compute_db_diff`` produces a tables →
records → fields diff for human debugging.

Path-independent scoring: any sequence of agent actions that lands in the
right end state passes, regardless of how the agent got there.
"""

import hashlib
import json
import math
from typing import Any

# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3
ORDER_INDEPENDENT_LIST_FIELDS: set[str] = {
    "standby_list",
    "notifications",
    "bookings",
    "system_accounts",
    "group_memberships",
    "asset_recoveries",
}

# Top-level keys in the DB excluded from hashing. eva uses {"session"} for auth
# session metadata. We may extend this if any internal markers ever migrate
# from `shared_state` into `state["db"]` (today they don't — `_call_counts` etc.
# live in `shared_state` itself, never in the hashed payload).
HASH_EXCLUDED_KEYS: set[str] = {"session"}


def normalize_for_comparison(obj: Any) -> Any:
    """Recursively normalize values for consistent comparison and hashing.

    Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — verbatim.

    Normalizations applied:
        - float → int when the value is a whole number and finite (e.g. 1.0 → 1)
        - "none" / "null" string → None (case-insensitive, stripped)
        - Recurse into dict values and list elements
        - Lists under ``ORDER_INDEPENDENT_LIST_FIELDS`` are sorted before
          serialization so element ordering doesn't affect hashes.

    Eliminates common false-mismatches between the agent's DB and the expected
    DB (e.g. one side stores `1` and the other `1.0`).
    """
    if isinstance(obj, dict):
        normalized = {}
        for k, v in obj.items():
            norm_v = normalize_for_comparison(v)
            if k in ORDER_INDEPENDENT_LIST_FIELDS and isinstance(norm_v, list):
                norm_v = sorted(norm_v, key=lambda x: json.dumps(x, sort_keys=True, default=str))
            normalized[k] = norm_v
        return normalized
    if isinstance(obj, list):
        return [normalize_for_comparison(item) for item in obj]
    if isinstance(obj, float):
        if math.isfinite(obj) and obj.is_integer():
            return int(obj)
        return obj
    if isinstance(obj, str):
        if obj.strip().lower() in ("none", "null"):
            return None
        return obj
    return obj


def get_dict_hash(obj: dict) -> str:
    """Compute SHA-256 hash of a dict (canonical JSON serialization).

    Adapted from https://github.com/ServiceNow/eva/tree/0.1.3.

    Follows tau-2 bench's approach:
    - Drop keys in ``HASH_EXCLUDED_KEYS`` (e.g. ``session``) from the top level.
    - Normalize via ``normalize_for_comparison`` so float/int and "none"/None
      don't cause false mismatches.
    - Serialize with ``sort_keys=True``, no whitespace, ``default=str`` for
      non-JSON-serializable types.
    - SHA-256 of the resulting string.
    """
    obj_for_hash = {k: v for k, v in obj.items() if k not in HASH_EXCLUDED_KEYS} if isinstance(obj, dict) else obj
    normalized = normalize_for_comparison(obj_for_hash)
    serialized = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_db_diff(expected_db: dict, actual_db: dict) -> dict:
    """Compute a structured diff between expected and actual DB states.

    Adapted from https://github.com/ServiceNow/eva/tree/0.1.3.

    Returned dict has shape::

        {
            "tables_added":    [name, ...],   # in actual but not expected
            "tables_removed":  [name, ...],   # in expected but not actual
            "tables_modified": {table_name: {records_added, records_removed, records_modified}},
        }

    Used for human debugging when ``get_dict_hash(expected) != get_dict_hash(actual)``.
    Not used for scoring itself (the hash comparison is the verdict).
    """
    diff: dict[str, Any] = {"tables_added": [], "tables_removed": [], "tables_modified": {}}

    expected_tables = set(expected_db.keys())
    actual_tables = set(actual_db.keys())

    diff["tables_added"] = sorted(actual_tables - expected_tables)
    diff["tables_removed"] = sorted(expected_tables - actual_tables)

    common_tables = expected_tables & actual_tables
    for table in sorted(common_tables):
        expected_table = expected_db[table]
        actual_table = actual_db[table]

        # Handle non-dict tables (e.g. a list-valued root key)
        if not isinstance(expected_table, dict) or not isinstance(actual_table, dict):
            if expected_table != actual_table:
                diff["tables_modified"][table] = {
                    "type": "non_dict_table",
                    "expected": expected_table,
                    "actual": actual_table,
                }
            continue

        table_diff = _compute_table_diff(expected_table, actual_table)
        if table_diff:
            diff["tables_modified"][table] = table_diff

    return diff


def _compute_table_diff(expected_table: dict, actual_table: dict) -> dict | None:
    """Diff between two dict-valued tables; returns None if identical."""
    table_diff: dict[str, Any] = {"records_added": [], "records_removed": [], "records_modified": {}}

    expected_keys = set(expected_table.keys())
    actual_keys = set(actual_table.keys())

    table_diff["records_added"] = sorted(actual_keys - expected_keys)
    table_diff["records_removed"] = sorted(expected_keys - actual_keys)

    common_keys = expected_keys & actual_keys
    for key in sorted(common_keys):
        record_diff = _compute_record_diff(expected_table[key], actual_table[key])
        if record_diff:
            table_diff["records_modified"][str(key)] = record_diff

    if not table_diff["records_added"] and not table_diff["records_removed"] and not table_diff["records_modified"]:
        return None

    return table_diff


def _compute_record_diff(
    expected_record: Any, actual_record: Any, path: str = "", field_name: str = ""
) -> dict | None:
    """Recursively diff two record values; returns None if identical."""
    expected_record = normalize_for_comparison(expected_record)
    actual_record = normalize_for_comparison(actual_record)

    if expected_record == actual_record:
        return None

    if type(expected_record) is not type(actual_record):
        return {
            "type": "type_mismatch",
            "expected": expected_record,
            "actual": actual_record,
            "expected_type": type(expected_record).__name__,
            "actual_type": type(actual_record).__name__,
        }

    if isinstance(expected_record, dict):
        field_diff: dict[str, Any] = {"fields_added": [], "fields_removed": [], "fields_modified": {}}
        expected_keys = set(expected_record.keys())
        actual_keys = set(actual_record.keys())
        field_diff["fields_added"] = sorted(actual_keys - expected_keys)
        field_diff["fields_removed"] = sorted(expected_keys - actual_keys)
        for key in sorted(expected_keys & actual_keys):
            nested_path = f"{path}.{key}" if path else key
            nested_diff = _compute_record_diff(expected_record[key], actual_record[key], nested_path, field_name=key)
            if nested_diff:
                field_diff["fields_modified"][key] = nested_diff
        if not field_diff["fields_added"] and not field_diff["fields_removed"] and not field_diff["fields_modified"]:
            return None
        return field_diff

    if isinstance(expected_record, list):
        if field_name in ORDER_INDEPENDENT_LIST_FIELDS:
            sort_key = lambda x: json.dumps(x, sort_keys=True, default=str)  # noqa: E731
            expected_record = sorted(expected_record, key=sort_key)
            actual_record = sorted(actual_record, key=sort_key)

        if len(expected_record) != len(actual_record):
            return {
                "type": "list_length_mismatch",
                "expected": expected_record,
                "actual": actual_record,
                "expected_length": len(expected_record),
                "actual_length": len(actual_record),
            }

        list_diffs = []
        for i, (exp_item, act_item) in enumerate(zip(expected_record, actual_record)):
            item_diff = _compute_record_diff(exp_item, act_item, f"{path}[{i}]")
            if item_diff:
                list_diffs.append({"index": i, "diff": item_diff})

        if not list_diffs:
            return None

        return {"type": "list_differences", "differences": list_diffs}

    # Primitive
    return {"type": "value_mismatch", "expected": expected_record, "actual": actual_record}
