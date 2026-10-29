# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The packaged MER scorer preserves the public ASCEND tutorial convention."""

import importlib.util
from pathlib import Path

import pytest

from nemo.collections.speechlm2.parts.metrics.mer import mer_tokens


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("你好World！", "你 好 world"),
        ("We can't go_今天.", "we can't go 今 天"),
        ("我有42元", "我 有 42 元"),
        ("HELLO, WORLD!", "hello world"),
        ("!? _", ""),
        ("東京 カナ", "東 京 カ ナ"),
    ],
)
def test_mer_units(text, expected):
    assert mer_tokens(text) == expected
    tutorial = Path(__file__).resolve().parents[3] / "examples/speechlm2/finetuning/task_metrics.py"
    spec = importlib.util.spec_from_file_location("tutorial_metrics", tutorial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert mer_tokens(text) == module.mer_tokens(text)
