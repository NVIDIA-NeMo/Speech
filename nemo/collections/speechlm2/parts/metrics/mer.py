# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASCEND mixed error rate tokenization, shared by evaluation and RL rewards."""

import re

_CJK = re.compile(r"([㐀-䶿一-鿿豈-﫿぀-ヿ])")
_MER_PUNCT = re.compile(r"[^\w\s']|_", re.UNICODE)


def mer_tokens(text: str) -> str:
    """Characters for CJK, words for everything else (the ASCEND / SEAME convention)."""
    t = _MER_PUNCT.sub(" ", text.lower())
    t = _CJK.sub(r" \1 ", t)
    return " ".join(t.split())
