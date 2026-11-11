# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import sys
import types
from unittest.mock import MagicMock

import pytest


class TestPanguCompat:
    @pytest.mark.unit
    def test_pangu_legacy_spacing(self, monkeypatch):
        """Test fallback when pangu exposes spacing (<=4.0.6.1)."""
        mock_pangu = types.ModuleType("pangu")
        mock_pangu.spacing = MagicMock(side_effect=lambda x: f"spaced:{x}")
        monkeypatch.setitem(sys.modules, "pangu", mock_pangu)

        from nemo.collections.common.tokenizers import chinese_tokenizers

        monkeypatch.setattr(chinese_tokenizers, "spacing", mock_pangu.spacing)

        processor = chinese_tokenizers.ChineseProcessor()
        # Mock normalizer convert to avoid opencc requirement
        processor.normalizer = MagicMock()
        processor.normalizer.convert = lambda x: x

        res = processor.detokenize(["测试", "test"])
        assert "spaced:" in res

    @pytest.mark.unit
    def test_pangu_space_text_v6(self, monkeypatch):
        """Test fallback when pangu exposes space_text (>=6.0.0)."""
        mock_pangu = types.ModuleType("pangu")
        mock_pangu.space_text = MagicMock(side_effect=lambda x: f"spaced_v6:{x}")
        monkeypatch.setitem(sys.modules, "pangu", mock_pangu)

        from nemo.collections.common.tokenizers import chinese_tokenizers

        monkeypatch.setattr(chinese_tokenizers, "spacing", mock_pangu.space_text)

        processor = chinese_tokenizers.ChineseProcessor()
        processor.normalizer = MagicMock()
        processor.normalizer.convert = lambda x: x

        res = processor.detokenize(["测试", "test"])
        assert "spaced_v6:" in res

    @pytest.mark.unit
    def test_pangu_spacing_text_v5(self, monkeypatch):
        """Test fallback when pangu exposes spacing_text (==5.0.0)."""
        mock_pangu = types.ModuleType("pangu")
        mock_pangu.spacing_text = MagicMock(side_effect=lambda x: f"spaced_v5:{x}")
        monkeypatch.setitem(sys.modules, "pangu", mock_pangu)

        from nemo.collections.common.tokenizers import chinese_tokenizers

        monkeypatch.setattr(chinese_tokenizers, "spacing", mock_pangu.spacing_text)

        processor = chinese_tokenizers.ChineseProcessor()
        processor.normalizer = MagicMock()
        processor.normalizer.convert = lambda x: x

        res = processor.detokenize(["测试", "test"])
        assert "spaced_v5:" in res

    @pytest.mark.unit
    def test_pangu_missing_raises_informative_error(self, monkeypatch):
        """Test that missing pangu raises a clear ImportError on detokenize."""
        from nemo.collections.common.tokenizers import chinese_tokenizers

        monkeypatch.setattr(chinese_tokenizers, "spacing", None)

        processor = chinese_tokenizers.ChineseProcessor()
        processor.normalizer = MagicMock()
        processor.normalizer.convert = lambda x: x

        with pytest.raises(ImportError, match="installed `pangu`"):
            processor.detokenize(["测试", "test"])

    @pytest.mark.unit
    def test_en_ja_pangu_fallback(self, monkeypatch):
        """Test JaMecabProcessor detokenize fallback with pangu space_text (>=6.0.0)."""
        mock_pangu = types.ModuleType("pangu")
        mock_pangu.space_text = MagicMock(side_effect=lambda x: f"ja_spaced:{x}")
        monkeypatch.setitem(sys.modules, "pangu", mock_pangu)

        from nemo.collections.common.tokenizers import en_ja_tokenizers

        processor = en_ja_tokenizers.JaMecabProcessor.__new__(en_ja_tokenizers.JaMecabProcessor)
        res = processor.detokenize(["hello", "world"])
        assert "ja_spaced:" in res

