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

import string

import pytest

from nemo.collections.common.parts.preprocessing import cleaners
from nemo.collections.common.parts.preprocessing.parsers import ENCharParser

LABELS = [' ', "'"] + list(string.ascii_lowercase)


class TestCleanAbbreviations:
    @pytest.mark.unit
    def test_expanded_version_expands_common_abbreviations(self):
        """`version="expanded"` must still apply the common abbreviations."""
        assert cleaners.clean_abbreviations("mr. smith", version="expanded") == "mister smith"

    @pytest.mark.unit
    def test_expanded_version_expands_the_expanded_only_abbreviations(self):
        """`ltd` exists only in ABBREVIATIONS_EXPANDED, so it pins the extra list to `expanded`."""
        assert cleaners.clean_abbreviations("ltd. co.", version="expanded") == "limited company"

    @pytest.mark.unit
    @pytest.mark.parametrize("version,expected", [(None, "mister smith"), ("fastpitch", "mister smith")])
    def test_other_versions_are_unchanged(self, version, expected):
        """Control: the sibling branches of the same function."""
        assert cleaners.clean_abbreviations("mr. smith", version=version) == expected

    @pytest.mark.unit
    def test_expanded_version_does_not_mutate_module_level_lists(self):
        """`abbbreviations` aliases ABBREVIATIONS_COMMON; an in-place `.extend()` fix would leak."""
        common_before = list(cleaners.ABBREVIATIONS_COMMON)
        fastpitch_before = list(cleaners.ABBREVIATIONS_TTS_FASTPITCH)

        cleaners.clean_abbreviations("mr. smith", version="expanded")

        assert cleaners.ABBREVIATIONS_COMMON == common_before
        assert cleaners.ABBREVIATIONS_TTS_FASTPITCH == fastpitch_before

    @pytest.mark.unit
    def test_expanded_version_does_not_leak_into_later_calls(self):
        """After an `expanded` call, `version=None` must not gain the expanded-only abbreviations."""
        cleaners.clean_abbreviations("ltd. co.", version="expanded")

        assert cleaners.clean_abbreviations("ltd. co.", version=None) == "ltd. company"
        assert cleaners.clean_abbreviations("ltd. co.", version="fastpitch") == "limited company"


class TestENCharParserAbbreviationVersion:
    @pytest.mark.unit
    def test_expanded_parser_does_not_drop_utterances(self):
        """`CharParser._normalize` swallows every exception, so a raise here silently filters samples."""
        parser = ENCharParser(abbreviation_version="expanded", labels=LABELS)
        reference = ENCharParser(abbreviation_version=None, labels=LABELS)

        assert parser("hello") == reference("hello")
        assert parser("mr. smith") == reference("mr. smith")
        assert parser("ltd. co.") == reference("limited company")

    @pytest.mark.unit
    def test_parser_instances_do_not_share_abbreviation_state(self):
        """Two parsers must stay independent regardless of construction order."""
        expanded = ENCharParser(abbreviation_version="expanded", labels=LABELS)
        plain = ENCharParser(abbreviation_version=None, labels=LABELS)

        assert expanded("ltd. co.") == plain("limited company")
        assert plain("ltd. co.") != plain("limited company")
        # and the plain parser is still unaffected after the expanded one has run again
        expanded("ltd. co.")
        assert plain("ltd. co.") != plain("limited company")
