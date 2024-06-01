# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import Mock

import pytest
import sentencepiece as spm
from omegaconf import OmegaConf

from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.common.tokenizers.canary_tokenizer import DEFAULT_TOKENS, CanaryTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model
from nemo.core import Serialization


@pytest.fixture(scope="session")
def special_tokenizer_path(tmp_path_factory) -> str:
    tokens = ["asr", "ast", "en", "de", "fr", "es"]
    tmpdir = tmp_path_factory.mktemp("spl_tokens")
    CanaryTokenizer.build_special_tokenizer(tokens, tmpdir)
    return str(tmpdir)


@pytest.fixture(scope="session")
def lang_tokenizer_path(tmp_path_factory) -> str:
    tmpdir = tmp_path_factory.mktemp("klingon_tokens")
    text_path = tmpdir / "text.txt"
    text_path.write_text("a\nb\nc\nd\n")
    create_spt_model(text_path, vocab_size=8, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir))
    return str(tmpdir)


def test_canary_tokenizer_build_special_tokenizer(tmp_path):
    tokens = ["asr", "ast", "en", "de", "fr", "es"]
    tokenizer = CanaryTokenizer.build_special_tokenizer(tokens, tmp_path)
    expected_tokens = DEFAULT_TOKENS + [f"<|{t}|>" for t in tokens] + ["▁", "<unk>"]
    tokens = []
    for i in range(tokenizer.tokenizer.vocab_size()):
        tokens.append(tokenizer.tokenizer.IdToPiece(i))
    expected_tokens.sort(), tokens.sort()
    print(expected_tokens, tokens)
    assert expected_tokens == tokens


def test_canary_tokenizer_init_from_cfg(special_tokenizer_path, lang_tokenizer_path):
    class DummyModel(ASRBPEMixin, Serialization):
        pass

    model = DummyModel()
    model.register_artifact = Mock(side_effect=lambda self, x: x)
    config = OmegaConf.create(
        {
            "type": "agg",
            "dir": None,
            "langs": {
                "spl_tokens": {"dir": special_tokenizer_path, "type": "bpe"},
                "en": {"dir": lang_tokenizer_path, "type": "bpe"},
            },
            "custom_tokenizer": {
                "_target_": "nemo.collections.common.tokenizers.canary_tokenizer.CanaryTokenizer",
            },
        }
    )
    model._setup_aggregate_tokenizer(config)
    tokenizer = model.tokenizer

    assert isinstance(tokenizer, CanaryTokenizer)
    assert len(tokenizer.tokenizers_dict) == 2
    assert set(tokenizer.tokenizers_dict.keys()) == {"spl_tokens", "en"}

    assert isinstance(tokenizer.tokenizers_dict["spl_tokens"], SentencePieceTokenizer)
    assert tokenizer.tokenizers_dict["spl_tokens"].vocab_size == 14

    assert isinstance(tokenizer.tokenizers_dict["en"], SentencePieceTokenizer)
    assert tokenizer.tokenizers_dict["en"].vocab_size == 6

    assert tokenizer.text_to_ids("<|startoftranscript|><|en|><|asr|><|en|><|pnc|>", lang_id="spl_tokens") == [
        4,
        9,
        7,
        9,
        5,
    ]
    assert tokenizer.text_to_ids("a", lang_id="en") == [14 + 1, 14 + 2]
