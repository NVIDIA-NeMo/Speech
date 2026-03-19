# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

nemo_text_processing = pytest.importorskip("nemo_text_processing", reason="Requires nemo_text_processing to run")

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

from nemo.collections.asr.data.text_to_text import TextToTextDataset, TextToTextItem, TextToTextIterableDataset
from nemo.collections.common import tokenizers

BASE_DIR = Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="module")
def set_multiprocessing_method():
    """
    Try to set 'fork' multiprocessing method to avoid problems with multiprocessing in PyTest on MacOS
    """
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)


@pytest.fixture(scope="module")
def speakers_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("textonly") / "speakers.txt"
    with open(path, "w", encoding="utf-8") as f:
        for speaker in [1, 2, 3]:
            print(f"{speaker}", file=f)
    return path


@pytest.fixture(scope="module")
def textonly_manifest_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("textonly") / "manifest.json"
    texts = [
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "nullam rhoncus sapien eros eu mollis sem euismod non",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            print(json.dumps(dict(text=text, tts_text_normalized=text)), file=f)
    return path


@pytest.fixture(scope="module")
def textonly_unnormalized_manifest_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("textonly") / "manifest_nonorm.json"
    texts = [
        (
            "lorem ipsum dolor sit amet consectetur adipiscing elit",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        ),
        (
            "nullam rhoncus sapien eros eu mollis sem euismod non nineteen",
            "Nullam rhoncus sapien eros, eu mollis sem euismod non 19.",
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for asr_text, tts_text in texts:
            print(json.dumps(dict(text=asr_text, tts_text=tts_text)), file=f)
    return path


@pytest.fixture(scope="module")
def tts_normalizer():
    normalizer = Normalizer(
        lang="en",
        input_case="cased",
        overwrite_cache=True,
        cache_dir=None,
    )
    return normalizer


@pytest.fixture(scope="module")
def asr_tokenizer(test_data_dir):
    tokenizer_path = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
    tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=tokenizer_path)
    return tokenizer


@pytest.fixture(scope="module")
def tts_tokenizer():
    @dataclass
    class G2PConfig:
        _target_: str = "nemo.collections.tts.g2p.models.en_us_arpabet.EnglishG2p"
        phoneme_dict: str = str(BASE_DIR / "scripts/tts_dataset_files/cmudict-0.7b_nv22.10")
        heteronyms: str = str(BASE_DIR / "scripts/tts_dataset_files/heteronyms-052722")
        phoneme_probability: float = 0.5

    @dataclass
    class TextTokenizerCfg:
        _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer"
        punct: bool = True
        stresses: bool = True
        chars: bool = True
        apostrophe: bool = True
        pad_with_space: bool = True
        add_blank_at: bool = True
        g2p: G2PConfig = field(default_factory=lambda: G2PConfig())

    config = OmegaConf.create(OmegaConf.to_yaml(TextTokenizerCfg()))
    return instantiate(config)


class TestTextToTextDataset:
    @pytest.mark.unit
    @pytest.mark.parametrize("tokenizer_workers", [1, 2])
    def test_text_to_text_dataset(
        self,
        textonly_manifest_path,
        tokenizer_workers,
        speakers_path,
        asr_tokenizer,
        tts_tokenizer,
        tts_normalizer,
        set_multiprocessing_method,
    ):
        """
        Test map-style text-to-text dataset with ASR and TTS tokenizers with normalized text
        """
        dataset = TextToTextDataset(
            manifest_filepath=textonly_manifest_path,
            speakers_filepath=speakers_path,
            asr_tokenizer=asr_tokenizer,
            asr_use_start_end_token=False,
            tts_parser=tts_tokenizer,
            tts_text_pad_id=0,
            tts_text_normalizer=tts_normalizer,
            tts_text_normalizer_call_kwargs=dict(),
            tokenizer_workers=tokenizer_workers,
        )
        assert len(dataset) == 2
        item = dataset[0]
        assert isinstance(item, TextToTextItem)

    @pytest.mark.unit
    def test_text_to_text_dataset_unnormalized(
        self, textonly_unnormalized_manifest_path, speakers_path, asr_tokenizer, tts_tokenizer, tts_normalizer
    ):
        """
        Test TextToTextDataset with ASR and TTS tokenizers with non-normalized text
        """
        dataset = TextToTextDataset(
            manifest_filepath=textonly_unnormalized_manifest_path,
            speakers_filepath=speakers_path,
            asr_tokenizer=asr_tokenizer,
            asr_use_start_end_token=False,
            tts_parser=tts_tokenizer,
            tts_text_pad_id=0,
            tts_text_normalizer=tts_normalizer,
            tts_text_normalizer_call_kwargs=dict(),
        )
        assert len(dataset) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize("tokenizer_workers", [1, 2])
    def test_text_to_text_iterable_dataset(
        self,
        textonly_manifest_path,
        tokenizer_workers,
        speakers_path,
        asr_tokenizer,
        tts_tokenizer,
        tts_normalizer,
        set_multiprocessing_method,
    ):
        """
        Test iterable text-to-text dataset with ASR and TTS tokenizers with normalized text
        """
        dataset = TextToTextIterableDataset(
            manifest_filepath=textonly_manifest_path,
            speakers_filepath=speakers_path,
            asr_tokenizer=asr_tokenizer,
            asr_use_start_end_token=False,
            tts_parser=tts_tokenizer,
            tts_text_pad_id=0,
            tts_text_normalizer=tts_normalizer,
            tts_text_normalizer_call_kwargs=dict(),
            tokenizer_workers=tokenizer_workers,
        )
        assert len(dataset) == 2
        item = next(iter(dataset))
        assert isinstance(item, TextToTextItem)
