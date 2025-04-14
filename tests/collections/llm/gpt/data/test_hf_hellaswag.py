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


from unittest.mock import MagicMock, patch

import pytest

from nemo.collections.llm.gpt.data.hf_dataset import HellaSwagHFDataModule


@pytest.fixture
def mock_tokenizer():
    """
    Provides a mock tokenizer that simulates huggingface's tokenizer behavior.
    """
    tokenizer = MagicMock()
    tokenizer.tokenizer = tokenizer
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|endoftext|>"

    def tokenizer_call(text, max_length, truncation=True):
        return {
            "input_ids": [[101, 102, 103]],
            "attention_mask": [[1, 1, 1]],
        }

    # Replace __call__ with a MagicMock and set the side_effect
    tokenizer.__call__ = MagicMock(side_effect=tokenizer_call)
    tokenizer.text_to_ids = MagicMock(side_effect=tokenizer_call)
    return tokenizer


@pytest.fixture
def sample_doc():
    """
    Provides a sample 'doc' from the HellaSwag dataset structure.
    """
    return {
        "ctx_a": "First part of the context.",
        "ctx_b": "continuation of the context.",
        "endings": ["Option 1 ending.", "Option 2 ending."],
        "label": 1,
        "activity_label": "SampleActivity",
    }


@pytest.mark.parametrize(
    "input_text, expected",
    [
        (" [title]  Something [extra] text ", " Something text"),
        ("Some text [title] [more]", "Some text. "),
        ("   [Foo]    [title]", "  . "),
    ],
)
def test_preprocess(input_text, expected):
    """
    Tests the static preprocess method. Ensures bracketed text and known artifacts are removed.
    """
    out = HellaSwagHFDataModule.preprocess(input_text)
    assert out == expected


def test_process_doc(sample_doc):
    """
    Tests the static process_doc method to ensure it creates the expected fields.
    """
    out_doc = HellaSwagHFDataModule.process_doc(sample_doc)
    assert "query" in out_doc
    assert "choices" in out_doc
    assert "gold" in out_doc
    assert "text" in out_doc

    # Check some details
    assert out_doc["gold"] == sample_doc["label"]
    assert len(out_doc["choices"]) == len(sample_doc["endings"])
    assert out_doc["query"].startswith("SampleActivity: ")


@patch("nemo.collections.llm.gpt.data.hf_dataset.load_dataset")
def test_preprocess_dataset(mock_load_dataset, mock_tokenizer):
    """
    Tests the static preprocess_dataset method by mocking out load_dataset
    and ensuring mapping/shuffling are correctly applied.
    """
    # Create a mock dataset object
    mock_data = MagicMock()
    # mock_data.map returns itself to allow chaining
    mock_data.map.return_value = mock_data
    # mock_data.shuffle returns itself to allow chaining
    mock_data.shuffle.return_value = mock_data

    # mock_load_dataset returns a dict with "train": mock_data
    mock_load_dataset.return_value = {"train": mock_data}

    # Call the preprocess_dataset function
    processed_data = HellaSwagHFDataModule.preprocess_dataset(
        tokenizer=mock_tokenizer,
        max_length=128,
        dataset=mock_data,
    )

    # Assertions
    mock_data.map.assert_called()
