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
import pytest
import torch
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.testing.dummies import dummy_recording
from torch import tensor

from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import PromptFormatterTextProcessing


@pytest.fixture
def tokenizer(capsys, tmp_path_factory):
    TOKENIZER_TRAIN_TEXT = """
    Example system message.
    Example user message.
    Example assistant message.
    TEST
    [INST]
    [/INST]
    <s>
    </s>
    <<SYS>>
    <</SYS>>
    User: Assistant:
    user model
    Instruct Output
    \n\n
    <start_of_turn> <end_of_turn>
    <|
    |>
    <|en|> <|de|> <|fr|> <|es|> <|transcribe|> <|translate|> <|pnc|> <|nopnc|> <|startoftranscript|> <|endoftext|>
    Feel free to add new tokens for your own tests!?
    But know that if you do so, you may need to update the token IDs in the existing tests!
    So, it might be a good idea to create a new tokenizer instead when adding new prompt formats.
    """
    tmpdir = tmp_path_factory.mktemp("bpe_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(TOKENIZER_TRAIN_TEXT)
    with capsys.disabled():
        create_spt_model(
            str(text_path),
            vocab_size=512,
            sample_size=-1,
            do_lower_case=False,
            output_dir=str(tmpdir),
            remove_extra_whitespaces=True,
        )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.fixture
def cuts():
    return CutSet(
        [
            MonoCut(
                id="ex0",
                start=0,
                duration=5.0,
                channel=0,
                supervisions=[
                    SupervisionSegment(
                        id="ex0",
                        recording_id="dummy-recording-0000",
                        start=0,
                        duration=5.0,
                        text="some transcription",
                        language="en",
                    )
                ],
                recording=dummy_recording(0, duration=5.0, with_data=True),
                custom={
                    "context": "non default prompt context",
                    "answer": "some desired answer",
                    "system_prompt": "Please answer the following based on the previous speech feature.",
                },
            ),
        ]
    )


def test_speechllm_dataset(tokenizer, cuts):
    text_processor = PromptFormatterTextProcessing(tokenizer=tokenizer, prompt_format="plain")
    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="do this task",
        tokens_to_generate=0,
        pad_to_max_length=True,
        max_seq_length=64,
    )

    batch = dataset[cuts]

    expected_keys = {
        "sample_ids",
        "audio_signal",
        "audio_signal_length",
        "audio_ratio",
        "metadata",
        "tokens",
        "tokens_length",
        "labels",
        "loss_mask",
        "position_ids",
        "contexts",
        "context_lengths",
        "max_length",
        "answers",
    }
    missing_keys = expected_keys - set(batch)
    unexpected_keys = set(batch) - expected_keys
    assert not missing_keys and not unexpected_keys, f"{missing_keys=} {unexpected_keys=}"

    assert batch["sample_ids"] == ["ex0"]
    assert batch["metadata"] == [{'audio_filepath': 'ex0.wav'}]
    torch.testing.assert_close(batch["audio_ratio"], tensor([1.0]))
    torch.testing.assert_close(batch["max_length"], tensor([64]))

    assert torch.is_tensor(batch["audio_signal"])
    assert torch.is_floating_point(batch["audio_signal"])
    assert batch["audio_signal"].shape == (1, 80000)
    torch.testing.assert_close(batch["audio_signal_length"], tensor([80000], dtype=torch.int32))

    # fmt: off
    expected = tensor([[  1,  78,   9,   1,  64,  80,   5,  75,  15,   6,   1,  12,  24,  14,
               23,   6,   1,  27,  14,   9,   6,  63,   6,  76,  14,  73,   2,   1,
               56, 100,  41,  14,   9,  0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0]])
    torch.testing.assert_close(batch["tokens"], expected)
    torch.testing.assert_close(batch["tokens_length"], tensor([32]))
    assert tokenizer.ids_to_text(expected[0, :33].tolist()) == "non default prompt context some transcription"

    expected = tensor([[1, 78, 9, 1, 64, 80, 5, 75, 15, 6, 1, 12, 24, 14, 23, 6, 1, 27,
             14, 9, 6, 63, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(batch["contexts"], expected)
    torch.testing.assert_close(batch["context_lengths"], tensor([23]))
    assert tokenizer.ids_to_text(expected[0, :23].tolist()) == "non default prompt context"

    expected = tensor([[76, 14, 73, 2, 1, 56, 100, 41, 14, 9, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(batch["answers"], expected)
    assert tokenizer.ids_to_text(expected[0, :10].tolist()) == "some transcription"

    expected = tensor([[78, 9, 1, 64, 80, 5, 75, 15, 6, 1, 12, 24, 14, 23,
             6, 1, 27, 14, 9, 6, 63, 6, 76, 14, 73, 2, 1, 56,
             100, 41, 14, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(batch["labels"], expected)
    assert tokenizer.ids_to_text(expected[0, :32].tolist()) == "non default prompt context some transcription"

    torch.testing.assert_close(
        batch["position_ids"],
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]])
    )

    torch.testing.assert_close(
        batch["loss_mask"],
        tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    )
    # fmt: on


@pytest.fixture
def llama_tokenizer(capsys, tmp_path_factory):
    TOKENIZER_TRAIN_TEXT = """
    a b c d e f g h i j k l m n o p q r s t u v x y z
    A B C D E F G H I J K L M N O P Q R S T U V X Y Z
    [EOG]
    Example system message.
    Example user message.
    Example assistant message.
    TEST
    [INST]
    [/INST]
    <s>
    </s>
    <<SYS>>
    <</SYS>>
    User: Assistant:
    user model
    Instruct Output
    \n\n
    <start_of_turn> <end_of_turn>
    <|
    |>
    <|en|> <|de|> <|fr|> <|es|> <|transcribe|> <|translate|> <|pnc|> <|nopnc|> <|startoftranscript|> <|endoftext|>
    Feel free to add new tokens for your own tests!?
    But know that if you do so, you may need to update the token IDs in the existing tests!
    So, it might be a good idea to create a new tokenizer instead when adding new prompt formats.
    """
    tmpdir = tmp_path_factory.mktemp("bpe_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(TOKENIZER_TRAIN_TEXT)
    with capsys.disabled():
        create_spt_model(
            str(text_path),
            vocab_size=512,
            sample_size=-1,
            do_lower_case=False,
            output_dir=str(tmpdir),
            bos=True,
            eos=True,
            user_defined_symbols=["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "[EOG]"],
            remove_extra_whitespaces=True,
        )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


def test_speechllm_dataset_prompt_template(llama_tokenizer, cuts):
    tokenizer = llama_tokenizer
    text_processor = PromptFormatterTextProcessing(tokenizer=tokenizer, prompt_format="llama2")
    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="do this task",
        tokens_to_generate=0,
        pad_to_max_length=True,
        max_seq_length=128,
    )

    batch = dataset[cuts]
    print(batch)

    expected_keys = {
        "sample_ids",
        "audio_signal",
        "audio_signal_length",
        "audio_ratio",
        "metadata",
        "tokens",
        "tokens_length",
        "labels",
        "loss_mask",
        "position_ids",
        "contexts",
        "context_lengths",
        "max_length",
        "answers",
    }
    missing_keys = expected_keys - set(batch)
    unexpected_keys = set(batch) - expected_keys
    assert not missing_keys and not unexpected_keys, f"{missing_keys=} {unexpected_keys=}"

    assert batch["sample_ids"] == ["ex0"]
    assert batch["metadata"] == [{'audio_filepath': 'ex0.wav'}]
    torch.testing.assert_close(batch["audio_ratio"], tensor([1.0]))
    torch.testing.assert_close(batch["max_length"], tensor([128]))

    assert torch.is_tensor(batch["audio_signal"])
    assert torch.is_floating_point(batch["audio_signal"])
    assert batch["audio_signal"].shape == (1, 80000)
    torch.testing.assert_close(batch["audio_signal_length"], tensor([80000], dtype=torch.int32))

    for k in ("tokens", "contexts", "answers", "labels"):
        print(f"batch['{k}']=", tokenizer.ids_to_text(batch[k][0]))
    # fmt: off
    expected = tensor([[  1,   8,   3,   8,   5,   8, 105,  18,   9,  12,  17,   9,  41,  14,
          17,  22, 125,  43,   9, 117,  19,  18,  18,  79,  48,  15,  92,  12,
          17,   9,  42,   8,  19,  14,  43,   9,  85,  21,   9, 114,  45,  19,
          86,  17,  72,  20,   9,   9,  32,  46, 117,   9, 123,  69,   9,  25,
           8,   6,   8,  93,  14,   8,  74,  88,  12,  86,  18,  13,  85,  21,
          19,  27,  13, 116,  19,  14,  13,  78,  13,   8,   4,  72,  19,  84,
           9,   8,  65, 120,  45,  19,  14,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2]])
    torch.testing.assert_close(batch["tokens"], expected)
    torch.testing.assert_close(batch["tokens_length"], tensor([91]))
    assert tokenizer.ids_to_text(expected[0, :91].tolist()) == "[INST] <<SYS>> Please answer the following based on the previous speech feature. <</SYS>> non default prompt context [/INST] some transcription"

    expected = tensor([[  1,   8,   3,   8,   5,   8, 105,  18,   9,  12,  17,   9,  41,  14,
          17,  22, 125,  43,   9, 117,  19,  18,  18,  79,  48,  15,  92,  12,
          17,   9,  42,   8,  19,  14,  43,   9,  85,  21,   9, 114,  45,  19,
          86,  17,  72,  20,   9,   9,  32,  46, 117,   9, 123,  69,   9,  25,
           8,   6,   8,  93,  14,   8,  74,  88,  12,  86,  18,  13,  85,  21,
          19,  27,  13, 116,  19,  14,  13,  78,  13,   8,   4,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2]])
    torch.testing.assert_close(batch["contexts"], expected)
    torch.testing.assert_close(batch["context_lengths"], tensor([81]))
    assert tokenizer.ids_to_text(expected[0, :81].tolist()) == "[INST] <<SYS>> Please answer the following based on the previous speech feature. <</SYS>> non default prompt context [/INST]"

    expected = tensor([[ 72,  19,  84,   9,   8,  65, 120,  45,  19,  14,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2]])
    torch.testing.assert_close(batch["answers"], expected)
    assert tokenizer.ids_to_text(expected[0, :11].tolist()) == "some transcription"

    expected = tensor([[  8,   3,   8,   5,   8, 105,  18,   9,  12,  17,   9,  41,  14,  17,
          22, 125,  43,   9, 117,  19,  18,  18,  79,  48,  15,  92,  12,  17,
           9,  42,   8,  19,  14,  43,   9,  85,  21,   9, 114,  45,  19,  86,
          17,  72,  20,   9,   9,  32,  46, 117,   9, 123,  69,   9,  25,   8,
           6,   8,  93,  14,   8,  74,  88,  12,  86,  18,  13,  85,  21,  19,
          27,  13, 116,  19,  14,  13,  78,  13,   8,   4,  72,  19,  84,   9,
           8,  65, 120,  45,  19,  14,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
           2]])
    torch.testing.assert_close(batch["labels"], expected)
    assert tokenizer.ids_to_text(expected[0, :90].tolist()) == "[INST] <<SYS>> Please answer the following based on the previous speech feature. <</SYS>> non default prompt context [/INST] some transcription"

    torch.testing.assert_close(
        batch["position_ids"],
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                 126, 127]])
    )

    torch.testing.assert_close(
        batch["loss_mask"],
        tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0.]])
    )
    # fmt: on


def test_speechllm_dataset_tokens_to_generate_increases_seq_len(llama_tokenizer, cuts):
    tokenizer = llama_tokenizer
    text_processor = PromptFormatterTextProcessing(tokenizer=tokenizer, prompt_format="llama2")

    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="do this task",
        tokens_to_generate=0,
        pad_to_max_length=False,
        max_seq_length=512,
    )
    batch = dataset[cuts]
    assert batch["tokens"].shape == (1, 91)
    assert batch["labels"].shape == (1, 91)
    assert batch["contexts"].shape == (1, 81)  # was 92 before padding optimization
    assert batch["answers"].shape == (1, 11)  # was 92 before padding optimization
    assert batch["position_ids"].shape == (1, 92)

    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="do this task",
        tokens_to_generate=256,
        pad_to_max_length=False,
        max_seq_length=512,
    )
    batch = dataset[cuts]
    assert batch["tokens"].shape == (1, 91)
    assert batch["labels"].shape == (1, 91)
    assert batch["contexts"].shape == (1, 337)
    assert batch["answers"].shape == (1, 11)
    assert batch["position_ids"].shape == (1, 92)
