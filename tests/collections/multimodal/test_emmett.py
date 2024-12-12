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
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import PromptFormatterTextProcessing


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        return cuts


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


"""
TEST FOR AUDIO DATALOADING WITH EMMETT
"""


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
                    "context": "<en>",
                    "answer": "some desired answer",
                },
            ),
        ]
    )


@pytest.fixture
def cuts_path(tmp_path_factory, cuts):
    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


def test_audio_example_with_prompt_emmett_t5(cuts_path, tokenizer):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse",
                    "cuts_path": cuts_path,
                },
            ],
            "prompt_format": "t5nmt",
            "force_finite": True,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # First test that sampling is correct and tokenizer + prompt formatter is applied there

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=tokenizer
    )
    batches = [batch for batch in dl]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, CutSet)
    assert len(b) == 1
    ex = b[0]
    assert isinstance(ex, MonoCut)

    assert ex.has_custom("context_ids")
    assert torch.is_tensor(ex.context_ids)
    assert tokenizer.ids_to_text(ex.context_ids) == "<en>"

    assert ex.has_custom("answer_ids")
    assert torch.is_tensor(ex.answer_ids)
    assert tokenizer.ids_to_text(ex.answer_ids) == "some transcription"

    assert ex.has_custom("input_ids")
    assert torch.is_tensor(ex.input_ids)
    assert tokenizer.ids_to_text(ex.input_ids) == "<en> some transcription"

    # Test that speechlm dataset processes the example correctly

    text_processor = PromptFormatterTextProcessing(tokenizer=tokenizer, prompt_format="t5nmt")
    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="<en>",
        tokens_to_generate=0,
        pad_to_max_length=False,
        max_seq_length=64,
    )

    batch = dataset[batches[0]]
    assert tokenizer.ids_to_text(batch["tokens"][0]) == "<en> some transcriptio"
    assert tokenizer.ids_to_text(batch["labels"][0]) == "en> some transcription"
    assert tokenizer.ids_to_text(batch["contexts"][0]) == "<en>"
    assert tokenizer.ids_to_text(batch["answers"][0]) == "some transcription"


"""
TEST FOR TEXT DATALOADING WITH EMMETT
"""


@pytest.fixture
def nmt_paths(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("nmtdata")
    src = tmp_path / "src.txt"
    tgt = tmp_path / "tgt.txt"
    q = tmp_path / "q.txt"
    src.write_text("fake german")
    tgt.write_text("real english")
    q.write_text("<en>")
    return src, tgt, q


def test_text_example_with_prompt_emmett_t5(nmt_paths, tokenizer):
    src, tgt, q = nmt_paths
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": src,
                    "target_paths": tgt,
                    "source_language": "de",
                    "target_language": "en",
                    "questions_path": q,
                    "questions_language": "en",
                },
            ],
            "prompt_format": "t5nmt",
            "force_finite": True,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # First test that sampling is correct and tokenizer + prompt formatter is applied there

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=tokenizer
    )
    batches = [batch for batch in dl]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, CutSet)
    assert len(b) == 1
    ex = b[0]
    assert isinstance(ex, SourceTargetTextExample)

    assert torch.is_tensor(ex.context_ids)
    assert tokenizer.ids_to_text(ex.context_ids) == "<en> fake german"

    assert torch.is_tensor(ex.answer_ids)
    assert tokenizer.ids_to_text(ex.answer_ids) == "real english"

    assert torch.is_tensor(ex.input_ids)
    assert tokenizer.ids_to_text(ex.input_ids) == "<en> fake german real english"

    # Test that speechlm dataset processes the example correctly

    text_processor = PromptFormatterTextProcessing(tokenizer=tokenizer, prompt_format="t5nmt")
    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="<en>",
        tokens_to_generate=0,
        pad_to_max_length=False,
        max_seq_length=64,
    )

    batch = dataset[batches[0]]

    assert tokenizer.ids_to_text(batch["text_input_ids"][0]) == "<en> fake german real english"
    assert tokenizer.ids_to_text(batch["text_context_ids"][0]) == "<en> fake german"
    assert tokenizer.ids_to_text(batch["text_answer_ids"][0]) == "real english"
