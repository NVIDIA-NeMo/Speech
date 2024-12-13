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
import lhotse.serialization
import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.cut import Cut
from lhotse.testing.dummies import dummy_cut

from nemo.collections.common.data.lhotse import (
    NeMoSFTExample,
    SourceTargetTextExample,
    get_lhotse_dataloader_from_config,
)
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model


@pytest.fixture
def tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("tok")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(chr(i) for i in range(256)))
    create_spt_model(
        text_path,
        vocab_size=512,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        user_defined_symbols=[
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>",
            "[audio]",
            "<end_of_turn>",
            "<start_of_turn>",
        ],
        remove_extra_whitespaces=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.fixture
def cuts_path(tmp_path_factory):
    tmp_path = tmp_path_factory.getbasetemp() / "cuts.jsonl"
    c = dummy_cut(0, duration=1.0, supervisions=[SupervisionSegment("", "", 0, 1.0, text="dummy text")])
    c.context = "dummy context"
    CutSet([c]).to_file(tmp_path)
    return tmp_path


@pytest.fixture
def src_tgt_example(tmp_path_factory):
    d = tmp_path_factory.mktemp("src_tgt")
    (d / "src.txt").write_text("an example")
    (d / "tgt.txt").write_text("elpmaxe na")
    return (d / "src.txt"), (d / "tgt.txt")


@pytest.fixture
def nemo_sft_example(tmp_path_factory):
    tmp_path = tmp_path_factory.getbasetemp() / "nemo_sft.jsonl"
    lhotse.serialization.save_to_jsonl(
        [
            {
                "system": "",
                "mask": "User",
                "dataset": "",
                "conversations": [
                    {
                        "from": "User",
                        "value": "Hi, how are you?",
                    },
                    {
                        "from": "Assistant",
                        "value": "Good day, I'm a useful assistant.",
                    },
                ],
            }
        ],
        tmp_path,
    )
    return tmp_path


class Identity:
    def __getitem__(self, item):
        return item


def test_prompt_format_cut(cuts_path, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "batch_size": 1,
            "prompt_format": "llama2",
            "min_duration": 0,
            "max_duration": 10,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )

    batch = next(iter(dl))
    ex = batch[0]
    assert isinstance(ex, Cut)
    assert tokenizer.ids_to_text(ex.input_ids) == "[INST] dummy context [/INST] dummy text"
    assert tokenizer.ids_to_text(ex.context_ids) == "[INST] dummy context [/INST]"
    assert tokenizer.ids_to_text(ex.answer_ids) == "dummy text"


def test_prompt_format_cut_filtered_out(cuts_path, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "batch_size": 1,
            "prompt_format": "llama2",
            "min_duration": 0,
            "max_duration": 0.5,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )
    with pytest.raises(StopIteration):
        next(iter(dl))


def test_prompt_format_cut_max_tokens_has_no_filtering_effect(cuts_path, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "batch_size": 1,
            "prompt_format": "llama2",
            "use_multimodal_dataloading": True,
            "token_equivalent_duration": 0.1,
            "min_tokens": 1,
            "max_tokens": 2,
            "use_total_length": True,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )

    batch = next(iter(dl))
    ex = batch[0]
    assert isinstance(ex, Cut)


def test_prompt_format_src_tgt(src_tgt_example, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "input_cfg": [
                {"type": "txt_pair", "source_paths": src_tgt_example[0], "target_paths": src_tgt_example[1]}
            ],
            "batch_size": 1,
            "force_finite": True,
            "prompt_format": "llama2",
            "use_multimodal_dataloading": True,
            "min_tokens": 1,
            "max_tokens": 50,
            "use_total_length": True,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )

    batch = next(iter(dl))
    ex = batch[0]
    assert isinstance(ex, SourceTargetTextExample)
    assert tokenizer.ids_to_text(ex.input_ids) == "[INST] an example [/INST] elpmaxe na"
    assert tokenizer.ids_to_text(ex.context_ids) == "[INST] an example [/INST]"
    assert tokenizer.ids_to_text(ex.answer_ids) == "elpmaxe na"


def test_prompt_format_src_tgt_filtered_out(src_tgt_example, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "input_cfg": [
                {"type": "txt_pair", "source_paths": src_tgt_example[0], "target_paths": src_tgt_example[1]}
            ],
            "batch_size": 1,
            "force_finite": True,
            "prompt_format": "llama2",
            "use_multimodal_dataloading": True,
            "min_tokens": 1,
            "max_tokens": 10,
            "use_total_length": True,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )
    with pytest.raises(StopIteration):
        batch = next(iter(dl))


def test_prompt_format_src_tgt_2d(src_tgt_example, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": src_tgt_example[0],
                    "target_paths": src_tgt_example[1],
                    "target_language": "reversed",
                }
            ],
            "batch_size": 1,
            "force_finite": True,
            "prompt_format": "t5nmt",
            "use_multimodal_dataloading": True,
            "min_tokens": 1,
            "max_tokens": 50,
            "use_total_length": False,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )

    batch = next(iter(dl))
    ex = batch[0]
    assert isinstance(ex, SourceTargetTextExample)
    assert tokenizer.ids_to_text(ex.input_ids) == "<reversed> an example elpmaxe na"
    assert tokenizer.ids_to_text(ex.context_ids) == "<reversed> an example"
    assert tokenizer.ids_to_text(ex.answer_ids) == "elpmaxe na"


def test_prompt_format_nemo_sft(nemo_sft_example, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "input_cfg": [{"type": "nemo_sft_jsonl", "paths": nemo_sft_example}],
            "batch_size": 1,
            "force_finite": True,
            "prompt_format": "llama2",
            "use_multimodal_dataloading": True,
            "min_tokens": 1,
            "max_tokens": 100,
            "use_total_length": True,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )

    batch = next(iter(dl))
    ex = batch[0]
    assert isinstance(ex, NeMoSFTExample)
    assert tokenizer.ids_to_text(ex.input_ids) == "[INST] Hi, how are you? [/INST] Good day, I'm a useful assistant."
    assert tokenizer.ids_to_text(ex.context_ids) == "[INST] Hi, how are you? [/INST]"
    assert tokenizer.ids_to_text(ex.answer_ids) == "Good day, I'm a useful assistant."


def test_prompt_format_nemo_sft_filtered_out(nemo_sft_example, tokenizer):
    dl = get_lhotse_dataloader_from_config(
        {
            "input_cfg": [{"type": "nemo_sft_jsonl", "paths": nemo_sft_example}],
            "batch_size": 1,
            "force_finite": True,
            "prompt_format": "llama2",
            "use_multimodal_dataloading": True,
            "min_tokens": 1,
            "max_tokens": 5,
            "use_total_length": True,
        },
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=tokenizer,
    )
    with pytest.raises(StopIteration):
        batch = next(iter(dl))
