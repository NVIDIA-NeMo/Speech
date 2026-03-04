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
import os
import random
from itertools import islice
from pathlib import Path

import lhotse
import numpy as np
import pytest
import torch
from lhotse import CutSet, SupervisionSegment, compute_num_samples
from lhotse.shar import JsonlShardWriter
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.indexed_adapters import (
    IndexedTarSampleReader,
    LazyShuffledRange,
    create_index,
    create_tar_index,
)
from nemo.collections.common.data.lhotse.sampling import (
    DurationFilter,
    MultimodalFixedBucketBatchSizeConstraint2D,
    MultimodalSamplingConstraint,
)
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    NeMoMultimodalConversation,
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationShareGPTJsonlAdapter,
    NeMoMultimodalConversationShareGPTWebdatasetAdapter,
    NeMoMultimodalConversationTarWriter,
    TextTurn,
)
from nemo.collections.common.prompts import Llama2PromptFormatter
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> lhotse.CutSet:
        return cuts


@pytest.fixture(scope="session")
def multimodal_conversations_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("text_data")
    en_path = tmp_path / "manifest.json"
    data = [
        {
            "id": "convo_1",
            "conversations": [
                {
                    "value": "Can you help summarize the following?",
                    "from": "User",
                    "type": "text",
                },
                {
                    "value": "123.wav",
                    "from": "User",
                    "type": "audio",
                    "duration": 5.73,
                },
                {
                    "value": "I'm glad to assist you with your request. Here's a summary:",
                    "from": "Assistant",
                    "type": "text",
                },
                {
                    "value": "123_answer.wav",
                    "from": "Assistant",
                    "type": "audio",
                    "duration": 7.11,
                },
                {
                    "value": "Can you further shorten it?",
                    "from": "User",
                    "type": "text",
                },
                {
                    "value": "Of course!",
                    "from": "Assistant",
                    "type": "text",
                },
                {"value": "123_answer.wav", "from": "Assistant", "type": "audio", "offset": 7.11},
            ],
        }
    ]
    lhotse.serialization.save_to_jsonl(data, en_path)
    dummy_recording(0, 5.73, with_data=True).to_cut().save_audio(tmp_path / "123.wav")
    dummy_recording(0, 12.11, with_data=True).to_cut().save_audio(tmp_path / "123_answer.wav")
    return en_path


@pytest.fixture(scope="session")
def tarred_multimodal_conversations_path(multimodal_conversations_path, tmp_path_factory):
    (conversation,) = list(NeMoMultimodalConversationJsonlAdapter(multimodal_conversations_path, "[audio]"))
    tar_dir = tmp_path_factory.mktemp("multi_convo_tarred")
    with NeMoMultimodalConversationTarWriter(tar_dir, shard_size=5) as writer:
        for i in range(10):
            conversation.id = f'convo-{i}'
            writer.write(conversation)
    return str(tar_dir / "manifest_{0..1}.jsonl"), str(tar_dir / "audio_{0..1}.tar")


def test_multimodal_conversation_input(multimodal_conversations_path):

    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": multimodal_conversations_path,
                    "audio_locator_tag": "[audio]",
                },
            ],
            "force_finite": True,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches = [batch for batch in dl]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b) == 1
    ex = b[0]
    assert isinstance(ex, NeMoMultimodalConversation)
    assert ex.id == "convo_1"
    assert len(ex.turns) == 7
    t = ex.turns[0]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "Can you help summarize the following?"
    t = ex.turns[1]
    assert isinstance(t, AudioTurn)
    assert t.role == "user"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 5.73
    assert t.cut.load_audio().shape == (1, 91680)
    t = ex.turns[2]
    assert isinstance(t, TextTurn)
    assert t.role == "assistant"
    assert t.value == "I'm glad to assist you with your request. Here's a summary:"
    t = ex.turns[3]
    assert isinstance(t, AudioTurn)
    assert t.role == "assistant"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 7.11
    assert t.cut.start == 0.0
    assert t.cut.load_audio().shape == (1, 113760)
    t = ex.turns[4]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "Can you further shorten it?"
    t = ex.turns[5]
    assert isinstance(t, TextTurn)
    assert t.role == "assistant"
    assert t.value == "Of course!"
    t = ex.turns[6]
    assert isinstance(t, AudioTurn)
    assert t.role == "assistant"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 5.0
    assert t.cut.start == 7.11
    assert t.cut.load_audio().shape == (1, 80000)


@pytest.fixture(scope="session")
def sharegpt_conversations_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sharegpt_data")
    en_path = tmp_path / "sharegpt_manifest.json"
    data = [
        {
            "id": "sharegpt_convo_1",
            "sound": "audio_123.wav",
            "conversations": [
                {
                    "from": "human",
                    "value": "Please analyze the following audio <sound> and tell me what you hear.",
                },
                {
                    "from": "gpt",
                    "value": "Based on the audio analysis, I can hear various sounds including...",
                },
            ],
            "ori_sound": "/original/path/audio_123.wav",
        },
        {
            "id": "sharegpt_convo_2",
            "sound": "speech_456.wav",
            "conversations": [
                {
                    "from": "human",
                    "value": "Transcribe this speech: <speech>",
                    "duration": 1,
                },
                {
                    "from": "gpt",
                    "value": "The transcription is: Hello, how are you today?",
                },
                {"from": "human", "value": "And this one: <speech>", "offset": 1},
            ],
        },
    ]
    lhotse.serialization.save_to_jsonl(data, en_path)
    # Create dummy audio files
    dummy_recording(0, 5.73, with_data=True).to_cut().save_audio(tmp_path / "audio_123.wav")
    dummy_recording(1, 3.45, with_data=True).to_cut().save_audio(tmp_path / "speech_456.wav")
    return en_path


def test_multimodal_conversation_input_sharegpt(sharegpt_conversations_path):
    """Test ShareGPT format conversation input with audio placeholders."""

    # Test the ShareGPT adapter directly since it's not integrated into the NeMo dataloader registry
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=sharegpt_conversations_path,
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>", "<speech>"],
    )

    conversations = list(adapter)
    assert len(conversations) == 2

    # Test first conversation with <sound> placeholder
    ex1 = conversations[0]
    assert isinstance(ex1, NeMoMultimodalConversation)
    assert ex1.id == "sharegpt_convo_1"
    assert len(ex1.turns) == 4  # text + audio + text + assistant response

    # First turn: text before <sound>
    t = ex1.turns[0]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "Please analyze the following audio"

    # Second turn: audio from <sound> placeholder
    t = ex1.turns[1]
    assert isinstance(t, AudioTurn)
    assert t.role == "user"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 5.73
    assert t.cut.start == 0
    assert t.cut.load_audio().shape == (1, 91680)

    # Third turn: text after <sound>
    t = ex1.turns[2]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "and tell me what you hear."

    # Fourth turn: GPT response
    t = ex1.turns[3]
    assert isinstance(t, TextTurn)
    assert t.role == "assistant"
    assert t.value == "Based on the audio analysis, I can hear various sounds including..."

    # Test second conversation with <speech> placeholder
    ex2 = conversations[1]
    assert isinstance(ex2, NeMoMultimodalConversation)
    assert ex2.id == "sharegpt_convo_2"
    assert len(ex2.turns) == 5  # text + audio + text + text + audio

    # First turn: text before <speech>
    t = ex2.turns[0]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "Transcribe this speech:"

    # Second turn: audio from <speech> placeholder
    t = ex2.turns[1]
    assert isinstance(t, AudioTurn)
    assert t.role == "user"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 1
    assert t.cut.start == 0
    assert t.cut.load_audio().shape == (1, 16000)

    # Third turn: GPT response
    t = ex2.turns[2]
    assert isinstance(t, TextTurn)
    assert t.role == "assistant"
    assert t.value == "The transcription is: Hello, how are you today?"

    # Fourth turn: text before <speech>
    t = ex2.turns[3]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "And this one:"

    # Fifth turn: audio from <speech> placeholder
    t = ex2.turns[4]
    assert isinstance(t, AudioTurn)
    assert t.role == "user"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 2.45
    assert t.cut.start == 1
    assert t.cut.load_audio().shape == (1, 39200)


@pytest.fixture
def tokenizer(tmp_path_factory, multimodal_conversations_path):
    tmpdir = tmp_path_factory.mktemp("multi_convo_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(
        "\n".join(
            turn["value"]
            for item in lhotse.serialization.load_jsonl(multimodal_conversations_path)
            for turn in item["conversations"]
        )
    )
    create_spt_model(
        text_path,
        vocab_size=128,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        user_defined_symbols=["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "[audio]"],
        remove_extra_whitespaces=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


def test_multimodal_conversation_input_with_prompt(multimodal_conversations_path, tokenizer):

    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": multimodal_conversations_path,
                    "audio_locator_tag": "[audio]",
                },
            ],
            "prompt_format": "llama2",
            "force_finite": True,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=tokenizer
    )
    batches = [batch for batch in dl]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b) == 1
    ex = b[0]
    assert isinstance(ex, NeMoMultimodalConversation)
    assert torch.is_tensor(ex.input_ids)
    assert ex.input_ids.shape == (107,)
    assert (
        tokenizer.ids_to_text(ex.input_ids)
        == "[INST] Can you help summarize the following? [audio] [/INST] I'm glad to assist you with your request. Here's a summary: [audio] [INST] Can you further shorten it? [/INST] Of course! [audio]"
    )

    assert torch.is_tensor(ex.context_ids)
    assert ex.context_ids.shape == (95,)
    assert (
        tokenizer.ids_to_text(ex.context_ids)
        == "[INST] Can you help summarize the following? [audio] [/INST] I'm glad to assist you with your request. Here's a summary: [audio] [INST] Can you further shorten it? [/INST]"
    )

    assert torch.is_tensor(ex.answer_ids)
    assert ex.answer_ids.shape == (12,)
    assert tokenizer.ids_to_text(ex.answer_ids) == "Of course! [audio]"

    assert torch.is_tensor(ex.mask)
    assert ex.mask.shape == (107,)
    assert (ex.mask[:30] == False).all()  # user turn
    assert (ex.mask[30:72] == True).all()  # assistant turn
    assert (ex.mask[72:95] == False).all()  # user turn
    assert (ex.mask[95:] == True).all()  # assistant turn


def test_text_only_conversation_length_measurement(tokenizer):
    convo = NeMoMultimodalConversation(
        id="textonly-1",
        turns=[
            TextTurn("hello", "user"),
            TextTurn("hi", "assistant"),
        ],
    )
    convo = convo.apply_prompt_format(Llama2PromptFormatter(tokenizer))
    assert tokenizer.ids_to_text(convo.input_ids) == "[INST] hello [/INST] hi"
    assert tokenizer.ids_to_text(convo.context_ids) == "[INST] hello [/INST]"
    assert tokenizer.ids_to_text(convo.answer_ids) == "hi"

    assert convo.input_length == len(convo.context_ids) == 10
    assert convo.output_length == len(convo.answer_ids) == 4
    assert convo.total_length == len(convo.input_ids) == 14

    constr = MultimodalSamplingConstraint(measure_total_length=False)
    assert constr.measure_length(convo) == 10

    constr = MultimodalSamplingConstraint(measure_total_length=True)
    assert constr.measure_length(convo) == 14

    constr = MultimodalFixedBucketBatchSizeConstraint2D(
        max_seq_len_buckets=[5, 10, 15], batch_sizes=[3, 2, 1], measure_total_length=True
    )
    assert constr.measure_length(convo) == 14
    assert constr.select_bucket(constr.max_seq_len_buckets, convo) == 2

    constr = MultimodalFixedBucketBatchSizeConstraint2D(
        max_seq_len_buckets=[(5, 2), (5, 5), (15, 3), (15, 6), (15, 10)],
        batch_sizes=[5, 4, 3, 2, 1],
        measure_total_length=False,
    )
    assert constr.measure_length(convo) == (10, 4)
    assert constr.select_bucket(constr.max_seq_len_buckets, convo) == 3


def test_audio_only_conversation_length_measurement(tokenizer, tmp_path_factory):
    audio_dir = tmp_path_factory.mktemp("audio")
    c1 = dummy_recording(0, duration=7.16, with_data=True).to_cut().save_audio(audio_dir / "1.wav")
    c2 = dummy_recording(1, duration=15.96, with_data=True).to_cut().save_audio(audio_dir / "2.wav")
    convo = NeMoMultimodalConversation(
        id="audioonly-1",
        turns=[
            AudioTurn(c1, "user", "[audio]"),
            AudioTurn(c2, "assistant", "[audio]"),
        ],
        token_equivalent_duration=0.1,  # 10ms frame_shift * 10x subsampling for easy testing
    )
    convo = convo.apply_prompt_format(Llama2PromptFormatter(tokenizer))
    assert tokenizer.ids_to_text(convo.input_ids) == "[INST] [audio] [/INST] [audio]"
    assert tokenizer.ids_to_text(convo.context_ids) == "[INST] [audio] [/INST]"
    assert tokenizer.ids_to_text(convo.answer_ids) == "[audio]"

    # NOTE: Unlike text-only, len(context_ids) != convo.input_length! The same is true for answer and input ids.
    # 7.16s with 100ms frame is 72 tokens, we have 7 context tokens, but replace 1 audio locator tag.
    assert len(convo.context_ids) == 7
    assert convo.input_length == 78

    # 15.96s with 100ms frame is 160 tokens, we have 3 answer tokens, but replace 1 audio locator tag.
    assert len(convo.answer_ids) == 3
    assert convo.output_length == 162

    assert len(convo.input_ids) == 10
    assert convo.total_length == 162 + 78

    constr = MultimodalSamplingConstraint(measure_total_length=False)
    assert constr.measure_length(convo) == 78

    constr = MultimodalSamplingConstraint(measure_total_length=True)
    assert constr.measure_length(convo) == 162 + 78

    constr = MultimodalFixedBucketBatchSizeConstraint2D(
        max_seq_len_buckets=[100, 200, 300, 400], batch_sizes=[3, 2, 1, 1], measure_total_length=True
    )
    assert constr.measure_length(convo) == 162 + 78
    assert constr.select_bucket(constr.max_seq_len_buckets, convo) == 2

    constr = MultimodalFixedBucketBatchSizeConstraint2D(
        max_seq_len_buckets=[
            (50, 50),
            (50, 100),
            (50, 200),
            (100, 50),
            (100, 150),
            (100, 200),
            (100, 300),
            (400, 400),
        ],
        batch_sizes=[8, 7, 6, 5, 4, 3, 2, 1],
        measure_total_length=False,
    )
    assert constr.measure_length(convo) == (78, 162)
    assert constr.select_bucket(constr.max_seq_len_buckets, convo) == 5


def test_multimodal_conversation_length_measurement(tokenizer, tmp_path_factory):
    audio_dir = tmp_path_factory.mktemp("audio")
    c1 = dummy_recording(0, duration=7.16, with_data=True).to_cut().save_audio(audio_dir / "1.wav")
    c2 = dummy_recording(1, duration=15.96, with_data=True).to_cut().save_audio(audio_dir / "2.wav")
    convo = NeMoMultimodalConversation(
        id="multimodal-1",
        turns=[
            TextTurn("listen to this and tell me your opinion", "user"),
            AudioTurn(c1, "user", "[audio]"),
            TextTurn("its fine", "assistant"),
            TextTurn("remove the noise", "user"),
            TextTurn("sure", "assistant"),
            AudioTurn(c2, "assistant", "[audio]"),
        ],
        token_equivalent_duration=0.1,  # 10ms frame_shift * 10x subsampling for easy testing
    )
    convo = convo.apply_prompt_format(Llama2PromptFormatter(tokenizer))
    print(convo)
    assert (
        tokenizer.ids_to_text(convo.input_ids)
        == "[INST] listen to this and tell me your opinion [audio] [/INST] its fine [INST] remove the noise [/INST] sure [audio]"
    )
    assert (
        tokenizer.ids_to_text(convo.context_ids)
        == "[INST] listen to this and tell me your opinion [audio] [/INST] its fine [INST] remove the noise [/INST]"
    )
    assert tokenizer.ids_to_text(convo.answer_ids) == "sure [audio]"

    assert len(convo.context_ids) == 66
    assert convo.input_length == 66 + 72 - 1 == 137

    # 15.96s with 100ms frame is 160 tokens, we have 3 answer tokens, but replace 1 audio locator tag.
    assert len(convo.answer_ids) == 7
    assert convo.output_length == 7 + 160 - 1 == 166

    assert len(convo.input_ids) == 73
    assert convo.total_length == 137 + 166 == 303

    constr = MultimodalSamplingConstraint(measure_total_length=False)
    assert constr.measure_length(convo) == 137

    constr = MultimodalSamplingConstraint(measure_total_length=True)
    assert constr.measure_length(convo) == 303

    constr = MultimodalFixedBucketBatchSizeConstraint2D(
        max_seq_len_buckets=[100, 200, 300, 400], batch_sizes=[3, 2, 1, 1], measure_total_length=True
    )
    assert constr.measure_length(convo) == 303
    assert constr.select_bucket(constr.max_seq_len_buckets, convo) == 3

    constr = MultimodalFixedBucketBatchSizeConstraint2D(
        max_seq_len_buckets=[
            (50, 50),
            (50, 100),
            (50, 200),
            (100, 50),
            (100, 150),
            (100, 200),
            (100, 300),
            (400, 400),
        ],
        batch_sizes=[8, 7, 6, 5, 4, 3, 2, 1],
        measure_total_length=False,
    )
    assert constr.measure_length(convo) == (137, 166)
    assert constr.select_bucket(constr.max_seq_len_buckets, convo) == 7


def test_multimodal_conversation_tarred_format(multimodal_conversations_path, tmp_path_factory):
    (conversation,) = list(NeMoMultimodalConversationJsonlAdapter(multimodal_conversations_path, "[audio]"))
    tar_dir = tmp_path_factory.mktemp("multi_convo_tarred")
    with NeMoMultimodalConversationTarWriter(tar_dir) as writer:
        writer.write(conversation)

    (restored_conversation,) = list(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=tar_dir / "manifest_0.jsonl",
            audio_locator_tag="[audio]",
            tarred_audio_filepaths=tar_dir / "audio_0.tar",
        )
    )
    assert conversation.id == restored_conversation.id
    assert len(conversation.turns) == len(restored_conversation.turns)
    for lhs, rhs in zip(conversation.turns, restored_conversation.turns):
        assert type(lhs) == type(rhs)
        assert lhs.role == lhs.role
        if isinstance(lhs, TextTurn):
            assert lhs.value == rhs.value
        else:
            assert lhs.audio_locator_tag == rhs.audio_locator_tag
            assert lhs.cut.id == rhs.cut.id
            assert lhs.cut.duration == rhs.cut.duration
            assert lhs.cut.recording.id == rhs.cut.recording.id
            np.testing.assert_allclose(lhs.cut.load_audio(), rhs.cut.load_audio())


def test_multimodal_conversation_tarred_format_sharding_works(multimodal_conversations_path, tmp_path_factory):
    (conversation,) = list(NeMoMultimodalConversationJsonlAdapter(multimodal_conversations_path, "[audio]"))
    tar_dir = tmp_path_factory.mktemp("multi_convo_tarred")
    with NeMoMultimodalConversationTarWriter(tar_dir, shard_size=10) as writer:
        for i in range(30):
            writer.write(conversation)

    loader = NeMoMultimodalConversationJsonlAdapter(
        manifest_filepath=tar_dir / "manifest_{0..2}.jsonl",
        audio_locator_tag="[audio]",
        tarred_audio_filepaths=tar_dir / "audio_{0..2}.tar",
    )
    restored = list(loader)
    assert len(restored) == 30
    assert all(c == restored[0] for c in restored[1:])


def test_multimodal_conversation_duration_filter():
    fltr = DurationFilter(d_min=1.0, d_max=5.0)

    # Passthrough for text-only.
    conv_text_only = NeMoMultimodalConversation(
        id="text",
        turns=[
            TextTurn("abc", role="user"),
            TextTurn("def", role="assistant"),
        ],
    )
    assert fltr(conv_text_only) is True

    # 1 <= 3s <= 5 -> OK
    conv_3s = NeMoMultimodalConversation(
        "audio-3s",
        turns=[
            AudioTurn(dummy_cut(0, duration=3.0), role="user", audio_locator_tag="<|audio|>"),
            TextTurn("def", role="assistant"),
        ],
    )
    assert fltr(conv_3s) is True

    # 1 <= 0.5s <= 5 -> reject
    conv_05s = NeMoMultimodalConversation(
        "audio-05s",
        turns=[
            AudioTurn(dummy_cut(0, duration=0.5), role="user", audio_locator_tag="<|audio|>"),
            TextTurn("def", role="assistant"),
        ],
    )
    assert fltr(conv_05s) is False

    # 1 <= 3 + 4 <= 5 -> reject
    conv_s2s_7s = NeMoMultimodalConversation(
        "audio-audio-7s",
        turns=[
            AudioTurn(dummy_cut(0, duration=3.0), role="user", audio_locator_tag="<|audio|>"),
            AudioTurn(dummy_cut(0, duration=4.0), role="assistant", audio_locator_tag="<|audio|>"),
        ],
    )
    assert fltr(conv_s2s_7s) is False


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """3 utterances of lengths 1s, 2s, and 3s, with different context/system_prompt, as a Lhotse CutSet."""
    cuts = CutSet(
        [
            dummy_cut(
                0,
                duration=1.0,
                supervisions=[SupervisionSegment("e1", "e1", 0.0, 1.0, text="transcript")],
                with_data=True,
            ),
            dummy_cut(
                1,
                duration=2.0,
                recording_duration=2.0,
                supervisions=[SupervisionSegment("e2", "e2", 0.0, 2.0, text="context and transcript")],
                with_data=True,
            ),
            dummy_cut(
                2,
                duration=3.0,
                recording_duration=3.0,
                supervisions=[SupervisionSegment("e3", "e3", 0.0, 2.0, text="system context and transcript")],
                with_data=True,
            ),
        ]
    )
    cuts[1].context = "some prompt"
    cuts[2].context = "other prompt"
    cuts[2].system_prompt = "system prompt"

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).drop_in_memory_data().to_file(p)
    return p


def test_cut_to_conversation_conversion(cutset_path, tokenizer):
    cuts = CutSet.from_file(cutset_path)
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_as_conversation",
                    "cuts_path": cutset_path,
                    "audio_locator_tag": "[audio]",
                    "tags": {"test_key": "test_value"},
                },
            ],
            "token_equivalent_duration": 0.08,
            "prompt_format": "llama3",
            "force_finite": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=tokenizer
    )
    batches = [batch for batch in dl]
    assert len(batches) == 1

    # Check the cut that has no 'context' or 'system_prompt'
    conv = batches[0][0]
    assert isinstance(conv, NeMoMultimodalConversation)
    assert conv.id == cuts[0].id
    assert len(conv.turns) == 2
    assert isinstance(conv.turns[0], AudioTurn)
    assert conv.turns[0].role == "user"
    assert isinstance(conv.turns[1], TextTurn)
    assert conv.turns[1].role == "assistant"
    assert conv.turns[1].value == "transcript"
    assert conv.custom["test_key"] == "test_value"
    assert conv.turns[0].cut.custom["test_key"] == "test_value"

    # Check the cut that has only 'context' and no 'system_prompt'
    conv = batches[0][1]
    assert isinstance(conv, NeMoMultimodalConversation)
    assert conv.id == cuts[1].id
    assert len(conv.turns) == 3
    assert isinstance(conv.turns[0], TextTurn)
    assert conv.turns[0].role == "user"
    assert conv.turns[0].value == "some prompt"
    assert isinstance(conv.turns[1], AudioTurn)
    assert conv.turns[1].role == "user"
    assert isinstance(conv.turns[2], TextTurn)
    assert conv.turns[2].role == "assistant"
    assert conv.turns[2].value == "context and transcript"
    assert conv.custom["test_key"] == "test_value"
    assert conv.turns[1].cut.custom["test_key"] == "test_value"

    # Check the cut that has both 'context' and 'system_prompt'
    conv = batches[0][2]
    assert isinstance(conv, NeMoMultimodalConversation)
    assert conv.id == cuts[2].id
    assert len(conv.turns) == 4
    assert isinstance(conv.turns[0], TextTurn)
    assert conv.turns[0].role == "system"
    assert conv.turns[0].value == "system prompt"
    assert isinstance(conv.turns[1], TextTurn)
    assert conv.turns[1].role == "user"
    assert conv.turns[1].value == "other prompt"
    assert isinstance(conv.turns[2], AudioTurn)
    assert conv.turns[2].role == "user"
    assert isinstance(conv.turns[3], TextTurn)
    assert conv.turns[3].role == "assistant"
    assert conv.turns[3].value == "system context and transcript"
    assert conv.custom["test_key"] == "test_value"
    assert conv.turns[2].cut.custom["test_key"] == "test_value"


@pytest.fixture(scope="session")
def s2s_cutset_path(tmp_path_factory) -> Path:
    """
    1 session of 60s with 4 turns (2 user, 2 assistant).
    It ignores the typical presence of 'target_audio' attribute in S2S data, unnecessary for this test.
    """

    cut = dummy_cut(
        0,
        duration=60.0,
        recording_duration=60.0,
        supervisions=[
            SupervisionSegment("ut1", "c1", start=1.5, duration=7.13, text="greetings, assistant", speaker="user"),
            SupervisionSegment("at1", "c1", start=10.2, duration=8.49, text="welcome, user", speaker="assistant"),
            SupervisionSegment(
                "ut2", "c1", start=21.3, duration=15.2, text="lengthy issue description", speaker="user"
            ),
            SupervisionSegment("at2", "c1", start=38.1, duration=20.0, text="lengthy response", speaker="assistant"),
        ],
        with_data=True,
    )
    cuts = CutSet([cut])
    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).drop_in_memory_data().to_file(p)
    return p


def test_s2s_cut_to_conversation_conversion(s2s_cutset_path, tokenizer):
    cuts = CutSet.from_file(s2s_cutset_path)
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "s2s_as_conversation",
                    "cuts_path": s2s_cutset_path,
                    "audio_locator_tag": "[audio]",
                    "tags": {"test_key": "test_value"},
                },
            ],
            "token_equivalent_duration": 0.08,
            "prompt_format": "llama3",
            "force_finite": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=tokenizer
    )
    batches = [batch for batch in dl]
    assert len(batches) == 1

    # Check the multimodal conversation has 4 turns, with user audio turns, and assistant text turns
    conv = batches[0][0]
    assert isinstance(conv, NeMoMultimodalConversation)
    assert conv.id == cuts[0].id
    assert len(conv.turns) == 4

    assert isinstance(conv.turns[0], AudioTurn)
    assert conv.turns[0].role == "user"
    assert conv.turns[0].text == "greetings, assistant"
    audio = conv.turns[0].cut.load_audio()
    assert audio.shape[1] == compute_num_samples(7.13, conv.turns[0].cut.sampling_rate)

    assert isinstance(conv.turns[1], TextTurn)
    assert conv.turns[1].role == "assistant"
    assert conv.turns[1].value == "welcome, user"

    assert isinstance(conv.turns[2], AudioTurn)
    assert conv.turns[2].role == "user"
    assert conv.turns[2].text == "lengthy issue description"
    audio = conv.turns[2].cut.load_audio()
    assert audio.shape[1] == compute_num_samples(15.2, conv.turns[0].cut.sampling_rate)

    assert isinstance(conv.turns[3], TextTurn)
    assert conv.turns[3].role == "assistant"
    assert conv.turns[3].value == "lengthy response"

    assert conv.custom["test_key"] == "test_value"
    assert conv.turns[0].cut.custom["test_key"] == "test_value"


def test_dataloader_multimodal_conversation_tarred_slice_length_multi_epoch_different_sample(
    tarred_multimodal_conversations_path,
):
    jsonl, tar = tarred_multimodal_conversations_path
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    # 2 shards, 5 utterances each
                    "type": "multimodal_conversation",
                    "manifest_filepath": jsonl,
                    "tarred_audio_filepaths": tar,
                },
            ],
            "slice_length": 2,
            "shuffle": False,  # shuffle is disabled, but the slice offset still must be random!
            "num_workers": 0,
            "seed": 0,
            "shard_seed": 0,
            "batch_size": 2,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # 2 batches == 1 epoch => 4 batches == 2 epochs
    batches = [b for b in islice(dl, 4)]
    assert len(batches) == 4

    epoch0_ids = [cut.id for b in batches[:2] for cut in b]
    epoch1_ids = [cut.id for b in batches[2:] for cut in b]
    print(epoch0_ids, epoch1_ids)
    assert epoch0_ids != epoch1_ids
    assert epoch0_ids + epoch1_ids != sorted(epoch0_ids + epoch1_ids)  # true when slice_length=None


@pytest.fixture(scope="session")
def multiple_multimodal_conversations_path(multimodal_conversations_path, tmp_path_factory):
    (conversation,) = list(NeMoMultimodalConversationJsonlAdapter(multimodal_conversations_path, "[audio]"))
    out_dir = tmp_path_factory.mktemp("multi_convo")
    with JsonlShardWriter(f"{out_dir}/manifest_%d.jsonl", shard_size=5) as writer:
        for i in range(10):
            conversation.id = f'convo-{i}'
            writer.write(conversation.to_dict())
    return str(out_dir) + "/manifest_{0..1}.jsonl"


def test_dataloader_multimodal_conversation_nontarred_slice_length_ignored(multiple_multimodal_conversations_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    # 2 shards, 5 utterances each
                    "type": "multimodal_conversation",
                    "manifest_filepath": multiple_multimodal_conversations_path,
                },
            ],
            "slice_length": 2,
            "shuffle": False,  # shuffle is disabled, but the slice offset still must be random!
            "num_workers": 0,
            "seed": 0,
            "shard_seed": 0,
            "batch_size": 2,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # 5 batches = 1 epoch
    batches = [b for b in islice(dl, 5)]
    assert len(batches) == 5
    ids = [c.id for b in batches for c in b]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert ids == sorted(ids)


@pytest.fixture(scope="session")
def indexed_sharegpt_conversations_path(tmp_path_factory):
    """
    Creates a ShareGPT JSONL manifest with 10 text-only conversations
    and a corresponding .idx index file.
    """
    tmp_path = tmp_path_factory.mktemp("indexed_sharegpt_data")
    manifest_path = tmp_path / "sharegpt_manifest.jsonl"
    data = [
        {
            "id": f"convo_{i}",
            "conversations": [
                {"from": "human", "value": f"Question number {i}"},
                {"from": "gpt", "value": f"Answer number {i}"},
            ],
        }
        for i in range(10)
    ]
    lhotse.serialization.save_to_jsonl(data, manifest_path)
    create_index(str(manifest_path), str(manifest_path) + ".idx")
    return manifest_path


def test_sharegpt_indexed_sequential_no_shuffle(indexed_sharegpt_conversations_path):
    """When shuffle is off, indexed reading is NOT used — sequential order is preserved."""
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(indexed_sharegpt_conversations_path),
        audio_locator_tag="[audio]",
        shuffle_shards=False,
        shard_seed=0,
    )
    assert adapter._has_index is True
    conversations = list(adapter)
    assert len(conversations) == 10
    ids = [c.id for c in conversations]
    assert ids == [f"convo_{i}" for i in range(10)]


def test_sharegpt_indexed_shuffle_uses_random_access(indexed_sharegpt_conversations_path):
    """When shuffle is on and .idx files exist, all items are yielded in shuffled order."""
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(indexed_sharegpt_conversations_path),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=0,
    )
    assert adapter._has_index is True
    conversations = list(adapter)
    assert len(conversations) == 10
    ids = [c.id for c in conversations]
    # All items present
    assert sorted(ids) == [f"convo_{i}" for i in range(10)]
    # Order is shuffled (with 10 items the chance of identical order is 1/10! ≈ 0)
    assert ids != [f"convo_{i}" for i in range(10)]


def test_sharegpt_indexed_different_epochs_different_order(indexed_sharegpt_conversations_path):
    """Different epochs produce different shuffled orders."""
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(indexed_sharegpt_conversations_path),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=0,
    )
    epoch0_ids = [c.id for c in adapter]
    epoch1_ids = [c.id for c in adapter]
    # Both epochs have all items
    assert sorted(epoch0_ids) == sorted(epoch1_ids)
    # But in different order (epoch counter increments the seed)
    assert epoch0_ids != epoch1_ids


def test_sharegpt_no_index_falls_back_to_in_memory_shuffle(tmp_path_factory):
    """When .idx files don't exist, shuffle_shards still works via in-memory shuffle."""
    tmp_path = tmp_path_factory.mktemp("sharegpt_no_idx")
    manifest_path = tmp_path / "manifest.jsonl"
    data = [
        {
            "id": f"convo_{i}",
            "conversations": [
                {"from": "human", "value": f"Question {i}"},
                {"from": "gpt", "value": f"Answer {i}"},
            ],
        }
        for i in range(10)
    ]
    lhotse.serialization.save_to_jsonl(data, manifest_path)
    # No .idx file created

    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest_path),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=0,
    )
    assert adapter._has_index is False
    conversations = list(adapter)
    assert len(conversations) == 10
    ids = [c.id for c in conversations]
    assert sorted(ids) == [f"convo_{i}" for i in range(10)]
    # Should still be shuffled (in-memory path)
    assert ids != [f"convo_{i}" for i in range(10)]


def test_sharegpt_indexed_with_audio(tmp_path_factory):
    """Indexed reading works correctly with audio turns (ShareGPT format with <sound> placeholders)."""
    tmp_path = tmp_path_factory.mktemp("indexed_sharegpt_audio")
    manifest_path = tmp_path / "manifest.jsonl"

    # Create audio files
    for i in range(5):
        dummy_recording(i, duration=1.0 + i * 0.5, with_data=True).to_cut().save_audio(tmp_path / f"audio_{i}.wav")

    data = [
        {
            "id": f"audio_convo_{i}",
            "sound": f"audio_{i}.wav",
            "conversations": [
                {"from": "human", "value": f"Listen to this: <sound> What do you think?"},
                {"from": "gpt", "value": f"Response {i}"},
            ],
        }
        for i in range(5)
    ]
    lhotse.serialization.save_to_jsonl(data, manifest_path)
    create_index(str(manifest_path), str(manifest_path) + ".idx")

    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest_path),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=42,
    )
    assert adapter._has_index is True
    conversations = list(adapter)
    assert len(conversations) == 5

    ids = sorted([c.id for c in conversations])
    assert ids == [f"audio_convo_{i}" for i in range(5)]

    # Verify audio turns were created correctly
    for conv in conversations:
        assert conv.has_audio_turns
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        assert len(audio_turns) == 1
        assert audio_turns[0].audio_locator_tag == "[audio]"
        assert audio_turns[0].cut.load_audio().shape[0] == 1  # mono audio


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 100, 1000, 1023, 1024, 1025])
def test_lazy_shuffled_range_is_a_permutation(n):
    """LazyShuffledRange must yield every element of [0, n) exactly once."""
    rng = random.Random(42)
    result = list(LazyShuffledRange(n, rng))
    assert len(result) == n
    assert sorted(result) == list(range(n))


def test_lazy_shuffled_range_is_shuffled():
    """LazyShuffledRange should not produce the identity permutation (for non-trivial n)."""
    rng = random.Random(0)
    result = list(LazyShuffledRange(50, rng))
    assert result != list(range(50))


def test_lazy_shuffled_range_different_seeds():
    """Different RNG seeds produce different permutations."""
    a = list(LazyShuffledRange(100, random.Random(0)))
    b = list(LazyShuffledRange(100, random.Random(1)))
    assert a != b
    assert sorted(a) == sorted(b) == list(range(100))


# ─── WebDataset ShareGPT adapter tests ──────────────────────────────────────


def _make_webdataset_dir(
    tmp_path, num_samples=6, num_shards=2, create_idx=True, audio_first=False, create_meta=True, add_dir_entries=False
):
    """
    Helper: create a WebDataset directory layout with optional wids-meta.json,
    tar shards containing N.json + N.wav pairs, and optional .idx files.
    When *audio_first* is True the wav member is written before the json member.
    When *add_dir_entries* is True, directory entries are inserted into each tar shard.
    """
    import io
    import json
    import struct
    import tarfile

    import soundfile as sf
    from lhotse.testing.dummies import dummy_recording

    shard_size = num_samples // num_shards
    shard_dir = tmp_path / "0" / "sharded_manifests"
    shard_dir.mkdir(parents=True)

    shardlist = []
    sample_idx = 0
    for shard_id in range(num_shards):
        tar_path = shard_dir / f"shard-{shard_id}.tar"
        with tarfile.open(tar_path, "w:") as tar:
            if add_dir_entries:
                d = tarfile.TarInfo(name="./")
                d.type = tarfile.DIRTYPE
                tar.addfile(d)
            for local_idx in range(shard_size):
                data = {
                    "id": f"sample_{sample_idx}",
                    "sound": f"audio_{sample_idx}.wav",
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Listen to this: <sound> What is it?",
                        },
                        {
                            "from": "gpt",
                            "value": f"Response for sample {sample_idx}",
                        },
                    ],
                }
                json_bytes = json.dumps(data).encode("utf-8")

                rec = dummy_recording(sample_idx, duration=1.0 + sample_idx * 0.1, with_data=True)
                audio_array = rec.load_audio().squeeze()
                buf = io.BytesIO()
                sf.write(buf, audio_array, rec.sampling_rate, format="WAV")
                wav_bytes = buf.getvalue()

                json_info = tarfile.TarInfo(name=f"{local_idx}.json")
                json_info.size = len(json_bytes)
                wav_info = tarfile.TarInfo(name=f"{local_idx}.wav")
                wav_info.size = len(wav_bytes)

                if audio_first:
                    tar.addfile(wav_info, io.BytesIO(wav_bytes))
                    tar.addfile(json_info, io.BytesIO(json_bytes))
                else:
                    tar.addfile(json_info, io.BytesIO(json_bytes))
                    tar.addfile(wav_info, io.BytesIO(wav_bytes))

                sample_idx += 1

        shardlist.append(
            {
                "url": f"0/sharded_manifests/shard-{shard_id}.tar",
                "nsamples": shard_size,
                "filesize": tar_path.stat().st_size,
            }
        )

        if create_idx:
            create_tar_index(str(tar_path), str(tar_path) + ".idx")

    if create_meta:
        meta = {"name": "test-dataset", "__kind__": "test-WebDataset", "wids_version": 1, "shardlist": shardlist}
        (tmp_path / "wids-meta.json").write_text(json.dumps(meta, indent=2))

    return tmp_path


@pytest.fixture(scope="session")
def webdataset_dir(tmp_path_factory):
    """WebDataset directory with 2 shards of 3 samples each, including .idx files."""
    return _make_webdataset_dir(tmp_path_factory.mktemp("webdataset"))


@pytest.fixture(scope="session")
def webdataset_dir_no_idx(tmp_path_factory):
    """WebDataset directory without .idx files."""
    return _make_webdataset_dir(tmp_path_factory.mktemp("webdataset_no_idx"), create_idx=False)


def test_webdataset_sequential_no_shuffle(webdataset_dir):
    """Sequential iteration reads all samples in order."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=False,
        shard_seed=0,
    )
    conversations = list(adapter)
    assert len(conversations) == 6
    ids = [c.id for c in conversations]
    assert ids == [f"sample_{i}" for i in range(6)]


def test_webdataset_sequential_turn_structure(webdataset_dir):
    """Verify turn structure: text + audio + text + text (GPT response)."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=False,
    )
    conv = next(iter(adapter))
    assert conv.id == "sample_0"
    assert conv.has_audio_turns

    # "Listen to this:" (text) + <sound> (audio) + "What is it?" (text) + GPT response (text)
    assert len(conv.turns) == 4
    assert isinstance(conv.turns[0], TextTurn)
    assert conv.turns[0].role == "user"
    assert conv.turns[0].value == "Listen to this:"
    assert isinstance(conv.turns[1], AudioTurn)
    assert conv.turns[1].role == "user"
    assert conv.turns[1].audio_locator_tag == "[audio]"
    assert conv.turns[1].cut.load_audio().shape[0] == 1  # mono
    assert isinstance(conv.turns[2], TextTurn)
    assert conv.turns[2].role == "user"
    assert conv.turns[2].value == "What is it?"
    assert isinstance(conv.turns[3], TextTurn)
    assert conv.turns[3].role == "assistant"
    assert conv.turns[3].value == "Response for sample 0"


def test_webdataset_indexed_shuffle(webdataset_dir):
    """When shuffle is on and .idx files exist, all items are yielded in shuffled order."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=0,
    )
    assert adapter._has_index is True
    conversations = list(adapter)
    assert len(conversations) == 6
    ids = [c.id for c in conversations]
    assert sorted(ids) == [f"sample_{i}" for i in range(6)]
    # Order is shuffled (1/6! ≈ 0 chance of identity)
    assert ids != [f"sample_{i}" for i in range(6)]


def test_webdataset_indexed_different_epochs(webdataset_dir):
    """Different epochs produce different shuffled orders."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=0,
    )
    epoch0_ids = [c.id for c in adapter]
    epoch1_ids = [c.id for c in adapter]
    assert sorted(epoch0_ids) == sorted(epoch1_ids)
    assert epoch0_ids != epoch1_ids


def test_webdataset_no_index_falls_back_to_sequential_shuffle(webdataset_dir_no_idx):
    """Without .idx files, shuffle_shards still works (shard-level shuffle, sequential within)."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir_no_idx),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=0,
    )
    assert adapter._has_index is False
    conversations = list(adapter)
    assert len(conversations) == 6
    ids = [c.id for c in conversations]
    assert sorted(ids) == [f"sample_{i}" for i in range(6)]


def test_webdataset_audio_loads_correctly(webdataset_dir):
    """Audio loaded from tar matches the expected duration per sample."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=False,
    )
    for i, conv in enumerate(adapter):
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        assert len(audio_turns) == 1
        expected_dur = 1.0 + i * 0.1
        assert abs(audio_turns[0].cut.duration - expected_dur) < 0.01
        audio = audio_turns[0].cut.load_audio()
        assert audio.shape[0] == 1  # mono


def test_webdataset_indexed_audio_loads_correctly(webdataset_dir):
    """Audio loaded via indexed random access is valid and decodable."""
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(webdataset_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=True,
        shard_seed=42,
    )
    assert adapter._has_index is True
    conversations = list(adapter)
    assert len(conversations) == 6
    for conv in conversations:
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        assert len(audio_turns) == 1
        audio = audio_turns[0].cut.load_audio()
        assert audio.shape[0] == 1
        assert audio.shape[1] > 0


def test_sharegpt_audio_root(tmp_path_factory):
    """When audio_root is set, audio files are resolved relative to it, not the manifest directory."""
    manifest_dir = tmp_path_factory.mktemp("sharegpt_manifest_dir")
    audio_dir = tmp_path_factory.mktemp("sharegpt_audio_dir")

    # Create audio files in audio_dir (separate from manifest)
    dummy_recording(0, 2.0, with_data=True).to_cut().save_audio(audio_dir / "clip_a.wav")
    dummy_recording(1, 3.0, with_data=True).to_cut().save_audio(audio_dir / "clip_b.wav")

    manifest_path = manifest_dir / "manifest.jsonl"
    data = [
        {
            "id": "root_convo_1",
            "sound": "clip_a.wav",
            "conversations": [
                {"from": "human", "value": "Describe this: <sound>"},
                {"from": "gpt", "value": "It sounds nice."},
            ],
        },
        {
            "id": "root_convo_2",
            "sound": "clip_b.wav",
            "conversations": [
                {"from": "human", "value": "What about <sound>?"},
                {"from": "gpt", "value": "Also nice."},
            ],
        },
    ]
    lhotse.serialization.save_to_jsonl(data, manifest_path)

    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest_path),
        audio_locator_tag="[audio]",
        audio_root=str(audio_dir),
    )
    conversations = list(adapter)
    assert len(conversations) == 2
    assert conversations[0].id == "root_convo_1"
    assert conversations[1].id == "root_convo_2"

    # Verify audio was loaded from audio_dir
    audio_turns_1 = [t for t in conversations[0].turns if isinstance(t, AudioTurn)]
    assert len(audio_turns_1) == 1
    assert abs(audio_turns_1[0].cut.duration - 2.0) < 0.01
    assert audio_turns_1[0].cut.load_audio().shape[0] == 1

    audio_turns_2 = [t for t in conversations[1].turns if isinstance(t, AudioTurn)]
    assert len(audio_turns_2) == 1
    assert abs(audio_turns_2[0].cut.duration - 3.0) < 0.01
    assert audio_turns_2[0].cut.load_audio().shape[0] == 1


@pytest.mark.parametrize("use_index", [False, True])
def test_webdataset_audio_first_ordering(tmp_path_factory, use_index):
    """Tar samples with wav before json are handled correctly."""
    wds_dir = _make_webdataset_dir(
        tmp_path_factory.mktemp("webdataset_audio_first"),
        num_samples=4,
        num_shards=2,
        create_idx=use_index,
        audio_first=True,
    )
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(wds_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=use_index,
        shard_seed=0,
    )
    conversations = list(adapter)
    assert len(conversations) == 4
    ids = sorted(c.id for c in conversations)
    assert ids == [f"sample_{i}" for i in range(4)]
    for conv in conversations:
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        assert len(audio_turns) == 1
        assert audio_turns[0].cut.load_audio().shape[0] == 1


@pytest.mark.parametrize("create_idx", [False, True])
def test_webdataset_auto_discover_shards_no_meta(tmp_path_factory, create_idx):
    """When wids-meta.json is missing, tar shards are auto-discovered via rglob."""
    wds_dir = _make_webdataset_dir(
        tmp_path_factory.mktemp("webdataset_no_meta"),
        num_samples=4,
        num_shards=2,
        create_idx=create_idx,
        create_meta=False,
    )
    assert not (wds_dir / "wids-meta.json").exists()
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(wds_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=False,
    )
    assert adapter._has_index == create_idx
    conversations = list(adapter)
    assert len(conversations) == 4
    ids = sorted(c.id for c in conversations)
    assert ids == [f"sample_{i}" for i in range(4)]
    for conv in conversations:
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        assert len(audio_turns) == 1
        assert audio_turns[0].cut.load_audio().shape[0] == 1


def test_indexed_tar_reader_rejects_bad_idx(webdataset_dir):
    """IndexedTarSampleReader rejects an idx file with offsets that don't point to valid tar headers."""
    import struct

    shard_paths = sorted(webdataset_dir.rglob("*.tar"))
    tar_path = str(shard_paths[0])
    bad_idx = tar_path + ".bad.idx"
    # Write garbage offsets (huge values well past file size)
    with open(bad_idx, "wb") as f:
        for i in range(5):
            f.write(struct.pack("<Q", 10**15 + i))
    with pytest.raises(ValueError, match="beyond file size"):
        IndexedTarSampleReader(tar_path, bad_idx)


def test_indexed_tar_reader_strips_trailing_zero_block_sentinel(tmp_path_factory):
    """IndexedTarSampleReader silently strips trailing sentinel pointing to end-of-archive zero blocks."""
    import struct

    wds_dir = _make_webdataset_dir(
        tmp_path_factory.mktemp("webdataset_zero_idx"), num_samples=2, num_shards=1, create_idx=True
    )
    tar_path = str(next(wds_dir.rglob("*.tar")))
    tar_size = os.path.getsize(tar_path)
    # Read the valid index to get the real offsets, then append a bad trailing entry
    good_idx = tar_path + ".idx"
    with open(good_idx, "rb") as f:
        good_data = f.read()
    bad_idx = tar_path + ".bad.idx"
    # Take the valid sample offsets but replace the sentinel with a zero-block offset
    with open(bad_idx, "wb") as f:
        f.write(good_data[:-8])  # all offsets except the file-size sentinel
        f.write(struct.pack("<Q", tar_size - 1024))  # add zero-block offset as sentinel
    reader = IndexedTarSampleReader(tar_path, bad_idx)
    assert len(reader) == 2  # trailing zero-block entry was stripped


def test_indexed_tar_reader_rejects_all_zero_block_offsets(tmp_path_factory):
    """IndexedTarSampleReader raises when all offsets (including first) point to zero blocks."""
    import struct

    wds_dir = _make_webdataset_dir(
        tmp_path_factory.mktemp("webdataset_all_zero"), num_samples=2, num_shards=1, create_idx=False
    )
    tar_path = str(next(wds_dir.rglob("*.tar")))
    tar_size = os.path.getsize(tar_path)
    bad_idx = tar_path + ".idx"
    with open(bad_idx, "wb") as f:
        f.write(struct.pack("<Q", tar_size - 1024))
        f.write(struct.pack("<Q", tar_size))
    with pytest.raises(ValueError, match="zero block"):
        IndexedTarSampleReader(tar_path, bad_idx)


def test_webdataset_auto_discover_no_tars_raises(tmp_path_factory):
    """When wids-meta.json is missing and no tar files exist, raise FileNotFoundError."""
    empty_dir = tmp_path_factory.mktemp("webdataset_empty")
    with pytest.raises(FileNotFoundError, match="No wids-meta.json and no .tar files"):
        NeMoMultimodalConversationShareGPTWebdatasetAdapter(
            data_dir=str(empty_dir),
            audio_locator_tag="[audio]",
        )


@pytest.mark.parametrize("use_index", [False, True])
def test_webdataset_tar_with_directory_entries(tmp_path_factory, use_index):
    """Tar shards containing directory entries are handled correctly."""
    wds_dir = _make_webdataset_dir(
        tmp_path_factory.mktemp("webdataset_dir_entries"),
        num_samples=4,
        num_shards=2,
        create_idx=use_index,
        add_dir_entries=True,
    )
    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(wds_dir),
        audio_locator_tag="[audio]",
        shuffle_shards=use_index,
        shard_seed=0,
    )
    conversations = list(adapter)
    assert len(conversations) == 4
    ids = sorted(c.id for c in conversations)
    assert ids == [f"sample_{i}" for i in range(4)]
    for conv in conversations:
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        assert len(audio_turns) == 1
        assert audio_turns[0].cut.load_audio().shape[0] == 1
