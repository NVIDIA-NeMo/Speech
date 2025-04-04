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

import base64
from unittest.mock import patch

import pytest
from PIL import Image

from nemo.collections.vlm.neva.data.conversation import (
    Conversation,
    SeparatorStyle,
    conv_chatml_direct,
    conv_gemma_instruct,
    conv_llama_2,
    conv_llava_llama_3,
    conv_llava_plain,
    conv_llava_v0,
    conv_llava_v1,
    conv_mistral_direct,
    conv_mistral_orca,
    conv_mistral_vila,
    conv_mistral_zephyr,
    conv_mpt,
    conv_nv_dpo,
    conv_nvgpt,
    conv_qwen,
)


@pytest.fixture
def sample_image():
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def basic_conversation():
    return Conversation(
        system="Test system",
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )


def test_conversation_initialization():
    conv = Conversation(system="Test system", roles=("User", "Assistant"), messages=[], offset=0)
    assert conv.system == "Test system"
    assert conv.roles == ("User", "Assistant")
    assert conv.messages == []
    assert conv.offset == 0
    assert conv.sep_style == SeparatorStyle.SINGLE
    assert conv.sep == "###"


def test_get_prompt_single_style(basic_conversation):
    basic_conversation.append_message("User", "Hello")
    basic_conversation.append_message("Assistant", "Hi there")
    prompt = basic_conversation.get_prompt()
    assert "Test system###" in prompt
    assert "User: Hello###" in prompt
    assert "Assistant: Hi there###" in prompt


def test_get_prompt_two_style():
    conv = Conversation(
        system="Test system",
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    conv.append_message("User", "Hello")
    conv.append_message("Assistant", "Hi there")
    prompt = conv.get_prompt()
    assert "Test system " in prompt
    assert "User: Hello" in prompt
    assert "Assistant: Hi there</s>" in prompt


def test_get_prompt_mistral_vila():
    conv = conv_mistral_vila.copy()
    conv.append_message("USER", "Hello")
    conv.append_message("ASSISTANT", "Hi there")
    prompt = conv.get_prompt()
    assert "<s>" in prompt


# def test_get_prompt_llama_2():
#     conv = conv_llama_2.copy()
#     conv.append_message("USER", "Hello")
#     conv.append_message("ASSISTANT", "Hi there")
#     prompt = conv.get_prompt()
#     assert "You are a helpful" in prompt
#
#
# def test_get_prompt_llama_3():
#     conv = conv_llava_llama_3.copy()
#     conv.append_message("user", "Hello")
#     conv.append_message("assistant", "Hi there")
#     prompt = conv.get_prompt()
#     assert "<|begin_of_text|>" in prompt
#     assert "user" in prompt
#     assert "assistant" in prompt


def test_get_prompt_nvgpt():
    conv = conv_nvgpt.copy()
    conv.append_message("User", "Hello")
    conv.append_message("Assistant", "Hi there")
    prompt = conv.get_prompt()
    assert "System" in prompt
    assert "User" in prompt
    assert "Assistant" in prompt


def test_get_prompt_plain():
    conv = conv_llava_plain.copy()
    conv.append_message("", "Hello")
    conv.append_message("", "Hi there")
    prompt = conv.get_prompt()
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_v0():
    conv = conv_llava_v0.copy()
    conv.append_message("Human", "Hello")
    conv.append_message("Assistant", "Hi there")
    prompt = conv.get_prompt()
    assert "Human: Hello###" in prompt
    assert "Assistant: Hi there###" in prompt


def test_get_prompt_v1():
    conv = conv_llava_v1.copy()
    conv.append_message("USER", "Hello")
    conv.append_message("ASSISTANT", "Hi there")
    prompt = conv.get_prompt()
    assert "USER: Hello" in prompt
    assert "ASSISTANT: Hi there" in prompt


def test_get_prompt_mistral_orca():
    conv = conv_mistral_orca.copy()
    conv.append_message("<|im_start|>user\n", "Hello")
    conv.append_message("<|im_start|>assistant\n", "Hi there")
    prompt = conv.get_prompt()
    assert "You are MistralOrca" in prompt
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_mistral_zephyr():
    conv = conv_mistral_zephyr.copy()
    conv.append_message("<|user|>\n", "Hello")
    conv.append_message("<|assistant|>\n", "Hi there")
    prompt = conv.get_prompt()
    assert "You are a helpful AI assistant" in prompt
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_mistral_direct():
    conv = conv_mistral_direct.copy()
    conv.append_message("<|im_start|>user\n", "Hello")
    conv.append_message("<|im_start|>assistant\n", "Hi there")
    prompt = conv.get_prompt()
    assert "Answer the questions" in prompt
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_chatml_direct():
    conv = conv_chatml_direct.copy()
    conv.append_message("<|im_start|>user\n", "Hello")
    conv.append_message("<|im_start|>assistant\n", "Hi there")
    prompt = conv.get_prompt()
    assert "Answer the questions" in prompt
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_mpt():
    conv = conv_mpt.copy()
    conv.append_message("<|im_start|>user\n", "Hello")
    conv.append_message("<|im_start|>assistant\n", "Hi there")
    prompt = conv.get_prompt()
    assert "A conversation between a user" in prompt
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_qwen():
    conv = conv_qwen.copy()
    conv.append_message("<|im_start|>user", "Hello")
    conv.append_message("<|im_start|>assistant", "Hi there")
    prompt = conv.get_prompt()
    assert "You are a helpful assistant" in prompt
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_gemma():
    conv = conv_gemma_instruct.copy()
    conv.append_message("<start_of_turn>user\n", "Hello")
    conv.append_message("<start_of_turn>model\n", "Hi there")
    prompt = conv.get_prompt()
    assert "Hello" in prompt
    assert "Hi there" in prompt


def test_get_prompt_nv_dpo():
    conv = conv_nv_dpo.copy()
    conv.append_message("User", "Hello")
    conv.append_message("Assistant", "Hi there")
    prompt = conv.get_prompt()
    assert "System" in prompt
    assert "User" in prompt
    assert "Assistant" in prompt


def test_process_image_pad(sample_image):
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[], offset=0)
    processed = conv.process_image(sample_image, "Pad", return_pil=True)
    assert isinstance(processed, Image.Image)
    assert processed.size[0] == processed.size[1]  # Should be square


def test_process_image_resize(sample_image):
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[], offset=0)
    processed = conv.process_image(sample_image, "Resize", return_pil=True)
    assert isinstance(processed, Image.Image)
    assert processed.size == (336, 336)


def test_process_image_default(sample_image):
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[], offset=0)
    processed = conv.process_image(sample_image, "Default", return_pil=True)
    assert isinstance(processed, Image.Image)


def test_process_image_base64(sample_image):
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[], offset=0)
    processed = conv.process_image(sample_image, "Default", return_pil=False)
    assert isinstance(processed, str)
    # Verify it's a valid base64 string
    try:
        base64.b64decode(processed)
    except Exception:
        pytest.fail("Not a valid base64 string")


def test_get_images(basic_conversation, sample_image):
    basic_conversation.append_message("User", ("Hello", sample_image, "Default"))
    images = basic_conversation.get_images(return_pil=True)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_get_images_return_path(basic_conversation, sample_image):
    basic_conversation.append_message("User", ("Hello", sample_image, "Default"))
    images = basic_conversation.get_images(return_path=True)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_to_gradio_chatbot(basic_conversation, sample_image):
    basic_conversation.append_message("User", ("Hello", sample_image, "Default"))
    basic_conversation.append_message("Assistant", "Hi there")
    chatbot = basic_conversation.to_gradio_chatbot()
    assert len(chatbot) == 1
    assert isinstance(chatbot[0], list)
    assert len(chatbot[0]) == 2
    assert "Hello" in chatbot[0][0]
    assert "Hi there" == chatbot[0][1]


def test_copy():
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[["User", "Hello"]], offset=0)
    copied = conv.copy()
    assert copied.system == conv.system
    assert copied.roles == conv.roles
    assert copied.messages == conv.messages
    assert copied.offset == conv.offset
    assert copied.sep_style == conv.sep_style
    assert copied.sep == conv.sep
    assert copied.sep2 == conv.sep2


def test_dict():
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[["User", "Hello"]], offset=0)
    conv_dict = conv.dict()
    assert conv_dict["system"] == "Test"
    assert conv_dict["roles"] == ("User", "Assistant")
    assert conv_dict["messages"] == [["User", "Hello"]]
    assert conv_dict["offset"] == 0
    assert conv_dict["sep"] == "###"


def test_dict_with_images(basic_conversation, sample_image):
    basic_conversation.append_message("User", ("Hello", sample_image, "Default"))
    conv_dict = basic_conversation.dict()
    assert conv_dict["messages"] == [["User", "Hello"]]


def test_process_chat_template():
    conv = Conversation(
        system="Test system", roles=("user", "assistant"), messages=[], offset=0, sep_style=SeparatorStyle.LLAMA_2
    )
    messages = [("user", "Hello"), ("assistant", "Hi there")]
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer.return_value.apply_chat_template.return_value = "Test template"
        result = conv.process_chat_template("test-tokenizer", messages)
        assert result == "Test template"
        mock_tokenizer.assert_called_once_with("test-tokenizer")


def test_invalid_sep_style():
    conv = Conversation(
        system="Test", roles=("User", "Assistant"), messages=[], offset=0, sep_style="INVALID"  # type: ignore
    )
    with pytest.raises(ValueError):
        conv.get_prompt()


def test_invalid_image_process_mode():
    conv = Conversation(system="Test", roles=("User", "Assistant"), messages=[], offset=0)
    with pytest.raises(ValueError):
        conv.process_image(None, "InvalidMode")
