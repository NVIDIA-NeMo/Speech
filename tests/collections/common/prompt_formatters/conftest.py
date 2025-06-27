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

import pytest

from nemo.collections.common.tokenizers import CanaryTokenizer, SentencePieceTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model

# Note: We don't really define special tokens for this test so every 'special token'
#       will be represented as a number of regular tokens.
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
SYSTEM
"""


@pytest.fixture(scope="session")
def bpe_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("bpe_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(TOKENIZER_TRAIN_TEXT)
    create_spt_model(
        str(text_path),
        vocab_size=512,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        remove_extra_whitespaces=True,
        bos=True,
        eos=True,
        user_defined_symbols=['\n', '<|im_start|>', '<|im_end|>', '<SPECIAL_10>', '<SPECIAL_11>'],
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.fixture(scope="session")
def canary_tokenizer(bpe_tokenizer, tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("spl_tokens")
    spl_tokens = CanaryTokenizer.build_special_tokenizer(["transcribe", "en"], tmpdir)
    return CanaryTokenizer(
        tokenizers={
            "spl_tokens": spl_tokens,
            "en": bpe_tokenizer,
        }
    )
