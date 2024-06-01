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
"""


@pytest.fixture(scope="session")
def bpe_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("bpe_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(TOKENIZER_TRAIN_TEXT)
    create_spt_model(str(text_path), vocab_size=512, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir))
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
