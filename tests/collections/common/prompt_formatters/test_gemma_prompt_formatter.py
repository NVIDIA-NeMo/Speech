from nemo.collections.common.prompts.gemma import GemmaPromptFormatter


def test_gemma_prompt_formatter_training(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [ 21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
         30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
         18,   6,  60,   9,   7,  73,  61,  69,   1,  81,  20,  30, 104,  59,
         18,  26,  18,   6,  60,   9,   7]
    assert ans["context_ids"].tolist() == [ 21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
         30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
         18,   6,  60,   9,   7,  73,  61,  69]
    assert ans["answer_ids"].tolist() == [1,  81,  20,  30, 104,  59,
         18,  26,  18,   6,  60,   9,   7]
    assert ans["mask"].tolist() == [False] * 36 + [True] * 13
    # fmt: on


def test_gemma_prompt_formatter_inference(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [ 21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
                                          30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
                                          18,   6,  60,   9,   7,  73,  61,  69]
    # fmt: on
