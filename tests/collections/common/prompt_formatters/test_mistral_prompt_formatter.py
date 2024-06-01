from nemo.collections.common.prompts.mistral import MistralPromptFormatter


def test_mistral_prompt_formatter_training(bpe_tokenizer):
    formatter = MistralPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, 66, 8, 7]
    assert ans["context_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    assert ans["answer_ids"].tolist() == [1, 81, 20, 30, 66, 8, 7]
    assert ans["mask"].tolist() == [False] * 18 + [True] * 7
    # fmt: on


def test_mistral_prompt_formatter_inference(bpe_tokenizer):
    formatter = MistralPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    # fmt: on
