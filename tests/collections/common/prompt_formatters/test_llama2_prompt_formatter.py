from nemo.collections.common.prompts.llama import Llama2PromptFormatter


def test_llama2_prompt_formatter_training(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [-1, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, -1]
    assert ans["context_ids"].tolist() == [-1, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    assert ans["answer_ids"].tolist() == [1, 81, 20, 30, -1]
    assert ans["mask"].tolist() == [False] * 16 + [True] * 5
    # fmt: on


def test_llama2_prompt_formatter_inference(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [-1, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    # fmt: on


def test_llama2_prompt_formatter_training_with_system(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "system_and_user", "slots": {"system": "TEST", "message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [-1, 54, 42, 49, 30, 50, 77, 13, 45, 13, 7, 7, 1, 81, 20, 30, 21, 66, 13, 45, 13, 7, 7, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, -1]
    assert ans["context_ids"].tolist() == [-1, 54, 42, 49, 30, 50, 77, 13, 45, 13, 7, 7, 1, 81, 20, 30, 21, 66, 13, 45, 13, 7, 7, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    assert ans["answer_ids"].tolist() == [1, 81, 20, 30, -1]
    assert ans["mask"].tolist() == [False] * 33 + [True] * 5
    # fmt: on


def test_llama2_prompt_formatter_inference_with_system(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "system_and_user", "slots": {"system": "TEST", "message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [-1, 54, 42, 49, 30, 50, 77, 13, 45, 13, 7, 7, 1, 81, 20, 30, 21, 66, 13, 45, 13, 7, 7, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    # fmt: on
