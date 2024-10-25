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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:-1]) == '[INST] TEST [/INST] TEST'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()[1:]) == '[INST] TEST [/INST]'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()[:-1]) == 'TEST'
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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:]) == '[INST] TEST [/INST]'
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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:-1]) == '[INST] <<SYS>> TEST <</SYS>> TEST [/INST] TEST'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()[1:]) == '[INST] <<SYS>> TEST <</SYS>> TEST [/INST]'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()[:-1]) == 'TEST'
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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:]) == '[INST] <<SYS>> TEST <</SYS>> TEST [/INST]'
    # fmt: on
