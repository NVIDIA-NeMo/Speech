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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<s> [INST] TEST [/INST] TEST</s>'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<s> [INST] TEST [/INST]'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == 'TEST</s>'
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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<s> [INST] TEST [/INST]'
    # fmt: on
