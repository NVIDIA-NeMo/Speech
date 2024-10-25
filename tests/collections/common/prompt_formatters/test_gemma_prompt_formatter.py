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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<start_of_turn>user TEST<end_of_turn> <start_of_turn>model TEST<end_of_turn>'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<start_of_turn>user TEST<end_of_turn> <start_of_turn>model'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == 'TEST<end_of_turn>'
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
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<start_of_turn>user TEST<end_of_turn> <start_of_turn>model'
    # fmt: on
