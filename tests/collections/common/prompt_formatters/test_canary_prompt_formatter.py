from nemo.collections.common.prompts.canary import CanaryPromptFormatter


def test_canary_prompt_formatter_training(canary_tokenizer):
    formatter = CanaryPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "source_lang": "<|en|>",
                    "target_lang": "<|en|>",
                    "task": "<|transcribe|>",
                    "pnc": "<|pnc|>",
                    "prompt_language": "spl_tokens",
                },
            },
            {"role": "assistant", "slots": {"text": "TEST", "prompt_language": "en"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [4, 8, 7, 8, 5, 11, 91, 30, 40, 3]
    assert ans["context_ids"].tolist() == [4, 8, 7, 8, 5]
    assert ans["answer_ids"].tolist() == [11, 91, 30, 40, 3]
    assert ans["mask"].tolist() == [False] * 5 + [True] * 5
    # fmt: on


def test_canary_prompt_formatter_inference(canary_tokenizer):
    formatter = CanaryPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "source_lang": "<|en|>",
                    "target_lang": "<|en|>",
                    "task": "<|transcribe|>",
                    "pnc": "<|pnc|>",
                    "prompt_language": "spl_tokens",
                },
            },
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [4, 8, 7, 8, 5]
    # fmt: on
