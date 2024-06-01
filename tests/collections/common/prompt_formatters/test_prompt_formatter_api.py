import pytest

from nemo.collections.common.prompts.canary import PromptFormatter
from nemo.collections.common.prompts.formatter import Modality


class _DummyPromptFormatter(PromptFormatter):
    NAME = "_dummy_test_formatter"
    TEMPLATE = {
        "user": {"template": "<s>|text|</s>", "slots": {"text": Modality.Text}},
        "assistant": {"template": "|text|</s>", "slots": {"text": Modality.Text}},
    }
    OUTPUT_ROLE = "assistant"


def test_prompt_formatter_empty_dialog_exception(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    with pytest.raises(AssertionError):
        formatter.encode_dialog([])


def test_prompt_formatter_inference(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "user", "slots": {"text": "hi"}}])
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "<s>hi</s>"


def test_prompt_formatter_training(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"text": "hi"}},
            {"role": "assistant", "slots": {"text": "hello"}},
        ]
    )
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "<s>hi</s> hello</s>", recovered


def test_prompt_formatter_missing_role(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    with pytest.raises(AssertionError, match="A turn must have have a 'role' key"):
        formatter.encode_dialog([{"slots": {"text": "hi"}}])


def test_prompt_formatter_missing_slots(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    with pytest.raises(
        AssertionError, match="A turn for role user must have have a non-empty value under 'slots' key"
    ):
        formatter.encode_dialog([{"role": "user"}])
    with pytest.raises(
        AssertionError, match="A turn for role user must have have a non-empty value under 'slots' key"
    ):
        formatter.encode_dialog([{"role": "user", "slots": {}}])


def test_prompt_formatter_aggregate_tokenizer(canary_tokenizer):
    # Note the 'canary_tokenizer' arg which is an aggregate tokenizer fixture.
    formatter = _DummyPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "text": "hi",
                    "prompt_language": "en",
                },
            }
        ]
    )
    recovered = canary_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == " <s>hi</s>"


def test_prompt_formatter_aggregate_tokenizer_missing_prompt_language(canary_tokenizer):
    # Note the 'canary_tokenizer' arg which is an aggregate tokenizer fixture.
    formatter = _DummyPromptFormatter(canary_tokenizer)

    with pytest.raises(AssertionError, match="Missing key 'prompt_language' in slot_values"):
        formatter.encode_dialog([{"role": "user", "slots": {"text": "hi"}}])


class _DummyPreamblePromptFormatter(PromptFormatter):
    NAME = "_dummy_preamble_test_formatter"
    TEMPLATE = {
        "preamble": {"template": "TEST"},
        "user": {"template": "<s>|text|</s>", "slots": {"text": Modality.Text}},
        "assistant": {"template": "|text|</s>", "slots": {"text": Modality.Text}},
    }
    OUTPUT_ROLE = "assistant"


def test_prompt_formatter_preamble_inference(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "user", "slots": {"text": "hi"}}])
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "TEST <s>hi</s>", recovered


def test_prompt_formatter_premble_training(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"text": "hi"}},
            {"role": "assistant", "slots": {"text": "hello"}},
        ]
    )
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "TEST <s>hi</s> hello</s>"


def test_prompt_formatter_explicit_preamble(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "preamble"}, {"role": "user", "slots": {"text": "hi"}}])
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "TEST <s>hi</s>"


def test_prompt_formatter_wrong_preamble_excpetions(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    with pytest.raises(AssertionError):
        # Error: 2 preambles
        formatter.encode_dialog(
            [
                {"role": "preamble"},
                {"role": "preamble"},
                {"role": "user", "slots": {"text": "hi"}},
            ]
        )
    with pytest.raises(AssertionError):
        # Error: preamble not at the beginning
        formatter.encode_dialog(
            [
                {"role": "user", "slots": {"text": "hi"}},
                {"role": "preamble"},
            ]
        )
    with pytest.raises(AssertionError):
        # Error: preamble with slots
        formatter.encode_dialog(
            [
                {"role": "user", "slots": {"text": "hi"}},
                {"role": "preamble", "slots": {"abc": "abc"}},
            ]
        )
