from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY_BOS,
    CANARY_EOS,
    CANARY_NOPNC,
    CANARY_PNC,
    CANARY_SPECIAL_TOKENIZER,
)


class CanaryPromptFormatter(PromptFormatter):
    NAME = "canary"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{CANARY_BOS}|source_lang||task||target_lang||pnc|",
            "slots": {
                "source_lang": Modality.Text,
                "task": Modality.TextLiteral(
                    "asr", "ast", "translate", "transcribe", "s2t_translation", "<|transcribe|>", "<|translate|>"
                ),
                "target_lang": Modality.Text,
                "pnc": Modality.TextLiteral(
                    "yes", "no", "true", "True", "false", "False", "1", "0", "pnc", "nopnc", "<|pnc|>", "<|nopnc|>"
                ),
            },
        },
        OUTPUT_ROLE: {
            "template": f"|text|{CANARY_EOS}",
            "slots": {
                "text": Modality.Text,
            },
        },
    }

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        # This method handles a level of indirection for Canary.
        # It maps values provided in trcfg to the actual special tokens
        # expected to be present in canary prompt.
        # It used to be done in prompt_format_fn inside Dataset class corresponding to Canary,
        # but we are not using it here anymore.
        # This maps things such as '|task|: "asr"' to '|TASK|: "<|transcribe|>"'.
        slot_values = map_manifest_values_to_special_tokens(slot_values)
        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


def map_manifest_values_to_special_tokens(slot_values: dict[str, str]) -> dict[str, str]:
    slot_values = slot_values.copy()

    any_special_token_present = False

    for k in ("source_lang", "target_lang"):
        if k in slot_values and not ((v := slot_values[k]).startswith("<|") and v.endswith("|>")):
            slot_values[k] = "<|" + slot_values[k] + "|>"
            any_special_token_present = True

    k = "pnc"
    if k in slot_values and slot_values[k] not in (CANARY_PNC, CANARY_NOPNC):
        slot_values[k] = CANARY_PNC if slot_values[k] in ("yes", "1", "True", "true", "pnc") else CANARY_NOPNC
        any_special_token_present = True

    # Note: we re-map 'taskname' to 'task' for compatibility with earlier versions of Canary training.
    for k in ("task", "taskname"):
        if k in slot_values and slot_values[k] not in ("<|transcribe|>", "<|translate|>"):
            if slot_values[k] in {"translate", "ast", "s2t_translation"}:
                slot_values["task"] = "<|translate|>"
            elif slot_values[k] in {"transcribe", "asr"}:
                slot_values["task"] = "<|transcribe|>"
            else:
                assert False, f"Task {slot_values[k]} invalid task for slot {k}"
            any_special_token_present = True

    # Auto-inject which tokenizer to look up in CanaryTokenizer if not provided,
    # and slots for this turn correspond to user prompt.
    if any_special_token_present and PromptFormatter.PROMPT_LANGUAGE_SLOT not in slot_values:
        slot_values[PromptFormatter.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER

    return slot_values
