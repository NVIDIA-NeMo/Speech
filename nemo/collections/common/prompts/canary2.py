# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from lhotse import CutSet, MonoCut
from lhotse.cut import MixedCut
from lhotse.utils import ifnone

from nemo.collections.common.prompts.fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY2_BOCTX,
    CANARY_BOS,
    CANARY_EOS,
    CANARY_SPECIAL_TOKENIZER,
)


class Canary2PromptFormatter(PromptFormatter):
    NAME = "canary2"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        # User prompt.
        # This is the main role used for training and ASR/AST inference.
        "user": {
            "template": f"{CANARY2_BOCTX}|decodercontext|{CANARY_BOS}|emotion||source_lang||target_lang||pnc||itn||timestamp||diarize|",
            "slots": {
                # Empty string or previous transcript / other context to bias predictions.
                "decodercontext": Modality.Text,
                # Emotion of the speaker - may be predicted by the model with a partial prompt.
                "emotion": Modality.TextLiteral(
                    "<|emo:undefined|>", "<|emo:neutral|>", "<|emo:angry|>", "<|emo:happy|>", "<|emo:sad|>"
                ),
                # Audio input language - may be predicted by the model with a partial prompt.
                "source_lang": Modality.Text,
                # Transcription language - specified by the user.
                "target_lang": Modality.Text,
                # Should we predict punctuation and capitalization?
                "pnc": Modality.TextLiteral(
                    "yes", "no", "true", "True", "false", "False", "1", "0", "pnc", "nopnc", "<|pnc|>", "<|nopnc|>"
                ),
                # Should we predict with inverse text normalization (numerals as digits, abbreviations, etc.)
                "itn": Modality.TextLiteral(
                    "yes", "no", "true", "True", "false", "False", "1", "0", "itn", "noitn", "<|itn|>", "<|noitn|>"
                ),
                # Should we predict timestamps?
                "timestamp": Modality.TextLiteral(
                    "yes",
                    "no",
                    "true",
                    "True",
                    "false",
                    "False",
                    "1",
                    "0",
                    "timestamp",
                    "notimestamp",
                    "<|timestamp|>",
                    "<|notimestamp|>",
                ),
                # Should we diarize speech?
                "diarize": Modality.TextLiteral(
                    "yes",
                    "no",
                    "true",
                    "True",
                    "false",
                    "False",
                    "1",
                    "0",
                    "diarize",
                    "nodiarize",
                    "<|diarize|>",
                    "<|nodiarize|>",
                ),
            },
        },
        # User prompt.
        # This role is used for emotion / LID inference only - use it for two just two decoder inference steps
        # to retrieve the recognized emotion and language tokens.
        "user_partial": {
            "template": f"{CANARY2_BOCTX}|decodercontext|{CANARY_BOS}",
            "slots": {
                # Empty string or previous transcript / other context to bias predictions.
                "decodercontext": Modality.Text,
            },
        },
        # System's reponse.
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
            val = slot_values[k]
            slot_values[k] = "<|" + val + "|>"
            any_special_token_present = True

    # Handle boolean slots
    for k in ("pnc", "itn", "timestamp", "diarize"):
        true_token = f"<|{k}|>"
        false_token = f"<|no{k}|>"
        if k in slot_values and slot_values[k] not in (true_token, false_token):
            slot_values[k] = true_token if slot_values[k] in ("yes", "1", "True", "true", k) else false_token
            any_special_token_present = True

    # Auto-inject which tokenizer to look up in CanaryTokenizer if not provided,
    # and slots for this turn correspond to user prompt.
    if any_special_token_present and PromptFormatter.PROMPT_LANGUAGE_SLOT not in slot_values:
        slot_values[PromptFormatter.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER

    return slot_values


@registered_prompt_format_fn
def canary2(
    cuts: CutSet, tokenizer: TokenizerSpec
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Prepend and append control tokens to the token sequence as per Canary 2.0 format.

    The prompt format syntax is defined in :class:`Canary2PromptFormatter`
    """
    formatter = Canary2PromptFormatter(tokenizer)

    prompts_with_answers, prompts, answers = [], [], []
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        if not isinstance(cut, MonoCut):
            raise TypeError(
                f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
            )

        # first, validate the utterance
        expected_slots = {"source_lang", "target_lang"}
        missing_keys = expected_slots - set(cut.custom)
        if missing_keys:
            raise RuntimeError(
                f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
                f"Please ensure that every utterance in the input manifests contains these keys."
            )

        optional_slots = {
            "decodercontext": "",
            "emotion": "<|emo:undefined|>",
            "itn": "<|noitn|>",
            "timestamp": "<|notimestamp|>",
            "diarize": "<|nodiarize|>",
            "pnc": "<|pnc|>",  # consistent with canary1
        }
        slots = {slot: cut.custom[slot] for slot in expected_slots}
        slots[formatter.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER
        for k, v in optional_slots.items():
            slots[k] = cut.custom[k] if k in cut.custom else v

        turns = [dict(role="user", slots=slots)]
        # If data has no transcript, create empty response with <eos> only.
        text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
        turns.append(
            dict(
                role="assistant",
                slots={
                    "text": text,
                    formatter.PROMPT_LANGUAGE_SLOT: ifnone(
                        cut.supervisions[0].language, cut.custom.get("target_lang")
                    ),
                },
            ),
        )
        encoded = formatter.encode_dialog(turns)
        prompts_with_answers.append(encoded["input_ids"])
        prompts.append(encoded["context_ids"])
        if "answer_ids" in encoded:
            assert (
                encoded["answer_ids"][-1].item() == formatter.tokenizer.eos
            ), f"Expected the last token in answer_ids to be EOS, but we got {encoded['answer_ids']=}"
            answers.append(encoded["answer_ids"][:-1])  # Strip Canary's EOS
        else:
            answers.append([])

    return prompts_with_answers, prompts, answers
