"""
Implemented following the guide at https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

MISTRAL_BOS = "<s>"
MISTRAL_PROMPT_BEGIN = "[INST]"
MISTRAL_PROMPT_END = "[/INST]"
MISTRAL_END_OF_TURN = "</s>"
MISTRAL_NL = "\n\n"


class MistralPromptFormatter(PromptFormatter):
    NAME = "mistral"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "preamble": {
            "template": MISTRAL_BOS,
        },
        "user": {
            "template": f"{MISTRAL_PROMPT_BEGIN} |message| {MISTRAL_PROMPT_END} ",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|{MISTRAL_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
