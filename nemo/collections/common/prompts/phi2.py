"""
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class Phi2QAPromptFormatter(PromptFormatter):
    NAME = "phi2_qa"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Instruct: |message|\nOutput: ",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


class Phi2ChatPromptFormatter(PromptFormatter):
    NAME = "phi2_chat"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Human: |message|\nAI: ",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


class Phi2CodePromptFormatter(PromptFormatter):
    NAME = "phi2_code"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
