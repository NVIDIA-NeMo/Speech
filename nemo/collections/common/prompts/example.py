"""
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class ExamplePromptFormatter(PromptFormatter):
    """
    The simplest possible prompt formatter implementation.

    It defines a dialog of the form:

        User: Hi.
        Assistant: Hi, how can I help you?
        User: What's the time?
        Assistant: It's 9 o'clock.

    """

    NAME = "example_prompt_format"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"User: |message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"Assistant: |message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
