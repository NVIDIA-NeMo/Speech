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
from typing import Callable, Type

import torch


PromptFormatFnReturnType = dict[str, list[torch.Tensor]]
PromptFormatSignature = Callable[[object, object], PromptFormatFnReturnType]
PROMPT_FORMAT_FNS: dict[tuple[Type, Type] | Type, PromptFormatSignature] = {}


def registered_prompt_format_fn(example_type: Type, formatter_type: Type = None):
    """
    Decorator for registering text prompt functions.
    It allows to select the right prompt formatting function based on the types of the
    example and the prompt formatter, allowing different strategies for formatting different
    types of data with different prompt formats.

    When formatter_type is set None, registers a default prompt format function for a given data type.

    Example::

        >>> @registered_prompt_format_fn(SourceTargetTextExample, Llama2PromptFormatter)
        ... def my_src_tgt_text_prompt(example, formatter):
        ...     pass
        ...
        ... @registered_prompt_format_fn(Cut, Llama2PromptFormatter)
        ... def my_audio_prompt(example, formatter):
        ...     pass
        ...
        ... prompt_fn = get_prompt_format_fn(SourceTargetTextExample, Llama2PromptFormatter)
    """

    def _decorator(prompt_fn: Callable[[object, object], dict[str, list[torch.Tensor]]]):
        global PROMPT_FORMAT_FNS
        if formatter_type is None:
            PROMPT_FORMAT_FNS[example_type] = prompt_fn
        else:
            PROMPT_FORMAT_FNS[(example_type, formatter_type)] = prompt_fn
        return prompt_fn

    return _decorator


def get_prompt_format_fn(example: Type | object, prompt: Type | object = None) -> PromptFormatSignature:
    """See the documentation of ``text_prompt_formatter`` above."""

    # If the user provided objects, resolve their types.
    if not isinstance(example, type):
        example = type(example)
    if not isinstance(prompt, type):
        prompt = type(prompt)

    # For the example type, first try to match it directly, then fall back to its parent classes.
    for example_subtype in example.mro():

        # First check the match for specific example type and a specific prompt format,
        # and all parent types of that specific prompt formatter type.
        for prompt_subtype in prompt.mro():
            if (example_subtype, prompt_subtype) in PROMPT_FORMAT_FNS:
                return PROMPT_FORMAT_FNS[(example_subtype, prompt_subtype)]

        # Then for the same specific example type, fall back to its default prompt formatter implementation.
        if example_subtype in PROMPT_FORMAT_FNS:
            return PROMPT_FORMAT_FNS[example_subtype]

    raise ValueError(
        f"Unknown prompt format function for ({example}, {prompt}). "
        f"Available choices are: {list(PROMPT_FORMAT_FNS.keys())}"
    )


def apply_prompt_format_fn(example: object | Type, prompt: object | Type) -> PromptFormatFnReturnType:
    """
    Utility for resolving the prompt format function and applying it to an example in one go.
    See the documentation of ``text_prompt_formatter`` above.
    """
    fn = get_prompt_format_fn(example, prompt)
    return fn(example, prompt)
