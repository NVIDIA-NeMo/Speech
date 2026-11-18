# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Language-ID prompt resolution shared by prompt-conditioned ASR models, datasets and pipelines."""

from typing import Any, Mapping, Optional, Tuple

from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

__all__ = [
    'AUTO_PROMPT_LANGUAGE',
    'resolve_prompt_language',
    'resolve_prompt_index',
    'resolve_deprecated_prompt_arguments',
]

# The language-agnostic prompt, used when no language is given.
AUTO_PROMPT_LANGUAGE = "auto"


def resolve_prompt_language(language: Optional[str], prompt_dictionary: Mapping[str, int]) -> str:
    """Map a language name to its key in a model's prompt dictionary.

    Resolution rules:
        * ``None`` selects the language-agnostic ``"auto"`` prompt.
        * An exact key (``"de-DE"``, ``"de"``) is used as is.
        * A bare language code (``"de"``) matches a locale key (``"de-DE"``) only when exactly one
          key has that code.
        * Anything else raises, so an unsupported language is never silently replaced.

    Args:
        language: Language name, or None for the language-agnostic prompt.
        prompt_dictionary: Mapping from language name to prompt index.

    Returns:
        (str) A key of ``prompt_dictionary``.

    Raises:
        ValueError: If the dictionary is empty, or the language is unsupported or ambiguous.
    """
    if not prompt_dictionary:
        raise ValueError("The model's prompt dictionary is empty, so no language prompt can be selected.")

    if language is None:
        if AUTO_PROMPT_LANGUAGE not in prompt_dictionary:
            raise ValueError(
                f"No language was given and this model has no '{AUTO_PROMPT_LANGUAGE}' prompt. "
                f"Pass one of the supported languages: {_format_languages(prompt_dictionary)}"
            )
        return AUTO_PROMPT_LANGUAGE

    if language in prompt_dictionary:
        return language

    prefix = f"{language.lower()}-"
    matches = sorted(key for key in prompt_dictionary if key.lower().startswith(prefix))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Language '{language}' is ambiguous for this model: it matches {matches}. Pass one of them.")

    message = f"Language '{language}' is not supported by this model. Supported languages: " + _format_languages(
        prompt_dictionary
    )
    if AUTO_PROMPT_LANGUAGE in prompt_dictionary:
        message += (
            f". You can try the language-agnostic '{AUTO_PROMPT_LANGUAGE}' prompt instead, "
            "but its performance on unsupported languages is not guaranteed."
        )
    raise ValueError(message)


def resolve_prompt_index(language: Optional[str], prompt_dictionary: Mapping[str, int]) -> int:
    """Map a language name to its prompt index; see :func:`resolve_prompt_language` for the rules."""
    return prompt_dictionary[resolve_prompt_language(language, prompt_dictionary)]


def resolve_deprecated_prompt_arguments(
    language_prompt_features: Any, language_prompt_indices: Any, prompt: Any, prompt_indices: Any
) -> Tuple[Any, Any]:
    """Fold the deprecated ``prompt`` / ``prompt_indices`` forward arguments into their new names.

    Args:
        language_prompt_features: Value passed as ``language_prompt_features``.
        language_prompt_indices: Value passed as ``language_prompt_indices``.
        prompt: Value passed as the deprecated ``prompt``.
        prompt_indices: Value passed as the deprecated ``prompt_indices``.

    Returns:
        (tuple) ``(language_prompt_features, language_prompt_indices)``.

    Raises:
        ValueError: If a deprecated argument and its replacement are both given.
    """
    return (
        _resolve_deprecated_argument(language_prompt_features, 'language_prompt_features', prompt, 'prompt'),
        _resolve_deprecated_argument(
            language_prompt_indices, 'language_prompt_indices', prompt_indices, 'prompt_indices'
        ),
    )


def _resolve_deprecated_argument(value: Any, name: str, deprecated_value: Any, deprecated_name: str) -> Any:
    if deprecated_value is None:
        return value
    if value is not None:
        raise ValueError(f"Pass either `{name}` or the deprecated `{deprecated_name}`, not both.")
    logging.warning(
        f"The `{deprecated_name}` argument of forward() is deprecated and will be removed in a future release. "
        f"Use `{name}` instead.",
        mode=LogMode.ONCE,
    )
    return deprecated_value


def _format_languages(prompt_dictionary: Mapping[str, int]) -> str:
    return ", ".join(sorted(key for key in prompt_dictionary if key != AUTO_PROMPT_LANGUAGE))
