# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import re
from typing import Optional, Union

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger

from nemo.collections.asr.parts.utils.eval_utils import clean_label, remove_punctuations


def match_str_and_float(
    ref_value: Union[str, float],
    pred_value: Union[str, float],
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Match the reference and prediction value.

    Args:
        ref_value: The reference value, can be a string or a float.
        pred_value: The prediction value, can be a string or a float.
        ignore_capitalization: Whether to ignore capitalization when comparing strings.
        ignore_punctuation: Whether to ignore punctuation when comparing strings.
        clean_text: Whether to clean the text by replacing special characters before comparing.
    Returns:
        True if the reference and prediction value match, False otherwise.
    """
    try:
        # try to convert to float for input like "1.0"
        ref_value = float(ref_value)
        pred_value = float(pred_value)
        is_string = False
    except Exception:
        is_string = True

    if is_string:
        ref_value = str(ref_value)
        pred_value = str(pred_value)
        logger.debug(f"before processing: ref_value: {ref_value}, pred_value: {pred_value}")
        if ignore_capitalization:
            ref_value = ref_value.lower()
            pred_value = pred_value.lower()
        if ignore_punctuation:
            ref_value = remove_punctuations(ref_value)
            pred_value = remove_punctuations(pred_value)
        if clean_text:
            ref_value = clean_label(ref_value, langid="en", num_to_words=False, lowercase=ignore_capitalization)
            pred_value = clean_label(pred_value, langid="en", num_to_words=False, lowercase=ignore_capitalization)
        logger.debug(f"after processing: ref_value: {ref_value}, pred_value: {pred_value}")
        return ref_value == pred_value
    else:
        try:
            is_close = np.isclose(ref_value, pred_value)
            logger.debug(f"ref_value: {ref_value}, pred_value: {pred_value}")
            if isinstance(is_close, np.ndarray):
                is_close = all(is_close)
            return bool(is_close)
        except Exception as e:
            logger.error(f"Error checking for np.isclose(ref_value: {ref_value}, pred_value: {pred_value}): {e}")
            return False


def match_item(
    ref_value,
    pred_value,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Recursively match a reference value against a prediction value.
    Handles dicts, lists, strings, and numbers.
    """
    if isinstance(ref_value, dict):
        if not isinstance(pred_value, dict):
            return False
        return match_dict(
            ref_value,
            pred_value,
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        )
    elif isinstance(ref_value, list):
        if not isinstance(pred_value, list):
            return False
        return match_list(
            ref_value,
            pred_value,
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        )
    else:
        return match_str_and_float(
            ref_value,
            pred_value,
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        )


def match_dict(
    ref_dict: dict,
    pred_dict: dict,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Check if pred_dict contains all keys and matching values from ref_dict.
    Additional keys in pred_dict are allowed.
    """
    for key, ref_val in ref_dict.items():
        if key not in pred_dict:
            return False
        if not match_item(
            ref_val,
            pred_dict[key],
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        ):
            return False
    return True


def match_list(
    ref_list: list,
    pred_list: list,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Check if each item in ref_list has a matching item in pred_list (order-independent).
    Each prediction item can only be matched once.
    """
    matched_indices = set()
    for ref_item in ref_list:
        found = False
        for i, pred_item in enumerate(pred_list):
            if i in matched_indices:
                continue
            if match_item(
                ref_item,
                pred_item,
                ignore_capitalization=ignore_capitalization,
                ignore_punctuation=ignore_punctuation,
                clean_text=clean_text,
            ):
                matched_indices.add(i)
                found = True
                break
        if not found:
            return False
    return True


def check_if_task_success(
    *,
    reference: str,
    prediction: str,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Check if the prediction is matches with the reference answer.

    Situations:
    1. If the reference is a dictionary, and the prediction is a dictionary:
      - The prediction should have the same keys and values as the reference.
      - Additional keys in prediction are allowed.

    2. If the reference is a dictionary, and the prediction is a list of dictionaries:
      -  the last dictionary in the prediction would be matched with the reference.

    3. If the reference is a list of dictionaries, and the prediction is a list of dictionaries:
      - For each dictionary in the reference, there should be a dictionary in the prediction that matches it
        according to the criteria in Situation 1.
      - The order of the dictionaries in the reference/prediction is not important.
      - All dictionaries in the reference should be matched with a dictionary in the prediction to be considered as a success.

    Args:
        reference: The path to the reference json file.
        prediction: The path to the prediction json file.
        ignore_capitalization: Whether to ignore case when comparing strings.
        ignore_punctuation: Whether to ignore punctuation when comparing strings.
        clean_text: Whether to clean the text before comparing.
    Returns:
        True if the task is considered as successful, False otherwise.
    """
    with open(reference, "r") as f:
        reference_answer = json.load(f)
    with open(prediction, "r") as f:
        prediction_answer = json.load(f)

    # Situation 1: If the reference is a dictionary, and the prediction is a dictionary,
    # Convert to Situation 3
    if isinstance(reference_answer, dict):
        reference_answer = [reference_answer]
    if isinstance(prediction_answer, dict):
        prediction_answer = [prediction_answer]

    # Situation 2: If the reference is a dictionary, and the prediction is a list of dictionaries,
    # the last dictionary in the prediction would be matched with the reference.
    # Convert to Situation 3
    if len(reference_answer) == 1 and len(prediction_answer) > 1:
        prediction_answer = [prediction_answer[-1]]

    logger.debug(f"reference_answer: {reference_answer}")
    logger.debug(f"prediction_answer: {prediction_answer}")
    result = True
    # Situation 3: For each reference dict, find a matching prediction dict (order-independent).
    matched_indices = set()
    for ref_dict in reference_answer:
        found = False
        for i, pred_dict in enumerate(prediction_answer):
            if i in matched_indices:
                continue
            if match_dict(
                ref_dict,
                pred_dict,
                ignore_capitalization=ignore_capitalization,
                ignore_punctuation=ignore_punctuation,
                clean_text=clean_text,
            ):
                matched_indices.add(i)
                found = True
                break
        if not found:
            result = False
            break
    logger.debug(f"success: {result}")
    return result


class LLMJudge:
    """
    LLM-based judge for evaluating voice agent responses.

    Uses an OpenAI-compatible chat completions API to score how well a prediction
    matches a reference answer. Returns a float score between 0 and 1.

    Args:
        url: The URL of the OpenAI-compatible chat completions endpoint.
        model: The model name to use for judging.
        api_key: The API key. If None, will be loaded from environment variable.
        api_key_name: The environment variable name for the API key (default: "API_KEY").
        default_prompt: Custom default system prompt. If None, uses DEFAULT_PROMPT.
        **kwargs: Additional keyword arguments passed to the API payload (e.g., temperature, max_tokens).
    """

    DEFAULT_PROMPT = """You are a judge that evaluates the similarity between a reference answer and a prediction.
You will be given a reference and a prediction wrapped in XML tags.
Judge how well the prediction matches the reference in terms of correctness and completeness.
Return a score between 0 and 1, where 0 means completely wrong and 1 means a perfect match.
You MUST return ONLY a JSON object in the following format, with no other text:
{"score": <score>, "reason": "<brief explanation>"}"""

    SCENARIO_PROMPT = """You are a judge that evaluates voice agent performance in a conversational scenario.
You will be given:
- A reference answer (the expected outcome)
- A prediction (the actual agent output)
- The full conversation transcript between the user and the agent
- The LLM context history, which includes tool/function calls made by the agent

Evaluate how well the agent performed by considering:
1. Whether the prediction matches the reference answer
2. Whether the agent followed instructions correctly during the conversation
3. Whether the agent called the correct tools with the correct arguments at the right time
4. Whether the agent avoided unnecessary or incorrect tool calls
5. Whether the agent handled the conversation naturally and helpfully

Return a score between 0 and 1, where 0 means complete failure and 1 means perfect performance.
You MUST return ONLY a JSON object in the following format, with no other text:
{"score": <score>, "reason": "<brief explanation>"}"""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_name: str = "API_KEY",
        default_prompt: Optional[str] = None,
        **kwargs,
    ):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.api_key_name = api_key_name
        if self.api_key is None:
            load_dotenv(override=True)
            self.api_key = os.getenv(self.api_key_name)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.default_prompt = default_prompt or self.DEFAULT_PROMPT
        self.kwargs = kwargs

    def _get_payload(self, user_content: str, prompt: Optional[str] = None) -> dict:
        if not prompt:
            prompt = self.default_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            **self.kwargs,
        }
        return payload

    def _parse_response(self, response: requests.Response) -> dict:
        """
        Parse the LLM response and extract the judgement JSON.

        Args:
            response: The HTTP response from the API.
        Returns:
            A dict with "score" (float) and optionally "reason" (str).
        Raises:
            ValueError: If the response cannot be parsed.
        """
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Try to parse JSON directly
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Try to find any JSON object in the content
        match = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', content)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"Could not parse judgement JSON from LLM response: {content}")

    def judge(self, reference: str, prediction: str, prompt: Optional[str] = None) -> dict:
        """
        Judge the similarity between a reference and a prediction.

        Args:
            reference: The reference answer string.
            prediction: The prediction answer string.
            prompt: Optional custom system prompt. Uses default_prompt if not provided.
        Returns:
            A dict with "score" (float between 0 and 1) and "reason" (str).
            On error, returns {"score": 0.0, "reason": "<error message>"}.
        """
        user_content = f"<reference>\n{reference}\n</reference>\n\n<prediction>\n{prediction}\n</prediction>"
        payload = self._get_payload(user_content, prompt)
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            result = self._parse_response(response)
            result["score"] = float(result["score"])
            if "reason" not in result:
                result["reason"] = ""
            logger.debug(f"LLMJudge result: {result}")
            return result
        except Exception as e:
            logger.error(f"LLMJudge error: {e}")
            return {"score": 0.0, "reason": f"Error: {e}"}

    def judge_file(self, reference: str, prediction: str, prompt: Optional[str] = None) -> dict:
        """
        Judge the similarity between a reference file and a prediction file.

        Args:
            reference: Path to the reference JSON file.
            prediction: Path to the prediction JSON file.
            prompt: Optional custom system prompt.
        Returns:
            A dict with "score" (float between 0 and 1) and "reason" (str).
        """
        with open(reference, "r") as f:
            reference_content = f.read()
        with open(prediction, "r") as f:
            prediction_content = f.read()
        return self.judge(reference_content, prediction_content, prompt)

    def judge_scenario(
        self,
        reference: str,
        prediction: str,
        conversation: Optional[list] = None,
        context_history: Optional[list] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Judge agent performance with full scenario context including conversation history.

        Args:
            reference: The reference answer string (or JSON string).
            prediction: The prediction answer string (or JSON string).
            conversation: List of conversation turns, each a dict with "role" and "text" keys.
            context_history: LLM context messages (from _retrieve_context_history).
            prompt: Optional custom system prompt. Uses SCENARIO_PROMPT if not provided.
        Returns:
            A dict with "score" (float between 0 and 1) and "reason" (str).
        """
        if not prompt:
            prompt = self.SCENARIO_PROMPT

        sections = [
            f"<reference>\n{reference}\n</reference>",
            f"<prediction>\n{prediction}\n</prediction>",
        ]

        if conversation:
            turns_text = "\n".join(f"[{turn.get('role', 'unknown')}]: {turn.get('text', '')}" for turn in conversation)
            sections.append(f"<conversation>\n{turns_text}\n</conversation>")

        if context_history:
            sections.append(f"<context_history>\n{json.dumps(context_history, indent=2)}\n</context_history>")

        user_content = "\n\n".join(sections)
        payload = self._get_payload(user_content, prompt)
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            result = self._parse_response(response)
            result["score"] = float(result["score"])
            result.setdefault("reason", "")
            return result
        except Exception as e:
            logger.error(f"LLMJudge error: {e}")
            return {"score": 0.0, "reason": f"Error: {e}"}
