# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
# pylint: disable=C0115
import random

import torch
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

QWEN_BOT = "<|im_start|>"
QWEN_EOT = "<|im_end|>"


class QwenPromptFormatter(PromptFormatter):
    NAME = "qwen"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"{QWEN_BOT}assistant\n"
    TEMPLATE = {
        "user": {
            "template": f"{QWEN_BOT}user\n|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


class Qwen3PromptFormatter(PromptFormatter):
    NAME = "qwen3"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"{QWEN_BOT}assistant\n"
    NO_THINK_PREFIX = "<think>\n\n</think>\n\n"
    TEMPLATE = {
        "system": {
            "template": f"{QWEN_BOT}system\n|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{QWEN_BOT}user\n|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

    def encode_dialog(self, turns: list[dict], enable_thinking: bool = True) -> dict[str, torch.Tensor]:
        """Overrides the base class method.

        Args:
            turns (list[dict]): List of turns. Each turn is a dict with "role" and "slots" keys.
            enable_thinking (bool): Whether to enable thinking. If False, an empty thinking block will be added.
        """

        roles = self.get_roles()
        assert len(turns) > 0, "Empty dialog is not supported."
        for turn in turns:
            assert "role" in turn, f"A turn must have a 'role' key. We received {turn=}"
            assert turn["role"] in roles, f"Found turn with {turn['role']=}, but available roles are {roles}"

        # Preprocess turns based on Qwen3 prompt format
        # Our training data format:
        # - The final assistant turn can have: (1) no thinking tags, (2) empty thinking, (3) non-empty thinking
        # - System and user turns do not have "/think" or "/no_think" tags
        # - There can be an empty system turn at the beginning
        # Our inference data format:
        # - System and user turns can have "/think" or "/no_think" tags

        # 0) Unify the format of turns to have "role" and "slots" keys.
        for turn in turns:
            if "content" in turn:
                turn["slots"] = {"message": turn.pop("content")}

        # 1) (Inference, Optional) Determine if thinking is enabled in user or system turns.
        # If multiple turns have the tag, we will use the last one.
        # enable_thinking = True  # By default, it is enabled according to Qwen3 prompt format
        # for turn in turns:
        #     if turn["role"] == "user" or turn["role"] == "system":
        #         if "/think" in turn["slots"]["message"]:
        #             enable_thinking = True
        #         elif "/no_think" in turn["slots"]["message"]:
        #             enable_thinking = False

        # 2) (Training and Inference) Remove thinking content from previous turns.
        for turn in turns[:-1]:
            if turn["role"] == self.OUTPUT_ROLE:
                if "</think>" in turn["slots"]["message"]:
                    turn["slots"]["message"] = turn["slots"]["message"].split("</think>")[1].strip()

        # 3) (Training) Handle the thinking content of the last assistant turn.
        # Also normalize the thinking format:
        # <think>\n" + reasoning_content.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
        # Find all system and user turns
        system_and_user_turns = [turn for turn in turns if turn["role"] == "system" or turn["role"] == "user"]
        turn = turns[-1]
        if turn["role"] == self.OUTPUT_ROLE:
            if "<think>" not in turn["slots"]["message"]:
                assert "</think>" not in turn["slots"]["message"], turn["slots"]["message"]
                # Add empty thinking block
                turn["slots"]["message"] = self.NO_THINK_PREFIX + turn["slots"]["message"]
                # A simplified version: add only one "/no_think" to a previous user or system turn
                self._inject_thinking_tag(system_and_user_turns, "/no_think")
            else:
                assert turn["slots"]["message"].startswith("<think>"), turn["slots"]["message"]
                assert "</think>" in turn["slots"]["message"], turn["slots"]["message"]
                reasoning_content = turn["slots"]["message"].split("<think>")[1].split("</think>")[0].strip()
                content = turn["slots"]["message"].split("</think>")[1].lstrip("\n")
                turn["slots"]["message"] = f"<think>\n{reasoning_content}\n</think>\n\n{content}"

                if not reasoning_content:
                    # A simplified version: add only one "/no_think" to a previous user or system turn
                    self._inject_thinking_tag(system_and_user_turns, "/no_think")
                else:
                    # Add "/think" or nothing
                    if random.random() < 0.5:
                        self._inject_thinking_tag(system_and_user_turns, "/think")

        # 4) (Training and Inference) Remove empty system turn
        if turns[0]["role"] == "system" and turns[0]["slots"]["message"].strip() == "":
            turns = turns[1:]

        turn_tokens = []
        turn_token_counts = []
        turn_mask_values = []

        if self.INSERT_BOS:
            turn_tokens.append(self.tokenizer.bos)
            turn_token_counts.append(1)
            turn_mask_values.append(False)

        is_inference = turns[-1]["role"] != self.OUTPUT_ROLE
        for idx, turn in enumerate(turns):
            role = turn["role"]
            expected_slots = self.get_slots(role)
            if "content" in turn and len(expected_slots) == 1:
                # User is leveraging the "standard" API prompting LLM; we'll map "content" value
                # to whatever is the name of the slot, when there's only one slot.
                slot_values = {k: turn["content"] for k in expected_slots.keys()}  # 1-item dict
            else:
                slot_values = turn.get("slots", {})
                if expected_slots:
                    assert slot_values, (
                        f"A turn for role {role} must have have a non-empty value under 'slots' key. "
                        f"We received {turn=}"
                    )
                    self._validate_slot_values(expected_slots, slot_values)
            template = self.get_template(role)
            tokens = self.encode_turn(template, expected_slots, slot_values)
            turn_tokens.extend(tokens)
            turn_token_counts.append(len(tokens))
            # Set loss mask as True only for the last assistant turn.
            turn_mask_values.append(role == self.OUTPUT_ROLE and idx == len(turns) - 1)

        if is_inference and self.INFERENCE_PREFIX is not None:
            inference_prefix = self._apply_tokenizer(self._inference_prefix(enable_thinking))
            turn_tokens.extend(inference_prefix)
            turn_token_counts.append(len(inference_prefix))
            turn_mask_values.append(False)  # not a training example

        # Insert EOS only when the last turn comes from the OUTPUT_ROLE.
        if self.INSERT_EOS and not is_inference:
            turn_tokens.append(self.tokenizer.eos)
            turn_token_counts[-1] += 1
            turn_mask_values.append(True)

        ans = {"input_ids": torch.tensor(turn_tokens, dtype=torch.long)}
        if turn_mask_values[-1]:
            # The last turn comes from OUTPUT_ROLE, i.e. it's a response from the system.
            # This indicates it's a training example for which we provide context/answer/mask.
            ans["context_ids"] = ans["input_ids"][: -turn_token_counts[-1]]
            ans["answer_ids"] = ans["input_ids"][-turn_token_counts[-1] :]
            ans["mask"] = torch.tensor(
                [
                    turn_mask_values[turn_idx]
                    for turn_idx, turn_len in enumerate(turn_token_counts)
                    for _ in range(turn_len)
                ],
                dtype=torch.bool,
            )
        else:
            ans["context_ids"] = ans["input_ids"]  # context == input for inference

        return ans

    def _inject_thinking_tag(self, turns: list[dict], tag: str) -> None:
        """Attach a ``/think`` or ``/no_think`` marker to a random system or user turn."""
        turn = random.choice(turns)
        if random.random() < 0.5:
            turn["slots"]["message"] = turn["slots"]["message"] + f" {tag}"
        else:
            turn["slots"]["message"] = f"{tag} " + turn["slots"]["message"]

    def _inference_prefix(self, enable_thinking: bool) -> str:
        """Return the generation prompt appended after the last non-assistant turn."""
        if enable_thinking:
            return self.INFERENCE_PREFIX
        return self.INFERENCE_PREFIX + self.NO_THINK_PREFIX


class Qwen3_5PromptFormatter(Qwen3PromptFormatter):
    """Qwen3.5 shares Qwen3's ChatML templates but controls reasoning differently.

    Its chat template has no ``/think`` / ``/no_think`` markers — reasoning is selected purely by
    the generation prompt, which always opens a thinking block: ``<think>\\n`` to reason, or an
    immediately closed block to skip it.
    """

    NAME = "qwen3.5"
    THINK_PREFIX = "<think>\n"

    def encode_dialog(self, turns: list[dict], enable_thinking: bool = False) -> dict[str, torch.Tensor]:
        """Overrides the base class method to default to non-thinking.

        Qwen3.5 gates reasoning on the generation prompt: ``<think>\\n`` opens a reasoning block,
        an immediately closed one skips it. Its chat template opens the block only when
        ``enable_thinking`` is explicitly true, and training targets carry a closed block, so
        thinking is off unless the caller asks for it.
        """
        return super().encode_dialog(turns, enable_thinking=enable_thinking)

    def _inject_thinking_tag(self, turns: list[dict], tag: str) -> None:
        """Disable Qwen3's tag injection: Qwen3.5's template defines no such markers."""

    def _inference_prefix(self, enable_thinking: bool) -> str:
        return self.INFERENCE_PREFIX + (self.THINK_PREFIX if enable_thinking else self.NO_THINK_PREFIX)
