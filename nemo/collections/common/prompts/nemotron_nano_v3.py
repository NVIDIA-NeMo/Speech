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
# pylint: disable=C0116
import torch
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

NANO_BOT = "<|im_start|>"
NANO_EOT = "<|im_end|>"


class NemotronNanoV3PromptFormatter(PromptFormatter):
    NAME = "nemotron-nano-v3"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"{NANO_BOT}assistant\n"
    TEMPLATE = {
        "system": {
            "template": f"{NANO_BOT}system\n|message|{NANO_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{NANO_BOT}user\n|message|{NANO_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "tool": {
            "template": f"{NANO_BOT}tool\n|message|{NANO_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{NANO_BOT}assistant\n|message|{NANO_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

    def encode_dialog(self, turns: list[dict], enable_thinking: bool = True) -> dict[str, torch.Tensor]:
        """Encode a dialog for Nemotron Nano v3 with <think> reasoning support.

        Training dialogs supervise every assistant response and end-of-turn marker.
        Role headers, prefilled thinking tags, formatting after the end marker, and
        system/user/tool turns are masked out. Retained reasoning and its generated
        closing tag remain supervised. Thinking prefills are inferred per response;
        ``enable_thinking`` only controls the inference prefix.
        ``context_ids`` and ``answer_ids`` still split at the final assistant turn;
        inference dialogs return no loss mask. History thinking normalization is
        unchanged, so removed reasoning text is not supervised.

        Args:
            turns: List of turns with "role" and "slots"/"content" keys.
            enable_thinking: If True, inference prefix ends with ``<think>\\n``;
                if False, ends with ``<think></think>``.
        """
        roles = self.get_roles()
        assert len(turns) > 0, "Empty dialog is not supported."
        for turn in turns:
            assert "role" in turn, f"A turn must have a 'role' key. We received {turn=}"
            assert turn["role"] in roles, f"Found turn with {turn['role']=}, but available roles are {roles}"

        # 0) Normalize "content" → "slots" format.
        for turn in turns:
            if "content" in turn:
                turn["slots"] = {"message": turn.pop("content")}

        # 1) Auto-insert empty system turn if first turn isn't system.
        if turns[0]["role"] != "system":
            turns.insert(0, {"role": "system", "slots": {"message": ""}})

        # 2) Find last user turn index.
        last_user_idx = None
        for i, turn in enumerate(turns):
            if turn["role"] == "user":
                last_user_idx = i

        # 3) Normalize past assistant turns to mirror HF jinja's truncate_history_thinking:
        #    - Has both tags  → truncate to "<think></think>" + post-</think> content
        #    - Has neither    → prepend "<think></think>"
        #    - Has only one   → leave as-is (matches jinja)
        if last_user_idx is not None:
            for i, turn in enumerate(turns):
                if i < last_user_idx and turn["role"] == self.OUTPUT_ROLE:
                    msg = turn["slots"]["message"]
                    has_open = "<think>" in msg
                    has_close = "</think>" in msg
                    if has_open and has_close:
                        content_after_think = msg.split("</think>", 1)[1].strip()
                        turn["slots"]["message"] = "<think></think>" + content_after_think
                    elif not has_open and not has_close:
                        turn["slots"]["message"] = "<think></think>" + msg

        # 4) For last assistant turn (training): prepend <think></think> if no think tags.
        if turns[-1]["role"] == self.OUTPUT_ROLE:
            msg = turns[-1]["slots"]["message"].strip()
            if "<think>" not in msg:
                turns[-1]["slots"]["message"] = "<think></think>" + msg
            else:
                turns[-1]["slots"]["message"] = msg

        # 5) Strip all assistant content.
        for turn in turns:
            if turn["role"] == self.OUTPUT_ROLE:
                turn["slots"]["message"] = turn["slots"]["message"].strip()

        # 6) Tokenize all turns.
        turn_tokens = []
        turn_token_counts = []
        loss_mask = []

        if self.INSERT_BOS:
            turn_tokens.append(self.tokenizer.bos)
            turn_token_counts.append(1)
            loss_mask.append(False)

        is_inference = turns[-1]["role"] != self.OUTPUT_ROLE
        for turn in turns:
            role = turn["role"]
            expected_slots = self.get_slots(role)
            slot_values = turn.get("slots", {})
            if expected_slots:
                self._validate_slot_values(expected_slots, slot_values)
            template = self.get_template(role)
            tokens = self.encode_turn(template, expected_slots, slot_values)
            turn_tokens.extend(tokens)
            turn_token_counts.append(len(tokens))
            if not is_inference and role == self.OUTPUT_ROLE:
                loss_mask.extend(self._assistant_loss_mask(tokens, template, expected_slots, slot_values))
            else:
                loss_mask.extend([False] * len(tokens))

        # 7) Append inference prefix with thinking toggle.
        if is_inference and self.INFERENCE_PREFIX is not None:
            if enable_thinking:
                inference_prefix = self.INFERENCE_PREFIX + "<think>\n"
            else:
                inference_prefix = self.INFERENCE_PREFIX + "<think></think>"
            inference_tokens = self._apply_tokenizer(inference_prefix)
            turn_tokens.extend(inference_tokens)
            turn_token_counts.append(len(inference_tokens))
            loss_mask.extend([False] * len(inference_tokens))

        # Insert EOS only when the last turn comes from the OUTPUT_ROLE.
        if self.INSERT_EOS and not is_inference:
            turn_tokens.append(self.tokenizer.eos)
            turn_token_counts[-1] += 1
            loss_mask.append(True)

        ans = {"input_ids": torch.tensor(turn_tokens, dtype=torch.long)}
        if not is_inference:
            ans["context_ids"] = ans["input_ids"][: -turn_token_counts[-1]]
            ans["answer_ids"] = ans["input_ids"][-turn_token_counts[-1] :]
            ans["mask"] = torch.tensor(loss_mask, dtype=torch.bool)
        else:
            ans["context_ids"] = ans["input_ids"]

        return ans

    def _assistant_loss_mask(
        self, tokens: list[int], template: str, expected_slots: dict, slot_values: dict
    ) -> list[bool]:
        # The caller provides the assistant header at generation time. Match only
        # leading thinking prefills present in this response; never mask reasoning
        # or a closing </think> that the model must generate after reasoning.
        message = slot_values["message"]
        prefix = template.split("|message|", 1)[0]
        for thinking_prefix in ("<think></think>", "<think>\n", "<think>"):
            if message.startswith(thinking_prefix):
                prefix += thinking_prefix
                break
        prefix_tokens = self._apply_tokenizer(prefix, lang=slot_values.get(self.PROMPT_LANGUAGE_SLOT))
        start = _common_prefix_length(tokens, prefix_tokens)

        # Keep the EOT target but exclude formatting after it. Tokenize complete
        # turns as before: concatenating independently encoded prefix/body pieces
        # would change BPE segmentation. If a boundary merges tokens, supervise
        # the ambiguous token(s) rather than dropping any response/EOT targets.
        through_eot = self.encode_turn(template.removesuffix("\n"), expected_slots, slot_values)
        end = _common_prefix_length(tokens, through_eot)
        if end < len(through_eot):
            end = len(tokens)
        return [False] * start + [True] * (end - start) + [False] * (len(tokens) - end)


@registered_prompt_format_fn(Cut, NemotronNanoV3PromptFormatter)
def nemotron_nano_v3(cut: Cut, prompt: NemotronNanoV3PromptFormatter):
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut

    turns = []

    system = ""
    if cut.has_custom("system_prompt"):
        system = cut.system_prompt
    turns.append({"role": "system", "content": system})

    if cut.has_custom("context"):
        ctx = cut.context
    else:
        ctx = ""
    turns.append({"role": "user", "content": ctx})

    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "content": answer})

    return prompt.encode_dialog(turns)


def _common_prefix_length(tokens: list[int], prefix: list[int]) -> int:
    for i, (token, prefix_token) in enumerate(zip(tokens, prefix)):
        if token != prefix_token:
            return i
    return min(len(tokens), len(prefix))
