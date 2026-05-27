# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
        OUTPUT_ROLE: {
            "template": f"{NANO_BOT}assistant\n|message|{NANO_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

    def encode_dialog(self, turns: list[dict], enable_thinking: bool = True) -> dict[str, torch.Tensor]:
        """Encode a dialog for Nemotron Nano v3 with <think> reasoning support.

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
        turn_mask_values = []

        if self.INSERT_BOS:
            turn_tokens.append(self.tokenizer.bos)
            turn_token_counts.append(1)
            turn_mask_values.append(False)

        is_inference = turns[-1]["role"] != self.OUTPUT_ROLE
        for idx, turn in enumerate(turns):
            role = turn["role"]
            expected_slots = self.get_slots(role)
            slot_values = turn.get("slots", {})
            if expected_slots:
                self._validate_slot_values(expected_slots, slot_values)
            template = self.get_template(role)
            tokens = self.encode_turn(template, expected_slots, slot_values)
            turn_tokens.extend(tokens)
            turn_token_counts.append(len(tokens))
            # Loss mask only on the last assistant turn.
            turn_mask_values.append(role == self.OUTPUT_ROLE and idx == len(turns) - 1)

        # 7) Append inference prefix with thinking toggle.
        if is_inference and self.INFERENCE_PREFIX is not None:
            if enable_thinking:
                inference_prefix = self.INFERENCE_PREFIX + "<think>\n"
            else:
                inference_prefix = self.INFERENCE_PREFIX + "<think></think>"
            inference_tokens = self._apply_tokenizer(inference_prefix)
            turn_tokens.extend(inference_tokens)
            turn_token_counts.append(len(inference_tokens))
            turn_mask_values.append(False)

        # Insert EOS only when the last turn comes from the OUTPUT_ROLE.
        if self.INSERT_EOS and not is_inference:
            turn_tokens.append(self.tokenizer.eos)
            turn_token_counts[-1] += 1
            turn_mask_values.append(True)

        ans = {"input_ids": torch.tensor(turn_tokens, dtype=torch.long)}
        if turn_mask_values[-1]:
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
            ans["context_ids"] = ans["input_ids"]

        return ans


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
