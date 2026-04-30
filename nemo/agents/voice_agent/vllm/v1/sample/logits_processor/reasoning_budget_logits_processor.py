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

"""vLLM V1 LogitsProcessor that enforces a per-request reasoning (thinking) token budget.

Models like Nemotron-Nano use ``<think>``/``</think>`` delimiters (same convention as
DeepSeek-R1).  This processor monitors generated tokens, counts those inside the
thinking block, and forces end tokens when the budget is reached.

Per-request parameters (via ``SamplingParams.extra_args``):

    thinking_budget (int):
        Maximum number of thinking tokens allowed before forcing the end
        sequence.  Required to activate the processor for a given request.

    thinking_budget_grace_period (int, optional):
        Number of tokens *before* the budget at which ``\\n`` and end-token
        logits start being boosted.  Defaults to 10 % of ``thinking_budget``.

    think_start_tokens (str, optional):
        Text that marks the beginning of a thinking block.  The processor
        tokenizes this string at request time.  Defaults to ``"<think>"``.

    think_end_tokens (str, optional):
        Text to force when the budget is reached.  The processor tokenizes
        this string at request time.  Defaults to ``"\\n</think>"``.  Can be
        set to a custom closing such as ``"Reached thinking limit.\\n</think>"``.

Usage — offline with ``vllm.LLM``::

    llm = LLM(model=model, logits_processors=[ReasoningBudgetLogitsProcessor], ...)
    params = SamplingParams(
        temperature=0.6, max_tokens=256,
        extra_args={"thinking_budget": 64},
    )
    outputs = llm.generate(prompts, params)

Usage — online with ``vllm serve``::

    vllm serve <model> \\
        --logits-processors '[".../reasoning_budget_logits_processor:ReasoningBudgetLogitsProcessor"]'

    # then per-request via the OpenAI client:
    extra_body={"vllm_xargs": {"thinking_budget": 64}}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Default grace-period ratio when ``thinking_budget_grace_period`` is not set.
_DEFAULT_GRACE_RATIO = 0.10
# Additive logit boost applied during the grace period.
_GRACE_LOGIT_BOOST = 5.0


@dataclass
class RequestState:
    """Mutable per-request tracking state."""

    max_thinking_tokens: int
    grace_period: int
    # Token IDs used to *detect* thinking boundaries in the output stream.
    detect_start_ids: list[int]
    detect_end_ids: list[int]
    # Token IDs *forced* into the output when the budget is reached.
    # May differ from detect_end_ids (e.g. "\n</think>" vs "</think>").
    force_end_ids: list[int]
    thinking_token_count: int = 0
    inside_thinking: bool = False
    stopped_thinking: bool = False
    # Index into ``force_end_ids`` for multi-token forcing.
    # -1 means "not currently forcing end tokens".
    forcing_end_idx: int = -1
    # Live reference to the request's growing output token list.
    output_tok_ids: list[int] = field(default_factory=list)
    # How many output tokens we had already inspected on the previous step.
    prev_output_length: int = 0


class ReasoningBudgetLogitsProcessor(LogitsProcessor):
    """Enforce a per-request thinking-token budget for reasoning models.

    The processor tracks thinking-start / thinking-end boundaries in each
    request's output.  Once a request's thinking token count enters the
    grace window, ``\\n`` and end-token logits are boosted.  At the hard
    limit all logits except the next forced end token are set to ``-inf``.
    """

    # ------------------------------------------------------------------
    # Construction & validation
    # ------------------------------------------------------------------

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self.pin_memory = is_pin_memory

        # Tokenize the delimiter strings once.  Keep the tokenizer for
        # encoding per-request ``think_start_tokens`` / ``think_end_tokens``
        # strings later.
        self.tokenizer = self._load_tokenizer(vllm_config)
        # Detection patterns — bare delimiters for scanning output tokens.
        self.think_start_ids: list[int] = self._encode(self.tokenizer, "<think>")
        self.think_end_detect_ids: list[int] = self._encode(self.tokenizer, "</think>")
        # Default forcing sequence — what gets injected at budget cutoff.
        self.think_end_force_ids: list[int] = self._encode(self.tokenizer, "\n</think>")
        self.newline_ids: list[int] = self._encode(self.tokenizer, "\n")

        logger.info(
            f"think_start_ids={self.think_start_ids}, "
            f"think_end_detect_ids={self.think_end_detect_ids}, "
            f"think_end_force_ids={self.think_end_force_ids}"
        )

        # Sparse dict: batch-index → RequestState (only for requests with a budget).
        self.req_states: dict[int, RequestState] = {}

        self.neg_inf = torch.tensor(-float("inf"), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tokenizer(vllm_config: "VllmConfig"):
        """Obtain a tokenizer from the vLLM config."""
        model_cfg = vllm_config.model_config
        from vllm.transformers_utils.tokenizer import get_tokenizer

        return get_tokenizer(
            model_cfg.tokenizer,
            trust_remote_code=model_cfg.trust_remote_code,
            revision=model_cfg.tokenizer_revision,
        )

    @staticmethod
    def _encode(tokenizer, text: str) -> list[int]:
        """Encode *text* without special tokens."""
        return tokenizer.encode(text, add_special_tokens=False)

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(data, device="cpu", dtype=dtype, pin_memory=self.pin_memory).to(
            device=self.device, non_blocking=True
        )

    # ------------------------------------------------------------------
    # LogitsProcessor interface
    # ------------------------------------------------------------------

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate thinking-budget-related extra_args on the provided sampling params."""
        if sampling_params.extra_args is None:
            return
        budget = sampling_params.extra_args.get("thinking_budget")
        if budget is None:
            return
        if not isinstance(budget, int) or budget < 0:
            raise ValueError(f"thinking_budget must be a non-negative int, got {budget!r}")
        grace = sampling_params.extra_args.get("thinking_budget_grace_period")
        if grace is not None and (not isinstance(grace, int) or grace < 0):
            raise ValueError(f"thinking_budget_grace_period must be a non-negative int, got {grace!r}")
        for key in ("think_start_tokens", "think_end_tokens"):
            val = sampling_params.extra_args.get(key)
            if val is not None and not isinstance(val, str):
                raise ValueError(f"{key} must be a string, got {val!r}")

    def is_argmax_invariant(self) -> bool:
        """Return whether this processor preserves argmax behavior (it does not)."""
        # This processor forces specific tokens, changing the argmax outcome.
        return False

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_ends_with(
        prompt_tok_ids: list[int] | None,
        pattern: list[int],
        skip_ids: list[int],
    ) -> bool:
        """Return True if *prompt_tok_ids* ends with *pattern*, ignoring
        any trailing tokens whose ID is in *skip_ids* (e.g. newlines).
        """
        if not prompt_tok_ids:
            return False
        idx = len(prompt_tok_ids)
        skip_set = set(skip_ids)
        while idx > 0 and prompt_tok_ids[idx - 1] in skip_set:
            idx -= 1
        plen = len(pattern)
        return idx >= plen and prompt_tok_ids[idx - plen : idx] == pattern

    def _new_state(
        self,
        params: SamplingParams,
        prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> RequestState | None:
        """Called by ``process_dict_updates`` for each newly added request."""
        if params.extra_args is None:
            return None
        budget = params.extra_args.get("thinking_budget")
        if budget is None or budget <= 0:
            return None

        grace = params.extra_args.get("thinking_budget_grace_period")
        if grace is None:
            grace = max(1, int(budget * _DEFAULT_GRACE_RATIO))
        elif grace < 1.0:
            # if grace is a percentage, convert it to an integer
            grace = max(1, int(budget * grace))
        else:
            # make sure grace is an integer
            grace = max(1, int(grace))

        # ensure grace is not greater than the budget
        if grace > budget:
            logger.warning(
                f"thinking_budget_grace_period={grace} is greater than the "
                f"thinking_budget={budget}, setting it to thinking_budget."
            )
            grace = budget

        # Per-request start/end token overrides.
        think_start_str = params.extra_args.get("think_start_tokens")
        if think_start_str is not None:
            detect_start_ids = self._encode(self.tokenizer, think_start_str)
        else:
            detect_start_ids = list(self.think_start_ids)

        think_end_str = params.extra_args.get("think_end_tokens")
        if think_end_str is not None:
            force_end_ids = self._encode(self.tokenizer, think_end_str)
        else:
            force_end_ids = list(self.think_end_force_ids)

        # Detection always uses the bare </think> delimiter so it works
        # regardless of what token precedes it (e.g. ".\n</think>" where
        # ".\n" is merged into a single token by the tokenizer).
        detect_end_ids = list(self.think_end_detect_ids)

        state = RequestState(
            max_thinking_tokens=budget,
            grace_period=grace,
            detect_start_ids=detect_start_ids,
            detect_end_ids=detect_end_ids,
            force_end_ids=force_end_ids,
            output_tok_ids=output_tok_ids,
        )

        # If the prompt already ends with <think>, the model is generating
        # inside a thinking block from the very first output token.
        if self._prompt_ends_with(prompt_tok_ids, detect_start_ids, self.newline_ids):
            state.inside_thinking = True

        # Catch up on any tokens already generated.
        self._scan_tokens(state, from_idx=0)
        state.prev_output_length = len(output_tok_ids)
        return state

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """Sync per-request thinking state with the batch update from vLLM."""
        process_dict_updates(self.req_states, batch_update, self._new_state)

        if not self.req_states:
            return

        to_remove: list[int] = []
        for idx, state in self.req_states.items():
            # Incrementally scan newly generated tokens.
            self._scan_tokens(state, from_idx=state.prev_output_length)
            state.prev_output_length = len(state.output_tok_ids)

            # If the model already emitted </think> naturally, stop tracking.
            if state.stopped_thinking:
                to_remove.append(idx)

        for idx in to_remove:
            del self.req_states[idx]

    # ------------------------------------------------------------------
    # Token scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_tokens(state: RequestState, from_idx: int) -> None:
        """Update ``state`` by scanning ``output_tok_ids[from_idx:]``."""
        toks = state.output_tok_ids
        start_ids = state.detect_start_ids
        end_ids = state.detect_end_ids
        start_len = len(start_ids)
        end_len = len(end_ids)

        for i in range(from_idx, len(toks)):
            end = i + 1

            # Check for think-start sequence ending at position i.
            if end >= start_len and toks[end - start_len : end] == start_ids:
                if not state.stopped_thinking:
                    state.inside_thinking = True
                continue  # delimiter token does not count toward budget

            # Check for think-end sequence ending at position i.
            if end >= end_len and toks[end - end_len : end] == end_ids:
                if state.inside_thinking:
                    state.inside_thinking = False
                    state.stopped_thinking = True
                    state.forcing_end_idx = -1
                    return
                continue  # delimiter token does not count toward budget

            # Count thinking tokens (excludes delimiter tokens).
            if state.inside_thinking:
                state.thinking_token_count += 1

    # ------------------------------------------------------------------
    # Logits manipulation
    # ------------------------------------------------------------------

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Modify the logits in place to force thinking-budget exits when the limit is reached."""
        if not self.req_states:
            return logits

        for idx, state in list(self.req_states.items()):
            if state.stopped_thinking or not state.inside_thinking:
                continue

            budget = state.max_thinking_tokens
            count = state.thinking_token_count
            grace_start = budget - state.grace_period

            # --- Hard cutoff: force end tokens one at a time ---
            if count >= budget or state.forcing_end_idx >= 0:
                self._force_end_token(logits, idx, state)
                continue

            # --- Grace period: boost \n and end tokens ---
            if count >= grace_start:
                self._apply_grace_boost(logits, idx, state)

        return logits

    def _force_end_token(self, logits: torch.Tensor, batch_idx: int, state: RequestState) -> None:
        """Set all logits to -inf except the next token in the end sequence.
        Advances ``forcing_end_idx`` each call."""
        if state.forcing_end_idx < 0:
            state.forcing_end_idx = 0

        if state.forcing_end_idx < len(state.force_end_ids):
            forced_tok = state.force_end_ids[state.forcing_end_idx]
            original = logits[batch_idx, forced_tok].clone()
            logits[batch_idx].fill_(-float("inf"))
            logits[batch_idx, forced_tok] = original if original != -float("inf") else 0.0
            state.forcing_end_idx += 1
        else:
            # All end tokens have been forced.  Mark thinking as done so
            # the request is cleaned up even if the end sequence did not
            # literally end with the </think> delimiter.
            state.inside_thinking = False
            state.stopped_thinking = True

    @staticmethod
    def _apply_grace_boost(logits: torch.Tensor, batch_idx: int, state: RequestState) -> None:
        """Additively boost newline and end-token logits."""
        for tok_id in state.force_end_ids:
            logits[batch_idx, tok_id] += _GRACE_LOGIT_BOOST
