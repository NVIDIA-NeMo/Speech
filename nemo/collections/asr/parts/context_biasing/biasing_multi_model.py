# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import abc
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, cast

import torch
import torch.nn as nn

from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.core.utils.optional_libs import TRITON_AVAILABLE, triton_required

if TRITON_AVAILABLE:
    import triton

    # from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_triton import multi_model_advance_triton_kernel


@dataclass
class BiasingRequestItemConfig:
    boosting_model_cfg: BoostingTreeModelConfig = field(default_factory=BoostingTreeModelConfig)
    boosting_model_alpha: float = 1.0
    multi_model_id: int | None = None  # compiled model id
    auto_manage_multi_model: bool = True

    def is_empty(self):
        if self.multi_model_id is not None:
            return False
        if not self.boosting_model_cfg.is_empty(self.boosting_model_cfg):
            return False
        return True

    def get_model(self, tokenizer: TokenizerSpec) -> NGramGPULanguageModel | GPUBoostingTreeModel | None:
        if self.boosting_model_cfg.is_empty(self.boosting_model_cfg):
            return None
        boosting_model = GPUBoostingTreeModel.from_config(self.boosting_model_cfg, tokenizer=tokenizer)
        return boosting_model

    def add_to_multi_model(self, tokenizer: TokenizerSpec, biasing_multi_model: "GPUBiasingMultiModelBase"):
        boosting_model = self.get_model(tokenizer=tokenizer)
        if boosting_model is None:
            raise ValueError("Nothing to add, biasing model is empty")
        self.multi_model_id = biasing_multi_model.add_model(model=boosting_model, alpha=self.boosting_model_alpha)

    def remove_from_multi_model(self, biasing_multi_model: "GPUBiasingMultiModelBase"):
        if self.multi_model_id is None:
            # nothing to remove
            return
        biasing_multi_model.remove_model(self.multi_model_id)
        self.multi_model_id = None


class GPUBiasingMultiModelBase(abc.ABC, nn.Module):
    @abstractmethod
    def add_model(self, model: NGramGPULanguageModel, alpha: float = 1.0) -> int:
        raise NotImplementedError

    @abstractmethod
    def remove_model(self, model_id: int):
        raise NotImplementedError

    @staticmethod
    def compatible_with_cuda_graphs() -> bool:
        """True if model can be compiled as a part of CUDA graph, False otherwise"""
        return False

    @abstractmethod
    def advance(
        self, states: torch.Tensor, model_ids: torch.Tensor, eos_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            model_ids: ids of models for each state
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        pass

    @abstractmethod
    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        pass


class GPUBiasingMultiModelReference(GPUBiasingMultiModelBase):
    """Reference implementation (incompatible with CUDA graphs)"""

    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([])
        self.alphas: list[float] = []
        self.vocab_size: int | None = None
        self.float_dtype: torch.dtype | None = None
        self.bos_state: int | None = None
        self.start_state: int | None = None
        self._params_defined = False
        self.free_ids = set()
        self._device = torch.device("cpu")

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self._device = device
        return super().to(*args, **kwargs)

    def _check_model_compatibility(self, model: NGramGPULanguageModel):
        if self.vocab_size != model.vocab_size:
            raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.bos_state != model.bos_state:
            raise ValueError(f"Inconsistent bos state: {self.bos_state} vs {model.bos_state}")
        if self.start_state != model.START_STATE:
            raise ValueError(f"Inconsistent start state: {self.start_state} vs {model.START_STATE}")

    def add_model(self, model: NGramGPULanguageModel, alpha: float = 1.0) -> int:
        if not self._params_defined:
            # there were no previous models
            self.vocab_size = model.vocab_size
            self.bos_state = model.bos_state
            self.start_state = model.START_STATE
            self.float_dtype = model.arcs_weights.dtype
            self._params_defined = True
        self._check_model_compatibility(model=model)
        try:
            model_id = self.free_ids.pop()
        except KeyError:
            model_id = None
        if model_id is None:
            model_id = len(self.models)
            self.models.append(model)
            self.alphas.append(alpha)
        else:
            self.models[model_id] = model
            self.alphas[model_id] = alpha
        return model_id

    def remove_model(self, model_id: int):
        self.models[model_id] = nn.Identity()  # dummy nn model
        self.alphas[model_id] = 0.0
        self.free_ids.add(model_id)

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        if not self._params_defined:
            return torch.zeros([batch_size], device=self._device, dtype=torch.long)
        device = self.models[0].arcs_weights.device
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.start_state, device=device, dtype=torch.long
        )

    def advance(
        self, states: torch.Tensor, model_ids: torch.Tensor, eos_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            model_ids: ids of models for each state
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        batch_size = states.shape[0]
        assert model_ids.shape[0] == batch_size
        device = next(iter(self.parameters())).device
        scores = torch.zeros([batch_size, self.vocab_size], device=device, dtype=self.float_dtype)
        new_states = torch.zeros([batch_size, self.vocab_size], dtype=torch.long, device=device)
        model_ids = model_ids.to("cpu").tolist()
        for batch_i, model_id in enumerate(model_ids):
            if model_id < 0:
                continue
            model = cast(NGramGPULanguageModel, self.models[model_id])
            scores_i, new_states_i = model.advance(states[batch_i : batch_i + 1], eos_id=eos_id)
            scores[batch_i : batch_i + 1] = scores_i * self.alphas[model_id]
            new_states[batch_i : batch_i + 1] = new_states_i
        return scores, new_states


class GPUBiasingMultiModel(GPUBiasingMultiModelBase):
    """Efficient multi-model implementation"""

    INIT_NUM_ARCS = 1_000_000
    INIT_NUM_STATES = 1_000_000
    INIT_NUM_MODELS = 128

    def __init__(self, reallocation_callback_fn: Callable | None = None):
        super().__init__()
        self.vocab_size: int | None = None
        self.float_dtype: torch.dtype | None = None
        self.bos_state: int | None = None
        self.start_state: int | None = None
        self._params_defined = False
        self.free_ids = set()

        self.reallocation_callbacks = []
        if reallocation_callback_fn is not None:
            self.reallocation_callbacks.append(reallocation_callback_fn)

        self.use_triton = TRITON_AVAILABLE

        int_dtype = torch.int64

        self.num_models = 0
        self.num_models_reserved = self.INIT_NUM_MODELS

        # store each model properties
        self.alphas = nn.Buffer(torch.zeros([self.num_models_reserved]))
        self.num_states = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.num_arcs = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.num_arcs_extended = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))

        self.num_states_all = self.INIT_NUM_STATES
        self.num_arcs_extended_all = self.INIT_NUM_ARCS  # + extra padding
        self.num_states_reserved = self.INIT_NUM_STATES
        self.num_arcs_extended_reserved = self.INIT_NUM_ARCS  # + extra padding

        # arcs-related data
        self.arcs_weights = nn.Parameter(torch.zeros([self.num_arcs_extended_reserved]))
        self.from_states = nn.Buffer(torch.zeros([self.num_arcs_extended_reserved], dtype=int_dtype))
        self.to_states = nn.Buffer(torch.zeros([self.num_arcs_extended_reserved], dtype=int_dtype))
        self.ilabels = nn.Buffer(torch.zeros([self.num_arcs_extended_reserved], dtype=int_dtype))

        # states-related data
        self.start_end_arcs = nn.Buffer(torch.zeros([self.num_states_reserved, 2], dtype=int_dtype))
        self.state_order = nn.Buffer(torch.zeros([self.num_states_reserved], dtype=int_dtype))
        self.backoff_to_states = nn.Buffer(torch.zeros([self.num_states_reserved], dtype=int_dtype))
        self.backoff_weights = nn.Parameter(torch.zeros([self.num_states_reserved]))
        self.final_weights = nn.Parameter(torch.zeros([self.num_states_reserved]))

    def _check_model_compatibility(self, model: NGramGPULanguageModel):
        if self.vocab_size != model.vocab_size:
            raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.bos_state != model.bos_state:
            raise ValueError(f"Inconsistent bos state: {self.bos_state} vs {model.bos_state}")
        if self.start_state != model.START_STATE:
            raise ValueError(f"Inconsistent start state: {self.start_state} vs {model.START_STATE}")
        if not model._final_resolved:
            model._resolve_final()

    def _maybe_extend_arcs_and_states(self, add_num_states: int, add_num_arcs_extended: int) -> bool:
        """Extend memory, return True if any tensor is reallocated"""
        reallocated = False
        device = self.arcs_weights.device
        float_dtype = self.arcs_weights.dtype
        int_dtype = self.from_states.dtype
        if self.num_arcs_extended_all + add_num_arcs_extended > self.num_arcs_extended_reserved:
            # min allocation: 2x
            add_num_arcs = max(
                self.num_arcs_extended_reserved,
                self.num_arcs_extended_all + add_num_arcs_extended - self.num_arcs_extended_reserved,
            )
            self.arcs_weights.data = torch.cat(
                (self.arcs_weights.data, torch.zeros([add_num_arcs], dtype=float_dtype, device=device))
            )
            self.from_states.data = torch.cat(
                (self.from_states.data, torch.zeros([add_num_arcs], dtype=int_dtype, device=device))
            )
            self.to_states.data = torch.cat(
                (self.to_states.data, torch.zeros([add_num_arcs], dtype=int_dtype, device=device))
            )
            self.ilabels.data = torch.cat(
                (self.ilabels.data, torch.zeros([add_num_arcs], dtype=int_dtype, device=device))
            )
            self.num_arcs_extended_reserved += add_num_arcs
            reallocated = True

        if self.num_states_all + add_num_states > self.num_states_reserved:
            # min allocation: 2x
            add_num_states = max(
                self.num_states_reserved, self.num_states_all + add_num_states - self.num_states_reserved
            )
            self.start_end_arcs.data = torch.cat(
                (self.start_end_arcs.data, torch.zeros([add_num_states], dtype=int_dtype, device=device))
            )
            self.state_order.data = torch.cat(
                (self.state_order.data, torch.zeros([add_num_states], dtype=int_dtype, device=device))
            )
            self.backoff_to_states.data = torch.cat(
                (self.backoff_to_states.data, torch.zeros([add_num_states], dtype=int_dtype, device=device))
            )
            self.backoff_weights.data = torch.cat(
                (self.backoff_weights.data, torch.zeros([add_num_states], dtype=float_dtype, device=device))
            )
            self.final_weights.data = torch.cat(
                (self.final_weights.data, torch.zeros([add_num_states], dtype=float_dtype, device=device))
            )
            self.num_states_reserved += add_num_states
            reallocated = True

        return reallocated

    def _extend_num_models(self):
        assert self.num_models_reserved > 0
        self.num_models_reserved *= 2
        self.alphas.data = torch.cat((self.alphas.data, torch.zeros_like(self.alphas.data)), dim=-1)
        self.num_states.data = torch.cat((self.num_states.data, torch.zeros_like(self.num_states.data)), dim=-1)
        self.num_arcs.data = torch.cat((self.num_arcs.data, torch.zeros_like(self.num_arcs.data)), dim=-1)
        self.num_arcs_extended.data = torch.cat(
            (self.num_arcs_extended.data, torch.zeros_like(self.num_arcs_extended.data)), dim=-1
        )

    def add_model(self, model: GPUBoostingTreeModel, alpha: float = 1.0) -> int:
        if not self._params_defined:
            # there were no previous models
            self.vocab_size = model.vocab_size
            self.bos_state = model.bos_state
            self.start_state = model.START_STATE
            self.float_dtype = model.arcs_weights.dtype
            self._params_defined = True
        self._check_model_compatibility(model=model)

        reallocated = False
        if self.num_models >= self.num_models_reserved:
            self._extend_num_models()
            reallocated = True
        model_id = self.num_models

        reallocated |= self._maybe_extend_arcs_and_states(
            add_num_states=model.num_states,
            add_num_arcs_extended=model.num_arcs_extended,
        )
        self.num_states[model_id] = model.num_states
        self.num_arcs[model_id] = model.num_arcs
        self.num_arcs_extended[model_id] = model.num_arcs_extended

        states_start = self.num_states_all
        arcs_start = self.num_arcs_extended_all

        # arcs-related data
        self.arcs_weights.data[arcs_start : arcs_start + model.num_arcs].copy_(
            model.arcs_weights.data[: model.num_arcs]
        )
        self.from_states.data[arcs_start : arcs_start + model.num_arcs].copy_(model.from_states.data[: model.num_arcs])
        self.to_states.data[arcs_start : arcs_start + model.num_arcs].copy_(model.to_states.data[: model.num_arcs])
        self.ilabels.data[arcs_start : arcs_start + model.num_arcs].copy_(model.ilabels.data[: model.num_arcs])

        # states-related data
        self.start_end_arcs.data[states_start : states_start + model.num_states].copy_(
            model.start_end_arcs.data[: model.num_states]
        )
        self.state_order.data[states_start : states_start + model.num_states].copy_(
            model.state_order.data[: model.num_states]
        )
        self.backoff_to_states.data[states_start : states_start + model.num_states].copy_(
            model.backoff_to_states.data[: model.num_states]
        )
        self.backoff_weights.data[states_start : states_start + model.num_states].copy_(
            model.backoff_weights.data[: model.num_states]
        )
        self.final_weights.data[states_start : states_start + model.num_states].copy_(
            model.final_weights.data[: model.num_states]
        )

        self.num_states_all += model.num_states
        self.num_arcs_extended_all += model.num_arcs_extended

        self.alphas[model_id] = alpha
        self.num_models += 1
        if reallocated:
            for reallocation_callback_fn in self.reallocation_callbacks:
                reallocation_callback_fn()
        return model_id

    def remove_model(self, model_id: int):
        raise NotImplementedError

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        device = self.arcs_weights.device
        if not self._params_defined:
            return torch.zeros([batch_size], device=device, dtype=torch.long)
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.start_state, device=device, dtype=torch.long
        )

    def advance(
        self, states: torch.Tensor, model_ids: torch.Tensor, eos_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            model_ids: ids of models for each state
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        assert model_ids.shape[0] == states.shape[0]

        if self.use_triton and states.device.type == "cuda":
            scores, next_states = self._advance_triton(states=states, model_ids=model_ids)
        else:
            scores, next_states = self._advance_pytorch(states=states, model_ids=model_ids)

        # replace eos_id score with maximum state weight to prevent from hallucinating in case of AED models (e.g. Canary)
        if eos_id is not None:
            raise NotImplementedError

        return scores, next_states

    @triton_required
    def _advance_triton(self, states: torch.Tensor, model_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        Triton implementation. Currently not differentiable.

        Args:
            states: batch of states

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        scores = torch.zeros([batch_size, self.vocab_size], device=device, dtype=self.arcs_weights.dtype)
        new_states = torch.zeros([batch_size, self.vocab_size], dtype=torch.long, device=device)

        raise NotImplementedError

        # ngram_advance_triton_kernel[batch_size,](
        #     vocab_size=self.vocab_size,
        #     states_ptr=states,
        #     new_states_ptr=new_states,
        #     scores_ptr=scores,
        #     start_state=self.START_STATE,
        #     to_states_ptr=self.to_states,
        #     ilabels_ptr=self.ilabels,
        #     arcs_weights_ptr=self.arcs_weights,
        #     start_end_arcs_ptr=self.start_end_arcs,
        #     backoff_to_states_ptr=self.backoff_to_states,
        #     backoff_weights_ptr=self.backoff_weights,
        #     BLOCK_SIZE=triton.next_power_of_2(self.vocab_size),
        # )

        return scores, new_states

    def _advance_pytorch(self, states: torch.Tensor, model_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        PyTorch implementation (slow, differentiable).

        Args:
            states: batch of states

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        current_states = states.clone()
        states_dtype = current_states.dtype

        # init output tensors
        out_scores = torch.zeros(batch_size, self.vocab_size, device=device)
        out_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=states_dtype, device=device)

        # helper ranges
        vocab_range = torch.arange(self.vocab_size, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # backoff weight accumulator
        accumulated_backoff = torch.zeros(batch_size, device=device)
        # loop condition
        start_state_not_processed = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)

        num_iterations = 0
        while start_state_not_processed.any():
            # assert num_iterations <= self.max_order, "Infinite loop in LM advance"
            num_iterations += 1
            # get arc boundaries
            start, end = self.start_end_arcs[current_states].unbind(dim=1)
            # number of arcs for each state cannot be larger than vocab size
            indices = start[:, None] + vocab_range[None, :]
            mask = indices < end[:, None]
            mask &= start_state_not_processed[:, None]
            mask_flat = mask.view(-1)
            indices_flat = indices.view(-1)
            # map indices outside the mask to vocab_size + 1
            scores_add = torch.zeros([batch_size, self.vocab_size + 1], device=device, dtype=out_scores.dtype)
            out_states_add = torch.full(
                [batch_size, self.vocab_size + 1], fill_value=-1, device=device, dtype=states_dtype
            )
            ilabels = self.ilabels[indices_flat] * mask_flat + ~mask_flat * self.vocab_size
            scores_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.arcs_weights[indices_flat]
            out_states_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.to_states[
                indices_flat
            ].to(states_dtype)
            # fill out_scores and out_states with new values where state is not found yet
            state_found = out_states != -1
            out_scores = torch.where(
                state_found, out_scores, accumulated_backoff.unsqueeze(-1) + scores_add[:, : self.vocab_size]
            )
            out_states = torch.where(state_found, out_states, out_states_add[:, : self.vocab_size])
            # update loop condition; process backoffs
            start_state_not_processed &= current_states != self.START_STATE
            accumulated_backoff += self.backoff_weights[current_states] * start_state_not_processed
            torch.where(
                start_state_not_processed, self.backoff_to_states[current_states], current_states, out=current_states
            )
        return out_scores, out_states
