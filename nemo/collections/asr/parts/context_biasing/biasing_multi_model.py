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
from nemo.utils import logging

if TRITON_AVAILABLE:
    import triton

    from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_triton import ngram_multi_advance_triton_kernel

_BIASING_MODEL_CACHE = dict()


@dataclass
class BiasingRequestItemConfig:
    boosting_model_cfg: BoostingTreeModelConfig = field(default_factory=BoostingTreeModelConfig)
    boosting_model_alpha: float = 1.0  # boosting weight
    cache_key: str | None = None  # cache key for memory cache; NB: cache key should be unique for (tokenizer, phrases)
    multi_model_id: int | None = None  # compiled model id
    auto_manage_multi_model: bool = True  # if model should be added to the decoder and removed automatically

    def __post_init__(self):
        # if BiasingRequestItemConfig initialized from dict, we need to fix boosting_model_cfg field
        # see solution https://stackoverflow.com/a/60383031
        if isinstance(self.boosting_model_cfg, dict):
            self.boosting_model_cfg = BoostingTreeModelConfig(**self.boosting_model_cfg)

    def is_empty(self) -> bool:
        """Return True if biasing request (or model) is empty"""
        if self.cache_key and self.cache_key in _BIASING_MODEL_CACHE:
            return False
        if self.multi_model_id is not None:
            return False
        if not BoostingTreeModelConfig.is_empty(self.boosting_model_cfg):
            return False
        return True

    def get_model(self, tokenizer: TokenizerSpec) -> NGramGPULanguageModel | GPUBoostingTreeModel | None:
        """Create biasing model or get from cache, return the model. `None` is returned if biasing config is empty"""
        if self.cache_key and self.cache_key in _BIASING_MODEL_CACHE:
            return _BIASING_MODEL_CACHE[self.cache_key]
        if self.boosting_model_cfg.is_empty(self.boosting_model_cfg):
            return None
        boosting_model = GPUBoostingTreeModel.from_config(self.boosting_model_cfg, tokenizer=tokenizer)
        if self.cache_key:
            _BIASING_MODEL_CACHE[self.cache_key] = boosting_model
        return boosting_model

    def add_to_multi_model(self, tokenizer: TokenizerSpec, biasing_multi_model: "GPUBiasingMultiModelBase"):
        """Add biasing model to biasing multi-model"""
        boosting_model = self.get_model(tokenizer=tokenizer)
        if boosting_model is None:
            raise ValueError("Nothing to add, biasing model is empty")
        self.multi_model_id = biasing_multi_model.add_model(model=boosting_model, alpha=self.boosting_model_alpha)

    def remove_from_cache(self):
        """Remove model from cache (if cache entry exists)"""
        if self.cache_key and self.cache_key in _BIASING_MODEL_CACHE:
            del _BIASING_MODEL_CACHE[self.cache_key]

    def remove_from_multi_model(self, biasing_multi_model: "GPUBiasingMultiModelBase"):
        """Remove biasing model from multi-model"""
        if self.multi_model_id is None:
            # nothing to remove
            return
        biasing_multi_model.remove_model(self.multi_model_id)
        self.multi_model_id = None


class GPUBiasingMultiModelBase(abc.ABC, nn.Module):
    """
    Base class for implementing biasing multi-model:
    model that contains multiple biasing models and handles batched requests for them
    """

    START_STATE = 0

    @abstractmethod
    def add_model(self, model: NGramGPULanguageModel, alpha: float = 1.0) -> int:
        pass

    @abstractmethod
    def remove_model(self, model_id: int):
        pass

    @abstractmethod
    def has_models(self) -> bool:
        """Return True if the multi-model has at least one model"""
        pass

    def compatible_with_cuda_graphs(self) -> bool:
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

    def __init__(self, vocab_size: int, *args, **kwargs):
        """

        Args:
            vocab_size: vocabulary size of the model
            *args, **kwargs: added for easiness of switching between this model and efficient implementation
        """
        super().__init__()
        self.models = nn.ModuleList([])
        self.buffer_for_device_handling = nn.Buffer(torch.zeros([1], dtype=torch.long))
        self.alphas: list[float] = []
        self.vocab_size: int = vocab_size
        self.float_dtype: torch.dtype | None = None
        self.bos_state: int | None = None
        self._params_defined = False
        self.free_ids = set()
        self.num_models = 0

    def has_models(self) -> bool:
        """Return True if the multi-model has at least one model"""
        return self.num_models > 0

    def _check_model_compatibility(self, model: NGramGPULanguageModel):
        if self.vocab_size != model.vocab_size:
            raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.bos_state != model.bos_state:
            raise ValueError(f"Inconsistent bos state: {self.bos_state} vs {model.bos_state}")
        if self.START_STATE != model.START_STATE:
            raise ValueError(f"Inconsistent start state: {self.START_STATE} vs {model.START_STATE}")

    def add_model(self, model: NGramGPULanguageModel, alpha: float = 1.0) -> int:
        if not self._params_defined:
            # there were no previous models
            self.bos_state = model.bos_state
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
        self.num_models += 1
        return model_id

    def remove_model(self, model_id: int):
        self.models[model_id] = nn.Identity()  # dummy nn model
        self.alphas[model_id] = 0.0
        self.free_ids.add(model_id)
        self.num_models -= 1

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        device = self.buffer_for_device_handling.device
        if not self._params_defined:
            return torch.zeros([batch_size], device=device, dtype=torch.long)
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.START_STATE, device=device, dtype=torch.long
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
        next_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=torch.long, device=device)
        model_ids = model_ids.to("cpu").tolist()
        for batch_i, model_id in enumerate(model_ids):
            if model_id < 0:
                continue
            model = cast(NGramGPULanguageModel, self.models[model_id])
            scores_i, next_states_i = model.advance(states[batch_i : batch_i + 1], eos_id=eos_id)
            scores[batch_i : batch_i + 1] = scores_i * self.alphas[model_id]
            next_states[batch_i : batch_i + 1] = next_states_i
        return scores, next_states


class GPUBiasingMultiModel(GPUBiasingMultiModelBase):
    """Efficient multi-model implementation"""

    INIT_NUM_ARCS = 1_000_000
    INIT_NUM_STATES = 1_000_000
    INIT_NUM_MODELS = 128

    def __init__(
        self, vocab_size: int, reallocation_callback_fn: Callable | None = None, use_triton: bool | None = None
    ):
        """

        Args:
            vocab_size: vocabulary size of the model
            reallocation_callback_fn: function to call when reallocation occurred (needed for decoders with CUDA graphs)
            use_triton: allow using Triton, `None` means "auto" (used if available)
        """
        super().__init__()
        self.vocab_size: int = vocab_size
        self.float_dtype: torch.dtype | None = None
        self.bos_state: int | None = None
        self._params_defined = False
        self.free_ids = set()

        self.reallocation_callbacks = []
        if reallocation_callback_fn is not None:
            self.reallocation_callbacks.append(reallocation_callback_fn)

        self.use_triton = use_triton if use_triton is not None else TRITON_AVAILABLE

        int_dtype = torch.int64

        self.num_models = 0
        self.num_models_reserved = self.INIT_NUM_MODELS

        # store each model properties
        self.model2alpha = nn.Buffer(torch.zeros([self.num_models_reserved]))
        self.model2active = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.bool))
        self.model2num_states = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.model2num_arcs = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.model2num_arcs_extended = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.model2states_offset = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.model2arcs_offset = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))

        self.num_states_total = 0
        self.num_arcs_extended_total = 0  # + extra padding
        self.num_states_reserved = self.INIT_NUM_STATES
        self.num_arcs_extended_reserved = self.INIT_NUM_ARCS  # + extra padding

        # arcs-related data
        self.all_arcs_weights = nn.Parameter(torch.zeros([self.num_arcs_extended_reserved]))
        self.all_from_states = nn.Buffer(torch.zeros([self.num_arcs_extended_reserved], dtype=int_dtype))
        self.all_to_states = nn.Buffer(torch.zeros([self.num_arcs_extended_reserved], dtype=int_dtype))
        self.all_ilabels = nn.Buffer(torch.zeros([self.num_arcs_extended_reserved], dtype=int_dtype))

        # states-related data
        self.all_start_end_arcs = nn.Buffer(torch.zeros([self.num_states_reserved, 2], dtype=int_dtype))
        self.all_state_order = nn.Buffer(torch.zeros([self.num_states_reserved], dtype=int_dtype))
        self.all_backoff_to_states = nn.Buffer(torch.zeros([self.num_states_reserved], dtype=int_dtype))
        self.all_backoff_weights = nn.Parameter(torch.zeros([self.num_states_reserved]))
        self.all_final_weights = nn.Parameter(torch.zeros([self.num_states_reserved]))

    def compatible_with_cuda_graphs(self) -> bool:
        """True if model can be compiled as a part of CUDA graph, False otherwise"""
        return self.use_triton

    def has_models(self) -> bool:
        """Return True if the multi-model has at least one model"""
        return self.num_models > 0

    def _check_model_compatibility(self, model: NGramGPULanguageModel):
        """Check that the new model parameters are the same compared to already stored models"""
        if self.vocab_size != model.vocab_size:
            raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.bos_state != model.bos_state:
            raise ValueError(f"Inconsistent bos state: {self.bos_state} vs {model.bos_state}")
        if self.START_STATE != model.START_STATE:
            raise ValueError(f"Inconsistent start state: {self.START_STATE} vs {model.START_STATE}")
        if not model._final_resolved:
            model._resolve_final()

    @staticmethod
    def _extend_buffer_or_param(buffer_or_param: nn.Buffer | nn.Parameter, add_len: int):
        """Extend buffer or parameter"""
        buffer_or_param.data = torch.cat(
            (
                buffer_or_param.data,
                torch.zeros(
                    [add_len] + list(buffer_or_param.shape)[1:],
                    dtype=buffer_or_param.dtype,
                    device=buffer_or_param.device,
                ),
            )
        )

    def _maybe_extend_arcs_and_states(self, add_num_states: int, add_num_arcs_extended: int) -> bool:
        """Extend memory allocated for arcs and states, return True if any tensor is reallocated"""
        reallocated = False

        if self.num_arcs_extended_total + add_num_arcs_extended > self.num_arcs_extended_reserved:
            # min allocation: 2x
            add_num_arcs = max(
                self.num_arcs_extended_reserved,
                self.num_arcs_extended_total + add_num_arcs_extended - self.num_arcs_extended_reserved,
            )
            self._extend_buffer_or_param(self.all_arcs_weights, add_len=add_num_arcs)
            self._extend_buffer_or_param(self.all_from_states, add_len=add_num_arcs)
            self._extend_buffer_or_param(self.all_to_states, add_len=add_num_arcs)
            self._extend_buffer_or_param(self.all_ilabels, add_len=add_num_arcs)
            self.num_arcs_extended_reserved += add_num_arcs
            reallocated = True

        if self.num_states_total + add_num_states > self.num_states_reserved:
            # min allocation: 2x
            add_num_states = max(
                self.num_states_reserved, self.num_states_total + add_num_states - self.num_states_reserved
            )
            self._extend_buffer_or_param(self.all_start_end_arcs, add_len=add_num_states)
            self._extend_buffer_or_param(self.all_state_order, add_len=add_num_states)
            self._extend_buffer_or_param(self.all_backoff_to_states, add_len=add_num_states)
            self._extend_buffer_or_param(self.all_backoff_weights, add_len=add_num_states)
            self._extend_buffer_or_param(self.all_final_weights, add_len=add_num_states)
            self.num_states_reserved += add_num_states
            reallocated = True

        return reallocated

    @staticmethod
    def _extend_buffer_2x(buffer: nn.Buffer):
        buffer.data = torch.cat((buffer.data, torch.zeros_like(buffer.data)), dim=-1)

    def _extend_num_models(self):
        """Extend memory allocated for models with properties"""
        assert self.num_models_reserved > 0
        self.num_models_reserved *= 2

        self._extend_buffer_2x(self.model2alpha)
        self._extend_buffer_2x(self.model2active)
        self._extend_buffer_2x(self.model2num_states)
        self._extend_buffer_2x(self.model2num_arcs)
        self._extend_buffer_2x(self.model2num_arcs_extended)
        self._extend_buffer_2x(self.model2states_offset)
        self._extend_buffer_2x(self.model2arcs_offset)

    @torch.no_grad()
    def add_model(self, model: GPUBoostingTreeModel, alpha: float = 1.0) -> int:
        """
        Add boosting model with `alpha` weight. Returns id for the added model

        Args:
            model: boosting model
            alpha: weight of the boosting model

        Returns:
            model id (to use in queries)
        """
        if not self._params_defined:
            # there were no previous models
            self.bos_state = model.bos_state
            self.float_dtype = model.arcs_weights.dtype
            self._params_defined = True
        self._check_model_compatibility(model=model)

        reallocated = False
        # select model id: either any free id, or num_models
        if self.free_ids:
            model_id = self.free_ids.pop()
        else:
            if self.num_models >= self.num_models_reserved:
                self._extend_num_models()
                reallocated = True
            model_id = self.num_models
            self.num_models += 1
        self.model2alpha[model_id] = alpha
        self.model2active[model_id] = True

        reallocated |= self._maybe_extend_arcs_and_states(
            add_num_states=model.num_states,
            add_num_arcs_extended=model.num_arcs_extended,
        )
        self.model2num_states[model_id] = model.num_states
        self.model2num_arcs[model_id] = model.num_arcs
        self.model2num_arcs_extended[model_id] = model.num_arcs_extended
        self.model2states_offset[model_id] = self.num_states_total
        self.model2arcs_offset[model_id] = self.num_arcs_extended_total

        # model is added always to the end of data storage
        states_start = self.num_states_total
        arcs_start = self.num_arcs_extended_total

        # arcs-related data
        self.all_arcs_weights.data[arcs_start : arcs_start + model.num_arcs].copy_(
            model.arcs_weights.data[: model.num_arcs]
        )
        self.all_from_states.data[arcs_start : arcs_start + model.num_arcs].copy_(
            model.from_states.data[: model.num_arcs]
        )
        self.all_to_states.data[arcs_start : arcs_start + model.num_arcs].copy_(model.to_states.data[: model.num_arcs])
        self.all_ilabels.data[arcs_start : arcs_start + model.num_arcs].copy_(model.ilabels.data[: model.num_arcs])

        # states-related data
        self.all_start_end_arcs.data[states_start : states_start + model.num_states].copy_(
            model.start_end_arcs.data[: model.num_states]
        )
        self.all_state_order.data[states_start : states_start + model.num_states].copy_(
            model.state_order.data[: model.num_states]
        )
        self.all_backoff_to_states.data[states_start : states_start + model.num_states].copy_(
            model.backoff_to_states.data[: model.num_states]
        )
        self.all_backoff_weights.data[states_start : states_start + model.num_states].copy_(
            model.backoff_weights.data[: model.num_states]
        )
        self.all_final_weights.data[states_start : states_start + model.num_states].copy_(
            model.final_weights.data[: model.num_states]
        )

        self.num_states_total += model.num_states
        self.num_arcs_extended_total += model.num_arcs_extended

        if reallocated:
            logging.info("Biasing multi-model reallocated memory. Executing reallocation callbacks")
            for reallocation_callback_fn in self.reallocation_callbacks:
                reallocation_callback_fn()
        return model_id

    @staticmethod
    def _clear_buffer_or_param_range(
        buffer_or_param: nn.Buffer | nn.Parameter, start: int, end: int, buffer_len: int | None = None
    ):
        if buffer_len is None:
            buffer_len = buffer_or_param.shape[0]
        remove_len = end - start
        buffer_or_param[start : buffer_len - remove_len].copy_(buffer_or_param[end:buffer_len].clone())
        buffer_or_param[buffer_len - remove_len : buffer_len].fill_(0)

    @torch.no_grad()
    def remove_model(self, model_id: int):
        """
        Remove boosting model.

        Args:
            model_id: boosting model id provided by the `add_model` method
        """
        logging.debug(f"Removing model: {model_id}")
        if model_id in self.free_ids or model_id >= self.num_models:
            raise ValueError(
                f"Trying to remove already deleted or non-existing model {model_id}. Total models in reserve: {self.num_models}"
            )

        # set model as inactive (we do not decrease num_models, only set to inactive!)
        self.model2active[model_id] = False
        self.model2alpha[model_id] = 0.0
        self.free_ids.add(model_id)

        start_state = self.model2states_offset[model_id].item()
        num_states = self.model2num_states[model_id].item()
        end_state = start_state + num_states

        start_arc = self.model2arcs_offset[model_id].item()
        num_arcs = self.model2num_arcs_extended[model_id].item()
        end_arc = start_arc + num_arcs

        assert num_arcs > 0 and num_states > 0, "Unexpected zero-size model"

        # clean up arcs-related data: cut [start_arc, end_arc) from the buffer (shifting right part to the left)
        self._clear_buffer_or_param_range(self.all_arcs_weights, start_arc, end_arc, self.num_arcs_extended_total)
        self._clear_buffer_or_param_range(self.all_from_states, start_arc, end_arc, self.num_arcs_extended_total)
        self._clear_buffer_or_param_range(self.all_to_states, start_arc, end_arc, self.num_arcs_extended_total)
        self._clear_buffer_or_param_range(self.all_ilabels, start_arc, end_arc, self.num_arcs_extended_total)

        # clean up states-related data: cut [start_state, end_state) from the buffer (shifting right part to the left)
        self._clear_buffer_or_param_range(self.all_start_end_arcs, start_state, end_state, self.num_states_total)
        self._clear_buffer_or_param_range(self.all_state_order, start_state, end_state, self.num_states_total)
        self._clear_buffer_or_param_range(self.all_backoff_to_states, start_state, end_state, self.num_states_total)
        self._clear_buffer_or_param_range(self.all_backoff_weights, start_state, end_state, self.num_states_total)
        self._clear_buffer_or_param_range(self.all_final_weights, start_state, end_state, self.num_states_total)

        # set num states/arcs to zero
        self.num_states_total -= num_states
        self.num_arcs_extended_total -= num_arcs

        self.model2num_states[model_id] = 0
        self.model2num_arcs[model_id] = 0
        self.model2num_arcs_extended[model_id] = 0
        # shift model offsets
        self.model2states_offset[model_id] = 0
        self.model2arcs_offset[model_id] = 0
        # shift states and arcs offsets
        torch.where(
            self.model2states_offset < start_state,
            self.model2states_offset,
            self.model2states_offset - num_states,
            out=self.model2states_offset,
        )
        torch.where(
            self.model2arcs_offset < start_arc,
            self.model2arcs_offset,
            self.model2arcs_offset - num_arcs,
            out=self.model2arcs_offset,
        )

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        device = self.all_arcs_weights.device
        if not self._params_defined:
            return torch.full([batch_size], fill_value=self.START_STATE, device=device, dtype=torch.long)
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.START_STATE, device=device, dtype=torch.long
        )

    def advance(
        self, states: torch.Tensor, model_ids: torch.Tensor, eos_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            model_ids: batch of ids of the models (`-1` to apply dummy model with zero weight)
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        assert model_ids.shape[0] == states.shape[0]

        if self.use_triton and states.device.type == "cuda":
            scores, next_states = self._advance_triton(states=states, model_ids=model_ids)
        else:
            scores, next_states = self._advance_pytorch(states=states, model_ids=model_ids)
        # NB: model_id can be -1, but we assume that there at least 1 element in self.alphas
        scores *= self.model2alpha[model_ids][:, None]

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
            model_ids: ids of the models (`-1` to apply dummy model with zero weight)

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        scores = torch.zeros([batch_size, self.vocab_size], device=device, dtype=self.all_arcs_weights.dtype)
        next_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=torch.long, device=device)

        ngram_multi_advance_triton_kernel[batch_size,](
            vocab_size=self.vocab_size,
            states_ptr=states,
            new_states_out_ptr=next_states,
            scores_out_ptr=scores,
            start_state=self.START_STATE,
            model_ids_ptr=model_ids,
            states_offsets_ptr=self.model2states_offset,
            arcs_offsets_ptr=self.model2arcs_offset,
            to_states_ptr=self.all_to_states,
            ilabels_ptr=self.all_ilabels,
            arcs_weights_ptr=self.all_arcs_weights,
            start_end_arcs_ptr=self.all_start_end_arcs,
            backoff_to_states_ptr=self.all_backoff_to_states,
            backoff_weights_ptr=self.all_backoff_weights,
            BLOCK_SIZE=triton.next_power_of_2(self.vocab_size),
        )

        return scores, next_states

    def _advance_pytorch(self, states: torch.Tensor, model_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        PyTorch implementation (slow, differentiable).

        Args:
            states: batch of states
            model_ids: ids of the models (`-1` to apply dummy model with zero weight)

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
        start_state_not_processed = model_ids != -1

        states_offsets = self.model2states_offset[model_ids]
        arcs_offsets = self.model2arcs_offset[model_ids]

        num_iterations = 0
        while start_state_not_processed.any():
            num_iterations += 1
            # get arc boundaries
            start, end = self.all_start_end_arcs[current_states + states_offsets].unbind(dim=1)
            # number of arcs for each state cannot be larger than vocab size
            start += arcs_offsets
            end += arcs_offsets
            arc_indices = start[:, None] + vocab_range[None, :]
            mask = arc_indices < end[:, None]
            mask &= start_state_not_processed[:, None]
            mask_flat = mask.view(-1)
            arc_indices_flat = arc_indices.view(-1)
            # map indices outside the mask to vocab_size + 1
            scores_add = torch.zeros([batch_size, self.vocab_size + 1], device=device, dtype=out_scores.dtype)
            out_states_add = torch.full(
                [batch_size, self.vocab_size + 1], fill_value=-1, device=device, dtype=states_dtype
            )
            ilabels = self.all_ilabels[arc_indices_flat] * mask_flat + ~mask_flat * self.vocab_size
            scores_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.all_arcs_weights[
                arc_indices_flat
            ]
            out_states_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.all_to_states[
                arc_indices_flat
            ].to(states_dtype)
            # fill out_scores and out_states with new values where state is not found yet
            state_found = out_states != -1
            out_scores = torch.where(
                state_found, out_scores, accumulated_backoff.unsqueeze(-1) + scores_add[:, : self.vocab_size]
            )
            out_states = torch.where(state_found, out_states, out_states_add[:, : self.vocab_size])
            # update loop condition; process backoffs
            start_state_not_processed &= current_states != self.START_STATE
            accumulated_backoff += (
                self.all_backoff_weights[current_states + states_offsets] * start_state_not_processed
            )
            torch.where(
                start_state_not_processed,
                self.all_backoff_to_states[current_states + states_offsets],
                current_states,
                out=current_states,
            )
        return out_scores, out_states
