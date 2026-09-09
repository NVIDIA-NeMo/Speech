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

from typing import Dict, Optional

import torch
from torch import Tensor

from nemo.utils import logging

__all__ = ['LangIdPromptMixin']


class LangIdPromptMixin:
    """Language-ID prompt conditioning for encoder-decoder ASR models ("unified" models).

    A one-hot language/task vector is concatenated to every encoder output frame and projected back
    to the encoder dimension, letting the decoder condition on the requested language. The mixin owns
    the projection module, the prompt vocabulary, and the prompt tensor construction.

    Inference only. The projection is part of the trained forward pass but no training entry point
    conditions on it, so :meth:`setup_lang_id_prompt` refuses to enable conditioning when a trainer
    is attached. Training support arrives with the dedicated unified model class.

    Host models need to call :meth:`setup_lang_id_prompt` during ``__init__``. Conditioning is then
    applied through whichever entry point suits the inference mode:

    - :meth:`apply_lang_id_prompt` — explicit prompt tensor, one row per utterance. Used by offline
      transcription, buffered/chunked streaming, and batched cache-aware streaming, where different
      streams in a batch may request different languages.
    - :meth:`apply_lang_id_prompt_for_transcribe` — resolves a language name and applies it to a
      whole batch. A no-op for models without prompt conditioning, so transcription paths can call
      it unconditionally.

    Cache-aware streaming through ``conformer_stream_step`` is not supported: that path belongs to
    ``PromptStreamingMixin``, whose models are trained for it.

    Enabled by the following model config, which must be present in the checkpoint:

    .. code-block:: yaml

        model:
          model_defaults:
            initialize_lang_id_prompt: true
            num_lang_id_prompts: 128
            lang_id_prompt_dictionary: {en-US: 0, de-DE: 9, ..., unk: 127}

    Models that were not trained with language-ID prompts keep ``use_lang_id_prompt = False`` and are
    entirely unaffected.
    """

    # Plain class-level defaults so any host model can be interrogated without a `getattr` guard and
    # before (or without) calling ``setup_lang_id_prompt``. ``lang_id_prompt_kernel`` is
    # intentionally NOT declared here: it is an ``nn.Module`` and a class attribute would shadow
    # ``nn.Module.__getattr__``'s lookup into ``_modules`` once the real module is registered.
    use_lang_id_prompt: bool = False
    num_lang_id_prompts: Optional[int] = None
    lang_id_prompt_dictionary: Optional[Dict[str, int]] = None

    # Keys tried in order when no language is requested, most language-agnostic first. Only keys the
    # checkpoint actually defines are considered, so a model with a narrower vocabulary still works.
    DEFAULT_LANG_ID_PROMPT_PREFERENCE = ('unk', 'auto', 'en-US')

    def setup_lang_id_prompt(self) -> None:
        """Build the prompt projection if the model config asks for it, otherwise do nothing.

        Safe to call unconditionally from a host model's ``__init__``.

        Raises:
            ValueError: If the config enables conditioning while a trainer is attached, since no
                training entry point passes a prompt (see the class docstring).
        """
        # Models carrying their own prompt machinery (``PromptStreamingMixin``, used by the
        # prompt-aware training classes) build and own their projection themselves.
        if hasattr(self, 'initialize_prompt_feature'):
            return

        model_defaults = self.cfg.get('model_defaults') or {}
        if not model_defaults.get('initialize_lang_id_prompt', False):
            return

        if getattr(self, '_trainer', None) is not None:
            raise ValueError(
                "`model_defaults.initialize_lang_id_prompt=true` is inference-only: the prompt "
                "projection would be a trainable parameter that no training or validation step "
                "conditions on, which fails under DDP and otherwise scores unconditioned metrics. "
                "Transcribe with `transcribe(source_lang=...)` or the chunked streaming script "
                "instead of attaching a trainer."
            )

        prompt_dictionary = model_defaults.get('lang_id_prompt_dictionary', None)
        if not prompt_dictionary:
            raise ValueError(
                "`model_defaults.lang_id_prompt_dictionary` must be a non-empty mapping of language/task "
                "name to prompt index when `model_defaults.initialize_lang_id_prompt=true`."
            )

        enc_hidden = model_defaults.get('enc_hidden', None)
        if enc_hidden is None:
            raise ValueError(
                "`model_defaults.enc_hidden` is required to size the prompt projection when "
                "`model_defaults.initialize_lang_id_prompt=true`."
            )

        self.use_lang_id_prompt = True
        self.num_lang_id_prompts = int(model_defaults.get('num_lang_id_prompts', 128))
        self.lang_id_prompt_dictionary = prompt_dictionary
        self.lang_id_prompt_kernel = _Float32PromptProjection(
            torch.nn.Linear(self.num_lang_id_prompts + enc_hidden, enc_hidden * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_hidden * 2, enc_hidden),
        )
        logging.info(
            f"Language-ID prompt conditioning enabled: num_lang_id_prompts={self.num_lang_id_prompts}, "
            f"languages={list(self.lang_id_prompt_dictionary.keys())}"
        )

    @property
    def default_lang_id_prompt(self) -> Optional[str]:
        """The language this model conditions on when a request does not name one.

        Lets callers stay out of the business of guessing language keys: a multilingual model
        advertises its own language-agnostic prompt instead of every call site hardcoding one.

        Returns:
            The most language-agnostic key the checkpoint defines, or None if the model has no
            language-ID prompts or defines none of the candidate keys.
        """
        if not self.use_lang_id_prompt:
            return None
        return next(
            (key for key in self.DEFAULT_LANG_ID_PROMPT_PREFERENCE if key in self.lang_id_prompt_dictionary), None
        )

    def resolve_lang_id_prompt(self, source_lang: Optional[str]) -> int:
        """Resolve a spoken-language name to its prompt index.

        The prompt projection is part of the trained forward pass and must always be applied, so this
        always returns an index. When ``source_lang`` is missing or unknown, the model's
        :attr:`default_lang_id_prompt` is used and a warning is logged.

        Args:
            source_lang: A key of ``lang_id_prompt_dictionary`` (e.g. ``"en-US"``), or None. This is
                the language spoken in the audio; the unified model is ASR-only, so there is no
                separate translation target.

        Returns:
            The prompt index to condition on.
        """
        self._assert_lang_id_prompt_supported()

        if source_lang is not None and source_lang in self.lang_id_prompt_dictionary:
            return self.lang_id_prompt_dictionary[source_lang]

        preview = self._language_preview()
        fallback = self.default_lang_id_prompt
        if fallback is None:
            raise ValueError(
                f"Cannot pick a default language-ID prompt: the model defines none of "
                f"{list(self.DEFAULT_LANG_ID_PROMPT_PREFERENCE)}. Please pass an explicit "
                f"`source_lang`. Available: {preview}"
            )

        if source_lang is None:
            logging.warning(
                f"No `source_lang` provided for a language-ID prompt model; falling back to the "
                f"'{fallback}' prompt. Pass `source_lang=<lang>` (e.g. source_lang=en-US) to force a "
                f"specific language."
            )
        else:
            logging.warning(
                f"Unknown source_lang='{source_lang}' (available: {preview}); falling back to the "
                f"'{fallback}' prompt."
            )
        return self.lang_id_prompt_dictionary[fallback]

    def create_lang_id_prompt(
        self, batch_size: int, prompt_id: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """Create a one-hot language-ID prompt shared by a whole batch.

        Args:
            batch_size: Number of utterances in the batch.
            prompt_id: Prompt index, e.g. from :meth:`resolve_lang_id_prompt`.
            dtype: Dtype of the returned tensor.
            device: Device of the returned tensor.

        Returns:
            One-hot prompt of shape ``(batch_size, num_lang_id_prompts)``.
        """
        prompt = torch.zeros(batch_size, self.num_lang_id_prompts, dtype=dtype, device=device)
        prompt[:, prompt_id] = 1.0
        return prompt

    def create_lang_id_prompts(self, prompt_ids: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Create a one-hot language-ID prompt with a possibly different language per utterance.

        Use this when a batch mixes languages (e.g. per-sample ``lang`` fields from a manifest);
        pass a single index to :meth:`create_lang_id_prompt` instead when the whole batch shares one.

        Args:
            prompt_ids: 1-D tensor (or sequence) of per-utterance prompt indices, e.g. each from
                :meth:`resolve_lang_id_prompt`.
            dtype: Dtype of the returned tensor.
            device: Device of the returned tensor.

        Returns:
            One-hot prompt of shape ``(len(prompt_ids), num_lang_id_prompts)``.
        """
        self._assert_lang_id_prompt_supported()
        prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
        prompt = torch.zeros(prompt_ids.shape[0], self.num_lang_id_prompts, dtype=dtype, device=device)
        prompt.scatter_(1, prompt_ids.unsqueeze(1), 1.0)
        return prompt

    def apply_lang_id_prompt(self, encoded: Tensor, prompt: Tensor) -> Tensor:
        """Condition the encoder output on a language-ID prompt.

        Each row of ``prompt`` may select a different language, so this also serves batched streaming
        where every stream requests its own language.

        Args:
            encoded: Encoder output of shape ``(B, D, T)``.
            prompt: One-hot prompt of shape ``(B, num_lang_id_prompts)``, broadcast across time; or a
                per-frame prompt of shape ``(B, T, num_lang_id_prompts)`` whose ``T`` must match
                ``encoded``.

        Returns:
            Prompt-conditioned encoder output of shape ``(B, D, T)``.
        """
        self._assert_lang_id_prompt_supported()

        encoded = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        out_dtype = encoded.dtype
        batch_size, time_steps, _ = encoded.shape

        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(1).expand(-1, time_steps, -1)
        elif prompt.dim() == 3:
            # Reject a mismatch rather than truncating or zero-padding: a zero-padded frame is a prompt
            # the model never saw in training, and silently mis-conditions the tail of every chunk.
            if prompt.shape[1] != time_steps:
                raise ValueError(
                    f"prompt has {prompt.shape[1]} time steps but the encoder produced {time_steps}. "
                    "Pass a (B, num_lang_id_prompts) prompt to broadcast across time instead of "
                    "precomputing the time dimension."
                )
        else:
            raise ValueError(f"Expected a 2D or 3D prompt, got shape {tuple(prompt.shape)}.")

        if prompt.shape[0] != batch_size or prompt.shape[-1] != self.num_lang_id_prompts:
            raise ValueError(
                f"Expected a prompt with batch size {batch_size} and {self.num_lang_id_prompts} classes, "
                f"got shape {tuple(prompt.shape)}."
            )

        # The projection runs in float32 (see ``_Float32PromptProjection``).
        with torch.amp.autocast(device_type=encoded.device.type, enabled=False):
            encoded = self.lang_id_prompt_kernel(torch.cat([encoded.float(), prompt.float()], dim=-1))
        return encoded.to(out_dtype).transpose(1, 2)  # (B, T, D) -> (B, D, T)

    def apply_lang_id_prompt_for_transcribe(self, encoded: Tensor, source_lang: Optional[str]) -> Tensor:
        """Resolve ``source_lang`` and condition the encoder output on it, for whole-batch inference.

        A no-op for models without language-ID prompts, so transcription paths can call it directly.

        Args:
            encoded: Encoder output of shape ``(B, D, T)``.
            source_lang: A key of ``lang_id_prompt_dictionary`` (the language spoken in the audio), or
                None to use the default (``unk``) prompt.

        Returns:
            Encoder output of shape ``(B, D, T)``, conditioned if the model supports prompts.
        """
        if not self.use_lang_id_prompt:
            return encoded

        prompt_id = self.resolve_lang_id_prompt(source_lang)
        prompt = self.create_lang_id_prompt(encoded.shape[0], prompt_id, dtype=encoded.dtype, device=encoded.device)
        return self.apply_lang_id_prompt(encoded, prompt)

    def _assert_lang_id_prompt_supported(self) -> None:
        if not self.use_lang_id_prompt:
            raise ValueError(f"{type(self).__name__} was not trained with language-ID prompt conditioning.")

    def _language_preview(self) -> str:
        available = list(self.lang_id_prompt_dictionary.keys())
        return f"{available[:10]}{'...' if len(available) > 10 else ''}"


class _Float32PromptProjection(torch.nn.Sequential):
    """Prompt projection that stays in float32 through model-wide dtype casts.

    The one-hot prompt contributes a single unit-magnitude feature alongside ``enc_hidden`` encoder
    activations of far larger magnitude. Measured on a released 0.6B unified checkpoint, the
    language-discriminative part of the output is 0.1-1% of its total magnitude, so in bfloat16 it
    falls into the mantissa noise and the projection becomes effectively language-independent while
    decoding silently ignores the requested language. Keeping just this small MLP in float32 restores
    conditioning at negligible cost.

    Device moves are honoured as usual; only the floating point dtype is pinned. Note that the
    dtype change is dropped rather than undone: casting down and back up would already have
    discarded the low-order mantissa bits this class exists to protect.
    """

    def _apply(self, fn, recurse: bool = True):
        def keep_float32(tensor: Tensor) -> Tensor:
            transformed = fn(tensor)
            if tensor.is_floating_point() and transformed.dtype is not torch.float32:
                # Follow the device (and any other placement change) but keep full precision by
                # re-deriving from the original tensor instead of the already-downcast result.
                return tensor.to(device=transformed.device, dtype=torch.float32)
            return transformed

        return super()._apply(keep_float32, recurse=recurse)
