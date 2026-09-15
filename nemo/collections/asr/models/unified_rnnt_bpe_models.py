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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.mixins import LangIdPromptMixin, TranscribeConfig, TranscriptionReturnType
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType, SpectrogramType

__all__ = ['EncDecUnifiedRNNTBPEModel', 'UnifiedTranscribeConfig']


@dataclass
class UnifiedTranscribeConfig(TranscribeConfig):
    """Transcription config for unified models, adding the spoken language of the audio."""

    # A key of the model's ``language_dictionary`` (e.g. "en-US"). This is the language spoken in the
    # audio, not a translation target: unified models are ASR-only. None uses the model's default.
    source_lang: Optional[str] = None


class EncDecUnifiedRNNTBPEModel(LangIdPromptMixin, EncDecRNNTBPEModel):
    """RNNT BPE model that conditions on the spoken language of the audio ("unified" model).

    Unified models are trained on many languages at once and take the language as an input, which
    both removes the need to pick a per-language checkpoint and lets a caller resolve the ambiguity
    that a purely acoustic model cannot. The conditioning itself lives in
    :class:`~nemo.collections.asr.parts.mixins.lang_id_prompt.LangIdPromptMixin`: a one-hot language
    vector is projected into the encoder output.

    Select a language with :meth:`transcribe`'s ``source_lang``, or with
    :meth:`~nemo.collections.asr.parts.mixins.lang_id_prompt.LangIdPromptMixin.set_inference_language`
    for streaming, where there is no single call to carry it. Omitting it falls back to the model's
    :attr:`~nemo.collections.asr.parts.mixins.lang_id_prompt.LangIdPromptMixin.default_language`.

    Inference only. The projection is part of the trained forward pass, but no training step in this
    class conditions on it, so construction with a trainer attached is refused rather than silently
    training an unconditioned model. Language-conditioned training arrives with the cache-aware
    unified architecture.

    Enabled by the model config carried in the checkpoint:

    .. code-block:: yaml

        model:
          model_defaults:
            initialize_lang_id_prompt: true
            num_lang_id_prompts: 128
            language_dictionary: {en-US: 0, de-DE: 9, ..., unk: 127}
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.setup_lang_id_prompt()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()

        types = {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }
        if self.use_lang_id_prompt:
            # Optional so that callers which condition the encoder output themselves — buffered and
            # chunked streaming apply the prompt per chunk — can still call forward() without one.
            types["lang_id_prompt"] = NeuralType(('B', 'D'), LabelsType(), optional=True)
        return types

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        lang_id_prompt=None,
    ):
        """Encoder forward pass, optionally conditioned on a language.

        Args:
            input_signal: Batch of raw audio signals of shape [B, T].
            input_signal_length: Vector of length B with the individual audio lengths.
            processed_signal: Batch of processed audio signals of shape [B, D, T].
            processed_signal_length: Vector of length B with the individual processed lengths.
            lang_id_prompt: Optional one-hot language prompt of shape (B, num_lang_id_prompts),
                broadcast across time. Build it with
                :meth:`~nemo.collections.asr.parts.mixins.lang_id_prompt.LangIdPromptMixin.create_lang_id_prompt`.

        Returns:
            A tuple of the encoder output of shape [B, D, T] and its lengths of shape [B].
        """
        encoded, encoded_len = super().forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

        if lang_id_prompt is not None:
            encoded = self.apply_lang_id_prompt(encoded, lang_id_prompt)

        return encoded, encoded_len

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        use_lhotse: bool = True,
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        source_lang: Optional[str] = None,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """Transcribe audio, conditioning the model on the language spoken in it.

        Args and return value match
        :meth:`~nemo.collections.asr.models.rnnt_models.EncDecRNNTModel.transcribe`, with one
        addition:

        Args:
            source_lang: A key of the model's ``language_dictionary`` (e.g. ``"en-US"``), naming the
                language spoken in the audio. None falls back to the model's
                :attr:`~nemo.collections.asr.parts.mixins.lang_id_prompt.LangIdPromptMixin.default_language`.
        """
        if override_config is None:
            override_config = UnifiedTranscribeConfig(
                use_lhotse=use_lhotse,
                batch_size=batch_size,
                # The base class derives this from `timestamps` too, but it does so on a local it
                # cannot pass on once an override config is in play.
                return_hypotheses=return_hypotheses or bool(timestamps),
                partial_hypothesis=partial_hypothesis,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                timestamps=timestamps,
                source_lang=source_lang,
            )
        elif source_lang is not None:
            override_config.source_lang = source_lang

        return super().transcribe(
            audio=audio,
            use_lhotse=use_lhotse,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            partial_hypothesis=partial_hypothesis,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=override_config,
        )

    @classmethod
    def get_transcribe_config(cls) -> UnifiedTranscribeConfig:
        """Return the transcription config this model understands, including ``source_lang``."""
        return UnifiedTranscribeConfig()

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """This method returns a list of pre-trained models which can be instantiated directly."""
        return []

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig) -> Dict[str, Any]:
        encoded, encoded_len = super().forward(input_signal=batch[0], input_signal_length=batch[1])
        encoded = self.apply_lang_id_prompt_for_transcribe(encoded, getattr(trcfg, 'source_lang', None))
        return dict(encoded=encoded, encoded_len=encoded_len)
