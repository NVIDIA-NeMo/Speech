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

from __future__ import annotations


from typing import TYPE_CHECKING
import torch

from nemo.collections.asr.inference.utils.device_utils import setup_device
from nemo.collections.common.prompts import PromptFormatter

if TYPE_CHECKING:
    from nemo.collections.speechlm2.models import SALM


class SALMASRInferenceWrapper:

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        device_id: int = 0,
        compute_dtype: str = 'bfloat16',
        use_amp: bool = True,
    ):
        """
        Initialize the SALM ASR inference wrapper.
        Args:
            model_name: (str) model name at Hugging Face or NGC cloud.
            device: (str) device to run the model on.
            device_id: (int) device ID to run the model on.
            compute_dtype: (str) compute dtype to run the model on.
            use_amp: (bool) Use Automatic Mixed Precision
        """

        self.device_str, self.device_id, self.compute_dtype = setup_device(device.strip(), device_id, compute_dtype)
        self.use_amp = use_amp
        self.device = torch.device(self.device_str)
        self.salm_model = self.load_model(model_name, self.device)
        self.audio_locator_tag = self.salm_model.audio_locator_tag
        self.tokenizer = self.salm_model.tokenizer
        self.set_dither_to_zero()

    @property
    def eos_token_ids(self) -> list[int]:
        """Returns the end of sentence token ids."""
        return [self.salm_model.text_eos_id]

    @property
    def word_separator(self) -> str:
        """Returns word separator."""
        return ' '

    @property
    def word_separator_ids(self) -> list[int]:
        """Returns the word separator token ids."""
        return self.tokenizer.text_to_ids(self.word_separator)

    @staticmethod
    def load_model(model_name: str, device: torch.device) -> SALM:
        """
        Load the SALM model.
        Args:
            model_name: (str) model name at Hugging Face or NGC cloud.
            device: (torch.device) device to load the model on.
        Returns:
            (SALM) loaded SALM model.
        """
        try:
            from nemo.collections.speechlm2.models import SALM

            model = SALM.from_pretrained(model_name).eval()
            model.to(device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def get_window_stride(self) -> float:
        """Returns the window stride of the model."""
        return self.salm_model.cfg.perception.preprocessor.window_stride

    def get_subsampling_factor(self) -> int:
        """Returns the subsampling factor of the model."""
        return self.salm_model.cfg.perception.encoder.subsampling_factor

    def get_model_stride(self, in_secs: bool = False, in_milliseconds: bool = False) -> float:
        """
        Returns the model stride in seconds or milliseconds.
        Args:
            in_secs: (bool) Whether to return the model stride in seconds.
            in_milliseconds: (bool) Whether to return the model stride in milliseconds.
        Returns:
            (float) model stride in seconds or milliseconds.
        """
        if in_secs and in_milliseconds:
            raise ValueError("Cannot return both seconds and milliseconds at the same time.")
        token_duration = self.salm_model.token_equivalent_duration
        if in_secs:
            return token_duration
        if in_milliseconds:
            return token_duration * 1000
        return token_duration

    def set_dither_to_zero(self) -> None:
        """Sets the dither to zero."""
        self.salm_model.cfg.perception.preprocessor.dither = 0.0
        self.salm_model.perception.preprocessor.featurizer.dither = 0.0

    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """
        Generate the model output.
        Args:
            prompts: (list[list[dict[str]]] | torch.Tensor) List of prompts or token ids.
            audios: (torch.Tensor) Audio tensor of shape (batch_size, num_samples).
            audio_lens: (torch.Tensor) Audio length tensor of shape (batch_size).
            max_new_tokens: (int) Maximum number of new tokens to generate.
        Returns:
            (torch.Tensor) Model output.
        """
        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            answer_ids = self.salm_model.generate(
                prompts=prompts,
                audios=audios,
                audio_lens=audio_lens,
                max_new_tokens=max_new_tokens,
            )
        return answer_ids

    def preprocess_prompts(self, prompts: list[list[dict[str]]]) -> torch.Tensor:
        """
        Convert the prompts to token ids.
        Args:
            prompts: (list[list[dict[str]]]) List of prompts.
        Returns:
            (torch.Tensor) Token ids of size (batch_size, max_prompt_length).
        """
        from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors

        formatter = PromptFormatter.resolve(self.salm_model.cfg.prompt_format)(self.tokenizer)
        tokens = left_collate_vectors(
            [formatter.encode_dialog(turns=prompt)["input_ids"] for prompt in prompts],
            padding_value=self.salm_model.text_pad_id,
        ).to(self.device)
        return tokens
