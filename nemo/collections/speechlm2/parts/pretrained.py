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
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import torch
from omegaconf import open_dict
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.
    """
    if pretrained_weights:
        asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        model.perception.load_state_dict(asr.state_dict(), strict=False)
    else:
        model.perception = AudioPerceptionModule(model.cfg.perception).train()


def set_model_dict_for_partial_init(
    pretrained_dict: Dict[str, torch.Tensor], model_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Partially initialize a model's state dictionary with a pretrained state dictionary.
    This function safely copies compatible layers from a pretrained model into a new model,
    ignoring layers with mismatched shapes or missing keys.

    Steps:
        1. Remove layers from the pretrained dictionary if their shape does not match the target model.
        2. Keep only keys that exist in the target model.
        3. Update the model dictionary with the filtered pretrained weights.

    Args:
        pretrained_dict (Dict[str, torch.Tensor]):
            The state dictionary of the pretrained model.
        model_dict (Dict[str, torch.Tensor]):
            The state dictionary of the target model to be partially initialized.

    Returns:
        Dict[str, torch.Tensor]:
            The updated model state dictionary with compatible layers loaded from the pretrained dictionary.

    Example:
        >>> model_dict = model.state_dict()
        >>> pretrained_dict = load_checkpoint("pretrained_model.ckpt")
        >>> model_dict = set_model_dict_for_partial_init(pretrained_dict, model_dict)
        >>> model.load_state_dict(model_dict)
    """
    # 1. Remove layers where pretrained shape differs from model shape
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(f" | > Layer with shape mismatch in the model definition: {k}")

    # 2. Keep only keys that exist in the target model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 3. Update model dictionary with filtered pretrained layers
    model_dict.update(pretrained_dict)
    logging.info(f" | > {len(pretrained_dict)} / {len(model_dict)} layers are restored.")

    return model_dict


def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from disk.

    Supports loading checkpoints stored in either PyTorch (`.ckpt`, `.pt`) or
    SafeTensors (`.safetensors`) formats. All parameters are loaded onto CPU
    regardless of the original device.

    Args:
        checkpoint_path (str):
            Path to the checkpoint file. If the filename contains `.safetensors`,
            it is loaded using the SafeTensors backend; otherwise, it is assumed
            to be a PyTorch checkpoint containing a `state_dict` field.

    Returns:
        dict:
            A state dictionary mapping parameter names to tensors.
    """
    if ".safetensors" in checkpoint_path:
        checkpoint_state = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    return checkpoint_state
