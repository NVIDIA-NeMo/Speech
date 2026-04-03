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
from omegaconf import OmegaConf, open_dict
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


def load_pretrained_hf(
    model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32, trust_remote_code: bool = False
):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Args:
        model_path_or_name: Path or name of the model to load
        pretrained_weights: Whether to load pretrained weights (True) or random init (False)
        dtype: Data type for the model
        trust_remote_code: Whether to trust remote code when loading model (needed for some models like Nemotron)
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)


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

    If user config specifies encoder parameters, they will override the pretrained model's config.
    """
    if pretrained_weights:
        # Save user-specified encoder config before loading pretrained model
        user_encoder_config = {}

        if 'encoder' in model.cfg.perception:
            user_encoder_config = OmegaConf.to_container(model.cfg.perception.encoder, resolve=True)

        asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size
            # Override with user-specified encoder parameters, e.g. initializiing a non-causal encoder for causal setup.
            if user_encoder_config:
                for key, value in user_encoder_config.items():
                    if value is not None:  # Only override if user explicitly set a value
                        model.cfg.perception.encoder[key] = value
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


def _load_checkpoint_state(checkpoint_path: str) -> dict:
    """Load checkpoint state dict from a file or HF directory.

    Args:
        checkpoint_path: Path to checkpoint file or HF directory with model.safetensors
    """
    import os

    if os.path.isdir(checkpoint_path):
        from safetensors.torch import load_file

        return load_file(os.path.join(checkpoint_path, "model.safetensors"))
    else:
        return torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']


def init_perception_from_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load perception module from another STT/S2S checkpoint.

    Args:
        model: The model whose perception module will be initialized
        checkpoint_path: Path to checkpoint file or HF directory
    """
    if checkpoint_path is None:
        return

    from nemo.utils import logging

    logging.info(f"Loading perception from checkpoint: {checkpoint_path}")
    checkpoint_state = _load_checkpoint_state(checkpoint_path)

    checkpoint_state = {k.replace("perception.", ""): v for k, v in checkpoint_state.items() if "perception." in k}
    checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.perception.state_dict())
    model.perception.load_state_dict(checkpoint_state, strict=True)


def init_model_from_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load full model state from a checkpoint.

    Args:
        model: The model to initialize
        checkpoint_path: Path to checkpoint file or HF directory
    """
    if checkpoint_path is None:
        return

    from nemo.utils import logging

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint_state = _load_checkpoint_state(checkpoint_path)

    checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.state_dict())
    model.load_state_dict(checkpoint_state, strict=True)


def load_pretrained_model(model: torch.nn.Module, checkpoint_path: str):
    """Load a pretrained S2S model from a checkpoint path.

    Supports both incremental loading from safetensors (for large models to avoid OOM)
    and standard loading from various checkpoint formats.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint file or HF directory
    """
    if checkpoint_path is None:
        return

    import gc
    import os
    from nemo.utils import logging

    logging.info(f"Loading pretrained s2s model from {checkpoint_path}")

    if os.path.isdir(checkpoint_path) and model.cfg.get("incremental_loading", False):
        # Hugging Face format with incremental loading
        from safetensors import safe_open

        # Load tensors incrementally to avoid OOM
        model_state_dict = model.state_dict()
        loaded_keys = []
        missing_keys = []

        with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt", device="cpu") as f:
            available_keys = f.keys()
            for key in available_keys:
                if key in model_state_dict:
                    # Load tensor and copy to model parameter
                    tensor = f.get_tensor(key)
                    model_state_dict[key].copy_(tensor)
                    loaded_keys.append(key)
                    del tensor  # Free memory immediately
                else:
                    missing_keys.append(key)

                # Periodic garbage collection for very large models
                if len(loaded_keys) % 100 == 0:
                    gc.collect()

        logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
        if missing_keys:
            logging.warning(f"Keys in checkpoint but not in model: {len(missing_keys)} keys")

        del model_state_dict
        gc.collect()
    else:
        init_model_from_checkpoint(model, checkpoint_path)


def maybe_load_pretrained_models(model: torch.nn.Module):
    """
    Optionally load pretrained model weights based on configuration.

    Checks for and loads:
    - ``pretrained_perception_from_s2s``: Perception module weights from another S2S checkpoint
    - ``pretrained_s2s_model``: Full S2S model weights from a checkpoint (supports incremental loading)
    """
    if model.cfg.get("pretrained_perception_from_s2s", None):
        init_perception_from_checkpoint(model, model.cfg.pretrained_perception_from_s2s)

    if model.cfg.get("pretrained_s2s_model", None):
        load_pretrained_model(model, model.cfg.pretrained_s2s_model)
