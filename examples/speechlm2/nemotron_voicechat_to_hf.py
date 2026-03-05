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
"""
Utility script for exporting Duplex Speech-to-Chat models to HuggingFace format.

This exporter supports:
- Standard PyTorch checkpoints (non-distributed)
- Distributed checkpoints (FSDP / TP / torch.distributed checkpoint directories)
- safetensors checkpoints

The script loads STT and TTS checkpoints, constructs a joint NemotronVoiceChat
model instance, and saves a HuggingFace-compatible checkpoint that can be
loaded via:

    NemotronVoiceChat.from_pretrained(path)

Arguments
---------
tts_ckpt_path : str
    Path to the TTS checkpoint.
    - Can be a standard PyTorch checkpoint (.ckpt or .pt)
    - Can be a directory containing distributed checkpoints (FSDP/TP)
    - Can be a safetensors file

tts_ckpt_config : str
    Path to the experiment configuration used to instantiate the TTS model.
    This configuration defines architecture, hyperparameters, and data settings
    required to reconstruct the model before loading weights.

stt_ckpt_path : str
    Path to the STT (speech-to-text) checkpoint.
    - Supports PyTorch checkpoints
    - Supports distributed checkpoints
    - Supports safetensors

stt_ckpt_config : str
    Path to the experiment configuration for the STT model.
    Used to reconstruct the STT module before loading weights.

output_dir : str
    Directory where the HuggingFace-compatible checkpoint will be stored.
    After export, the model can be loaded via:

        NemotronVoiceChat.from_pretrained(output_dir)

dtype : str (optional, default="float32")
    Target dtype for storing parameters.
    Typical values: "float32", "float16", "bfloat16".
    Controls the precision of saved weights.

register_speaker_dict : Dict[str, str] (optional)
    Dictionary mapping speaker names to reference audio paths.
    Used for speaker registration and voice cloning at inference time.

    Example:
        {
            "Megan": "/path/to/megan_reference.wav",
            "Emma": "/path/to/emma_reference.wav"
        }

    Each entry:
    - Key: Speaker identifier (string)
    - Value: Path to an audio file containing the speaker’s voice

    The exporter loads the reference audio, resamples it to the model’s target
    sample rate, and registers it as an audio prompt latent so the model can
    generate speech in that speaker’s voice.

reinit_audio_prompt_frozen_projection : bool (optional, default=False)
    If True, reinitializes the frozen audio prompt projection layer.

    Purpose:
    - Disables voice cloning effects during inference
    - Useful when exporting models where voice cloning is not desired

    When enabled:
    - The projection matrix is replaced with random weights
    - Speaker conditioning is effectively disabled

Notes
-----
- Distributed checkpoints are detected when `tts_ckpt_path` or `stt_ckpt_path`
  points to a directory containing checkpoint shards.
- safetensors checkpoints are supported when the path ends with `.safetensors`.
- Non-distributed PyTorch checkpoints are loaded normally with state_dict mapping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.models.duplex_ear_tts import load_audio_librosa
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class HfExportConfig:
    """
    Configuration for exporting Speech-to-Chat/TTS models to HuggingFace format.

    Attributes:
        tts_ckpt_path:
            Path to the TTS checkpoint (PyTorch Lightning ckpt or directory).
        tts_ckpt_config:
            Path to the experiment config used to instantiate the TTS model.
        stt_ckpt_path:
            Path to the STT checkpoint (PyTorch Lightning ckpt or directory).
        stt_ckpt_config:
            Path to the experiment config used to instantiate the STT model.
        output_dir:
            Directory where the HuggingFace-compatible checkpoint will be stored.
        dtype:
            Target dtype for storing parameters (default: float32).
        register_speaker_dict:
            Dictionary mapping speaker names to reference audio paths.
        reinit_audio_prompt_frozen_projection:
            If True, reinitialize the frozen projection to disable voice cloning.
    """

    tts_ckpt_path: str
    tts_ckpt_config: str
    stt_ckpt_path: str
    stt_ckpt_config: str
    output_dir: str
    dtype: str = "float32"
    register_speaker_dict: Dict[str, str] = field(default_factory=dict)
    reinit_audio_prompt_frozen_projection: bool = False


def load_checkpoint_by_module(model: torch.nn.Module, checkpoint_path: str, module_name: str):
    """
    Load checkpoint weights into a submodule of the model.

    Supports:
        - Distributed checkpoints (FSDP/TP)
        - safetensors checkpoints
        - standard PyTorch checkpoints

    Args:
        model:
            Parent torch module.
        checkpoint_path:
            Path to checkpoint or distributed checkpoint directory.
        module_name:
            Name of the submodule (e.g., "stt_model" or "tts_model").
    """

    module = getattr(model, module_name)

    # Distributed checkpoint (FSDP/TP)
    if Path(checkpoint_path).is_dir():
        from torch.distributed.checkpoint import load

        state_dict = {"state_dict": module.state_dict()}
        load(state_dict, checkpoint_id=checkpoint_path)
        module.load_state_dict(state_dict["state_dict"])
        return

    # safetensors checkpoint
    if ".safetensors" in checkpoint_path:
        if hasattr(model, "init_from_safetensors_ckpt"):
            model.init_from_safetensors_ckpt(checkpoint_path, prefix=f"{module_name}.")
            return
        raise RuntimeError("Model does not support safetensors loader.")

    # Standard PyTorch checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt_data.get("state_dict", ckpt_data)

    try:
        module.load_state_dict(sd, strict=False)
    except Exception:
        # Fallback: filter keys matching module prefix
        prefix = module_name + "."
        filtered = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        module.load_state_dict(filtered, strict=False)


@hydra_runner(config_name="HfExportConfig", schema=HfExportConfig)
def main(cfg: HfExportConfig):
    """
    Entry point for exporting Nemotron VoiceChat checkpoint.

    Steps:
        1. Load STT and TTS experiment configs.
        2. Instantiate NemotronVoiceChat model.
        3. Load STT/TTS checkpoints.
        4. Optionally register speaker audio prompts.
        5. Save HuggingFace-compatible checkpoint.

    Args:
        cfg:
            Hydra configuration object containing export parameters.
    """

    # Load STT configuration
    stt_model_cfg = OmegaConf.load(cfg.stt_ckpt_config)

    # Prevent model from reloading pretrained perception module during export.
    # Some STT configs define these fields.
    # If it exists, we set it to None so the exporter uses the checkpoint weights only.
    if hasattr(stt_model_cfg.model, "pretrained_perception_from_s2s"):
        stt_model_cfg.model.pretrained_perception_from_s2s = None
    if hasattr(stt_model_cfg.model, "pretrained_s2s_model"):
        stt_model_cfg.model.pretrained_s2s_model = None

    # Load TTS experiment configuration
    tts_model_cfg = OmegaConf.load(cfg.tts_ckpt_config)

    # Disable codec model reloading during export.
    # Some recipes reference an external pretrained codec.
    # Setting this to None forces the exporter to use checkpoint weights only,
    # avoiding additional model downloads or initialization.
    if hasattr(tts_model_cfg.model, "pretrained_codec_model"):
        tts_model_cfg.model.pretrained_codec_model = None

    # Joint model configuration
    model_cfg = {
        "model": {
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "stt": stt_model_cfg,
            "speech_generation": tts_model_cfg,
        },
        "data": {
            "frame_length": 0.08,
            "source_sample_rate": stt_model_cfg.data.source_sample_rate,
            "target_sample_rate": tts_model_cfg.data.target_sample_rate,
        },
        "exp_manager": {"explicit_log_dir": " "},
        "torch_dtype": cfg.dtype,
    }

    model_cfg = OmegaConf.create(model_cfg)

    # Instantiate model
    model = NemotronVoiceChat(OmegaConf.to_container(model_cfg, resolve=True))

    # Load checkpoints
    load_checkpoint_by_module(model, cfg.stt_ckpt_path, "stt_model")
    load_checkpoint_by_module(model, cfg.tts_ckpt_path, "tts_model")

    # Register inference speakers (voice cloning)
    if cfg.register_speaker_dict:
        model.tts_model.to(model.device)
        for speaker_name, audio_path in cfg.register_speaker_dict.items():
            speaker_audio, sr = load_audio_librosa(audio_path)
            speaker_audio = resample(speaker_audio, sr, model.tts_model.target_sample_rate).to(model.device)

            speaker_audio_lens = (
                torch.tensor([speaker_audio.size(1)]).long().repeat(speaker_audio.size(0)).to(model.device)
            )

            model.tts_model.set_audio_prompt_lantent(
                speaker_audio,
                speaker_audio_lens,
                system_prompt=None,
                batch_size=1,
                name=speaker_name,
            )
            logging.info(f"Speaker {speaker_name} registered!")

    # Optionally reinitialize projection (disables voice cloning)
    if cfg.reinit_audio_prompt_frozen_projection:
        D = model.tts_model.tts_model.hidden_size
        model.tts_model.tts_model.audio_prompt_projection_W.copy_(
            torch.randn(
                D,
                D,
                device=model.tts_model.tts_model.audio_prompt_projection_W.device,
                dtype=model.tts_model.tts_model.audio_prompt_projection_W.dtype,
            )
        )
        logging.info("Audio frozen projection reinitialized!")

    # Cast model to target dtype and save HuggingFace checkpoint
    model = model.to(getattr(torch, cfg.dtype))
    model.save_pretrained(cfg.output_dir, config=OmegaConf.to_container(model_cfg, resolve=True))

    logging.info(f"HuggingFace-compatible checkpoint saved at: {cfg.output_dir}")


if __name__ == "__main__":
    main()
