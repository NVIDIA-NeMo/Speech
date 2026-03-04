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

"""
Evaluation and export script for NemotronVoiceChat models.

This script runs validation for a NemotronVoiceChat checkpoint using a
Duplex S2S/STT-style Lhotse dataset. It evaluates the full speech-to-speech
pipeline, including both Duplex STT and duplex TTS model.

Metrics
-------

During validation, the script computes:

- Text BLEU score (reference text vs predicted text)
- ASR BLEU score (reference text vs ASR-transcribed generated speech)

The ASR model used for scoring is defined by the configuration parameter:
    model.scoring_asr

This model is used to transcribe generated speech and compute BLEU-based
speech consistency metrics. The specific ASR checkpoint is fully controlled
via config, in the same way as other parameters such as:

    exp_manager.explicit_log_dir

Outputs
-------

All generated artifacts are saved under:

    exp_manager.explicit_log_dir + "/validation_logs"

The script:

- Saves generated audio files (e.g., under "pred_wavs/")
- Saves per-utterance logs in JSON format
- Saves predicted text, target text, and ASR-transcribed speech

Each validation example is exported as a JSON entry with the following format:

{
    "target_text": "...",
    "pred_text": "...",
    "speech_pred_transcribed": "...",
    "audio_path": "pred_wavs/example.wav"
}

Where:
    target_text:
        Ground-truth target text.

    pred_text:
        Text predicted by the STT/S2S model.

    speech_pred_transcribed:
        Transcription of the generated speech using the ASR model
        defined by ``model.scoring_asr``.

    audio_path:
        Relative path to the generated waveform inside
        exp_manager.explicit_log_dir.

Hugging Face Export
-------------------

If ``hf_export_dir`` is provided, the script exports a Hugging Face–compatible
checkpoint containing BOTH:

- The STT model (``model.stt.model.pretrained_s2s_model``)
- The TTS model (``model.speech_generation.model.pretrained_model``)

Before export, the script can:

- Register multiple speaker reference audios via ``register_speaker_dict``
- Reinitialize the frozen audio prompt projection matrix if
  ``reinit_audio_prompt_frozen_projection=True``

This produces a single HF-style checkpoint directory containing the full
NemotronVoiceChat pipeline (STT + TTS).

Key Config Overrides (commonly used)
-------------------------------------

STT:
    ++model.stt.model.pretrained_s2s_model=...
    ++model.stt.model.pretrained_asr=...

TTS:
    ++model.speech_generation.model.pretrained_model=...
    ++model.speech_generation.model.inference_guidance_enabled=True
    ++model.speech_generation.model.inference_guidance_scale=0.2
    ++model.speech_generation.model.inference_top_p_or_k=0.95
    ++model.speech_generation.model.inference_noise_scale=0.001

Note:
    For export-only runs, it is common to set:
        ++trainer.limit_val_batches=0.0
        ++trainer.max_steps=1

"""
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset
from nemo.collections.speechlm2.models.duplex_ear_tts import load_audio_librosa
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    with trainer.init_module():
        model_config = OmegaConf.to_container(cfg, resolve=True)

        # if available load directly from huggingface like path
        if cfg.get("checkpoint_path", None):
            # instanciate and load the model using from_pretrained 
            model = NemotronVoiceChat.from_pretrained(cfg.checkpoint_path)
        else:
            # load from individual STT and TTS checkpoints
            model = NemotronVoiceChat(model_config)
            # load pretrained checkpoint and rescale the weights if needed
            if model.tts_model.cfg.get("pretrained_model", None):
                model.tts_model.restore_from_pretrained_checkpoint(model.tts_model.cfg.pretrained_model)

    # update model internal configs using the new configs
    model.full_cfg.merge_with(cfg)
    model.cfg.merge_with(cfg.model)
    OmegaConf.save(model.full_cfg, log_dir / "exp_config.yaml")
    model.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

    dataset = DuplexSTTDataset(
        tokenizer=model.stt_model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.stt_model.tokenizer, dataset=dataset)
    # export file to huggingface
    hf_export_dir = model_config.get("hf_export_dir", None)
    if hf_export_dir:
        if model_config.get("register_speaker_dict", None):
            model.tts_model.to(model.device)
            speaker_dict = model_config.get("register_speaker_dict", None)
            for speaker_name in speaker_dict:
                speaker_audio, sr = load_audio_librosa(speaker_dict[speaker_name])
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
                logging.info(f"Speaker {speaker_name} registered !")

        if cfg.get("reinit_audio_prompt_frozen_projection", False):
            D = model.tts_model.tts_model.hidden_size
            Q, _ = torch.linalg.qr(
                torch.randn(
                    D,
                    D,
                    device=model.tts_model.tts_model.audio_prompt_projection_W.device,
                    dtype=model.tts_model.tts_model.audio_prompt_projection_W.dtype,
                )
            )
            model.tts_model.tts_model.audio_prompt_projection_W.copy_(Q)
            logging.info("Audio frozen projection reinited !")

        model.save_pretrained(hf_export_dir, config=model_config)
        logging.info("Hugging face compatible checkpoint saved at:", hf_export_dir)

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    inference()
