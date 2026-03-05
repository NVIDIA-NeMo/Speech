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
Evaluation script for NemotronVoiceChat models.

This script runs validation for a NemotronVoiceChat checkpoint using a 
Duplex S2S/STT-style Lhotse dataset. It evaluates the full speech-to-speech 
pipeline, including both the Duplex STT and Duplex TTS models.

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

Arguments
---------
For a complete configuration reference, please look at the example config located at:
    examples/speechlm2/conf/nemotron_voicechat.yaml

cfg : omegaconf.DictConfig
    The main Hydra configuration object defining the evaluation parameters. 
    It is expected to contain the following top-level configurations:
    
    checkpoint_path (str | null)
        Path to the pre-trained NemotronVoiceChat checkpoint for evaluation.

    model (DictConfig)
        Model-specific settings encompassing both STT and TTS subsystems. Key parameters include:
        * scoring_asr (str): The ASR model name/path used to evaluate generated speech (e.g., 'stt_en_fastconformer_transducer_large').
        * inference_speaker_reference (str | null): Path to the reference audio used to condition the speaker's voice. Set to `null` if using `inference_speaker_name`.
        * inference_speaker_name (str): Named speaker identifier (e.g., 'Megan'); overrides `inference_speaker_reference`.
        * stt (DictConfig): Sub-config for the `DuplexSTTModel` (e.g., `eval_text_turn_taking`).
        * speech_generation (DictConfig): Sub-config for the `DuplexEARTTS` model. Includes codec configs, EAR-TTS backbone, and inference behavior like `inference_guidance_scale` (CFG) and `inference_noise_scale` (sampling temperature for the MoG head).

    data (DictConfig)
        Configuration for the data pipelines, datasets, and DataModule. Key parameters include:
        * source_sample_rate (int): Sample rate of the input/user audio (e.g., 16000).
        * target_sample_rate (int): Sample rate of the generated output audio (e.g., 22050).
        * frame_length (float): Duration of audio frames in seconds (e.g., 0.08).
        * input_roles (list): Conversation roles mapped to the input prompt (e.g., ["user", "User"]).
        * output_roles (list): Conversation roles targeted for model generation (e.g., ["agent", "Assistant"]).
        * validation_ds (DictConfig): Paths and settings for the Lhotse validation shards (e.g., `shar_path`, `batch_size`). Note that the data format for `data.validation_ds.evaluation_set` must follow the `duplexs2s-dataset-structure`. For detailed specifications, see: https://docs.nvidia.com/nemo/speech/nightly/speechlm2/datasets.html#duplexs2s-dataset-structure

    exp_manager (DictConfig)
        Experiment manager configurations for logging. Must include:
        * name (str): Experiment name (e.g., 'nemotron-voicechat-eval').
        * explicit_log_dir (str): The root directory where output artifacts and metric logs are saved.

    trainer (DictConfig)
        PyTorch Lightning Trainer parameters dictating hardware usage. Key settings include `devices`, `num_nodes`, `limit_val_batches` (fraction of dataset to evaluate), and `precision` (e.g., 32).

Example Run
-----------
You can run the evaluation script and override parameters dynamically using Hydra command-line flags. 
Here is an example execution using dummy paths:

    python /path/to/nemo/examples/speechlm2/nemotron_voicechat_eval.py \
        --config-path=examples/speechlm2/conf/ \
        --config-name=nemotron_voicechat.yaml \
        exp_manager.name="Nemotron_VoiceChat_Eval" \
        ++model.stt.model.eval_text_turn_taking=True \
        ++checkpoint_path="/path/to/nemotron_voicechat_ckpt/" \
        ++model.inference_speaker_reference=null \
        ++model.inference_speaker_name="Megan" \
        ++model.speech_generation.model.inference_guidance_scale=0.2 \
        ++model.speech_generation.model.inference_guidance_enabled=True \
        ++model.speech_generation.model.inference_top_p_or_k=0.95 \
        ++model.speech_generation.model.inference_noise_scale=0.001 \
        trainer.num_nodes=1 \
        exp_manager.explicit_log_dir="/path/to/results_dir/Nemotron_VoiceChat_Eval/" \
        data.validation_ds.batch_size=2 \
        data.validation_ds.datasets.evaluation_set.shar_path="/path/to/validation_dataset/" \
        ++trainer.limit_val_batches=1.0 \
        ++trainer.precision=32 \
        data.validation_ds.seed=42

Outputs
-------
All generated artifacts are saved under:
    exp_manager.explicit_log_dir + "/validation_logs"

The script:
- Saves generated audio files
- Saves per-utterance logs in JSON format via `ResultsLogger`
- Saves predicted text, target text, and ASR-transcribed speech

Each validation example is exported as a JSON entry with the following format:
{
    "target_text": "...",
    "pred_text": "...",
    "speech_pred_transcribed": "...",
    "audio_path": "pred_wavs/example.wav"
}

Where:
    target_text: Ground-truth target text.
    pred_text: Text predicted by the STT/S2S model.
    speech_pred_transcribed: Transcription of the generated speech using the ASR model defined by `model.scoring_asr`.
    audio_path: Relative path to the generated waveform inside exp_manager.explicit_log_dir.
"""

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="nemotron_voicechat")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    with trainer.init_module():
        # instanciate and load the model using from_pretrained
        model = NemotronVoiceChat.from_pretrained(cfg.checkpoint_path).eval()

    # update model internal configs using the new configs
    model.full_cfg.merge_with(cfg)
    model.cfg.merge_with(cfg.model)
    OmegaConf.save(model.full_cfg, log_dir / "exp_config.yaml")
    model.validation_save_path = os.path.join(log_dir, "validation_logs")

    dataset = DuplexSTTDataset(
        tokenizer=model.stt_model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.stt_model.tokenizer, dataset=dataset)
    trainer.validate(model, datamodule)


if __name__ == "__main__":
    inference()
