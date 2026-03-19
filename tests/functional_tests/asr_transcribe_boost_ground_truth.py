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

from dataclasses import dataclass

import torch
from omegaconf import MISSING, open_dict

from nemo.collections.asr.inference.utils.manifest_io import prepare_audio_data
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import get_auto_inference_device
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exceptions import NeMoBaseException


@dataclass
class TranscriptionBoostGroundTruthConfig:
    dataset_manifest: str = MISSING
    model_path: str | None = None  # Path to a .nemo file
    pretrained_name: str | None = None  # Name of a pretrained model
    batch_size: int = 128
    boosting_alpha: float = 1.0
    output_filename: str | None = None
    device: str | None = None


@hydra_runner(config_name="TranscriptionBoostGroundTruthConfig", schema=TranscriptionBoostGroundTruthConfig)
def main(cfg: TranscriptionBoostGroundTruthConfig):
    """
    Script to test per-utterance boosting. We boost ground truth tests with `asr_model.transcribe(...)`.
    Sanity check: boosting ground truth should result in better WER (for CTC and RNN-T –
    not always 0 even with high boosting weight if the transcription is inconsistent with the audio)
    """
    # Reading audio filepaths
    audio_filepaths, manifest, _, _ = prepare_audio_data(cfg.dataset_manifest, sort_by_duration=True)
    logging.info(f"Found {len(audio_filepaths)} audio files")
    assert manifest is not None, "This script works only with manifest"
    device = torch.device(cfg.device) if cfg.device is not None else get_auto_inference_device()

    asr_model: EncDecRNNTBPEModel
    if cfg.model_path is not None:
        asr_model = EncDecRNNTBPEModel.restore_from(cfg.model_path)
    elif cfg.pretrained_name is not None:
        asr_model = EncDecRNNTBPEModel.from_pretrained(model_name=cfg.pretrained_name)
    else:
        raise NeMoBaseException("Either `model_path` or `pretrained_name` should be not None")
    assert isinstance(asr_model, EncDecRNNTBPEModel), "Only RNN-T model supported"
    asr_model.to(device)

    # Change Decoding Config: ensure greedy_batch + label-looping enabled
    with open_dict(asr_model.cfg.decoding):
        asr_model.cfg.decoding.strategy = "greedy_batch"
        asr_model.cfg.decoding.greedy.loop_labels = True
        asr_model.cfg.decoding.greedy.enable_per_stream_biasing = True
    asr_model.change_decoding_strategy(asr_model.cfg.decoding)

    batch_size = cfg.batch_size
    for start_batch_i in range(0, len(manifest), batch_size):
        end_batch_i = min(start_batch_i + batch_size, len(manifest))
        # use transcribe with empty partial hypotheses with boosting requests with one phrase
        results = asr_model.transcribe(
            audio=audio_filepaths[start_batch_i : start_batch_i + batch_size],
            partial_hypothesis=[
                Hypothesis.empty_with_biasing_cfg(
                    biasing_cfg=BiasingRequestItemConfig(
                        boosting_model_cfg=BoostingTreeModelConfig(
                            key_phrases_list=[manifest[i]["text"]],
                        ),
                        boosting_model_alpha=cfg.boosting_alpha,
                    ),
                )
                for i in range(start_batch_i, end_batch_i)
            ],
            return_hypotheses=True,
            batch_size=end_batch_i - start_batch_i,
        )

        for i, result in zip(range(start_batch_i, end_batch_i), results):
            manifest[i]["pred_text"] = result.text

    cer = word_error_rate(
        hypotheses=[record["pred_text"] for record in manifest],
        references=[record["text"] for record in manifest],
        use_cer=True,
    )
    wer = word_error_rate(
        hypotheses=[record["pred_text"] for record in manifest],
        references=[record["text"] for record in manifest],
        use_cer=False,
    )
    logging.info(f"Dataset WER/CER {wer:.2%}/{cer:.2%}")

    # Dump the transcriptions to a output file
    if cfg.output_filename is not None:
        write_manifest(output_path=cfg.output_filename, target_manifest=manifest)

    logging.info("Done!")


if __name__ == "__main__":
    main()
