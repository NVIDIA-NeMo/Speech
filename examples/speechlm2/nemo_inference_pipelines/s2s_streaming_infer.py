# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
S2S Streaming Inference Client

Usage:
    python s2s_streaming_infer.py \
        audio_file=/path/to/audio_or_directory \
        s2s.model_path=/path/to/eartts_ckpt \
        s2s.speaker_reference=/path/to/speaker.wav \
        streaming.chunk_size_in_secs=0.08 \
        streaming.buffer_size_in_secs=5.6
"""

import hydra
from omegaconf import DictConfig

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.utils.stepprogressbar import StepProgressBar
from nemo.collections.speechlm2.inference.utils.audio_data import (
    calculate_durations_incl_padding,
    dump_output_json,
    prepare_audio_data,
)
from nemo.collections.speechlm2.inference.utils.pipeline_utils import clean_pred_text
from nemo.utils import logging
from nemo.utils.timers import SimpleTimer


@hydra.main(config_path="./conf", config_name="s2s_streaming", version_base=None)
def main(cfg: DictConfig):
    default_system_prompt = cfg.get("s2s", {}).get("system_prompt", None)
    audio_filepaths, options, ground_truths = prepare_audio_data(
        cfg.audio_file, default_system_prompt=default_system_prompt, sort_by_duration=False,
    )
    logging.info(f"Found {len(audio_filepaths)} audio files to generate")

    pipeline = S2SPipelineBuilder.build_pipeline(cfg)

    progress_bar = StepProgressBar.from_audio_filepaths(
        audio_filepaths,
        chunk_size_in_secs=pipeline.chunk_size_in_secs,
        pad_audio_to_sec=cfg.get("pad_audio_to_sec"),
        pad_silence_ratio=cfg.get("pad_silence_ratio"),
        pad_audio_by_sec=cfg.get("pad_audio_by_sec"),
    )

    timer = SimpleTimer()
    timer.start()
    output = pipeline.run(audio_filepaths, options=options, progress_bar=progress_bar)
    timer.stop()
    exec_dur = timer.total_sec()
    logging.info(f"Generated {len(audio_filepaths)} files in {exec_dur:.2f}s")

    data_dur = sum(calculate_durations_incl_padding(
        audio_filepaths, cfg.get("pad_audio_to_sec"), cfg.get("pad_silence_ratio"), cfg.get("pad_audio_by_sec"),
    ))
    rtfx = data_dur / exec_dur if exec_dur > 0 else float('inf')
    logging.info(f"RTFX: {rtfx:.2f} ({data_dur:.2f}s / {exec_dur:.2f}s)")

    # Compute WER when ground-truth texts are available (micro-average,
    # matching the offline eval in speechlm2.parts.metrics.asr_cer_wer)
    asr_texts = output.asr_texts_with_timestamps or [None] * len(audio_filepaths)
    all_refs, all_hyps = [], []
    for gt, asr_text in zip(ground_truths, asr_texts):
        if gt and asr_text:
            cleaned_gt = clean_pred_text(gt)
            cleaned_pred = clean_pred_text(asr_text)
            if cleaned_gt.strip() and cleaned_pred.strip():
                all_refs.append(cleaned_gt)
                all_hyps.append(cleaned_pred)
    if all_refs:
        wer = word_error_rate(hypotheses=all_hyps, references=all_refs)
        logging.info(f"WER: {wer:.4f} ({wer * 100:.2f}%), n={len(all_refs)}")

    output_dir = cfg.get("output_dir", "./generated")
    dump_output_json(audio_filepaths, output, output_dir, options, ground_truths)
    logging.info(f"Transcriptions written to {output_dir}/output_processed.json and {output_dir}/output_raw.json")


if __name__ == "__main__":
    main()
