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

from pathlib import Path

import hydra

from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.utils.manifest_io import calculate_duration, dump_output, get_audio_filepaths
from nemo.collections.asr.inference.utils.progressbar import TQDMProgressBar
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.utils import logging
from nemo.utils.timers import SimpleTimer

BASE_PATH = Path(__file__).parent


@hydra.main(version_base=None)
def main(cfg):
    # Reading audio filepaths
    audio_filepaths, manifest = get_audio_filepaths(cfg.audio_file, sort_by_duration=True)
    logging.info(f"Found {len(audio_filepaths)} audio files")
    num_files = len(audio_filepaths)
    assert manifest is not None, "This script works only with manifest"
    cfg.asr.decoding.greedy.enable_per_stream_biasing = True

    # Build the pipeline
    pipeline = PipelineBuilder.build_pipeline(cfg)

    # initialize options with biasing phrase for each utterance (sanity check: ground truth boosting)
    options = [
        ASRRequestOptions(
            biasing_cfg=BiasingRequestItemConfig(
                boosting_model_cfg=BoostingTreeModelConfig(
                    key_phrases_list=[manifest[i]["text"]],
                ),
                boosting_model_alpha=1.0,
            )
        )
        for i in range(num_files)
    ]

    # Run the pipeline
    recognition_timer = SimpleTimer()
    recognition_timer.start(device=pipeline.asr_model.device)
    output = pipeline.run(audio_filepaths, options=options, progress_bar=TQDMProgressBar())
    recognition_timer.stop()
    exec_dur = recognition_timer.total_sec()

    # Calculate RTFX
    data_dur, _ = calculate_duration(audio_filepaths)
    rtfx = data_dur / exec_dur if exec_dur > 0 else float('inf')
    logging.info(f"RTFx: {rtfx:.2f} ({data_dur:.2f}s / {exec_dur:.2f}s)")

    for i, record in enumerate(manifest):
        record["pred_text"] = output[i]["text"]

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
    dump_output(output, cfg.output_filename, cfg.output_dir)
    logging.info("Done!")


if __name__ == "__main__":
    main()
