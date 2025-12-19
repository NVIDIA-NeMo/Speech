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
This script serves as the entry point for local ASR inference, supporting buffered CTC/RNNT/TDT and cache-aware CTC/RNNT inference.

The script performs the following steps:
    (1) Accepts as input a single audio file, a directory of audio files, or a manifest file.
        - Note: Input audio files must be 16 kHz, mono-channel WAV files.
    (2) Creates a pipeline object to perform inference.
    (3) Runs inference on the input audio files.
    (4) Writes the transcriptions to an output json/jsonl file. Word/Segment level output is written to a separate JSON file.

Example usage:
python asr_streaming_infer.py \
        --config-path=../conf/asr_streaming_inference/ \
        --config-name=config.yaml \
        audio_file=<path to audio file, directory of audio files, or manifest file> \
        output_filename=<path to output jsonfile> \
        lang=en \
        enable_pnc=False \
        enable_itn=False \
        enable_nmt=False \
        asr_output_granularity=segment \
        ...
        # See ../conf/asr_streaming_inference/*.yaml for all available options

Note:
    The output file is a json file with the following structure:
    {"audio_filepath": "path/to/audio/file", "text": "transcription of the audio file", "json_filepath": "path/to/json/file"}
"""


from time import time

import hydra

from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
from nemo.collections.asr.inference.utils.manifest_io import calculate_duration, dump_output, get_audio_filepaths
from nemo.collections.asr.inference.utils.pipeline_eval import calculate_pipeline_laal, evaluate_pipeline
from nemo.collections.asr.inference.utils.progressbar import TQDMProgressBar
from nemo.utils import logging

# disable nemo_text_processing logging
try:
    from nemo_text_processing.utils import logger as nemo_text_logger

    nemo_text_logger.propagate = False
except ImportError:
    # NB: nemo_text_processing requires pynini, which is tricky to install on MacOS
    # since nemo_text_processing is not necessary for ASR, wrap the import
    logging.warning("NeMo text processing library is unavailable.")


@hydra.main(version_base=None)
def main(cfg):

    # Set the logging level
    logging.setLevel(cfg.log_level)

    # Reading audio filepaths
    audio_filepaths, manifest = get_audio_filepaths(cfg.audio_file, sort_by_duration=True)
    logging.info(f"Found {len(audio_filepaths)} audio files")
    if manifest:
        keys = list(manifest[0].keys())
        logging.info(f"Found {len(keys)} keys in the input manifest: {keys}")

    # Build the pipeline
    pipeline = PipelineBuilder.build_pipeline(cfg)
    progress_bar = TQDMProgressBar()

    # Run the pipeline
    start = time()
    output = pipeline.run(audio_filepaths, progress_bar=progress_bar)
    exec_dur = time() - start

    # Calculate RTFX
    data_dur, durations = calculate_duration(audio_filepaths)
    rtfx = data_dur / exec_dur if exec_dur > 0 else float('inf')
    logging.info(f"RTFX: {rtfx:.2f} ({data_dur:.2f}s / {exec_dur:.2f}s)")

    # Calculate LAAL
    laal = calculate_pipeline_laal(output, durations, manifest, cfg)
    if laal is not None:
        logging.info(f"LAAL: {laal:.2f}ms")

    # Dump the transcriptions to a output file
    dump_output(output, cfg.output_filename, cfg.output_dir, manifest)

    # Evaluate the pipeline
    evaluate_pipeline(cfg.output_filename, cfg)
    logging.info("Done!")


if __name__ == "__main__":
    main()
