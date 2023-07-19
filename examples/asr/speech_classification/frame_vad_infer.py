# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
This script peforms VAD on each 20ms frames of the input audio files. 
Postprocessing is also performed to generate speech segments and store them as RTTM files.
Long audio files will be splitted into smaller chunks to avoid OOM issues, but the frames close
to the split points might have worse performance due to truncated context.

## Usage:
python frame_vad_infer.py \
    --config-path="../conf/vad" --config-name="frame_vad_infer_postprocess" \
    input_manifest=<Path of manifest file containing evaluation data. Audio files should have unique names> \
    output_dir=<Path of output directory>

The manifest json file should have the following format (each line is a Python dictionary):
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000}  
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000}  

If you want to evaluate tne model's AUROC and DER performance, you need to set `evaluate=True` in config yaml,
and also provide groundtruth in either RTTM files or label strings:
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000, "label": "0 1 0 0 0 1 1 1 0"}
or
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000, "rttm_filepath": "/path/to/rttm_file1.rttm"}

"""

import os
from pathlib import Path

import torch

from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.asr.parts.utils.vad_utils import (
    frame_vad_eval_detection_error,
    frame_vad_infer_load_manifest,
    generate_overlap_vad_seq,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    init_frame_vad_model,
    prepare_manifest,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="../conf/vad", config_name="frame_vad_infer_postprocess")
def main(cfg):
    if not cfg.input_manifest:
        raise ValueError("You must input the path of json file of evaluation data")
    output_dir = cfg.output_dir if cfg.output_dir else "frame_vad_outputs"
    if os.path.exists(output_dir):
        logging.warning(
            f"Output directory {output_dir} already exists, use this only if you're tuning post-processing params."
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cfg.frame_out_dir = os.path.join(output_dir, "frame_preds")
    cfg.smoothing_out_dir = os.path.join(output_dir, "smoothing_preds")
    cfg.rttm_out_dir = os.path.join(output_dir, "rttm_preds")

    # each line of input_manifest should be have different audio_filepath and unique name to simplify edge cases or conditions
    logging.info(f"Loading manifest file {cfg.input_manifest}")
    manifest_orig, key_labels_map, key_rttm_map = frame_vad_infer_load_manifest(cfg)

    # Prepare manifest for streaming VAD
    manifest_vad_input = cfg.input_manifest
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        config = {
            'input': manifest_vad_input,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'split_duration': cfg.prepare_manifest.split_duration,
            'num_workers': cfg.num_workers,
            'prepared_manifest_vad_input': cfg.prepared_manifest_vad_input,
            'out_dir': output_dir,
        }
        manifest_vad_input = prepare_manifest(config)
    else:
        logging.warning(
            "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
        )

    torch.set_grad_enabled(False)
    vad_model = init_frame_vad_model(cfg.vad.model_path)

    # setup_test_data
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'sample_rate': 16000,
            'manifest_filepath': manifest_vad_input,
            'labels': ['infer'],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'normalize_audio_db': cfg.vad.parameters.normalize_audio_db,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    if not os.path.exists(cfg.frame_out_dir):
        logging.info(f"Frame predictions do not exist at {cfg.frame_out_dir}, generating frame prediction.")
        os.mkdir(cfg.frame_out_dir)
        extract_frame_preds = True
    else:
        logging.info(f"Frame predictions already exist at {cfg.frame_out_dir}, skipping frame prediction generation.")
        extract_frame_preds = False

    if extract_frame_preds:
        logging.info("Generating frame-level prediction ")
        pred_dir = generate_vad_frame_pred(
            vad_model=vad_model,
            window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
            manifest_vad_input=manifest_vad_input,
            out_dir=cfg.frame_out_dir,
        )
        logging.info(f"Finish generating VAD frame level prediction. You can find the prediction in {pred_dir}")
    else:
        pred_dir = cfg.frame_out_dir

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    # overlap smoothing filter
    if cfg.vad.parameters.smoothing:
        # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
        # smoothing_method would be either in majority vote (median) or average (mean)
        logging.info("Generating predictions with overlapping input segments")
        smoothing_pred_dir = generate_overlap_vad_seq(
            frame_pred_dir=pred_dir,
            smoothing_method=cfg.vad.parameters.smoothing,
            overlap=cfg.vad.parameters.overlap,
            window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=cfg.smoothing_out_dir,
        )
        logging.info(
            f"Finish generating predictions with overlapping input segments with smoothing_method={cfg.vad.parameters.smoothing} and overlap={cfg.vad.parameters.overlap}"
        )
        pred_dir = smoothing_pred_dir

    # postprocessing and generate speech segments
    logging.info("Converting frame level prediction to RTTM files.")
    rttm_out_dir = generate_vad_segment_table(
        vad_pred_dir=pred_dir,
        postprocessing_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
        use_rttm=cfg.vad.use_rttm,
        out_dir=cfg.rttm_out_dir,
    )
    logging.info(
        f"Finish generating speech semgents table with postprocessing_params: {cfg.vad.parameters.postprocessing}"
    )

    logging.info("Writing VAD output to manifest")
    key_pred_rttm_map = {}
    manifest_new = []
    for entry in manifest_orig:
        key = Path(entry['audio_filepath']).stem
        entry['rttm_filepath'] = Path(os.path.join(rttm_out_dir, key + ".rttm")).absolute().as_posix()
        if not Path(entry['rttm_filepath']).is_file():
            logging.warning(f"Not able to find {entry['rttm_filepath']} for {entry['audio_filepath']}")
            entry['rttm_filepath'] = ""
        manifest_new.append(entry)
        key_pred_rttm_map[key] = entry['rttm_filepath']

    if not cfg.out_manifest_filepath:
        out_manifest_filepath = os.path.join(output_dir, "manifest_vad_output.json")
    else:
        out_manifest_filepath = cfg.out_manifest_filepath
    write_manifest(out_manifest_filepath, manifest_new)
    logging.info(f"Finished writing VAD output to manifest: {out_manifest_filepath}")

    if cfg.get("evaluate", False):
        logging.info("Evaluating VAD results")
        auroc, report = frame_vad_eval_detection_error(
            pred_dir=pred_dir,
            key_labels_map=key_labels_map,
            key_rttm_map=key_rttm_map,
            key_pred_rttm_map=key_pred_rttm_map,
            frame_length_in_sec=frame_length_in_sec,
        )
        DetER = report.iloc[[-1]][('detection error rate', '%')].item()
        FA = report.iloc[[-1]][('false alarm', '%')].item()
        MISS = report.iloc[[-1]][('miss', '%')].item()
        logging.info(f"AUROC: {auroc:.4f}")
        logging.info(f"DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
        logging.info(f"with params: {cfg.vad.parameters.postprocessing}")
    logging.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
