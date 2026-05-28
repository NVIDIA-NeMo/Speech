# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
During inference, we perform frame-level prediction by two approaches: 
    1) shift the window of length window_length_in_sec (e.g. 0.63s) by shift_length_in_sec (e.g. 10ms) to generate the frame and use the prediction of the window to represent the label for the frame;
       [this script demonstrate how to do this approach]
    2) generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments. 
       [get frame level prediction by this script and use vad_overlap_posterior.py in NeMo/scripts/voice_activity_detection
       One can also find posterior about converting frame level prediction 
       to speech/no-speech segment in start and end times format in that script.]
       
       Image https://raw.githubusercontent.com/NVIDIA/NeMo/main/tutorials/asr/images/vad_post_overlap_diagram.png 
       will help you understand this method.

This script will also help you perform postprocessing and generate speech segments if needed

Usage:
python vad_infer.py --config-path="../conf/vad" --config-name="vad_inference_postprocessing.yaml" dataset=<Path of json file of evaluation data. Audio files should have unique names>

"""
import json
import os

import torch

from nemo.collections.asr.parts.utils.speaker_utils import write_rttm2manifest
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    init_vad_model,
    prepare_manifest,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="../conf/vad", config_name="vad_inference_postprocessing.yaml")
def main(cfg):
    if not cfg.dataset:
        raise ValueError("You must input the path of json file of evaluation data")

    # each line of dataset should be have different audio_filepath and unique name to simplify edge cases or conditions
    key_meta_map = {}
    with open(cfg.dataset, 'r') as manifest:
        for line in manifest.readlines():
            audio_filepath = json.loads(line.strip())['audio_filepath']
            uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio_filepath! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath}

    # Prepare manifest for streaming VAD
    manifest_vad_input = cfg.dataset
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        config = {
            'input': manifest_vad_input,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'split_duration': cfg.prepare_manifest.split_duration,
            'num_workers': cfg.num_workers,
            'prepared_manifest_vad_input': cfg.prepared_manifest_vad_input,
        }
        manifest_vad_input = prepare_manifest(config)
    else:
        logging.warning(
            "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
        )

    torch.set_grad_enabled(False)
    vad_model = init_vad_model(cfg.vad.model_path)

    # setup_test_data
    vad_model.setup_test_data(
        test_data_config={
            'vad_stream': True,
            'sample_rate': 16000,
            'manifest_filepath': manifest_vad_input,
            'labels': [
                'infer',
            ],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'shift_length_in_sec': cfg.vad.parameters.shift_length_in_sec,
            'trim_silence': False,
            'normalize_audio': cfg.vad.parameters.normalize_audio,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    if not os.path.exists(cfg.frame_out_dir):
        os.mkdir(cfg.frame_out_dir)
    else:
        logging.warning(
            "Note frame_out_dir exists. If new file has same name as file inside existing folder, it will append result to existing file and might cause mistakes for next steps."
        )

    logging.info("Generating frame level prediction ")
    pred_dir = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
        shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
        manifest_vad_input=manifest_vad_input,
        out_dir=cfg.frame_out_dir,
    )
    logging.info(
        f"Finish generating VAD frame level prediction with window_length_in_sec={cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={cfg.vad.parameters.shift_length_in_sec}"
    )
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
        frame_length_in_sec = 0.01

    # postprocessing and generate speech segments
    if cfg.gen_seg_table:
        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=pred_dir,
            postprocessing_params=cfg.vad.parameters.postprocessing,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=cfg.table_out_dir,
        )
        logging.info(
            f"Finish generating speech semgents table with postprocessing_params: {cfg.vad.parameters.postprocessing}"
        )

    if cfg.write_to_manifest:
        for i in key_meta_map:
            key_meta_map[i]['rttm_filepath'] = os.path.join(table_out_dir, i + ".txt")

        if not cfg.out_manifest_filepath:
            out_manifest_filepath = "vad_out.json"
        else:
            out_manifest_filepath = cfg.out_manifest_filepath
        out_manifest_filepath = write_rttm2manifest(key_meta_map, out_manifest_filepath)
        logging.info(f"Writing VAD output to manifest: {out_manifest_filepath}")


if __name__ == '__main__':
    main()
