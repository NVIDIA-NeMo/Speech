# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import time
from argparse import ArgumentParser

from nemo.collections.asr.parts.utils.vad_utils import generate_overlap_vad_seq, generate_vad_segment_table
from nemo.utils import logging

"""
Note you can use NeMo/examples/asr/speech_classification/vad_infer.py which includes the functionalities appeared in this function directly. 

You are encouraged to use this script if you want to try overlapped mean/median smoothing filter and postprocessing technique without perform costly NN inference several times.
You can also use this script to write RTTM-like files if you have frame level prediction already.

This script serves two purposes:
    1) gen_overlap_seq: 
        Generate predictions with overlapping input segments by using the frame level prediction from NeMo/examples/asr/speech_classification/vad_infer.py. 
        Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments. 
       
    2）gen_seg_table: 
        Converting frame level prediction to speech/no-speech segment in start and end times format with postprocessing technique.
   
Usage:

python vad_overlap_posterior.py --gen_overlap_seq --gen_seg_table --frame_folder=<FULL PATH OF YOU STORED FRAME LEVEL PREDICTION> --method='median' --overlap=0.875 --num_workers=20
 
You can play with different postprocesing parameters. Here we just show the simpliest condition onset=offset=threshold=0.5
See more details about postprocesing in function binarization and filtering in NeMo/nemo/collections/asr/parts/utils/vad_utils

"""
postprocessing_params = {"onset": 0.5, "offset": 0.5}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gen_overlap_seq", default=False, action='store_true')
    parser.add_argument("--gen_seg_table", default=False, action='store_true')
    parser.add_argument("--frame_folder", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Use mean/median for overlapped prediction. Use frame for gen_seg_table of frame prediction",
    )
    parser.add_argument("--overlap_out_dir", type=str)
    parser.add_argument("--table_out_dir", type=str)
    parser.add_argument("--overlap", type=float, default=0.875, help="Overlap percentatge. Default is 0.875")
    parser.add_argument("--window_length_in_sec", type=float, default=0.63)
    parser.add_argument("--shift_length_in_sec", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.gen_overlap_seq:
        start = time.time()
        logging.info("Generating predictions with overlapping input segments")
        overlap_out_dir = generate_overlap_vad_seq(
            frame_pred_dir=args.frame_folder,
            smoothing_method=args.method,
            overlap=args.overlap,
            window_length_in_sec=args.window_length_in_sec,
            shift_length_in_sec=args.shift_length_in_sec,
            num_workers=args.num_workers,
            out_dir=args.overlap_out_dir,
        )
        logging.info(
            f"Finish generating predictions with overlapping input segments with smoothing_method={args.method} and overlap={args.overlap}"
        )
        end = time.time()
        logging.info(f"Generate overlapped prediction takes {end-start:.2f} seconds!\n Save to {overlap_out_dir}")

    if args.gen_seg_table:
        start = time.time()
        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")

        frame_length_in_sec = args.shift_length_in_sec
        if args.gen_overlap_seq:
            logging.info("Use overlap prediction. Change if you want to use basic frame level prediction")
            vad_pred_dir = overlap_out_dir
            frame_length_in_sec = 0.01
        else:
            logging.info("Use basic frame level prediction")
            vad_pred_dir = args.frame_folder

        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=vad_pred_dir,
            postprocessing_params=postprocessing_params,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=args.num_workers,
            out_dir=args.table_out_dir,
        )
        logging.info(f"Finish generating speech semgents table with postprocessing_params: {postprocessing_params}")
        end = time.time()
        logging.info(
            f"Generating rttm-like tables for {vad_pred_dir} takes {end-start:.2f} seconds!\n Save to {table_out_dir}"
        )
