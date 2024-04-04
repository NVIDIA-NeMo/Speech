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

import multiprocessing
import os
import pickle
import shutil
import tarfile
import time
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SHAPES = {
    'prompt_embeds': (77, 2048),
    'pooled_prompt_embeds': (1280,),
    'latents_256': (4, 32, 32),
}


def convert_single_parquet_to_tar(parquet_file):
    pf = pd.read_parquet(parquet_file)
    tmp_folder = Path(parquet_file.split('.')[0] + '-tmp-pickle-files')
    os.makedirs(tmp_folder, exist_ok=True)
    tar_file = Path(args.output_folder) / os.path.basename(parquet_file).replace('parquet', 'tar')
    with tarfile.open(tar_file, 'w') as f:
        tmp_pickle_files = []
        for i in range(len(pf.index)):
            data = pf.iloc[i]
            info = dict()
            for key, shape in SHAPES.items():
                info[key] = np.frombuffer(data[key], dtype=np.float32).reshape(shape)
            tmp_pickle_filename = f'{i}.pickle'
            pickle.dump(info, open(tmp_folder / tmp_pickle_filename, 'wb'))
            f.add(tmp_folder / tmp_pickle_filename, tmp_pickle_filename)
            tmp_pickle_files.append(tmp_pickle_filename)
    shutil.rmtree(tmp_folder)


def generate_wdinfo(tar_folder: str, chunk_size: int, output_path: Optional[str]):
    if not output_path:
        return
    tar_files = []
    for fname in glob(os.path.join(tar_folder, '*.tar')):
        # only glob one level of folder structure because we only write basename to the tar files
        if os.path.getsize(fname) > 0 and not os.path.exists(f"{fname}.INCOMPLETE"):
            tar_files.append(os.path.basename(fname))
    data = {'tar_files': sorted(tar_files), 'chunk_size': chunk_size, 'total_key_count': len(tar_files) * chunk_size}
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print("Generated", output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--parquet_folder', type=str, default='data/parquet')
    parser.add_argument('--output_folder', type=str, default='data/output')
    parser.add_argument('--num_process', type=int, default=-1)
    parser.add_argument('--num_files', type=int, default=-1)
    args = parser.parse_args()

    PROFILE = True
    if PROFILE:
        shutil.rmtree(args.output_folder)

    os.makedirs(args.output_folder, exist_ok=True)
    parquets = glob(f'{args.parquet_folder}/*.parquet')
    if args.num_files > 0:
        parquets = parquets[: args.num_files]
    args.num_files = len(parquets)
    print(f'Processing {args.num_files} files.')
    if args.num_process <= 0:
        args.num_process = min(len(parquets), multiprocessing.cpu_count())
    print(f'Converting using {args.num_process} processes.')
    assert args.num_process <= args.num_files

    t0 = time.time()
    with Pool(processes=args.num_process) as pool:
        pool.map(convert_single_parquet_to_tar, parquets)
    t1 = time.time()
    if PROFILE:
        print("====== Summary ======")
        print(f"{args.num_process} processes and {args.num_files} files.")
        print(f"Total time {t1-t0:.2f}")
        print(f"Time per file {(t1-t0)/len(parquets):.2f}")

    generate_wdinfo(args.output_folder, chunk_size=5000, output_path=os.path.join(args.output_folder, 'wdinfo.pkl'))
