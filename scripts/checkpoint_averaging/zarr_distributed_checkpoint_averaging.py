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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
Example: python scripts/checkpoint_averaging/zarr_distributed_checkpoint_averaging.py \
             --name_prefix=<checkpoint name> \
             --checkpoint_dir=<folder containing checkpoints> \
             --steps <list of checkpoint steps to average, if not provided, it will average all the checkpoints>

will generate a new directory in each of the distributed checkpoint subfolders named <checkpoint name>-averaged
"""

import argparse
import logging
import os
import shutil
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name_prefix',
        help='Name of the final checkpoint. Will append -averaged automatically.',
    )
    parser.add_argument(
        '--checkpoint_dir',
        help='Folder containing all the distributed checkpoints.',
    )
    # list of checkpoint steps to average
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        help='List of checkpoint steps to average. If not specified, will average all.',
    )

    args = parser.parse_args()

    if args.steps is not None:
        logging.info(f"Will average only steps {args.steps}")

    # repeating for all ranks

    checkpoint_paths = []
    for ckpt_dir in os.listdir(args.checkpoint_dir):
        logging.info("Processing %s", ckpt_dir)
        if ckpt_dir.endswith('0-last'):
            continue
        if args.steps is None:
            checkpoint_paths.append(ckpt_dir)
        else:
            for step in args.steps:
                key = f"-step={step}-"
                if key in ckpt_dir:
                    checkpoint_paths.append(ckpt_dir)

    n = len(checkpoint_paths)
    # initialize dict, will be used to store the weights that need to be averaged
    avg_weights = {}
    chunk_info = {}

    logging.info(f"Averaging {n} checkpoints ... {'at steps:' + str(args.steps) if args.steps is not None else ''}")

    # item that needs to be copied to the new checkpoint folder
    copy_items = []
    for ix, path in enumerate(checkpoint_paths):
        full_path = os.path.join(args.checkpoint_dir, path)

        for item in os.listdir(full_path):

            # if item is not a directory, skip it
            if not os.path.isdir(os.path.join(full_path, item)):
                if ix == 0:
                    copy_items.append(os.path.join(full_path, item))
                continue

            # transformer engine states, leave them out
            if item.endswith('._extra_state'):
                if ix == 0:
                    copy_items.append(os.path.join(full_path, item))
                continue

            # optimizer states, no point of averaing them
            if item.startswith('optimizer.'):
                if ix == 0:
                    copy_items.append(os.path.join(full_path, item))
                continue

            if item not in avg_weights:
                logging.info(f"Initialized average weights dict with: {item}")
                array = zarr.open(os.path.join(full_path, item), mode='r')
                avg_weights[item] = array[:]
                chunk_info[item] = array.chunks
            else:
                logging.info(f"Updated average weights dict with weight: {item}")
                array_z = zarr.open(os.path.join(full_path, item), mode='r')
                sum_array = avg_weights[item] + array_z[:]
                avg_weights[item] = sum_array

    for k in avg_weights:
        logging.info(f"Average weights dict key : {k}, dtype : {avg_weights[k].dtype}, shape : {avg_weights[k].shape}")
        if str(avg_weights[k].dtype).startswith("int"):
            raise ValueError("Int type not supported")
        else:
            array_z = avg_weights[k] / n
            avg_weights[k] = array_z

    # Save model
    if args.steps is None:
        ckpt_name = os.path.join(args.checkpoint_dir, args.name_prefix + '-averaged')
    else:
        steps_combined = '_'.join([str(x) for x in args.steps])
        ckpt_name = os.path.join(args.checkpoint_dir, args.name_prefix + '-' + steps_combined + '-averaged')

    # save avg_weights
    for k in avg_weights:
        logging.info(f"Saving {k} to {ckpt_name}")
        input_arr = avg_weights[k]
        chunks = chunk_info[k]
        # create the zarr array
        output_array = zarr.create(
            input_arr.shape,
            dtype=input_arr.dtype,
            store=os.path.join(ckpt_name, k),
            chunks=chunks,
            compressor=None,
            fill_value=None,
            write_empty_chunks=True,
        )
        if input_arr.dtype == np.dtype('bfloat16'):
            arr = output_array
            arr._dtype = input_arr.dtype
            zarray = arr.store['.zarray']
            arr.store['.zarray'] = zarray.replace(b'<V2', b'bfloat16')
        output_array[:] = input_arr

    # copy other files
    for item in copy_items:
        is_file = os.path.isfile(item)
        logging.info(f"Copying {'directory' if is_file else 'file'} {item} to {ckpt_name}")
        if os.path.isfile(item):
            # copy single file
            shutil.copy(item, ckpt_name)
        else:
            # copy directory
            shutil.copytree(item, os.path.join(ckpt_name, os.path.basename(item)), dirs_exist_ok=True)

    logging.info(f"Averaged distributed checkpoint saved as : {ckpt_name}")


if __name__ == '__main__':
    main()
