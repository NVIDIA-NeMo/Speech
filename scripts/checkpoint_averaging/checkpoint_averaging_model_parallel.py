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
Example: python scripts/checkpoint_averaging/average_model_checkpoints.py \
             --name_prefix=<checkpoint name> \
             --checkpoint_dir=<folder with mp_rank_X subfolders containing checkpoints>

will generate a new file in each of the mp_rank_X subfolders named <checkpoint name>-averaged.ckpt

Typically you should follow up this script with a call to examples/nlp/language_modeling/megatron_ckpt_to_nemo.py
to convert .ckpt checkpoint to .nemo format.
"""

import argparse
import os

import torch

from nemo.utils import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name_prefix', help='Name of the final checkpoint. Will append -averaged.ckpt automatically.',
    )
    parser.add_argument(
        '--checkpoint_dir', help='Folder containing all mp_rank_X subfolders.',
    )
    args = parser.parse_args()

    # repeating for all ranks
    for rank_dir in os.listdir(args.checkpoint_dir):
        if not rank_dir.startswith('mp_rank_'):
            continue
        logging.info("Processing %s", rank_dir)
        full_checkpoint_dir = os.path.join(args.checkpoint_dir, rank_dir)
        checkpoint_paths = [
            os.path.join(full_checkpoint_dir, x)
            for x in os.listdir(full_checkpoint_dir)
            if x.endswith('.ckpt') and not x.endswith('-last.ckpt')
        ]

        # everything below is copied over from average_model_checkpoints.py
        """ < Checkpoint Averaging Logic > """
        # load state dicts
        n = len(checkpoint_paths)
        avg_state = None

        logging.info(f"Averaging {n} checkpoints ...")

        for ix, path in enumerate(checkpoint_paths):
            checkpoint = torch.load(path, map_location='cpu')

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            if ix == 0:
                # Initial state
                avg_state = checkpoint

                logging.info(f"Initialized average state dict with checkpoint : {path}")
            else:
                # Accumulated state
                for k in avg_state:
                    avg_state[k] = avg_state[k] + checkpoint[k]

                logging.info(f"Updated average state dict with state from checkpoint : {path}")

        for k in avg_state:
            if str(avg_state[k].dtype).startswith("torch.int"):
                # For int type, not averaged, but only accumulated.
                # e.g. BatchNorm.num_batches_tracked
                pass
            else:
                avg_state[k] = avg_state[k] / n

        # Save model
        ckpt_name = os.path.join(full_checkpoint_dir, args.name_prefix + '-averaged.ckpt')
        torch.save({'state_dict': avg_state}, ckpt_name)

        logging.info(f"Averaged pytorch checkpoint saved as : {ckpt_name}")


if __name__ == '__main__':
    main()
