# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This script is designed to extract features from different layers of a pretrained SSL model.
The extracted features will be in *.npy format, and in the shape of [L, D, T], where L is the 
number of layers, D is the feature dimension, and T is the time dimension.

Example usage:

python extract_features.py \
    --model_path="nvidia/ssl_en_nest_large_v1.0" \
    --input=<path to input manifest, or a dir containing audios, or path to audio> \
    --output=<output directory to store features and manifest> \
    --layers="all" \
    --batch_size=8 \
    --workers=8 \
    --max_cache=1000 # save features every 1000 samples to avoid OOM in system memory
"""


import argparse
import os
import tempfile
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_text_dataset import get_char_dataset
from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
from nemo.collections.asr.modules import ConformerMultiLayerFeatureExtractor
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.classes.common import typecheck
from nemo.utils import logging

typecheck.set_typecheck_enabled(enabled=False)

parser = argparse.ArgumentParser(description="Extract audio features using an SSL model")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the .nemo model file or a pretrained model name from the NGC/HF model hub",
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to the input audio file, or list of files, directory or jsonl manifest",
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Path to the output directory that contains .npy file"
)
parser.add_argument(
    "-l",
    "--layers",
    type=str,
    default="all",
    help="Layers to extract features from, use 'all' to extract from all layer, 'last' for last layer, "
    "or comma-separated indices of the target layers (e.g. '0,1,2')",
)
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size for feature extraction")
parser.add_argument("-w", "--workers", type=int, default=8, help="Number of workers for feature extraction")
parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use for feature extraction")
parser.add_argument("-t", "--type", type=str, default="wav", help="audio file type, only needed for directory input")
parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
parser.add_argument(
    "--amp_dtype",
    type=str,
    default="float16",
    choices=["float16", "bfloat16"],
    help="Data type for automatic mixed precision",
)
parser.add_argument("-mc", "--max_cache", type=int, default=-1, help="Max cache size before saving features")
args = parser.parse_args()


def get_input_manifest(input: str) -> List[dict]:
    """
    Build manifest from input path or directory
    """
    if input.endswith(".json") or input.endswith(".jsonl") and os.path.isfile(input):
        logging.info(f"Reading manifest from: {input}")
        manifest = [
            {"audio_filepath": str(get_full_path(item["audio_filepath"], input)), "duration": None, "text": "-"}
            for item in read_manifest(input)
        ]
    elif os.path.isdir(input):
        logging.info(f"Creating manifest from directory: {input}")
        manifest = [
            {"audio_filepath": str(p), "duration": None, "text": "-"} for p in Path(input).rglob(f"*.{args.type}")
        ]
        logging.info(f"Found {len(manifest)} items of {args.type} files")
    elif os.path.isfile(input):
        logging.info(f"Reading single file: {input}")
        manifest = [{"audio_filepath": Path(input).absolute.as_posix(), "duration": None, "text": "-"}]
    else:
        raise ValueError(f"Invalid input: {input}")
    return manifest


def load_model(model_path):
    """
    Load SSL model from local or pretrained
    """
    if model_path.endswith(".nemo") and os.path.isfile(model_path):
        logging.info(f"Loading model from local: {model_path}")
        model = EncDecDenoiseMaskedTokenPredModel.restore_from(model_path)
    else:
        logging.info(f"Loading model from pretrained: {model_path}")
        model = EncDecDenoiseMaskedTokenPredModel.from_pretrained(model_name=model_path)
    return model


class FeatureExtractor(pl.LightningModule):
    """
    Wrapper class for extracting features from SSL model
    """

    def __init__(self, ssl_model: EncDecDenoiseMaskedTokenPredModel, layer: str = "all"):
        super().__init__()
        self.preprocessor = ssl_model.preprocessor
        self.encoder = ssl_model.encoder
        self.layer_idx_list = None
        self.sample_rate = ssl_model.cfg.sample_rate
        if layer == "all":
            self.layer_idx_list = None
        elif layer == "last":
            self.layer_idx_list = [len(self.encoder.layers) - 1]
        else:
            try:
                self.layer_idx_list = [int(l) for l in layer.split(",")]
            except Exception as e:
                raise ValueError(f"Invalid layer argument: {layer}. Error: {e}")
        self.feature_extractor = ConformerMultiLayerFeatureExtractor(
            self.encoder, aggregator=None, layer_idx_list=self.layer_idx_list
        )

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        """
        Forward pass to extract features, same input interface as EncDecDenoiseMaskedTokenPredModel.forward
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        encoded, encoded_len = self.feature_extractor(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len


def maybe_save_features(output_dir, results, max_cache, manifest):
    """
    Check if the cache is full and save features to disk
    """
    if len(results) == 0 or max_cache < 0 or len(results) < max_cache:
        return
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving {len(results)} features to {output_dir}")

    for sample_id, audio_file, features_np in tqdm(results, desc="Saving features", total=len(results)):
        filename = str(audio_file).replace("/", "_").replace(".", "_")
        if len(filename) > 256:
            filename = filename[-256:]
        output_path = os.path.join(output_dir, f"{filename}.npy")
        np.save(output_path, features_np)
        manifest[sample_id]["feature_path"] = output_path

    logging.info(f"Saved {len(results)} features to {output_dir}")
    results.clear()


def extract_features(args):
    """
    Main function to extract and save features from SSL model
    """

    logging.info(f"Extracting features using params: {vars(args)}")

    # Load model
    model = load_model(args.model_path)
    feature_extractor = FeatureExtractor(model, args.layers)
    device = torch.device(args.device)
    feature_extractor.to(device)

    # Load data
    logging.info(f"Building dataset from input: {args.input}")
    tmp_manifest = tempfile.NamedTemporaryFile(mode="w", delete=False)
    manifest = get_input_manifest(args.input)
    write_manifest(tmp_manifest.name, manifest)
    total_num_samples = len(manifest)

    # Build dataloader
    config = {
        "manifest_filepath": tmp_manifest.name,
        "sample_rate": feature_extractor.sample_rate,
        "return_sample_id": True,
    }
    dataset = get_char_dataset(config)
    logging.info(f"Built dataset with {len(dataset)} samples")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # Extract features
    indices = set()
    results = []
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    logging.info(f"Extracting features using AMP: {args.use_amp}, dtype: {amp_dtype}")
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=amp_dtype, enabled=args.use_amp):
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Extracting features"):
                batch = move_data_to_device(batch, device)
                audio_signal, audio_signal_len, _, _, sample_id = batch
                features, features_len = feature_extractor(
                    input_signal=audio_signal, input_signal_length=audio_signal_len
                )
                batch_size = features[0].size(0)
                num_layers = len(features)
                for i in range(batch_size):
                    sid_i = sample_id[i]
                    if sid_i in indices:
                        logging.warning(f"Skipping duplicated sample_id: {sample_id}")
                        continue

                    feat_i_len = features_len[0][i]
                    feat_i = []
                    for j in range(num_layers):
                        feat_i.append(features[j][i][:, :feat_i_len])

                    feat_i_np = torch.stack(feat_i, dim=0).cpu().numpy()

                    indices.add(sid_i)
                    results.append((sid_i, manifest[sid_i]['audio_filepath'], feat_i_np))

                maybe_save_features(args.output, results, args.max_cache, manifest)

    maybe_save_features(args.output, results, 0, manifest)

    output_manifest = Path(args.output) / "features.json"
    write_manifest(output_manifest, manifest)
    os.remove(tmp_manifest.name)
    logging.info(f"Extracted features from {total_num_samples} samples to {args.output}")
    logging.info(f"Manifest saved to: {output_manifest}")


if __name__ == "__main__":
    extract_features(args)
