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
import os
from pathlib import Path

try:
    import utmosv2
except ImportError:
    raise ImportError(
        "UTMOSv2 is not installed. Please install it using `pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1`."
    )
from typing import Optional
import torch
from threadpoolctl import threadpool_limits

# If UTMOSv2 cache is not set but HF_HOME is, use an area under HF_HOME for the cache location
# This avoids re-downloading the UTMSOv2 model each time.
# Note that "UTMOSV2_CHACHE" is not a typo -- that is the string used in the UTMOSv2 library.
if "UTMOSV2_CHACHE" not in os.environ and "HF_HOME" in os.environ:
    utmos_cache_dir = os.path.join(os.environ["HF_HOME"], "utmosv2")
    os.makedirs(utmos_cache_dir, exist_ok=True)
    os.environ["UTMOSV2_CHACHE"] = utmos_cache_dir

"""
Uses the UTMOSv2 model to estimate the MOS of a speech audio file.
"""


class UTMOSv2Calculator:
    """
    Wrapper around UTMOSv2 MOS estimator to make it easy to use.
    Args:
        device: The device to place the model on. If None, the best available device will be used.
            Default is None.
    """

    def __init__(self, device: Optional[str] = None, verbose: bool = True):
        if device is None:
            device = get_available_device()
        self.device = torch.device(device)
        self.model = utmosv2.create_model()
        self.model.eval()
        self.model.to(self.device)
        self.verbose = verbose

    def to(self, device: str | torch.device):
        """Move the scorer to ``device`` and return this calculator."""
        device = torch.device(device)
        if device != self.device:
            self.model.to(device)
            self.device = device
        return self

    def __call__(self, file_path):
        """
        Estimate the MOS of the given speech audio file using UTMOSv2.
        """
        with torch.inference_mode():
            # UTMOSv2 tends to launch many OpenMP threads which can overload the machine's CPUs
            # without actually speeding up prediction. Limit to 4 threads.
            with threadpool_limits(limits=4):
                mos_score = self.model.predict(
                    input_path=file_path,
                    device=self.device,
                    num_repetitions=1,
                    num_workers=0,
                    verbose=self.verbose,
                )
        return mos_score

    def process_directory(
        self,
        input_dir: str,
        batch_size: int = 16,
        num_workers: int = None,
        val_list: list[str] | None = None,
    ) -> list[dict[str, str | float]]:
        """
        Computes UTMOSv2 scores for `*.wav` files in the given directory.

        Args:
            input_dir: The directory containing the audio files.
            batch_size: The number of audio files per scoring batch.
            num_workers: Number of worker processes used by UTMOS internals.
                Set to 0 to avoid multiprocessing pickling issues.
            val_list: If provided, only score these basenames (e.g. ``["000000.wav", "000001.wav"]``)
                via the library's ``val_list`` parameter instead of globbing the whole directory.
                If None, all ``*.wav`` files in ``input_dir`` are scored.
        Returns:
            A list of dictionaries, each containing the file path and the UTMOSv2 score.
        """
        if num_workers is None:
            num_workers = batch_size

        with torch.inference_mode():
            # UTMOSV2 tends to launch many of OpenMP threads which overloads the machine's CPUs
            # while actually slowing down the prediction. Limit the number of threads here.
            with threadpool_limits(limits=1):
                results = self.model.predict(
                    input_dir=input_dir,
                    device=self.device,
                    num_repetitions=1,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    val_list=val_list,
                    verbose=self.verbose,
                )
        if val_list is not None:
            requested_names = [Path(path).name for path in val_list]
            results_by_name = {Path(str(item["file_path"])).name: item for item in results}
            if len(results_by_name) != len(results):
                raise RuntimeError("UTMOSv2 returned duplicate file paths.")
            missing_names = [name for name in requested_names if name not in results_by_name]
            if missing_names:
                raise RuntimeError(f"UTMOSv2 did not return scores for: {missing_names}")
            results = [results_by_name[name] for name in requested_names]
        return results


def get_available_device():
    """
    Get the best available device (prefer GPU, fallback to CPU).
    """
    if torch.cuda.is_available():
        return "cuda:0"  # Use first GPU
    else:
        return "cpu"
