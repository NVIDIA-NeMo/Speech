# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = get_available_device()
        self.model = utmosv2.create_model()
        self.model.eval()
        self.model.to(torch.device(device))

    def __call__(self, file_path):
        """
        Estimate the MOS of the given speech audio file using UTMOSv2.
        """
        with torch.inference_mode():
            # UTMOSv2 tends to launch many OpenMP threads which can overload the machine's CPUs
            # without actually speeding up prediction. Limit to 4 threads.
            with threadpool_limits(limits=4):
                mos_score = self.model.predict(input_path=file_path, num_repetitions=1, num_workers=0)
        return mos_score

    def process_directory(self, input_dir: str, batch_size: int = 16) -> list[dict[str, str | float]]:
        """
        Computes UTMOSv2 scores for all `*.wav` files in the given directory.
        Args:
            input_dir: The directory containing the audio files.
            batch_size: The number of audio files to process in parallel.
        Returns:
            A list of dictionaries, each containing the file path and the UTMOSv2 score.
        """
        with torch.inference_mode():
            # UTMOSV2 tends to launch many of OpenMP threads which overloads the machine's CPUs
            # while actually slowing down the prediction. Limit the number of threads here.
            with threadpool_limits(limits=1):
                results = self.model.predict(
                    input_dir=input_dir, num_repetitions=1, num_workers=batch_size, batch_size=batch_size
                )
        return results


def get_available_device():
    """
    Get the best available device (prefer GPU, fallback to CPU).
    """
    if torch.cuda.is_available():
        return "cuda:0"  # Use first GPU
    else:
        return "cpu"
