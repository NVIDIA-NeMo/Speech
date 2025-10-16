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
try:
    import utmosv2
except ImportError:
    raise ImportError(
        "UTMOSv2 is not installed. Please install it using `pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1`."
    )
import torch
from threadpoolctl import threadpool_limits

"""
Uses the UTMOSv2 model to estimate the MOS of a speech audio file.
"""


class UTMOSv2Calculator:
    """
    Wrapper around UTMOSv2 MOS estimator to make it easy to use.
    """

    def __init__(self):
        device = get_available_device()
        self.model = utmosv2.create_model()
        self.model.eval()
        self.model.to(torch.device(device))

    def __call__(self, file_path):
        """
        Estimate the MOS of the given speech audio file using UTMOSv2.
        """
        with torch.inference_mode():
            # UTMOSv2 tends to launch many of OpenMP threads which can overload the machine's CPUs
            # without actually speeding up prediction. Limit to 4 threads.
            with threadpool_limits(limits=4):
                mos_score = self.model.predict(input_path=file_path, num_repetitions=1, num_workers=0)
        return mos_score


def get_available_device():
    """
    Get the best available device (prefer GPU, fallback to CPU).
    """
    if torch.cuda.is_available():
        return "cuda:0"  # Use first GPU
    else:
        return "cpu"
