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

import glob
import os
from typing import List


from nemo.utils import logging

LLM_VOCAB_SIZE_MAP = {
    "gpt3": 51200,
    "llama2": 32000,
    "llama3": 128256,
    "nemotron": 256000,
    "bert": 29000,
    "mixtral": 32000,
}


def read_tb_log(path: str, summary_name: str) -> List:
    """
    Reads a TensorBoard Events file from the input path, and returns the
    summary specified.

    Args:
        path: str, path to the dir where the events file is located.
        summary_name: str, name of the summary to read from the TB logs.
    Returns:
        summary_list: list, the values in the read summary list, formatted as a list.
    """
    from tensorboard.backend.event_processing import event_accumulator

    files = glob.glob(f"{path}/events*tfevents*")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if len(files) == 0 or not os.path.isfile(files[0]):
        raise FileNotFoundError("Missing TensorBoard log file.")

    events_file = files[0]
    try:
        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()
        summary = ea.Scalars(summary_name)
        summary_list = [round(x.value, 2) for x in summary]
        logging.info(f"{summary_name}: {summary_list}")
    except KeyError:
        raise KeyError(f"{summary_name} not found in {events_file}")

    return summary_list
