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

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MLFlowParams:
    # name of experiment, if none, defaults to the globally set experiment name
    experiment_name: Optional[str] = None
    # no run_name because it's set by version
    # local or remote tracking seerver. If tracking_uri is not set, it defaults to save_dir
    tracking_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    save_dir: Optional[str] = "./mlruns"
    prefix: str = ""
    artifact_location: Optional[str] = None
    # provide run_id if resuming a previously started run
    run_id: Optional[str] = None
    # Log checkpoints created by ModelCheckpoint as MLFlow artifacts.
    log_model: bool = False
