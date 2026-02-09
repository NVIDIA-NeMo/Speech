# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Simulstream integration (optional dependency)
try:
    from nemo.collections.asr.inference.simulstream_manifest_utils import (
        load_manifest_audio_paths,
        manifest_to_audio_definition,
        prepare_simulstream_files,
    )
    from nemo.collections.asr.inference.simulstream_pipeline_adapter import (
        NeMoStreamingPipelineAdapter,
        create_nemo_pipeline_adapter,
        create_nemo_pipeline_from_config,
        load_nemo_config,
    )
    __all__ = [
        'NeMoStreamingPipelineAdapter',
        'create_nemo_pipeline_adapter',
        'load_nemo_config',
        'create_nemo_pipeline_from_config',
        'prepare_simulstream_files',
        'get_language_from_manifest',
    ]
except ImportError:
    __all__ = []
