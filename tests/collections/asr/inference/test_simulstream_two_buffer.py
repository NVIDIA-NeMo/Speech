# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.inference.pipelines.base_pipeline import TranscribeStepOutput
from nemo.collections.asr.inference.utils.simulstream_pipeline_adapter import NeMoStreamingPipelineAdapter


class _NMTEnabledPipeline:
    nmt_enabled = True


def test_final_and_new_partial_are_one_incremental_transition():
    adapter = NeMoStreamingPipelineAdapter.__new__(NeMoStreamingPipelineAdapter)
    adapter.latency_unit = "word"
    adapter.pipeline = _NMTEnabledPipeline()
    adapter._prev_partial_translation = "Die Konferenz beginnt"
    output = TranscribeStepOutput(
        stream_id=0,
        final_translation="Die Konferenz beginnt am Montag.",
        partial_translation="Die Anmeldung öffnet",
    )

    incremental = adapter._convert_to_incremental_output(output)

    assert incremental.deleted_tokens == []
    assert incremental.new_tokens == ["am", "Montag.", "Die", "Anmeldung", "öffnet"]
    assert adapter._prev_partial_translation == "Die Anmeldung öffnet"


def test_chinese_boundary_transition_does_not_insert_whitespace():
    adapter = NeMoStreamingPipelineAdapter.__new__(NeMoStreamingPipelineAdapter)
    adapter.latency_unit = "char"
    adapter.pipeline = _NMTEnabledPipeline()
    adapter._prev_partial_translation = "会议"
    output = TranscribeStepOutput(
        stream_id=0,
        final_translation="会议开始。",
        partial_translation="注册开放",
    )

    incremental = adapter._convert_to_incremental_output(output)

    assert incremental.deleted_tokens == []
    assert incremental.new_string == "开始。注册开放"
