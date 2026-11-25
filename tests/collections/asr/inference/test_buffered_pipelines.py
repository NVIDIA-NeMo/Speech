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


from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

CONFIG_DIR = Path(__file__).parents[4] / "examples" / "asr" / "conf" / "asr_streaming_inference"
# A hybrid RNNT/CTC model, so one checkpoint serves both buffered pipelines.
HYBRID_MODEL = "nvidia/parakeet-tdt_ctc-110m"


def _buffered_pipeline(
    decoding_type: str,
    left_padding_size: float,
    chunk_size: float,
    right_padding_size: float,
    stateful: bool | None = None,
):
    cfg = OmegaConf.load(CONFIG_DIR / f"buffered_{decoding_type}.yaml")
    cfg.asr.model_name = HYBRID_MODEL
    cfg.asr.device = "cpu"
    cfg.asr.compute_dtype = "float32"
    cfg.streaming.left_padding_size = left_padding_size
    cfg.streaming.chunk_size = chunk_size
    cfg.streaming.right_padding_size = right_padding_size
    if stateful is not None:
        cfg.streaming.stateful = stateful
    return PipelineBuilder.build_pipeline(cfg)


@pytest.mark.with_downloads
@pytest.mark.unit
@pytest.mark.parametrize("decoding_type, stateful", [("rnnt", True), ("rnnt", False), ("ctc", None)])
def test_whole_frame_contexts_keep_their_frame_counts(decoding_type, stateful):
    # 0.56 s is 7 frames of 0.08 s, but 0.56 / 0.08 is 7.000000000000001 in floating point.
    pipeline = _buffered_pipeline(
        decoding_type, left_padding_size=5.6, chunk_size=0.56, right_padding_size=0.56, stateful=stateful
    )

    assert pipeline.tokens_per_frame == 7
    assert pipeline.mid_delay == 14
    assert pipeline.chunk_size == 0.56
    assert pipeline.right_padding_size == 0.56


@pytest.mark.with_downloads
@pytest.mark.unit
def test_stateful_rnnt_pipeline_keeps_whole_frame_contexts_and_rounds_partial_ones_up():
    pipeline = _buffered_pipeline("rnnt", left_padding_size=5.6, chunk_size=0.5, right_padding_size=0.56)

    frames = (pipeline.tokens_per_left_padding, pipeline.tokens_per_frame, pipeline.tokens_per_right_padding)
    assert frames == (70, 7, 7)
    assert pipeline.chunk_size == pytest.approx(0.56)
    assert pipeline.right_padding_size == 0.56
