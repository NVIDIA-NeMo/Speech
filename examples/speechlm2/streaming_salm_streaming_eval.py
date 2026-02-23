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
"""
Streaming (online) evaluation script for StreamingSALM.

Uses the true-streaming pipeline (:class:`StreamingSALMPipeline`) to process
audio chunk-by-chunk with KV-cache, just as it would run in a real-time
server.  Computes WER against reference transcripts from a Lhotse manifest.

Usage::

    python streaming_salm_streaming_eval.py \
        pretrained_name=path/to/hf_model \
        inputs=/data/test_cuts.jsonl.gz \
        chunk_size=0.5 \
        latency=1
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.inference.model_wrappers.streaming_salm_inference_wrapper import (
    StreamingSALMInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.streaming_salm_pipeline import StreamingSALMPipeline
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class StreamingSALMStreamingEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    chunk_size: float = 0.5
    latency: int = 1
    context: Optional[str] = None
    batch_size: int = 1
    device: str = "cuda"
    compute_dtype: str = "bfloat16"
    output_manifest: Optional[str] = "streaming_salm_streaming_generations.jsonl"
    verbose: bool = True
    use_normalizer: Optional[str] = "english"


@hydra_runner(config_name="StreamingSALMStreamingEvalConfig", schema=StreamingSALMStreamingEvalConfig)
def main(cfg: StreamingSALMStreamingEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Load model via inference wrapper
    wrapper = StreamingSALMInferenceWrapper(
        model_name=cfg.pretrained_name,
        device=cfg.device,
        compute_dtype=cfg.compute_dtype,
        latency=cfg.latency,
        context=cfg.context,
    )

    # 2. Build pipeline config
    pipeline_cfg = OmegaConf.create(
        {
            "streaming": {
                "sample_rate": wrapper.sample_rate,
                "batch_size": cfg.batch_size,
                "chunk_size": cfg.chunk_size,
                "latency": cfg.latency,
                "context": cfg.context,
            }
        }
    )

    # 3. Construct pipeline
    pipeline = StreamingSALMPipeline(pipeline_cfg, wrapper)

    # 4. Extract audio filepaths and references from Lhotse manifest
    cuts = guess_parse_cutset(cfg.inputs)
    audio_filepaths = []
    refs = []
    cut_metadata = []  # (id, duration) for output manifest
    for cut in cuts:
        audio_filepaths.append(cut.recording.sources[0].source)
        refs.append(cut.supervisions[0].text)
        cut_metadata.append({"id": cut.id, "duration": cut.duration})

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(
        _normalizer_key, lambda x: x
    )
    refs = [normalizer(r) for r in refs]

    # 5. Run pipeline
    logging.info(f"Running streaming pipeline on {len(audio_filepaths)} utterances...")
    ts = perf_counter()
    output = pipeline.run(audio_filepaths)
    elapsed = perf_counter() - ts

    # 6. Collect hypotheses (stream IDs are 0-indexed, matching input order)
    hyps = []
    for sid in range(len(audio_filepaths)):
        text = output[sid]["text"] if sid in output else ""
        hyps.append(normalizer(text.strip()))

    # 7. Compute WER
    total_duration = sum(m["duration"] for m in cut_metadata)
    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = total_duration / elapsed if elapsed > 0 else float("inf")
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f} ({total_duration:.1f}s audio in {elapsed:.1f}s)")

    if cfg.verbose:
        for i, (ref, hyp) in enumerate(zip(refs, hyps)):
            logging.info(f"  [{i}] REF: {ref}")
            logging.info(f"  [{i}] HYP: {hyp}")

    # 8. Write output manifest
    if cfg.output_manifest is not None:
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for meta, ref, hyp in zip(cut_metadata, refs, hyps):
                writer.write({"id": meta["id"], "duration": meta["duration"], "text": ref, "pred_text": hyp})
        logging.info(f"Wrote {len(hyps)} entries to {cfg.output_manifest}")


if __name__ == "__main__":
    main()
