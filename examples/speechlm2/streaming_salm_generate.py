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
Offline evaluation script for StreamingSALM.

Usage::

    python streaming_salm_generate.py \
        pretrained_name=nvidia/streaming-salm-v1 \
        inputs=/data/test.jsonl \
        batch_size=32 \
        latency=3

Unlike ``salm_eval.py``, StreamingSALM builds its own prompt internally
(no chat template, no prompt formatting).  The model's ``generate()`` method
returns ``list[str]`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2.models import StreamingSALM
from nemo.core.config import hydra_runner
from nemo.utils import logging


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class StreamingSALMEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    batch_size: int = 64
    latency: int = 1
    context: Optional[str] = None
    output_manifest: Optional[str] = "streaming_salm_generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none"


@hydra_runner(config_name="StreamingSALMEvalConfig", schema=StreamingSALMEvalConfig)
def main(cfg: StreamingSALMEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    model = StreamingSALM.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size),
        num_workers=1,
        batch_size=None,
    )

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(
        _normalizer_key, lambda x: x
    )

    refs = []
    hyps = []
    input_durations = []
    infer_durations = []

    for batch_idx, batch in enumerate(dloader):
        ts = perf_counter()
        batch_hyps_raw = model.generate(
            audio=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            latency=cfg.latency,
            context=cfg.context,
        )
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [normalizer(h.strip()) for h in batch_hyps_raw]

        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info(
                f"Batch {batch_idx}: "
                f"WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] "
                f"RTFx={batch_rtfx:.1f}"
            )

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for cut, ref, hyp in zip(cuts, refs, hyps):
                writer.write({"id": cut.id, "duration": cut.duration, "text": ref, "pred_text": hyp})


if __name__ == "__main__":
    main()
