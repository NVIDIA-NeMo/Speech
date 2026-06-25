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
"""
Offline evaluation script for StreamingSTTModel.

Usage::

    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/test.jsonl \
        batch_size=32

    # Simulate streaming (chunk-by-chunk with blanks):
    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/test.jsonl \
        simulate_streaming=true

The model's ``generate()`` method returns ``list[str]`` directly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import GenerationConfig
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.data.lhotse.dataloader import pad_extra_duration
from nemo.collections.speechlm2.models import StreamingSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class StreamingSTTGenerationConfig:
    """
    A proxy class for GenerationConfig so that we can use OmegaConf with hydra overrides.
    All parameters will be passed to GenerationConfig.
    """

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


@dataclass
class StreamingSTTEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    batch_size: int = 64
    num_workers: int = 4
    max_new_tokens: int = 64
    system_prompt: str = "Transcribe the audio into text."
    output_manifest: Optional[str] = "streaming_stt_generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none"
    use_offline_embs: bool = False
    seed: Optional[int] = None  # Set for deterministic results
    pad_extra_duration: Optional[float] = 0.0
    use_state_machine_inference: bool = (
        False  # recommended turned off for chunk_size > 0, no effect for chunk_size <= 0
    )
    dynamic_min_chunk_size: int = 0  # dynamic chunking: min frames before allowing generation
    dynamic_max_chunk_size: Optional[int] = None  # dynamic chunking: max frames before forcing generation
    # Fixed-chunk size (frames) to run inference at. None → use the model config
    # chunk_size (the longest value when the model was trained with a list of sizes).
    chunk_size_override: Optional[int] = None
    # Probability threshold for the boundary decision.
    #   - When use_chunk_classifier_at_inference=True: threshold on the aux
    #     head's sigmoid output. None → 0.5 default.
    #   - When False: threshold on p(user_footer_first_id) from the LM head.
    #     None → fall back to argmax (legacy behavior).
    emit_threshold: Optional[float] = None
    # When True, dump per-LISTENING-frame diagnostics (LM head top-5, prob of
    # user_footer_first / blank, aux head sigmoid, decision taken) to a
    # sibling JSONL alongside output_manifest. Slows inference; use on
    # small eval sets when debugging boundary-decision behavior.
    emit_delay_frames: int = 0
    # K-frame grouping for dynamic-chunking read/write decisions. When None,
    # falls back to model.core_cfg.dynamic_chunk_step (= the value the model
    # was trained with). Override for ablations only.
    dynamic_chunk_step: Optional[int] = None
    disable_emit_for_debug: bool = False
    debug_log_audio_frames: bool = False
    # NOTE: ``use_chunk_classifier_at_inference`` is removed — the aux head is
    # used automatically when the model was trained with ``use_chunk_classifier=True``.
    # When True, save per-word alignments alongside pred_text in the output
    # manifest. Each predicted word inherits the start_time / end_time of the
    # audio chunk it was generated from. Format matches the GT manifest:
    #   [{"text": "...", "start_time": float_s, "end_time": float_s}, ...]
    save_alignments: bool = True
    generation_config: StreamingSTTGenerationConfig = field(default_factory=StreamingSTTGenerationConfig)


@hydra_runner(config_name="StreamingSTTEvalConfig", schema=StreamingSTTEvalConfig)
def main(cfg: StreamingSTTEvalConfig):
    if cfg.max_new_tokens is not None or cfg.max_new_tokens > 0:
        cfg.generation_config.max_new_tokens = cfg.max_new_tokens
        logging.warning(f"Setting generation_config.max_new_tokens to {cfg.max_new_tokens}")
        logging.warning(
            f"Using `max_new_tokens` is deprecated, please use `generation_config.max_new_tokens` instead."
        )

    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.seed is not None:
        logging.warning(f"Setting random seed to {cfg.seed}, this will slow down the inference")
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        logging.warning("Random seed not set, results will not be deterministic")

    model = StreamingSTTModel.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    cuts = guess_parse_cutset(cfg.inputs)
    # Resample to model's expected sample rate if needed.
    sample_cut = next(iter(cuts))
    if sample_cut.sampling_rate != model.sampling_rate:
        logging.info(f"Resampling cuts from {sample_cut.sampling_rate} to {model.sampling_rate} Hz")
        cuts = CutSet.from_cuts(c.resample(model.sampling_rate) for c in cuts)
    cuts = cuts.sort_by_duration()
    cuts = cuts.map(partial(pad_extra_duration, extra_duration=cfg.pad_extra_duration))
    sampler = lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size)
    num_batches = math.ceil(len(cuts) / cfg.batch_size)
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=sampler,
        num_workers=cfg.num_workers,
        batch_size=None,
    )

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(_normalizer_key, lambda x: x)

    refs = []
    hyps = []
    input_durations = []
    infer_durations = []
    # Per-cut diagnostic lists, ordered to match `cuts` after `sort_by_duration`.
    cut_alignments: Optional[list[list[dict]]] = [] if cfg.save_alignments else None
    cut_content_scores: list[list[float]] = []
    cut_pred_text_annotated: list[str] = []
    content_score_mode: Optional[str] = None

    # Optional per-frame debug log file (one record per LISTENING frame per
    # cut, keyed by cut id). Only opened when debug_log_audio_frames=True.
    debug_log_writer = None
    if cfg.debug_log_audio_frames and cfg.output_manifest is not None:
        manifest_path = Path(cfg.output_manifest)
        debug_log_path = manifest_path.with_name(
            manifest_path.stem.replace("_generations", "") + "_audio_frame_log.jsonl"
        )
        debug_log_writer = SequentialJsonlWriter(str(debug_log_path))
        logging.info(f"Audio frame debug log → {debug_log_path}")

    for batch_idx, batch in tqdm(enumerate(dloader), total=num_batches):
        ts = perf_counter()
        generation_config = GenerationConfig(**OmegaConf.to_container(cfg.generation_config))
        result = model.generate(
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            system_prompt=cfg.system_prompt,
            max_new_tokens=cfg.max_new_tokens,
            generation_config=generation_config,
            use_offline_embs=cfg.use_offline_embs,
            use_state_machine_inference=cfg.use_state_machine_inference,
            dynamic_min_chunk_size=cfg.dynamic_min_chunk_size,
            dynamic_max_chunk_size=cfg.dynamic_max_chunk_size,
            emit_threshold=cfg.emit_threshold,
            emit_delay_frames=cfg.emit_delay_frames,
            dynamic_chunk_step=cfg.dynamic_chunk_step,
            disable_emit_for_debug=cfg.disable_emit_for_debug,
            chunk_size_override=cfg.chunk_size_override,
            return_alignments=cfg.save_alignments,
            return_debug_logs=cfg.debug_log_audio_frames,
        )
        batch_infer_duration = perf_counter() - ts

        # Write per-frame debug records keyed by cut id.
        if debug_log_writer is not None and result.debug_logs is not None:
            for cut, frames in zip(batch["cuts"], result.debug_logs):
                debug_log_writer.write({"id": cut.id, "duration": cut.duration, "frames": frames})

        if result.content_score_mode is not None:
            content_score_mode = result.content_score_mode
        if result.content_scores is not None:
            cut_content_scores.extend(result.content_scores)
        if result.pred_text_annotated is not None:
            cut_pred_text_annotated.extend(result.pred_text_annotated)

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [normalizer(h.strip()) for h in result.texts]

        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info("--------------------------------")
            logging.info(
                f"Batch {batch_idx}: "
                f"WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] "
                f"RTFx={batch_rtfx:.1f}"
            )
            for ref, hyp in zip(batch_refs, batch_hyps):
                logging.info(f"\n[REF]\t`{ref}`\n[HYP]\t`{hyp}`\n")
            logging.info("--------------------------------")

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        if cut_alignments is not None and result.pred_alignments is not None:
            cut_alignments.extend(result.pred_alignments)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    if debug_log_writer is not None:
        debug_log_writer.close()

    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        log_file = Path(cfg.output_manifest).parent / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"======{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}======\n")
            f.write(f"Input: {cfg.inputs}\n")
            f.write(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]\n")
            f.write(f"RTFx: {rtfx:.1f}\n")
            f.write(f"=============================================\n\n")
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for idx, (cut, ref, hyp) in enumerate(zip(cuts, refs, hyps)):
                wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)
                record = {
                    "id": cut.id,
                    "duration": cut.duration,
                    "text": ref,
                    "pred_text": hyp,
                    "wer": wer,
                    "ins": nins,
                    "del": ndel,
                    "sub": nsub,
                }
                # Per-word predicted timestamps (same schema as the GT
                # manifest's `alignments` field: text / start_time / end_time,
                # in seconds). Stored as `pred_alignments` to mirror the
                # `pred_text` / `text` naming convention. Words from the
                # same chunk share that chunk's audio span.
                if cut_alignments is not None and idx < len(cut_alignments):
                    record["pred_alignments"] = cut_alignments[idx]
                if idx < len(cut_content_scores):
                    record["content_scores"] = cut_content_scores[idx]
                    record["content_score_mode"] = content_score_mode
                if idx < len(cut_pred_text_annotated):
                    record["pred_text_annotated"] = cut_pred_text_annotated[idx]
                writer.write(record)


if __name__ == "__main__":
    main()
