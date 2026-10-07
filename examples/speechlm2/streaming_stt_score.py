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
"""Score a prediction manifest offline: cpWER and WER, no model, no GPU.

`streaming_stt_generate.py` still scores inline by default. This exists so that CHANGING how
scoring works does not mean re-running inference on a GPU: scoring is CPU-only and takes seconds,
inference takes a GPU and minutes. Re-running is reproducible (two runs at identical settings give
byte-identical manifests), but re-scoring a fixed manifest is both exact and free.

Requires the raw fields. A manifest written before those existed carries only `text` / `pred_text`,
which were normalized AND tag-stripped before they were written, so re-normalizing them is not the
same as normalizing the original and the result would be wrong undetectably. Such a manifest is
refused, with a message naming the recovery options.

Usage::

    # score a run with the defaults it was produced under
    python streaming_stt_score.py manifest=eval/run.jsonl

    # the same manifest under a different normalizer -- this is the point of the split
    python streaming_stt_score.py manifest=eval/run.jsonl cpwer_normalizer=chime8

    # a suffix-placement hypothesis against references written in prefix form
    python streaming_stt_score.py manifest=eval/run.jsonl cpwer_placement_ref=prefix cpwer_placement_hyp=suffix

    # inspect what a manifest was produced with, score nothing
    python streaming_stt_score.py manifest=eval/run.jsonl dry_run=true

    # recover a legacy manifest by joining the raw reference from its source
    python streaming_stt_score.py manifest=eval/old.jsonl \\
        reference_manifest=data/test.json hypothesis_field=pred_text_annotated

Hydra prints no field descriptions in ``--help``, so the knobs are documented here.

Scoring axes -- each independently selects a behaviour some other scorer has; every default
reproduces this repo's historical behaviour:

    cpwer_normalizer            None -> inherit use_normalizer; whisper | hf | chime8 | none
    cpwer_tag_syntax_ref/_hyp   spk | bracket | spk+bracket | canonical (+ aliases)
    cpwer_tag_case_sensitive    False treats <SPK:0> as a tag rather than two words
    cpwer_untagged_speaker_ref/_hyp   bucket for words before the first tag; null discards them
    cpwer_keep_empty_streams    False drops a stream whose raw text is empty
    cpwer_drop_tag_residue      False scores "<spk:0" as text
    cpwer_speaker_order         index | first_seen -- changes only the ins/del/sub split
    cpwer_ceiling_source        strip_tags | streams -- affects the no-tag ceiling only

Tag placement -- not an axis, since the reference scorer knows prefix only:

    cpwer_placement             prefix | suffix -- whether a tag opens or closes its speaker's run
    cpwer_placement_ref/_hyp    the same per role; null (the default) follows cpwer_placement

Scorer-only knobs:

    manifest              input jsonl (required)
    output_manifest       default <stem>.scored.jsonl -- never rewritten in place
    output_metrics        default <stem>.metrics.json
    log_file              default <stem>.scored.log; never the run's own log.txt
    reference_manifest    join raw references from a source manifest by id or sample_id
    reference_field       force a reference field instead of the resolution chain
    hypothesis_field      force a hypothesis field
    dry_run               report what the manifest was produced with, score nothing
    compute_wer           recompute the speaker-agnostic WER from the raw fields
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import MISSING, OmegaConf

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.asr.parts.utils.sot_speaker_alignment import remove_speaker_tags
from nemo.collections.asr.parts.utils.text_normalizers import build_normalizer
from nemo.collections.speechlm2.parts.metrics import (
    CpWERScoringConfig,
    join_reference_manifest,
    resolve_placements,
    score_rows,
)
from nemo.collections.speechlm2.parts.metrics.cpwer_report import cpwer_metrics_dict, format_cpwer_report
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class CpWERScoreConfig(CpWERScoringConfig):
    """Scorer settings: the shared scoring axes, plus where to read and write.

    Subclasses the shared config so every axis is spelled exactly as it is on the inference script;
    an override copied from one command line works on the other.
    """

    manifest: str = MISSING
    output_manifest: Optional[str] = None
    output_metrics: Optional[str] = None
    log_file: Optional[str] = None
    reference_manifest: Optional[str] = None
    reference_field: Optional[str] = None
    hypothesis_field: Optional[str] = None
    dry_run: bool = False
    compute_wer: bool = True


@hydra_runner(config_name="CpWERScoreConfig", schema=CpWERScoreConfig)
def main(cfg: CpWERScoreConfig):
    logging.info(f"Scoring config:\n{OmegaConf.to_yaml(cfg)}")
    # Hydra hands back a DictConfig; materialise the dataclass so its methods are bound and the
    # scoring code sees the same object type whether it was called from a CLI or from Python.
    cfg = OmegaConf.to_object(cfg)
    cfg.validate()

    manifest = Path(cfg.manifest)
    rows = _read_jsonl(manifest)
    logging.info(f"Read {len(rows)} rows from {manifest}")

    if cfg.reference_manifest:
        ref_rows = _read_jsonl(Path(cfg.reference_manifest))
        rows = join_reference_manifest(rows, ref_rows)
        logging.info(f"Joined raw references from {cfg.reference_manifest}")

    if cfg.dry_run:
        _report_provenance(rows, cfg)
        return

    _refuse_unscorable(rows)
    _warn_placement_mismatch(rows, cfg)

    per_row, corpus, subsets = score_rows(
        rows, cfg, reference_field=cfg.reference_field, hypothesis_field=cfg.hypothesis_field
    )
    metrics = cpwer_metrics_dict(corpus, subsets, cfg)

    wer = None
    if cfg.compute_wer:
        wer = _speaker_agnostic_wer(rows, cfg)
        metrics["wer"] = round(wer * 100, 2)

    report = format_cpwer_report(metrics, cfg, wer=wer)
    logging.info("\n" + report)

    out_manifest = Path(cfg.output_manifest or _default_path(manifest, ".scored.jsonl"))
    out_metrics = Path(cfg.output_metrics or _default_path(manifest, ".metrics.json"))
    out_log = Path(cfg.log_file or _default_path(manifest, ".scored.log"))

    scored = [{**row, **row_metrics} for row, row_metrics in zip(rows, per_row)]
    _write_jsonl(out_manifest, scored)
    _write_json(out_metrics, metrics)
    with open(out_log, "a") as handle:
        handle.write(report + "\n")
    logging.info(f"Wrote {out_manifest}, {out_metrics}, {out_log}")


def _refuse_unscorable(rows: list) -> None:
    """Stop before scoring text that cannot produce a correct number.

    A pre-split manifest carries only `text` / `pred_text`, already normalized and tag-stripped.
    Scoring those would silently produce a wrong denominator rather than fail, so it is refused
    here with the recovery options named.
    """
    if not rows:
        raise ValueError("manifest is empty")
    first = rows[0]
    if not (first.get("text_raw") or first.get("expected_answer")):
        raise ValueError(
            "This manifest has no raw reference (no `text_raw`, no `expected_answer`). Its `text` "
            "was normalized and tag-stripped before it was written, so re-scoring it would give a "
            "wrong number rather than an error. Either pass `reference_manifest=<source manifest>` "
            "to recover the raw reference, or re-run inference to produce a manifest with raw "
            "fields."
        )
    seg_rows = [r for r in rows if isinstance(r.get("_run"), dict) and r["_run"].get("seg_mode")]
    if seg_rows:
        raise ValueError(
            "cpWER requires globally consistent speaker indices, but <spk:N> is arrival-ordered "
            "WITHIN each decode window -- segments are decoded independently, so <spk:0> in one "
            "segment is generally a different person than in the next, and a session-global "
            "permutation cannot undo a per-segment relabeling. Score cpWER per cut "
            "(max_segment_duration=0), or add cross-segment speaker stitching."
        )
    placements = {_run_placements(r["_run"]) for r in rows if isinstance(r.get("_run"), dict)}
    if len(placements) > 1:
        raise ValueError(
            f"Rows disagree on tag placement (reference, hypothesis: {sorted(placements, key=str)}). This manifest "
            "is a concatenation of runs that are not comparable; score them separately."
        )


def _report_provenance(rows: list, cfg: CpWERScoreConfig) -> None:
    """Print what the manifest was produced with, and how the request differs. Score nothing."""
    runs = [r["_run"] for r in rows if isinstance(r.get("_run"), dict)]
    if not runs:
        logging.info("No `_run` block on these rows: a pre-split manifest, provenance unknown.")
        return
    first = runs[0]
    logging.info("Manifest was produced with:")
    for key in sorted(first):
        values = {json.dumps(r.get(key), sort_keys=True, default=str) for r in runs}
        suffix = "" if len(values) == 1 else f"   (!! {len(values)} distinct values across rows)"
        logging.info(f"    {key}: {first[key]!r}{suffix}")
    logging.info(f"Requested cpWER normalizer: {cfg.effective_normalizer()!r}")
    logging.info(f"Requested tag placement (reference, hypothesis): {resolve_placements(cfg)}")
    _warn_placement_mismatch(rows, cfg)
    if first.get("inference_normalizer") != cfg.effective_normalizer():
        logging.info(
            f"    differs from the inference-time normalizer {first.get('inference_normalizer')!r}; "
            "`text` / `pred_text` on these rows were written with the latter, but scoring reads the "
            "raw fields, so this is fine."
        )


def _speaker_agnostic_wer(rows: list, cfg: CpWERScoreConfig) -> float:
    """Corpus WER over tag-stripped text, recomputed from the raw fields.

    Tags are stripped BEFORE normalizing, so a pure speaker swap with identical words is not scored
    as word errors.
    """
    normalizer = build_normalizer(cfg.use_normalizer, cfg.normalizer_language)
    refs, hyps = [], []
    for row in rows:
        ref = row.get("text_raw") or row.get("expected_answer") or ""
        hyp = row.get("pred_text_raw") or row.get("predicted_answer") or row.get("generation") or ""
        refs.append(normalizer(remove_speaker_tags(ref)))
        hyps.append(normalizer(remove_speaker_tags(hyp)))
    wer, _, _, _, _ = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    return wer


def _default_path(manifest: Path, suffix: str) -> Path:
    """Sibling output path, never the input. Re-scoring a scored file must not stack suffixes."""
    stem = manifest.name
    for drop in (".jsonl", ".json"):
        if stem.endswith(drop):
            stem = stem[: -len(drop)]
    if stem.endswith(".scored"):
        stem = stem[: -len(".scored")]
    return manifest.parent / (stem + suffix)


def _read_jsonl(path: Path) -> list:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w") as handle:
        for row in rows:
            # allow_nan=False so no bare `Infinity` reaches a scored artifact: it is not valid JSON
            # and a strict reader rejects the whole file.
            handle.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as handle:
        json.dump(_json_safe(obj), handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _json_safe(obj):
    """Replace non-finite floats with None, which JSON can represent and a reader can test."""
    import math

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _run_placements(run: dict) -> tuple:
    """The ``(reference, hypothesis)`` placements of a ``_run`` block: a role's own where stamped, else ``placement``.

    A run that sets no role placement stamps neither, so its pair is ``(placement, placement)``.
    """
    placement = run.get("placement")
    ref, hyp = run.get("placement_ref"), run.get("placement_hyp")
    return (placement if ref is None else ref, placement if hyp is None else hyp)


def _warn_placement_mismatch(rows: list, cfg: CpWERScoreConfig) -> None:
    """Warn when the requested ``(reference, hypothesis)`` placements are not the ones the run stamped.

    Not refused: re-scoring under another placement is legitimate (references joined from another source, or a run
    scored with the wrong one), but it changes the number, so it must not pass silently.
    """
    runs = [r["_run"] for r in rows if isinstance(r.get("_run"), dict) and r["_run"].get("placement")]
    stamped = {_run_placements(run) for run in runs}
    requested = resolve_placements(cfg)
    if stamped and requested not in stamped:
        logging.warning(
            f"Scoring with tag placement (reference, hypothesis) {requested}, but the run stamped "
            f"{sorted(stamped)}. Set cpwer_placement (or cpwer_placement_ref / cpwer_placement_hyp) to the "
            "stamped pair unless the change is intended."
        )


if __name__ == "__main__":
    main()
