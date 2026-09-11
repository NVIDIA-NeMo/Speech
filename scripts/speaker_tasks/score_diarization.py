#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Score diarization from RTTM files, RTTM directories, or diarization manifests."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Prefer this checkout over any NeMo version installed in the active environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.parts.utils.speaker_utils import labels_to_supervisions


LabelsByRecording = Dict[str, List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate NeMo diarization error metrics from RTTM files or directories.",
        add_help=False,
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Reference manifest, compound RTTM, or directory of per-recording RTTMs",
    )
    parser.add_argument(
        "-h",
        "--hypothesis",
        type=Path,
        required=True,
        help="Hypothesis manifest, compound RTTM, or directory of per-recording RTTMs",
    )
    parser.add_argument("-c", "--collar", type=float, default=0.0, help="NeMo collar half-width in seconds")
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    args = parser.parse_args()
    if args.collar < 0:
        parser.error("collar must be non-negative")
    return args


def read_rttm(path: Path, empty_recording_id: str | None = None, allow_empty: bool = False) -> LabelsByRecording:
    """Read an RTTM into NeMo ``start end speaker`` labels grouped by recording ID."""
    if not path.is_file():
        raise FileNotFoundError(f"RTTM file not found: {path}")

    labels: LabelsByRecording = defaultdict(list)
    with path.open(encoding="utf-8") as rttm:
        for line_number, line in enumerate(rttm, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] != "SPEAKER":
                continue
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected at least 8 RTTM fields")
            try:
                start = round(float(fields[3]), 3)
                duration = round(float(fields[4]), 3)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid start or duration") from error
            if duration < 0:
                raise ValueError(f"{path}:{line_number}: duration must be non-negative")

            end = start + duration
            labels[fields[1]].append(f"{start} {end} {fields[7]}")

    if not labels:
        if empty_recording_id is not None:
            return {empty_recording_id: []}
        if allow_empty:
            return {}
        raise ValueError(f"No SPEAKER records found in {path}")
    return dict(labels)


def merge_recordings(destination: LabelsByRecording, source: LabelsByRecording, source_path: Path) -> None:
    """Merge recording labels while rejecting duplicate IDs across per-file RTTMs."""
    duplicate_ids = sorted(set(destination).intersection(source))
    if duplicate_ids:
        raise ValueError(
            f"Duplicate recording IDs across RTTM files in {source_path.parent}: {', '.join(duplicate_ids[:5])}"
        )
    destination.update(source)


def get_rttm_files(directory: Path) -> Dict[str, Path]:
    """Return direct child RTTMs keyed by exact filename."""
    files = {path.name: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".rttm"}
    if not files:
        raise ValueError(f"No RTTM files found in directory: {directory}")
    return files


def read_rttm_directory(directory: Path) -> LabelsByRecording:
    """Read every direct-child RTTM in a directory."""
    recordings: LabelsByRecording = {}
    for path in sorted(get_rttm_files(directory).values()):
        merge_recordings(recordings, read_rttm(path, empty_recording_id=path.stem), path)
    return recordings


def read_rttm_source(path: Path, allow_empty: bool = False) -> LabelsByRecording:
    """Read a compound RTTM file or a directory of per-recording RTTMs."""
    if path.is_file():
        return read_rttm(path, allow_empty=allow_empty)
    if path.is_dir():
        return read_rttm_directory(path)
    raise FileNotFoundError(f"RTTM path not found: {path}")


def resolve_manifest_path(path_value: str | None, manifest_path: Path) -> Path | None:
    """Resolve an optional manifest entry path relative to the manifest directory."""
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def read_rttm_manifest(manifest_path: Path) -> Tuple[LabelsByRecording, Dict[str, Dict]]:
    """Read RTTMs and scoring metadata from an e2e diarization JSON-lines manifest."""
    recordings: LabelsByRecording = {}
    audio_rttm_map = {}
    with manifest_path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON") from error
            if "audio_filepath" not in item:
                raise ValueError(f"{manifest_path}:{line_number}: audio_filepath is required")
            if not item.get("rttm_filepath"):
                raise ValueError(f"{manifest_path}:{line_number}: rttm_filepath is required")

            recording_id = item.get("uniq_id") or Path(item["audio_filepath"]).stem
            if recording_id in recordings:
                raise ValueError(f"{manifest_path}:{line_number}: duplicate recording ID: {recording_id}")

            rttm_path = resolve_manifest_path(item["rttm_filepath"], manifest_path)
            rttm_recordings = read_rttm(rttm_path, empty_recording_id=recording_id)
            if recording_id in rttm_recordings:
                recordings[recording_id] = rttm_recordings[recording_id]
            elif len(rttm_recordings) == 1:
                recordings[recording_id] = next(iter(rttm_recordings.values()))
            else:
                raise ValueError(f"{rttm_path}: recording ID {recording_id} was not found in a multi-recording RTTM")

            uem_path = resolve_manifest_path(item.get("uem_filepath"), manifest_path)
            audio_rttm_map[recording_id] = {
                "audio_filepath": item["audio_filepath"],
                "rttm_filepath": str(rttm_path),
                "offset": item.get("offset"),
                "duration": item.get("duration"),
                "uem_filepath": None if uem_path is None else str(uem_path),
            }

    if not recordings:
        raise ValueError(f"No manifest entries found in {manifest_path}")
    return recordings, audio_rttm_map


def read_rttm_inputs(reference_path: Path, hypothesis_path: Path) -> Tuple[LabelsByRecording, LabelsByRecording]:
    """Read any file/directory RTTM pairing, requiring exact filenames when both are directories."""
    if reference_path.is_file() and hypothesis_path.is_file():
        reference = read_rttm(reference_path)
        hypothesis = read_rttm(hypothesis_path, allow_empty=True)
        if not hypothesis:
            hypothesis = {recording_id: [] for recording_id in reference}
        return reference, hypothesis

    if reference_path.is_dir() and hypothesis_path.is_dir():
        reference_files = get_rttm_files(reference_path)
        hypothesis_files = get_rttm_files(hypothesis_path)
        reference_names = set(reference_files)
        hypothesis_names = set(hypothesis_files)
        if reference_names != hypothesis_names:
            missing = sorted(reference_names - hypothesis_names)
            extra = sorted(hypothesis_names - reference_names)
            details = []
            if missing:
                details.append(f"missing in hypothesis: {', '.join(missing[:5])}")
            if extra:
                details.append(f"not in reference: {', '.join(extra[:5])}")
            raise ValueError("Reference and hypothesis RTTM filenames do not match (" + "; ".join(details) + ")")

        reference: LabelsByRecording = {}
        hypothesis: LabelsByRecording = {}
        for filename in sorted(reference_names):
            reference_file_recordings = read_rttm(
                reference_files[filename], empty_recording_id=reference_files[filename].stem
            )
            empty_hypothesis_id = (
                next(iter(reference_file_recordings))
                if len(reference_file_recordings) == 1
                else hypothesis_files[filename].stem
            )
            hypothesis_file_recordings = read_rttm(hypothesis_files[filename], empty_recording_id=empty_hypothesis_id)
            merge_recordings(reference, reference_file_recordings, reference_files[filename])
            merge_recordings(hypothesis, hypothesis_file_recordings, hypothesis_files[filename])
        return reference, hypothesis

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference path not found: {reference_path}")
    if not hypothesis_path.exists():
        raise FileNotFoundError(f"Hypothesis path not found: {hypothesis_path}")
    reference = read_rttm_source(reference_path)
    hypothesis = read_rttm_source(hypothesis_path, allow_empty=True)
    if not hypothesis:
        hypothesis = {recording_id: [] for recording_id in reference}
    return reference, hypothesis


def align_recording_ids(
    reference: LabelsByRecording, hypothesis: LabelsByRecording
) -> Tuple[LabelsByRecording, str | None]:
    """Align recording IDs, filling missing hypotheses while rejecting extra IDs."""
    ref_ids = set(reference)
    extra = sorted(set(hypothesis) - ref_ids)
    if extra:
        raise ValueError(f"Hypothesis contains recording IDs not in reference: {', '.join(extra[:5])}")

    aligned = dict(hypothesis)
    missing = sorted(ref_ids - set(hypothesis))
    if missing:
        for recording_id in missing:
            aligned[recording_id] = []

    aligned = {recording_id: aligned[recording_id] for recording_id in reference}
    message = f"Treating missing hypothesis recording IDs as empty: {', '.join(missing[:5])}." if missing else None
    return aligned, message


def main() -> None:
    args = parse_args()
    reference_is_manifest = args.reference.is_file() and args.reference.suffix.lower() in {".json", ".jsonl"}
    hypothesis_is_manifest = args.hypothesis.is_file() and args.hypothesis.suffix.lower() in {".json", ".jsonl"}
    if reference_is_manifest or hypothesis_is_manifest:
        if reference_is_manifest:
            reference, audio_rttm_map = read_rttm_manifest(args.reference)
        else:
            reference = read_rttm_source(args.reference)
            audio_rttm_map = {recording_id: {} for recording_id in reference}
        if hypothesis_is_manifest:
            hypothesis, _ = read_rttm_manifest(args.hypothesis)
        else:
            hypothesis = read_rttm_source(args.hypothesis, allow_empty=True)
            if not hypothesis:
                hypothesis = {recording_id: [] for recording_id in reference}
    else:
        reference, hypothesis = read_rttm_inputs(args.reference, args.hypothesis)
        audio_rttm_map = {recording_id: {} for recording_id in reference}
    hypothesis, alignment_message = align_recording_ids(reference, hypothesis)
    recording_ids = sorted(reference)

    all_reference = []
    all_hypothesis = []
    for recording_id in recording_ids:
        all_reference.append([recording_id, labels_to_supervisions(reference[recording_id], uniq_name=recording_id)])
        all_hypothesis.append([recording_id, labels_to_supervisions(hypothesis[recording_id], uniq_name=recording_id)])

    if alignment_message:
        print(alignment_message, flush=True)
    print(f"Recordings: {len(recording_ids)}", flush=True)
    print(f"Collar: {args.collar:g} sec; ignore_overlap: False", flush=True)

    result = score_labels(
        AUDIO_RTTM_MAP=audio_rttm_map,
        all_reference=all_reference,
        all_hypothesis=all_hypothesis,
        collar=args.collar,
        ignore_overlap=False,
        verbose=True,
    )
    if result is None:
        raise RuntimeError("NeMo DER scoring failed")


if __name__ == "__main__":
    main()
