# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import IO, Any, Dict, Iterable, List, Optional, Tuple
from lhotse import SupervisionSegment

from nemo.collections.asr.metrics.md_eval import (
    EPSILON,
    NOEVAL_SD,
    DiarizationErrorResult,
    SpeakerMap,
    SpeakerOverlap,
    _annotation_to_rttm_data,
    _iter_annotation_segments,
    _labels_to_rttm_data,
    _merge_rttm_dicts,
    _merge_uem_dicts,
    _uem_list_to_uem_data,
    add_exclusion_zones_to_uem,
    create_speaker_segs,
    evaluate,
    get_uem_data,
    map_speakers,
    uem_from_rttm,
)
from nemo.utils import logging

__all__ = [
    'get_partial_ref_labels',
    'get_online_DER_stats',
    'score_labels',
    'evaluate_der',
    'score_labels_from_rttm_labels',
    # Lhotse-backed annotation/segment/timeline helpers.
    'make_diar_segment',
    'make_diar_annotation',
    'make_uem_timeline',
    'unique_speakers',
    'write_supervisions_to_rttm',
]


# ─── Lhotse-backed annotation helpers ──────────────────────────────────────
#
# NeMo's diarization code uses lhotse's ``SupervisionSegment`` /
# ``SupervisionSet`` (already a hard dependency via ``requirements_asr.txt``)
# as the carrier type for diarization annotations and UEM timelines. The
# helpers below provide a small adapter layer used throughout the DER
# pipeline:
#
#   * ``make_diar_segment``        — build a single ``SupervisionSegment``
#     ``(start, end, speaker)``.
#   * ``make_diar_annotation``     — build an annotation as a list of
#     ``SupervisionSegment`` from ``"start end speaker"`` label strings.
#   * ``make_uem_timeline``        — build a UEM (evaluation regions) timeline
#     as a list of ``SupervisionSegment`` with ``speaker="UEM"`` so all
#     annotation-like objects are uniform.
#   * ``unique_speakers``          — return the unique speaker labels in an
#     annotation.
#   * ``write_supervisions_to_rttm`` — serialize an annotation to RTTM lines.
#
# Downstream consumers (``md_eval._annotation_to_rttm_data``) duck-type the
# input so any iterable of objects exposing ``start``/``end`` (or
# ``duration``) and ``speaker`` is accepted.


_DIAR_RECORDING_ID_PLACEHOLDER = "__diar__"


def make_diar_segment(
    start: float,
    end: float,
    speaker: str,
    recording_id: str = _DIAR_RECORDING_ID_PLACEHOLDER,
    segment_id: Optional[str] = None,
) -> SupervisionSegment:
    """Build a single diarization segment as a lhotse ``SupervisionSegment``.

    Args:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        speaker: Speaker label.
        recording_id: Recording (file) identifier (analogous to a recording
            URI). Defaults to a placeholder when the caller does not yet
            know the file id.
        segment_id: Optional unique segment id; auto-generated when omitted.

    Returns:
        A ``SupervisionSegment`` with ``start``, ``duration``, and
        ``speaker`` populated.
    """
    duration = max(0.0, float(end) - float(start))
    if segment_id is None:
        segment_id = f"{recording_id}-{float(start):.6f}-{float(end):.6f}-{speaker}"
    return SupervisionSegment(
        id=segment_id,
        recording_id=recording_id,
        start=float(start),
        duration=duration,
        speaker=str(speaker),
    )


def make_diar_annotation(
    labels: Iterable[str],
    uniq_name: str = "",
    audio_end: Optional[float] = None,
) -> List[SupervisionSegment]:
    """Build a diarization annotation from ``"start end speaker"`` label strings.

    Returns a list of ``SupervisionSegment`` accepted by every NeMo DER
    helper.

    Args:
        labels (Iterable[str]): Iterable of label strings, each formatted as
            ``"start end speaker"``.
        uniq_name (str): Recording / file identifier (used as the recording id
            of each emitted supervision).
        audio_end (Optional[float]): If provided, segment end times are capped
            at this value (typically ``offset + duration`` from the manifest).
            Segments that fall entirely past ``audio_end`` are dropped.

    Returns:
        List[SupervisionSegment]: Supervision segments, one per valid label line.
    """
    recording_id = uniq_name or _DIAR_RECORDING_ID_PLACEHOLDER
    segments: List[SupervisionSegment] = []
    for idx, label in enumerate(labels):
        parts = label.strip().split()
        if len(parts) < 3:
            continue
        start, end = float(parts[0]), float(parts[1])
        speaker = parts[2]
        if audio_end is not None:
            end = min(end, audio_end)
            if end <= start:
                continue
        segments.append(
            make_diar_segment(
                start=start,
                end=end,
                speaker=speaker,
                recording_id=recording_id,
                segment_id=f"{recording_id}-{idx}",
            )
        )
    return segments


def make_uem_timeline(
    uem_lines: Iterable[Iterable[float]],
    uniq_id: str,
) -> List[SupervisionSegment]:
    """Build a UEM (evaluation region) timeline as a list of supervisions.

    Each region is represented as a ``SupervisionSegment`` with
    ``speaker="UEM"`` so the same iteration patterns used for annotations
    also work for UEMs.

    Args:
        uem_lines: Iterable of ``[start, end]`` pairs in seconds.
        uniq_id: Recording / file identifier.

    Returns:
        List of ``SupervisionSegment`` objects representing the evaluation
        regions for ``uniq_id``.
    """
    segments: List[SupervisionSegment] = []
    for idx, span in enumerate(uem_lines):
        span_list = list(span)
        if len(span_list) < 2:
            continue
        start, end = float(span_list[0]), float(span_list[1])
        segments.append(
            SupervisionSegment(
                id=f"{uniq_id}-uem-{idx}",
                recording_id=uniq_id,
                start=start,
                duration=max(0.0, end - start),
                speaker="UEM",
            )
        )
    return segments


def unique_speakers(annotation: Any) -> List[str]:
    """Return the unique speaker labels in an annotation-like object.

    Accepts the lhotse-based annotation objects used throughout NeMo's DER
    pipeline (list of ``SupervisionSegment`` / ``SupervisionSet``) as well as
    any iterable whose items expose ``start`` / ``end`` (or ``duration``) and
    ``speaker``. If the input exposes a ``.labels()`` method it is used
    directly.
    """
    if hasattr(annotation, "labels") and not isinstance(annotation, (list, tuple)):
        try:
            return list(annotation.labels())
        except TypeError:
            pass
    seen: List[str] = []
    seen_set = set()
    for _start, _end, speaker in _iter_annotation_segments(annotation):
        if speaker not in seen_set:
            seen.append(speaker)
            seen_set.add(speaker)
    return seen


def write_supervisions_to_rttm(
    annotation: Any,
    file_handle: IO[str],
    recording_id: Optional[str] = None,
    channel: int = 1,
) -> None:
    """Write an annotation-like object to ``file_handle`` in NIST RTTM format.

    Args:
        annotation: Iterable of ``SupervisionSegment`` (or any object
            accepted by :func:`md_eval._iter_annotation_segments`).
        file_handle: An open text file handle.
        recording_id: Recording identifier emitted in the second RTTM
            column. When omitted, the ``recording_id`` of the first
            supervision (or an empty string) is used.
        channel: Channel id (1-indexed); RTTM convention defaults to ``1``.
    """
    if recording_id is None:
        first = next(iter(annotation), None)
        recording_id = getattr(first, "recording_id", "") if first is not None else ""

    # RTTM lines must be sorted by onset time to match the standard format.
    segments = sorted(
        _iter_annotation_segments(annotation),
        key=lambda x: (x[0], x[1], x[2]),
    )

    # Write RTTM lines in sorted order
    for start, end, speaker in segments:
        duration = end - start
        if duration <= 0:
            continue
        file_handle.write(
            f"SPEAKER {recording_id} {channel} {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )


def get_partial_ref_labels(pred_labels: List[str], ref_labels: List[str]) -> List[str]:
    """
    For evaluation of online diarization performance, generate partial reference labels
    from the last prediction time.

    Args:
        pred_labels (list[str]): list of partial prediction labels
        ref_labels (list[str]): list of full reference labels

    Returns:
        ref_labels_out (list[str]): list of partial reference labels
    """
    # If there is no reference, return empty list
    if len(ref_labels) == 0:
        return []

    # If there is no prediction, set the last prediction time to 0
    if len(pred_labels) == 0:
        last_pred_time = 0
    else:
        # The lastest prediction time in the prediction labels
        last_pred_time = max([float(labels.split()[1]) for labels in pred_labels])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        # If the current [start, end] interval extends beyond the end of hypothesis time stamps
        if start < last_pred_time:
            end_time = min(end, last_pred_time)
            label = f"{start} {end_time} {speaker}"
            ref_labels_out.append(label)
        # Other cases where the current [start, end] interval is before the last prediction time
        elif end < last_pred_time:
            ref_labels_out.append(label)
    return ref_labels_out


def get_online_DER_stats(
    DER: float,
    CER: float,
    FA: float,
    MISS: float,
    diar_eval_count: int,
    der_stat_dict: Dict[str, float],
    deci: int = 3,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    For evaluation of online diarization performance, add cumulative, average, and maximum DER/CER.

    Args:
        DER (float): Diarization Error Rate from the start to the current point
        CER (float): Confusion Error Rate from the start to the current point
        FA (float): False Alarm from the start to the current point
        MISS (float): Miss rate from the start to the current point
        diar_eval_count (int): Number of evaluation sessions
        der_stat_dict (dict): Dictionary containing cumulative, average, and maximum DER/CER
        deci (int): Number of decimal places to round

    Returns:
        der_dict (dict): Dictionary containing DER, CER, FA, and MISS
        der_stat_dict (dict): Dictionary containing cumulative, average, and maximum DER/CER
    """
    der_dict = {
        "DER": round(100 * DER, deci),
        "CER": round(100 * CER, deci),
        "FA": round(100 * FA, deci),
        "MISS": round(100 * MISS, deci),
    }
    der_stat_dict['cum_DER'] += DER
    der_stat_dict['cum_CER'] += CER
    der_stat_dict['avg_DER'] = round(100 * der_stat_dict['cum_DER'] / diar_eval_count, deci)
    der_stat_dict['avg_CER'] = round(100 * der_stat_dict['cum_CER'] / diar_eval_count, deci)
    der_stat_dict['max_DER'] = round(max(der_dict['DER'], der_stat_dict['max_DER']), deci)
    der_stat_dict['max_CER'] = round(max(der_dict['CER'], der_stat_dict['max_CER']), deci)
    return der_dict, der_stat_dict


def _build_mapping_from_data(
    ref_data: Dict,
    sys_data: Dict,
    uem_data: Optional[Dict],
) -> Dict[str, SpeakerMap]:
    """Build per-file optimal speaker mappings from parsed RTTM data."""
    mapping_dict: Dict[str, SpeakerMap] = {}
    for file_id in sorted(ref_data.keys()):
        for chnl in sorted(ref_data[file_id].keys()):
            ref_spkr_data = ref_data[file_id][chnl].get("SPEAKER")
            sys_spkr_data = sys_data.get(file_id, {}).get(chnl, {}).get("SPEAKER", {})
            if not ref_spkr_data:
                continue

            for segs in sys_spkr_data.values():
                for seg in segs:
                    seg.setdefault("RTBEG", seg["TBEG"])
                    seg.setdefault("RTEND", seg["TEND"])
                    seg["RTDUR"] = seg["RTEND"] - seg["RTBEG"]
                    seg["RTMID"] = seg["RTBEG"] + seg["RTDUR"] / 2.0

            ref_rttm = ref_data[file_id][chnl].get("RTTM", [])
            uem = uem_data.get(file_id, {}).get(chnl) if uem_data else None
            if uem is None:
                uem = uem_from_rttm(ref_rttm)

            uem_sd_eval = add_exclusion_zones_to_uem(NOEVAL_SD, uem, ref_rttm)
            if not uem_sd_eval:
                uem_sd_eval = uem

            eval_segs = create_speaker_segs(uem_sd_eval, ref_spkr_data, sys_spkr_data)
            spkr_overlap: SpeakerOverlap = {}
            for seg in eval_segs:
                if not seg["REF"]:
                    continue
                for rs in seg["REF"]:
                    for ss in seg["SYS"]:
                        spkr_overlap.setdefault(rs, {})[ss] = spkr_overlap.get(rs, {}).get(ss, 0.0) + seg["TDUR"]
            mapping_dict[file_id] = map_speakers(spkr_overlap)
    return mapping_dict


def _extract_errors(cum: Dict) -> Tuple[float, float, float, float]:
    """Extract (DER, CER, FA, MISS) from aggregate stats."""
    scored = cum.get("SCORED_SPEAKER", 0.0) or EPSILON
    if scored <= EPSILON:
        raise ValueError("Total evaluation time is 0. Abort.")
    missed = cum.get("MISSED_SPEAKER", 0.0)
    falarm = cum.get("FALARM_SPEAKER", 0.0)
    error = cum.get("SPEAKER_ERROR", 0.0)
    DER = (missed + falarm + error) / scored
    CER = error / scored
    FA = falarm / scored
    MISS = missed / scored
    return DER, CER, FA, MISS


def _default_uem_from_ref_sys(
    ref_data: Dict[str, Dict[str, Dict[str, Any]]],
    sys_data: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Auto-derive a UEM that spans the union of reference and system extents.

    NeMo's DER wrappers historically delegated to an external scoring engine
    that, when no UEM was provided, built its evaluation map from the union of
    reference and hypothesis extents. Matching that convention here keeps DER
    numbers reported by NeMo consistent with previously published results
    (any system time extending past the last reference segment is correctly
    counted as false alarm).

    The underlying ``md_eval.evaluate`` function defaults to a stricter
    NIST ``md-eval-22.pl`` behaviour (eval map = reference extent only). This
    helper bridges the two by constructing an explicit single-segment UEM
    per ``(file_id, channel)`` pair that covers
    ``[min(ref ∪ sys TBEG), max(ref ∪ sys TEND)]`` and passing it down so
    ``evaluate`` uses it verbatim.

    Args:
        ref_data: Merged reference RTTM data dict
            (``{file_id: {chnl: {"RTTM": [...], "SPEAKER": {...}, ...}}}``).
        sys_data: Merged system RTTM data dict, same shape as ``ref_data``.

    Returns:
        UEM data dict ``{file_id: {chnl: [{"TBEG": ..., "TEND": ...}]}}``
        with one segment per ``(file_id, chnl)`` pair found in either input.
        Empty when both inputs are empty.
    """
    valid_types = {"SEGMENT", "SPEAKER", "SU", "EDIT", "FILLER", "IP", "CB", "A/P", "LEXEME", "NON-LEX"}
    file_ids = set(ref_data.keys()) | set(sys_data.keys())
    uem_data: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for file_id in file_ids:
        chnls = set(ref_data.get(file_id, {}).keys()) | set(sys_data.get(file_id, {}).keys())
        for chnl in chnls:
            tbeg, tend = float("inf"), float("-inf")
            for src in (ref_data, sys_data):
                rttm = src.get(file_id, {}).get(chnl, {}).get("RTTM", [])
                for tok in rttm:
                    if tok.get("TYPE") in valid_types:
                        tbeg = min(tbeg, tok["TBEG"])
                        tend = max(tend, tok["TEND"])
            if tend > tbeg:
                uem_data.setdefault(file_id, {})[chnl] = [{"TBEG": tbeg, "TEND": tend}]
    return uem_data


def _clamp_uem_to_manifest(
    uem_data: Dict[str, Dict[str, List[Dict[str, float]]]],
    audio_rttm_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Clamp each per-file UEM segment to the manifest-declared audio extent.

    The default auto-UEM (:func:`_default_uem_from_ref_sys`) spans the union of
    reference and system extents. When the hypothesis overshoots the actual
    audio duration (e.g., the last predicted segment ends past the audio's
    final sample), the auto-UEM extends past the audio end and any such
    overshoot is silently counted as false alarm. Historically NeMo's DER
    pipeline clipped scoring to the manifest duration, which is also what
    NIST-style external scorers (e.g., meeteval) do when their UEM is set
    from the manifest. This helper restores that behaviour by clamping each
    UEM segment to ``[offset, offset + duration]`` when those fields are
    available in the manifest.

    Args:
        uem_data: UEM data dict
            ``{file_id: {chnl: [{"TBEG": ..., "TEND": ...}, ...]}}``.
        audio_rttm_map: Manifest map keyed by ``file_id``; per-file entries
            may carry ``offset`` and ``duration`` fields (seconds).

    Returns:
        The same dict, mutated in place. Entries whose manifest does not
        carry both ``offset`` and ``duration`` are left untouched. Segments
        that fall entirely outside ``[offset, offset + duration]`` are
        dropped, and a ``(file_id, chnl)`` whose segments all get dropped
        is removed.
    """
    if not uem_data or not audio_rttm_map:
        return uem_data
    for file_id in list(uem_data.keys()):
        manifest_entry = audio_rttm_map.get(file_id)
        if not manifest_entry:
            continue
        offset = manifest_entry.get('offset', None)
        duration = manifest_entry.get('duration', None)
        if offset is None or duration is None:
            continue
        try:
            audio_tbeg = float(offset)
            audio_tend = audio_tbeg + float(duration)
        except (TypeError, ValueError):
            continue
        if audio_tend <= audio_tbeg:
            continue
        for chnl in list(uem_data[file_id].keys()):
            clamped: List[Dict[str, float]] = []
            for seg in uem_data[file_id][chnl]:
                new_tbeg = max(float(seg["TBEG"]), audio_tbeg)
                new_tend = min(float(seg["TEND"]), audio_tend)
                if new_tend > new_tbeg + EPSILON:
                    clamped.append({"TBEG": new_tbeg, "TEND": new_tend})
            if clamped:
                uem_data[file_id][chnl] = clamped
            else:
                del uem_data[file_id][chnl]
        if not uem_data[file_id]:
            del uem_data[file_id]
    return uem_data


def score_labels(
    AUDIO_RTTM_MAP,
    all_reference: list,
    all_hypothesis: list,
    all_uem: List[List[float]] = None,
    collar: float = 0.25,
    ignore_overlap: bool = True,
    verbose: bool = True,
) -> Optional[Tuple[DiarizationErrorResult, Dict]]:
    """
    Calculate DER, CER, FA and MISS rate from hypotheses and references. Hypothesis and
    reference annotations are lists of :class:`lhotse.SupervisionSegment` (typically
    produced by :func:`labels_to_supervisions` from RTTM label strings).

    Internally uses the md-eval engine (a Python port of NIST md-eval-22.pl)
    for DER computation.

    Args:
        AUDIO_RTTM_MAP (dict): Dictionary containing information provided from manifestpath
        all_reference (list[uniq_name, list[SupervisionSegment]]): reference annotations
            for score calculation.
        all_hypothesis (list[uniq_name, list[SupervisionSegment]]): hypothesis annotations
            for score calculation.
        all_uem (list[list[float]]): List of UEM segments for each audio file. If UEM file is not provided,
                                     it will be read from manifestpath
        collar (float): No-score collar **half-width** in seconds, following NIST
            ``md-eval-22.pl`` semantics. The total no-score zone around every
            reference boundary is ``2 * collar`` seconds. This matches the
            historical NeMo public contract.

            Note on cross-implementation parity: some external annotation
            libraries define their ``collar`` argument as the **total** width
            of the no-score zone (i.e., they use ``collar / 2`` on each side).
            To reproduce a NeMo result with such a library, pass ``2 * collar``
            there. For example, NeMo's ``collar=0.05`` is equivalent to those
            libraries' ``collar=0.10``.
        ignore_overlap (bool): If True, overlapping segments in reference and hypothesis will be ignored
        verbose (bool): If True, warning messages will be printed

    Returns:
        metric (DiarizationErrorResult): Diarization Error Rate metric object.
                                         This object contains detailed scores of each audiofile.
        mapping (dict): Mapping dict containing the mapping speaker label for each audio input
        itemized_errors (tuple): Tuple containing (DER, CER, FA, MISS) for each audio file.
            - DER: Diarization Error Rate, which is sum of all three errors, CER + FA + MISS.
            - CER: Confusion Error Rate, which is sum of all errors
            - FA: False Alarm Rate, which is the number of false alarm segments
            - MISS: Missed Detection Rate, which is the number of missed detection segments
    """
    if len(all_reference) != len(all_hypothesis):
        if verbose:
            logging.warning(
                "Check if each ground truth RTTMs were present in the provided manifest file. "
                "Skipping calculation of Diarization Error Rate"
            )
        return None

    ref_dicts = []
    sys_dicts = []
    uem_dicts = []
    correct_spk_count = 0

    for idx, (reference, hypothesis) in enumerate(zip(all_reference, all_hypothesis)):
        ref_key, ref_labels = reference
        _, hyp_labels = hypothesis

        ref_n_spk = len(unique_speakers(ref_labels))
        hyp_n_spk = len(unique_speakers(hyp_labels))
        if ref_n_spk == hyp_n_spk:
            correct_spk_count += 1
        if verbose and ref_n_spk != hyp_n_spk:
            logging.info(f"Wrong Spk. Count with uniq_id:...{ref_key[-10:]}, " f"Ref: {ref_n_spk}, Hyp: {hyp_n_spk}")

        ref_dicts.append(_annotation_to_rttm_data(ref_key, ref_labels))
        sys_dicts.append(_annotation_to_rttm_data(ref_key, hyp_labels))

        if all_uem is not None:
            uem_obj = all_uem[idx]
            uem_segs = [[seg.start, seg.end] for seg in uem_obj]
            uem_dicts.append(_uem_list_to_uem_data(ref_key, uem_segs))
        elif AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None) is not None:
            uem_file_data = get_uem_data(AUDIO_RTTM_MAP[ref_key]['uem_filepath'])
            if uem_file_data:
                uem_dicts.append(uem_file_data)

    ref_data = _merge_rttm_dicts(ref_dicts)
    sys_data = _merge_rttm_dicts(sys_dicts)
    uem_data = _merge_uem_dicts(uem_dicts) if uem_dicts else None
    if uem_data is None:
        uem_data = _default_uem_from_ref_sys(ref_data, sys_data)
        # When the manifest declares an audio extent, clamp the auto-derived UEM to
        # ``[offset, offset + duration]`` so hypothesis overshoots past the actual
        # audio end don't inflate false alarm. Matches the historical NeMo pipeline
        # and the manifest-based UEM convention used by external scorers (e.g.
        # meeteval).
        uem_data = _clamp_uem_to_manifest(uem_data, AUDIO_RTTM_MAP)

    all_scores, cum = evaluate(
        ref_data,
        sys_data,
        uem_data=uem_data,
        collar=collar,
        opt_1=ignore_overlap,
        verbose=False,
    )

    mapping_dict = _build_mapping_from_data(ref_data, sys_data, uem_data)

    DER, CER, FA, MISS = _extract_errors(cum)
    itemized_errors = (DER, CER, FA, MISS)
    spk_count_acc = correct_spk_count / len(all_reference)

    metric = DiarizationErrorResult(
        all_scores=all_scores,
        cum=cum,
        mapping_dict=mapping_dict,
        collar=collar,
        ignore_overlap=ignore_overlap,
    )

    if verbose:
        logging.info(f"\n{metric.report()}")
    logging.info(
        f"Cumulative Results for collar {collar} sec and ignore_overlap {ignore_overlap}: \n"
        f"| FA: {FA:.4f} | MISS: {MISS:.4f} | CER: {CER:.4f} | DER: {DER:.4f} | "
        f"Spk. Count Acc. {spk_count_acc:.4f}\n"
    )

    return metric, mapping_dict, itemized_errors


def evaluate_der(audio_rttm_map_dict, all_reference, all_hypothesis, diar_eval_mode='all'):
    """
    Evaluate with a selected diarization evaluation scheme

    AUDIO_RTTM_MAP (dict):
        Dictionary containing information provided from manifestpath
    all_reference (list[uniq_name,annotation]):
        reference annotations for score calculation
    all_hypothesis (list[uniq_name,annotation]):
        hypothesis annotations for score calculation
    diar_eval_mode (str):
        Diarization evaluation modes

        diar_eval_mode == "full":
            DIHARD challenge style evaluation, the most strict way of evaluating diarization
            (collar, ignore_overlap) = (0.0, False)
        diar_eval_mode == "fair":
            Evaluation setup used in VoxSRC challenge
            (collar, ignore_overlap) = (0.25, False)
        diar_eval_mode == "forgiving":
            Traditional evaluation setup
            (collar, ignore_overlap) = (0.25, True)
        diar_eval_mode == "all":
            Compute all three modes (default)
    """
    eval_settings = []
    if diar_eval_mode == "full":
        eval_settings = [(0.0, False)]
    elif diar_eval_mode == "fair":
        eval_settings = [(0.25, False)]
    elif diar_eval_mode == "forgiving":
        eval_settings = [(0.25, True)]
    elif diar_eval_mode == "all":
        eval_settings = [(0.0, False), (0.25, False), (0.25, True)]
    else:
        raise ValueError("`diar_eval_mode` variable contains an unsupported value")

    for collar, ignore_overlap in eval_settings:
        diar_score = score_labels(
            AUDIO_RTTM_MAP=audio_rttm_map_dict,
            all_reference=all_reference,
            all_hypothesis=all_hypothesis,
            collar=collar,
            ignore_overlap=ignore_overlap,
        )
    return diar_score


def score_labels_from_rttm_labels(
    ref_labels_list: List[Tuple[str, List[str]]],
    hyp_labels_list: List[Tuple[str, List[str]]],
    uem_segments_list: Optional[List[Tuple[str, List[List[float]]]]] = None,
    collar: float = 0.25,
    ignore_overlap: bool = True,
    verbose: bool = True,
) -> Optional[Tuple[DiarizationErrorResult, Dict[str, SpeakerMap], Tuple[float, float, float, float]]]:
    """Score diarization directly from plain ``"start end speaker"`` label strings.

    Convenience function for callers that have labels as ``"start end speaker"``
    strings rather than pre-built supervision lists.

    Args:
        ref_labels_list: List of ``(uniq_id, [label_strings])`` reference labels.
        hyp_labels_list: List of ``(uniq_id, [label_strings])`` hypothesis labels.
        uem_segments_list: Optional list of ``(uniq_id, [[start, end], ...])`` UEM segments.
        collar: No-score collar **half-width** in seconds, following NIST
            ``md-eval-22.pl`` semantics. The total no-score zone around every
            reference boundary is ``2 * collar`` seconds. To reproduce a NeMo
            result with an external annotation library that defines its
            ``collar`` argument as the **total** width of the no-score zone,
            pass ``2 * collar`` there. For example, NeMo's ``collar=0.05`` is
            equivalent to those libraries' ``collar=0.10``.
        ignore_overlap: If ``True``, restrict scoring to single-speaker regions.
        verbose: If ``True``, log detailed results.

    Returns:
        Same format as :func:`score_labels`, or ``None`` if counts don't match.
    """
    if len(ref_labels_list) != len(hyp_labels_list):
        if verbose:
            logging.warning(
                "Reference and hypothesis label lists must have the same length. "
                "Skipping calculation of Diarization Error Rate"
            )
        return None

    ref_dicts = [_labels_to_rttm_data(uid, labels) for uid, labels in ref_labels_list]
    sys_dicts = [_labels_to_rttm_data(uid, labels) for uid, labels in hyp_labels_list]

    uem_dicts = []
    if uem_segments_list:
        for uid, segs in uem_segments_list:
            uem_dicts.append(_uem_list_to_uem_data(uid, segs))

    ref_data = _merge_rttm_dicts(ref_dicts)
    sys_data = _merge_rttm_dicts(sys_dicts)
    uem_data = _merge_uem_dicts(uem_dicts) if uem_dicts else None
    if uem_data is None:
        uem_data = _default_uem_from_ref_sys(ref_data, sys_data)

    all_scores, cum = evaluate(
        ref_data,
        sys_data,
        uem_data=uem_data,
        collar=collar,
        opt_1=ignore_overlap,
        verbose=False,
    )

    mapping_dict = _build_mapping_from_data(ref_data, sys_data, uem_data)

    DER, CER, FA, MISS = _extract_errors(cum)
    itemized_errors = (DER, CER, FA, MISS)

    correct_spk_count = 0
    for (_, ref_labels), (_, hyp_labels) in zip(ref_labels_list, hyp_labels_list):
        ref_spkrs = {lbl.strip().split()[2] for lbl in ref_labels}
        hyp_spkrs = {lbl.strip().split()[2] for lbl in hyp_labels}
        if len(ref_spkrs) == len(hyp_spkrs):
            correct_spk_count += 1
    spk_count_acc = correct_spk_count / len(ref_labels_list)

    metric = DiarizationErrorResult(
        all_scores=all_scores,
        cum=cum,
        mapping_dict=mapping_dict,
        collar=collar,
        ignore_overlap=ignore_overlap,
    )

    if verbose:
        logging.info(f"\n{metric.report()}")
    logging.info(
        f"Cumulative Results for collar {collar} sec and ignore_overlap {ignore_overlap}: \n"
        f"| FA: {FA:.4f} | MISS: {MISS:.4f} | CER: {CER:.4f} | DER: {DER:.4f} | "
        f"Spk. Count Acc. {spk_count_acc:.4f}\n"
    )

    return metric, mapping_dict, itemized_errors
