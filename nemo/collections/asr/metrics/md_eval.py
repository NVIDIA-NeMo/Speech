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
This script is mostly a Python port of the NIST 'md-eval' (version 22)
Perl script originally found in the dscore repository:
https://github.com/nryant/dscore/blob/master/scorelib/md-eval-22.pl

Original Author: NIST (National Institute of Standards and Technology)
Ported by: [Your Name/Organization]

Bipartite speaker matching uses ``scipy.optimize.linear_sum_assignment``
(Hungarian algorithm, minimisation) in place of the Perl ``weighted_bipartite_graph_match``,
with negated overlap costs so that maximising total overlap becomes a minimisation problem.

Data-flow is a direct translation of the Perl:
    ``get_rttm_data`` → ``evaluate`` → ``score_speaker_diarization``
        → ``create_speaker_segs`` (builds overlap matrix)
        → ``map_speakers``          (``linear_sum_assignment``)
        → ``score_speaker_segments`` (tallies DER components)
    → ``format_sd_scores``
"""
# ==============================================================================
# ORIGINAL BSD-2-CLAUSE LICENSE NOTICE (from dscore)
# ==============================================================================
# Copyright (c) 2017, Neville Ryant
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import re
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from nemo.utils import logging

__all__ = [
    'EPSILON',
    'DEFAULT_COLLAR',
    'DEFAULT_EXTEND',
    'NOEVAL_SD',
    'NOSCORE_SD',
    'RTTM_DATATYPES',
    'get_rttm_data',
    'get_uem_data',
    'uem_from_rttm',
    'add_exclusion_zones_to_uem',
    'add_collars_to_uem',
    'exclude_overlapping_speech_from_uem',
    'create_speaker_segs',
    'map_speakers',
    'score_speaker_diarization',
    'evaluate',
    'DiarizationErrorResult',
]

# ─── Type aliases ──────────────────────────────────────────────────────────
Token = Dict[str, Any]
UEMSegment = Dict[str, float]
SpeakerData = Dict[str, List[Token]]
SpeakerOverlap = Dict[str, Dict[str, float]]
SpeakerMap = Dict[str, str]
ScoreSegment = Dict[str, Any]
SDStats = Dict[str, Any]

# ─── Constants ─────────────────────────────────────────────────────────────

EPSILON: float = 1e-8
MISS_NAME: str = "  MISS"
FA_NAME: str = "  FALSE ALARM"
DEFAULT_COLLAR: float = 0.00
DEFAULT_EXTEND: float = 0.50

NOEVAL_SD: Dict[str, Dict[str, int]] = {
    "NOSCORE": {"<na>": 1},
}
NOSCORE_SD: Dict[str, Dict[str, int]] = {
    "NOSCORE": {"<na>": 1},
    "NON-LEX": {"laugh": 1, "breath": 1, "lipsmack": 1, "cough": 1, "sneeze": 1, "other": 1},
}

RTTM_DATATYPES: Dict[str, Dict[str, int]] = {
    "SEGMENT": {"eval": 1, "<na>": 1},
    "NOSCORE": {"<na>": 1},
    "NO_RT_METADATA": {"<na>": 1},
    "LEXEME": {
        "lex": 1,
        "fp": 1,
        "frag": 1,
        "un-lex": 1,
        "for-lex": 1,
        "alpha": 1,
        "acronym": 1,
        "interjection": 1,
        "propernoun": 1,
        "other": 1,
    },
    "NON-LEX": {"laugh": 1, "breath": 1, "lipsmack": 1, "cough": 1, "sneeze": 1, "other": 1},
    "NON-SPEECH": {"noise": 1, "music": 1, "other": 1},
    "FILLER": {
        "filled_pause": 1,
        "discourse_marker": 1,
        "discourse_response": 1,
        "explicit_editing_term": 1,
        "other": 1,
    },
    "EDIT": {"repetition": 1, "restart": 1, "revision": 1, "simple": 1, "complex": 1, "other": 1},
    "IP": {"edit": 1, "filler": 1, "edit&filler": 1, "other": 1},
    "SU": {
        "statement": 1,
        "backchannel": 1,
        "question": 1,
        "incomplete": 1,
        "unannotated": 1,
        "other": 1,
    },
    "CB": {"coordinating": 1, "clausal": 1, "other": 1},
    "A/P": {"<na>": 1},
    "SPEAKER": {"<na>": 1},
    "SPKR-INFO": {"adult_male": 1, "adult_female": 1, "child": 1, "unknown": 1},
}


# ─── RTTM / UEM parsing ───────────────────────────────────────────────────


def get_rttm_data(data: Dict[str, Dict[str, Dict[str, Any]]], rttm_file: Optional[str]) -> None:
    """Parse one RTTM file into a nested data dictionary (in-place).

    The resulting structure is::

        data[file_id][chnl]['SPEAKER'][spkr]  = [token, ...]
        data[file_id][chnl]['RTTM']           = [token, ...]   (all non-SPKR-INFO)
        data[file_id][chnl]['SPKR-INFO'][spkr]= {'GENDER': str}
        data[file_id][chnl]['LEXEME']         = [token, ...]

    Args:
        data: Mutable dictionary to populate with parsed RTTM data.
        rttm_file: Path to the RTTM file. If ``None``, the function returns immediately.

    Raises:
        ValueError: If a record has fewer than 9 fields or contains a negative duration.
    """
    if rttm_file is None:
        return

    with open(rttm_file, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line[0] in "#;":
                continue
            fields = line.split()
            if fields[0] == "":
                fields = fields[1:]
            if len(fields) < 9:
                raise ValueError(f"Insufficient fields in RTTM file '{rttm_file}'\n  record: {line!r}")

            data_type = fields[0].upper()
            tbeg_str = fields[3].replace("*", "")
            tdur_str = fields[4].replace("*", "")
            token: Token = {
                "TYPE": data_type,
                "FILE": fields[1],
                "CHNL": fields[2].lower(),
                "TBEG": 0.0 if tbeg_str.lower() == "<na>" else float(tbeg_str),
                "TDUR": 0.0 if tdur_str.lower() == "<na>" else float(tdur_str),
                "WORD": fields[5].lower(),
                "SUBT": fields[6].lower(),
                "SPKR": fields[7] if len(fields) > 7 else "<na>",
                "CONF": fields[8].lower() if len(fields) > 8 else "-",
            }
            if token["TDUR"] < 0:
                raise ValueError(f"Negative duration in '{rttm_file}': {line!r}")

            file_id = token["FILE"]
            chnl = token["CHNL"]

            if data_type != "SPKR-INFO":
                token["TEND"] = token["TBEG"] + token["TDUR"]
                token["TMID"] = token["TBEG"] + token["TDUR"] / 2.0

            if data_type == "SPKR-INFO":
                (data.setdefault(file_id, {}).setdefault(chnl, {}).setdefault("SPKR-INFO", {}))[token["SPKR"]] = {
                    "GENDER": token["SUBT"]
                }

            elif data_type == "SPEAKER":
                (
                    data.setdefault(file_id, {})
                    .setdefault(chnl, {})
                    .setdefault("SPEAKER", {})
                    .setdefault(token["SPKR"], [])
                ).append(token)
                data[file_id][chnl].setdefault("RTTM", []).append(token)

            elif data_type == "LEXEME":
                data.setdefault(file_id, {}).setdefault(chnl, {})
                data[file_id][chnl].setdefault("LEXEME", []).append(token)
                data[file_id][chnl].setdefault("RTTM", []).append(token)

            else:
                data.setdefault(file_id, {}).setdefault(chnl, {})
                data[file_id][chnl].setdefault("RTTM", []).append(token)

    # Post-parse: sort SPEAKER segments; stamp gender from SPKR-INFO
    for file_id in data:
        for chnl in data[file_id]:
            spkr_info = data[file_id][chnl].get("SPKR-INFO", {})
            spkr_data = data[file_id][chnl].get("SPEAKER", {})
            for spkr, segs in spkr_data.items():
                gender = spkr_info.get(spkr, {}).get("GENDER") or "unknown"
                spkr_info.setdefault(spkr, {})["GENDER"] = gender
                segs.sort(key=lambda t: t["TMID"])
                for tok in segs:
                    tok["SUBT"] = gender


def get_uem_data(
    uem_file: Optional[str], keep_directory: bool = False
) -> Optional[Dict[str, Dict[str, List[UEMSegment]]]]:
    """Parse a UEM (Un-partitioned Evaluation Map) file.

    Args:
        uem_file: Path to the UEM file. If ``None``, returns ``None``.
        keep_directory: If ``False``, strip directory prefixes from file IDs.

    Returns:
        Nested dictionary ``data[file_id][chnl] = [seg, ...]`` where each segment is a
        dict with keys ``FILE``, ``CHNL``, ``TBEG``, ``TEND``. Returns ``None`` when
        *uem_file* is ``None``.

    Raises:
        ValueError: If a record has fewer than 4 fields.
    """
    if uem_file is None:
        return None

    data: Dict[str, Dict[str, List[UEMSegment]]] = {}
    with open(uem_file, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line[0] in "#;":
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Insufficient UEM fields: {line!r}")
            seg: UEMSegment = {
                "FILE": fields[0],
                "CHNL": fields[1].lower(),
                "TBEG": float(re.sub(r"[^0-9.]", "", fields[2])),
                "TEND": float(re.sub(r"[^0-9.]", "", fields[3])),
            }
            if not keep_directory:
                seg["FILE"] = re.sub(r".*/", "", seg["FILE"])
            seg["FILE"] = re.sub(r"\.[^.]*$", "", seg["FILE"])
            data.setdefault(seg["FILE"], {}).setdefault(seg["CHNL"], []).append(seg)

    for file_id in data:
        for chnl in data[file_id]:
            data[file_id][chnl].sort(key=lambda s: s["TBEG"])
    return data


def uem_from_rttm(rttm_data: List[Token]) -> List[UEMSegment]:
    """Derive a single UEM partition spanning all RTTM tokens.

    Args:
        rttm_data: List of RTTM token dicts, each with keys ``TYPE``, ``TBEG``, ``TEND``.

    Returns:
        A single-element list ``[{"TBEG": ..., "TEND": ...}]`` covering the full time range.
    """
    valid = {"SEGMENT", "SPEAKER", "SU", "EDIT", "FILLER", "IP", "CB", "A/P", "LEXEME", "NON-LEX"}
    tbeg, tend = 1e30, 0.0
    for tok in rttm_data:
        if tok["TYPE"] in valid:
            tbeg = min(tbeg, tok["TBEG"])
            tend = max(tend, tok["TEND"])
    return [{"TBEG": tbeg, "TEND": tend}]


# ─── UEM manipulation helpers ─────────────────────────────────────────────


def _key_end_before_beg(e: Dict[str, Any]) -> Tuple[float, int]:
    """Sort key: order by time; END (0) before BEG (1) at equal times."""
    return (e["TIME"], 0 if e["EVENT"] == "END" else 1)


def _key_beg_before_end(e: Dict[str, Any]) -> Tuple[float, int]:
    """Sort key: order by time; BEG (0) before END (1) at equal times."""
    return (e["TIME"], 0 if e["EVENT"] == "BEG" else 1)


def add_exclusion_zones_to_uem(
    excluded_tokens: Dict[str, Dict[str, int]],
    uem_score: Optional[List[UEMSegment]],
    rttm_data: List[Token],
    max_extend: Optional[float] = None,
) -> List[UEMSegment]:
    """Remove excluded-token regions from the UEM.

    Direct port of the Perl ``add_exclusion_zones_to_uem`` subroutine.  Extends NON-LEX
    no-score zones by *max_extend* seconds toward speech anchors.

    Args:
        excluded_tokens: Mapping ``{type: {subtype: 1}, ...}`` specifying which tokens to exclude.
        uem_score: Current scored UEM segments. May be ``None``.
        rttm_data: List of all RTTM tokens for this file/channel.
        max_extend: Maximum extension (seconds) for NON-LEX no-score zones.

    Returns:
        Updated UEM segment list with exclusion zones removed. Returns *uem_score* unchanged
        when there is nothing to exclude.
    """
    if not excluded_tokens:
        return uem_score

    max_ext = max_extend if (max_extend and max_extend >= EPSILON) else EPSILON
    ns_events: List[Dict[str, Any]] = []

    for tok in rttm_data:
        if tok.get("TDUR", 0) <= 0:
            continue
        ttype = tok["TYPE"]
        subt = tok["SUBT"]

        if ttype == "LEXEME":
            if not excluded_tokens.get("LEXEME", {}).get(subt):
                ns_events.append({"TYPE": "LEX", "EVENT": "BEG", "TIME": tok["TBEG"]})
                ns_events.append({"TYPE": "LEX", "EVENT": "END", "TIME": tok["TEND"]})
        elif ttype == "SPEAKER":
            ns_events.append({"TYPE": "SEG", "EVENT": "BEG", "TIME": tok["TBEG"]})
            ns_events.append({"TYPE": "SEG", "EVENT": "END", "TIME": tok["TEND"]})
        elif excluded_tokens.get(ttype, {}).get(subt):
            ns_events.append({"TYPE": "NSZ", "EVENT": "BEG", "TIME": tok["TBEG"]})
            ns_events.append({"TYPE": "NSZ", "EVENT": "END", "TIME": tok["TEND"]})

    ns_events.sort(key=_key_end_before_beg)

    # Phase 1: build noscore-zone boundary events
    events: List[Dict[str, Any]] = []
    evaluating = 1
    tseg = tend_nsz = tend_lex = 0.0
    lex_cnt = nsz_cnt = 0

    for ev in ns_events:
        etype, eevt, etime = ev["TYPE"], ev["EVENT"], ev["TIME"]

        if etype == "LEX":
            if eevt == "BEG":
                lex_cnt += 1
            else:
                lex_cnt -= 1
                if lex_cnt == 0:
                    tend_lex = etime
        elif etype == "NSZ":
            if eevt == "BEG":
                nsz_cnt += 1
            else:
                nsz_cnt -= 1
                if nsz_cnt == 0:
                    tend_nsz = etime
        elif etype == "SEG":
            tseg = etime

        if evaluating:
            if nsz_cnt == 0 or etype != "NSZ":
                continue
            tstop = etime if lex_cnt > 0 else max(tend_lex, tseg, etime - max_ext)
            events.append({"TYPE": "NSZ", "EVENT": "BEG", "TIME": tstop})
            evaluating = 0
        elif nsz_cnt == 0 and (lex_cnt > 0 or etype == "SEG"):
            tstart = min(tend_nsz + max_ext, etime)
            events.append({"TYPE": "NSZ", "EVENT": "END", "TIME": tstart})
            evaluating = 1
        elif nsz_cnt == 1 and etype == "NSZ" and eevt == "BEG" and etime > tend_nsz + 2 * max_ext:
            events.append({"TYPE": "NSZ", "EVENT": "END", "TIME": tend_nsz + max_ext})
            events.append({"TYPE": "NSZ", "EVENT": "BEG", "TIME": etime - max_ext})
            evaluating = 0

    # Phase 2: merge NSZ events with UEM
    for uem in uem_score or []:
        if uem["TEND"] - uem["TBEG"] > 0:
            events.append({"TYPE": "UEM", "EVENT": "BEG", "TIME": uem["TBEG"]})
            events.append({"TYPE": "UEM", "EVENT": "END", "TIME": uem["TEND"]})

    events.sort(key=_key_end_before_beg)

    evl_cnt = nsz_cnt = evaluating = 0
    tbeg = 0.0
    uem_ex: List[UEMSegment] = []
    for ev in events:
        if ev["TYPE"] == "UEM":
            evl_cnt += 1 if ev["EVENT"] == "BEG" else -1
        elif ev["TYPE"] == "NSZ":
            nsz_cnt += 1 if ev["EVENT"] == "BEG" else -1

        if evaluating and (evl_cnt == 0 or nsz_cnt > 0) and ev["TIME"] > tbeg:
            uem_ex.append({"TBEG": tbeg, "TEND": ev["TIME"]})
            evaluating = 0
        elif evl_cnt > 0 and nsz_cnt == 0:
            tbeg = ev["TIME"]
            evaluating = 1

    return uem_ex if uem_ex else uem_score


def add_collars_to_uem(
    uem_eval: List[UEMSegment],
    ref_spkr_data: SpeakerData,
    collar: float,
) -> List[UEMSegment]:
    """Remove ±collar-second zones around every reference speaker boundary.

    Args:
        uem_eval: Evaluation UEM segments.
        ref_spkr_data: Reference speaker data ``{spkr: [token, ...]}``.
        collar: No-score collar width in seconds.

    Returns:
        New UEM segment list with collar zones removed.
    """
    events: List[Dict[str, Any]] = []
    for uem in uem_eval:
        events.append({"EVENT": "BEG", "TIME": uem["TBEG"]})
        events.append({"EVENT": "END", "TIME": uem["TEND"]})
    for segs in ref_spkr_data.values():
        for seg in segs:
            events.append({"EVENT": "END", "TIME": seg["TBEG"] - collar})
            events.append({"EVENT": "BEG", "TIME": seg["TBEG"] + collar})
            events.append({"EVENT": "END", "TIME": seg["TEND"] - collar})
            events.append({"EVENT": "BEG", "TIME": seg["TEND"] + collar})

    events.sort(key=_key_beg_before_end)

    evaluate = 0
    tbeg = 0.0
    uem_out: List[UEMSegment] = []
    for ev in events:
        if ev["EVENT"] == "BEG":
            evaluate += 1
            if evaluate == 1:
                tbeg = ev["TIME"]
        else:
            evaluate -= 1
            if evaluate == 0 and ev["TIME"] > tbeg:
                uem_out.append({"TBEG": tbeg, "TEND": ev["TIME"]})
    return uem_out


def exclude_overlapping_speech_from_uem(uem_data: List[UEMSegment], rttm_data: List[Token]) -> List[UEMSegment]:
    """Remove regions where two or more reference speakers overlap simultaneously.

    Args:
        uem_data: Current UEM segments to modify.
        rttm_data: List of all RTTM tokens for this file/channel.

    Returns:
        New UEM segment list with overlap regions excluded.
    """
    spkr_evs: List[Dict[str, Any]] = []
    for tok in rttm_data:
        if tok["TYPE"] == "SPEAKER" and tok["TDUR"] > 0:
            spkr_evs.append({"EVENT": "BEG", "TIME": tok["TBEG"]})
            spkr_evs.append({"EVENT": "END", "TIME": tok["TEND"]})

    spkr_evs.sort(key=_key_end_before_beg)

    events: List[Dict[str, Any]] = []
    spkr_cnt = 0
    tbeg_ovlap = 0.0
    for ev in spkr_evs:
        if ev["EVENT"] == "BEG":
            spkr_cnt += 1
            if spkr_cnt == 2:
                tbeg_ovlap = ev["TIME"]
        else:
            spkr_cnt -= 1
            if spkr_cnt == 1:
                events.append({"TYPE": "NSZ", "EVENT": "BEG", "TIME": tbeg_ovlap})
                events.append({"TYPE": "NSZ", "EVENT": "END", "TIME": ev["TIME"]})

    for uem in uem_data:
        if uem["TEND"] - uem["TBEG"] > 0:
            events.append({"TYPE": "UEM", "EVENT": "BEG", "TIME": uem["TBEG"]})
            events.append({"TYPE": "UEM", "EVENT": "END", "TIME": uem["TEND"]})

    events.sort(key=_key_end_before_beg)

    tbeg = 0.0
    evl_cnt = nsz_cnt = evaluating = 0
    uem_ex: List[UEMSegment] = []
    for ev in events:
        if ev["TYPE"] == "UEM":
            evl_cnt += 1 if ev["EVENT"] == "BEG" else -1
        elif ev["TYPE"] == "NSZ":
            nsz_cnt += 1 if ev["EVENT"] == "BEG" else -1

        if evaluating and (evl_cnt == 0 or nsz_cnt > 0) and ev["TIME"] > tbeg:
            uem_ex.append({"TBEG": tbeg, "TEND": ev["TIME"]})
            evaluating = 0
        elif evl_cnt > 0 and nsz_cnt == 0:
            tbeg = ev["TIME"]
            evaluating = 1
    return uem_ex


# ─── Speaker segment timeline ─────────────────────────────────────────────


def create_speaker_segs(
    uem_score: Optional[List[UEMSegment]],
    ref_data: SpeakerData,
    sys_data: SpeakerData,
) -> List[ScoreSegment]:
    """Build a piecewise-constant timeline of ``(ref_spkrs, sys_spkrs)`` sets.

    Ports the Perl ``create_speaker_segs`` exactly:
      - UEM gates which time regions are evaluated
      - Segments are cut at every event boundary
      - At equal times, END events are processed before BEG events
        (with ε-tolerance matching the Perl epsilon comparison)

    Args:
        uem_score: Scored UEM segments. May be ``None``.
        ref_data: Reference speaker data ``{spkr: [token, ...]}``.
        sys_data: System speaker data ``{spkr: [token, ...]}``.

    Returns:
        List of score segments, each a dict with keys ``REF``, ``SYS``, ``TBEG``,
        ``TEND``, ``TDUR``.
    """
    events: List[Dict[str, Any]] = []
    for uem in uem_score or []:
        if uem["TEND"] > uem["TBEG"] + EPSILON:
            events.append({"TYPE": "UEM", "EVENT": "BEG", "TIME": uem["TBEG"]})
            events.append({"TYPE": "UEM", "EVENT": "END", "TIME": uem["TEND"]})

    for spkr, segs in ref_data.items():
        for seg in segs:
            if seg["TDUR"] > 0:
                events.append({"TYPE": "REF", "SPKR": spkr, "EVENT": "BEG", "TIME": seg["TBEG"]})
                events.append({"TYPE": "REF", "SPKR": spkr, "EVENT": "END", "TIME": seg["TEND"]})

    for spkr, segs in sys_data.items():
        for seg in segs:
            if seg["TDUR"] > 0:
                tbeg = seg.get("RTBEG", seg["TBEG"])
                tend = seg.get("RTEND", seg["TEND"])
                events.append({"TYPE": "SYS", "SPKR": spkr, "EVENT": "BEG", "TIME": tbeg})
                events.append({"TYPE": "SYS", "SPKR": spkr, "EVENT": "END", "TIME": tend})

    events.sort(key=_key_end_before_beg)

    evaluate = 0
    tbeg = 0.0
    ref_spkrs: Dict[str, int] = {}
    sys_spkrs: Dict[str, int] = {}
    segments: List[ScoreSegment] = []

    for ev in events:
        if evaluate and tbeg < ev["TIME"] - EPSILON:
            tend = ev["TIME"]
            segments.append(
                {
                    "REF": dict(ref_spkrs),
                    "SYS": dict(sys_spkrs),
                    "TBEG": tbeg,
                    "TEND": tend,
                    "TDUR": tend - tbeg,
                }
            )
            tbeg = tend

        if ev["TYPE"] == "UEM":
            if ev["EVENT"] == "BEG":
                evaluate = 1
                tbeg = ev["TIME"]
            else:
                evaluate = 0
        else:
            spkrs = ref_spkrs if ev["TYPE"] == "REF" else sys_spkrs
            spkr = ev["SPKR"]
            if ev["EVENT"] == "BEG":
                spkrs[spkr] = spkrs.get(spkr, 0) + 1
                if spkrs[spkr] > 1:
                    warnings.warn(f"Speaker {spkr} speaking more than once at t={ev['TIME']}")
            else:
                cnt = spkrs.get(spkr, 0) - 1
                if cnt <= 0:
                    spkrs.pop(spkr, None)
                else:
                    spkrs[spkr] = cnt

    return segments


# ─── Bipartite speaker matching ───────────────────────────────────────────


def map_speakers(spkr_overlap: SpeakerOverlap) -> SpeakerMap:
    """Map reference speakers to system speakers to maximise total overlap time.

    Direct replacement for the Perl ``weighted_bipartite_graph_match``.
    ``scipy.optimize.linear_sum_assignment`` minimises cost; the overlap matrix is
    negated to turn maximisation into minimisation.

    Args:
        spkr_overlap: Mapping ``{ref_spkr: {sys_spkr: seconds_overlap}}``.

    Returns:
        Mapping ``{ref_spkr: sys_spkr}`` containing only genuinely-overlapping pairs.
    """
    if not spkr_overlap:
        return {}

    ref_spkrs = sorted(spkr_overlap.keys())
    sys_spkrs_set: set = set()
    for r in spkr_overlap:
        sys_spkrs_set.update(spkr_overlap[r].keys())
    sys_spkrs = sorted(sys_spkrs_set)

    nref = len(ref_spkrs)
    nsys = len(sys_spkrs)

    cost = np.zeros((nref, nsys))
    for i, ref in enumerate(ref_spkrs):
        for j, sys_ in enumerate(sys_spkrs):
            cost[i, j] = -spkr_overlap[ref].get(sys_, 0.0)

    row_ind, col_ind = linear_sum_assignment(cost)

    result: SpeakerMap = {}
    for i, j in zip(row_ind, col_ind):
        ref = ref_spkrs[i]
        sys_ = sys_spkrs[j]
        if spkr_overlap[ref].get(sys_, 0.0) > 0:
            result[ref] = sys_
    return result


# ─── Per-segment speaker scoring ─────────────────────────────────────────


def _speakers_match(ref_spkrs: Dict[str, int], sys_spkrs: Dict[str, int], spkr_map: SpeakerMap) -> bool:
    """Check whether every ref speaker in a segment maps to a present sys speaker."""
    if len(ref_spkrs) != len(sys_spkrs):
        return False
    for rs in ref_spkrs:
        mapped = spkr_map.get(rs)
        if mapped is None or mapped not in sys_spkrs:
            return False
    return True


def _speaker_mapping_scores(
    spkr_map: SpeakerMap,
    spkr_info: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict]:
    """Count-based (NSPK) speaker-type confusion statistics."""
    stats: Dict[str, Dict] = {"REF": {}, "SYS": {}, "JOINT": {}}
    imap: Dict[str, str] = {}

    for rs, info in spkr_info["REF"].items():
        if not info.get("TIME"):
            continue
        rt = info.get("TYPE", "unknown")
        stats["REF"][rt] = stats["REF"].get(rt, 0) + 1
        ss = spkr_map.get(rs)
        st = spkr_info["SYS"].get(ss, {}).get("TYPE", MISS_NAME) if ss else MISS_NAME
        stats["JOINT"].setdefault(rt, {})[st] = stats["JOINT"].get(rt, {}).get(st, 0) + 1
        if ss:
            imap[ss] = rs

    for ss, info in spkr_info["SYS"].items():
        if not info.get("TIME"):
            continue
        st = info.get("TYPE", "unknown")
        stats["SYS"][st] = stats["SYS"].get(st, 0) + 1
        if ss not in imap:
            stats["JOINT"].setdefault(FA_NAME, {})[st] = stats["JOINT"].get(FA_NAME, {}).get(st, 0) + 1

    return stats


def _score_speaker_segments(
    stats: SDStats,
    score_segs: List[ScoreSegment],
    ref_wds: List[Token],
    spkr_map: SpeakerMap,
    spkr_info: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Accumulate DER components over all scored segments (in-place).

    Args:
        stats: Mutable statistics dictionary to update.
        score_segs: Scored timeline segments.
        ref_wds: Sorted reference word tokens.
        spkr_map: Reference-to-system speaker mapping.
        spkr_info: Speaker metadata dict with REF/SYS sub-dicts.
    """
    ref_wds_list = sorted(ref_wds, key=lambda w: w["TMID"])
    wi = 0

    for seg in score_segs:
        dur = seg["TDUR"]
        nref = len(seg["REF"])
        nsys = len(seg["SYS"])

        stats["SCORED_TIME"] = stats.get("SCORED_TIME", 0.0) + dur
        stats["SCORED_SPEECH"] = stats.get("SCORED_SPEECH", 0.0) + (dur if nref else 0.0)
        stats["MISSED_SPEECH"] = stats.get("MISSED_SPEECH", 0.0) + (dur if nref and not nsys else 0.0)
        stats["FALARM_SPEECH"] = stats.get("FALARM_SPEECH", 0.0) + (dur if nsys and not nref else 0.0)
        stats["SCORED_SPEAKER"] = stats.get("SCORED_SPEAKER", 0.0) + dur * nref
        stats["MISSED_SPEAKER"] = stats.get("MISSED_SPEAKER", 0.0) + dur * max(nref - nsys, 0)
        stats["FALARM_SPEAKER"] = stats.get("FALARM_SPEAKER", 0.0) + dur * max(nsys - nref, 0)

        nmap = sum(1 for rs in seg["REF"] if spkr_map.get(rs) in seg["SYS"])
        stats["SPEAKER_ERROR"] = stats.get("SPEAKER_ERROR", 0.0) + dur * (min(nref, nsys) - nmap)

        while wi < len(ref_wds_list) and ref_wds_list[wi]["TMID"] < seg["TBEG"]:
            wi += 1
        tmp = wi
        sw = mw = ew = 0
        while tmp < len(ref_wds_list) and ref_wds_list[tmp]["TMID"] <= seg["TEND"]:
            if ref_wds_list[tmp].get("SCOREABLE"):
                sw += 1
                if not nsys:
                    mw += 1
                if not _speakers_match(seg["REF"], seg["SYS"], spkr_map):
                    ew += 1
            tmp += 1
        stats["SCORED_WORDS"] = stats.get("SCORED_WORDS", EPSILON) + sw
        stats["MISSED_WORDS"] = stats.get("MISSED_WORDS", EPSILON) + mw
        stats["ERROR_WORDS"] = stats.get("ERROR_WORDS", EPSILON) + ew

        num_ref: Dict[str, int] = {}
        num_sys: Dict[str, int] = {}
        for rs in seg["REF"]:
            rt = spkr_info["REF"].get(rs, {}).get("TYPE", "unknown")
            num_ref[rt] = num_ref.get(rt, 0) + 1
        for ss in seg["SYS"]:
            st = spkr_info["SYS"].get(ss, {}).get("TYPE", "unknown")
            num_sys[st] = num_sys.get(st, 0) + 1

        tt = stats["TYPE"].setdefault("TIME", {"REF": {}, "SYS": {}, "JOINT": {}})
        for rt, nrt in num_ref.items():
            tt["REF"][rt] = tt["REF"].get(rt, 0.0) + nrt * dur
            for st, nst in num_sys.items():
                tt["JOINT"].setdefault(rt, {})[st] = tt["JOINT"].get(rt, {}).get(st, 0.0) + min(nrt, nst) * dur
            tt["JOINT"].setdefault(rt, {})[MISS_NAME] = (
                tt["JOINT"].get(rt, {}).get(MISS_NAME, 0.0) + max(nrt - nsys, 0) * dur
            )
        for st, nst in num_sys.items():
            tt["SYS"][st] = tt["SYS"].get(st, 0.0) + nst * dur
            tt["JOINT"].setdefault(FA_NAME, {})[st] = (
                tt["JOINT"].get(FA_NAME, {}).get(st, 0.0) + max(nst - nref, 0) * dur
            )


# ─── Main diarization scoring ─────────────────────────────────────────────


def score_speaker_diarization(
    file_id: str,
    chnl: str,
    ref_spkr_data: SpeakerData,
    sys_spkr_data: SpeakerData,
    ref_wds: List[Token],
    uem_eval: List[UEMSegment],
    rttm_data: List[Token],
    collar: float = DEFAULT_COLLAR,
    opt_1: bool = False,
    noscore_sd: Optional[Dict[str, Dict[str, int]]] = None,
    max_extend: float = DEFAULT_EXTEND,
) -> Tuple[SDStats, SpeakerMap]:
    """Score speaker diarization for a single file/channel pair.

    Ports the Perl ``score_speaker_diarization`` subroutine exactly.

    Args:
        file_id: File identifier string.
        chnl: Channel identifier string.
        ref_spkr_data: Reference speaker data ``{spkr: [token, ...]}``.
        sys_spkr_data: System speaker data ``{spkr: [token, ...]}``.
        ref_wds: Reference word (LEXEME) tokens.
        uem_eval: Evaluation UEM segments.
        rttm_data: All RTTM tokens for this file/channel.
        collar: No-score collar width in seconds.
        opt_1: If ``True``, restrict scoring to single-speaker regions.
        noscore_sd: No-score conditions for speaker diarization.
        max_extend: Maximum extension for NON-LEX no-score zones.

    Returns:
        A tuple ``(stats, spkr_map)`` where *stats* is a dictionary of DER component
        accumulators and *spkr_map* is the optimal ref→sys speaker mapping.
    """
    stats: SDStats = {
        "EVAL_WORDS": EPSILON,
        "SCORED_WORDS": EPSILON,
        "MISSED_WORDS": EPSILON,
        "ERROR_WORDS": EPSILON,
        "EVAL_TIME": 0.0,
        "EVAL_SPEECH": 0.0,
        "SCORED_TIME": 0.0,
        "SCORED_SPEECH": 0.0,
        "MISSED_SPEECH": 0.0,
        "FALARM_SPEECH": 0.0,
        "SCORED_SPEAKER": 0.0,
        "MISSED_SPEAKER": 0.0,
        "FALARM_SPEAKER": 0.0,
        "SPEAKER_ERROR": 0.0,
        "TYPE": {},
    }

    ref_wds_list = sorted(ref_wds, key=lambda w: w["TMID"])
    wi = 0
    for seg in uem_eval or []:
        stats["EVAL_TIME"] += seg["TEND"] - seg["TBEG"]
        while wi < len(ref_wds_list) and ref_wds_list[wi]["TMID"] < seg["TBEG"]:
            wi += 1
        while wi < len(ref_wds_list) and ref_wds_list[wi]["TMID"] <= seg["TEND"]:
            stats["EVAL_WORDS"] += 1
            wi += 1

    eval_segs = create_speaker_segs(uem_eval, ref_spkr_data, sys_spkr_data)
    spkr_info: Dict[str, Dict[str, Dict[str, Any]]] = {"REF": {}, "SYS": {}}
    spkr_overlap: SpeakerOverlap = {}

    for seg in eval_segs:
        for rs in seg["REF"]:
            spkr_info["REF"].setdefault(rs, {"TIME": 0.0})
            spkr_info["REF"][rs]["TIME"] += seg["TDUR"]
            if ref_spkr_data.get(rs):
                spkr_info["REF"][rs]["TYPE"] = ref_spkr_data[rs][0].get("SUBT", "unknown")
        for ss in seg["SYS"]:
            spkr_info["SYS"].setdefault(ss, {"TIME": 0.0})
            spkr_info["SYS"][ss]["TIME"] += seg["TDUR"]
            if sys_spkr_data.get(ss):
                spkr_info["SYS"][ss]["TYPE"] = sys_spkr_data[ss][0].get("SUBT", "unknown")

        if not seg["REF"]:
            continue
        stats["EVAL_SPEECH"] += seg["TDUR"]
        for rs in seg["REF"]:
            for ss in seg["SYS"]:
                spkr_overlap.setdefault(rs, {})[ss] = spkr_overlap.get(rs, {}).get(ss, 0.0) + seg["TDUR"]

    spkr_map = map_speakers(spkr_overlap)

    uem_score = add_collars_to_uem(uem_eval, ref_spkr_data, collar) if collar > 0 else uem_eval

    if noscore_sd:
        uem_score = add_exclusion_zones_to_uem(noscore_sd, uem_score, rttm_data)
        noscore_nl = noscore_sd.get("NON-LEX")
        if noscore_nl:
            uem_score = add_exclusion_zones_to_uem({"NON-LEX": noscore_nl}, uem_score, rttm_data, max_extend)

    if opt_1:
        uem_score = exclude_overlapping_speech_from_uem(uem_score, rttm_data)

    score_segs = create_speaker_segs(uem_score, ref_spkr_data, sys_spkr_data)
    _score_speaker_segments(stats, score_segs, ref_wds_list, spkr_map, spkr_info)
    stats["TYPE"]["NSPK"] = _speaker_mapping_scores(spkr_map, spkr_info)

    return stats, spkr_map


# ─── Output formatting ────────────────────────────────────────────────────


def _summarize_speaker_type_performance(cls: str, stats: Dict[str, Dict]) -> str:
    """Format speaker-type confusion matrix as a multi-line string.

    Args:
        cls: Either ``"NSPK"`` (count-weighted) or ``"TIME"`` (time-weighted).
        stats: Type confusion statistics dict with keys ``REF``, ``SYS``, ``JOINT``.

    Returns:
        Formatted confusion matrix string.
    """
    sys_types = sorted(stats.get("SYS", {}).keys())
    label = "  REF\\SYS (count)      " if cls == "NSPK" else "  REF\\SYS (seconds)    "

    lines = [label + "".join(f"{st:<20}" for st in sys_types + [MISS_NAME])]
    ref_tot = sum(stats.get("REF", {}).values())

    for rt in sorted(stats.get("REF", {}).keys()) + [FA_NAME]:
        parts = [f"{rt:<16}"]
        for st in sys_types + [MISS_NAME]:
            if rt == FA_NAME and st == MISS_NAME:
                continue
            val = stats.get("JOINT", {}).get(rt, {}).get(st, 0)
            pct = min(999.9, 100.0 * val / ref_tot) if ref_tot else 9e9
            if cls == "NSPK":
                parts.append(f"{int(val):>11} /{pct:>6.1f}%")
            else:
                parts.append(f"{val:>11.2f} /{pct:>6.1f}%")
        lines.append("".join(parts))

    return "\n".join(lines)


def format_sd_scores(condition: str, scores: SDStats) -> str:
    """Format speaker diarization scores as a human-readable string.

    Args:
        condition: Label for the evaluation condition (e.g. ``"ALL"``).
        scores: Aggregated DER component statistics dictionary.

    Returns:
        Multi-line formatted string containing DER breakdown and confusion matrices.
    """
    scored = scores.get("SCORED_SPEAKER", 0.0) or EPSILON
    missed = scores.get("MISSED_SPEAKER", 0.0)
    falarm = scores.get("FALARM_SPEAKER", 0.0)
    error = scores.get("SPEAKER_ERROR", 0.0)
    der = 100.0 * (missed + falarm + error) / scored

    lines = [
        f"\n*** Performance analysis for Speaker Diarization for {condition} ***\n",
        f"SCORED SPEAKER TIME ={scored:f} secs",
        f"MISSED SPEAKER TIME ={missed:f} secs",
        f"FALARM SPEAKER TIME ={falarm:f} secs",
        f"SPEAKER ERROR TIME ={error:f} secs",
        f" OVERALL SPEAKER DIARIZATION ERROR = {der:.2f} percent of scored speaker time" f"  `({condition})",
        "---------------------------------------------",
        " Speaker type confusion matrix -- speaker weighted",
        _summarize_speaker_type_performance("NSPK", scores.get("TYPE", {}).get("NSPK", {})),
        "---------------------------------------------",
        " Speaker type confusion matrix -- time weighted",
        _summarize_speaker_type_performance("TIME", scores.get("TYPE", {}).get("TIME", {})),
        "---------------------------------------------",
    ]
    return "\n".join(lines)


# ─── Top-level evaluate ───────────────────────────────────────────────────


def evaluate(
    ref_data: Dict[str, Dict[str, Dict[str, Any]]],
    sys_data: Dict[str, Dict[str, Dict[str, Any]]],
    uem_data: Optional[Dict[str, Dict[str, List[UEMSegment]]]] = None,
    collar: float = DEFAULT_COLLAR,
    opt_1: bool = False,
    noeval_sd: Optional[Dict[str, Dict[str, int]]] = None,
    noscore_sd: Optional[Dict[str, Dict[str, int]]] = None,
    max_extend: float = DEFAULT_EXTEND,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, SDStats]], SDStats]:
    """Evaluate speaker diarization across all files and channels.

    Ports the Perl ``evaluate`` subroutine (speaker-diarization path only).

    Args:
        ref_data: Parsed reference RTTM data
            ``{file_id: {chnl: {"SPEAKER": ..., "RTTM": ...}}}``.
        sys_data: Parsed system RTTM data (same structure as *ref_data*).
        uem_data: Parsed UEM data ``{file_id: {chnl: [seg, ...]}}``. If ``None``,
            UEM partitions are derived from the reference RTTM.
        collar: No-score collar width in seconds.
        opt_1: If ``True``, restrict scoring to single-speaker regions.
        noeval_sd: No-eval conditions. Defaults to :data:`NOEVAL_SD`.
        noscore_sd: No-score conditions. Defaults to :data:`NOSCORE_SD`.
        max_extend: Maximum extension for NON-LEX no-score zones.
        verbose: If ``True``, log the final DER summary.

    Returns:
        A tuple ``(all_scores, cum)`` where *all_scores* maps
        ``file_id → chnl → stats`` and *cum* is the aggregate statistics dictionary.
    """
    noeval_sd = noeval_sd if noeval_sd is not None else NOEVAL_SD
    noscore_sd = noscore_sd if noscore_sd is not None else NOSCORE_SD

    all_scores: Dict[str, Dict[str, SDStats]] = {}

    for file_id in sorted(ref_data.keys()):
        for chnl in sorted(ref_data[file_id].keys()):
            ref_rttm = ref_data[file_id][chnl].get("RTTM", [])
            ref_spkr_data = ref_data[file_id][chnl].get("SPEAKER")
            if not ref_spkr_data:
                continue

            sys_spkr_data = sys_data.get(file_id, {}).get(chnl, {}).get("SPEAKER", {})
            ref_wds = ref_data[file_id][chnl].get("LEXEME", [])

            uem = uem_data.get(file_id, {}).get(chnl) if uem_data else None
            if uem is None:
                uem = uem_from_rttm(ref_rttm)

            for segs in sys_spkr_data.values():
                for seg in segs:
                    seg.setdefault("RTBEG", seg["TBEG"])
                    seg.setdefault("RTEND", seg["TEND"])
                    seg["RTDUR"] = seg["RTEND"] - seg["RTBEG"]
                    seg["RTMID"] = seg["RTBEG"] + seg["RTDUR"] / 2.0

            uem_sd_eval = add_exclusion_zones_to_uem(noeval_sd, uem, ref_rttm)
            if not uem_sd_eval:
                uem_sd_eval = uem

            stats, _ = score_speaker_diarization(
                file_id,
                chnl,
                ref_spkr_data,
                sys_spkr_data,
                ref_wds,
                uem_sd_eval,
                ref_rttm,
                collar=collar,
                opt_1=opt_1,
                noscore_sd=noscore_sd,
                max_extend=max_extend,
            )
            all_scores.setdefault(file_id, {})[chnl] = stats

    # Aggregate across files/channels
    cum: SDStats = {
        "EVAL_TIME": 0.0,
        "EVAL_SPEECH": 0.0,
        "SCORED_TIME": 0.0,
        "SCORED_SPEECH": 0.0,
        "MISSED_SPEECH": 0.0,
        "FALARM_SPEECH": 0.0,
        "SCORED_SPEAKER": 0.0,
        "MISSED_SPEAKER": 0.0,
        "FALARM_SPEAKER": 0.0,
        "SPEAKER_ERROR": 0.0,
        "EVAL_WORDS": 0.0,
        "SCORED_WORDS": 0.0,
        "MISSED_WORDS": 0.0,
        "ERROR_WORDS": 0.0,
        "TYPE": {
            "NSPK": {"REF": {}, "SYS": {}, "JOINT": {}},
            "TIME": {"REF": {}, "SYS": {}, "JOINT": {}},
        },
    }

    _scalar_keys = (
        "EVAL_TIME",
        "EVAL_SPEECH",
        "SCORED_TIME",
        "SCORED_SPEECH",
        "MISSED_SPEECH",
        "FALARM_SPEECH",
        "SCORED_SPEAKER",
        "MISSED_SPEAKER",
        "FALARM_SPEAKER",
        "SPEAKER_ERROR",
        "EVAL_WORDS",
        "SCORED_WORDS",
        "MISSED_WORDS",
        "ERROR_WORDS",
    )

    for file_id in all_scores:
        for chnl in all_scores[file_id]:
            s = all_scores[file_id][chnl]
            for k in _scalar_keys:
                cum[k] += s.get(k, 0.0)
            for cls in ("NSPK", "TIME"):
                src = s.get("TYPE", {}).get(cls, {})
                dst = cum["TYPE"][cls]
                for kind in ("REF", "SYS"):
                    for t, v in src.get(kind, {}).items():
                        dst[kind][t] = dst[kind].get(t, 0) + v
                for rt, sm in src.get("JOINT", {}).items():
                    for st, v in sm.items():
                        dst["JOINT"].setdefault(rt, {})[st] = dst["JOINT"].get(rt, {}).get(st, 0) + v

    if verbose:
        logging.info(format_sd_scores("ALL", cum))

    return all_scores, cum


# ─── DER result wrapper ────────────────────────────────────────────────────
#
# ``DiarizationErrorResult`` is the result object returned by the public DER
# entry points (``score_labels`` and ``score_labels_from_rttm_labels`` in
# ``der.py``). It exposes a small, dict-like interface that the rest of NeMo
# (and downstream user code) consume.
# ───────────────────────────────────────────────────────────────────────────


class DiarizationErrorResult:
    """Result object returned by NeMo's DER entry points.

    Supports:
      - ``abs(result)``  → overall DER (float)
      - ``result['total']``, ``result['confusion']``, ``result['false alarm']``,
        ``result['missed detection']``
      - ``result.results_`` → list of ``(uniq_id, score_dict)`` per file
      - ``result.optimal_mapping(ref, hyp)`` → speaker mapping dict for a file
      - ``result.report()`` → formatted string summary

    Args:
        all_scores: Per-file stats ``{file_id: {chnl: SDStats}}``.
        cum: Aggregate stats dict.
        mapping_dict: ``{uniq_id: {ref_spkr: sys_spkr}}``.
        collar: Collar value used.
        ignore_overlap: Whether overlap was ignored.
    """

    def __init__(
        self,
        all_scores: Dict[str, Dict[str, SDStats]],
        cum: SDStats,
        mapping_dict: Dict[str, SpeakerMap],
        collar: float,
        ignore_overlap: bool,
    ):
        self._all_scores = all_scores
        self._cum = cum
        self._mapping_dict = mapping_dict
        self._collar = collar
        self._ignore_overlap = ignore_overlap

        scored = cum.get("SCORED_SPEAKER", 0.0) or EPSILON
        self._total = scored
        self._confusion = cum.get("SPEAKER_ERROR", 0.0)
        self._false_alarm = cum.get("FALARM_SPEAKER", 0.0)
        self._missed = cum.get("MISSED_SPEAKER", 0.0)
        self._der = (self._confusion + self._false_alarm + self._missed) / self._total

        self.results_: List[Tuple[str, Dict[str, float]]] = []
        for file_id in sorted(all_scores.keys()):
            for chnl in sorted(all_scores[file_id].keys()):
                s = all_scores[file_id][chnl]
                s_scored = s.get("SCORED_SPEAKER", 0.0) or EPSILON
                self.results_.append(
                    (
                        file_id,
                        {
                            "total": s_scored,
                            "confusion": s.get("SPEAKER_ERROR", 0.0),
                            "false alarm": s.get("FALARM_SPEAKER", 0.0),
                            "missed detection": s.get("MISSED_SPEAKER", 0.0),
                        },
                    )
                )

    def __abs__(self) -> float:
        return self._der

    def __getitem__(self, key: str) -> float:
        return {
            "total": self._total,
            "confusion": self._confusion,
            "false alarm": self._false_alarm,
            "missed detection": self._missed,
        }[key]

    def optimal_mapping(self, ref_labels: Any, hyp_labels: Any) -> SpeakerMap:
        """Return the optimal speaker mapping for a given ref/hyp pair.

        When called with a string key, the mapping is looked up directly. For
        annotation-like objects, the recording id is taken from a ``.uri`` /
        ``.recording_id`` attribute when present, otherwise the object's
        ``str(...)`` representation.
        """
        if isinstance(ref_labels, str):
            key = ref_labels
        else:
            key = getattr(ref_labels, 'uri', None) or getattr(ref_labels, 'recording_id', None) or str(ref_labels)
        return self._mapping_dict.get(key, {})

    def report(self) -> str:
        """Return a human-readable string of per-file DER scores."""
        lines = []
        header = f"{'file':<40} {'total':>10} {'confusion':>10} {'false alarm':>12} {'missed':>10} {'DER':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for file_id, score in self.results_:
            total = score["total"]
            conf = score["confusion"]
            fa = score["false alarm"]
            miss = score["missed detection"]
            der = 100.0 * (conf + fa + miss) / total if total > 0 else 0.0
            lines.append(f"{file_id:<40} {total:>10.2f} {conf:>10.2f} {fa:>12.2f} {miss:>10.2f} {der:>7.2f}%")
        total = self._total
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<40} {total:>10.2f} {self._confusion:>10.2f} "
            f"{self._false_alarm:>12.2f} {self._missed:>10.2f} {abs(self) * 100:>7.2f}%"
        )
        return "\n".join(lines)


def _iter_annotation_segments(annotation: Any) -> Iterator[Tuple[float, float, str]]:
    """Yield ``(start, end, speaker)`` tuples from an annotation-like object.

    Supports lhotse ``SupervisionSet`` / iterable of ``SupervisionSegment``
    (the preferred representation), as well as any iterable of objects
    exposing ``.start`` plus either ``.end`` or ``.duration`` and ``.speaker``.
    Inputs that expose an ``.itertracks(yield_label=True)`` iterator are also
    supported for compatibility with annotation objects from external
    annotation libraries.
    """
    if hasattr(annotation, "itertracks"):
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            yield float(segment.start), float(segment.end), str(speaker)
        return

    for item in annotation:
        start = float(item.start)
        if hasattr(item, "end") and item.end is not None:
            end = float(item.end)
        elif hasattr(item, "duration"):
            end = start + float(item.duration)
        else:
            raise TypeError(f"Annotation item of type {type(item).__name__} has no 'end' or 'duration' attribute.")
        speaker = getattr(item, "speaker", None)
        if speaker is None:
            raise TypeError(f"Annotation item of type {type(item).__name__} has no 'speaker' attribute.")
        yield start, end, str(speaker)


def _annotation_to_rttm_data(
    uniq_id: str,
    annotation: Any,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert an annotation-like object into the nested RTTM data structure
    expected by :func:`evaluate`.

    Accepts any of the following:
      * a lhotse ``SupervisionSet`` or iterable of ``SupervisionSegment``
        (each item exposes ``.start``, ``.end``/``.duration``, ``.speaker``).
      * any iterable of objects that have ``.start``, ``.end`` (or
        ``.duration``) and ``.speaker``.
      * any object exposing ``.itertracks(yield_label=True)`` (compatibility
        with annotation objects from external annotation libraries).

    Args:
        uniq_id: Unique file identifier used as ``file_id``.
        annotation: An annotation-like object as described above.

    Returns:
        RTTM data dict ``{file_id: {chnl: {"SPEAKER": ..., "RTTM": ...}}}``.
    """
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    chnl = "1"

    for tbeg, tend, speaker in _iter_annotation_segments(annotation):
        tdur = tend - tbeg
        if tdur <= 0:
            continue
        token: Token = {
            "TYPE": "SPEAKER",
            "FILE": uniq_id,
            "CHNL": chnl,
            "TBEG": tbeg,
            "TDUR": tdur,
            "TEND": tend,
            "TMID": tbeg + tdur / 2.0,
            "WORD": "<na>",
            "SUBT": "<na>",
            "SPKR": str(speaker),
            "CONF": "-",
        }
        (
            data.setdefault(uniq_id, {}).setdefault(chnl, {}).setdefault("SPEAKER", {}).setdefault(str(speaker), [])
        ).append(token)
        data[uniq_id][chnl].setdefault("RTTM", []).append(token)

    # Sort speaker segments by midpoint (matching get_rttm_data post-parse)
    for file_id in data:
        for ch in data[file_id]:
            for spkr, segs in data[file_id][ch].get("SPEAKER", {}).items():
                segs.sort(key=lambda t: t["TMID"])

    return data


def _labels_to_rttm_data(
    uniq_id: str,
    labels: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert a list of ``"start end speaker"`` label strings into the nested RTTM
    data structure expected by :func:`evaluate`.

    Args:
        uniq_id: Unique file identifier.
        labels: List of label strings, each formatted as ``"start end speaker"``.

    Returns:
        RTTM data dict ``{file_id: {chnl: {"SPEAKER": ..., "RTTM": ...}}}``.
    """
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    chnl = "1"

    for label in labels:
        parts = label.strip().split()
        tbeg, tend = float(parts[0]), float(parts[1])
        speaker = parts[2]
        tdur = tend - tbeg
        if tdur <= 0:
            continue
        token: Token = {
            "TYPE": "SPEAKER",
            "FILE": uniq_id,
            "CHNL": chnl,
            "TBEG": tbeg,
            "TDUR": tdur,
            "TEND": tend,
            "TMID": tbeg + tdur / 2.0,
            "WORD": "<na>",
            "SUBT": "<na>",
            "SPKR": speaker,
            "CONF": "-",
        }
        (data.setdefault(uniq_id, {}).setdefault(chnl, {}).setdefault("SPEAKER", {}).setdefault(speaker, [])).append(
            token
        )
        data[uniq_id][chnl].setdefault("RTTM", []).append(token)

    for file_id in data:
        for ch in data[file_id]:
            for spkr, segs in data[file_id][ch].get("SPEAKER", {}).items():
                segs.sort(key=lambda t: t["TMID"])

    return data


def _uem_list_to_uem_data(
    uniq_id: str,
    uem_segments: List[List[float]],
) -> Dict[str, Dict[str, List[UEMSegment]]]:
    """Convert a list of ``[start, end]`` pairs into UEM data structure.

    Args:
        uniq_id: Unique file identifier.
        uem_segments: List of ``[start_time, end_time]`` pairs.

    Returns:
        UEM data dict ``{file_id: {chnl: [seg, ...]}}``.
    """
    chnl = "1"
    segs = [{"TBEG": float(s), "TEND": float(e)} for s, e in uem_segments]
    return {uniq_id: {chnl: segs}}


def _merge_rttm_dicts(dicts: List[Dict[str, Dict[str, Dict[str, Any]]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Merge multiple single-file RTTM data dicts into one combined dict."""
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for d in dicts:
        for file_id, channels in d.items():
            merged.setdefault(file_id, {}).update(channels)
    return merged


def _merge_uem_dicts(dicts: List[Dict[str, Dict[str, List[UEMSegment]]]]) -> Dict[str, Dict[str, List[UEMSegment]]]:
    """Merge multiple single-file UEM data dicts into one combined dict."""
    merged: Dict[str, Dict[str, List[UEMSegment]]] = {}
    for d in dicts:
        for file_id, channels in d.items():
            merged.setdefault(file_id, {}).update(channels)
    return merged
