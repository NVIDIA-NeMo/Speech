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
"""Concatenated minimum-permutation WER for SOT-tagged multi-speaker transcripts."""
from collections import defaultdict
from typing import NamedTuple, Optional

from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.cpwer import calculate_session_cpWER_detail
from nemo.collections.asr.parts.utils.sot_speaker_alignment import remove_speaker_tags, sot_to_speaker_texts
from nemo.utils import logging


class CpWERSessionResult(NamedTuple):
    """One session's cpWER, with the pieces a per-record dump needs."""

    cpwer: Optional[float]  # None when the reference has no words -- excluded from aggregates
    errors: int
    ref_words: int
    ins: int
    dels: int
    subs: int
    num_ref_speakers: int
    num_hyp_speakers: int
    ref_by_speaker: list
    hyp_in_ref_order: list
    assignment: list
    notag_ceiling: Optional[float]


class CpWER:
    """Score SOT-tagged hypotheses against SOT-tagged references, permutation-invariantly.

    Owns one ordering contract that is easy to get wrong: **split on speaker tags FIRST, then
    normalize each speaker's text**. Every Whisper-style normalizer opens with
    ``re.sub(r"[<\\[][^>\\]]*[>\\]]", "", s)``, so normalizing first deletes every ``<spk:N>`` tag and
    collapses the session to a single speaker -- silently, with a plausible-looking score.

    Empty speaker buckets are kept deliberately. The bucket count must be decided by the tags, not
    by the normalizer: on the multi-speaker debug set, 22/600 references contain a speaker whose
    entire contribution is a filler word that the normalizer deletes. Dropping the emptied bucket
    would remove a reference speaker outright and misalign the permutation search. Empty buckets
    cost nothing -- zero errors and zero denominator on either side.
    """

    def __init__(
        self,
        normalize: bool = True,
        normalizer=None,
        untagged_speaker: Optional[int] = 0,
        max_speakers: Optional[int] = None,
        report_notag_ceiling: bool = True,
        verbose: bool = True,
        placement: str = 'prefix',
    ):
        if normalize:
            self.normalizer = normalizer if normalizer is not None else EnglishTextNormalizer()
        else:
            self.normalizer = _identity
        self.untagged_speaker = untagged_speaker
        self.max_speakers = max_speakers
        self.report_notag_ceiling = report_notag_ceiling
        self.verbose = verbose
        self.reset()
        # 'suffix' when targets close a run with `<spk:N>` instead of opening it.
        self.placement = placement

    def reset(self):
        """Drop all accumulated sessions."""
        self._errors = defaultdict(int)
        self._ref_words = defaultdict(int)
        self._rates = defaultdict(list)
        self._by_num_speakers = defaultdict(lambda: [0, 0])
        self._ceiling = defaultdict(lambda: [0, 0])
        self._skipped_empty_ref = defaultdict(int)
        self._untagged_hyp = defaultdict(int)
        self._sessions = defaultdict(int)
        return self

    def _speaker_lists(self, text: str) -> list:
        """Tagged text -> one normalized string per speaker, ascending by speaker index."""
        grouped = sot_to_speaker_texts(
            text,
            default_speaker=self.untagged_speaker,
            keep_empty=True,
            max_speakers=self.max_speakers,
            placement=self.placement,
        )
        return [self.normalizer(t).strip() for t in grouped.values()]

    def score_session(self, ref_raw: str, hyp_raw: str) -> CpWERSessionResult:
        """Score one session from RAW (still tagged) reference and hypothesis strings."""
        ref_list = self._speaker_lists(ref_raw)
        hyp_list = self._speaker_lists(hyp_raw)
        detail = calculate_session_cpWER_detail(hyp_list, ref_list)

        ceiling = None
        if self.report_notag_ceiling and detail.ref_words:
            # A word-perfect but completely untagged hypothesis. Reference-only and model
            # independent, with the same denominator -- so it is the ceiling a system that
            # attributes nothing would score, and makes the control arm's number interpretable.
            flat = self.normalizer(remove_speaker_tags(ref_raw)).strip()
            ceiling = calculate_session_cpWER_detail([flat], ref_list).cpwer

        return CpWERSessionResult(
            cpwer=detail.cpwer if detail.ref_words else None,
            errors=detail.errors,
            ref_words=detail.ref_words,
            ins=detail.ins,
            dels=detail.dels,
            subs=detail.subs,
            num_ref_speakers=len(ref_list),
            num_hyp_speakers=len(hyp_list),
            ref_by_speaker=ref_list,
            hyp_in_ref_order=detail.hyp_in_ref_order,
            assignment=detail.assignment,
            notag_ceiling=ceiling,
        )

    def update(self, name: str, refs: list, hyps: list) -> None:
        """Accumulate a batch of raw tagged reference/hypothesis pairs under dataset ``name``."""
        for ref, hyp in zip(refs, hyps):
            result = self.score_session(ref, hyp)
            self._sessions[name] += 1
            if result.cpwer is None:
                # An empty reference yields inf; one such session would poison every aggregate.
                self._skipped_empty_ref[name] += 1
                continue
            self._errors[name] += result.errors
            self._ref_words[name] += result.ref_words
            self._rates[name].append(result.cpwer)
            bucket = self._by_num_speakers[(name, result.num_ref_speakers)]
            bucket[0] += result.errors
            bucket[1] += result.ref_words
            if result.num_hyp_speakers <= 1 and result.num_ref_speakers > 1:
                self._untagged_hyp[name] += 1
            if result.notag_ceiling is not None:
                ceiling = self._ceiling[name]
                ceiling[0] += result.notag_ceiling * result.ref_words
                ceiling[1] += result.ref_words
        if self.verbose and refs and hyps:
            logging.info(f"[cpWER REF]\t{refs[0]}\n[cpWER HYP]\t{hyps[0]}")

    def compute(self) -> dict:
        """Corpus cpWER: micro (the headline), macro, and per-reference-speaker-count breakdowns."""
        out = {}
        for name in self._sessions:
            words = self._ref_words[name]
            micro = self._errors[name] / words if words else float("nan")
            out[f"cpwer_{name}"] = micro
            out[f"cpwer_macro_{name}"] = (
                sum(self._rates[name]) / len(self._rates[name]) if self._rates[name] else float("nan")
            )
            out[f"cpwer_sessions_{name}"] = self._sessions[name]
            if self._skipped_empty_ref[name]:
                out[f"cpwer_skipped_empty_ref_{name}"] = self._skipped_empty_ref[name]
            if self._untagged_hyp[name]:
                out[f"cpwer_untagged_hyp_sessions_{name}"] = self._untagged_hyp[name]
            errs, denom = self._ceiling[name]
            if denom:
                out[f"cpwer_notag_ceiling_{name}"] = errs / denom
        for (name, n_spk), (errs, words) in sorted(self._by_num_speakers.items()):
            if words:
                out[f"cpwer_{name}_{n_spk}spk"] = errs / words
        totals = [(self._errors[n], self._ref_words[n]) for n in self._sessions]
        total_words = sum(w for _, w in totals)
        if total_words:
            out["cpwer"] = sum(e for e, _ in totals) / total_words
        self.reset()
        return out


def _identity(x):
    return x
