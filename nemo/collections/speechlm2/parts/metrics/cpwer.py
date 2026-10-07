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
from collections import OrderedDict, defaultdict
from typing import NamedTuple, Optional

from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.cpwer import calculate_session_cpWER_detail
from nemo.collections.asr.parts.utils.sot_speaker_alignment import remove_speaker_tags, sot_to_speaker_texts
from nemo.collections.asr.parts.utils.text_normalizers import build_normalizer
from nemo.utils import logging

# Distinguishes "caller did not pass this" from "caller passed None", which is a meaningful value
# for the untagged-speaker axes: None means DISCARD the untagged run.
_UNSET = object()


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
    # True when the row was not scored at all (the reference parsed to zero streams, or the
    # normalizer failed). Distinct from `cpwer is None`, which also covers a reference that parsed
    # into streams but normalized to zero words -- that row IS scored and its errors still count.
    abstained: bool = False
    # The reference stream keys, parallel to `ref_by_speaker`. Without them a dumped stream list is
    # positional only, and positions are not comparable across roles once the two are parsed with
    # different settings.
    ref_speakers: list = ()


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
        *,
        normalizer_name: Optional[str] = None,
        tag_syntax_ref: str = 'spk',
        tag_syntax_hyp: str = 'spk',
        tag_case_sensitive: bool = True,
        untagged_speaker_ref: Optional[int] = _UNSET,
        untagged_speaker_hyp: Optional[int] = _UNSET,
        keep_empty_streams: bool = True,
        drop_tag_residue: bool = True,
        speaker_order: str = 'index',
        ceiling_source: str = 'strip_tags',
        placement_ref: Optional[str] = None,
        placement_hyp: Optional[str] = None,
    ):
        if normalizer is not None and normalizer_name is not None:
            raise ValueError("Pass either normalizer= (a callable) or normalizer_name= (a family name), not both.")
        if normalizer_name is not None:
            self.normalizer = build_normalizer(normalizer_name, 'en')
        elif normalize:
            self.normalizer = normalizer if normalizer is not None else EnglishTextNormalizer()
        else:
            self.normalizer = _identity
        _validate('placement', placement, ('prefix', 'suffix'))
        for name, value in (('placement_ref', placement_ref), ('placement_hyp', placement_hyp)):
            if value is not None:
                _validate(name, value, ('prefix', 'suffix'))
        _validate('speaker_order', speaker_order, ('index', 'first_seen'))
        _validate('ceiling_source', ceiling_source, ('strip_tags', 'streams'))
        self.untagged_speaker = untagged_speaker
        # Per-role, because a reference and a hypothesis are not symmetric: some scorers discard a
        # reference run that precedes the first tag while keeping the hypothesis equivalent. Each
        # falls back to the single `untagged_speaker` when not given, so old call sites are unchanged.
        self.untagged_speaker_ref = untagged_speaker if untagged_speaker_ref is _UNSET else untagged_speaker_ref
        self.untagged_speaker_hyp = untagged_speaker if untagged_speaker_hyp is _UNSET else untagged_speaker_hyp
        self.tag_syntax_ref = tag_syntax_ref
        self.tag_syntax_hyp = tag_syntax_hyp
        self.tag_case_sensitive = tag_case_sensitive
        self.keep_empty_streams = keep_empty_streams
        self.drop_tag_residue = drop_tag_residue
        self.speaker_order = speaker_order
        self.ceiling_source = ceiling_source
        self.max_speakers = max_speakers
        self.report_notag_ceiling = report_notag_ceiling
        self.verbose = verbose
        self.reset()
        # 'suffix' when targets close a run with `<spk:N>` instead of opening it.
        self.placement = placement
        # Per role, because a suffix-placement model is scored against references written in prefix form. None
        # follows `placement`, so old call sites are unchanged.
        self.placement_ref = placement_ref
        self.placement_hyp = placement_hyp

    @classmethod
    def from_config(cls, cfg, normalizer=None) -> "CpWER":
        """Build from a ``CpWERScoringConfig``, so every entry point configures this the same way.

        Naming the axes at each call site instead would let an inference script and an offline
        scorer drift apart silently -- which is the one failure this whole surface exists to avoid.

        Args:
            cfg: any object carrying the ``CpWERScoringConfig`` fields.
            normalizer: pre-built callable; when omitted it is built from the config's resolved
                normalizer family.

        Returns:
            CpWER: configured, with counters reset.
        """
        from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import resolve_placements

        if normalizer is None:
            from nemo.collections.asr.parts.utils.text_normalizers import build_normalizer
            from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import resolve_normalizer

            normalizer = build_normalizer(resolve_normalizer(cfg), cfg.normalizer_language)
        placement_ref, placement_hyp = resolve_placements(cfg)
        return cls(
            normalize=True,
            normalizer=normalizer,
            untagged_speaker=cfg.cpwer_untagged_speaker_ref,
            max_speakers=cfg.cpwer_max_speakers,
            report_notag_ceiling=cfg.cpwer_report_notag_ceiling,
            verbose=False,
            placement=cfg.cpwer_placement,
            tag_syntax_ref=cfg.cpwer_tag_syntax_ref,
            tag_syntax_hyp=cfg.cpwer_tag_syntax_hyp,
            tag_case_sensitive=cfg.cpwer_tag_case_sensitive,
            untagged_speaker_ref=cfg.cpwer_untagged_speaker_ref,
            untagged_speaker_hyp=cfg.cpwer_untagged_speaker_hyp,
            keep_empty_streams=cfg.cpwer_keep_empty_streams,
            drop_tag_residue=cfg.cpwer_drop_tag_residue,
            speaker_order=cfg.cpwer_speaker_order,
            ceiling_source=cfg.cpwer_ceiling_source,
            placement_ref=placement_ref,
            placement_hyp=placement_hyp,
        )

    def reset(self):
        """Drop all accumulated sessions."""
        self._errors = defaultdict(int)
        self._ref_words = defaultdict(int)
        self._rates = defaultdict(list)
        self._by_num_speakers = defaultdict(lambda: [0, 0])
        self._ceiling = defaultdict(lambda: [0, 0])
        self._ins = defaultdict(int)
        self._dels = defaultdict(int)
        self._subs = defaultdict(int)
        self._abstained = defaultdict(int)
        self._zero_ref_words = defaultdict(int)
        self._admitted = defaultdict(int)
        self._empty_hyp = defaultdict(int)
        self._untagged_hyp = defaultdict(int)
        self._sessions = defaultdict(int)
        return self

    def _speaker_streams(self, text: str, role: str) -> "OrderedDict":
        """Tagged text -> ``{speaker index: normalized text}`` for one role.

        Returns the keys, not just the values, because a caller that dumps streams for inspection
        needs to know WHICH speaker each string belongs to -- and because the two roles can be
        parsed with different settings, so positions are not comparable across them.

        Args:
            text: raw, still-tagged text.
            role: ``'ref'`` or ``'hyp'``; selects the per-role axes.

        Returns:
            OrderedDict[int, str]: speaker index -> normalized text, in `speaker_order`.
        """
        grouped = sot_to_speaker_texts(
            text,
            default_speaker=self.untagged_speaker_ref if role == 'ref' else self.untagged_speaker_hyp,
            keep_empty=self.keep_empty_streams,
            max_speakers=self.max_speakers,
            placement=self._placement(role),
            tag_syntax=self.tag_syntax_ref if role == 'ref' else self.tag_syntax_hyp,
            case_sensitive=self.tag_case_sensitive,
            drop_tag_residue=self.drop_tag_residue,
            speaker_order=self.speaker_order,
        )
        return OrderedDict((i, self.normalizer(t).strip()) for i, t in grouped.items())

    def _placement(self, role: str) -> str:
        """The tag placement ``role`` (``'ref'`` or ``'hyp'``) is parsed with: its own when set, else `placement`."""
        own = self.placement_ref if role == 'ref' else self.placement_hyp
        return self.placement if own is None else own

    def score_session(self, ref_raw: str, hyp_raw: str) -> CpWERSessionResult:
        """Score one session from RAW (still tagged) reference and hypothesis strings."""
        ref_streams = self._speaker_streams(ref_raw, 'ref')
        hyp_streams = self._speaker_streams(hyp_raw, 'hyp')
        ref_list = list(ref_streams.values())
        hyp_list = list(hyp_streams.values())
        if not ref_streams:
            # The reference parsed to zero streams -- nothing to score against. Abstain rather than
            # invent a single empty speaker, which would turn every hypothesis word into an
            # insertion against a zero denominator.
            return CpWERSessionResult(
                cpwer=None,
                errors=0,
                ref_words=0,
                ins=0,
                dels=0,
                subs=0,
                num_ref_speakers=0,
                num_hyp_speakers=len(hyp_list),
                ref_by_speaker=[],
                hyp_in_ref_order=[],
                assignment=[],
                notag_ceiling=None,
                abstained=True,
            )
        detail = calculate_session_cpWER_detail(hyp_list, ref_list)

        ceiling = None
        if self.report_notag_ceiling and detail.ref_words:
            # A word-perfect but completely untagged hypothesis. Reference-only and model
            # independent, with the same denominator -- so it is the ceiling a system that
            # attributes nothing would score, and makes the control arm's number interpretable.
            if self.ceiling_source == 'streams':
                # `remove_speaker_tags` only knows `<spk:N>`, so under any other tag syntax the tag
                # text would leak into the pseudo-hypothesis and be scored as words. Joining the
                # parser's own RAW streams and normalizing once avoids that. Join raw, normalize
                # once -- normalizing each stream and joining is not the same string.
                raw = sot_to_speaker_texts(
                    ref_raw,
                    default_speaker=self.untagged_speaker_ref,
                    keep_empty=False,
                    max_speakers=self.max_speakers,
                    placement=self._placement('ref'),
                    tag_syntax=self.tag_syntax_ref,
                    case_sensitive=self.tag_case_sensitive,
                    drop_tag_residue=self.drop_tag_residue,
                    speaker_order=self.speaker_order,
                )
                flat = self.normalizer(" ".join(raw.values())).strip()
            else:
                flat = self.normalizer(remove_speaker_tags(ref_raw)).strip()
            ceiling = calculate_session_cpWER_detail([flat], ref_list).cpwer

        return CpWERSessionResult(
            # The two gates, named separately because they are different situations that both
            # leave no rate to compute:
            #   GATE 1  ref_list empty      -> the reference parsed to zero streams. Not scored at
            #                                  all; handled by the early return above.
            #   GATE 2  ref_words == 0      -> it parsed into streams that normalize to nothing (a
            #                                  reference of pure filler). Scored as 0.0: the errors
            #                                  still pool into the micro numerator against a zero
            #                                  denominator contribution, so micro can exceed 100%,
            #                                  and a literal 0.0 joins the macro list. Both follow
            #                                  the reference scorer; agreement is the point.
            cpwer=detail.cpwer if detail.ref_words else 0.0,
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
            ref_speakers=list(ref_streams.keys()),
        )

    def update(self, name: str, refs: list, hyps: list) -> None:
        """Accumulate a batch of raw tagged reference/hypothesis pairs under dataset ``name``.

        Args:
            name: dataset label the counts accumulate under.
            refs: raw, still-tagged reference strings.
            hyps: raw, still-tagged hypothesis strings, parallel to ``refs``.

        Raises:
            ValueError: if the two lists differ in length. ``zip`` used to truncate silently, which
                scores a prefix of the corpus and reports it as the whole thing.
        """
        if len(refs) != len(hyps):
            raise ValueError(f"refs and hyps must be the same length, got {len(refs)} and {len(hyps)}")
        for ref, hyp in zip(refs, hyps):
            result = self.score_session(ref, hyp)
            self._sessions[name] += 1
            if result.abstained:
                # Not scored at all: the reference parsed to zero streams, so there is nothing to
                # align against. Excluded from micro AND macro, and counted so it cannot hide.
                self._abstained[name] += 1
                continue
            if not result.ref_words:
                # GATE 2: parsed into streams but normalized to zero words. The errors are real and
                # still pool into the micro numerator; the denominator contribution is zero, so this
                # can push micro above 100%. Counted separately from an abstain.
                self._zero_ref_words[name] += 1
            self._admitted[name] += 1
            if not any(result.hyp_in_ref_order):
                self._empty_hyp[name] += 1
            self._errors[name] += result.errors
            self._ref_words[name] += result.ref_words
            self._ins[name] += result.ins
            self._dels[name] += result.dels
            self._subs[name] += result.subs
            # A gate-2 row contributes a literal 0.0 here, which is what dilutes the macro.
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
            # Integer totals, so a micro can be re-pooled across runs without re-scoring.
            out[f"cpwer_errors_{name}"] = self._errors[name]
            out[f"cpwer_ref_words_{name}"] = self._ref_words[name]
            out[f"cpwer_insertions_{name}"] = self._ins[name]
            out[f"cpwer_deletions_{name}"] = self._dels[name]
            out[f"cpwer_substitutions_{name}"] = self._subs[name]
            # Counters are emitted even at zero. A counter that appears only when non-zero cannot be
            # distinguished from one that was never computed, which is exactly when it matters.
            out[f"cpwer_sessions_{name}"] = self._sessions[name]
            out[f"cpwer_admitted_{name}"] = self._admitted[name]
            out[f"cpwer_abstained_{name}"] = self._abstained[name]
            out[f"cpwer_zero_ref_words_{name}"] = self._zero_ref_words[name]
            out[f"cpwer_empty_hyp_{name}"] = self._empty_hyp[name]
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


def _validate(name: str, value, accepted) -> None:
    """Reject an unrecognised axis value instead of silently taking a default branch.

    ``placement='Prefix'`` used to fall through to the prefix branch, so a typo produced a
    plausible number from the wrong setting.
    """
    if value not in accepted:
        raise ValueError(f"Unknown {name}={value!r}. Accepted: {', '.join(map(repr, accepted))}")
