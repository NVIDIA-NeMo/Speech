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
"""Pin the output-manifest schema written by ``examples/speechlm2/streaming_stt_generate.py``.

The point of the raw fields is that a manifest can be re-scored offline under a *different*
normalizer or parser. ``text`` / ``pred_text`` cannot serve that purpose: every Whisper-style
normalizer opens by deleting ``<...>`` spans, so the speaker tags are gone by the time they are
written, and re-normalizing an already-normalized string is not the same as normalizing the
original. These tests pin the verbatim round-trip, the count fields, the nested ``custom`` block
and the ``_run`` provenance block.

Imports nothing that later commits in this series introduce.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.speechlm2.streaming_stt_generate import (  # noqa: E402
    ReducedRecord,
    StreamingSTTEvalConfig,
    _build_record,
    _build_run_block,
    _cut_meta,
    _wer_counts,
)

# The keys a pre-split manifest already had. C1 adds keys; it renames and drops none --
# `speech_to_text_eval.py` and `analyze_word_latency.py` read these by name.
_PRE_EXISTING_KEYS = {"id", "duration", "text", "pred_text", "wer", "ins", "del", "sub"}

_RUN_KEYS = {
    "build",
    "head_sha",
    "placement",
    "seg_mode",
    "max_segment_duration",
    "seg_method",
    "inference_normalizer",
    "normalizer_language",
    "pretrained_name",
    "inputs",
    "seed",
    "pad_extra_duration",
    "max_new_tokens",
    "system_prompt",
    "oracle_spk_targets",
    "use_state_machine_inference",
    "chunk_size_override",
    "emit_threshold",
}


class _Cut:
    """Stand-in for a lhotse cut carrying only what ``_cut_meta`` reads."""

    def __init__(self, custom):
        self.custom = custom


def _record(ref_raw, hyp_raw, *, meta=None, session=None, normalizer=None):
    rec = ReducedRecord(
        id="cut-000000-000100",
        duration=1.0,
        ref_raw=ref_raw,
        hyp_raw=hyp_raw,
        alignments=None,
        content_scores=None,
        annotated=None,
        meta=meta or {},
    )
    run_block = _build_run_block(StreamingSTTEvalConfig(), seg_mode=False)
    return _build_record(rec, normalizer or (lambda x: x), run_block, session, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "<spk:0> hello there <spk:1> hi",
        "<spk:0> alpha <spk_switch> bravo",
        "<spk:0 alpha bravo",  # unclosed tag -- residue the parser drops but the manifest must keep
        "",
    ],
)
def test_raw_fields_round_trip_verbatim(raw):
    """``text_raw`` / ``pred_text_raw`` are byte-identical to what went in."""
    # A realistic normalizer: deletes bracket spans, exactly what makes `text` unusable for rescoring.
    normalizer = lambda s: re.sub(r"[<\[][^>\]]*[>\]]", "", s).lower().strip()  # noqa: E731
    record = _record(raw, raw, normalizer=normalizer)
    assert record["text_raw"] == raw
    assert record["pred_text_raw"] == raw


@pytest.mark.unit
def test_normalized_fields_are_lossy_so_raw_is_required():
    """The regression this schema exists to prevent: tags do not survive into ``text``."""
    normalizer = lambda s: re.sub(r"[<\[][^>\]]*[>\]]", "", s).lower().strip()  # noqa: E731
    raw = "<spk:0> hello there <spk:1> hi"
    record = _record(raw, raw, normalizer=normalizer)
    assert "<spk:" not in record["text"]
    assert "<spk:" in record["text_raw"]


@pytest.mark.unit
def test_custom_is_nested_whole_and_does_not_clobber_text():
    """`cut.custom` rides under ``custom`` so its ``text`` cannot overwrite the record's own."""
    custom = {
        "sample_id": "s1",
        "audio_filepath": "/a.wav",
        "subset_for_metrics": "ami-ihm",
        "num_speakers": 2,
        "dataset_id": "d1",
        "text": "RAW REFERENCE THAT MUST NOT BECOME record['text']",
        "duration": 99.0,
        "alignments": [{"text": "x", "start_time": 0.0, "end_time": 0.1}],
    }
    record = _record("<spk:0> hello", "<spk:0> hello", meta=_cut_meta(_Cut(custom)))
    # Tags are stripped before the normalizer runs, so `text` is the derived value -- and crucially
    # NOT the `text` that rode in on cut.custom.
    assert record["text"] == "hello"
    assert record["duration"] == 1.0  # the record's padded duration, not custom's 99.0
    # ...but nothing is lost: the whole input row is carried through verbatim.
    assert record["custom"] == custom
    assert record["custom"]["text"] == "RAW REFERENCE THAT MUST NOT BECOME record['text']"
    assert "subset_for_metrics" not in record  # lives only inside `custom`


@pytest.mark.unit
def test_subset_key_lives_only_inside_custom():
    """Per-subset bucketing reads `custom.subset_for_metrics`; nothing is mirrored onto the row."""
    record = _record("<spk:0> hello", "<spk:0> hello", meta=_cut_meta(_Cut({"subset_for_metrics": "ami-ihm"})))
    assert record["custom"]["subset_for_metrics"] == "ami-ihm"
    assert "subset_for_metrics" not in record


@pytest.mark.unit
def test_no_custom_writes_neither_key():
    record = _record("<spk:0> hello", "<spk:0> hello", meta=_cut_meta(_Cut({})))
    assert "custom" not in record


@pytest.mark.unit
@pytest.mark.parametrize("custom", [None, {}])
def test_cut_meta_tolerates_missing_custom(custom):
    """A MonoCut with no custom dict, and a MixedCut whose tracks carry none."""
    assert _cut_meta(_Cut(custom)) == {}


@pytest.mark.unit
def test_cut_meta_copies_custom_as_is():
    """No allowlist: whatever the input carried is carried through, unmodified."""
    custom = {"sample_id": "s1", "anything_at_all": {"nested": [1, 2]}}
    assert _cut_meta(_Cut(custom)) == {"custom": custom}


@pytest.mark.unit
def test_wer_counts_are_internally_consistent():
    counts = _wer_counts("a b c d", "a x c")
    assert counts["wer_errors"] == counts["wer_sub"] + counts["wer_ins"] + counts["wer_del"]
    assert counts["wer_ref_words"] == 4


@pytest.mark.unit
def test_wer_counts_exact_on_empty_reference():
    """No division, no sentinel -- unlike the ``wer`` rate beside them, which is ``inf`` here."""
    counts = _wer_counts("", "a b c")
    assert counts == {"wer_errors": 3, "wer_ref_words": 0, "wer_ins": 3, "wer_del": 0, "wer_sub": 0}


@pytest.mark.unit
def test_empty_reference_keeps_the_infinity_rate_and_finite_counts():
    record = _record("", "a b c")
    assert record["wer"] == float("inf")  # unchanged legacy behaviour
    assert record["wer_errors"] == 3 and record["wer_ref_words"] == 0


@pytest.mark.unit
def test_key_inventory_preserves_pre_existing_keys():
    record = _record("<spk:0> hello", "<spk:0> hello")
    assert _PRE_EXISTING_KEYS <= set(record)
    assert {"text_raw", "pred_text_raw", "_run"} <= set(record)
    assert {"wer_errors", "wer_ref_words", "wer_ins", "wer_del", "wer_sub"} <= set(record)


@pytest.mark.unit
def test_cpwer_count_keys_keep_their_existing_short_names():
    """C1 renames nothing: `cpwer_ins`/`del`/`sub` are what every existing manifest already uses."""

    class _Session:
        cpwer, errors, ref_words, ins, dels, subs = 0.25, 1, 4, 0, 0, 1
        num_ref_speakers = num_hyp_speakers = 1
        ref_by_speaker = ["hello"]
        hyp_in_ref_order = ["hallo"]
        assignment = [0]
        notag_ceiling = None

    record = _record("<spk:0> hello", "<spk:0> hallo", session=_Session())
    assert {"cpwer_ins", "cpwer_del", "cpwer_sub"} <= set(record)
    assert not {"cpwer_substitutions", "cpwer_insertions", "cpwer_deletions"} & set(record)


@pytest.mark.unit
def test_run_block_is_written_on_every_row_including_unscored():
    """``placement`` and ``seg_mode`` are what let an offline scorer refuse invalid work."""
    record = _record("<spk:0> hello", "<spk:0> hello", session=None)
    assert "cpwer" not in record  # cpWER never ran
    assert set(record["_run"]) == _RUN_KEYS


@pytest.mark.unit
def test_run_block_provenance():
    run = _build_run_block(StreamingSTTEvalConfig(), seg_mode=False)
    assert run["build"]  # non-null: nemo.package_info.__version__
    assert run["head_sha"] is None or re.fullmatch(r"[0-9a-f]{7,40}", run["head_sha"])


@pytest.mark.unit
def test_run_block_records_seg_mode_and_placement():
    cfg = StreamingSTTEvalConfig()
    cfg.cpwer_placement = "suffix"
    run = _build_run_block(cfg, seg_mode=True)
    assert run["placement"] == "suffix"
    assert run["seg_mode"] is True


@pytest.mark.unit
def test_run_block_records_a_role_placement_only_when_set():
    """A run that sets no role placement writes the block it always did; a role that is set is recorded."""
    cfg = StreamingSTTEvalConfig()
    cfg.cpwer_placement = "suffix"
    assert set(_build_run_block(cfg, seg_mode=False)) == _RUN_KEYS
    cfg.cpwer_placement_ref = "prefix"
    run = _build_run_block(cfg, seg_mode=False)
    assert set(run) == _RUN_KEYS | {"placement_ref"}
    assert (run["placement"], run["placement_ref"]) == ("suffix", "prefix")
    cfg.cpwer_placement_hyp = "suffix"
    assert _build_run_block(cfg, seg_mode=False)["placement_hyp"] == "suffix"
