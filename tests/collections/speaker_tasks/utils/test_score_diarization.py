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

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.speaker_tasks import score_diarization
from scripts.speaker_tasks.score_diarization import align_recording_ids, read_rttm_inputs, read_rttm_manifest


def write_rttm(path, recording_id, speaker="speaker_0"):
    path.write_text(
        f"SPEAKER {recording_id} 1 0.000 1.000 <NA> <NA> {speaker} <NA> <NA>\n",
        encoding="utf-8",
    )


class TestScoreDiarizationInputs:
    @pytest.mark.unit
    @pytest.mark.parametrize("recording_ids", [("session_a", "session_b")])
    def test_compound_rttm_files(self, tmp_path, recording_ids):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        reference_path.write_text(
            "".join(
                f"SPEAKER {recording_id} 1 0.000 1.000 <NA> <NA> ref <NA> <NA>\n" for recording_id in recording_ids
            ),
            encoding="utf-8",
        )
        hypothesis_path.write_text(
            "".join(
                f"SPEAKER {recording_id} 1 0.000 1.000 <NA> <NA> hyp <NA> <NA>\n" for recording_id in recording_ids
            ),
            encoding="utf-8",
        )

        reference, hypothesis = read_rttm_inputs(reference_path, hypothesis_path)

        assert set(reference) == set(recording_ids)
        assert set(hypothesis) == set(recording_ids)

    @pytest.mark.unit
    @pytest.mark.parametrize("recordings", [(("a.rttm", "session_a"), ("b.rttm", "session_b"))])
    def test_matching_per_file_rttm_directories(self, tmp_path, recordings):
        reference_dir = tmp_path / "reference"
        hypothesis_dir = tmp_path / "hypothesis"
        reference_dir.mkdir()
        hypothesis_dir.mkdir()
        for filename, recording_id in recordings:
            write_rttm(reference_dir / filename, recording_id, "ref")
            write_rttm(hypothesis_dir / filename, recording_id, "hyp")

        reference, hypothesis = read_rttm_inputs(reference_dir, hypothesis_dir)

        expected_recording_ids = {recording_id for _, recording_id in recordings}
        assert set(reference) == expected_recording_ids
        assert set(hypothesis) == expected_recording_ids

    @pytest.mark.unit
    @pytest.mark.parametrize("reference_is_directory", [False, True])
    def test_mixed_compound_file_and_directory_inputs(self, tmp_path, reference_is_directory):
        compound_path = tmp_path / "compound.rttm"
        per_file_dir = tmp_path / "per_file"
        per_file_dir.mkdir()
        compound_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> speaker_a <NA> <NA>\n"
            "SPEAKER session_b 1 0.000 1.000 <NA> <NA> speaker_b <NA> <NA>\n",
            encoding="utf-8",
        )
        write_rttm(per_file_dir / "a.rttm", "session_a")
        write_rttm(per_file_dir / "b.rttm", "session_b")

        reference_path, hypothesis_path = (
            (per_file_dir, compound_path) if reference_is_directory else (compound_path, per_file_dir)
        )
        reference, hypothesis = read_rttm_inputs(reference_path, hypothesis_path)

        assert set(reference) == {"session_a", "session_b"}
        assert set(hypothesis) == {"session_a", "session_b"}

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "reference_filename,hypothesis_filename,error_match",
        [("a.rttm", "b.rttm", r"missing in hypothesis: a\.rttm; not in reference: b\.rttm")],
    )
    def test_directory_filenames_must_match_exactly(
        self, tmp_path, reference_filename, hypothesis_filename, error_match
    ):
        reference_dir = tmp_path / "reference"
        hypothesis_dir = tmp_path / "hypothesis"
        reference_dir.mkdir()
        hypothesis_dir.mkdir()
        write_rttm(reference_dir / reference_filename, "session_a")
        write_rttm(hypothesis_dir / hypothesis_filename, "session_a")

        with pytest.raises(ValueError, match=error_match):
            read_rttm_inputs(reference_dir, hypothesis_dir)

    @pytest.mark.unit
    @pytest.mark.parametrize("recording_id", ["session_a"])
    def test_empty_per_file_hypothesis_rttm_means_no_speech(self, tmp_path, recording_id):
        reference_dir = tmp_path / "reference"
        hypothesis_dir = tmp_path / "hypothesis"
        reference_dir.mkdir()
        hypothesis_dir.mkdir()
        write_rttm(reference_dir / f"{recording_id}.rttm", recording_id)
        (hypothesis_dir / f"{recording_id}.rttm").write_text("", encoding="utf-8")

        reference, hypothesis = read_rttm_inputs(reference_dir, hypothesis_dir)

        assert set(reference) == {recording_id}
        assert hypothesis == {recording_id: []}

    @pytest.mark.unit
    @pytest.mark.parametrize("recording_ids", [("session_a", "session_b")])
    def test_empty_compound_hypothesis_uses_reference_recording_ids(self, tmp_path, recording_ids):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        reference_path.write_text(
            "".join(
                f"SPEAKER {recording_id} 1 0.000 1.000 <NA> <NA> speaker <NA> <NA>\n" for recording_id in recording_ids
            ),
            encoding="utf-8",
        )
        hypothesis_path.write_text("", encoding="utf-8")

        reference, hypothesis = read_rttm_inputs(reference_path, hypothesis_path)

        assert set(reference) == set(recording_ids)
        assert hypothesis == {recording_id: [] for recording_id in recording_ids}

    @pytest.mark.unit
    @pytest.mark.parametrize("missing_recording_id", ["session_b"])
    def test_missing_hypothesis_recording_is_scored_as_empty(self, missing_recording_id):
        reference = {"session_a": ["0.0 1.0 ref_a"], "session_b": ["0.0 1.0 ref_b"]}
        hypothesis = {"session_a": ["0.0 1.0 hyp_a"]}

        aligned_hypothesis, message = align_recording_ids(reference, hypothesis)

        assert aligned_hypothesis == {"session_a": ["0.0 1.0 hyp_a"], missing_recording_id: []}
        assert missing_recording_id in message

    @pytest.mark.unit
    @pytest.mark.parametrize("extra_recording_id", ["session_b"])
    def test_extra_hypothesis_recording_is_rejected(self, extra_recording_id):
        reference = {"session_a": ["0.0 1.0 ref_a"]}
        hypothesis = {"session_a": ["0.0 1.0 hyp_a"], extra_recording_id: ["0.0 1.0 hyp_b"]}

        with pytest.raises(ValueError, match=f"not in reference: {extra_recording_id}"):
            align_recording_ids(reference, hypothesis)

    @pytest.mark.unit
    @pytest.mark.parametrize("missing_recording_id", ["session_b"])
    def test_main_scores_partially_missing_hypothesis_as_complete_miss(self, tmp_path, missing_recording_id):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        reference_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> ref_a <NA> <NA>\n"
            f"SPEAKER {missing_recording_id} 1 0.000 1.000 <NA> <NA> ref_b <NA> <NA>\n",
            encoding="utf-8",
        )
        write_rttm(hypothesis_path, "session_a", "hyp_a")
        args = SimpleNamespace(reference=reference_path, hypothesis=hypothesis_path, collar=0.0)

        with (
            patch.object(score_diarization, "parse_args", return_value=args),
            patch("nemo.collections.asr.metrics.der.logging.info") as log_info,
        ):
            score_diarization.main()

        log_text = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert missing_recording_id in log_text
        assert "MISS: 0.5000" in log_text

    @pytest.mark.unit
    @pytest.mark.parametrize("recording_id,offset,duration", [("session_a", 2.0, 5.0)])
    def test_reference_manifest_resolves_rttm_and_scoring_region(self, tmp_path, recording_id, offset, duration):
        reference_path = tmp_path / "reference.rttm"
        manifest_path = tmp_path / "manifest.json"
        write_rttm(reference_path, "internal_rttm_id", "ref")
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": f"audio/{recording_id}.wav",
                    "rttm_filepath": reference_path.name,
                    "offset": offset,
                    "duration": duration,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        reference, audio_rttm_map = read_rttm_manifest(manifest_path)

        assert set(reference) == {recording_id}
        assert audio_rttm_map[recording_id]["rttm_filepath"] == str(reference_path)
        assert audio_rttm_map[recording_id]["offset"] == offset
        assert audio_rttm_map[recording_id]["duration"] == duration

    @pytest.mark.unit
    @pytest.mark.parametrize("recording_id", ["session_a"])
    def test_main_logs_full_per_file_der_report_with_manifest_reference(self, tmp_path, recording_id):
        reference_path = tmp_path / "reference.rttm"
        manifest_path = tmp_path / "manifest.json"
        hypothesis_dir = tmp_path / "hypothesis"
        hypothesis_dir.mkdir()
        write_rttm(reference_path, recording_id, "ref")
        write_rttm(hypothesis_dir / f"{recording_id}.rttm", recording_id, "hyp")
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": f"{recording_id}.wav",
                    "rttm_filepath": str(reference_path),
                    "duration": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(reference=manifest_path, hypothesis=hypothesis_dir, collar=0.0)

        with (
            patch.object(score_diarization, "parse_args", return_value=args),
            patch("nemo.collections.asr.metrics.der.logging.info") as log_info,
        ):
            score_diarization.main()

        log_text = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert recording_id in log_text
        assert "false alarm" in log_text
        assert "TOTAL" in log_text

    @pytest.mark.unit
    @pytest.mark.parametrize("recording_id,internal_hypothesis_id", [("session_a", "internal_hypothesis_id")])
    def test_main_accepts_hypothesis_manifest(self, tmp_path, recording_id, internal_hypothesis_id):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        hypothesis_manifest = tmp_path / "hypothesis.json"
        write_rttm(reference_path, recording_id, "ref")
        write_rttm(hypothesis_path, internal_hypothesis_id, "hyp")
        hypothesis_manifest.write_text(
            json.dumps(
                {
                    "audio_filepath": f"{recording_id}.wav",
                    "rttm_filepath": hypothesis_path.name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(reference=reference_path, hypothesis=hypothesis_manifest, collar=0.0)

        with (
            patch.object(score_diarization, "parse_args", return_value=args),
            patch("nemo.collections.asr.metrics.der.logging.info") as log_info,
        ):
            score_diarization.main()

        log_text = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert recording_id in log_text
        assert "TOTAL" in log_text
