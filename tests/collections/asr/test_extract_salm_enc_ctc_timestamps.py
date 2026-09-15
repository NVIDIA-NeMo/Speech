# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


_TIMESTAMP_SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "asr" / "extract_salm_enc_ctc_timestamps.py"
_SPEC = importlib.util.spec_from_file_location("pee_transformer_ctc_timestamp_cli", _TIMESTAMP_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
timestamp_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(timestamp_cli)


def _speaker_word_timestamps():
    # The first and third speaker-zero words are one semantic segment, but a
    # speaker-one word interrupts them in chronological order. Gecko can only
    # group adjacent time-sorted CTM rows, so its writer must start a fresh ID
    # for the final speaker-zero word.
    return {
        "1": [
            {"word": "yeah", "start": 0.20, "end": 0.25, "speaker_tag": 1, "word_index": 2, "turn_index": 1},
            {"word": "later", "start": 0.70, "end": 0.80, "speaker_tag": 1, "word_index": 5, "turn_index": 3},
        ],
        "0": [
            {"word": "hello", "start": 0.00, "end": 0.10, "speaker_tag": 0, "word_index": 0, "turn_index": 0},
            {"word": "there", "start": 0.11, "end": 0.19, "speaker_tag": 0, "word_index": 1, "turn_index": 0},
            {"word": "again", "start": 0.27, "end": 0.35, "speaker_tag": 0, "word_index": 3, "turn_index": 0},
            {"word": "after", "start": 0.50, "end": 0.60, "speaker_tag": 0, "word_index": 4, "turn_index": 2},
        ],
    }


def test_build_ctm_lines_emits_gecko_speaker_segment_ids():
    lines = timestamp_cli.build_ctm_lines(_speaker_word_timestamps(), merge_threshold=0.1)
    fields = [line.split() for line in lines]

    assert all(len(row) == 6 for row in fields)
    assert [row[1] for row in fields] == ["1"] * len(fields)
    assert [row[4] for row in fields] == ["hello", "there", "yeah", "again", "after", "later"]
    assert [row[5] for row in fields] == ["-1.00"] * len(fields)
    assert [row[0] for row in fields] == [
        "spk0_00000_audio",
        "spk0_00000_audio",
        "spk1_00001_audio",
        "spk0_00002_audio",
        "spk0_00003_audio",
        "spk1_00004_audio",
    ]

    # These are precisely the values Gecko derives from the first CTM field.
    assert [row[0].split("_")[0] for row in fields] == ["spk0", "spk0", "spk1", "spk0", "spk0", "spk1"]
    assert [int(row[0].split("_")[1]) for row in fields] == [0, 0, 1, 2, 3, 4]


def test_output_dir_ctm_uses_the_same_gecko_segmenting(tmp_path):
    timestamps = _speaker_word_timestamps()
    result = {"speaker_word_timestamps": timestamps}
    written = timestamp_cli.write_record_output_dir(
        tmp_path,
        session_id="recording_name",
        manifest_record_index=None,
        audio_path=Path("/audio/recording_name.wav"),
        result=result,
        output={"speaker_word_timestamps": timestamps},
        merge_threshold=0.1,
    )

    assert written["ctm"].read_text(encoding="utf-8").splitlines() == timestamp_cli.build_ctm_lines(
        timestamps, merge_threshold=0.1
    )
    seglst = timestamp_cli.build_seglst_segments(
        "recording_name", Path("/audio/recording_name.wav"), timestamps, merge_threshold=0.1
    )
    assert [segment["words"] for segment in seglst] == ["hello there again", "yeah", "after", "later"]
