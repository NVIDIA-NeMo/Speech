# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt")
pytest.importorskip("cuda.bindings.runtime")

from examples.asr.export import run_trt_preprocessor


@pytest.mark.unit
@pytest.mark.parametrize("name", ["features", "features.npz", "features.data", ".npz", "features.NPZ"])
@pytest.mark.parametrize("collision", [None, "existing", "during_preprocessing"])
def test_preprocessor_output_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, collision: str | None
) -> None:
    output = tmp_path / name
    destination = tmp_path / (name if name.endswith(".npz") else name + ".npz")
    input_path = tmp_path / "input.npz"
    signal = np.arange(8, dtype=np.float32).reshape(1, 8)
    lengths = np.array([8], dtype=np.int64)
    np.savez(input_path, input_signal=signal, length=lengths)
    sentinel = b"existing archive must remain unchanged"
    if collision == "existing":
        destination.write_bytes(sentinel)

    def preprocess(engine_path: Path, waveforms: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        np.testing.assert_array_equal(waveforms, signal)
        np.testing.assert_array_equal(counts, lengths)
        if collision == "during_preprocessing":
            destination.write_bytes(sentinel)
        return signal, lengths

    monkeypatch.setattr(run_trt_preprocessor, "preprocess", preprocess)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_trt_preprocessor", "--engine", "unused.plan", "--input", str(input_path), "--output", str(output)],
    )
    if collision is not None:
        with pytest.raises(FileExistsError):
            run_trt_preprocessor.main()
        assert destination.read_bytes() == sentinel
    else:
        run_trt_preprocessor.main()
        with np.load(destination, allow_pickle=False) as result:
            assert set(result.files) == {"processed_signal", "processed_length"}
            np.testing.assert_array_equal(result["processed_signal"], signal)
            np.testing.assert_array_equal(result["processed_length"], lengths)
    if output != destination:
        assert not output.exists()
