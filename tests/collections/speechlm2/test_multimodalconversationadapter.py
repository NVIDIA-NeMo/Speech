import json
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversationJsonlAdapter,
    AudioTurn,
)

@pytest.fixture
def dummy_audio_manifest(tmp_path: Path):
    """
    Creates a dummy WAV file with 3 segments:
        - 0.0 -> 0.25s: 0.25
        - 0.25 -> 1.0s: 0.5
        - 1.0 -> 1.5s: 0.75
    And a JSONL manifest with three conversation rows:
        - row1: first 0.25s
        - row2: next 0.75s
        - row3: full audio
    Returns:
       Path to the JSONL manifest
       Sample rate
    """
    sample_rate = 16_000
    audio_path = tmp_path / "audio.wav"

    # Generate piecewise audio
    audio = np.concatenate([
        np.full(int(0.25 * sample_rate), 0.25, dtype=np.float32),
        np.full(int(0.75 * sample_rate), 0.5, dtype=np.float32),
        np.full(int(0.5 * sample_rate), 0.75, dtype=np.float32),
    ])
    sf.write(audio_path, audio, samplerate=sample_rate)

    # Create JSONL manifest
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "id": "row1",
            "conversations": [
                {
                    "from": "User",
                    "value": "audio.wav",
                    "type": "audio",
                    "duration": 0.25,
                    "offset": 0.0,
                },
            ],
        },
        {
            "id": "row2",
            "conversations": [
                {
                    "from": "User",
                    "value": "audio.wav",
                    "type": "audio",
                    "duration": 0.75,
                    "offset": 0.25,
                },
            ],
        },
        {
            "id": "row3",
            "conversations": [
                {
                    "from": "User",
                    "value": "audio.wav",
                    "type": "audio",
                },
            ],
        }
    ]

    with open(manifest_path, "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    return manifest_path, sample_rate
            
def test_adapter_loading_manifest_jsonl(dummy_audio_manifest):
    """
    Manifest row:
    - two audio turns from the same file
    - first: 0.5s starting at 0.0
    - second: 1.5s starting at 0.5
    """

    manifest_path, sample_rate = dummy_audio_manifest

    adapter = NeMoMultimodalConversationJsonlAdapter(
        manifest_path,
        audio_locator_tag="<dummy>",
    )

    items = list(adapter)
    assert len(items) == 3

    expected_samples_0 = int(0.25 * sample_rate)
    audio = [t for t in items[0].turns if isinstance(t, AudioTurn)][0].cut.load_audio()
    assert audio.shape[1] == expected_samples_0
    assert np.allclose(audio[0], 0.25, atol=1e-3)

    expected_samples_1 = int(0.75 * sample_rate)
    audio = [t for t in items[1].turns if isinstance(t, AudioTurn)][0].cut.load_audio()
    assert audio.shape[1] == expected_samples_1
    assert np.allclose(audio[0], 0.5, atol=1e-3)

    expected_samples_2 = int(1.5 * sample_rate)
    audio = [t for t in items[2].turns if isinstance(t, AudioTurn)][0].cut.load_audio()
    assert audio.shape[1] == expected_samples_2
    assert audio[0][0] == 0.25
    assert audio[0][-1] == 0.75
    assert not np.allclose(audio, 0.5, atol=1e-3)
