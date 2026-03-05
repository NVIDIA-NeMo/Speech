# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import librosa
import pytest

from nemo.collections.tts.metrics.eou_classifier import EoUClassification, EoUClassifier, EoUType, TokenSegment

# ---------------------------------------------------------------------------
# TODO: Fill in (audio_path, text) pairs per EoU class.
# Paths are relative to the repo root. Multiple examples per class are supported.
# ---------------------------------------------------------------------------
DATA_PATH = "/home/TestData/tts/eou_classifier_unit_test"
# TEST_NAME, EoU_TYPE, AUDIO_PATH, TEXT
_CLASSIFICATION_CASES: list[tuple[str, EoUType, str, str]] = [
    (
        "good ending",
        EoUType.GOOD,
        f"{DATA_PATH}/rodney.wav",
        "Yes, it is quite amazing to watch and I love all of it.",
    ),
    (
        "cut-off ending",
        EoUType.CUTOFF,
        f"{DATA_PATH}/libritts_test_clean_1320_122612_000056_000003.wav",
        "Having reached within a few yards of the latter, he arose to his feet, silently and slowly.",
    ),
    ("silence tail", EoUType.SILENCE, f"{DATA_PATH}/magpie_silence_wood.wav", "w o o d"),
    ("noise tail", EoUType.NOISE, f"{DATA_PATH}/magpie_noisy_yes.wav", "yes"),
    (
        "noise tail with looping end",
        EoUType.NOISE,
        f"{DATA_PATH}/magpie_repeated_tail.wav",
        "Put them away quick before Andella and Rosalie see them.",
    ),
]


@pytest.fixture(scope="module")
def classifier(request):
    """Load the Wav2Vec2 model once for the entire test module."""
    device = "cpu" if request.config.getoption("--cpu") else "cuda"
    return EoUClassifier(device=device)


# ── classification tests (one per class) ──────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "eou_type, audio_path, text",
    [(t, a, tx) for _, t, a, tx in _CLASSIFICATION_CASES],
    ids=[p for p, _, _, _ in _CLASSIFICATION_CASES],
)
def test_classification_matches_expected_class(classifier, eou_type, audio_path, text):
    """Each sample should be classified as its expected EoU type."""
    result = classifier.classify(audio_path, text)

    assert isinstance(result, EoUClassification)
    assert result.eou_type == eou_type, (
        f"Expected {eou_type.value!r} but got {result.eou_type.value!r} "
        f"(trailing={result.trailing_duration:.3f}s, rms_ratio={result.trail_rms_ratio:.4f}, "
        f"last_conf={result.last_token_confidence:.3f})"
    )


# ── numpy array input ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_classify_accepts_numpy_array(classifier):
    """Classifier should accept a pre-loaded numpy array instead of a path."""
    _, _, audio_path, text = next(c for c in _CLASSIFICATION_CASES if c[1] == EoUType.GOOD)
    samples, _ = librosa.load(audio_path, sr=16000)

    result_from_path = classifier.classify(audio_path, text)
    result_from_array = classifier.classify(samples, text)

    assert result_from_path.eou_type == result_from_array.eou_type
    assert abs(result_from_path.trailing_duration - result_from_array.trailing_duration) < 1e-4


# ── return value structure ────────────────────────────────────────────────


@pytest.mark.unit
def test_classification_result_structure(classifier):
    """Verify the returned dataclass fields have correct types and reasonable ranges."""
    _, _, audio_path, text = next(c for c in _CLASSIFICATION_CASES if c[1] == EoUType.GOOD)
    result = classifier.classify(audio_path, text)

    assert isinstance(result.eou_type, EoUType)
    assert result.speech_end >= 0.0
    assert result.audio_duration > 0.0
    assert result.trailing_duration >= 0.0
    assert result.speech_end <= result.audio_duration + 0.5  # small tolerance for frame rounding
    assert 0.0 <= result.trail_rms_ratio
    assert result.last_token_duration >= 0.0
    assert 0.0 <= result.last_token_confidence <= 1.0
    assert isinstance(result.last_token, str)
    assert result.last_token_gap >= 0.0
    assert 0.0 <= result.last_two_phoneme_avg_confidence <= 1.0

    assert isinstance(result.token_segments, list)
    assert len(result.token_segments) > 0
    for seg in result.token_segments:
        assert isinstance(seg, TokenSegment)
        assert seg.end >= seg.start
        assert seg.duration >= 0.0
        assert 0.0 <= seg.confidence <= 1.0


# ── batched inference ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_batch_matches_unbatched(classifier):
    """Batched inference must produce identical classifications to single-sample."""
    items = [(a, tx) for _, _, a, tx in _CLASSIFICATION_CASES]
    batch_results = classifier.classify_batch(items)

    assert len(batch_results) == len(_CLASSIFICATION_CASES)

    for i, (name, expected_type, audio_path, text) in enumerate(_CLASSIFICATION_CASES):
        single_result = classifier.classify(audio_path, text)
        assert (
            batch_results[i].eou_type == single_result.eou_type
        ), f"Mismatch for {name!r}: batch={batch_results[i].eou_type}, single={single_result.eou_type}"
        assert abs(batch_results[i].trailing_duration - single_result.trailing_duration) < 1e-4, (
            f"trailing_duration mismatch for {name!r}: "
            f"batch={batch_results[i].trailing_duration:.6f}, single={single_result.trailing_duration:.6f}"
        )
        assert abs(batch_results[i].speech_end - single_result.speech_end) < 1e-4, (
            f"speech_end mismatch for {name!r}: "
            f"batch={batch_results[i].speech_end:.6f}, single={single_result.speech_end:.6f}"
        )


@pytest.mark.unit
def test_batch_naive_matches_unbatched(classifier):
    """Naive batched inference (full model including CNN) vs single-sample.

    Tests whether the GroupNorm-in-CNN concern actually causes divergence
    once the attention mask dtype bug is fixed.
    """
    audios = []
    texts = []
    for _, _, audio_path, text in _CLASSIFICATION_CASES:
        samples, _ = librosa.load(audio_path, sr=16000)
        audios.append(samples)
        texts.append(text)

    naive_infos = classifier._forced_align_batch_naive(audios, texts)

    mismatches = []
    for i, (name, expected_type, audio_path, text) in enumerate(_CLASSIFICATION_CASES):
        naive_result = classifier._classify_from_alignment(audios[i], texts[i], naive_infos[i])
        single_result = classifier.classify(audio_path, text)

        type_match = naive_result.eou_type == single_result.eou_type
        trailing_delta = abs(naive_result.trailing_duration - single_result.trailing_duration)
        speech_end_delta = abs(naive_result.speech_end - single_result.speech_end)

        if not type_match or trailing_delta > 1e-4 or speech_end_delta > 1e-4:
            mismatches.append(
                f"  {name!r}: type={naive_result.eou_type}(expected {single_result.eou_type}), "
                f"trailing_delta={trailing_delta:.6f}, speech_end_delta={speech_end_delta:.6f}"
            )

    if mismatches:
        pytest.fail("Naive batched (full-model) vs unbatched mismatches:\n" + "\n".join(mismatches))


@pytest.mark.unit
def test_batch_with_timing(classifier):
    """Smoke-test that log_timing=True doesn't crash."""
    items = [(a, tx) for _, _, a, tx in _CLASSIFICATION_CASES[:2]]
    results = classifier.classify_batch(items, log_timing=True)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, EoUClassification)
