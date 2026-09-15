# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo.collections.asr.parts.utils.sortformer_utils import (
    get_prediction_cache_metadata,
    load_prediction_tensors,
    save_prediction_tensors,
    validate_prediction_tensors,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "cache_version, model_filename, model_contents, manifest_filename, manifest_contents, recording_ids, "
        "num_speakers, output_subsampling_factor, precision, presort_manifest, streaming_flags, chunk_context, "
        "cache_lengths, score_boosts"
    ),
    [
        (
            1,
            "model.nemo",
            b"model-contents",
            "manifest.json",
            '{"audio_filepath": "audio.wav"}\n',
            ("recording-b", "recording-a"),
            4,
            8,
            "bf16",
            True,
            (True, True, True, True),
            (6, 1, 7),
            (188, 144, 188),
            (0.75, 1.5, 0.25),
        )
    ],
)
def test_get_prediction_cache_metadata(
    tmp_path,
    cache_version,
    model_filename,
    model_contents,
    manifest_filename,
    manifest_contents,
    recording_ids,
    num_speakers,
    output_subsampling_factor,
    precision,
    presort_manifest,
    streaming_flags,
    chunk_context,
    cache_lengths,
    score_boosts,
):
    streaming_mode, async_streaming, async_pad_to_max, async_desync_updates = streaming_flags
    chunk_len, chunk_left_context, chunk_right_context = chunk_context
    spkcache_len, spkcache_update_period, fifo_len = cache_lengths
    strong_boost_rate, weak_boost_rate, scores_boost_latest = score_boosts
    model_path = tmp_path / model_filename
    manifest_path = tmp_path / manifest_filename
    model_path.write_bytes(model_contents)
    manifest_path.write_text(manifest_contents)
    model_stat = model_path.stat()
    manifest_stat = manifest_path.stat()
    cfg = SimpleNamespace(
        model_path=str(model_path),
        dataset_manifest=str(manifest_path),
        output_subsampling_factor=output_subsampling_factor,
        precision=precision,
        presort_manifest=presort_manifest,
        async_streaming=async_streaming,
        async_pad_to_max=async_pad_to_max,
        async_desync_updates=async_desync_updates,
        chunk_len=chunk_len,
        chunk_left_context=chunk_left_context,
        chunk_right_context=chunk_right_context,
        spkcache_len=spkcache_len,
        spkcache_update_period=spkcache_update_period,
        fifo_len=fifo_len,
    )
    diar_model = SimpleNamespace(
        _cfg=SimpleNamespace(max_num_of_spks=num_speakers),
        streaming_mode=streaming_mode,
        sortformer_modules=SimpleNamespace(
            strong_boost_rate=strong_boost_rate,
            weak_boost_rate=weak_boost_rate,
            scores_boost_latest=scores_boost_latest,
        ),
    )
    infer_audio_rttm_dict = {recording_id: {} for recording_id in recording_ids}

    metadata = get_prediction_cache_metadata(cfg, diar_model, infer_audio_rttm_dict)

    assert metadata == {
        "version": cache_version,
        "model_path": str(model_path.resolve()),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_size": manifest_stat.st_size,
        "manifest_mtime_ns": manifest_stat.st_mtime_ns,
        "recording_ids": list(recording_ids),
        "num_speakers": num_speakers,
        "output_subsampling_factor": output_subsampling_factor,
        "precision": precision,
        "presort_manifest": presort_manifest,
        "streaming_mode": streaming_mode,
        "async_streaming": async_streaming,
        "async_pad_to_max": async_pad_to_max,
        "async_desync_updates": async_desync_updates,
        "chunk_len": chunk_len,
        "chunk_left_context": chunk_left_context,
        "chunk_right_context": chunk_right_context,
        "spkcache_len": spkcache_len,
        "spkcache_update_period": spkcache_update_period,
        "fifo_len": fifo_len,
        "strong_boost_rate": strong_boost_rate,
        "weak_boost_rate": weak_boost_rate,
        "scores_boost_latest": scores_boost_latest,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "container_type, prediction_shape, recording_ids, num_speakers",
    [
        ("list", (1, 3, 2), ("recording",), 2),
        ("tuple", (1, 3, 2), ("recording",), 2),
    ],
    ids=["list", "tuple"],
)
def test_validate_prediction_tensors_accepts_valid_predictions(
    container_type, prediction_shape, recording_ids, num_speakers
):
    prediction = torch.ones(prediction_shape)
    predictions = [prediction] if container_type == "list" else (prediction,)
    metadata = {"recording_ids": list(recording_ids), "num_speakers": num_speakers}

    validated = validate_prediction_tensors(predictions, metadata)

    assert isinstance(validated, list)
    assert len(validated) == 1
    assert torch.equal(validated[0], predictions[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    "container_type, prediction_specs, recording_ids, num_speakers, error_match",
    [
        ("tensor", (1, 3, 2), ("recording",), 2, "must contain a list"),
        ("list", (), ("recording",), 2, "contains 0 recordings"),
        ("list", ("not-a-tensor",), ("recording",), 2, "must have shape"),
        ("list", ((3, 2),), ("recording",), 2, "must have shape"),
        ("list", ((2, 3, 2),), ("recording",), 2, "must have shape"),
        ("list", ((1, 3, 3),), ("recording",), 2, "has 3 speakers"),
    ],
    ids=[
        "invalid-container",
        "recording-count",
        "non-tensor",
        "wrong-rank",
        "wrong-batch-size",
        "wrong-speaker-count",
    ],
)
def test_validate_prediction_tensors_rejects_invalid_predictions(
    container_type, prediction_specs, recording_ids, num_speakers, error_match
):
    if container_type == "tensor":
        predictions = torch.ones(prediction_specs)
    else:
        predictions = [
            torch.ones(prediction_spec) if isinstance(prediction_spec, tuple) else prediction_spec
            for prediction_spec in prediction_specs
        ]
    metadata = {"recording_ids": list(recording_ids), "num_speakers": num_speakers}

    with pytest.raises(ValueError, match=error_match):
        validate_prediction_tensors(predictions, metadata)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "cache_filename, metadata_version, recording_ids, num_speakers, first_prediction_shape, "
        "first_fill_value, second_prediction_shape, second_fill_value"
    ),
    [("cache/predictions.pt", 1, ("session",), 2, (1, 4, 2), 1.0, (1, 5, 2), 0.0)],
)
def test_prediction_tensor_cache_round_trip_and_atomic_overwrite(
    tmp_path,
    cache_filename,
    metadata_version,
    recording_ids,
    num_speakers,
    first_prediction_shape,
    first_fill_value,
    second_prediction_shape,
    second_fill_value,
):
    tensor_path = tmp_path / cache_filename
    metadata = {
        "version": metadata_version,
        "recording_ids": list(recording_ids),
        "num_speakers": num_speakers,
    }
    first_predictions = [torch.full(first_prediction_shape, first_fill_value)]
    second_predictions = [torch.full(second_prediction_shape, second_fill_value)]

    save_prediction_tensors(str(tensor_path), first_predictions, metadata)
    assert torch.equal(load_prediction_tensors(str(tensor_path), metadata)[0], first_predictions[0])

    save_prediction_tensors(str(tensor_path), second_predictions, metadata)
    assert torch.equal(load_prediction_tensors(str(tensor_path), metadata)[0], second_predictions[0])
    assert list(tensor_path.parent.glob(f".{tensor_path.name}.*.tmp")) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "cache_filename, metadata_version, recording_ids, mismatched_recording_ids, num_speakers, prediction_shape, "
        "error_match"
    ),
    [("predictions.pt", 1, ("session",), ("different-session",), 2, (1, 4, 2), "recording_ids")],
)
def test_prediction_tensor_cache_rejects_metadata_mismatch(
    tmp_path,
    cache_filename,
    metadata_version,
    recording_ids,
    mismatched_recording_ids,
    num_speakers,
    prediction_shape,
    error_match,
):
    tensor_path = tmp_path / cache_filename
    metadata = {
        "version": metadata_version,
        "recording_ids": list(recording_ids),
        "num_speakers": num_speakers,
    }
    save_prediction_tensors(str(tensor_path), [torch.ones(prediction_shape)], metadata)

    incompatible_metadata = {**metadata, "recording_ids": list(mismatched_recording_ids)}
    with pytest.raises(ValueError, match=error_match):
        load_prediction_tensors(str(tensor_path), incompatible_metadata)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cache_filename, metadata_version, recording_id, num_speakers, prediction_shape",
    [("legacy.pt", 1, "session", 2, (1, 4, 2))],
)
def test_legacy_prediction_tensor_cache_is_supported(
    tmp_path, cache_filename, metadata_version, recording_id, num_speakers, prediction_shape
):
    tensor_path = tmp_path / cache_filename
    predictions = [torch.ones(prediction_shape)]
    metadata = {
        "version": metadata_version,
        "recording_ids": [recording_id],
        "num_speakers": num_speakers,
    }
    torch.save(predictions, tensor_path)

    loaded_predictions = load_prediction_tensors(str(tensor_path), metadata)

    assert torch.equal(loaded_predictions[0], predictions[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    "cache_filename, metadata_version, recording_ids, num_speakers, prediction_shape, missing_key, error_match",
    [
        (
            "malformed.pt",
            1,
            ("recording",),
            2,
            (1, 3, 2),
            "metadata",
            "must contain 'metadata' and 'predictions'",
        ),
        (
            "malformed.pt",
            1,
            ("recording",),
            2,
            (1, 3, 2),
            "predictions",
            "must contain 'metadata' and 'predictions'",
        ),
    ],
    ids=["missing-metadata", "missing-predictions"],
)
def test_load_prediction_tensors_rejects_malformed_dictionary_payload(
    tmp_path,
    cache_filename,
    metadata_version,
    recording_ids,
    num_speakers,
    prediction_shape,
    missing_key,
    error_match,
):
    tensor_path = tmp_path / cache_filename
    expected_metadata = {
        "version": metadata_version,
        "recording_ids": list(recording_ids),
        "num_speakers": num_speakers,
    }
    payload = {
        "metadata": expected_metadata,
        "predictions": [torch.ones(prediction_shape)],
    }
    del payload[missing_key]
    torch.save(payload, tensor_path)

    with pytest.raises(ValueError, match=error_match):
        load_prediction_tensors(str(tensor_path), expected_metadata)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "cache_filename, original_contents, metadata_version, recording_ids, num_speakers, prediction_shape, "
        "error_message"
    ),
    [("cache/predictions.pt", b"existing-cache", 1, ("recording",), 2, (1, 3, 2), "save failed")],
)
def test_save_prediction_tensors_removes_temporary_file_after_failure(
    tmp_path,
    cache_filename,
    original_contents,
    metadata_version,
    recording_ids,
    num_speakers,
    prediction_shape,
    error_message,
):
    tensor_path = tmp_path / cache_filename
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(original_contents)
    metadata = {
        "version": metadata_version,
        "recording_ids": list(recording_ids),
        "num_speakers": num_speakers,
    }

    with patch(
        "nemo.collections.asr.parts.utils.sortformer_utils.torch.save",
        side_effect=RuntimeError(error_message),
    ):
        with pytest.raises(RuntimeError, match=error_message):
            save_prediction_tensors(str(tensor_path), [torch.ones(prediction_shape)], metadata)

    assert tensor_path.read_bytes() == original_contents
    assert list(tensor_path.parent.glob(f".{tensor_path.name}.*.tmp")) == []
