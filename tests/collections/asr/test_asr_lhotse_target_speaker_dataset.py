# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import DummyManifest

from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseTargetSpeakerBpeDataset


def test_background_copy_removes_supervisions_without_mutating_source():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=1, with_data=True)
    source = cuts[0]
    source.supervisions = [
        SupervisionSegment(
            id="source-supervision",
            recording_id=source.recording_id,
            start=0.0,
            duration=source.duration,
            text="background transcript",
            speaker="background-speaker",
        )
    ]

    background = LhotseTargetSpeakerBpeDataset._without_supervisions(source)

    assert background is not source
    assert background.supervisions == []
    assert len(source.supervisions) == 1


def test_activity_targets_preserve_target_overlap_and_absence():
    dataset = LhotseTargetSpeakerBpeDataset.__new__(LhotseTargetSpeakerBpeDataset)
    dataset.num_sample_per_mel_frame = 160
    dataset.num_mel_frame_per_asr_frame = 8

    class Cut:
        def __init__(self, samples):
            self.num_samples = samples

    core = Cut(160 * 8 * 3)
    mixed = Cut(160 * 8 * 5)

    assert dataset._activity_target(core, mixed, target_absent=False).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert dataset._activity_target(core, mixed, target_absent=True).tolist() == [0.0] * 5


def test_embeddings_are_normalized_and_reject_inconsistent_widths(tmp_path):
    embeddings_path = tmp_path / "embeddings.pt"
    torch.save({"cut-a": torch.tensor([3.0, 4.0])}, embeddings_path)
    embeddings = LhotseTargetSpeakerBpeDataset._load_embeddings(str(embeddings_path))
    assert torch.allclose(embeddings["cut-a"].norm(), torch.tensor(1.0))

    invalid_path = tmp_path / "invalid_embeddings.pt"
    torch.save({"cut-a": torch.ones(2), "cut-b": torch.ones(3)}, invalid_path)
    with pytest.raises(ValueError, match="common width"):
        LhotseTargetSpeakerBpeDataset._load_embeddings(str(invalid_path))
