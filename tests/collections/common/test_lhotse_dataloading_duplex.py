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

from itertools import islice
from pathlib import Path

import lhotse
import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import CutSet, SupervisionSegment
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config


class Identity(torch.utils.data.Dataset):
    """Dummy dataset class to return the raw CutSet for testing dataloader output."""

    def __getitem__(self, cuts: lhotse.CutSet) -> lhotse.CutSet:
        return cuts


def create_wav_file(path: Path, duration: float, sample_rate: int = 16000):
    """Helper to create a valid, silent WAV file on disk to bypass memory bytes serialization."""
    samples = np.zeros((1, int(duration * sample_rate)), dtype=np.float32)
    sf.write(str(path), samples.T, sample_rate, format='WAV')


@pytest.fixture(scope="session")
def cutset_shar_s2s_overlap_path(tmp_path_factory) -> Path:
    """5 utterances representing conversational overlap data as Lhotse Shar."""
    tmp_dir = tmp_path_factory.mktemp("overlap_audio")
    cuts = []

    for i in range(5):
        main_path = tmp_dir / f"ov_main_{i}.wav"
        create_wav_file(main_path, duration=5.0)

        c = lhotse.MonoCut(
            id=f"ov_cut_{i}",
            start=0.0,
            duration=5.0,
            channel=0,
            recording=lhotse.Recording.from_file(main_path, recording_id=f"ov_main_{i}"),
        )
        # Add custom overlapping segments
        c.supervisions = []
        c.custom = {
            "agent_segments": [{"start": 0.5, "end": 2.0, "text": "agent speaking"}],
            "user_segments": [{"start": 1.0, "end": 3.0, "text": "user speaking"}],
        }
        cuts.append(c)

    cuts = CutSet.from_cuts(cuts)
    p = tmp_path_factory.mktemp("overlap_shar")
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture(scope="session")
def cutset_shar_magpietts_path(tmp_path_factory) -> Path:
    """5 utterances representing MagpieTTS data with target and context audio."""
    tmp_dir = tmp_path_factory.mktemp("magpie_audio")
    cuts = []

    for i in range(5):
        main_path = tmp_dir / f"mag_main_{i}.wav"
        tgt_path = tmp_dir / f"mag_target_{i}.wav"
        ctx_path = tmp_dir / f"mag_context_{i}.wav"

        create_wav_file(main_path, duration=2.0)
        create_wav_file(tgt_path, duration=2.0)
        create_wav_file(ctx_path, duration=1.0)

        c = lhotse.MonoCut(
            id=f"mag_cut_{i}",
            start=0.0,
            duration=2.0,
            channel=0,
            recording=lhotse.Recording.from_file(main_path, recording_id=f"mag_main_{i}"),
        )

        c.custom = {
            "target_audio": lhotse.Recording.from_file(tgt_path, recording_id=f"mag_target_{i}"),
            "context_audio": lhotse.Recording.from_file(ctx_path, recording_id=f"mag_context_{i}"),
        }

        c.supervisions = [
            SupervisionSegment(
                id=f"sup_{i}",
                recording_id=c.recording.id,
                start=0.0,
                duration=2.0,
                text="hello",
                speaker="agent",
                custom={"cer": 0.01, "context_speaker_similarity": 0.9, "validation_status": "pass"},
            )
        ]
        cuts.append(c)

    cuts = CutSet.from_cuts(cuts)
    p = tmp_path_factory.mktemp("magpie_shar")
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture(scope="session")
def regular_duplex_s2s_format(tmp_path_factory) -> Path:
    """5 utterances representing duplex conversational data for role reversal."""
    tmp_dir = tmp_path_factory.mktemp("reverse_role_audio")
    cuts = []

    for i in range(5):
        main_path = tmp_dir / f"rr_main_{i}.wav"
        tgt_path = tmp_dir / f"rr_target_{i}.wav"

        create_wav_file(main_path, duration=3.0)
        create_wav_file(tgt_path, duration=3.0)

        c = lhotse.MonoCut(
            id=f"rr_cut_{i}",
            start=0.0,
            duration=3.0,
            channel=0,
            recording=lhotse.Recording.from_file(main_path, recording_id=f"rr_main_{i}"),
        )

        # Store an alternative target recording in the custom field
        c.custom = {"target_audio": lhotse.Recording.from_file(tgt_path, recording_id=f"rr_target_{i}")}

        c.supervisions = [
            SupervisionSegment(
                id=f"sup_{i}_1", recording_id=c.recording.id, start=0.0, duration=1.0, speaker="user", text="hello"
            ),
            SupervisionSegment(
                id=f"sup_{i}_2", recording_id=c.recording.id, start=1.5, duration=1.0, speaker="agent", text="hi"
            ),
        ]
        cuts.append(c)

    cuts = CutSet.from_cuts(cuts)
    p = tmp_path_factory.mktemp("reverse_role_shar")
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


def test_data_input_cfg_s2s_overlap(cutset_shar_s2s_overlap_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "s2s_duplex_overlap_as_s2s_duplex",
                    "shar_path": str(cutset_shar_s2s_overlap_path),
                    "weight": 1.0,
                    "move_agent_text_back_by": 0.1,
                    "filter_samples_starting_with_agent": False,
                    "tags": {
                        "dataset_name": "OverlapData",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Verify dataloader and transformations
    batches = [batch for batch in islice(dl, 1)]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["dataset_name"] == "OverlapData" for c in b)

    for cut in b:
        assert cut.task == "s2s_duplex_overlap_as_s2s_duplex"
        assert len(cut.supervisions) == 2

        # Verify chronological sorting and offsets applied correctly
        sups = sorted(cut.supervisions, key=lambda s: s.start)
        assert sups[0].speaker == "agent"  # agent starts at 0.5 - 0.1 = 0.4
        assert sups[0].start == pytest.approx(0.4)
        assert sups[1].speaker == "user"  # user starts at 1.0


def test_data_input_cfg_magpietts(cutset_shar_magpietts_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_magpietts_data_as_continuation",
                    "shar_path": str(cutset_shar_magpietts_path),
                    "weight": 1.0,
                    "sample_rate": 22050,
                    "add_extra_end_silence": False,
                    "tags": {
                        "dataset_name": "MagpieData",
                    },
                },
            ],
            "sample_rate": 22050,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batches = [batch for batch in islice(dl, 1)]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["dataset_name"] == "MagpieData" for c in b)

    for cut in b:
        assert cut.task == "lhotse_magpietts_data_as_continuation"
        assert hasattr(cut, "target_audio")
        assert hasattr(cut, "context_audio")
        assert hasattr(cut, "recording")
        assert len(cut.supervisions) == 2

        # Verify synthetic user/agent split behavior
        assert cut.supervisions[0].speaker == "user"
        assert cut.supervisions[0].duration == pytest.approx(0.08)
        assert cut.supervisions[1].speaker == "agent"


def test_data_input_cfg_reverse_role(regular_duplex_s2s_format):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "s2s_duplex_reverse_role",
                    "shar_path": str(regular_duplex_s2s_format),
                    "weight": 1.0,
                    "target_agent_name": "swapped_agent",
                    "target_user_name": "swapped_user",
                    "tags": {
                        "dataset_name": "ReverseRoleData",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batches = [batch for batch in islice(dl, 1)]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["dataset_name"] == "ReverseRoleData" for c in b)

    for cut in b:
        assert cut.task == "s2s_duplex_reverse_role"

        # Verify the roles have been inverted according to configuration overrides
        sups = sorted(cut.supervisions, key=lambda s: s.start)
        assert sups[0].speaker == "swapped_agent"  # Originally "user"
        assert sups[1].speaker == "swapped_user"  # Originally "agent"

        # Ensure the recording streams were swapped
        assert cut.recording.id.startswith("rr_target")
        assert cut.target_audio.id.startswith("rr_main")
