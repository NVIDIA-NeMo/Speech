# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Tests for the is_training flag in DuplexSTTDataset.

Verifies that training-only augmentations (force alignment, audio augmentation,
early interruption) are skipped when is_training=False.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.duplex_stt_dataset import DuplexSTTDataset
from nemo.collections.speechlm2.data.utils import get_pad_id

SR = 16000
FL = 0.08


@pytest.fixture(scope="session")
def tokenizer():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        model_path = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"
    else:
        model_path = "TinyLlama/TinyLlama_v1.1"
    return AutoTokenizer(model_path, use_fast=True)


@pytest.fixture(scope="session")
def cuts():
    cut = dummy_cut(0, duration=2.0, recording=dummy_recording(0, duration=2.0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id="s0-user", recording_id=cut.recording_id, start=0, duration=0.5, text="hello", speaker="user"
        ),
        SupervisionSegment(
            id="s0-agent", recording_id=cut.recording_id, start=0.6, duration=0.5, text="hi there", speaker="assistant"
        ),
        SupervisionSegment(
            id="s0-user2", recording_id=cut.recording_id, start=1.2, duration=0.3, text="thanks", speaker="user"
        ),
    ]
    return CutSet([cut])


def _make_dataset(tokenizer, is_training, cfg=None, model_cfg=None):
    default_cfg = {"prepend_word_space": False, "early_interruption_prob": 0.0}
    if cfg:
        default_cfg.update(cfg)
    default_model_cfg = {"predict_user_text": False, "force_align_user_text": False}
    if model_cfg:
        default_model_cfg.update(model_cfg)
    return DuplexSTTDataset(
        tokenizer=tokenizer,
        frame_length=FL,
        source_sample_rate=SR,
        input_roles=["user"],
        output_roles=["assistant"],
        cfg=default_cfg,
        model_cfg=default_model_cfg,
        is_training=is_training,
    )


def test_is_training_default_is_true(tokenizer):
    """Dataset defaults to is_training=True."""
    ds = DuplexSTTDataset(
        tokenizer=tokenizer,
        frame_length=FL,
        source_sample_rate=SR,
    )
    assert ds.is_training is True


def test_is_training_false_skips_early_interruption(tokenizer, cuts):
    """Early interruption augmentation should only run when is_training=True."""
    cfg = {"prepend_word_space": False, "early_interruption_prob": 1.0, "early_interruption_overlap_tokens": 8}

    train_ds = _make_dataset(tokenizer, is_training=True, cfg=cfg)
    val_ds = _make_dataset(tokenizer, is_training=False, cfg=cfg)

    train_batch = train_ds[cuts]
    val_batch = val_ds[cuts]

    train_targets = train_batch["audio_data"]["target_tokens"]
    val_targets = val_batch["audio_data"]["target_tokens"]

    # With early_interruption_prob=1.0 and is_training=True, targets should be modified.
    # With is_training=False, targets should be unmodified (same as no augmentation).
    no_aug_ds = _make_dataset(tokenizer, is_training=True, cfg={"prepend_word_space": False, "early_interruption_prob": 0.0})
    no_aug_batch = no_aug_ds[cuts]
    no_aug_targets = no_aug_batch["audio_data"]["target_tokens"]

    assert torch.equal(val_targets, no_aug_targets), "Validation dataset should not apply early interruption"


def test_is_training_false_skips_audio_augmentation(tokenizer, cuts, tmp_path):
    """Audio augmentation should only run when is_training=True."""
    from lhotse.dataset.collation import collate_audio

    noise_dir = tmp_path / "noise" / "all"
    noise_dir.mkdir(parents=True)
    for i in range(3):
        noise = np.random.randn(SR).astype(np.float32) * 0.01
        sf.write(str(noise_dir / f"noise_{i}.wav"), noise, SR)

    cfg = {
        "prepend_word_space": False,
        "use_noise_aug": True,
        "noise_prob": 1.0,
        "noise_aug_path": str(tmp_path / "noise"),
        "noise_min_snr": 0,
        "noise_max_snr": 0,
    }

    val_ds = _make_dataset(tokenizer, is_training=False, cfg=cfg)
    assert val_ds.audio_augmenter is not None

    original_audio, _ = collate_audio(cuts.resample(SR))
    val_batch = val_ds[cuts]

    assert torch.equal(val_batch["audio_data"]["source_audio"], original_audio), (
        "Validation dataset should not apply audio augmentation"
    )


def test_is_training_true_applies_audio_augmentation(tokenizer, cuts, tmp_path):
    """Audio augmentation should run when is_training=True."""
    from lhotse.dataset.collation import collate_audio

    noise_dir = tmp_path / "noise" / "all"
    noise_dir.mkdir(parents=True)
    for i in range(3):
        noise = np.random.randn(SR).astype(np.float32) * 0.01
        sf.write(str(noise_dir / f"noise_{i}.wav"), noise, SR)

    cfg = {
        "prepend_word_space": False,
        "use_noise_aug": True,
        "noise_prob": 1.0,
        "noise_aug_path": str(tmp_path / "noise"),
        "noise_min_snr": 0,
        "noise_max_snr": 0,
    }

    train_ds = _make_dataset(tokenizer, is_training=True, cfg=cfg)
    assert train_ds.audio_augmenter is not None

    original_audio, _ = collate_audio(cuts.resample(SR))
    train_batch = train_ds[cuts]

    assert not torch.equal(train_batch["audio_data"]["source_audio"], original_audio), (
        "Training dataset should apply audio augmentation"
    )


def test_is_training_false_skips_force_alignment(tokenizer, cuts):
    """Force alignment should only run when is_training=True."""
    model_cfg = {"predict_user_text": True, "force_align_user_text": True, "force_align_device": "cpu"}

    val_ds = _make_dataset(tokenizer, is_training=False, model_cfg=model_cfg)

    # Force aligner should be created but never called during validation
    val_ds.force_aligner = MagicMock()
    val_ds[cuts]
    val_ds.force_aligner.batch_force_align_user_audio.assert_not_called()


def test_is_training_true_calls_force_alignment(tokenizer, cuts):
    """Force alignment should run when is_training=True."""
    model_cfg = {"predict_user_text": True, "force_align_user_text": True, "force_align_device": "cpu"}

    train_ds = _make_dataset(tokenizer, is_training=True, model_cfg=model_cfg)

    # Mock the force aligner to avoid loading wav2vec2
    train_ds.force_aligner = MagicMock()
    train_ds.force_aligner.batch_force_align_user_audio.side_effect = lambda cuts, **kwargs: cuts
    train_ds[cuts]
    train_ds.force_aligner.batch_force_align_user_audio.assert_called_once()
