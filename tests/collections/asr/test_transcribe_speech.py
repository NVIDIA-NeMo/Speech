# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch
from examples.asr import transcribe_speech
from omegaconf import OmegaConf


class _TrainerReached(Exception):
    pass


def _devices_for(cuda, tmp_path):
    """Run transcribe_speech.main() up to the Trainer and return the device choices it made."""
    cfg = OmegaConf.structured(
        transcribe_speech.TranscriptionConfig(
            pretrained_name="any-model", audio_dir=str(tmp_path), cuda=cuda, matmul_precision="highest"
        )
    )
    with (
        mock.patch.object(transcribe_speech, "setup_model", return_value=(mock.Mock(), "any-model")) as setup_model,
        mock.patch.object(transcribe_speech.pl, "Trainer", side_effect=_TrainerReached) as trainer,
    ):
        with pytest.raises(_TrainerReached):
            transcribe_speech.main(cfg)
    return setup_model.call_args.args[1], trainer.call_args.kwargs


@pytest.mark.unit
def test_negative_cuda_runs_on_cpu(tmp_path):
    with mock.patch("torch.cuda.is_available", return_value=True):
        map_location, trainer_kwargs = _devices_for(-1, tmp_path)

    assert map_location == torch.device("cpu")
    assert trainer_kwargs == {"devices": 1, "accelerator": "cpu"}


@pytest.mark.unit
@pytest.mark.parametrize("cuda, index", [(None, 0), (1, 1)])
def test_cuda_device_selection(tmp_path, cuda, index):
    with mock.patch("torch.cuda.is_available", return_value=True):
        map_location, trainer_kwargs = _devices_for(cuda, tmp_path)

    assert map_location == torch.device(f"cuda:{index}")
    assert trainer_kwargs == {"devices": [index], "accelerator": "gpu"}
