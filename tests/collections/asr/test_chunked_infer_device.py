# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch
from examples.asr.asr_chunked_inference.aed import speech_to_text_aed_chunked_infer as aed_chunked_infer
from examples.asr.asr_chunked_inference.rnnt import speech_to_text_buffered_infer_rnnt as buffered_infer_rnnt
from omegaconf import OmegaConf


class _ModelSetupReached(Exception):
    pass


@pytest.fixture
def restore_torch_globals():
    # main() changes process-wide autograd and matmul precision settings
    grad_enabled = torch.is_grad_enabled()
    matmul_precision = torch.get_float32_matmul_precision()
    yield
    torch.set_grad_enabled(grad_enabled)
    torch.set_float32_matmul_precision(matmul_precision)


@pytest.mark.unit
@pytest.mark.parametrize("script", [buffered_infer_rnnt, aed_chunked_infer])
@pytest.mark.parametrize(
    "cuda, cuda_available, expected",
    [(-1, True, "cpu"), (None, False, "cpu"), (None, True, "cuda:0"), (1, True, "cuda:1")],
)
def test_device_selection(script, cuda, cuda_available, expected, tmp_path, restore_torch_globals):
    cfg = OmegaConf.structured(
        script.TranscriptionConfig(pretrained_name="any-model", audio_dir=str(tmp_path), cuda=cuda)
    )
    with (
        mock.patch("torch.cuda.is_available", return_value=cuda_available),
        mock.patch.object(script, "setup_model", side_effect=_ModelSetupReached) as setup_model,
    ):
        with pytest.raises(_ModelSetupReached):
            script.main(cfg)

    assert setup_model.call_args.args[1] == torch.device(expected)
