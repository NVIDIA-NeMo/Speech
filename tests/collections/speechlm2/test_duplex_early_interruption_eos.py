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

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock
from lightning.pytorch.callbacks import Callback
import lhotse.lazy

if not hasattr(lhotse.lazy, "get_graph_origin"):
    lhotse.lazy.get_graph_origin = MagicMock()
if not hasattr(lhotse.lazy, "resolve_iterator_source"):
    lhotse.lazy.resolve_iterator_source = MagicMock()

class _DummyTimeEventCallback(Callback):
    def __init__(self, *args, **kwargs):
        pass

for mod in [
    "lhotse.indexing",
    "lhotse.shar",
    "lhotse.shar.lazy_pointer",
    "nv_one_logger",
    "nv_one_logger.api",
    "nv_one_logger.api.config",
    "nv_one_logger.training_telemetry",
    "nv_one_logger.training_telemetry.api",
    "nv_one_logger.training_telemetry.api.callbacks",
    "nv_one_logger.training_telemetry.api.config",
    "nv_one_logger.training_telemetry.api.training_telemetry_provider",
    "nv_one_logger.training_telemetry.integration",
    "nv_one_logger.training_telemetry.integration.pytorch_lightning",
]:
    sys.modules[mod] = MagicMock()

sys.modules["nv_one_logger.training_telemetry.integration.pytorch_lightning"].TimeEventCallback = _DummyTimeEventCallback

# Pre-populate submodule stubs to isolate unit test from optional heavyweight dependencies
speechlm2_mod = types.ModuleType("nemo.collections.speechlm2")
speechlm2_data_mod = types.ModuleType("nemo.collections.speechlm2.data")
speechlm2_utils_mod = types.ModuleType("nemo.collections.speechlm2.data.utils")
speechlm2_utils_mod.get_pad_id = lambda tokenizer: getattr(tokenizer, "pad_id", getattr(tokenizer, "pad", 0))

sys.modules["nemo.collections.speechlm2"] = speechlm2_mod
sys.modules["nemo.collections.speechlm2.data"] = speechlm2_data_mod
sys.modules["nemo.collections.speechlm2.data.utils"] = speechlm2_utils_mod
sys.modules["nemo.collections.speechlm2.data.force_align"] = MagicMock()
sys.modules["nemo.collections.speechlm2.data.s2s_dataset"] = MagicMock()
sys.modules["nemo.collections.speechlm2.parts"] = types.ModuleType("parts")
sys.modules["nemo.collections.speechlm2.parts.augmentation"] = MagicMock()

import pytest
import torch

path = Path(__file__).resolve().parents[3] / "nemo/collections/speechlm2/data/duplex_stt_dataset.py"
spec = importlib.util.spec_from_file_location("duplex_stt_dataset", str(path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DuplexSTTDataset = mod.DuplexSTTDataset


@pytest.mark.unit
def test_early_interruption_preserves_relocated_eos():
    """Verify that early-interruption augmentation does not overwrite the relocated EOS token
    when original EOS is near the sequence end."""
    obj = types.SimpleNamespace()
    obj.tokenizer = types.SimpleNamespace(bos=1, eos=2, pad_id=0, pad=0)
    obj.cfg = {"early_interruption_overlap_tokens": 5}
    obj.frame_length = 0.08
    obj.source_sample_rate = 16000

    target_tokens = torch.full((1, 25), 0, dtype=torch.long)
    target_tokens[0, 0] = 1        # BOS
    target_tokens[0, 1:13] = torch.arange(10, 22)  # content
    target_tokens[0, 24] = 2       # original EOS at the last index

    source_tokens = torch.full((1, 25), 0, dtype=torch.long)
    source_tokens[0, 0] = 1
    source_tokens[0, 1:13] = torch.arange(10, 22)
    source_tokens[0, 24] = 2

    source_audio = torch.zeros((1, 25 * 1280), dtype=torch.float32)
    source_audio_lens = torch.tensor([25 * 1280], dtype=torch.long)

    # Call the augmentation
    DuplexSTTDataset._apply_early_interruption_augmentation(
        obj, target_tokens, source_tokens, source_audio, source_audio_lens, 0
    )

    # Relocated EOS token must be preserved in target_tokens
    eos_count = (target_tokens[0] == 2).sum().item()
    assert eos_count == 1, f"Expected 1 EOS token, but found {eos_count}"
