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
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).parents[3]


def test_salm_automodel_uses_portable_moe_dispatcher():
    cfg = OmegaConf.load(REPO_ROOT / "examples/speechlm2/conf/salm_automodel.yaml")

    assert cfg.model.automodel_backend.dispatcher == "torch"


def test_streaming_stt_multispeaker_suffix_changes_only_the_speaker_markup():
    import warnings
    from types import SimpleNamespace

    from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataConfig
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel, StreamingSTTModelConfig
    from nemo.collections.speechlm2.parts.utils import to_dataclass

    conf = REPO_ROOT / "examples/speechlm2/conf"
    prefix = OmegaConf.load(conf / "streaming_stt_multispeaker.yaml")
    suffix = OmegaConf.load(conf / "streaming_stt_multispeaker_suffix.yaml")

    assert suffix.data.dataset.speaker_tag_placement == "suffix"
    assert suffix.model.speaker_tokens.turn_start_token == "<|turn_start|>"
    assert suffix.data.dataset.turn_start_token == "<|turn_start|>"  # interpolated from the model key
    changed = OmegaConf.to_container(suffix)
    del changed["data"]["dataset"]["speaker_tag_placement"], changed["data"]["dataset"]["turn_start_token"]
    del changed["model"]["speaker_tokens"]["turn_start_token"]
    # The manifest paths are left for the command line.
    paths = ("data.train_ds.manifest_filepath", "data.validation_ds.datasets.val_set_0.manifest_filepath")
    assert [OmegaConf.select(suffix, path) for path in paths] == [None, None]
    expected = OmegaConf.to_container(prefix)
    expected["data"]["train_ds"]["manifest_filepath"] = None
    expected["data"]["validation_ds"]["datasets"]["val_set_0"]["manifest_filepath"] = None
    assert changed == expected
    # The prefix recipe keeps the default placement and has no turn-start token.
    assert "speaker_tag_placement" not in prefix.data.dataset and "turn_start_token" not in prefix.data.dataset

    for recipe, placement, token in ((prefix, "prefix", None), (suffix, "suffix", "<|turn_start|>")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)  # no deprecated key
            data = to_dataclass(StreamingSTTDataConfig, recipe.data.dataset)
            model = SimpleNamespace(core_cfg=to_dataclass(StreamingSTTModelConfig, recipe.model))
        assert (data.speaker_tag_placement, data.turn_start_token) == (placement, token)
        StreamingSTTModel._assert_speaker_config_matches_data(model, recipe.data.dataset)
