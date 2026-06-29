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
import json

import torch

from nemo.collections.speechlm2.parts import hf_hub


def test_normalize_torch_dtype_name_accepts_prefixed_strings():
    assert hf_hub._normalize_torch_dtype_name(torch.float32) == "float32"
    assert hf_hub._normalize_torch_dtype_name("float32") == "float32"
    assert hf_hub._normalize_torch_dtype_name("torch.float32") == "float32"


def test_checkpoint_state_dict_maps_activation_checkpoint_wrappers():
    wrapped = torch.nn.Parameter(torch.tensor([1.0]))
    ignored = torch.nn.Parameter(torch.tensor([2.0]))

    state_dict = hf_hub._checkpoint_state_dict(
        {
            "llm.layers.0._checkpoint_wrapped_module.norm.weight": wrapped,
            "runtime_only.weight": ignored,
        },
        {"llm.layers.0.norm.weight"},
    )

    assert list(state_dict) == ["llm.layers.0.norm.weight"]
    assert state_dict["llm.layers.0.norm.weight"] is wrapped


def test_distributed_loader_forwards_nested_automodel_controls(monkeypatch, tmp_path):
    config_path = tmp_path / hf_hub.CONFIG_NAME
    config_path.write_text(
        json.dumps(
            {
                "pretrained_weights": True,
                "use_nemo_automodel": True,
            }
        ),
        encoding="utf-8",
    )
    weight_path = tmp_path / hf_hub.SAFETENSORS_SINGLE_FILE
    weight_path.touch()

    def fake_cached_file(_model_id, filename, **_kwargs):
        if filename == hf_hub.CONFIG_NAME:
            return str(config_path)
        if filename == hf_hub.SAFETENSORS_SINGLE_FILE:
            return str(weight_path)
        raise AssertionError(filename)

    monkeypatch.setattr(hf_hub, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        hf_hub,
        "_load_state_dict_with_dtensors",
        lambda model, weight_dir: setattr(model, "loaded_from", weight_dir),
    )

    class FakeSALM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.configure_kwargs = None

        def configure_model(self, **kwargs):
            self.configure_kwargs = kwargs

    device_mesh = object()
    moe_mesh = object()
    distributed_config = object()
    moe_config = object()
    backend = object()

    model = hf_hub.HFHubMixin._from_pretrained.__func__(
        FakeSALM,
        model_id="checkpoint",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        moe_config=moe_config,
        activation_checkpointing=True,
        backend=backend,
        sdpa_method=["math"],
        use_liger_kernel=False,
        torch_dtype=torch.bfloat16,
        architectures=["NemotronHForCausalLM"],
    )

    assert model.configure_kwargs == {
        "device_mesh": device_mesh,
        "distributed_config": distributed_config,
        "moe_config": moe_config,
        "moe_mesh": moe_mesh,
        "activation_checkpointing_llm": True,
        "activation_checkpointing_perception": None,
        "llm_automodel_kwargs": {
            "backend": backend,
            "sdpa_method": ["math"],
            "use_liger_kernel": False,
        },
    }
    assert model.cfg["torch_dtype"] == "bfloat16"
    assert model.loaded_from == str(tmp_path)
