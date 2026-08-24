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

import pytest

from scripts.speechlm2.convert_dflash_to_dflash2 import (
    build_dflash2_config,
    convert_checkpoint,
    dflash2_tensor_shapes,
    initialize_dflash2_tensors,
)


@pytest.fixture
def lightning_dflash_config():
    return {
        "architectures": ["DFlashDraftModel"],
        "hidden_size": 2688,
        "num_hidden_layers": 6,
        "num_target_layers": 52,
        "vocab_size": 131072,
        "dflash_config": {
            "causal": False,
            "mask_token_id": 990,
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
        },
        "quantization_config": {
            "ignore": ["*embed_tokens*", "*self_attn*"],
            "exclude_modules": ["*embed_tokens*", "*self_attn*"],
        },
    }


@pytest.fixture
def tiny_dflash_checkpoint(tmp_path, lightning_dflash_config):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source"
    source.mkdir()
    config = {
        **lightning_dflash_config,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "dflash_config": {
            **lightning_dflash_config["dflash_config"],
            "target_layer_ids": [1],
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"exclude_modules": ["*embed_tokens*", "*self_attn*"]}})
    )
    (source / "mask_embedding.pt").write_bytes(b"test mask embedding")
    safetensors.save_file({"norm.weight": torch.arange(8, dtype=torch.bfloat16)}, source / "model.safetensors")
    return source


def test_build_config_is_lightning_compatible(lightning_dflash_config):
    converted = build_dflash2_config(lightning_dflash_config)

    assert converted["architectures"] == ["DFlash2DraftModel"]
    assert "num_target_layers" not in converted
    assert converted["is_causal"] is False
    assert converted["dflash_config"] == {
        "causal": False,
        "mask_token_id": 990,
        "target_layer_ids": [1, 5, 19, 29, 41, 51],
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "selector_rank": 256,
        "selector_top_k": 16,
    }
    expected_exclusions = [
        "*embed_tokens*",
        "*self_attn*",
        "*attention_conv*",
        "*mlp_conv*",
        "*candidate_selector*",
    ]
    assert converted["quantization_config"]["ignore"] == expected_exclusions
    assert converted["quantization_config"]["exclude_modules"] == expected_exclusions


def test_build_config_preserves_layer_type_causality(lightning_dflash_config):
    del lightning_dflash_config["dflash_config"]["causal"]
    lightning_dflash_config["layer_types"] = ["sliding_attention", "full_attention"]

    converted = build_dflash2_config(lightning_dflash_config)

    assert "is_causal" not in converted
    assert converted["layer_types"] == ["sliding_attention", "full_attention"]


def test_build_config_normalizes_top_level_target_layers(lightning_dflash_config):
    target_layer_ids = lightning_dflash_config["dflash_config"].pop("target_layer_ids")
    lightning_dflash_config["target_layer_ids"] = target_layer_ids

    converted = build_dflash2_config(lightning_dflash_config)

    assert converted["dflash_config"]["target_layer_ids"] == target_layer_ids


def test_tensor_shapes_match_vllm_dflash2(lightning_dflash_config):
    shapes = dflash2_tensor_shapes(build_dflash2_config(lightning_dflash_config))

    assert len(shapes) == 27
    assert shapes["layers.0.attention_conv.base_kernel"] == (2, 2, 2688)
    assert shapes["layers.5.mlp_conv.kernel_projection.weight"] == (672, 2688)
    assert shapes["candidate_selector.predecessor_codebook"] == (131072, 256)
    assert shapes["candidate_selector.successor_codebook"] == (131072, 256)
    assert shapes["candidate_selector.hidden_projection.weight"] == (256, 2688)


def test_tensor_shapes_match_installed_vllm_modules(monkeypatch, lightning_dflash_config):
    torch = pytest.importorskip("torch")
    linear = pytest.importorskip("vllm.model_executor.layers.linear")
    parameter = pytest.importorskip("vllm.model_executor.parameter")
    dflash2 = pytest.importorskip("vllm.model_executor.models.qwen3_dflash2")
    vllm_config = pytest.importorskip("vllm.config")

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)

    config = {
        **lightning_dflash_config,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "vocab_size": 32,
    }
    config = build_dflash2_config(config, conv_group_size=4, selector_rank=2, selector_top_k=4)
    with vllm_config.set_current_vllm_config(vllm_config.VllmConfig()):
        conv = dflash2.DFlashGroupedConv(
            hidden_size=8,
            taps=2,
            group_size=4,
            block_size=3,
            params_dtype=torch.bfloat16,
            prefix="attention_conv",
        )
        selector = dflash2.CandidateSelector(
            hidden_size=8,
            vocab_size=32,
            rank=2,
            top_k=4,
            params_dtype=torch.bfloat16,
            prefix="candidate_selector",
        )

    actual = {
        **{f"layers.0.attention_conv.{name}": tuple(param.shape) for name, param in conv.named_parameters()},
        **{f"candidate_selector.{name}": tuple(param.shape) for name, param in selector.named_parameters()},
    }
    expected = dflash2_tensor_shapes(config)
    covered_expected = {
        name for name in expected if name.startswith(("layers.0.attention_conv.", "candidate_selector."))
    }
    assert set(actual) == covered_expected
    for name, shape in actual.items():
        assert shape == expected[name]


def test_identity_convolutions_and_neutral_selector(lightning_dflash_config):
    torch = pytest.importorskip("torch")
    tensors = initialize_dflash2_tensors(build_dflash2_config(lightning_dflash_config))

    base = tensors["layers.0.attention_conv.base_kernel"]
    torch.testing.assert_close(base[:, 0], torch.ones_like(base[:, 0]))
    torch.testing.assert_close(base[:, 1], torch.zeros_like(base[:, 1]))
    assert not tensors["layers.0.attention_conv.kernel_projection.weight"].count_nonzero()
    assert not tensors["candidate_selector.predecessor_codebook"].count_nonzero()
    assert not tensors["candidate_selector.successor_codebook"].count_nonzero()
    assert not tensors["candidate_selector.hidden_projection.weight"].count_nonzero()


def test_convert_checkpoint_writes_loadable_artifact(tmp_path, tiny_dflash_checkpoint):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    original = safetensors.load_file(tiny_dflash_checkpoint / "model.safetensors")["norm.weight"]

    output = convert_checkpoint(
        str(tiny_dflash_checkpoint),
        tmp_path / "output",
        conv_group_size=4,
        selector_rank=2,
        selector_top_k=4,
    )

    output_config = json.loads((output / "config.json").read_text())
    output_weights = safetensors.load_file(output / "model.safetensors")
    manifest = json.loads((output / "dflash2_bootstrap.json").read_text())
    hf_quant_config = json.loads((output / "hf_quant_config.json").read_text())
    assert output_config["architectures"] == ["DFlash2DraftModel"]
    torch.testing.assert_close(output_weights["norm.weight"], original)
    assert output_weights["layers.0.attention_conv.kernel_projection.weight"].shape == (8, 8)
    assert manifest["trained_dflash2_parameters"] is False
    assert (output / "hf_quant_config.json").is_file()
    assert (output / "mask_embedding.pt").read_bytes() == b"test mask embedding"
    assert hf_quant_config["quantization"]["exclude_modules"] == [
        "*embed_tokens*",
        "*self_attn*",
        "*attention_conv*",
        "*mlp_conv*",
        "*candidate_selector*",
    ]
    assert output.stat().st_mode & 0o777 == 0o755
    assert (output / "model.safetensors").stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"conv_group_size": 17}, "must divide hidden_size"),
        ({"selector_top_k": 131073}, "cannot exceed vocab_size"),
        ({"conv_kernel_size": 0}, "must be a positive integer"),
    ],
)
def test_rejects_incompatible_dimensions(lightning_dflash_config, kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_dflash2_config(lightning_dflash_config, **kwargs)


def test_rejects_non_dflash_source(lightning_dflash_config):
    lightning_dflash_config["architectures"] = ["DFlash2DraftModel"]

    with pytest.raises(ValueError, match="must declare DFlashDraftModel"):
        build_dflash2_config(lightning_dflash_config)


def test_rejects_existing_output(tmp_path, tiny_dflash_checkpoint):
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Output already exists"):
        convert_checkpoint(str(tiny_dflash_checkpoint), output, conv_group_size=4, selector_rank=2)


def test_rejects_multiple_weight_files(tmp_path, tiny_dflash_checkpoint):
    (tiny_dflash_checkpoint / "second.safetensors").write_bytes(b"not read")

    with pytest.raises(ValueError, match="expects one safetensors file"):
        convert_checkpoint(
            str(tiny_dflash_checkpoint),
            tmp_path / "output",
            conv_group_size=4,
            selector_rank=2,
        )


def test_rejects_existing_dflash2_tensors(tmp_path, tiny_dflash_checkpoint):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file(
        {
            "norm.weight": torch.arange(8, dtype=torch.bfloat16),
            "layers.0.attention_conv.base_kernel": torch.zeros((2, 2, 8), dtype=torch.bfloat16),
        },
        tiny_dflash_checkpoint / "model.safetensors",
    )

    with pytest.raises(ValueError, match="already contains DFlash2 tensors"):
        convert_checkpoint(
            str(tiny_dflash_checkpoint),
            tmp_path / "output",
            conv_group_size=4,
            selector_rank=2,
        )
