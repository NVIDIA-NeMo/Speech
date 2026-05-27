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
"""Tests for SALM + Automodel LoRA integration."""
import os
import re

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.speechlm2.data import SALMDataset
from nemo.collections.speechlm2.models import SALMAutomodel
from nemo.collections.speechlm2.parts.automodel_lora import (
    LORA_PARAM_PATTERN,
    ensure_lora_trainable,
    make_peft_config,
    maybe_install_lora,
)

if torch.cuda.is_available():
    torch.set_default_device('cuda')


def resolve_pretrained_models():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/Qwen--Qwen3-1.7B",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo",
        }
    else:
        return {
            "pretrained_asr": "nvidia/canary-1b-flash",
            "pretrained_llm": "Qwen/Qwen3-1.7B",
        }


AUDIO_LOCATOR_TAG = "<|audioplaceholder|>"
PROMPT = "qwen"

LORA_CFG = {
    "dim": 8,
    "alpha": 16,
    "dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
}

BASE_CFG = {
    **resolve_pretrained_models(),
    "pretrained_weights": False,
    "prompt_format": PROMPT,
    "audio_locator_tag": AUDIO_LOCATOR_TAG,
    "freeze_params": ["^llm\\..+$", "^perception\\.preprocessor\\..+$", "^perception\\.encoder\\..+$"],
    "prevent_freeze_params": [],
    "lora": LORA_CFG,
    "perception": {
        "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
        "output_dim": 2048,
        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "att_context_size": [-1, -1],
            "causal_downsampling": False,
            "conv_context_size": None,
            "conv_kernel_size": 9,
            "conv_norm_type": "batch_norm",
            "d_model": 1024,
            "dropout": 0.1,
            "dropout_att": 0.1,
            "dropout_emb": 0.0,
            "dropout_pre_encoder": 0.1,
            "feat_in": 128,
            "feat_out": -1,
            "ff_expansion_factor": 4,
            "n_heads": 8,
            "n_layers": 2,
            "pos_emb_max_len": 5000,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
        },
        "modality_adapter": {
            "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
            "d_model": 1024,
        },
        "preprocessor": {
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "dither": 1e-05,
            "features": 128,
            "frame_splicing": 1,
            "log": True,
            "n_fft": 512,
            "normalize": "per_feature",
            "pad_to": 0,
            "pad_value": 0.0,
            "sample_rate": 16000,
            "window": "hann",
            "window_size": 0.025,
            "window_stride": 0.01,
        },
    },
    "optimizer": {"_target_": "torch.optim.AdamW"},
    "torch_dtype": "bfloat16",
}


# ---------------------------------------------------------------------------
# Unit tests for automodel_lora helpers (no model needed)
# ---------------------------------------------------------------------------


class TestMakePeftConfig:
    def test_basic(self):
        cfg = DictConfig({"dim": 32, "alpha": 64, "target_modules": ["q_proj"]})
        pc = make_peft_config(cfg)
        assert pc.dim == 32
        assert pc.alpha == 64
        assert pc.target_modules == ["*.q_proj"]

    def test_defaults(self):
        cfg = DictConfig({})
        pc = make_peft_config(cfg)
        assert pc is None

    def test_all_fields(self):
        cfg = DictConfig(
            {
                "dim": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "exclude_modules": ["lm_head"],
                "match_all_linear": False,
                "use_dora": True,
                "dropout_position": "pre",
                "lora_A_init": "kaiming_uniform",
                "use_triton": False,
            }
        )
        pc = make_peft_config(cfg)
        assert pc.dim == 16
        assert pc.use_dora is True
        assert pc.dropout_position == "pre"
        assert pc.lora_A_init == "kaiming_uniform"
        assert pc.exclude_modules == ["*.lm_head"]

    def test_short_names_get_wildcard_prefix(self):
        """Short leaf names like 'q_proj' should be auto-prefixed with '*.' so
        that automodel's ModuleMatcher matches them against full dotted paths."""
        cfg = DictConfig({"target_modules": ["q_proj", "v_proj"], "exclude_modules": ["lm_head"]})
        pc = make_peft_config(cfg)
        assert pc.target_modules == ["*.q_proj", "*.v_proj"]
        assert pc.exclude_modules == ["*.lm_head"]

    def test_already_qualified_names_unchanged(self):
        """Names that already contain '*' or '.' should not be double-prefixed."""
        cfg = DictConfig({"target_modules": ["*.q_proj", "model.layers.*.self_attn.v_proj"]})
        pc = make_peft_config(cfg)
        assert pc.target_modules == ["*.q_proj", "model.layers.*.self_attn.v_proj"]


class TestEnsureLoraTrainable:
    def _model_stub(self, cfg_dict):
        class Stub:
            pass

        s = Stub()
        s.cfg = DictConfig(cfg_dict)
        return s

    def test_creates_prevent_freeze_params(self):
        model = self._model_stub({})
        ensure_lora_trainable(model)
        assert LORA_PARAM_PATTERN in model.cfg.prevent_freeze_params

    def test_appends_to_existing(self):
        model = self._model_stub({"prevent_freeze_params": ["^some_other$"]})
        ensure_lora_trainable(model)
        assert len(model.cfg.prevent_freeze_params) == 2
        assert LORA_PARAM_PATTERN in model.cfg.prevent_freeze_params

    def test_idempotent(self):
        model = self._model_stub({"prevent_freeze_params": []})
        ensure_lora_trainable(model)
        ensure_lora_trainable(model)
        assert model.cfg.prevent_freeze_params.count(LORA_PARAM_PATTERN) == 1


class TestLoraParamPattern:
    """Verify the regex matches typical LoRA parameter names."""

    @pytest.mark.parametrize(
        "name",
        [
            "llm.model.layers.0.self_attn.q_proj.lora_A.weight",
            "llm.model.layers.0.self_attn.v_proj.lora_B.weight",
            "llm.model.layers.31.mlp.gate_proj.lora_A.weight",
        ],
    )
    def test_matches_lora_params(self, name):
        assert re.compile(LORA_PARAM_PATTERN).match(name)

    @pytest.mark.parametrize(
        "name",
        [
            "llm.model.layers.0.self_attn.q_proj.weight",
            "perception.encoder.layers.0.weight",
            "llm.lm_head.weight",
        ],
    )
    def test_does_not_match_base_params(self, name):
        assert re.compile(LORA_PARAM_PATTERN).match(name) is None


# ---------------------------------------------------------------------------
# Integration tests (require CUDA — Automodel's from_config needs a GPU)
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Automodel requires CUDA")


@pytest.fixture(scope="module")
def salm_with_lora():
    if not torch.cuda.is_available():
        pytest.skip("Automodel requires CUDA")
    model = SALMAutomodel(BASE_CFG)
    model.configure_model()
    model.to("cuda")
    return model


@pytest.fixture(scope="module")
def salm_without_lora():
    if not torch.cuda.is_available():
        pytest.skip("Automodel requires CUDA")
    cfg = {k: v for k, v in BASE_CFG.items() if k != "lora"}
    model = SALMAutomodel(cfg)
    model.configure_model()
    model.to("cuda")
    return model


@pytest.fixture(scope="module")
def dataset(salm_with_lora):
    return SALMDataset(salm_with_lora.tokenizer)


@pytest.fixture(scope="module")
def prompt_formatter(salm_with_lora):
    return PromptFormatter.resolve(PROMPT)(salm_with_lora.tokenizer)


@pytest.fixture(scope="module")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id, recording_id=cut.recording_id, start=0, duration=1.0, text='Some text transcription.'
        )
    ]
    return CutSet(
        [
            NeMoMultimodalConversation(
                id="example-0",
                turns=[
                    TextTurn(role="user", value="Repeat after me:"),
                    AudioTurn(role="user", cut=cut, audio_locator_tag=AUDIO_LOCATOR_TAG),
                    TextTurn(role="assistant", value=cut.supervisions[0].text),
                ],
                token_equivalent_duration=0.08,
            )
        ]
    )


@requires_cuda
def test_lora_params_exist(salm_with_lora):
    """After configure_model with lora config, LLM should have lora_A/lora_B parameters."""
    lora_params = {n for n, _ in salm_with_lora.named_parameters() if ".lora_" in n}
    assert len(lora_params) > 0, "No LoRA parameters found"
    # We targeted q_proj and v_proj, so each should have lora_A.weight and lora_B.weight
    assert any("q_proj.lora_A" in n for n in lora_params)
    assert any("q_proj.lora_B" in n for n in lora_params)
    assert any("v_proj.lora_A" in n for n in lora_params)
    assert any("v_proj.lora_B" in n for n in lora_params)


@requires_cuda
def test_no_lora_params_without_config(salm_without_lora):
    """Without lora config, no LoRA parameters should exist."""
    lora_params = [n for n, _ in salm_without_lora.named_parameters() if ".lora_" in n]
    assert len(lora_params) == 0


@requires_cuda
def test_lora_prevent_freeze_pattern_set(salm_with_lora):
    """The prevent_freeze_params list should include the LoRA pattern."""
    assert LORA_PARAM_PATTERN in salm_with_lora.cfg.prevent_freeze_params


@requires_cuda
def test_lora_params_are_trainable(salm_with_lora):
    """LoRA parameters should have requires_grad=True."""
    for name, param in salm_with_lora.named_parameters():
        if ".lora_" in name:
            assert param.requires_grad, f"LoRA param {name} is not trainable"


@requires_cuda
def test_base_llm_params_not_patched_elsewhere(salm_with_lora):
    """Only targeted modules (q_proj, v_proj) should have LoRA, not others like k_proj."""
    lora_params = {n for n, _ in salm_with_lora.named_parameters() if ".lora_" in n}
    for n in lora_params:
        assert "q_proj" in n or "v_proj" in n, f"Unexpected LoRA param on non-targeted module: {n}"


@requires_cuda
def test_lora_training_step(salm_with_lora, dataset, prompt_formatter, training_cutset_batch):
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=salm_with_lora.device)
    results = salm_with_lora.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


@requires_cuda
def test_lora_generation(salm_with_lora):
    answer = salm_with_lora.generate(
        prompts=[
            [{"role": "user", "slots": {"message": f"Repeat after me: {AUDIO_LOCATOR_TAG}"}}],
        ],
        audios=torch.randn(1, 16000),
        audio_lens=torch.tensor([16000]),
        max_new_tokens=4,
    )
    assert answer.shape == (1, 4)
    assert answer.dtype == torch.long


@requires_cuda
def test_lora_match_all_linear():
    """Test match_all_linear=True applies LoRA to all linear layers in the LLM."""
    cfg = {**BASE_CFG}
    cfg["lora"] = {"dim": 8, "alpha": 16, "match_all_linear": True}
    model = SALMAutomodel(cfg)
    model.configure_model()

    lora_params = {n for n, _ in model.named_parameters() if ".lora_" in n}
    # Should have LoRA on more than just q_proj/v_proj
    modules_with_lora = {n.rsplit(".lora_", 1)[0] for n in lora_params}
    assert len(modules_with_lora) > 2, f"match_all_linear should patch many modules, got: {modules_with_lora}"
