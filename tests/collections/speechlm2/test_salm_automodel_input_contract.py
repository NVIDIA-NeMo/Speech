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
"""SpeechLM2 embedding-only input contracts and packed GPU execution."""

import os

import pytest
import torch
import torch.nn.functional as F
from lightning import LightningModule
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel


@pytest.mark.parametrize(
    ("model_kind", "config_cls", "model_cls_name"),
    [
        ("llama", LlamaConfig, "LlamaForCausalLM"),
        ("qwen2", Qwen2Config, "Qwen2ForCausalLM"),
        ("qwen3", Qwen3Config, "Qwen3ForCausalLM"),
    ],
)
@pytest.mark.parametrize("layout", ["padded", "thd", "thd_with_batch_dimension"])
def test_salm_automodel_accepts_precomputed_embeddings(layout, model_kind, config_cls, model_cls_name):
    native_model = pytest.importorskip(f"nemo_automodel.components.models.{model_kind}.model")
    from nemo_automodel.components.models.common import BackendConfig

    # Zero decoder layers isolate native model input validation and THD
    # normalization from optional TE attention kernels. The LM head and
    # embeddings-to-logits backward path still run normally.
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=0,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", rope_fusion=False)
    model = SALMAutomodel.__new__(SALMAutomodel)
    LightningModule.__init__(model)
    model.llm = getattr(native_model, model_cls_name)(config, backend=backend)
    model._fused_linear_cross_entropy = None

    shape = (4, 16) if layout == "thd" else (1, 4, 16)
    inputs = torch.randn(*shape, requires_grad=True)
    kwargs = {}
    if layout != "padded":
        kwargs = {
            "qkv_format": "thd",
            "position_ids": torch.tensor([0, 1, 0, 1]),
            "cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32),
            "max_seqlen": 2,
        }

    logits = model(input_embeds=inputs, **kwargs)["logits"]
    assert logits.shape == (1, 4, 32)
    logits.float().square().sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.abs().sum() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_batch_dimension", [False, True])
def test_salm_automodel_qwen3_packed_gpu_matches_independent_sequences(with_batch_dimension, monkeypatch):
    """Real TE THD attention preserves logits and audio/text gradients."""
    from nemo.collections.speechlm2.parts.parallel import (
        installed_version,
        packed_attention_requires_flash_attention,
        validate_parallelism_compatibility,
    )

    device_capability = torch.cuda.get_device_capability()
    if packed_attention_requires_flash_attention(
        packed_sequences=True, attn_backend="te", device_capability=device_capability
    ):
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    try:
        # Skip TE / GPU combinations that SALMAutomodel training rejects as numerically broken.
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn=os.environ.get("NVTE_FUSED_ATTN"),
            device_capability=device_capability,
            te_version=installed_version("transformer-engine"),
            flash_attn_4_version=installed_version("flash-attn-4"),
        )
    except ValueError as error:
        pytest.skip(f"Unsupported packed attention environment: {error}")
    te = pytest.importorskip("transformer_engine.pytorch")
    qwen3 = pytest.importorskip("nemo_automodel.components.models.qwen3.model")
    from nemo_automodel.components.models.common import BackendConfig

    from nemo.collections.speechlm2.parts.packed_sequences import prepare_packed_llm_inputs

    torch.manual_seed(16366)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", rope_fusion=False)
    model = SALMAutomodel.__new__(SALMAutomodel)
    LightningModule.__init__(model)
    model.llm = qwen3.Qwen3ForCausalLM(config, backend=backend).to(device="cuda", dtype=torch.bfloat16).train()
    model._fused_linear_cross_entropy = None
    assert isinstance(model.llm.model.layers[0].self_attn.attn_module, te.DotProductAttention)

    input_ids = torch.tensor([[0, 31, 1, 2, 3], [31, 4, 5, 6, 7]], device="cuda")
    targets = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, 6, 7]], device="cuda")
    audio = [torch.randn(length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for length in (5, 7)]

    def prepare():
        return prepare_packed_llm_inputs(
            input_ids=input_ids,
            text_embs=None,
            audio_embs=audio,
            target_ids=targets,
            padding_id=0,
            placeholder_id=31,
            embed_tokens=model.llm.get_input_embeddings(),
        )

    packed = prepare()
    embeddings = packed["input_embeds"]
    if with_batch_dimension:
        embeddings = embeddings.unsqueeze(0)
    logits = model(input_embeds=embeddings, **packed["llm_kwargs"])["logits"]
    labels = packed["target_ids"].reshape(-1)
    F.cross_entropy(logits.float().reshape(-1, 32), labels, reduction="sum").backward()
    packed_grads = {name: param.grad.clone() for name, param in model.llm.named_parameters() if param.grad is not None}
    packed_audio_grads = [value.grad.clone() for value in audio]

    model.llm.zero_grad(set_to_none=True)
    for value in audio:
        value.grad = None
    reference_inputs = prepare()
    offsets = reference_inputs["llm_kwargs"]["cu_seqlens"].tolist()
    reference_logits = torch.cat(
        [
            model.llm(inputs_embeds=reference_inputs["input_embeds"][start:end].unsqueeze(0)).logits
            for start, end in zip(offsets, offsets[1:])
        ],
        dim=1,
    )
    F.cross_entropy(reference_logits.float().reshape(-1, 32), labels, reduction="sum").backward()

    # BF16 fused THD and per-document SDPA reductions may use different orders.
    torch.testing.assert_close(logits, reference_logits, atol=5e-3, rtol=2e-2)
    reference_grads = {name: param.grad for name, param in model.llm.named_parameters() if param.grad is not None}
    assert packed_grads.keys() == reference_grads.keys()
    for name, expected in reference_grads.items():
        torch.testing.assert_close(packed_grads[name], expected, atol=2e-2, rtol=2e-2)
    for actual, value in zip(packed_audio_grads, audio):
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, value.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("precomputed_mtp_embeddings", [False, True])
@pytest.mark.parametrize("with_batch_dimension", [False, True])
def test_salm_automodel_nemotron_h_mtp_packed_gpu_without_input_ids(with_batch_dimension, precomputed_mtp_embeddings):
    """Nemotron-H THD packing, including MTP heads, runs from embeddings alone."""
    pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("mamba_ssm")
    nemotron = pytest.importorskip("nemo_automodel.components.models.nemotron_v3.model")
    from nemo_automodel.components.models.common import BackendConfig
    from transformers import NemotronHConfig

    from nemo.collections.speechlm2.parts.packed_sequences import prepare_packed_llm_inputs

    torch.manual_seed(16366)
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        hybrid_override_pattern="M*-",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        mamba_num_heads=4,
        mamba_head_dim=32,
        ssm_state_size=16,
        n_groups=1,
        mamba_chunk_size=16,
        attention_dropout=0.0,
        use_cache=False,
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*",
    )
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", rope_fusion=False)
    model = SALMAutomodel.__new__(SALMAutomodel)
    LightningModule.__init__(model)
    model.llm = nemotron.NemotronHForCausalLM(config, backend=backend).to(device="cuda", dtype=torch.bfloat16).train()
    model._fused_linear_cross_entropy = None
    assert model.llm.mtp is not None

    input_ids = torch.tensor([[0, 31, 1, 2, 3], [31, 4, 5, 6, 7]], device="cuda")
    targets = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, 6, 7]], device="cuda")
    audio = [torch.randn(length, 128, device="cuda", dtype=torch.bfloat16) for length in (5, 7)]
    packed = prepare_packed_llm_inputs(
        input_ids=input_ids,
        text_embs=None,
        audio_embs=audio,
        target_ids=targets,
        padding_id=0,
        placeholder_id=31,
        embed_tokens=model.llm.get_input_embeddings(),
    )
    embeddings = packed["input_embeds"].detach().requires_grad_(True)
    num_tokens = embeddings.shape[0]
    llm_kwargs = dict(packed["llm_kwargs"])
    if precomputed_mtp_embeddings:
        # Context-parallel training passes globally shifted MTP embeddings positionally.
        llm_kwargs["mtp_embed_inputs"] = (torch.roll(embeddings, shifts=-1, dims=0),)
    inputs = embeddings.unsqueeze(0) if with_batch_dimension else embeddings

    out = model(input_embeds=inputs, **llm_kwargs)
    assert out["logits"].shape == (1, num_tokens, 32)
    assert len(out["mtp_per_depth_h"]) == 1
    (out["logits"].float().square().mean() + out["mtp_per_depth_h"][0].float().square().mean()).backward()
    assert torch.isfinite(embeddings.grad).all() and embeddings.grad.abs().sum() > 0

    offsets = packed["llm_kwargs"]["cu_seqlens"].tolist()
    with torch.no_grad():
        reference_logits = torch.cat(
            [
                model.llm(inputs_embeds=embeddings[start:end].unsqueeze(0)).logits
                for start, end in zip(offsets, offsets[1:])
            ],
            dim=1,
        )
    # BF16 THD kernels (TE attention, Mamba varlen scan) vs per-document BSHD reductions.
    torch.testing.assert_close(out["logits"].detach(), reference_logits, atol=2e-2, rtol=2e-2)
