# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpeechLM2 embedding-only input contracts and packed GPU execution."""

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_batch_dimension", [False, True])
def test_salm_automodel_qwen3_packed_gpu_matches_independent_sequences(with_batch_dimension, monkeypatch):
    """Real TE THD attention preserves logits and audio/text gradients."""
    if torch.cuda.get_device_capability() == (12, 0):
        pytest.importorskip(
            "flash_attn", reason="SM120 THD backward requires the documented FlashAttention workaround"
        )
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
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
    torch.testing.assert_close(logits, reference_logits, atol=2e-3, rtol=2e-2)
    reference_grads = {name: param.grad for name, param in model.llm.named_parameters() if param.grad is not None}
    assert packed_grads.keys() == reference_grads.keys()
    for name, expected in reference_grads.items():
        torch.testing.assert_close(packed_grads[name], expected, atol=2e-3, rtol=2e-2)
    for actual, value in zip(packed_audio_grads, audio):
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, value.grad, atol=2e-3, rtol=2e-2)
