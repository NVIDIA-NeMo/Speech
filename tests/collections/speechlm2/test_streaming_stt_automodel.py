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
"""Tests for :class:`StreamingSTTModelAutomodel`.

``nemo_automodel`` is an optional dependency, so these tests stub the single
entry point that needs it (``load_pretrained_automodel_llm``) with a small
HuggingFace LLM. Everything else — deferred construction, the vocabulary
reconciliation, the freeze policy, the embedding indirection and a real
training step — is exercised for real.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX, StreamingSTTBatch
from nemo.collections.speechlm2.models import StreamingSTTModelAutomodel
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

BLANK_TOKEN = "<blank>"
CHUNK_SIZE = 2
NUM_AUDIO_SLOTS = 8


def resolve_pretrained_llm() -> str:
    if os.path.exists("/home/TestData/speechlm/pretrained_models/Qwen--Qwen3-1.7B"):
        return "/home/TestData/speechlm/pretrained_models/Qwen--Qwen3-1.7B"
    return "Qwen/Qwen3-1.7B"


def make_cfg(**overrides) -> dict:
    """Model config with a tiny, offline-constructible perception module."""
    cfg = {
        "pretrained_llm": resolve_pretrained_llm(),
        "pretrained_asr": "unused-because-load_asr_weights-is-false",
        "load_llm_weights": False,
        "load_asr_weights": False,
        "use_nemo_automodel": True,
        "blank_token": BLANK_TOKEN,
        "chunk_size": CHUNK_SIZE,
        "att_context_size": [70, 1],
        "audio_pad_to": 0,
        "sample_rate": 16000,
        "frame_length_in_secs": 0.08,
        "dtype": "float32",
        "freeze_speech_encoder": False,
        "freeze_modality_adapter": False,
        "freeze_modality_proj": False,
        "freeze_llm_model": True,
        "freeze_llm_head": False,
        "freeze_embed_tokens": False,
        "freeze_params": [],
        "prevent_freeze_params": [],
        "perception": {
            "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
            "output_dim": 128,
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "att_context_size": [70, 1],
                "causal_downsampling": True,
                "conv_context_size": "causal",
                "conv_kernel_size": 9,
                "d_model": 64,
                "feat_in": 80,
                "feat_out": -1,
                "ff_expansion_factor": 4,
                "n_heads": 4,
                "n_layers": 2,
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
                "subsampling_conv_channels": 64,
                "subsampling_factor": 8,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                "d_model": 64,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "features": 80,
                "normalize": "per_feature",
                "sample_rate": 16000,
                "window_size": 0.025,
                "window_stride": 0.01,
            },
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-4},
    }
    cfg.update(overrides)
    return cfg


def tiny_llm_factory(monkeypatch):
    """Patch ``load_pretrained_automodel_llm`` with a small real HF LLM.

    The architecture and (crucially) ``vocab_size`` come from the real Qwen3
    config, so the spare-embedding-rows logic is exercised faithfully; only the
    layer sizes are shrunk.
    """
    import nemo.collections.speechlm2.models.streaming_stt_model_automodel as mod

    calls = {}

    def _fake_loader(model_path_or_name, pretrained_weights=True, dtype=torch.float32, **kwargs):
        calls["kwargs"] = kwargs
        calls["dtype"] = dtype
        calls["pretrained_weights"] = pretrained_weights
        config = AutoConfig.from_pretrained(model_path_or_name)
        config.num_hidden_layers = 2
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 32
        config.tie_word_embeddings = False
        return AutoModelForCausalLM.from_config(config, dtype=dtype)

    monkeypatch.setattr(mod, "load_pretrained_automodel_llm", _fake_loader)
    return calls


@pytest.fixture
def model(monkeypatch):
    calls = tiny_llm_factory(monkeypatch)
    m = StreamingSTTModelAutomodel(make_cfg())
    m.configure_model()
    m._automodel_calls = calls
    return m


def make_batch(model, batch_size: int = 2, n_text_tokens: int = 6) -> StreamingSTTBatch:
    """Synthetic batch: [audio slots ..., text tokens ...] with targets on text."""
    torch.manual_seed(0)
    total_len = NUM_AUDIO_SLOTS + n_text_tokens
    input_tokens = torch.full((batch_size, total_len), model.text_pad_id, dtype=torch.long)
    input_tokens[:, :NUM_AUDIO_SLOTS] = AUDIO_TOKEN_IDX
    text_ids = torch.randint(low=10, high=1000, size=(batch_size, n_text_tokens))
    input_tokens[:, NUM_AUDIO_SLOTS:] = text_ids

    target_tokens = torch.full((batch_size, total_len), IGNORE_INDEX, dtype=torch.long)
    target_tokens[:, NUM_AUDIO_SLOTS:] = text_ids

    n_samples = 16000  # 1 second
    audios = torch.randn(batch_size, n_samples) * 0.1
    return StreamingSTTBatch(
        audios=audios,
        audio_lens=torch.full((batch_size,), n_samples, dtype=torch.long),
        input_tokens=input_tokens,
        input_token_lens=torch.full((batch_size,), total_len, dtype=torch.long),
        target_tokens=target_tokens,
        target_token_lens=torch.full((batch_size,), total_len, dtype=torch.long),
        text=["dummy"] * batch_size,
        chunk_size=CHUNK_SIZE,
    )


# ===========================================================================
# Deferred construction
# ===========================================================================


def test_init_defers_llm_and_perception(monkeypatch):
    tiny_llm_factory(monkeypatch)
    m = StreamingSTTModelAutomodel(make_cfg())
    assert m.llm is None
    assert m.perception is None
    # The tokenizer is ready immediately (the DataModule needs it before fit).
    assert m.tokenizer is not None
    assert m.blank_token == BLANK_TOKEN
    assert m.blank_token_id > 0
    # ...and no embedding table exists yet.
    assert m.embed_tokens is None


def test_configure_model_builds_modules(model):
    assert model.llm is not None
    assert model.perception is not None
    # embed_tokens stays *inside* the LLM (unlike StreamingSTTModel).
    assert model.embed_tokens is model.llm.model.embed_tokens
    assert "embed_tokens" not in dict(model.named_children())
    assert any(k.startswith("llm.model.embed_tokens") for k in model.state_dict())


def test_configure_model_is_idempotent(model):
    llm_before = model.llm
    perception_before = model.perception
    model.configure_model()
    assert model.llm is llm_before
    assert model.perception is perception_before


def test_perception_output_dim_matches_llm(model):
    assert model.perception.proj.out_features == model.llm.config.hidden_size


def test_dtype_resolution_from_config(model):
    assert model._automodel_calls["dtype"] == torch.float32
    assert model._automodel_calls["pretrained_weights"] is False


# ===========================================================================
# Vocabulary reconciliation
# ===========================================================================


def test_added_special_token_fits_in_spare_rows(model):
    """Qwen3 ships 151936 embedding rows for a 151669-token vocab, so <blank>
    lands in a spare row and no resize is needed."""
    assert len(model.tokenizer.tokenizer) > 0
    assert model.blank_token_id < model.embed_tokens.weight.shape[0]
    assert model.text_vocab_size >= len(model.tokenizer.tokenizer)


def test_sync_vocab_grows_table_when_too_small(model):
    """When the checkpoint has no spare rows the table is grown, not shrunk."""
    target = len(model.tokenizer.tokenizer)
    model.llm.resize_token_embeddings(target - 5)
    assert model.embed_tokens.weight.shape[0] == target - 5
    model._sync_llm_vocab_size()
    assert model.embed_tokens.weight.shape[0] >= target


def test_sync_vocab_keeps_spare_rows(model):
    """A larger-than-needed table is left alone (the parent class would shrink it)."""
    rows_before = model.embed_tokens.weight.shape[0]
    assert rows_before > len(model.tokenizer.tokenizer)
    model._sync_llm_vocab_size()
    assert model.embed_tokens.weight.shape[0] == rows_before


# ===========================================================================
# Embedding indirection
# ===========================================================================


def test_embed_tokens_matches_module_call(model):
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.testing.assert_close(model._embed_tokens(ids), model.embed_tokens(ids))


def test_embed_ref_tensor_dtype(model):
    assert model._embed_ref_tensor.dtype == model.embed_tokens.weight.dtype


def test_move_embedding_ctx_is_a_noop(model):
    """The parent moves embed_tokens in/out of the LLM for generation; here it
    must stay put, otherwise the LLM loses its embedding table."""
    with model._move_embedding_ctx():
        assert model.llm.model.embed_tokens is not None
    assert model.llm.model.embed_tokens is not None


@pytest.fixture
def fake_dist():
    """Minimal env for the fake distributed backend used with LocalTensorMode."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "0")
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("requires_grad", [True, False])
def test_embed_tokens_gradient_is_reduced_across_dp_ranks(fake_dist, requires_grad):
    """A trainable embedding table is looked up OUTSIDE the LLM's FSDP2 forward,
    so FSDP2 never reduce-scatters its gradient. ``_embed_tokens`` must declare
    the gradient ``Partial`` so the redistribute back to ``Shard(0)`` performs
    the DP reduction — otherwise each rank keeps only its own micro-batch
    contribution and the ``<blank>`` embedding trains on 1/dp_size of the data.

    Rank 0 weighs its lookup by 1.0 and rank 1 by 3.0, so a correctly averaged
    gradient is 2.0 while an unreduced one is 1.0.
    """
    local_tensor_mode = pytest.importorskip(
        "torch.distributed._local_tensor", reason="LocalTensorMode requires PyTorch >= 2.10"
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    world_size = 2
    torch.distributed.init_process_group(backend="fake", rank=0, world_size=world_size)
    with local_tensor_mode.LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        weight = distribute_tensor(torch.zeros(8, 4), mesh, [Shard(0)]).requires_grad_(requires_grad)
        per_rank_scale = distribute_tensor(torch.tensor([1.0, 3.0]), mesh, [Shard(0)]).to_local()
        stub = SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))

        embedded = StreamingSTTModelAutomodel._embed_tokens(stub, torch.zeros(1, dtype=torch.long))
        assert embedded.shape == (1, 4)

        if not requires_grad:
            assert not embedded.requires_grad
            return
        (embedded * per_rank_scale).sum().backward()
        row0_grad = weight.grad.full_tensor()[0]
        torch.testing.assert_close(row0_grad, torch.full((4,), 2.0))


# ===========================================================================
# LLM output normalization
# ===========================================================================


class _TensorOutLLM(torch.nn.Module):
    """Stand-in for an Automodel backbone that returns bare logits."""

    def __init__(self, vocab=7, hidden=4):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, vocab)

    def forward(self, inputs_embeds=None, **kwargs):
        return self.proj(inputs_embeds)


def test_llm_forward_wraps_bare_tensor(model):
    model.llm = _TensorOutLLM()
    out = model._llm_forward(inputs_embeds=torch.randn(2, 3, 4))
    assert out.logits.shape == (2, 3, 7)
    assert out.past_key_values is None


def test_llm_forward_raises_without_cache(model):
    model.llm = _TensorOutLLM()
    with pytest.raises(RuntimeError, match="did not return a KV cache"):
        model._llm_forward(inputs_embeds=torch.randn(2, 3, 4), use_cache=True)


def test_llm_forward_raises_without_hidden_states(model):
    model.llm = _TensorOutLLM()
    with pytest.raises(RuntimeError, match="did not return hidden states"):
        model._llm_forward(inputs_embeds=torch.randn(2, 3, 4), output_hidden_states=True)


def test_llm_forward_passes_through_model_output(model):
    out = model._llm_forward(inputs_embeds=torch.randn(2, 3, model.llm.config.hidden_size))
    assert out.logits.shape[:2] == (2, 3)
    # Both mapping and attribute access work (the inherited code paths use both).
    torch.testing.assert_close(out["logits"], out.logits)


# ===========================================================================
# Freeze policy
# ===========================================================================


def test_freeze_llm_body_keeps_head_and_embeddings_trainable(model):
    assert all(not p.requires_grad for p in model.llm.model.layers.parameters())
    assert all(p.requires_grad for p in model.llm.lm_head.parameters())
    # embed_tokens lives inside llm.model, so the per-module switch must win.
    assert model.embed_tokens.weight.requires_grad
    assert all(p.requires_grad for p in model.perception.encoder.parameters())


def test_freeze_embed_tokens(monkeypatch):
    tiny_llm_factory(monkeypatch)
    m = StreamingSTTModelAutomodel(make_cfg(freeze_embed_tokens=True))
    m.configure_model()
    assert not m.embed_tokens.weight.requires_grad


# ===========================================================================
# Config validation
# ===========================================================================


def test_hf_peft_lora_keys_rejected(monkeypatch):
    tiny_llm_factory(monkeypatch)
    cfg = make_cfg(lora={"task_type": "CAUSAL_LM", "r": 128, "lora_alpha": 256, "lora_dropout": 0.01})
    with pytest.raises(ValueError, match="HuggingFace-PEFT keys"):
        StreamingSTTModelAutomodel(cfg)


class _FakeMesh:
    """Minimal stand-in for a DeviceMesh with per-dimension sizes."""

    def __init__(self, **sizes):
        self.mesh_dim_names = tuple(sizes)
        self._sizes = sizes

    def __getitem__(self, name):
        size = self._sizes[name]
        return SimpleNamespace(size=lambda: size)


def test_context_parallelism_rejected(model):
    model._device_mesh = _FakeMesh(dp=1, cp=2, tp=1)
    with pytest.raises(ValueError, match="context parallelism"):
        model._validate_parallelism_compatibility()


def test_tensor_parallelism_rejected(model):
    """TP shards lm_head's output; the inherited blank/non-blank loss breakdown
    indexes the per-token loss with a plain bool mask, which is not DTensor-safe."""
    model._device_mesh = _FakeMesh(dp=1, cp=1, tp=2)
    with pytest.raises(ValueError, match="tensor parallelism"):
        model._validate_parallelism_compatibility()


def test_data_parallel_only_mesh_accepted(model):
    model._device_mesh = _FakeMesh(dp=8, cp=1, tp=1)
    model._validate_parallelism_compatibility()  # must not raise


def test_configure_model_refuses_to_skip_sharding(model):
    """A model already built WITHOUT a mesh must not silently no-op when the
    strategy later calls configure_model() WITH one — that would train every
    rank unsharded and unsynchronized."""
    assert model._configured_with_mesh is False  # built by the fixture without a mesh
    with pytest.raises(RuntimeError, match="already ran without a device mesh"):
        model.configure_model(device_mesh=_FakeMesh(dp=2, cp=1, tp=1))


# ===========================================================================
# LoRA trainability (the freeze order that bit the sharded path)
# ===========================================================================


def _inject_fake_lora(model):
    """Attach lora_A/lora_B params the way Automodel does inside from_pretrained."""
    q_proj = model.llm.model.layers[0].self_attn.q_proj
    hidden = model.llm.config.hidden_size
    q_proj.lora_A = torch.nn.Linear(hidden, 4, bias=False)
    q_proj.lora_B = torch.nn.Linear(4, hidden, bias=False)
    return [q_proj.lora_A.weight, q_proj.lora_B.weight]


def test_unfreeze_lora_params_restores_grads(model):
    """`freeze_module(llm.model)` also freezes adapters installed by Automodel
    before the freeze; `prevent_freeze_params` alone cannot bring them back
    because freeze_and_subset skips requires_grad=False params first."""
    lora_params = _inject_fake_lora(model)
    model._apply_freeze_config()  # freeze_llm_model=True -> freezes the adapters too
    assert all(not p.requires_grad for p in lora_params)

    n = model._unfreeze_lora_params()
    assert n == len(lora_params)
    assert all(p.requires_grad for p in lora_params)


def test_lora_params_reach_the_optimizer(model):
    lora_params = _inject_fake_lora(model)
    model._apply_freeze_config()
    model.cfg.prevent_freeze_params = [r"^.+\.lora_.+$"]
    model._unfreeze_lora_params()

    optimizer = model.configure_optimizers()["optimizer"]
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimizer_params for p in lora_params)


# ===========================================================================
# End-to-end training step
# ===========================================================================


def test_training_step_produces_gradients(model):
    batch = make_batch(model)
    out = model.training_step(batch, batch_idx=0)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.perception.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert model.llm.lm_head.weight.grad is not None
    # The frozen LLM body must not accumulate gradients.
    assert all(p.grad is None for p in model.llm.model.layers.parameters())


def test_validation_step_runs(model):
    model.on_validation_epoch_start()
    model.validation_step({"val_set_0": make_batch(model)}, batch_idx=0)
    assert model._partial_val_losses["val_set_0"]
    assert torch.isfinite(model._partial_val_losses["val_set_0"][0])


def test_configure_optimizers_only_sees_trainable_params(model):
    ans = model.configure_optimizers()
    optimizer = ans["optimizer"]
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    frozen = [p for p in model.llm.model.layers.parameters()]
    assert optimizer_params, "optimizer received no parameters"
    assert all(id(p) not in optimizer_params for p in frozen)
    assert id(model.embed_tokens.weight) in optimizer_params


def test_oomptimizer_schema(model):
    schema = model.oomptimizer_schema
    assert schema["cls"] is StreamingSTTBatch
    names = {entry["name"] for entry in schema["inputs"]}
    assert {"input_tokens", "target_tokens", "audios", "audio_lens"} <= names


def test_hyperparameters_are_serializable(model):
    """PTL checkpoints store cfg as a plain dict."""
    assert isinstance(model.hparams["cfg"], dict)
    OmegaConf.create(model.hparams["cfg"])


# ===========================================================================
# Activation-checkpointing recompute under bf16-true
# ===========================================================================


def test_backward_replays_the_forward_default_dtype(model):
    """`precision: bf16-true` (Lightning HalfPrecision) sets the *global* default
    dtype around the forward only. Anything the LLM creates without an explicit
    dtype inside an activation-checkpointed block — e.g. the empty per-expert
    buffers a MoE layer builds when an expert receives zero tokens — is then bf16
    when saved and fp32 when recomputed, and torch raises CheckpointError.
    backward() must therefore run under the dtype the forward captured."""
    from lightning.fabric.plugins.precision.utils import _DtypeContextManager

    seen = {}

    class _Recorder(StreamingSTTModelAutomodel):
        def _setup_moe_fsdp_sync(self):
            pass

    model.__class__ = _Recorder
    # what Lightning's train_step_context() would have left behind
    with _DtypeContextManager(torch.bfloat16):
        model._forward_default_dtype = torch.get_default_dtype()
    assert model._forward_default_dtype == torch.bfloat16

    loss = (torch.ones(1, requires_grad=True) * 2).sum()
    original = StreamingSTTModel.backward

    def _capture(self, *args, **kwargs):
        seen["dtype"] = torch.get_default_dtype()

    StreamingSTTModel.backward = _capture
    try:
        assert torch.get_default_dtype() == torch.float32  # outside the forward ctx
        model.backward(loss)
    finally:
        StreamingSTTModel.backward = original

    assert seen["dtype"] == torch.bfloat16, "recompute would run at a different default dtype than the forward"
    assert torch.get_default_dtype() == torch.float32, "default dtype must be restored"


def test_backward_is_a_noop_when_forward_did_not_change_dtype(model):
    """With precision plugins that leave the default dtype alone (bf16-flash /
    FlashPrecision, fp32), nothing should be forced."""
    seen = {}
    model._forward_default_dtype = torch.float32
    original = StreamingSTTModel.backward

    def _capture(self, *args, **kwargs):
        seen["dtype"] = torch.get_default_dtype()

    StreamingSTTModel.backward = _capture
    try:
        model.backward((torch.ones(1, requires_grad=True) * 2).sum())
    finally:
        StreamingSTTModel.backward = original
    assert seen["dtype"] == torch.float32


# ===========================================================================
# Gradient-norm logging
# ===========================================================================


def _fake_optimizer(params):
    return SimpleNamespace(param_groups=[{"params": params}])


def test_grad_norm_is_logged(model, monkeypatch):
    """grad_norm is the earliest divergence signal: one inf gradient makes the
    global norm non-finite and clipping then scales every parameter by it."""
    logged = {}
    monkeypatch.setattr(type(model), "log", lambda self, name, value, **kw: logged.__setitem__(name, value))
    monkeypatch.setattr(StreamingSTTModel, "configure_gradient_clipping", lambda self, *a, **k: None, raising=False)
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm == 5

    model.configure_gradient_clipping(_fake_optimizer([p]), gradient_clip_val=1.0)

    assert "grad_norm" in logged
    torch.testing.assert_close(logged["grad_norm"], torch.tensor(5.0))


def test_non_finite_grad_norm_warns(model, monkeypatch):
    """An inf gradient must produce a loud, actionable warning — this is the
    failure mode where the forward looks healthy and the model NaNs one step later.
    NeMo's logger does not propagate to caplog, so capture it at the source."""
    import nemo.collections.speechlm2.models.streaming_stt_model_automodel as mod

    warnings_seen = []
    monkeypatch.setattr(type(model), "log", lambda self, name, value, **kw: None)
    monkeypatch.setattr(StreamingSTTModel, "configure_gradient_clipping", lambda self, *a, **k: None, raising=False)
    monkeypatch.setattr(mod.logging, "warning", lambda msg, *a, **k: warnings_seen.append(msg % a if a else msg))
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([float("inf"), 0.0])

    model.configure_gradient_clipping(_fake_optimizer([p]), gradient_clip_val=1.0)

    assert any("Non-finite gradient norm" in w for w in warnings_seen), warnings_seen
    # the warning must localize the failure, naming the offending parameter
    assert any("non-finite" in w for w in warnings_seen), warnings_seen


def test_grad_norm_logging_skipped_without_grads(model, monkeypatch):
    """No gradients yet (e.g. sanity-check val) must not log or crash."""
    logged = {}
    monkeypatch.setattr(type(model), "log", lambda self, name, value, **kw: logged.__setitem__(name, value))
    monkeypatch.setattr(StreamingSTTModel, "configure_gradient_clipping", lambda self, *a, **k: None, raising=False)
    model.configure_gradient_clipping(_fake_optimizer([]), gradient_clip_val=1.0)
    assert logged == {}
