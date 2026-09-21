# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import inspect
import os
from contextlib import contextmanager

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording
from lightning import LightningModule
from transformers import GenerationConfig

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.speechlm2.data import SALMDataset
from nemo.collections.speechlm2.models import SALMAutomodel
from tests.collections.speechlm2._chunking_helpers import (
    ChunkingTestPerception,
    ChunkingTestTokenizer,
    chunking_test_devices,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SALMAutomodel requires CUDA")


@pytest.fixture(autouse=True, scope="module")
def _default_device_cuda():
    """Run this module's tests on CUDA by default, but scope the change so it does
    not leak. ``torch.set_default_device`` is a global, process-wide mutation; setting
    it at import time bleeds into other modules collected in the same pytest session
    (e.g. device-agnostic ASR module tests),
    causing spurious cuda/cpu device-mismatch failures. The previous default device
    is always restored on teardown.
    """
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.get_default_device()
    torch.set_default_device('cuda')
    try:
        yield
    finally:
        torch.set_default_device(prev)


def resolve_pretrained_models():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        # CI pre-cached paths:
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/Qwen--Qwen3-1.7B",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo",
        }
    else:
        # HF URLs:
        return {
            "pretrained_asr": "nvidia/canary-1b-flash",
            "pretrained_llm": "Qwen/Qwen3-1.7B",
        }


AUDIO_LOCATOR_TAG = "<|audioplaceholder|>"
PROMPT = "qwen"


@pytest.fixture(scope="session")
def model():
    if not torch.cuda.is_available():
        pytest.skip("SALMAutomodel requires CUDA")
    cfg = {
        **resolve_pretrained_models(),
        "pretrained_weights": False,
        # Exercise backend preservation for native fixtures and removal for HF fixtures.
        "automodel_backend": {"dispatcher": "torch"},
        "prompt_format": PROMPT,
        "audio_locator_tag": AUDIO_LOCATOR_TAG,
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
    model = SALMAutomodel(cfg)
    model.configure_model()
    model.to("cuda")
    return model


@pytest.fixture(scope="session")
def dataset(model):
    return SALMDataset(model.tokenizer)


@pytest.fixture(scope="session")
def prompt_formatter(model):
    return PromptFormatter.resolve(PROMPT)(model.tokenizer)


@pytest.fixture(scope="session")
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
def test_salm_automodel_dataset(dataset, prompt_formatter, training_cutset_batch):
    # This first step pre-tokenizes the examples, usually handled within `get_lhotse_dataloder_from_config`.
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    # fmt: off
    tokenized = training_cutset_batch[0].input_ids
    assert (
        prompt_formatter.tokenizer.tokenizer.decode(tokenized) ==
        f"<|im_start|>user\nRepeat after me: {AUDIO_LOCATOR_TAG}<|im_end|>\n<|im_start|>assistant\nSome text transcription.<|im_end|>\n"
    )
    # fmt: on
    batch = dataset[training_cutset_batch]
    for key in ("audios", "audio_lens", "input_ids", "loss_mask"):
        assert key in batch
        assert torch.is_tensor(batch[key])


@requires_cuda
def test_salm_automodel_training_step(model, dataset, prompt_formatter, training_cutset_batch):
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model._training_step_batch(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


def test_salm_automodel_training_step_uses_dataloader_iter_signature():
    assert list(inspect.signature(SALMAutomodel.training_step).parameters) == ["self", "dataloader_iter"]


def test_salm_automodel_forward_enters_configured_te_fp8_context():
    events = []

    class FakeFP8:
        @contextmanager
        def maybe_te_autocast(self):
            events.append("enter")
            yield
            events.append("exit")

    class FakeLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = type("Backend", (), {"te_fp8": FakeFP8()})()

        def forward(self, *args, inputs_embeds, **kwargs):
            events.append("forward")
            return {"logits": inputs_embeds}

    model = SALMAutomodel.__new__(SALMAutomodel)
    LightningModule.__init__(model)
    model.llm = FakeLLM()
    model._fused_linear_cross_entropy = None

    outputs = model.forward(torch.randn(1, 2, 4))

    assert outputs["logits"].shape == (1, 2, 4)
    assert events == ["enter", "forward", "exit"]


def test_salm_automodel_backward_does_not_enter_te_fp8_context(monkeypatch):
    events = []

    class FakeFP8:
        @contextmanager
        def maybe_te_autocast(self):
            events.append("enter")
            yield
            events.append("exit")

    class FakeLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = type("Backend", (), {"te_fp8": FakeFP8()})()

    model = SALMAutomodel.__new__(SALMAutomodel)
    LightningModule.__init__(model)
    model.llm = FakeLLM()
    monkeypatch.setattr(model, "_setup_moe_fsdp_sync", lambda: events.append("setup"))
    monkeypatch.setattr(LightningModule, "backward", lambda *_args, **_kwargs: events.append("backward"))

    model.backward(torch.tensor(1.0))

    assert events == ["setup", "backward"]


def test_salm_automodel_pad_token_override_preserves_eot_labels(monkeypatch):
    seen = {}

    class FakeTokenizer:
        def __init__(self, _src, *, use_fast, trust_remote_code, pad_token):
            seen["pad_token"] = pad_token
            self.pad = 0 if pad_token == "<unk>" else 11
            self.unk_id = 0

        def add_special_tokens(self, _tokens):
            return 0

    salm_module = __import__("nemo.collections.speechlm2.models.salm_automodel", fromlist=["AutoTokenizer"])
    monkeypatch.setattr(salm_module, "AutoTokenizer", FakeTokenizer)
    model = SALMAutomodel(
        {
            "pretrained_llm": "unused",
            "audio_locator_tag": "<|audio|>",
            "pad_token": "<unk>",
        }
    )

    assert seen["pad_token"] == "<unk>"
    assert model.text_pad_id == 0

    from nemo.collections.speechlm2.parts.packed_sequences import prepare_packed_llm_inputs

    packed = prepare_packed_llm_inputs(
        input_ids=torch.tensor([[0, 10, 11, 10, 42, 11]]),
        text_embs=torch.randn(1, 6, 2),
        audio_embs=[],
        target_ids=torch.tensor([[-100, -100, -100, -100, 42, 11]]),
        padding_id=model.text_pad_id,
        placeholder_id=999,
    )
    assert packed["target_ids"].tolist() == [-100, -100, 42, 11, -100]


def test_salm_automodel_fused_linear_forward_keeps_hidden_states_without_logits():
    class FakeLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            hidden = kwargs["inputs_embeds"] + 3
            logits = hidden[..., :0] if kwargs.get("compute_logits") is False else hidden[..., :1]
            return {"logits": logits, "hidden_states": (hidden,)}

    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.cfg = {}
    model._fused_linear_cross_entropy = object()
    model.llm = FakeLLM()
    model.train()
    inputs = torch.randn(1, 5, 4)

    outputs = model.forward(inputs)

    assert model.llm.kwargs["compute_logits"] is False
    assert model.llm.kwargs["output_hidden_states"] is True
    assert "compute_mtp" not in model.llm.kwargs
    torch.testing.assert_close(outputs["hidden_states"], inputs + 3)
    assert outputs["logits"].shape == (1, 5, 0)


def test_salm_automodel_fused_linear_loss_consumes_hidden_states_and_lm_weight():
    calls = []

    class FakeFusedLoss:
        def __call__(self, hidden_states, target_ids, weight, grad_reduce_group):
            calls.append((hidden_states, target_ids, weight, grad_reduce_group))
            return hidden_states.new_tensor(7.0)

    class FakeLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(3, 5, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.llm = FakeLLM()
    model._fused_linear_cross_entropy = FakeFusedLoss()
    hidden = torch.randn(1, 4, 3)
    targets = torch.tensor([[0, 1, -100, 2]])
    group = object()

    loss_sum, logits = model._compute_training_cross_entropy_sum(
        {"hidden_states": hidden, "logits": torch.empty(0)}, targets, group
    )

    assert loss_sum.item() == 7.0
    assert logits is None
    assert calls == [(hidden, targets, model.llm.lm_head.weight, group)]


def test_salm_automodel_notifies_garbage_collection_after_optimizer_step(monkeypatch):
    calls = []

    class FakeGarbageCollectionManager:
        def on_optimizer_step(self):
            calls.append("gc")

    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model._garbage_collection = FakeGarbageCollectionManager()
    monkeypatch.setattr(
        LightningModule,
        "optimizer_step",
        lambda *args, **kwargs: calls.append("optimizer"),
    )
    model.optimizer_step(0, 0, object())
    assert calls == ["optimizer", "gc"]


def test_salm_automodel_record_training_stats_uses_thd_metadata():
    model = SALMAutomodel.__new__(SALMAutomodel)
    batch = {"input_ids": torch.zeros(3, 7, dtype=torch.long)}
    inputs = {
        "input_embeds": torch.zeros(5, 4),
        "attention_mask": None,
        "num_tokens": torch.tensor(11),
        "num_examples": torch.tensor(3),
    }

    model._record_training_stats(batch, inputs)

    assert model._last_batch_num_tokens == 11
    assert model._last_batch_num_examples == 3


@requires_cuda
def test_salm_automodel_validation_step(model, dataset, prompt_formatter, training_cutset_batch):
    model.on_validation_epoch_start()
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None


def test_salm_automodel_validation_epoch_end_uses_token_weighted_metrics():
    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.on_validation_epoch_start()
    model._get_moe_dp_group = lambda: None

    model._partial_val_loss_sums["dummy"].extend([torch.tensor(2.0), torch.tensor(18.0)])
    model._partial_val_corrects["dummy"].extend([torch.tensor(1.0), torch.tensor(3.0)])
    model._partial_val_num_frames["dummy"].extend([torch.tensor(1.0), torch.tensor(9.0)])

    logged = {}

    def fake_log(name, value, **kwargs):
        logged[name] = value.detach().cpu()

    model.log = fake_log
    model.on_validation_epoch_end()

    assert logged["val_loss_dummy"].item() == pytest.approx(2.0)
    assert logged["val_acc_dummy"].item() == pytest.approx(0.4)
    assert logged["val_loss"].item() == pytest.approx(2.0)
    assert logged["val_acc"].item() == pytest.approx(0.4)
    assert not model._partial_val_loss_sums
    assert not model._partial_val_corrects
    assert not model._partial_val_num_frames


@requires_cuda
def test_salm_automodel_generation(model):
    answer = model.generate(
        prompts=[
            [
                {"role": "user", "slots": {"message": f"Repeat after me: {AUDIO_LOCATOR_TAG}"}},
            ]
        ],
        audios=torch.randn(1, 16000),
        audio_lens=torch.tensor([16000]),
        max_new_tokens=4,
    )
    assert answer.shape == (1, 4)
    assert answer.dtype == torch.long
    assert (answer >= 0).all()
    assert (answer < model.text_vocab_size).all()


@requires_cuda
@pytest.mark.parametrize(
    ("enable_thinking", "expected_formatter_kwargs"),
    [
        (False, {"enable_thinking": False}),
        (None, {}),
    ],
)
def test_salm_automodel_generation_passes_enable_thinking(
    model, monkeypatch, enable_thinking, expected_formatter_kwargs
):
    seen = {}

    class _FakeFormatter:
        def __init__(self, tokenizer):
            pass

        def encode_dialog(self, turns, **kwargs):
            seen["turns"] = turns
            seen["formatter_kwargs"] = kwargs
            return {"input_ids": torch.tensor([1, 2], dtype=torch.long)}

    def fake_generate(*, input_ids, attention_mask, generation_config, **kwargs):
        seen["input_ids"] = input_ids
        seen["attention_mask"] = attention_mask
        max_new_tokens = kwargs["max_new_tokens"]
        return torch.zeros((input_ids.shape[0], max_new_tokens), dtype=torch.long, device=input_ids.device)

    monkeypatch.setattr(PromptFormatter, "resolve", staticmethod(lambda name: _FakeFormatter))
    monkeypatch.setattr(model.llm, "generate", fake_generate, raising=False)

    answer = model.generate(
        prompts=[[{"role": "user", "slots": {"message": "test"}}]],
        enable_thinking=enable_thinking,
        max_new_tokens=3,
    )

    assert seen["formatter_kwargs"] == expected_formatter_kwargs
    assert seen["turns"] == [{"role": "user", "slots": {"message": "test"}}]
    assert seen["input_ids"].shape == (1, 2)
    assert torch.equal(seen["attention_mask"], torch.ones_like(seen["input_ids"], dtype=torch.bool))
    assert answer.shape == (1, 3)


@requires_cuda
def test_salm_automodel_generation_audios_via_prompt(model, tmp_path):
    audio_path = tmp_path / "audio.wav"
    dummy_cut(0, with_data=True).save_audio(audio_path)

    answer = model.generate(
        prompts=[
            [{"role": "user", "content": f"Repeat after me: {AUDIO_LOCATOR_TAG}", "audio": [audio_path]}],
            [
                {
                    "role": "user",
                    "content": f"Repeat after me: {AUDIO_LOCATOR_TAG} and {AUDIO_LOCATOR_TAG}",
                    "audio": [audio_path, audio_path],
                }
            ],
        ],
        generation_config=GenerationConfig(max_new_tokens=4),
    )
    assert answer.shape == (2, 4)
    assert answer.dtype == torch.long
    assert (answer >= 0).all()
    assert (answer < model.text_vocab_size).all()


@requires_cuda
def test_salm_automodel_generation_prompts_as_tensor(model):
    answer = model.generate(
        prompts=torch.tensor([[1, 2, 3, 4, 5, 6, 7, model.audio_locator_tag_id]]),
        audios=torch.randn(1, 16000),
        audio_lens=torch.tensor([16000]),
        max_new_tokens=4,
    )
    assert answer.shape == (1, 4)
    assert answer.dtype == torch.long
    assert (answer >= 0).all()
    assert (answer < model.text_vocab_size).all()


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_prepare_inputs_chunks_long_audio(device):
    model = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device=device)
    spk_targets = torch.arange(10, dtype=torch.float32, device=device).reshape(1, 5, 2)
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        "audio_lens": torch.tensor([5], dtype=torch.long, device=device),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool, device=device),
        "spk_targets": spk_targets,
    }

    inputs = model.prepare_inputs(batch)

    chunked_signal, chunked_lens = model.perception.calls[0]
    assert chunked_signal.shape == (2, 3)
    assert torch.equal(chunked_lens, torch.tensor([2, 3], dtype=torch.long, device=device))
    # Ordinary encoders keep their existing audio-chunking path and ignore
    # unsupported speaker targets.
    assert model.perception.spk_targets_calls[0] is None
    assert torch.equal(inputs["input_embeds"][0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device))
    assert torch.equal(inputs["attention_mask"], torch.ones((1, 5), dtype=torch.bool, device=device))


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_prepare_inputs_merges_short_tail_chunk(device):
    model = _make_chunking_test_model(encoder_chunk_size_seconds=0.5, sampling_rate=8, hop_length=2, device=device)
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], device=device),
        "audio_lens": torch.tensor([9], dtype=torch.long, device=device),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool, device=device),
    }

    inputs = model.prepare_inputs(batch)

    chunked_signal, chunked_lens = model.perception.calls[0]
    assert chunked_signal.shape == (2, 5)
    assert torch.equal(chunked_lens, torch.tensor([4, 5], dtype=torch.long, device=device))
    assert model.perception.spk_targets_calls[0] is None
    assert torch.equal(
        inputs["input_embeds"][0, :, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=device),
    )


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_prepare_inputs_skips_chunking_when_size_is_null(device):
    model = _make_chunking_test_model(encoder_chunk_size_seconds=None, sampling_rate=2, device=device)
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        "audio_lens": torch.tensor([5], dtype=torch.long, device=device),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool, device=device),
    }

    model.prepare_inputs(batch)

    input_signal, input_signal_lens = model.perception.calls[0]
    assert input_signal.shape == (1, 5)
    assert torch.equal(input_signal_lens, torch.tensor([5], dtype=torch.long, device=device))


@pytest.mark.parametrize("native_dataset_batch", [False, True])
@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_packed_no_chunking_embeds_only_real_tokens(monkeypatch, device, native_dataset_batch):
    """Packed no-chunking batches compact IDs before the embedding lookup."""
    model = _make_chunking_test_model(encoder_chunk_size_seconds=None, sampling_rate=2, device=device)
    model.cfg["packed_sequences"] = True
    batch_size = 4
    sequence_length = 64
    row_lengths = [64, 8, 4, 2]
    input_ids = torch.full((batch_size, sequence_length), model.text_pad_id, dtype=torch.long, device=device)
    for row, length in enumerate(row_lengths):
        tokens = torch.arange(10, 10 + length, dtype=torch.long, device=device)
        tokens[0] = model.audio_locator_tag_id
        input_ids[row, -length:] = tokens
    loss_mask = input_ids != model.text_pad_id
    loss_mask[input_ids == model.audio_locator_tag_id] = False
    audios = torch.arange(1, batch_size * 3 + 1, dtype=torch.float32, device=device).reshape(batch_size, 3)
    batch = {
        "audio_lens": torch.full((batch_size,), 3, dtype=torch.long, device=device),
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }
    if native_dataset_batch:
        batch.update(
            {
                "packed_audio_samples": audios.flatten(),
                "audio_cu_seqlens": torch.arange(0, batch_size * 3 + 1, 3, dtype=torch.long, device=device),
                "input_ids": torch.cat([row[-length:] for row, length in zip(input_ids, row_lengths)]),
                "loss_mask": torch.cat([row[-length:] for row, length in zip(loss_mask, row_lengths)]),
                "text_cu_seqlens": torch.tensor(
                    [0, *torch.tensor(row_lengths, device=device).cumsum(0).tolist()],
                    dtype=torch.long,
                    device=device,
                ),
            }
        )
    else:
        batch["audios"] = audios
    original_embed_tokens = model._embed_tokens
    embedded_shapes = []

    def embed_tokens(flat_ids):
        embedded_shapes.append(tuple(flat_ids.shape))
        return original_embed_tokens(flat_ids)

    monkeypatch.setattr(model, "_embed_tokens", embed_tokens)

    inputs = model.prepare_inputs(batch)

    real_token_count = sum(row_lengths)
    assert embedded_shapes == [(real_token_count,)]
    if not native_dataset_batch:
        assert real_token_count < input_ids.numel()
    assert inputs["input_embeds"].ndim == 2
    inputs["input_embeds"].sum().backward()
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_prepare_inputs_preserves_chunked_audio_order(device):
    model = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device=device)
    batch = {
        "audios": torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
            ],
            device=device,
        ),
        "audio_lens": torch.tensor([3, 5], dtype=torch.long, device=device),
        "input_ids": torch.tensor(
            [[model.audio_locator_tag_id, model.audio_locator_tag_id, 10]], dtype=torch.long, device=device
        ),
        "loss_mask": torch.tensor([[False, False, True]], dtype=torch.bool, device=device),
    }

    inputs = model.prepare_inputs(batch)

    assert torch.equal(
        inputs["input_embeds"][0, :, 0],
        torch.tensor([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0], device=device),
    )


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_generate_chunks_audio_before_llm(device):
    model = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device=device)
    spk_targets = torch.arange(10, dtype=torch.float32, device=device).reshape(1, 5, 2)

    answer = model.generate(
        prompts=torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        audios=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        audio_lens=torch.tensor([5], dtype=torch.long, device=device),
        spk_targets=spk_targets,
        max_new_tokens=3,
    )

    chunked_signal, chunked_lens = model.perception.calls[0]
    assert chunked_signal.shape == (2, 3)
    assert torch.equal(chunked_lens, torch.tensor([2, 3], dtype=torch.long, device=device))
    # Ordinary encoders keep their existing audio-chunking path and ignore
    # unsupported speaker targets.
    assert model.perception.spk_targets_calls[0] is None
    assert torch.equal(
        model.llm.generate_kwargs["inputs_embeds"][0, :5, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device),
    )
    assert answer.shape == (1, 3)


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_limits_packed_encoder_opt_in_to_training(device):
    model = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device=device)
    model.cfg["packed_encoder_sequences"] = True
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        "audio_lens": torch.tensor([5], dtype=torch.long, device=device),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool, device=device),
    }

    inputs = model.prepare_inputs(batch)

    assert model.perception.sequence_packed_calls == 1
    assert torch.equal(
        inputs["input_embeds"][0, :, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device),
    )

    answer = model.generate(
        prompts=torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        audios=torch.tensor([[6.0, 7.0, 8.0, 9.0, 10.0]], device=device),
        audio_lens=torch.tensor([5], dtype=torch.long, device=device),
        max_new_tokens=2,
    )

    assert model.perception.sequence_packed_calls == 1
    assert torch.equal(
        model.llm.generate_kwargs["inputs_embeds"][0, :5, 0],
        torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0], device=device),
    )
    assert answer.shape == (1, 2)


@pytest.mark.parametrize("device", chunking_test_devices())
def test_salm_automodel_packed_audio_samples_match_padded_batch(device):
    padded_model = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device=device)
    packed_model = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device=device)
    padded_model.cfg["packed_encoder_sequences"] = True
    packed_model.cfg["packed_encoder_sequences"] = True
    audios = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [10.0, 11.0, 12.0, 13.0, 14.0]], device=device)
    audio_lens = torch.tensor([3, 5], dtype=torch.long, device=device)
    common = {
        "audio_lens": audio_lens,
        "input_ids": torch.tensor(
            [[padded_model.audio_locator_tag_id, padded_model.audio_locator_tag_id, 10]],
            dtype=torch.long,
            device=device,
        ),
        "loss_mask": torch.tensor([[False, False, True]], dtype=torch.bool, device=device),
    }
    padded_batch = {**common, "audios": audios}
    packed_batch = {
        **common,
        "packed_audio_samples": torch.cat([audios[0, :3], audios[1, :5]]),
        "audio_cu_seqlens": torch.tensor([0, 3, 8], dtype=torch.long, device=device),
    }

    expected = padded_model.prepare_inputs(padded_batch)
    actual = packed_model.prepare_inputs(packed_batch)

    torch.testing.assert_close(actual["input_embeds"], expected["input_embeds"], rtol=0.0, atol=0.0)
    assert torch.equal(actual["target_ids"], expected["target_ids"])


def _make_chunking_test_model(encoder_chunk_size_seconds, sampling_rate, device, hop_length=1):
    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.cfg = {"encoder_chunk_size_seconds": encoder_chunk_size_seconds}
    model.audio_locator_tag = AUDIO_LOCATOR_TAG
    model.tokenizer = ChunkingTestTokenizer(AUDIO_LOCATOR_TAG)
    model.llm = _AutomodelChunkingTestLLM(device=device)
    model.perception = ChunkingTestPerception(sampling_rate=sampling_rate, hop_length=hop_length)
    model._use_tp = False
    return model


class _AutomodelChunkingTestLLM(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(128, 1, device=device)
        self.generate_kwargs = None
        with torch.no_grad():
            self.model.embed_tokens.weight.zero_()

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        batch_size = kwargs["inputs_embeds"].shape[0]
        max_new_tokens = kwargs["max_new_tokens"]
        return torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=kwargs["inputs_embeds"].device)
