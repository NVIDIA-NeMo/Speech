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
import os

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording
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

if torch.cuda.is_available():
    torch.set_default_device('cuda')


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
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


@requires_cuda
def test_salm_automodel_validation_step(model, dataset, prompt_formatter, training_cutset_batch):
    model.on_validation_epoch_start()
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None


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
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        "audio_lens": torch.tensor([5], dtype=torch.long, device=device),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool, device=device),
    }

    inputs = model.prepare_inputs(batch)

    chunked_signal, chunked_lens = model.perception.calls[0]
    assert chunked_signal.shape == (2, 3)
    assert torch.equal(chunked_lens, torch.tensor([2, 3], dtype=torch.long, device=device))
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

    answer = model.generate(
        prompts=torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        audios=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        audio_lens=torch.tensor([5], dtype=torch.long, device=device),
        max_new_tokens=3,
    )

    chunked_signal, chunked_lens = model.perception.calls[0]
    assert chunked_signal.shape == (2, 3)
    assert torch.equal(chunked_lens, torch.tensor([2, 3], dtype=torch.long, device=device))
    assert torch.equal(
        model.llm.generate_kwargs["inputs_embeds"][0, :5, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device),
    )
    assert answer.shape == (1, 3)


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
