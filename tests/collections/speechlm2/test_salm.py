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
from nemo.collections.speechlm2.models import SALM

if torch.cuda.is_available():
    torch.set_default_device('cuda')


def resolve_pretrained_models():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        # CI pre-cached paths:
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo",
        }
    else:
        # HF URLs:
        return {
            "pretrained_asr": "nvidia/canary-1b-flash",
            "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
        }


AUDIO_LOCATOR_TAG = "<|audioplaceholder|>"
PROMPT = "llama2"


@pytest.fixture(scope="session")
def model():
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
    }
    model = SALM(cfg)
    if torch.cuda.is_available():
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


def test_salm_dataset(dataset, prompt_formatter, training_cutset_batch):
    # This first step pre-tokenizes the examples, usually handled within `get_lhotse_dataloder_from_config`.
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    # fmt: off
    tokenized = training_cutset_batch[0].input_ids
    assert (
        prompt_formatter.tokenizer.tokenizer.decode(tokenized) ==
        f"<s> [INST] Repeat after me: {AUDIO_LOCATOR_TAG}  [/INST] Some text transcription. </s>"
    )
    # fmt: on
    batch = dataset[training_cutset_batch]
    for key in ("audios", "audio_lens", "input_ids", "loss_mask"):
        assert key in batch
        assert torch.is_tensor(batch[key])


def test_salm_training_step(model, dataset, prompt_formatter, training_cutset_batch):
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


def test_salm_validation_step(model, dataset, prompt_formatter, training_cutset_batch):
    model.on_validation_epoch_start()
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None


def test_salm_generation(model):
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


def test_salm_generation_audios_via_prompt(model, tmp_path):
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


def test_salm_generation_prompts_as_tensor(model):
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
