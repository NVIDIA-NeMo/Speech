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

from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.speechlm2.data import DuplexS2SDataset
from nemo.collections.speechlm2.models import DuplexS2SModel

if torch.cuda.is_available():
    torch.set_default_device('cuda')


def resolve_pretrained_models():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        # CI pre-cached paths:
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1",
            "pretrained_audio_codec": "/home/TestData/speechlm/pretrained_models/low-frame-rate-speech-codec-22khz.nemo",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo",
            "scoring_asr": "/home/TestData/speechlm/pretrained_models/stt_en_fastconformer_transducer_large.nemo",
        }
    else:
        # HF URLs:
        return {
            "pretrained_asr": "stt_en_fastconformer_hybrid_large_streaming_80ms",
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
            "pretrained_audio_codec": "nvidia/low-frame-rate-speech-codec-22khz",
        }


@pytest.fixture(scope="session")
def model():
    cfg = {
        **resolve_pretrained_models(),
        "pretrained_weights": False,
        "freeze_params": ["^audio_codec\\..+$"],
        "audio_loss_weight": 1,
        "text_loss_weight": 3,
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
    model = DuplexS2SModel(cfg)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


@pytest.fixture(scope="session")
def dataset(model):
    return DuplexS2SDataset(
        model.tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        target_sample_rate=22050,
        input_roles=["user"],
        output_roles=["assistant"],
    )


@pytest.fixture(scope="session")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.target_audio = dummy_recording(1, with_data=True)
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=0.1,
            text='hi',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.1,
            text='hello',
            speaker="assistant",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.5,
            duration=0.1,
            text='ok',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.6,
            duration=0.4,
            text='okay',
            speaker="assistant",
        ),
    ]
    return CutSet([cut])


def test_s2s_dataset(dataset, training_cutset_batch):
    batch = dataset[training_cutset_batch]
    for key in (
        "source_audio",
        "target_audio",
        "source_audio_lens",
        "target_audio_lens",
        "target_tokens",
        "target_token_lens",
        "source_tokens",
        "source_token_lens",
    ):
        assert key in batch
        assert torch.is_tensor(batch[key])
    assert batch["source_audio"].shape == (1, 16000)
    assert batch["target_audio"].shape == (1, 22050)
    assert batch["target_texts"] == ["hello okay"]
    assert batch["target_tokens"].tolist() == [[0, 0, 0, 0, 1, 2, 0, 0, 1, 20759, 0, 0, 0]]
    assert batch["source_tokens"].tolist() == [[1, 2, 0, 0, 0, 0, 1, 3431, 2, 0, 0, 0, 0]]


def test_s2s_training_step(model, dataset, training_cutset_batch):
    model.on_train_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


def test_s2s_validation_step(model, dataset, training_cutset_batch):
    model.on_validation_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None  # no return value


def test_s2s_offline_generation(model):
    # 16000 samples == 1 second == 12.5 frames ~= 14 frames after encoder padding
    ans = model.offline_inference(
        input_signal=torch.randn(1, 16000),
        input_signal_lens=torch.tensor([16000]),
    )

    assert ans.keys() == {"text", "tokens_text", "tokens_audio", "audio", "audio_len", "tokens_len"}

    assert isinstance(ans["text"], list)
    assert isinstance(ans["text"][0], str)

    gen_text = ans["tokens_text"]
    assert gen_text.shape == (1, 13)
    assert gen_text.dtype == torch.long
    assert (gen_text >= 0).all()
    assert (gen_text < model.text_vocab_size).all()

    gen_audio_codes = ans["tokens_audio"]
    assert gen_audio_codes.shape == (1, 13, 8)
    assert gen_audio_codes.dtype == torch.long
    assert (gen_audio_codes >= 0).all()
    assert (gen_audio_codes < model.speech_vocab_size).all()

    gen_audio = ans["audio"]
    assert gen_audio.dtype == torch.float32
