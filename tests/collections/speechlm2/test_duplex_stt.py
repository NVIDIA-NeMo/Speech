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

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.speechlm2.data import DuplexS2SDataset
from nemo.collections.speechlm2.models import DuplexSTTModel

if torch.cuda.is_available():
    torch.set_default_device('cuda')


def resolve_pretrained_models():
    return {
        "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
    }


def create_model(
    predict_user_text=False,
    force_use_noise_augmentation=False,
    old_noise_prob=0.0,
    old_noise_min_snr=0.0,
    old_noise_max_snr=0.0,
):
    """Helper function to create a model with configurable settings."""
    cfg = {
        "model": {
            **resolve_pretrained_models(),
            "pretrained_weights": False,
            "freeze_params": ["^audio_codec\\..+$"],
            "audio_loss_weight": 1,
            "text_loss_weight": 3,
            "perception": {
                "_target_": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                "modality_adapter": {
                    "_target_": "torch.nn.Linear",
                    "in_features": 512,
                    "out_features": 512,
                },
            },
            "speech_decoder": {
                "n_layers": 1,
                "d_model": 768,
                "d_ffn": 3072,
                "sa_n_heads": 12,
                "kernel_size": 3,
                "is_causal": True,
            },
            "predict_user_text": predict_user_text,
            "force_use_noise_augmentation": force_use_noise_augmentation,
            "old_noise_prob": old_noise_prob,
            "old_noise_min_snr": old_noise_min_snr,
            "old_noise_max_snr": old_noise_max_snr,
            "optimizer": {"_target_": "torch.optim.AdamW"},
        },
        "data": {
            "target_sample_rate": 22050,
            "source_sample_rate": 16000,
        },
        "exp_manager": {
            "explicit_log_dir": "/tmp/test_duplex_stt_logs",
        },
    }
    model = DuplexSTTModel(cfg)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


@pytest.fixture(scope="session")
def model():
    return create_model(predict_user_text=False)


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


def test_s2s_speech_decoder_training_step(model, dataset, training_cutset_batch):
    model.on_train_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


@pytest.fixture(scope="function")
def model_with_asr():
    """Model fixture with ASR head enabled."""
    return create_model(predict_user_text=True)


def test_s2s_speech_decoder_training_step_with_asr(model_with_asr, dataset, training_cutset_batch):
    # Model is initialized with ASR head enabled
    model_with_asr.on_train_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model_with_asr.device)
    results = model_with_asr.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0

    assert "asr_loss" in results
    assert torch.is_tensor(results["asr_loss"])
    assert not torch.isnan(results["asr_loss"])
    assert results["asr_loss"] >= 0


@pytest.fixture(scope="function")
def model_with_noise():
    """Model fixture with noise augmentation enabled."""
    # Explicitly enable the old noise augmentation and force it
    # Use some reasonable nonzero dummy values for noise params
    model = create_model(
        force_use_noise_augmentation=True,
        old_noise_prob=0.9,
        old_noise_min_snr=5.0,
        old_noise_max_snr=15.0,
    )
    return model


@pytest.fixture(scope="function")
def model_with_asr_and_noise():
    """Model fixture with both ASR head and noise augmentation enabled."""
    model = create_model(
        predict_user_text=True,
        force_use_noise_augmentation=True,
        old_noise_prob=0.9,
        old_noise_min_snr=5.0,
        old_noise_max_snr=15.0,
    )
    return model


def test_s2s_speech_decoder_training_step_with_noise(model_with_asr_and_noise, dataset, training_cutset_batch):
    # Model is initialized with both ASR head and noise augmentation enabled
    model_with_asr_and_noise.on_train_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model_with_asr_and_noise.device)
    results = model_with_asr_and_noise.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0

    assert "asr_loss" in results
    assert torch.is_tensor(results["asr_loss"])
    assert not torch.isnan(results["asr_loss"])
    assert results["asr_loss"] >= 0


def test_s2s_speech_decoder_validation_step(model, dataset, training_cutset_batch):
    model.on_validation_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None  # no return value


def test_s2s_speech_decoder_offline_generation(model):
    # 16000 samples == 1 second == 12.5 frames ~= 14 frames after encoder padding
    ans = model.offline_inference(
        input_signal=torch.randn(1, 16000, device=model.device),
        input_signal_lens=torch.tensor([16000], device=model.device),
    )

    assert ans.keys() == {
        'text',
        'src_text',
        'tokens_text_src',
        'tokens_text',
        'tokens_audio',
        'tokens_len',
        'source_audio',
        'source_audio_len',
    }

    assert isinstance(ans["text"], list)
    assert isinstance(ans["text"][0], str)

    gen_text = ans["tokens_text"]
    assert gen_text.shape == (1, 14)
    assert gen_text.dtype == torch.long
    assert (gen_text >= 0).all()
    assert (gen_text < model.text_vocab_size).all()


def test_s2s_speech_decoder_offline_generation_regression_with_asr(model_with_asr):
    """Test offline generation with ASR head enabled for user text prediction."""
    # 16000 samples == 1 second == 12.5 frames ~= 14 frames after encoder padding
    ans = model_with_asr.offline_inference(
        input_signal=torch.randn(1, 16000, device=model_with_asr.device),
        input_signal_lens=torch.tensor([16000], device=model_with_asr.device),
    )

    # Verify all expected output keys are present
    assert ans.keys() == {
        'text',
        'src_text',
        'tokens_text_src',
        'tokens_text',
        'tokens_audio',
        'tokens_len',
        'source_audio',
        'source_audio_len',
    }

    # Verify agent text output
    assert isinstance(ans["text"], list)
    assert isinstance(ans["text"][0], str)

    # Verify user text (ASR) output
    assert isinstance(ans["src_text"], list)
    assert isinstance(ans["src_text"][0], str)

    # Verify generated text tokens
    gen_text = ans["tokens_text"]
    assert gen_text.shape == (1, 14)
    assert gen_text.dtype == torch.long
    assert (gen_text >= 0).all()
    assert (gen_text < model_with_asr.text_vocab_size).all()

    # Verify ASR tokens
    asr_tokens = ans["tokens_text_src"]
    assert asr_tokens.shape[0] == 1  # batch size
    assert asr_tokens.dtype == torch.long
    assert (asr_tokens >= 0).all()
    assert (asr_tokens < model_with_asr.text_vocab_size).all()
