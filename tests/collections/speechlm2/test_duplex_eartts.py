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
from nemo.collections.speechlm2.data.duplex_ear_tts_dataset import (
    DuplexEARTTSDataset,
    add_speech_delay,
    sample_audio_segments_repeat,
)
from nemo.collections.speechlm2.models import DuplexEARTTS


if torch.cuda.is_available():
    torch.set_default_device('cuda')


test_eartts_config = {
    "model": {
        "pretrained_lm_name": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "pretrained_ae_dir": None,
        "pretrained_tts_model": None,
        "scoring_asr": "stt_en_fastconformer_transducer_large",
        "freeze_params": [
            r"^audio_codec\..+$",  # Keep audio codec frozen as it only provides supervision for training.
            r"^embed_tokens\..+$",  # Keep embed_tokens frozen as done in eartts
        ],
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<SPECIAL_12>",
        "audio_codec_run_dtype": "float32",
        "prevent_freeze_params": [],
        "audio_save_path": "",
        "inference_guidance_scale": 0.5,
        "inference_noise_scale": 0.8,
        "inference_top_p_or_k": 0.8,
        "inference_guidance_enabled": False,
        "subword_mask_exactly_as_eartts": False,
        "context_hidden_mask_exactly_as_eartts": False,
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 4e-5,
            "betas": [0.9, 0.98],
            "weight_decay": 0,
            "foreach": True,
        },
        "lr_scheduler": {
            "_target_": "nemo.core.optim.lr_scheduler.InverseSquareRootAnnealing",
            "warmup_steps": 2500,
            "min_lr": 1e-6,
            "max_steps": 100_000_000,
        },
        "codec_config": {
            "latent_size": 512,
            "n_fft": 16,
            "hop_length": 4,
            "base_hidden_size": 384,
            "channel_mult": [1, 2, 4],
            "rates": [7, 7, 9],
            "num_blocks": 3,
            "kernel_size": 7,
            "groups": 1,
            "codebook_size": 1024,
            "num_quantizers": 31,
            "wav_to_token_ratio": 1764,
        },
        "tts_config": {
            "use_gated_fusion_for_text_audio": True,
            "disable_eos_prediction": True,
            "use_bos_eos_emb": True,
            "use_subword_flag_emb": True,
            "num_delay_speech_tokens": 2,
            "backbone_type": "gemma3_text",
            "backbone_model_class": None,
            "backbone_config_class": None,
            "backbone_config": {
                "hidden_size": 1152,
                "intermediate_size": 4608,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "head_dim": 72,
                "attention_dropout": 0.1,
                "use_cache": False,
            },
            "latent_size": 512,
            "codebook_size": 1024,
            "num_quantizers": 31,
            "context_hidden_size": None,
            "cas_config": {
                "backbone_type": "t5gemma",
                "backbone_model_class": None,
                "backbone_config_class": None,
                "backbone_config": {
                    "is_encoder_decoder": False,
                    "encoder": {
                        "hidden_size": 1152,
                        "intermediate_size": 4608,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 16,
                        "num_key_value_heads": 16,
                        "head_dim": 72,
                        "use_cache": False,
                        "attention_dropout": 0.1,
                    },
                },
            },
            "mog_head_config": {
                "intermediate_size": 4608,
                "num_layers": 3,
                "low_rank": 64,
                "num_predictions": 1024,
                "min_log_std": -4.0,
                "eps": 1e-6,
            },
            "p_uncond": 0.1,
            "label_smoothing": 0.01,
            "max_training_rate": 0.8,
            "quantizer_dropout": 0.5,
            "random_target_masking": False,
            "exponent": 3.0,
        },
    },
    "trainer": {
        "devices": -1,
        "accelerator": "gpu",
        "num_nodes": 1,
        "precision": 32,
        "logger": False,
        "enable_checkpointing": False,
        "use_distributed_sampler": False,
        "max_steps": 100_000_000,
        "val_check_interval": 1000,
        "limit_train_batches": "${trainer.val_check_interval}",
        "limit_val_batches": 2,
        "log_every_n_steps": 20,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "strategy": {
            "_target_": "lightning.pytorch.strategies.DDPStrategy",
            "gradient_as_bucket_view": True,
            "find_unused_parameters": True,
        },
    },
    "data": {
        "add_text_bos_and_eos_in_each_turn": True,
        "add_audio_prompt": True,
        "audio_prompt_duration": 3.0,
        "frame_length": 0.08,
        "source_sample_rate": 22050,
        "target_sample_rate": 22050,
        "input_roles": ["user", "User"],
        "output_roles": ["agent", "Assistant", "assistant", "Agent"],
    },
    "exp_manager": {
        "exp_dir": None,
        "explicit_log_dir": "",
        "name": "eartts",
        "create_tensorboard_logger": False,
        "create_checkpoint_callback": True,
        "use_datetime_version": True,
        "max_time_per_run": "00:03:50:00",
        "resume_from_checkpoint": None,
        "resume_if_exists": True,
        "resume_ignore_no_checkpoint": True,
        "create_wandb_logger": True,
        "wandb_logger_kwargs": {
            "name": "duplex_eartts_test",
            "project": "duplex_eartts",
            "resume": True,
        },
    },
}

# set CI cached path
if os.path.exists("/home/TestData/"):
    test_eartts_config["model"]["pretrained_lm_name"] = "/home/TestData/nvidia--NVIDIA-Nemotron-Nano-9B-v2/"


@pytest.fixture(scope="session")
def model():
    model = DuplexEARTTS(test_eartts_config)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


@pytest.fixture(scope="session")
def dataset(model):
    return DuplexEARTTSDataset(
        model.tokenizer,
        add_text_bos_and_eos_in_each_turn=True,
        add_audio_prompt=True,
        audio_prompt_duration=3.0,
        frame_length=0.08,
        source_sample_rate=22050,
        target_sample_rate=22050,
        input_roles=["user", "User"],
        output_roles=["agent", "Assistant", "assistant", "Agent"],
    )


@pytest.fixture(scope="session")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True, duration=1.0, sampling_rate=22050))
    cut.target_audio = dummy_recording(1, with_data=True, duration=1.0, sampling_rate=22050)
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
            duration=0.1,
            text='okay',
            speaker="assistant",
        ),
    ]
    return CutSet([cut])


def test_eartts_dataset(dataset, training_cutset_batch):
    batch = dataset[training_cutset_batch]
    expected_keys = {
        "sample_id",
        "non_prompt_mask",
        "prompt_lens",
        "aligned_attention_mask",
        "aligned_position_ids",
        "source_audio",
        "source_audio_lens",
        "target_audio",
        "target_audio_lens",
        "target_text_tokens",
        "target_token_lens",
        "source_tokens",
        "source_token_lens",
        "target_texts",
        "audio_prompt",
        "audio_prompt_lens",
        "task",
    }

    for key in expected_keys:
        assert key in batch, f"Missing key: {key}"

    tensor_keys = [
        "non_prompt_mask",
        "aligned_attention_mask",
        "aligned_position_ids",
        "source_audio",
        "source_audio_lens",
        "target_audio",
        "target_audio_lens",
        "target_text_tokens",
        "target_token_lens",
        "source_tokens",
        "source_token_lens",
        "audio_prompt",
        "audio_prompt_lens",
    ]

    for key in tensor_keys:
        assert torch.is_tensor(batch[key]), f"{key} must be a tensor"

    # Check target text consistency
    assert batch["target_texts"] == ["hello okay"]
    assert batch["source_tokens"].tolist() == [
        [
            2,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            2,
            1,
            2,
            12,
            12,
            12,
            12,
            1,
            1662,
            2,
            12,
            12,
            12,
            12,
        ]
    ]

    assert batch["target_text_tokens"].tolist() == [
        [
            2,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            2,
            12,
            12,
            12,
            12,
            1,
            2,
            12,
            12,
            1,
            2,
            1417,
            12,
            12,
        ]
    ]

    # Check task
    assert batch["task"] == ["s2s_duplex"]


# test extra functions inside of eartts dataset
def test_add_speech_delay():
    source_audio = torch.ones(1, 16000)
    target_audio = torch.ones(1, 22050)

    source_lens = torch.tensor([16000])
    target_lens = torch.tensor([22050])

    num_delays = 2

    # samples per frame (float → int handled explicitly)
    target_samples_per_frame = source_audio.size(1) / 12.5
    source_samples_per_frame = target_audio.size(1) / 12.5

    expected_extra_src_size = int(source_samples_per_frame * num_delays)
    expected_extra_tgt_size = int(target_samples_per_frame * num_delays)

    out_src, out_src_lens, out_tgt, out_tgt_lens = add_speech_delay(
        source_audio=source_audio,
        source_audio_lens=source_lens,
        target_audio=target_audio,
        target_audio_lens=target_lens,
        num_delay_speech_tokens=num_delays,
        target_samples_per_frame=target_samples_per_frame,
        source_samples_per_frame=source_samples_per_frame,
    )

    # --------------------------------------------------
    # Shape & length bookkeeping
    # --------------------------------------------------
    assert out_src.shape == (1, source_audio.size(1) + expected_extra_src_size)
    assert out_tgt.shape == (1, target_audio.size(1) + expected_extra_tgt_size)
    assert out_src_lens.item() == source_lens.item() + expected_extra_src_size
    assert out_tgt_lens.item() == target_lens.item() + expected_extra_tgt_size

    # --------------------------------------------------
    # Padding direction & content
    # --------------------------------------------------
    # Target audio is left-padded
    assert torch.all(out_tgt[:, :expected_extra_tgt_size] == 0)
    assert torch.all(out_tgt[:, expected_extra_tgt_size:] == 1)

    # Source audio is right-padded
    assert torch.all(out_src[:, : source_audio.size(1)] == 1)
    assert torch.all(out_src[:, source_audio.size(1) :] == 0)


def test_sample_audio_segments_repeat():
    cases = [
        # (audio, lens, n_sample, expected_when_sample_false)
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            torch.tensor([5]),
            3,
            torch.tensor([[1.0, 2.0, 3.0]]),
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([2]),
            5,
            torch.tensor([[1.0, 2.0, 1.0, 2.0, 1.0]]),
        ),
        (
            torch.zeros(1, 10),
            torch.tensor([0]),
            4,
            torch.zeros(1, 4),
        ),
    ]

    for prompt_audio, prompt_audio_lens, n_sample, expected in cases:
        # --------------------------------------------------
        # sample=False → deterministic + sequence check
        # --------------------------------------------------
        out = sample_audio_segments_repeat(
            prompt_audio,
            prompt_audio_lens,
            n_sample=n_sample,
            sample=False,
        )

        assert out.shape == expected.shape
        assert torch.equal(out, expected)

        # --------------------------------------------------
        # sample=True → stochastic, shape only
        # --------------------------------------------------
        out = sample_audio_segments_repeat(
            prompt_audio,
            prompt_audio_lens,
            n_sample=n_sample,
            sample=True,
        )

        assert out.shape == expected.shape


def test_eartts_training_step(model, dataset, training_cutset_batch):
    model.train()
    model.on_train_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


def test_eartts_validation_step(model, dataset, training_cutset_batch):
    model.eval()
    model.on_validation_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None  # no return value


def test_eartts_offline_generation(model):
    model.eval()
    # generate random subword_ids
    subword_ids = torch.ones(2, 10).long()

    # set init inputs and get it
    model.set_init_inputs(
        speaker_audio=torch.randn(1, 22050),
        speaker_audio_lens=torch.tensor([22050]),
    )
    init_inputs = model.get_init_inputs(B=subword_ids.size(0))
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    gen_audio, gen_audio_len = model.offline_inference(
        next_subword_ids=subword_ids,
        init_inputs=init_inputs,
    )
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    gen_audio_inc, gen_audio_len_inc = model.offline_inference(
        next_subword_ids=subword_ids, init_inputs=init_inputs, incremental_audio_decoding=True
    )

    assert torch.equal(
        gen_audio_len, gen_audio_len_inc
    ), "Audio lengths differ between incremental and non-incremental decoding."

    # compare waveform
    torch.testing.assert_close(
        gen_audio,
        gen_audio_inc,
        atol=1e-1,
        rtol=0,
    )

    assert gen_audio.shape == (2, 17640)
    assert gen_audio_len[0] == gen_audio.size(-1)
    assert gen_audio.dtype == torch.float32
