# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyModelInferenceParameters, TrainingMode
from tests.collections.tts.models.test_audio_codec import create_codec_config


if torch.cuda.is_available():
    torch.set_default_device("cuda")


pytestmark = pytest.mark.unit

BPE_TOKENIZER_NAME = "nemotron_bpe"
BPE_TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
BPE_TOKENIZER_CACHED_PATH = Path("/home/TestData/nvidia--NVIDIA-Nemotron-Nano-9B-v2/")
if BPE_TOKENIZER_CACHED_PATH.exists():
    BPE_TOKENIZER_MODEL = str(BPE_TOKENIZER_CACHED_PATH)


def _restore_codec_as_random_initialized_model(*args, **kwargs):
    del args
    if kwargs.get("return_config", False):
        return create_codec_config()

    codec_cfg = kwargs.get("override_config_path", None)
    if codec_cfg is None:
        codec_cfg = create_codec_config()
    codec_model = AudioCodecModel(cfg=codec_cfg)
    codec_model.freeze()
    return codec_model


@contextmanager
def _codec_restore_uses_random_initialized_audio_codec():
    from nemo.collections.tts.models import easy_magpietts_inference

    with patch.object(
        easy_magpietts_inference.AudioCodecModel,
        "restore_from",
        staticmethod(_restore_codec_as_random_initialized_model),
    ):
        yield


def _seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def tiny_easy_magpie_cfg(overrides=None):
    cfg = OmegaConf.create(
        {
            "codecmodel_path": "dummy_codec.nemo",
            "decoder_type": "nemotron_h",
            "embedding_dim": 32,
            "hidden_dim": 32,
            "audio_embedding_dim": 16,
            "frame_stacking_factor": 1,
            "local_transformer_type": "none",
            "disable_lm_text_head": True,
            "disable_subword_embedding": False,
            "use_bpe_char_tokenizer": True,
            "text_conditioning_tokenizer_name": BPE_TOKENIZER_NAME,
            "use_multiturn_dataset": False,
            "run_val_inference": False,
            "use_utmos": False,
            "cfg_unconditional_prob": 0.0,
            "dropout_text_input_prob": 0.0,
            "phoneme_corruption_batch_prob": 0.0,
            "phoneme_corruption_timestep_ratio": 0.0,
            "phoneme_as_text_prob": 0.0,
            "mask_user_on_loss": True,
            "text_tokenizers": {
                BPE_TOKENIZER_NAME: {
                    "_target_": "AutoTokenizer",
                    "pretrained_model": BPE_TOKENIZER_MODEL,
                }
            },
            "training_modes": [
                {
                    "text_input_mode": "streaming",
                    "streaming_phonemes_delay": 1,
                    "streaming_speech_delay": 2,
                }
            ],
            "nemotron_h_config": {
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "vocab_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 8,
                "ssm_state_size": 8,
                "n_groups": 2,
                "intermediate_size": 64,
                "hybrid_override_pattern": "*",
                "use_cache": True,
                "_attn_implementation": "sdpa",
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 0.001,
            },
        }
    )
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def _make_easy_magpie_model(cfg=None):
    _seed_everything()
    with _codec_restore_uses_random_initialized_audio_codec():
        model = EasyMagpieTTSModel(cfg or tiny_easy_magpie_cfg())
    model.eval()
    return model


@pytest.fixture()
def model():
    return _make_easy_magpie_model()


def _padded_token_tensor(model, texts):
    tokenized = [
        model.tokenizer.encode(text, tokenizer_name=BPE_TOKENIZER_NAME) + [model.eos_id] for text in texts
    ]
    lens = torch.tensor([len(tokens) for tokens in tokenized], dtype=torch.long)
    max_len = int(lens.max().item())
    padded = torch.full((len(tokenized), max_len), model.pad_id, dtype=torch.long)
    for idx, tokens in enumerate(tokenized):
        padded[idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return padded, lens


def _toy_codes(model, batch_size, num_frames):
    codes = torch.zeros(batch_size, model.num_audio_codebooks, num_frames, dtype=torch.long)
    frame_ids = torch.arange(num_frames, dtype=torch.long)
    for batch_idx in range(batch_size):
        for codebook_idx in range(model.num_audio_codebooks):
            codes[batch_idx, codebook_idx] = (frame_ids + batch_idx + codebook_idx * 7) % model.codebook_size
    return codes


def _toy_batch(model):
    text, text_lens = _padded_token_tensor(model, ["abc", "de"])
    context_text_tokens, context_text_tokens_lens = _padded_token_tensor(model, ["hi", "ok"])

    audio_codes = _toy_codes(model, batch_size=2, num_frames=4)
    audio_codes_lens = torch.tensor([4, 3], dtype=torch.long)
    context_audio_codes = _toy_codes(model, batch_size=2, num_frames=2)
    context_audio_codes[1, :, 1] = 0
    context_audio_codes_lens = torch.tensor([2, 1], dtype=torch.long)
    agent_mask = torch.tensor(
        [
            [False, True, True, False, False],
            [True, False, True, False, False],
        ],
        dtype=torch.bool,
    )

    return {
        "text": text,
        "text_lens": text_lens,
        "context_text_tokens": context_text_tokens,
        "context_text_tokens_lens": context_text_tokens_lens,
        "audio_codes": audio_codes,
        "audio_codes_lens": audio_codes_lens,
        "context_audio_codes": context_audio_codes,
        "context_audio_codes_lens": context_audio_codes_lens,
        "agent_mask": agent_mask,
        "task": ["tts", "tts"],
    }


@pytest.fixture()
def toy_batch(model):
    return _toy_batch(model)


def test_training_mode_and_inference_parameters():
    mode = TrainingMode(
        text_input_mode="streaming",
        streaming_phonemes_delay=4,
        streaming_speech_delay=8,
        mode_idx=2,
    )
    assert mode.name == "streaming_4_8"

    params = EasyModelInferenceParameters.from_dict(
        {
            "max_decoder_steps": 11,
            "temperature": 0.25,
            "topk": 7,
            "cfg_scale": 1.5,
            "unknown_key": "ignored",
        }
    )
    assert params == EasyModelInferenceParameters(
        max_decoder_steps=11,
        temperature=0.25,
        topk=7,
        cfg_scale=1.5,
    )


def test_easy_magpietts_model_construction(model):
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert next(model.parameters()).device.type == expected_device
    assert model.tokenizer is not None
    assert model.decoder is not None
    assert len(model.audio_embeddings) == model.num_audio_codebooks * model.frame_stacking_factor
    assert isinstance(model.audio_in_projection, nn.Linear)
    assert isinstance(model.audio_out_projection, nn.Linear)
    assert model.final_proj.out_features == model.num_audio_codebooks * model.num_all_tokens_per_codebook
    assert model.audio_bos_id == model.codebook_size
    assert model.audio_eos_id == model.codebook_size + 1
    assert model.training_modes[0].name == "streaming_1_2"
    assert model.default_inference_mode == "streaming_1_2"
    assert model.lm_text_head is None
    assert model.use_bpe_char_tokenizer
    assert model.text_conditioning_tokenizer_name == BPE_TOKENIZER_NAME
    assert hasattr(model, "cas_encoder")


def test_state_dict_excludes_codec(model):
    state = model.state_dict()
    assert state
    assert not any("_codec_model" in key for key in state)
    assert any(key.startswith("audio_embeddings.") for key in state)
    assert any(key.startswith("final_proj.") for key in state)


def test_audio_and_text_embedding_shapes(model):
    audio_tokens = _toy_codes(model, batch_size=2, num_frames=3)
    audio_tokens[0, :, -1] = model.audio_eos_id
    audio_embedded = model.embed_audio_tokens(audio_tokens)
    assert audio_embedded.shape == (2, 3, model.cfg.embedding_dim)
    assert audio_embedded.dtype == torch.float32
    assert torch.isfinite(audio_embedded).all()

    text_tokens, text_lens = _padded_token_tensor(model, ["abc", "de"])
    text_embedded = model.embed_text_tokens(text_tokens, text_lens=text_lens)
    assert text_embedded.shape == (2, text_tokens.size(1), model.cfg.embedding_dim)
    assert text_embedded.dtype == torch.float32
    assert torch.isfinite(text_embedded).all()


def test_stack_codes_round_trip_expected_shape(model):
    codes = _toy_codes(model, batch_size=2, num_frames=4)
    codes_lens = torch.tensor([4, 4], dtype=torch.long)

    stacked, stacked_lens = model.stack_codes(
        codes,
        codes_lens,
        bos_id=model.audio_bos_id,
        eos_id=model.audio_eos_id,
        stacking_factor=2,
        num_codebooks=model.num_audio_codebooks,
    )
    unstacked, unstacked_lens = model.unstack_codes(stacked, stacked_lens, stacking_factor=2)

    assert stacked.shape == (2, model.num_audio_codebooks * 2, 2)
    assert stacked_lens.tolist() == [2, 2]
    assert unstacked.shape == codes.shape
    assert unstacked_lens.tolist() == codes_lens.tolist()
    torch.testing.assert_close(unstacked, codes)


def test_compute_loss_with_and_without_agent_mask(model):
    _seed_everything()
    batch_size, num_frames = 2, 5
    audio_codes = torch.randint(
        low=0,
        high=model.num_all_tokens_per_codebook,
        size=(batch_size, model.num_audio_codebooks, num_frames),
    )
    audio_codes_lens = torch.tensor([5, 3], dtype=torch.long)
    logits = torch.randn(
        batch_size,
        num_frames,
        model.num_audio_codebooks * model.num_all_tokens_per_codebook,
    )
    agent_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [False, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    loss, loss_mask = model.compute_loss(logits, audio_codes, audio_codes_lens)
    masked_loss, masked_loss_mask = model.compute_loss(
        logits,
        audio_codes,
        audio_codes_lens,
        agent_mask_target=agent_mask,
    )

    assert loss.ndim == 0
    assert masked_loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(masked_loss)
    assert loss_mask.shape == (batch_size, model.num_audio_codebooks, num_frames)
    assert masked_loss_mask.shape == loss_mask.shape
    assert loss_mask.dtype == torch.bool


def test_prepare_audio_channel_embeddings_shapes(model):
    audio_codes = _toy_codes(model, batch_size=2, num_frames=3)
    audio_codes[1, :, 2] = 0
    audio_codes_lens = torch.tensor([3, 2], dtype=torch.long)
    delay = torch.tensor([2, 1], dtype=torch.long)
    agent_mask = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
        ],
        dtype=torch.bool,
    )

    embeddings, lens, targets, target_lens, loss_agent_mask = model.prepare_audio_channel_embeddings(
        audio_codes=audio_codes,
        audio_codes_lens=audio_codes_lens,
        delay=delay,
        agent_mask=agent_mask,
    )

    assert embeddings.shape == (2, int((delay + target_lens).max().item()), model.cfg.embedding_dim)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()
    assert lens.tolist() == (delay + target_lens).tolist()
    assert targets.shape == (2, model.num_audio_codebooks, int(target_lens.max().item()))
    assert target_lens.tolist() == [4, 3]
    assert loss_agent_mask.shape == (2, targets.size(2))
    assert loss_agent_mask.dtype == torch.bool


def test_forward_with_inputs_embeds(model):
    _seed_everything()
    inputs_embeds = torch.randn(2, 6, model.cfg.embedding_dim)
    attention_mask = torch.ones(2, 6, dtype=torch.bool)

    output = model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)

    assert output.last_hidden_state.shape == inputs_embeds.shape
    assert output.last_hidden_state.dtype == torch.float32
    assert torch.isfinite(output.last_hidden_state).all()
    assert output.past_key_values is not None


def test_logits_to_audio_codes_schema(model):
    logits = torch.zeros(2, 4, model.num_audio_codebooks * model.num_all_tokens_per_codebook)
    expected_tokens = []
    for codebook_idx in range(model.num_audio_codebooks):
        token_id = codebook_idx + 3
        expected_tokens.append(token_id)
        offset = codebook_idx * model.num_all_tokens_per_codebook
        logits[:, :, offset + token_id] = 5.0
    audio_codes_lens = torch.tensor([4, 2], dtype=torch.long)

    audio_codes = model.logits_to_audio_codes(logits, audio_codes_lens)

    assert audio_codes.shape == (2, model.num_audio_codebooks, 4)
    assert audio_codes.dtype == torch.long
    for codebook_idx, token_id in enumerate(expected_tokens):
        assert audio_codes[0, codebook_idx].tolist() == [token_id] * 4
    assert audio_codes[1, :, 2:].eq(0).all()


def test_process_batch_smoke(model, toy_batch):
    _seed_everything()
    output = model.process_batch(
        text=toy_batch["text"],
        text_lens=toy_batch["text_lens"],
        context_text_tokens=toy_batch["context_text_tokens"],
        context_text_tokens_lens=toy_batch["context_text_tokens_lens"],
        audio_codes=toy_batch["audio_codes"],
        audio_codes_lens=toy_batch["audio_codes_lens"],
        context_audio_codes=toy_batch["context_audio_codes"],
        context_audio_codes_lens=toy_batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=toy_batch["agent_mask"],
    )

    assert output.selected_training_mode == model.default_inference_mode
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert output.phoneme_loss is None
    assert output.local_transformer_loss is None
    assert output.logits.shape[:2] == output.audio_codes_target.shape[0::2]
    assert output.logits.shape[-1] == model.num_audio_codebooks * model.num_all_tokens_per_codebook
    assert output.audio_codes_target.dtype == torch.long
    assert output.context_audio_codes.shape[1] == model.num_audio_codebooks


def test_process_batch_with_multiturn_dataset_enabled():
    _seed_everything()
    model = _make_easy_magpie_model(tiny_easy_magpie_cfg({"use_multiturn_dataset": True}))
    batch = _toy_batch(model)
    text = batch["text"].clone()
    text[0, 0] = model.interruption_token_id

    output = model.process_batch(
        text=text,
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        task=batch["task"],
        agent_mask=batch["agent_mask"],
    )

    assert text[0, 0].item() == model.pad_id
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert output.local_transformer_loss is None


def test_process_batch_with_autoregressive_local_transformer():
    _seed_everything()
    model = _make_easy_magpie_model(
        tiny_easy_magpie_cfg(
            {
                "local_transformer_type": "autoregressive",
                "local_transformer_hidden_dim": 32,
                "local_transformer_n_layers": 1,
                "local_transformer_n_heads": 4,
                "local_transformer_loss_scale": 0.5,
            }
        )
    )
    batch = _toy_batch(model)

    output = model.process_batch(
        text=batch["text"],
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=batch["agent_mask"],
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert torch.isfinite(output.local_transformer_loss)
    assert output.local_transformer_logits is not None
    assert output.local_transformer_logits.shape == output.logits.shape


def test_training_step_smoke(model, toy_batch):
    _seed_everything()
    model.train()

    with patch.object(model, "log", lambda *args, **kwargs: None), patch.object(
        model, "log_dict", lambda *args, **kwargs: None
    ):
        loss = model.training_step(toy_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_validation_step_smoke(model, toy_batch, tmp_path):
    _seed_everything()
    model.eval()
    object.__setattr__(
        model,
        "_trainer",
        SimpleNamespace(world_size=1, global_rank=0, local_rank=0, log_dir=str(tmp_path), current_epoch=0),
    )

    with patch.object(model, "log_val_audio_example", lambda *args, **kwargs: {}):
        output = model.validation_step(toy_batch, batch_idx=1)

    assert set(output.keys()) == {"val_loss", "val_codebook_loss", "val_local_transformer_loss"}
    assert torch.isfinite(output["val_loss"])
    assert torch.isfinite(output["val_codebook_loss"])
    assert output["val_local_transformer_loss"] is None
    assert model.validation_step_outputs[-1] == output
