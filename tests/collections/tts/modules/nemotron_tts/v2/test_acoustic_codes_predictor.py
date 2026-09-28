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

import inspect
from dataclasses import fields

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig
from nemo.collections.tts.modules.nemotron_tts.v2 import AcousticCodesPredictor


pytestmark = pytest.mark.unit

NUM_CODES = 8
CODEBOOK_SIZE = 32
NUM_SPECIAL_TOKENS = 8
D_MODEL = 16
AUDIO_EOS_ID = CODEBOOK_SIZE + 1
MASK_TOKEN_ID = CODEBOOK_SIZE + 4
SCHEDULE = (2, 6)


class _CodeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(CODEBOOK_SIZE + NUM_SPECIAL_TOKENS, D_MODEL) for _ in range(NUM_CODES)]
        )
        self.calls = []

    def forward(self, codes, codebook_indices=None):
        if codebook_indices is None:
            codebook_indices = tuple(range(codes.size(1)))
        else:
            codebook_indices = tuple(codebook_indices)
        assert len(codebook_indices) == codes.size(1)
        self.calls.append((codebook_indices, codes.detach().clone()))

        embedded = self.embeddings[codebook_indices[0]](codes[:, 0, :])
        for input_idx, codebook_idx in enumerate(codebook_indices[1:], start=1):
            embedded = embedded + self.embeddings[codebook_idx](codes[:, input_idx, :])
        return embedded / codes.size(1)


def _backbone_config():
    return NemotronHConfig(
        hidden_size=D_MODEL,
        num_hidden_layers=3,
        hybrid_override_pattern="M*-",
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=True,
        intermediate_size=48,
        mamba_num_heads=4,
        mamba_head_dim=4,
        ssm_state_size=8,
        n_groups=2,
        _attn_implementation="sdpa",
    )


def _make_predictor(**overrides):
    torch.manual_seed(0)
    kwargs = {
        "backbone_config": _backbone_config(),
        "embed_codes": _CodeEmbedder(),
        "num_audio_codebooks": NUM_CODES,
        "audio_eos_id": AUDIO_EOS_ID,
        "mask_token_id": MASK_TOKEN_ID,
        "codebook_size": CODEBOOK_SIZE,
        "prediction_schedule": SCHEDULE,
        "n_layers": 2,
    }
    kwargs.update(overrides)
    predictor = AcousticCodesPredictor(**kwargs)
    predictor.eval()
    return predictor


def _cache(predictor, batch_size):
    return predictor.make_cache(batch_size, device=torch.device("cpu"), dtype=torch.float32)


def test_blocks_reuse_the_complete_backbone_config():
    backbone_config = _backbone_config()
    predictor = _make_predictor(backbone_config=backbone_config)

    for block in predictor.blocks:
        for config_field in fields(NemotronHConfig):
            if config_field.name not in {"num_hidden_layers", "hybrid_override_pattern"}:
                assert getattr(block.config, config_field.name) == getattr(backbone_config, config_field.name)
        assert block.config.num_hidden_layers == 4
        assert block.config.hybrid_override_pattern == "*-*-"
        assert [layer.block_type for layer in block.layers] == ["attention", "mlp", "attention", "mlp"]

    assert [block.codebook_indices for block in predictor.blocks] == [(0, 1), (2, 3, 4, 5, 6, 7)]
    assert [block.codebook_projection.out_features for block in predictor.blocks] == [66, 198]
    assert not hasattr(predictor, "out_proj")


def test_schedule_must_cover_every_code_channel():
    with pytest.raises(AssertionError, match="expected 8"):
        _make_predictor(prediction_schedule=(2, 5))


def test_target_channels_must_match_schedule():
    predictor = _make_predictor()
    with pytest.raises(AssertionError, match="target has 7 codes"):
        predictor.compute_loss(
            hidden_states=torch.randn(2, 3, D_MODEL),
            target_codes=torch.zeros(2, 3, NUM_CODES - 1, dtype=torch.long),
            lengths=torch.tensor([3, 3]),
        )


def test_loss_includes_audio_eos_and_excludes_other_special_tokens():
    predictor = _make_predictor()
    with torch.no_grad():
        for block in predictor.blocks:
            block.codebook_projection.weight.zero_()
            block.codebook_projection.bias.zero_()

    hidden_states = torch.randn(2, 3, D_MODEL)
    lengths = torch.tensor([3, 3])
    eos_targets = torch.full((2, 3, NUM_CODES), AUDIO_EOS_ID)
    eos_loss = predictor.compute_loss(hidden_states, eos_targets, lengths)
    torch.testing.assert_close(eos_loss, torch.tensor(float(torch.log(torch.tensor(CODEBOOK_SIZE + 1)))))

    masked_targets = torch.full((2, 3, NUM_CODES), MASK_TOKEN_ID)
    assert predictor.compute_loss(hidden_states, masked_targets, lengths) == 0.0


def test_loss_mask_controls_supervision_without_hiding_teacher_forced_codes():
    predictor = _make_predictor()
    target_codes = torch.randint(0, CODEBOOK_SIZE, (2, 3, NUM_CODES))
    hidden_states = torch.randn(2, 3, D_MODEL)
    lengths = torch.tensor([3, 3])
    loss_mask = torch.zeros(2, 3, dtype=torch.bool)

    predictor.embed_codes.calls.clear()
    loss = predictor.compute_loss(hidden_states, target_codes, lengths, loss_mask=loss_mask)

    assert loss == 0.0
    assert len(predictor.embed_codes.calls) == 1
    codebook_indices, embedded_codes = predictor.embed_codes.calls[0]
    assert codebook_indices == predictor.blocks[0].codebook_indices
    torch.testing.assert_close(
        embedded_codes,
        target_codes[:, :, : SCHEDULE[0]].transpose(1, 2),
    )


def test_next_block_embeds_only_codes_predicted_by_the_previous_block():
    predictor = _make_predictor()
    embedder = predictor.embed_codes
    hidden_states = torch.randn(2, 3, D_MODEL)

    codes = predictor.predict_codes(hidden_states, temperature=0.0)

    assert len(embedder.calls) == 1
    codebook_indices, embedded_codes = embedder.calls[0]
    assert codebook_indices == predictor.blocks[0].codebook_indices
    torch.testing.assert_close(
        embedded_codes,
        codes[:, :, : SCHEDULE[0]].transpose(1, 2),
    )


def test_prediction_returns_only_codec_tokens_or_audio_eos():
    predictor = _make_predictor()
    codes = predictor.predict_codes(torch.randn(3, 2, D_MODEL), temperature=0.7, topk=80)

    assert codes.shape == (3, 2, NUM_CODES)
    assert (((codes >= 0) & (codes < CODEBOOK_SIZE)) | (codes == AUDIO_EOS_ID)).all()


def test_predictor_does_not_take_backbone_code_predictions():
    predictor = _make_predictor()
    assert "pred_codes" not in inspect.signature(predictor.predict_codes).parameters
    codes = predictor.predict_codes(torch.randn(2, 1, D_MODEL), temperature=0.0)
    assert codes.shape == (2, 1, NUM_CODES)


def test_cached_streaming_matches_whole_sequence_prediction():
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 6, D_MODEL)

    whole = _make_predictor().predict_codes(hidden_states, temperature=0.0)

    predictor = _make_predictor()
    cache = _cache(predictor, batch_size=2)
    streamed = torch.cat(
        [
            predictor.predict_codes(hidden_states[:, frame : frame + 1], cache=cache, temperature=0.0)
            for frame in range(hidden_states.size(1))
        ],
        dim=1,
    )

    assert AcousticCodesPredictor.cached_frames(cache) == hidden_states.size(1)
    torch.testing.assert_close(streamed, whole)


def test_skipped_prediction_populates_the_same_cache_as_advance():
    torch.manual_seed(2)
    hidden_states = torch.randn(1, 3, D_MODEL)

    advanced = _make_predictor()
    advanced_cache = _cache(advanced, batch_size=1)
    advanced.advance(hidden_states[:, :2], cache=advanced_cache)
    after_advance = advanced.predict_codes(hidden_states[:, 2:], cache=advanced_cache, temperature=0.0)

    skipped = _make_predictor()
    skipped_cache = _cache(skipped, batch_size=1)
    for frame in range(2):
        codes = skipped.predict_codes(
            hidden_states[:, frame : frame + 1],
            cache=skipped_cache,
            temperature=0.0,
            predict=torch.tensor([False]),
        )
        assert (codes == MASK_TOKEN_ID).all()
    after_skips = skipped.predict_codes(hidden_states[:, 2:], cache=skipped_cache, temperature=0.0)

    assert AcousticCodesPredictor.cached_frames(advanced_cache) == 3
    assert AcousticCodesPredictor.cached_frames(skipped_cache) == 3
    torch.testing.assert_close(after_skips, after_advance)


def test_cfg_uses_doubled_hidden_states_and_returns_one_stream():
    predictor = _make_predictor()
    codes = predictor.predict_codes(
        hidden_states=torch.randn(6, 1, D_MODEL),
        temperature=0.0,
        use_cfg=True,
        cfg_scale=2.5,
    )
    assert codes.shape == (3, 1, NUM_CODES)


def test_flash_attention_config_handles_a_batched_suffix_behind_a_warm_cache():
    torch.manual_seed(3)
    hidden_states = torch.randn(1, 6, D_MODEL)
    backbone_config = _backbone_config()
    backbone_config._attn_implementation = "flash_attention_2"

    whole = _make_predictor(backbone_config=backbone_config).predict_codes(hidden_states, temperature=0.0)

    predictor = _make_predictor(backbone_config=backbone_config)
    cache = _cache(predictor, batch_size=1)
    prefix = predictor.predict_codes(hidden_states[:, :2], cache=cache, temperature=0.0)
    suffix = predictor.predict_codes(hidden_states[:, 2:], cache=cache, temperature=0.0)

    assert AcousticCodesPredictor.cached_frames(cache) == hidden_states.size(1)
    torch.testing.assert_close(torch.cat([prefix, suffix], dim=1), whole)
