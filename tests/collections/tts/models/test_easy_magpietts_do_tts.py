# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Unit tests for EasyMagpieTTSInferenceModel.do_tts local-transformer selection.

With use_local_transformer=None, do_tts derives it from local_transformer_type (AR -> use it;
MASKGIT/NO_LT -> parallel sampling, since EasyMagpie only supports an AR local transformer and raises
otherwise); an explicit value overrides that. The test drives the real do_tts with a mock self and
asserts the use_local_transformer_for_inference passed to infer_batch.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBaseV2
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType


def _make_easy_mock_model(local_transformer_type):
    """An EasyMagpieTTSInferenceModel mock wired enough for do_tts to reach the infer_batch call."""
    model = MagicMock(spec=EasyMagpieTTSInferenceModel)
    model.local_transformer_type = local_transformer_type
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])  # do_tts reads device off a param
    model.cfg = MagicMock()
    model.cfg.text_tokenizers = {"english_phoneme": object()}
    model.tokenizer = MagicMock()
    model.tokenizer.tokenizers = {"english_phoneme": object()}
    model.tokenizer.encode.return_value = [1, 2, 3]
    model.enable_phoneme_text_input = False
    model.phoneme_tokenizer = None
    model.text_phoneme_token_offset = None
    model.phoneme_text_bop_marker = "<bop>"
    model.phoneme_text_eop_marker = "<eop>"
    model.eos_id = 0
    model.text_conditioning_tokenizer_name = "text_ce_tokenizer"
    model.data_num_audio_codebooks = 4
    model.infer_batch.return_value = SimpleNamespace(
        predicted_audio=torch.zeros(1, 1), predicted_audio_lens=torch.zeros(1, dtype=torch.long)
    )
    return model


def _local_transformer_flag_passed_to_infer_batch(model):
    model.infer_batch.assert_called_once()
    _args, kwargs = model.infer_batch.call_args
    return kwargs["use_local_transformer_for_inference"]


class TestEasyDoTtsLocalTransformerSelection:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "local_transformer_type, use_local_transformer, expected_use_lt",
        [
            # use_local_transformer=None -> derive from local_transformer_type (EasyMagpie uses the
            # local transformer only for AR; MASKGIT/NO_LT decode via parallel sampling).
            (LocalTransformerType.NO_LT, None, False),
            (LocalTransformerType.AR, None, True),
            (LocalTransformerType.MASKGIT, None, False),
            # An explicit value overrides the derived default -- e.g. False flips AR's derived True.
            (LocalTransformerType.AR, True, True),
            (LocalTransformerType.AR, False, False),
            (LocalTransformerType.MASKGIT, False, False),
        ],
    )
    def test_easy_do_tts_local_transformer_selection(
        self, local_transformer_type, use_local_transformer, expected_use_lt
    ):
        """do_tts derives use_local_transformer from local_transformer_type when None, else honors the explicit value."""
        model = _make_easy_mock_model(local_transformer_type)

        EasyMagpieTTSInferenceModel.do_tts(model, "hello world", use_local_transformer=use_local_transformer)

        assert _local_transformer_flag_passed_to_infer_batch(model) is expected_use_lt


class TestEasyDoTtsPrecomputedContext:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_passes_precomputed_context_to_infer_batch(self):
        model = _make_easy_mock_model(LocalTransformerType.AR)
        context_embedding = torch.randn(7, 1536)

        EasyMagpieTTSInferenceModel.do_tts(
            model,
            "hello world",
            context_text="Speaker: Example",
            precomputed_context_audio_embedding=context_embedding,
        )

        _args, kwargs = model.infer_batch.call_args
        batch = kwargs["batch"]
        assert batch["precomputed_context_audio_embedding"].shape == (1, 7, 1536)
        assert batch["precomputed_context_audio_embedding_lens"].tolist() == [7]
        assert "context_audio_codes" not in batch
        assert batch["context_text_tokens"].tolist() == [[1, 2, 3]]
        model.tokenizer.encode.assert_any_call(
            "Speaker: Example", tokenizer_name=model.text_conditioning_tokenizer_name
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_rejects_raw_and_precomputed_context(self):
        model = _make_easy_mock_model(LocalTransformerType.AR)

        with pytest.raises(ValueError, match="context_audio_file_path"):
            EasyMagpieTTSInferenceModel.do_tts(
                model,
                "hello world",
                context_audio_file_path="context.wav",
                precomputed_context_audio_embedding=torch.randn(7, 1536),
            )


class TestNemotronTTSV2PrecomputedContext:
    @staticmethod
    def _make_model():
        model = MagicMock(spec=NemotronTTSBaseV2)
        model.parameters.side_effect = lambda: iter([torch.zeros(1)])
        model.cfg = MagicMock()
        model.cfg.embedding_dim = 8
        model.cfg.text_tokenizers = {"english_phoneme": object()}
        model.tokenizer = MagicMock()
        model.tokenizer.tokenizers = {"english_phoneme": object()}
        model.tokenizer.encode.return_value = [1, 2, 3]
        model.enable_phoneme_text_input = False
        model.phoneme_tokenizer = None
        model.text_phoneme_token_offset = None
        model.phoneme_text_bop_marker = "<bop>"
        model.phoneme_text_eop_marker = "<eop>"
        model.eos_id = 0
        model.text_conditioning_tokenizer_name = "text_ce_tokenizer"
        model.data_num_audio_codebooks = 4
        model._codec_helper = MagicMock()
        model.infer_batch.return_value = SimpleNamespace(
            predicted_audio=torch.zeros(1, 1), predicted_audio_lens=torch.zeros(1, dtype=torch.long)
        )
        return model

    @pytest.mark.unit
    def test_do_tts_passes_precomputed_context_without_v1_options(self):
        model = self._make_model()
        context_embedding = torch.randn(7, 8)

        NemotronTTSBaseV2.do_tts(
            model,
            "hello world",
            context_text="Speaker: Example",
            precomputed_context_audio_embedding=context_embedding,
        )

        _args, kwargs = model.infer_batch.call_args
        batch = kwargs["batch"]
        assert batch["precomputed_context_audio_embedding"].shape == (1, 7, 8)
        assert batch["precomputed_context_audio_embedding_lens"].tolist() == [7]
        assert "context_audio_codes" not in batch
        assert "use_local_transformer_for_inference" not in kwargs

    @pytest.mark.unit
    def test_do_tts_rejects_raw_and_precomputed_context(self):
        model = self._make_model()

        with pytest.raises(ValueError, match="context_audio_file_path"):
            NemotronTTSBaseV2.do_tts(
                model,
                "hello world",
                context_audio_file_path="context.wav",
                precomputed_context_audio_embedding=torch.randn(7, 8),
            )

    @pytest.mark.unit
    def test_prepare_context_bypasses_codec_and_speaker_encoder(self):
        model = self._make_model()
        model.training_modes = [SimpleNamespace(mode_idx=None)]
        model.task_embedding = None
        model.use_speaker_encoder = True
        model.embed_text_tokens.return_value = torch.zeros(2, 3, 8)
        model._validate_precomputed_context_audio.side_effect = (
            lambda embedding, lengths, batch_size: NemotronTTSBaseV2._validate_precomputed_context_audio(
                model, embedding, lengths, batch_size
            )
        )
        model.join_embeddings_temporally.side_effect = lambda embeddings, lengths: (
            torch.cat(embeddings, dim=1),
            torch.stack(lengths).sum(dim=0),
        )
        context_embedding = torch.randn(2, 4, 8)
        context_embedding_lens = torch.tensor([4, 2])

        combined, combined_lens, context_codes, context_code_lens = NemotronTTSBaseV2.prepare_context_tensors(
            model,
            context_text_tokens=torch.ones(2, 3, dtype=torch.long),
            context_text_tokens_lens=torch.tensor([3, 2]),
            precomputed_context_audio_embedding=context_embedding,
            precomputed_context_audio_embedding_lens=context_embedding_lens,
        )

        assert combined.shape == (2, 7, 8)
        assert combined_lens.tolist() == [7, 4]
        assert context_codes.shape == (2, 4, 0)
        assert context_code_lens.tolist() == [0, 0]
        model._codec_helper.audio_to_codes.assert_not_called()
        model.encode_context_audio_embeddings.assert_not_called()
