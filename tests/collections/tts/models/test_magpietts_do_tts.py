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

"""
Unit tests for MagpieTTSModel.do_tts local-transformer selection.

do_tts must use local transformer iff the model has one (local_transformer_type
!= 'none'); otherwise generate_speech raises. The test drives the real do_tts with a mock self and
asserts the use_local_transformer_for_inference passed to generate_speech.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.collections.tts.models.magpietts import MagpieTTSModel
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType


def _make_mock_model(local_transformer_type):
    """A MagpieTTSModel mock with just enough wired for do_tts to reach the generate_speech call."""
    model = MagicMock(spec=MagpieTTSModel)
    model.has_baked_context_embedding = True
    model.local_transformer_type = local_transformer_type
    model.device = torch.device("cpu")
    model.eos_id = 0
    model.tokenizer = MagicMock()
    model.tokenizer.tokenizers = {"english_phoneme": object()}
    # Zero-length predicted codes so do_tts skips codec decoding and returns silence.
    model.generate_speech.return_value = SimpleNamespace(
        predicted_codes=torch.zeros(1, 1, 0, dtype=torch.long),
        predicted_codes_lens=torch.zeros(1, dtype=torch.long),
    )
    return model


def _local_transformer_flag_passed_to_generate_speech(model):
    model.generate_speech.assert_called_once()
    _args, kwargs = model.generate_speech.call_args
    return kwargs["use_local_transformer_for_inference"]


class TestDoTtsLocalTransformerSelection:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "local_transformer_type, expected_use_lt",
        [
            # NO_LT has no local transformer -> do_tts must NOT request it (else generate_speech raises).
            (LocalTransformerType.NO_LT, False),
            # AR and MASKGIT both have a local transformer -> do_tts must request it.
            (LocalTransformerType.AR, True),
            (LocalTransformerType.MASKGIT, True),
        ],
    )
    @patch("nemo.collections.tts.models.magpietts.chunk_text_for_inference")
    @patch("nemo.collections.tts.models.magpietts.get_tokenizer_for_language")
    def test_do_tts_requests_local_transformer_iff_model_has_one(
        self, mock_get_tok, mock_chunk, local_transformer_type, expected_use_lt
    ):
        """do_tts must derive use_local_transformer_for_inference from the model's local_transformer_type."""
        mock_get_tok.return_value = "english_phoneme"
        mock_chunk.return_value = ([torch.zeros(3, dtype=torch.long)], [3], None)
        model = _make_mock_model(local_transformer_type)

        MagpieTTSModel.do_tts(model, "hello world", language="en")

        assert _local_transformer_flag_passed_to_generate_speech(model) is expected_use_lt
