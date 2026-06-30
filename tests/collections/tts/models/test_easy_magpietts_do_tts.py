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
