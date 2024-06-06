# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.submodules import adapters as adapter_modules
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.utils import config_utils


def _create_masks(att_mask, max_audio_length, padding_length):
    # pad_mask is the masking to be used to ignore paddings
    pad_mask = torch.arange(0, max_audio_length).expand(padding_length.size(0), -1) < padding_length.unsqueeze(-1)

    # pad_mask_for_att_mask is the mask which helps to ignore paddings
    pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
    pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
    # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
    att_mask = att_mask[:, :max_audio_length, :max_audio_length]
    # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
    att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))

    pad_mask = ~pad_mask
    att_mask = ~att_mask

    return pad_mask, att_mask


def get_mask(lengths: torch.Tensor):
    max_seq_len = lengths.max()
    att_mask = torch.ones(1, max_seq_len, max_seq_len, dtype=torch.bool)

    pad_mask, att_mask = _create_masks(att_mask, max_seq_len, lengths)
    return pad_mask, att_mask


class TestASRAdapterModules:
    @pytest.mark.unit
    def test_mha_adapter_config(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.MultiHeadAttentionAdapter,
            adapter_modules.MultiHeadAttentionAdapterConfig,
            ignore_args=IGNORED_ARGS,
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_relpos_mha_adapter_config(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.RelPositionMultiHeadAttentionAdapter,
            adapter_modules.RelPositionMultiHeadAttentionAdapterConfig,
            ignore_args=IGNORED_ARGS,
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_abs_pos_encoding_adapter_config(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.PositionalEncodingAdapter,
            adapter_modules.PositionalEncodingAdapterConfig,
            ignore_args=IGNORED_ARGS,
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_rel_pos_encoding_adapter_config(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.RelPositionalEncodingAdapter,
            adapter_modules.RelPositionalEncodingAdapterConfig,
            ignore_args=IGNORED_ARGS,
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    @pytest.mark.parametrize('n_head', [1, 2, 10])
    @pytest.mark.parametrize('proj_dim', [None, -1])
    def test_mha_adapter_init(self, n_head, proj_dim):
        torch.random.manual_seed(0)
        x = torch.randn(2, 32, 50)
        lengths = torch.randint(1, x.size(1), size=(x.size(0),))
        lengths[torch.randint(0, x.size(0), size=(1,))[0]] = x.size(1)

        adapter = adapter_modules.MultiHeadAttentionAdapter(
            n_head=n_head, n_feat=50, dropout_rate=0.0, proj_dim=proj_dim
        )

        pad_mask, att_mask = get_mask(lengths)

        with torch.no_grad():
            assert adapter.linear_out.weight.sum() == 0
            if hasattr(adapter.linear_out, 'bias') and adapter.linear_out.bias is not None:
                assert adapter.linear_out.bias.sum() == 0

            out = adapter(x, x, x, att_mask)
            assert out.sum().abs() <= 1e-8
            assert out.shape == x.shape

    @pytest.mark.unit
    @pytest.mark.parametrize('n_head', [1, 2, 10])
    @pytest.mark.parametrize('proj_dim', [None, -1])
    def test_relmha_adapter_init(self, n_head, proj_dim):
        torch.random.manual_seed(0)
        x = torch.randn(2, 32, 50)
        lengths = torch.randint(1, x.size(1), size=(x.size(0),))
        lengths[torch.randint(0, x.size(0), size=(1,))[0]] = x.size(1)

        adapter = adapter_modules.RelPositionMultiHeadAttentionAdapter(
            n_head=n_head, n_feat=50, dropout_rate=0.0, proj_dim=proj_dim
        )
        relpos_enc = adapter_modules.RelPositionalEncodingAdapter(d_model=50)

        pad_mask, att_mask = get_mask(lengths)
        relpos_enc.extend_pe(lengths.max(), device='cpu', dtype=torch.float32)

        with torch.no_grad():
            assert adapter.linear_out.weight.sum() == 0
            if hasattr(adapter.linear_out, 'bias') and adapter.linear_out.bias is not None:
                assert adapter.linear_out.bias.sum() == 0

            _, pos_emb = relpos_enc(x)
            out = adapter(x, x, x, att_mask, pos_emb)
            assert out.sum().abs() <= 1e-8
            assert out.shape == x.shape

    @pytest.mark.unit
    def test_abspos_encoding_init(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 32, 50)
        lengths = torch.randint(1, x.size(1), size=(x.size(0),))
        lengths[torch.randint(0, x.size(0), size=(1,))[0]] = x.size(1)

        relpos_enc = adapter_modules.PositionalEncodingAdapter(d_model=50)

        relpos_enc.extend_pe(lengths.max(), device='cpu', dtype=torch.float32)

        with torch.no_grad():
            out, pos_emb = relpos_enc(x)
            assert (out - pos_emb - x).sum().abs() <= 1e-5
            assert out.shape == x.shape

    @pytest.mark.unit
    def test_relpos_encoding_init(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 32, 50)
        lengths = torch.randint(1, x.size(1), size=(x.size(0),))
        lengths[torch.randint(0, x.size(0), size=(1,))[0]] = x.size(1)

        relpos_enc = adapter_modules.RelPositionalEncodingAdapter(d_model=50)

        relpos_enc.extend_pe(lengths.max(), device='cpu', dtype=torch.float32)

        with torch.no_grad():
            out, pos_emb = relpos_enc(x)
            assert (out - x).sum().abs() <= 1e-8
            assert out.shape == x.shape

    @pytest.mark.unit
    def test_mha_adapter_strategy(self):
        adapter = adapter_modules.MultiHeadAttentionAdapter(n_head=1, n_feat=50, dropout_rate=0.0)
        assert hasattr(adapter, 'adapter_strategy')
        assert adapter.adapter_strategy is not None
        # assert default strategy is set
        assert isinstance(adapter.adapter_strategy, adapter_modules.MHAResidualAddAdapterStrategy)

    @pytest.mark.unit
    def test_relpos_mha_adapter_strategy(self):
        adapter = adapter_modules.RelPositionMultiHeadAttentionAdapter(n_head=1, n_feat=50, dropout_rate=0.0)
        assert hasattr(adapter, 'adapter_strategy')
        assert adapter.adapter_strategy is not None
        # assert default strategy is set
        assert isinstance(adapter.adapter_strategy, adapter_modules.MHAResidualAddAdapterStrategy)

    @pytest.mark.unit
    def test_abspos_encoding_adapter_strategy(self):
        adapter = adapter_modules.PositionalEncodingAdapter(d_model=50)
        assert hasattr(adapter, 'adapter_strategy')
        assert adapter.adapter_strategy is not None
        # assert default strategy is set
        assert isinstance(adapter.adapter_strategy, adapter_mixin_strategies.ReturnResultAdapterStrategy)

    @pytest.mark.unit
    def test_relpos_encoding_adapter_strategy(self):
        adapter = adapter_modules.RelPositionalEncodingAdapter(d_model=50)
        assert hasattr(adapter, 'adapter_strategy')
        assert adapter.adapter_strategy is not None
        # assert default strategy is set
        assert isinstance(adapter.adapter_strategy, adapter_mixin_strategies.ReturnResultAdapterStrategy)
