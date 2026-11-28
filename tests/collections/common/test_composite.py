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

import pytest
import torch

from nemo.collections.common.losses import CompositeLoss, LossTerm
from nemo.collections.common.losses import composite as composite_module


def _terms():
    return {
        "ce": LossTerm(torch.tensor(12.0), torch.tensor(4.0)),
        "ctc": LossTerm(torch.tensor(30.0), torch.tensor(5.0)),
    }


class TestLossTerm:
    @pytest.mark.unit
    def test_from_elements_counts_the_valid_elements(self):
        losses = torch.tensor([[1.0, 2.0, 100.0], [3.0, 100.0, 100.0]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        term = LossTerm.from_elements(losses, mask)
        assert term.summed.item() == 6.0 and term.count.item() == 3.0
        whole = LossTerm.from_elements(losses)
        assert whole.summed.item() == 306.0 and whole.count.item() == 6.0

    @pytest.mark.unit
    def test_from_sequences_counts_the_lengths(self):
        term = LossTerm.from_sequences(torch.tensor([4.0, 6.0]), torch.tensor([3, 7]))
        assert term.summed.item() == 10.0 and term.count.item() == 10.0


class TestCompositeLoss:
    @pytest.mark.unit
    def test_total_is_the_scaled_sum_of_the_means(self):
        loss = CompositeLoss(scales={"ctc": 0.5})
        total, reported = loss(_terms())
        assert total.item() == pytest.approx(12.0 / 4.0 + 0.5 * 30.0 / 5.0)
        assert reported["ce"].item() == 3.0 and reported["ctc"].item() == 6.0
        assert not reported["ce"].requires_grad

    @pytest.mark.unit
    def test_hybrid_objective_with_six_terms(self):
        """Four CTC heads counted in target labels and two cross-entropy heads counted in tokens with EOS."""
        torch.manual_seed(3)
        target_lengths = torch.tensor([47, 5, 23])
        ctc_terms = {
            f"ctc_{layer}": LossTerm.from_sequences(torch.rand(3) * 50, target_lengths) for layer in (4, 10, 16, 24)
        }
        token_mask = torch.arange(48)[None, :] <= target_lengths[:, None]
        ce_terms = {
            name: LossTerm.from_elements(torch.rand(3, 48).requires_grad_(), token_mask) for name in ("ce", "ce_3")
        }
        terms = {**ctc_terms, **ce_terms}
        total, reported = CompositeLoss(scales={name: 1.0 for name in terms})(terms)
        for name, term in terms.items():
            assert term.count.item() == (75.0 if name.startswith("ctc") else 78.0)
            torch.testing.assert_close(reported[name], term.summed.detach() / term.count)
        torch.testing.assert_close(total, sum(reported.values()))
        total.backward()

    @pytest.mark.unit
    def test_scale_zero_reports_without_training(self):
        terms = _terms()
        terms["ce"] = LossTerm(terms["ce"].summed.clone().requires_grad_(), terms["ce"].count)
        total, reported = CompositeLoss(scales={"ce": 0.0})(terms)
        assert total.item() == 6.0 and reported["ce"].item() == 3.0
        total.backward()
        assert terms["ce"].summed.grad.item() == 0.0

    @pytest.mark.unit
    def test_unknown_scale_names_and_normalizations_are_rejected(self):
        with pytest.raises(ValueError, match="not provided"):
            CompositeLoss(scales={"rnnt": 1.0})(_terms())
        with pytest.raises(ValueError, match="normalization"):
            CompositeLoss(normalization="mean")
        with pytest.raises(ValueError, match="at least one"):
            CompositeLoss()({})

    @pytest.mark.unit
    def test_empty_term_contributes_nothing(self):
        total, reported = CompositeLoss()({"ce": LossTerm(torch.tensor(0.0), torch.tensor(0.0))})
        assert total.item() == 0.0 and reported["ce"].item() == 0.0

    @pytest.mark.unit
    def test_rank_normalizations_differ_with_unequal_counts(self):
        """The mean of per-rank means is not the global mean; both forms are derived by hand here."""
        rank_terms = [
            {"ce": LossTerm(torch.tensor(10.0), torch.tensor(2.0))},
            {"ce": LossTerm(torch.tensor(10.0), torch.tensor(8.0))},
        ]
        per_rank = [CompositeLoss()(terms)[0].item() for terms in rank_terms]
        assert sum(per_rank) / 2 == pytest.approx((10.0 / 2.0 + 10.0 / 8.0) / 2)
        global_mean = (10.0 + 10.0) / (2.0 + 8.0)
        assert sum(per_rank) / 2 != pytest.approx(global_mean)

    @pytest.mark.unit
    def test_global_normalization_uses_the_all_reduced_counts(self, monkeypatch):
        """Two ranks are simulated: rank 0 sees sum 10 over 2 elements, rank 1 sum 10 over 8 elements."""
        other = {"ce": (10.0, 8.0)}

        def fake_all_reduce(tensor, op):
            assert op == torch.distributed.ReduceOp.SUM
            half = tensor.numel() // 2
            tensor[:half] += torch.tensor([other[name][0] for name in names])
            tensor[half:] += torch.tensor([other[name][1] for name in names])

        names = ["ce"]
        monkeypatch.setattr(composite_module, "_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        summed = torch.tensor(10.0, requires_grad=True)
        total, reported = CompositeLoss(normalization="global")({"ce": LossTerm(summed, torch.tensor(2.0))})
        # The local loss is the local sum times the world size over the global count 10,
        # so the data-parallel average of both ranks' losses is the global mean 20 / 10 = 2.
        assert total.item() == pytest.approx(2 * 10.0 / 10.0)
        assert reported["ce"].item() == pytest.approx(2.0)
        total.backward()
        assert summed.grad.item() == pytest.approx(2 / 10.0)
