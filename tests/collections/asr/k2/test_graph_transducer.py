# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy

try:
    from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss
    from nemo.core.utils.k2_guard import k2
except (ImportError, ModuleNotFoundError):
    pytest.skip("k2 is not installed, skipping Graph-RNNT tests.", allow_module_level=True)

EPS_SM_INPUT = 1e-6
EPS_L_INPUT = 1e-4

DEVICES = ['cpu']

if torch.cuda.is_available() and k2.with_cuda:
    DEVICES.append('cuda')


class TestGraphRnnt:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    @pytest.mark.parametrize("num_frames", [1, 3, 6])
    @pytest.mark.parametrize("vocab_size", [3])
    def test_temporal_schema(self, device, blank_first, num_frames, vocab_size):
        blank_id = 0 if blank_first else vocab_size - 1
        loss = GraphRnntLoss(blank=blank_id)
        temporal_schema = loss.get_temporal_schema(
            num_frames=num_frames, vocab_size=vocab_size, device=torch.device(device)
        )

        etalon_schema_fst: List[List[int]] = []
        for time_i in range(num_frames):
            for label_i in range(vocab_size):
                if label_i == blank_id:
                    # transition to the next state
                    etalon_schema_fst.append([time_i, time_i + 1, label_i, time_i, 0])
                else:
                    # self-loop
                    etalon_schema_fst.append([time_i, time_i, label_i, time_i, 0])
        etalon_schema_fst.append([num_frames, num_frames + 1, -1, -1, 0])  # transition to final state
        etalon_schema_fst.append([num_frames + 1])  # final state
        etalon_schema_fst = sorted(etalon_schema_fst)  # required for k2.Fsa.from_str
        etalon_schema_fst_str = "\n".join([" ".join(map(str, line)) for line in etalon_schema_fst])
        etalon_temporal_schema = k2.Fsa.from_str(etalon_schema_fst_str, num_aux_labels=1)

        assert temporal_schema.num_arcs == etalon_temporal_schema.num_arcs
        assert temporal_schema.shape == etalon_temporal_schema.shape  # (num_states, None)
        assert k2.is_rand_equivalent(
            temporal_schema, etalon_temporal_schema, log_semiring=True, treat_epsilons_specially=False
        ), "Temporal schema mismatch"
        assert k2.is_rand_equivalent(
            temporal_schema.invert(),
            etalon_temporal_schema.invert(),
            log_semiring=True,
            treat_epsilons_specially=False,
        ), "Temporal schema output labels mismatch"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_unit_schema(self, device, blank_first):
        vocab_size = 3
        blank_id = 0 if blank_first else vocab_size - 1
        if blank_first:
            labels = [1, 1, 2, 1]
        else:
            labels = [1, 1, 0, 1]
        loss = GraphRnntLoss(blank=blank_id)
        unit_schema = loss.get_unit_schema(
            units_tensor=torch.tensor(labels, device=torch.device(device)), vocab_size=vocab_size
        )

        etalon_schema_fst: List[List[int]] = []
        for label_i, label in enumerate(labels):
            etalon_schema_fst.append([label_i, label_i + 1, label, label, label_i, 0])  # forward: label
            etalon_schema_fst.append([label_i, label_i, blank_id, blank_id, label_i, 0])  # self-loop: blank
        etalon_schema_fst.append([len(labels), len(labels), blank_id, blank_id, len(labels), 0])
        etalon_schema_fst.append([len(labels), len(labels) + 1, -1, -1, -1, 0])  # transition to final state
        etalon_schema_fst.append([len(labels) + 1])  # final state
        etalon_schema_fst = sorted(etalon_schema_fst)  # required for k2.Fsa.from_str
        etalon_schema_fst_str = "\n".join([" ".join(map(str, line)) for line in etalon_schema_fst])
        etalon_unit_schema = k2.Fsa.from_str(etalon_schema_fst_str, aux_label_names=["aux_labels", "unit_positions"])

        assert unit_schema.num_arcs == etalon_unit_schema.num_arcs
        assert unit_schema.shape == etalon_unit_schema.shape  # (num_states, None)
        assert k2.is_rand_equivalent(
            unit_schema, etalon_unit_schema, log_semiring=True, treat_epsilons_specially=False
        ), "Unit schema input labels mismatch"
        assert k2.is_rand_equivalent(
            unit_schema.invert(), etalon_unit_schema.invert(), log_semiring=True, treat_epsilons_specially=False
        ), "Unit schema output labels mismatch"

        # swap aux_labels and unit positions to test unit_positions
        unit_schema.aux_labels, unit_schema.unit_positions = unit_schema.unit_positions, unit_schema.aux_labels
        etalon_unit_schema.aux_labels, etalon_unit_schema.unit_positions = (
            etalon_unit_schema.unit_positions,
            etalon_unit_schema.aux_labels,
        )
        assert k2.is_rand_equivalent(
            unit_schema.invert(), etalon_unit_schema.invert(), log_semiring=True, treat_epsilons_specially=False
        ), "Unit schema unit positions mismatch"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_grid_schema(self, device, blank_first):
        vocab_size = 3
        blank_id = 0 if blank_first else vocab_size - 1
        if blank_first:
            labels = [1, 1, 2, 1]
        else:
            labels = [1, 1, 0, 1]
        text_length = len(labels)
        num_frames = 5
        loss = GraphRnntLoss(blank=blank_id)
        grid_schema = loss.get_grid(
            units_tensor=torch.tensor(labels, device=torch.device(device)),
            num_frames=num_frames,
            vocab_size=vocab_size,
        )

        etalon_schema_fst: List[List[int]] = []
        for frame_i in range(num_frames):
            for label_i in range(text_length + 1):
                state = frame_i * (text_length + 1) + label_i
                if label_i < text_length:
                    next_state_label = state + 1
                    # next unit
                    etalon_schema_fst.append([state, next_state_label, labels[label_i], frame_i, label_i, 0])
                if frame_i < num_frames - 1:
                    next_state_frame = (frame_i + 1) * (text_length + 1) + label_i
                    # next time frame (blank)
                    etalon_schema_fst.append([state, next_state_frame, blank_id, frame_i, label_i, 0])

        last_grid_state = num_frames * (text_length + 1) - 1
        etalon_schema_fst.append([last_grid_state, last_grid_state + 1, blank_id, num_frames - 1, text_length, 0])
        etalon_schema_fst.append(
            [last_grid_state + 1, last_grid_state + 2, -1, -1, -1, 0]
        )  # transition to final state
        etalon_schema_fst.append([last_grid_state + 2])  # final state
        etalon_schema_fst = sorted(etalon_schema_fst)  # required for k2.Fsa.from_str
        etalon_schema_fst_str = "\n".join([" ".join(map(str, line)) for line in etalon_schema_fst])
        etalon_grid_schema = k2.Fsa.from_str(etalon_schema_fst_str, aux_label_names=["aux_labels", "unit_positions"])

        assert grid_schema.num_arcs == etalon_grid_schema.num_arcs
        assert grid_schema.shape == etalon_grid_schema.shape  # (num_states, None)
        assert k2.is_rand_equivalent(
            grid_schema, etalon_grid_schema, log_semiring=True, treat_epsilons_specially=False
        ), "Grid schema input labels mismatch"
        assert k2.is_rand_equivalent(
            grid_schema.invert(), etalon_grid_schema.invert(), log_semiring=True, treat_epsilons_specially=False
        ), "Grid schema output labels mismatch"

        # swap aux_labels and unit positions to test unit_positions
        grid_schema.aux_labels, grid_schema.unit_positions = grid_schema.unit_positions, grid_schema.aux_labels
        etalon_grid_schema.aux_labels, etalon_grid_schema.unit_positions = (
            etalon_grid_schema.unit_positions,
            etalon_grid_schema.aux_labels,
        )
        assert k2.is_rand_equivalent(
            grid_schema.invert(), etalon_grid_schema.invert(), log_semiring=True, treat_epsilons_specially=False
        ), "Grid schema unit positions mismatch"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("connect_composed", [True, False])
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_small_compose_transducer(
        self, device, connect_composed, blank_first, rnnt_test_helper, rnn_loss_sample_data
    ):
        if blank_first:
            sample_data = rnn_loss_sample_data.get_sample_small()
        else:
            sample_data = rnn_loss_sample_data.get_sample_small_blank_last()
        graph_rnnt = GraphRnntLoss(
            blank=sample_data.blank_id, connect_composed=connect_composed, use_grid_implementation=False
        )
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_small_grid_transducer(self, device, rnnt_test_helper, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_small()
        graph_rnnt = GraphRnntLoss(blank=0, use_grid_implementation=True)
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_medium_grid_transducer(self, device, rnnt_test_helper, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_medium()
        graph_rnnt = GraphRnntLoss(blank=0, use_grid_implementation=True)
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_medium_random_var_size(self, device, rnnt_test_helper, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_medium_random_var_size(blank_first=True)
        graph_rnnt = GraphRnntLoss(blank=0, use_grid_implementation=True)
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt,
            sample_data.logits.detach(),
            sample_data.targets,
            device,
            input_lengths=sample_data.input_lengths,
            target_lengths=sample_data.target_lengths,
        )
        etalon_rnnt = RNNTLoss_Numpy(blank=0)
        etalon_cost, etalon_grads = rnnt_test_helper.wrap_and_call(
            etalon_rnnt,
            sample_data.logits.detach(),
            sample_data.targets,
            device,
            input_lengths=sample_data.input_lengths,
            target_lengths=sample_data.target_lengths,
        )
        assert np.allclose(graph_cost.sum(), etalon_cost, rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, etalon_grads, atol=1e-4), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_small_random_grid_compose_equivalent(self, device: torch.device, blank_first: bool, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_small_random(blank_first, device=device)
        criterion = GraphRnntLoss(blank=sample_data.blank_id, connect_composed=True, use_grid_implementation=False)
        text_tensor = sample_data.targets[0]
        num_frames = sample_data.logits.shape[1]
        graph_grid = criterion.get_grid(text_tensor, num_frames, sample_data.vocab_size)
        graph_composed = criterion.get_composed_lattice(text_tensor, num_frames, sample_data.vocab_size)
        assert k2.is_rand_equivalent(
            graph_grid, graph_composed, log_semiring=True, treat_epsilons_specially=False
        ), "Grid and composed graphs are not equivalent."
