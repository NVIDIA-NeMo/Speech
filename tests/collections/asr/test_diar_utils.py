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

import os

import numpy as np
import pytest
import torch

from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.parts.utils.offline_clustering import (
    SpeakerClustering,
    get_scale_interpolated_embs,
    getCosAffinityMatrix,
    getKneighborsConnections,
    split_input_data,
)
from nemo.collections.asr.parts.utils.online_clustering import (
    OnlineSpeakerClustering,
    get_closest_embeddings,
    get_merge_quantity,
    get_minimal_indices,
    merge_vectors,
    run_reducer,
    stitch_cluster_labels,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    check_ranges,
    get_new_cursor_for_update,
    get_online_subsegments_from_buffer,
    get_speech_labels_for_update,
    get_subsegments,
    get_target_sig,
    merge_float_intervals,
    merge_int_intervals,
)

MAX_SEED_COUNT = 2


def check_range_values(target, source):
    bool_list = []
    for tgt, src in zip(target, source):
        for x, y in zip(src, tgt):
            bool_list.append(abs(x - y) < 1e-6)
    return all(bool_list)


def check_labels(target, source):
    bool_list = []
    for x, y in zip(target, source):
        bool_list.append(abs(x - y) < 1e-6)
    return all(bool_list)


def matrix(mat, use_tensor=True, dtype=torch.long):
    if use_tensor:
        mat = torch.Tensor(mat).to(dtype)
    else:
        mat = np.array(mat)
    return mat


def generate_orthogonal_embs(total_spks, perturb_sigma, emb_dim):
    """Generate a set of artificial orthogonal embedding vectors from random numbers
    """
    gaus = torch.randn(emb_dim, emb_dim)
    _svd = torch.linalg.svd(gaus)
    orth = _svd[0] @ _svd[2]
    orth_embs = orth[:total_spks]
    # Assert orthogonality
    assert torch.abs(getCosAffinityMatrix(orth_embs) - torch.diag(torch.ones(total_spks))).sum() < 1e-4
    return orth_embs


def generate_toy_data(
    n_spks=2,
    spk_dur=3,
    emb_dim=192,
    perturb_sigma=0.0,
    ms_window=[1.5, 1.0, 0.5],
    ms_shift=[0.75, 0.5, 0.25],
    torch_seed=0,
):
    torch.manual_seed(torch_seed)
    spk_timestamps = [(spk_dur * k, spk_dur) for k in range(n_spks)]
    emb_list, seg_list = [], []
    multiscale_segment_counts = [0 for _ in range(len(ms_window))]
    ground_truth = []
    random_orthogonal_embs = generate_orthogonal_embs(n_spks, perturb_sigma, emb_dim)
    for scale_idx, (window, shift) in enumerate(zip(ms_window, ms_shift)):
        for spk_idx, (offset, dur) in enumerate(spk_timestamps):
            segments_stt_dur = get_subsegments(offset=offset, window=window, shift=shift, duration=dur)
            segments = [[x[0], x[0] + x[1]] for x in segments_stt_dur]
            emb_cent = random_orthogonal_embs[spk_idx, :]
            emb = emb_cent.tile((len(segments), 1)) + 0.1 * torch.rand(len(segments), emb_dim)
            seg_list.extend(segments)
            emb_list.append(emb)
            multiscale_segment_counts[scale_idx] += emb.shape[0]

            if scale_idx == len(multiscale_segment_counts) - 1:
                ground_truth.extend([spk_idx] * emb.shape[0])

    emb_tensor = torch.concat(emb_list)
    multiscale_segment_counts = torch.tensor(multiscale_segment_counts)
    segm_tensor = torch.tensor(seg_list)
    multiscale_weights = torch.ones(len(ms_window)).unsqueeze(0)
    ground_truth = torch.tensor(ground_truth)
    return emb_tensor, segm_tensor, multiscale_segment_counts, multiscale_weights, spk_timestamps, ground_truth


class TestDiarizationSequneceUtilFunctions:
    """Tests diarization and speaker-task related utils.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("Y", [[3, 3, 3, 4, 4, 5], [100, 100, 100, 104, 104, 1005]])
    @pytest.mark.parametrize("target", [[0, 0, 0, 1, 1, 2]])
    @pytest.mark.parametrize("offset", [1, 10])
    def test_minimal_index_ex2(self, Y, target, offset):
        Y = torch.tensor(Y)
        target = torch.tensor(target)
        min_Y = get_minimal_indices(Y)
        assert check_labels(target, min_Y)
        min_Y = get_minimal_indices(Y + offset)
        assert check_labels(target, min_Y)

    @pytest.mark.parametrize("Y", [[4, 0, 0, 5, 4, 5], [14, 12, 12, 19, 14, 19]])
    @pytest.mark.parametrize("target", [[1, 0, 0, 2, 1, 2]])
    @pytest.mark.parametrize("offset", [1, 10])
    def test_minimal_index_ex2(self, Y, target, offset):
        Y = torch.tensor(Y)
        target = torch.tensor(target)
        min_Y = get_minimal_indices(Y)
        assert check_labels(target, min_Y)
        min_Y = get_minimal_indices(Y + offset)
        assert check_labels(target, min_Y)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 4, 16, 64])
    def test_minimal_index_same(self, N):
        Y = matrix([0] * N + [1] * N + [2] * N)
        min_Y = get_minimal_indices(Y)
        target = matrix([0] * N + [1] * N + [2] * N)
        assert check_labels(target, min_Y)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 4, 16, 64])
    def test_stitch_cluster_labels_label_switch(self, N):
        Y_old = matrix([0] * N)
        Y_new = matrix([0] * N) + 1
        target = matrix([0] * N)
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 4, 16, 64])
    def test_stitch_cluster_labels_label_many_to_one(self, N):
        Y_old = matrix(np.arange(N).tolist())
        Y_new = matrix([0] * N)
        target = matrix([0] * N)
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 4, 16, 64])
    def test_stitch_cluster_labels_label_one_to_many(self, N):
        Y_old = matrix(np.arange(N).tolist())
        Y_new = matrix([k for k in range(N)])
        target = matrix([k for k in range(N)])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 4, 16, 64])
    def test_stitch_cluster_labels_one_label_replaced(self, N):
        Y_old = matrix([0] * N + [1] * N + [2] * N)
        Y_new = matrix([1] * N + [2] * N + [3] * N)
        target = matrix([0] * N + [1] * N + [2] * N)
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 4, 16, 64])
    def test_stitch_cluster_labels_confusion_error(self, N):
        Y_old = matrix([0] * N + [1] * (N - 1) + [2] * (N + 1))
        Y_new = matrix([1] * N + [2] * N + [3] * N)
        target = matrix([0] * N + [1] * N + [2] * N)
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 256])
    def test_stitch_cluster_labels_speaker_more_speakers(self, N):
        Y_old = matrix([0] * N + [1] * (N - 1) + [2] * (N + 1) + [0, 0, 0])
        Y_new = matrix([1] * N + [0] * N + [2] * N + [4, 5, 6])
        target = matrix([0] * N + [1] * N + [2] * N + [3, 4, 5])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 256])
    def test_stitch_cluster_labels_speaker_longer_sequence(self, N):
        Y_old = matrix([0] * N + [1] * N + [2] * N + [0, 0, 0] * N)
        Y_new = matrix([1] * N + [2] * N + [0] * N + [1, 2, 3, 1, 2, 3] * N)
        target = matrix([0] * N + [1] * N + [2] * N + [0, 1, 3, 0, 1, 3] * N)
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [2, 3, 4, 5])
    @pytest.mark.parametrize("merge_quantity", [2, 3])
    def test_embedding_merger(self, n_spks, merge_quantity):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(n_spks, spk_dur=5, perturb_sigma=10)
        em_s, ts_s = split_input_data(em, ts, mc)
        target_speaker_index = 0
        pre_clus_labels = gt
        ndx = torch.where(pre_clus_labels == target_speaker_index)[0]
        pre_embs = em_s[-1]
        affinity_mat = getCosAffinityMatrix(pre_embs)
        cmat = affinity_mat[:, ndx][ndx, :]
        # Check the dimension of the selected affinity values
        assert cmat.shape[0] == cmat.shape[1] == torch.sum(pre_clus_labels == target_speaker_index).item()
        index_2d, rest_inds = get_closest_embeddings(cmat, merge_quantity)
        # Check the most closest affinity value
        assert torch.max(cmat.sum(0)) == cmat.sum(0)[index_2d[0]]
        spk_cluster_labels, emb_ndx = pre_clus_labels[ndx], pre_embs[ndx]
        merged_embs, merged_clus_labels = merge_vectors(index_2d, emb_ndx, spk_cluster_labels)
        # Check the number of merged embeddings and labels
        assert (torch.sum(gt == target_speaker_index).item() - merge_quantity) == merged_clus_labels.shape[0]

    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1, 8])
    @pytest.mark.parametrize("spk_dur", [0.2, 0.25, 0.5, 1, 10])
    def test_cosine_affinity_calculation(self, n_spks, spk_dur):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(n_spks=n_spks, spk_dur=spk_dur)
        em_s, ts_s = split_input_data(em, ts, mc)
        affinity_mat = getCosAffinityMatrix(em_s[-1])
        # affinity_mat should not contain any nan element
        assert torch.any(torch.isnan(affinity_mat)) == False

    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1, 8])
    @pytest.mark.parametrize("spk_dur", [0.2, 0.25, 0.5, 1, 10])
    def test_cosine_affinity_calculation_scale_interpol(self, n_spks, spk_dur):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(n_spks=n_spks, spk_dur=spk_dur)
        em_s, ts_s = split_input_data(em, ts, mc)
        embs, _ = get_scale_interpolated_embs(mw, em_s, ts_s)
        affinity_mat = getCosAffinityMatrix(embs)
        # affinity_mat should not contain any nan element
        assert torch.any(torch.isnan(affinity_mat)) == False

    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [4, 5, 6])
    @pytest.mark.parametrize("target_speaker_index", [0, 1, 2])
    @pytest.mark.parametrize("merge_quantity", [2, 3])
    def test_embedding_reducer(self, n_spks, target_speaker_index, merge_quantity):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(n_spks=n_spks, spk_dur=10)
        em_s, ts_s = split_input_data(em, ts, mc)
        merged_embs, merged_clus_labels, _ = run_reducer(
            pre_embs=em_s[-1], target_spk_idx=target_speaker_index, merge_quantity=merge_quantity, pre_clus_labels=gt,
        )
        assert (torch.sum(gt == target_speaker_index).item() - merge_quantity) == merged_clus_labels.shape[0]

    @pytest.mark.unit
    @pytest.mark.parametrize("ntbr", [3])
    @pytest.mark.parametrize("pcl", [torch.tensor([0] * 70 + [1] * 32)])
    @pytest.mark.parametrize("mspb", [25])
    def test_merge_scheduler_2clus(self, ntbr, pcl, mspb):
        class_target_vol = get_merge_quantity(num_to_be_removed=ntbr, pre_clus_labels=pcl, min_count_per_cluster=mspb,)
        assert all(class_target_vol == torch.tensor([3, 0]))

    @pytest.mark.unit
    @pytest.mark.parametrize("ntbr", [3])
    @pytest.mark.parametrize("pcl", [torch.tensor([0] * 80 + [1] * 35 + [2] * 32)])
    @pytest.mark.parametrize("mspb", [0, 25])
    def test_merge_scheduler_3clus(self, ntbr, pcl, mspb):
        class_target_vol = get_merge_quantity(num_to_be_removed=ntbr, pre_clus_labels=pcl, min_count_per_cluster=mspb,)
        assert all(class_target_vol == torch.tensor([3, 0, 0]))

    @pytest.mark.unit
    @pytest.mark.parametrize("ntbr", [132 - 45])
    @pytest.mark.parametrize("pcl", [torch.tensor([2] * 70 + [0] * 32 + [1] * 27 + [3] * 3)])
    @pytest.mark.parametrize("mspb", [3, 10])
    def test_merge_scheduler_4clus_shuff(self, ntbr, pcl, mspb):
        class_target_vol = get_merge_quantity(num_to_be_removed=ntbr, pre_clus_labels=pcl, min_count_per_cluster=mspb,)
        assert all(class_target_vol == torch.tensor([18, 13, 56, 0]))

    @pytest.mark.unit
    @pytest.mark.parametrize("ntbr", [3])
    @pytest.mark.parametrize("pcl", [torch.tensor([0] * 5 + [1] * 4 + [2] * 3)])
    @pytest.mark.parametrize("mspb", [0, 2])
    def test_merge_scheduler_3clus(self, ntbr, pcl, mspb):
        class_target_vol = get_merge_quantity(num_to_be_removed=ntbr, pre_clus_labels=pcl, min_count_per_cluster=mspb,)
        assert all(class_target_vol == torch.tensor([2, 1, 0]))

    @pytest.mark.unit
    @pytest.mark.parametrize("ntbr", [2])
    @pytest.mark.parametrize("pcl", [torch.tensor([0] * 7 + [1] * 5 + [2] * 3 + [3] * 5)])
    @pytest.mark.parametrize("mspb", [2])
    def test_merge_scheduler_3clus_repeat(self, ntbr, pcl, mspb):
        class_target_vol = get_merge_quantity(num_to_be_removed=ntbr, pre_clus_labels=pcl, min_count_per_cluster=mspb,)
        assert all(class_target_vol == torch.tensor([2, 0, 0, 0]))


class TestDiarizationSegmentationUtils:
    """
    Test segmentation util functions
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "intervals",
        [
            [[1, 4], [2, 6], [8, 10], [15, 18]],
            [[1, 5], [2, 6], [8, 10], [15, 18]],
            [[2, 6], [1, 3], [15, 18], [8, 10]],
            [[8, 10], [15, 18], [2, 6], [1, 3]],
            [[8, 10], [15, 18], [2, 6], [1, 3], [3, 5]],
            [[8, 10], [8, 8], [15, 18], [2, 6], [1, 6], [2, 4]],
        ],
    )
    @pytest.mark.parametrize("target", [[[1, 6], [8, 10], [15, 18]]])
    def test_merge_int_intervals_ex1(self, intervals, target):
        merged = merge_int_intervals(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "intervals",
        [
            [[6, 8], [0, 9], [2, 4], [4, 7]],
            [[0, 9], [6, 8], [4, 7], [2, 4]],
            [[0, 4], [0, 0], [4, 9], [2, 4]],
            [[6, 8], [2, 8], [0, 3], [3, 4], [4, 5], [5, 9]],
        ],
    )
    @pytest.mark.parametrize("target", [[[0, 9]]])
    def test_merge_int_intervals_ex2(self, intervals, target):
        merged = merge_int_intervals(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    @pytest.mark.parametrize("intervals", [[[0, 1], [1, 9]], [[0, 0], [0, 9]], [[0, 9], [0, 9]]])
    @pytest.mark.parametrize("target", [[[0, 9]]])
    def test_merge_int_intervals_edge_test(self, intervals, target):
        merged = merge_int_intervals(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_merge_float_intervals_edge_margin_test(self):
        intervals = [[0.0, 1.0], [1.0, 2.0]]

        target_0 = [[0.0, 2.0]]
        merged_0 = merge_float_intervals(intervals, margin=0)
        assert check_range_values(target_0, merged_0)

        target_1 = [[0.0, 1.0], [1.0, 2.0]]
        merged_1 = merge_float_intervals(intervals, margin=1)
        assert check_range_values(target_1, merged_1)

        target_2 = [[0.0, 1.0], [1.0, 2.0]]
        merged_2 = merge_float_intervals(intervals, margin=2)
        assert check_range_values(target_2, merged_2)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "intervals",
        [
            [[0.25, 1.7], [1.5, 3.0], [2.8, 5.0], [5.5, 10.0]],
            [[0.25, 5.0], [5.5, 10.0], [1.5, 3.5]],
            [[5.5, 8.05], [8.0, 10.0], [0.25, 5.0]],
            [[0.25, 3.0], [1.5, 3.0], [5.5, 10.0], [2.8, 5.0]],
            [[0.25, 1.7], [1.5, 3.0], [2.8, 5.0], [5.5, 10.0]],
        ],
    )
    @pytest.mark.parametrize("target", [[[0.25, 5.0], [5.5, 10.0]]])
    def test_merge_float_overlaps(self, intervals, target):
        merged = merge_float_intervals(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_get_speech_labels_for_update(self):
        frame_start = 3.0
        buffer_end = 6.0
        cumulative_speech_labels = torch.tensor([[0.0000, 3.7600]])
        vad_timestamps = torch.tensor([[0.9600, 4.8400]])
        cursor_for_old_segments = 1.0
        speech_labels_for_update, cumulative_speech_labels = get_speech_labels_for_update(
            frame_start, buffer_end, cumulative_speech_labels, vad_timestamps, cursor_for_old_segments,
        )
        assert (speech_labels_for_update - torch.tensor([[1.0000, 3.7600]])).sum() < 1e-8
        assert (cumulative_speech_labels - torch.tensor([[0.9600, 4.8400]])).sum() < 1e-8

        # Check if the ranges are containing faulty values
        assert check_ranges(speech_labels_for_update)
        assert check_ranges(cumulative_speech_labels)

    @pytest.mark.unit
    def test_get_online_subsegments_from_buffer(self):
        torch.manual_seed(0)
        sample_rate = 16000
        speech_labels_for_update = torch.Tensor([[0.0000, 3.7600]])
        audio_buffer = torch.randn(5 * sample_rate)
        segment_indexes = []
        window = 2.0
        shift = 1.0
        slice_length = int(window * sample_rate)
        range_target = [[0.0, 2.0], [1.0, 3.0], [2.0, 3.76]]
        sigs_list, sig_rangel_list, sig_indexes = get_online_subsegments_from_buffer(
            buffer_start=0.0,
            buffer_end=5.0,
            sample_rate=sample_rate,
            speech_labels_for_update=speech_labels_for_update,
            audio_buffer=audio_buffer,
            segment_indexes=segment_indexes,
            window=window,
            shift=shift,
        )
        assert check_range_values(target=range_target, source=sig_rangel_list)
        for k, rg in enumerate(sig_rangel_list):
            signal = get_target_sig(audio_buffer, rg[0], rg[1], slice_length, sample_rate)
            if len(signal) < int(window * sample_rate):
                signal = repeat_signal(signal, len(signal), slice_length)
            assert len(signal) == int(slice_length), "Length mismatch"
            assert (np.abs(signal - sigs_list[k])).sum() < 1e-8, "Audio stream mismatch"
        assert (torch.tensor(sig_indexes) - torch.arange(len(range_target))).sum() < 1e-8, "Segment index mismatch"

    @pytest.mark.unit
    @pytest.mark.parametrize("frame_start", [3.0])
    @pytest.mark.parametrize("segment_range_ts", [[[0.0, 2.0]]])
    @pytest.mark.parametrize("gt_cursor_for_old_segments", [3.0])
    @pytest.mark.parametrize("gt_cursor_index", [1])
    def test_get_new_cursor_for_update_mulsegs_ex1(
        self, frame_start, segment_range_ts, gt_cursor_for_old_segments, gt_cursor_index
    ):
        cursor_for_old_segments, cursor_index = get_new_cursor_for_update(frame_start, segment_range_ts)
        assert cursor_for_old_segments == gt_cursor_for_old_segments
        assert cursor_index == gt_cursor_index


class TestClusteringUtilFunctions:
    @pytest.mark.parametrize("p_value", [1, 5, 9])
    @pytest.mark.parametrize("N", [9, 20])
    @pytest.mark.parametrize("mask_method", ['binary', 'sigmoid', 'drop'])
    def test_get_k_neighbors_connections(self, p_value: int, N: int, mask_method: str, seed=0):
        torch.manual_seed(seed)
        random_mat = torch.rand(N, N)
        affinity_mat = 0.5 * (random_mat + random_mat.T)
        affinity_mat = affinity_mat / torch.max(affinity_mat)
        binarized_affinity_mat = getKneighborsConnections(affinity_mat, p_value, mask_method)
        if mask_method == 'binary':
            assert all(binarized_affinity_mat.sum(dim=0) == float(p_value))
        elif mask_method == 'sigmoid':
            assert all(binarized_affinity_mat.sum(dim=0) <= float(p_value))
        elif mask_method == 'drop':
            assert all(binarized_affinity_mat.sum(dim=0) <= float(p_value))


class TestSpeakerClustering:
    """
    Test speaker clustering module
    """

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1, 2])
    def test_clus_script_export(self, n_spks, total_dur_sec=30):
        exported_filename = 'speaker_clustering_script.pt'
        speaker_clustering_python = SpeakerClustering(maj_vote_spk_count=False, cuda=True)
        speaker_clustering_scripted_source = torch.jit.script(speaker_clustering_python)
        torch.jit.save(speaker_clustering_scripted_source, exported_filename)
        speaker_clustering_scripted = torch.jit.load(exported_filename)
        assert os.path.exists(exported_filename)
        os.remove(exported_filename)
        assert not os.path.exists(exported_filename)

        each_spk_dur = float(total_dur_sec / n_spks)
        em, ts, mc, mw, _, _ = generate_toy_data(n_spks=n_spks, spk_dur=each_spk_dur)
        num_speakers = -1
        max_num_speakers = 8
        enhanced_count_thres = 80
        sparse_search_volume = 10
        max_rp_threshold = 0.15
        fixed_thres = -1.0

        # Function call for NeMo python pipeline (unexported) in python
        Y_py = speaker_clustering_python.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=num_speakers,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            max_rp_threshold=max_rp_threshold,
            fixed_thres=fixed_thres,
        )

        # Function call for exported module but in python
        Y_tjs = speaker_clustering_scripted.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=num_speakers,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            max_rp_threshold=max_rp_threshold,
            fixed_thres=fixed_thres,
        )

        clustering_param_dict = {
            'embeddings': em,
            'timestamps': ts,
            'multiscale_segment_counts': mc,
            'multiscale_weights': mw,
            'oracle_num_speakers': torch.Tensor([num_speakers]),
            'max_num_speakers': torch.Tensor([max_num_speakers]),
            'enhanced_count_thres': torch.Tensor([enhanced_count_thres]),
            'sparse_search_volume': torch.Tensor([sparse_search_volume]),
            'max_rp_threshold': torch.tensor([max_rp_threshold]),
            'fixed_thres': torch.tensor([fixed_thres]),
        }

        # Function call for an exported module in Triton server environment
        Y_prd = speaker_clustering_scripted.forward(clustering_param_dict)

        # All three types of function call should generate exactly the same output.
        assert len(set(Y_tjs.tolist())) == len(set(Y_py.tolist())) == len(set(Y_prd.tolist())) == n_spks
        assert (
            all(Y_tjs == Y_py) == all(Y_py == Y_prd) == True
        ), f"Script module and python module are showing different clustering results"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1, 2])
    @pytest.mark.parametrize("spk_dur", [20])
    @pytest.mark.parametrize("SSV", [10])
    @pytest.mark.parametrize("perturb_sigma", [0.1])
    def test_offline_speaker_clustering_cpu(self, n_spks, spk_dur, SSV, perturb_sigma):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(n_spks=n_spks, spk_dur=spk_dur, perturb_sigma=perturb_sigma)
        offline_speaker_clustering = SpeakerClustering(maj_vote_spk_count=False, cuda=False)
        assert isinstance(offline_speaker_clustering, SpeakerClustering)

        Y_out = offline_speaker_clustering.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=-1,
            max_num_speakers=8,
            enhanced_count_thres=40,
            sparse_search_volume=SSV,
            max_rp_threshold=0.15,
            fixed_thres=-1.0,
        )
        permuted_Y = stitch_cluster_labels(Y_old=gt, Y_new=Y_out)

        # mc[-1] is the number of base scale segments
        assert len(set(permuted_Y.tolist())) == n_spks
        assert Y_out.shape[0] == mc[-1]
        assert all(permuted_Y == gt)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("total_sec", [30])
    @pytest.mark.parametrize("SSV", [10])
    @pytest.mark.parametrize("perturb_sigma", [0.1])
    @pytest.mark.parametrize("seed", np.arange(MAX_SEED_COUNT).tolist())
    def test_offline_speaker_clustering_gpu(self, n_spks, total_sec, SSV, perturb_sigma, seed):
        spk_dur = total_sec / n_spks
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(
            n_spks=n_spks, spk_dur=spk_dur, perturb_sigma=perturb_sigma, torch_seed=seed
        )
        offline_speaker_clustering = SpeakerClustering(maj_vote_spk_count=False, cuda=True)
        assert isinstance(offline_speaker_clustering, SpeakerClustering)

        Y_out = offline_speaker_clustering.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=-1,
            max_num_speakers=8,
            enhanced_count_thres=40,
            sparse_search_volume=SSV,
            max_rp_threshold=0.15,
            fixed_thres=-1.0,
        )
        permuted_Y = stitch_cluster_labels(Y_old=gt, Y_new=Y_out)

        # mc[-1] is the number of base scale segments
        assert len(set(permuted_Y.tolist())) == n_spks
        assert Y_out.shape[0] == mc[-1]
        assert all(permuted_Y == gt)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1])
    @pytest.mark.parametrize("spk_dur", [0.25, 0.5, 0.75, 1, 1.5, 2])
    @pytest.mark.parametrize("SSV", [5])
    @pytest.mark.parametrize("enhanced_count_thres", [40])
    @pytest.mark.parametrize("min_samples_for_nmesc", [6])
    @pytest.mark.parametrize("seed", np.arange(MAX_SEED_COUNT).tolist())
    def test_offline_speaker_clustering_very_short_cpu(
        self, n_spks, spk_dur, SSV, enhanced_count_thres, min_samples_for_nmesc, seed,
    ):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(
            n_spks=n_spks, spk_dur=spk_dur, perturb_sigma=0.1, torch_seed=seed
        )
        offline_speaker_clustering = SpeakerClustering(maj_vote_spk_count=False, min_samples_for_nmesc=0, cuda=False)
        assert isinstance(offline_speaker_clustering, SpeakerClustering)
        Y_out = offline_speaker_clustering.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=-1,
            max_num_speakers=8,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=SSV,
            max_rp_threshold=0.15,
            fixed_thres=-1.0,
        )
        permuted_Y = stitch_cluster_labels(Y_old=gt, Y_new=Y_out)

        # mc[-1] is the number of base scale segments
        assert len(set(permuted_Y.tolist())) == n_spks
        assert Y_out.shape[0] == mc[-1]
        assert all(permuted_Y == gt)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1])
    @pytest.mark.parametrize("spk_dur", [0.25, 0.5, 0.75, 1, 2, 4])
    @pytest.mark.parametrize("SSV", [5])
    @pytest.mark.parametrize("enhanced_count_thres", [40])
    @pytest.mark.parametrize("min_samples_for_nmesc", [6])
    @pytest.mark.parametrize("seed", np.arange(MAX_SEED_COUNT).tolist())
    def test_offline_speaker_clustering_very_short_gpu(
        self, n_spks, spk_dur, SSV, enhanced_count_thres, min_samples_for_nmesc, seed
    ):
        em, ts, mc, mw, spk_ts, gt = generate_toy_data(
            n_spks=n_spks, spk_dur=spk_dur, perturb_sigma=0.1, torch_seed=seed
        )
        offline_speaker_clustering = SpeakerClustering(maj_vote_spk_count=False, min_samples_for_nmesc=0, cuda=True)
        assert isinstance(offline_speaker_clustering, SpeakerClustering)
        Y_out = offline_speaker_clustering.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=-1,
            max_num_speakers=8,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=SSV,
            max_rp_threshold=0.15,
            fixed_thres=-1.0,
        )
        permuted_Y = stitch_cluster_labels(Y_old=gt, Y_new=Y_out)

        # mc[-1] is the number of base scale segments
        assert Y_out.shape[0] == mc[-1]
        assert all(permuted_Y == gt)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [1, 2, 3, 4])
    @pytest.mark.parametrize("total_sec", [30])
    @pytest.mark.parametrize("buffer_size", [30])
    @pytest.mark.parametrize("sigma", [0.1])
    @pytest.mark.parametrize("seed", np.arange(MAX_SEED_COUNT).tolist())
    def test_online_speaker_clustering(self, n_spks, total_sec, buffer_size, sigma, seed):
        step_per_frame = 2
        spk_dur = total_sec / n_spks
        em, ts, mc, _, _, gt = generate_toy_data(n_spks, spk_dur=spk_dur, perturb_sigma=sigma, torch_seed=seed)
        em_s, ts_s = split_input_data(em, ts, mc)

        emb_gen = em_s[-1]
        segment_indexes = ts_s[-1]
        if torch.cuda.is_available():
            emb_gen, segment_indexes = emb_gen.to("cuda"), segment_indexes.to("cuda")
            cuda = True
        else:
            cuda = False

        history_buffer_size = buffer_size
        current_buffer_size = buffer_size

        online_clus = OnlineSpeakerClustering(
            max_num_speakers=8,
            max_rp_threshold=0.15,
            sparse_search_volume=30,
            history_buffer_size=history_buffer_size,
            current_buffer_size=current_buffer_size,
        )

        n_frames = int(emb_gen.shape[0] / step_per_frame)
        evaluation_list = []

        # Simulate online speaker clustering
        for frame_index in range(n_frames):
            curr_emb = emb_gen[0 : (frame_index + 1) * step_per_frame]
            base_segment_indexes = np.arange(curr_emb.shape[0])

            # Save history embeddings
            concat_emb, add_new = online_clus.get_reduced_mat(
                emb_in=curr_emb, base_segment_indexes=base_segment_indexes
            )

            # Check history_buffer_size and history labels
            assert (
                online_clus.history_embedding_buffer_emb.shape[0] <= history_buffer_size
            ), "History buffer size error"
            assert (
                online_clus.history_embedding_buffer_emb.shape[0]
                == online_clus.history_embedding_buffer_label.shape[0]
            )

            # Call clustering function
            Y_concat = online_clus.forward_infer(emb=concat_emb, frame_index=frame_index, cuda=cuda)

            # Resolve permutations
            merged_clus_labels = online_clus.match_labels(Y_concat, add_new=add_new)
            assert len(merged_clus_labels) == (frame_index + 1) * step_per_frame
            print(merged_clus_labels)
            # Resolve permutation issue by using stitch_cluster_labels function
            merged_clus_labels = stitch_cluster_labels(Y_old=gt[: len(merged_clus_labels)], Y_new=merged_clus_labels)
            evaluation_list.extend(list(merged_clus_labels == gt[: len(merged_clus_labels)]))

        assert online_clus.is_online
        assert add_new
        cumul_label_acc = sum(evaluation_list) / len(evaluation_list)
        assert cumul_label_acc > 0.9

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("n_spks", [4])
    @pytest.mark.parametrize("total_sec", [30])
    @pytest.mark.parametrize("buffer_size", [30])
    @pytest.mark.parametrize("sigma", [0.1])
    @pytest.mark.parametrize("seed", [0])
    def test_online_speaker_clustering_cpu(self, n_spks, total_sec, buffer_size, sigma, seed):
        self.test_online_speaker_clustering(n_spks, total_sec, buffer_size, sigma, seed)
