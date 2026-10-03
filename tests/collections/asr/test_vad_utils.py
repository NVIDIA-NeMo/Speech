"""Unit tests for VAD thresholding, segmentation, and post-processing logic.

Tests cover binarization, filtering, segment merging, gap detection,
short segment removal, and the full generate_vad_segment_table_per_tensor pipeline
from nemo.collections.asr.parts.utils.vad_utils.
"""

import importlib.util
import sys
import types

import pytest
import torch

sys.modules["nv_one_logger"] = types.ModuleType("nv_one_logger")
sys.modules["nv_one_logger.api"] = types.ModuleType("nv_one_logger.api")
sys.modules["nv_one_logger.api.config"] = types.ModuleType("nv_one_logger.api.config")
_m = types.ModuleType("nemo.collections.asr.models")
_m.EncDecClassificationModel = type("E", (), {})
_m.EncDecFrameClassificationModel = type("F", (), {})
sys.modules["nemo.collections.asr.models"] = _m
_ms = types.ModuleType("nemo.collections.common.parts.preprocessing.manifest")
_ms.get_full_path = lambda *a, **k: None
sys.modules["nemo.collections.common.parts.preprocessing.manifest"] = _ms
_l = types.ModuleType("nemo.utils.logging")
for _f in ("info", "debug", "warning", "error"):
    setattr(_l, _f, lambda *a, **k: None)
sys.modules["nemo.utils"] = types.ModuleType("nemo.utils")
sys.modules["nemo.utils"].logging = _l

_spec = importlib.util.spec_from_file_location(
    "vad_utils", "/home/shivanshsingh/Desktop/NEMO/NeMo/nemo/collections/asr/parts/utils/vad_utils.py"
)
_v = importlib.util.module_from_spec(_spec)
sys.modules["vad_utils"] = _v
_spec.loader.exec_module(_v)

binarization = _v.binarization
cal_vad_onset_offset = _v.cal_vad_onset_offset
filter_short_segments = _v.filter_short_segments
filtering = _v.filtering
generate_vad_segment_table_per_tensor = _v.generate_vad_segment_table_per_tensor
get_gap_segments = _v.get_gap_segments
merge_overlap_segment = _v.merge_overlap_segment
percentile = _v.percentile
remove_segments = _v.remove_segments
PostProcessingParams = _v.PostProcessingParams


class TestBinarization:
    """Tests for binarization: frame-level scores to speech segments."""

    def test_symmetric_threshold(self):
        torch.manual_seed(42)
        seq = torch.tensor([0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.0, 0.6, 0.8, 0.0])
        result = binarization(seq, {"onset": 0.5, "offset": 0.5, "frame_length_in_sec": 0.01})
        assert result.shape[0] == 2, f"Expected 2 segments, got {result.shape[0]}"
        assert result[0, 0] == pytest.approx(0.02)
        assert result[0, 1] == pytest.approx(0.05)
        assert result[1, 0] == pytest.approx(0.07)
        assert result[1, 1] == pytest.approx(0.09)

    def test_hysteresis_onset_gt_offset(self):
        seq = torch.tensor([0.0, 0.6, 0.4, 0.4, 0.6, 0.0])
        result = binarization(seq, {"onset": 0.5, "offset": 0.3, "frame_length_in_sec": 0.01})
        assert result.shape[0] >= 1, "Expected at least 1 segment with hysteresis"

    def test_empty_sequence(self):
        result = binarization(torch.tensor([]), {"onset": 0.5, "offset": 0.5, "frame_length_in_sec": 0.01})
        assert result.shape == torch.Size([0]), "Empty input should return empty segments"

    def test_all_zeros(self):
        result = binarization(torch.zeros(10), {"onset": 0.5, "offset": 0.5, "frame_length_in_sec": 0.01})
        assert result.shape == torch.Size([0]), "All zeros should yield no speech segments"

    def test_all_ones(self):
        result = binarization(torch.ones(10), {"onset": 0.5, "offset": 0.5, "frame_length_in_sec": 0.01})
        assert result.shape[0] == 1, "All ones should yield exactly 1 segment"
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.09)

    def test_single_frame(self):
        result = binarization(torch.tensor([0.9]), {"onset": 0.5, "offset": 0.5, "frame_length_in_sec": 0.01})
        assert result.shape[0] == 1, "Single high-value frame should produce 1 segment"


class TestBinarizationParametrized:
    """Parametrized threshold sweep: higher onset yields fewer or equal segments."""

    @pytest.mark.parametrize("onset", [0.3, 0.5, 0.7, 0.9])
    def test_higher_onset_fewer_segments(self, onset):
        torch.manual_seed(42)
        seq = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7, 0.9, 0.1, 0.5])
        result = binarization(seq, {"onset": onset, "offset": onset, "frame_length_in_sec": 0.01})
        counts = {0.3: 2, 0.5: 2, 0.7: 2, 0.9: 0}
        assert result.shape[0] == counts[onset], f"Expected {counts[onset]} segments for onset={onset}"


class TestFiltering:
    """Tests for filtering post-processing: short segment removal and gap merging."""

    def test_filter_short_speech(self):
        segs = torch.tensor([[0.0, 0.02], [0.10, 0.20]])
        result = filtering(segs, {"min_duration_on": 0.05, "min_duration_off": 0.0, "filter_speech_first": 1.0})
        assert result.shape[0] == 1, "Short segment should be filtered out"
        assert result[0, 0] == pytest.approx(0.10)

    def test_no_filtering_needed(self):
        segs = torch.tensor([[0.0, 0.10], [0.20, 0.35]])
        result = filtering(segs, {"min_duration_on": 0.05, "min_duration_off": 0.0, "filter_speech_first": 1.0})
        assert result.shape[0] == 2, "No segments should be filtered"

    def test_empty_input(self):
        result = filtering(
            torch.empty(0), {"min_duration_on": 0.05, "min_duration_off": 0.0, "filter_speech_first": 1.0}
        )
        assert result.shape == torch.Size([0])


class TestMergeOverlapSegment:
    """Tests for merging overlapping speech segments."""

    def test_merge_overlapping(self):
        result = merge_overlap_segment(torch.tensor([[0.0, 1.5], [1.0, 3.5]]))
        assert result.shape[0] == 1
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(3.5)

    def test_no_overlap(self):
        result = merge_overlap_segment(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
        assert result.shape[0] == 2

    def test_empty_input(self):
        result = merge_overlap_segment(torch.empty(0))
        assert result.shape == torch.Size([0])


class TestSegmentHelpers:
    """Tests for filter_short_segments, get_gap_segments, and remove_segments."""

    def test_filter_short_segments(self):
        segs = torch.tensor([[0.0, 1.5], [1.0, 3.5], [4.0, 7.0]])
        assert filter_short_segments(segs, 2.0).shape[0] == 2

    def test_get_gap_segments(self):
        segs = torch.tensor([[0.0, 1.0], [2.0, 3.0], [5.0, 6.0]])
        gaps = get_gap_segments(segs)
        assert gaps.shape[0] == 2
        assert gaps[0, 0] == pytest.approx(1.0) and gaps[0, 1] == pytest.approx(2.0)

    def test_remove_segments(self):
        orig = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        assert remove_segments(orig, torch.tensor([[2.0, 3.0]])).shape[0] == 2


class TestCalVadOnsetOffset:
    """Tests for threshold scale conversion."""

    def test_absolute_scale(self):
        onset, offset = cal_vad_onset_offset("absolute", 0.5, 0.5)
        assert onset == pytest.approx(0.5) and offset == pytest.approx(0.5)

    def test_relative_scale(self):
        onset, offset = cal_vad_onset_offset("relative", 0.5, 0.5, torch.tensor([0.2, 0.4, 0.6, 0.8]))
        assert onset == pytest.approx(0.5) and offset == pytest.approx(0.5)


class TestPercentile:
    """Tests for the percentile utility."""

    def test_percentile_basic(self):
        assert percentile(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), 50) == 3.0

    def test_percentile_extremes(self):
        assert percentile(torch.tensor([10.0, 20.0, 30.0]), 100) == 30.0


class TestGenerateVadSegmentTablePerTensor:
    """Integration tests for the full VAD segment table pipeline."""

    def test_full_pipeline(self):
        torch.manual_seed(42)
        seq = torch.tensor([0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.8, 0.8, 0.0])
        pa = {
            "onset": 0.5,
            "offset": 0.5,
            "frame_length_in_sec": 0.01,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
        result = generate_vad_segment_table_per_tensor(seq, pa)
        assert result.shape[0] == 2
        assert result[0, 0] < result[0, 1] and result[1, 0] < result[1, 1], "Start must be less than end"

    def test_determinism(self):
        torch.manual_seed(42)
        seq = torch.tensor([0.0, 0.7, 0.7, 0.0, 0.8, 0.8, 0.0])
        pa = {
            "onset": 0.5,
            "offset": 0.5,
            "frame_length_in_sec": 0.01,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
        r1 = generate_vad_segment_table_per_tensor(seq, pa)
        r2 = generate_vad_segment_table_per_tensor(seq, pa)
        assert torch.equal(r1, r2), "Results must be deterministic"

    def test_output_no_overlap(self):
        torch.manual_seed(42)
        seq = torch.tensor([0.0, 0.9, 0.9, 0.0, 0.0, 0.8, 0.8, 0.0])
        pa = {
            "onset": 0.5,
            "offset": 0.5,
            "frame_length_in_sec": 0.01,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
        result = generate_vad_segment_table_per_tensor(seq, pa)
        for i in range(result.shape[0] - 1):
            assert result[i, 1] <= result[i + 1, 0], "Segments must not overlap"
