# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for ``validate_parallelism_compatibility``.

Pure-function tests — no Lightning, no model, no device mesh required.
"""
import warnings

import pytest

from nemo.collections.speechlm2.parts.parallel import validate_parallelism_compatibility


# Combinations that must pass without raising or warning.


def test_bshd_cp1_te_passes():
    validate_parallelism_compatibility(
        packed_sequences=False,
        cp_size=1,
        attn_backend="te",
        nvte_fused_attn=None,
        device_capability=(9, 0),  # H100
    )


def test_bshd_cp1_sdpa_passes():
    validate_parallelism_compatibility(
        packed_sequences=False,
        cp_size=1,
        attn_backend="sdpa",
        nvte_fused_attn=None,
        device_capability=(9, 0),
    )


def test_thd_cp1_te_with_fused_attn_off_passes():
    validate_parallelism_compatibility(
        packed_sequences=True,
        cp_size=1,
        attn_backend="te",
        nvte_fused_attn="0",
        device_capability=(12, 0),  # sm_120 — still fine because env is set
    )


def test_thd_cp2_te_with_fused_attn_off_passes():
    validate_parallelism_compatibility(
        packed_sequences=True,
        cp_size=2,
        attn_backend="te",
        nvte_fused_attn="0",
        device_capability=(12, 0),
    )


# BSHD + CP > 1 — hard error regardless of other knobs.


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_bshd_with_cp_raises(cp_size):
    with pytest.raises(ValueError, match="BSHD .* incompatible with cp_size > 1"):
        validate_parallelism_compatibility(
            packed_sequences=False,
            cp_size=cp_size,
            attn_backend="te",
            nvte_fused_attn="0",
            device_capability=(9, 0),
        )


# THD + non-TE attention — hard error.


@pytest.mark.parametrize("attn_backend", ["sdpa", "flex"])
def test_thd_with_non_te_attn_raises(attn_backend):
    with pytest.raises(ValueError, match=r"THD.*requires.*attn=te"):
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend=attn_backend,
            nvte_fused_attn="0",
            device_capability=(9, 0),
        )


# THD + TE + NVTE_FUSED_ATTN unset — warns on non-sm_120, raises on sm_120.


@pytest.mark.parametrize("nvte_fused_attn", [None, "", "1", "true"])
def test_thd_te_without_fused_attn_off_warns_on_other_archs(nvte_fused_attn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn=nvte_fused_attn,
            device_capability=(9, 0),  # H100
        )
    assert len(caught) == 1
    assert "NVTE_FUSED_ATTN" in str(caught[0].message)


@pytest.mark.parametrize("nvte_fused_attn", [None, "", "1", "true"])
def test_thd_te_without_fused_attn_off_raises_on_sm120(nvte_fused_attn):
    with pytest.raises(ValueError, match="NVTE_FUSED_ATTN"):
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn=nvte_fused_attn,
            device_capability=(12, 0),  # sm_120
        )


def test_thd_te_with_fused_attn_off_does_not_warn_on_sm120():
    """The escape-hatch case: user has the env set, no warning fires."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn="0",
            device_capability=(12, 0),
        )
    assert len(caught) == 0


def test_unknown_device_capability_warns_not_raises():
    """``device_capability=None`` (CPU-only env) treats the THD/TE check as a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn=None,
            device_capability=None,
        )
    assert len(caught) == 1
    assert "NVTE_FUSED_ATTN" in str(caught[0].message)
