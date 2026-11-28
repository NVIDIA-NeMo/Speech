# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


@pytest.mark.parametrize("check_backward", [True, False])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_bshd_with_cp_raises(cp_size, check_backward):
    with pytest.raises(ValueError, match="BSHD .* incompatible with cp_size > 1"):
        validate_parallelism_compatibility(
            packed_sequences=False,
            cp_size=cp_size,
            attn_backend="te",
            nvte_fused_attn="0",
            device_capability=(9, 0),
            check_backward=check_backward,
        )


# THD + non-TE attention — hard error.


@pytest.mark.parametrize("check_backward", [True, False])
@pytest.mark.parametrize("attn_backend", ["sdpa", "flex"])
def test_thd_with_non_te_attn_raises(attn_backend, check_backward):
    with pytest.raises(ValueError, match=r"THD.*requires.*attn=te"):
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend=attn_backend,
            nvte_fused_attn="0",
            device_capability=(9, 0),
            check_backward=check_backward,
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


def test_forward_only_thd_te_without_fused_attn_off_passes_on_sm120():
    """The sm_120 fused-attention issue is backward-only."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=2,
            attn_backend="te",
            nvte_fused_attn=None,
            device_capability=(12, 0),
            check_backward=False,
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


# THD + TE + NVTE_FUSED_ATTN=0 routes causal attention to flash-attn-4, which TE < 2.21 drives with a window that
# admits no keys (zero attention). Raised regardless of ``check_backward`` and device capability.


@pytest.mark.parametrize("check_backward", [True, False])
@pytest.mark.parametrize("device_capability", [(12, 0), (9, 0), None])
@pytest.mark.parametrize("te_version", ["2.15.0+42b84005", "2.20.2"])
def test_thd_te_flash_attn_4_with_old_te_raises(check_backward, device_capability, te_version):
    with pytest.raises(ValueError, match="transformer-engine>=2.21"):
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn="0",
            device_capability=device_capability,
            check_backward=check_backward,
            te_version=te_version,
            flash_attn_4_version="4.0.0b39",
        )


@pytest.mark.parametrize(
    "packed_sequences,nvte_fused_attn,te_version,flash_attn_4_version",
    [
        (True, "0", "2.21.0+7457e47d", "4.0.0b39"),  # fixed TE
        (True, "0", "2.22.0.dev0", "4.0.0b39"),
        (True, "0", "2.15.0", None),  # flash-attn-4 not installed: TE uses another FlashAttention
        (True, "0", None, "4.0.0b39"),  # TE not installed
        (False, "0", "2.15.0", "4.0.0b39"),  # BSHD does not take this path
    ],
)
def test_thd_te_flash_attn_4_version_check_passes(packed_sequences, nvte_fused_attn, te_version, flash_attn_4_version):
    validate_parallelism_compatibility(
        packed_sequences=packed_sequences,
        cp_size=1,
        attn_backend="te",
        nvte_fused_attn=nvte_fused_attn,
        device_capability=(12, 0),
        te_version=te_version,
        flash_attn_4_version=flash_attn_4_version,
    )


def test_thd_te_fused_attn_on_sm120_still_reports_fused_attn_with_old_te():
    """Without NVTE_FUSED_ATTN=0 TE keeps cuDNN fused attention, so only the existing sm_120 check applies."""
    with pytest.raises(ValueError, match="NVTE_FUSED_ATTN"):
        validate_parallelism_compatibility(
            packed_sequences=True,
            cp_size=1,
            attn_backend="te",
            nvte_fused_attn=None,
            device_capability=(12, 0),
            te_version="2.15.0",
            flash_attn_4_version="4.0.0b39",
        )


def test_installed_version_handles_missing_packages():
    from nemo.collections.speechlm2.parts.parallel import installed_version

    assert installed_version("pytest") == pytest.__version__
    assert installed_version("surely-not-an-installed-package-xyz") is None
