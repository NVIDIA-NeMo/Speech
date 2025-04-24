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

import os
from unittest import mock

import pytest
import torch

from nemo.utils.get_rank import get_last_rank, get_rank, is_global_rank_zero


class TestIsGlobalRankZero:
    """Test the is_global_rank_zero function with various environment variable settings."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Clear all relevant environment variables before each test."""
        for var in ["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "NODE_RANK", "GROUP_RANK", "LOCAL_RANK"]:
            if var in os.environ:
                del os.environ[var]

    def test_default_behavior(self):
        """Test the default behavior when no environment variables are set."""
        assert is_global_rank_zero() is True

    def test_with_pytorch_rank_0(self):
        """Test when RANK=0 (pytorch environment)."""
        os.environ["RANK"] = "0"
        assert is_global_rank_zero() is True

    def test_with_pytorch_rank_nonzero(self):
        """Test when RANK is not 0 (pytorch environment)."""
        os.environ["RANK"] = "1"
        assert is_global_rank_zero() is False

    def test_with_slurm_rank_0(self):
        """Test when SLURM_PROCID=0 (SLURM environment)."""
        os.environ["SLURM_PROCID"] = "0"
        assert is_global_rank_zero() is True

    def test_with_slurm_rank_nonzero(self):
        """Test when SLURM_PROCID is not 0 (SLURM environment)."""
        os.environ["SLURM_PROCID"] = "1"
        assert is_global_rank_zero() is False

    def test_with_mpi_rank_0(self):
        """Test when OMPI_COMM_WORLD_RANK=0 (MPI environment)."""
        os.environ["OMPI_COMM_WORLD_RANK"] = "0"
        assert is_global_rank_zero() is True

    def test_with_mpi_rank_nonzero(self):
        """Test when OMPI_COMM_WORLD_RANK is not 0 (MPI environment)."""
        os.environ["OMPI_COMM_WORLD_RANK"] = "1"
        assert is_global_rank_zero() is False

    def test_with_node_rank_0_local_rank_0(self):
        """Test when NODE_RANK=0 and LOCAL_RANK=0."""
        os.environ["NODE_RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        assert is_global_rank_zero() is True

    def test_with_node_rank_0_local_rank_nonzero(self):
        """Test when NODE_RANK=0 but LOCAL_RANK is not 0."""
        os.environ["NODE_RANK"] = "0"
        os.environ["LOCAL_RANK"] = "1"
        assert is_global_rank_zero() is False

    def test_with_node_rank_nonzero(self):
        """Test when NODE_RANK is not 0."""
        os.environ["NODE_RANK"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        assert is_global_rank_zero() is False

    def test_with_group_rank_fallback(self):
        """Test using GROUP_RANK as fallback for NODE_RANK."""
        os.environ["GROUP_RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        assert is_global_rank_zero() is True

        os.environ["GROUP_RANK"] = "1"
        assert is_global_rank_zero() is False

    def test_env_var_precedence(self):
        """Test that environment variables are checked in the expected order of precedence."""
        # RANK has highest precedence
        os.environ["RANK"] = "0"
        os.environ["SLURM_PROCID"] = "1"
        os.environ["OMPI_COMM_WORLD_RANK"] = "1"
        assert is_global_rank_zero() is True

        os.environ["RANK"] = "1"
        os.environ["SLURM_PROCID"] = "0"
        assert is_global_rank_zero() is False

        # Without RANK, SLURM_PROCID has next precedence
        del os.environ["RANK"]
        assert is_global_rank_zero() is True

        os.environ["SLURM_PROCID"] = "1"
        os.environ["OMPI_COMM_WORLD_RANK"] = "0"
        assert is_global_rank_zero() is False

        # Without RANK and SLURM_PROCID, OMPI_COMM_WORLD_RANK has next precedence
        del os.environ["SLURM_PROCID"]
        assert is_global_rank_zero() is True


class TestGetRank:
    """Test the get_rank function."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Clear all relevant environment variables before each test."""
        for var in ["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "NODE_RANK", "GROUP_RANK", "LOCAL_RANK"]:
            if var in os.environ:
                del os.environ[var]

    @mock.patch("torch.distributed.is_initialized", return_value=False)
    def test_not_distributed(self, mock_is_initialized):
        """Test when not in a distributed environment."""
        assert get_rank() == 0

    @mock.patch("torch.distributed.is_initialized", return_value=True)
    @mock.patch("torch.distributed.get_rank", return_value=2)
    def test_distributed_not_global_rank_zero(self, mock_dist_get_rank, mock_is_initialized):
        """Test when in a distributed environment and not global rank zero."""
        # Make sure is_global_rank_zero() returns False
        os.environ["RANK"] = "1"
        assert get_rank() == 2
        mock_dist_get_rank.assert_called_once()

    @mock.patch("torch.distributed.is_initialized", return_value=True)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_distributed_global_rank_zero(self, mock_dist_get_rank, mock_is_initialized):
        """Test when in a distributed environment and is global rank zero."""
        # Global rank is zero
        os.environ["RANK"] = "0"
        assert get_rank() == 0
        # Should not call torch.distributed.get_rank() when is_global_rank_zero() is True
        mock_dist_get_rank.assert_not_called()


class TestGetLastRank:
    """Test the get_last_rank function."""

    @mock.patch("torch.distributed.is_initialized", return_value=False)
    def test_not_distributed(self, mock_is_initialized):
        """Test when not in a distributed environment."""
        assert get_last_rank() == 0
        mock_is_initialized.assert_called_once()

    @mock.patch("torch.distributed.is_initialized", return_value=True)
    @mock.patch("torch.distributed.get_world_size", return_value=4)
    def test_distributed(self, mock_get_world_size, mock_is_initialized):
        """Test when in a distributed environment."""
        assert get_last_rank() == 3  # world_size - 1
        mock_is_initialized.assert_called_once()
        mock_get_world_size.assert_called_once()
