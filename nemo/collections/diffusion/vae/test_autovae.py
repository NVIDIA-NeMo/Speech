# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import unittest

import torch
from autovae import VAEGenerator


class TestVAEGenerator(unittest.TestCase):
    """Unit tests for the VAEGenerator class."""

    def setUp(self):
        # Common setup for tests
        self.input_resolution = 1024
        self.compression_ratio = 8
        self.generator = VAEGenerator(input_resolution=self.input_resolution, compression_ratio=self.compression_ratio)

    def test_initialization_valid(self):
        """Test that valid initialization parameters set the correct properties."""
        generator = VAEGenerator(input_resolution=1024, compression_ratio=8)
        self.assertEqual(generator.input_resolution, 1024)
        self.assertEqual(generator.compression_ratio, 8)

        generator = VAEGenerator(input_resolution=2048, compression_ratio=16)
        self.assertEqual(generator.input_resolution, 2048)
        self.assertEqual(generator.compression_ratio, 16)

    def test_initialization_invalid(self):
        """Test that invalid initialization parameters raise an error."""
        with self.assertRaises(NotImplementedError):
            VAEGenerator(input_resolution=4096, compression_ratio=16)

    def test_generate_input(self):
        """Test that _generate_input produces a tensor with the correct shape and device."""
        input_tensor = self.generator._generate_input()
        expected_shape = (1, 3, self.input_resolution, self.input_resolution)
        self.assertEqual(input_tensor.shape, expected_shape)
        self.assertEqual(input_tensor.dtype, torch.float16)
        self.assertEqual(input_tensor.device.type, "cuda")

    def test_count_parameters(self):
        """Test that _count_parameters correctly counts model parameters."""
        model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5))
        expected_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_count = self.generator._count_parameters(model)
        self.assertEqual(param_count, expected_param_count)

    def test_load_base_json_skeleton(self):
        """Test that _load_base_json_skeleton returns the correct skeleton."""
        skeleton = self.generator._load_base_json_skeleton()
        expected_keys = {
            "_class_name",
            "_diffusers_version",
            "_name_or_path",
            "act_fn",
            "block_out_channels",
            "down_block_types",
            "force_upcast",
            "in_channels",
            "latent_channels",
            "layers_per_block",
            "norm_num_groups",
            "out_channels",
            "sample_size",
            "scaling_factor",
            "up_block_types",
        }
        self.assertEqual(set(skeleton.keys()), expected_keys)

    def test_generate_all_combinations(self):
        """Test that _generate_all_combinations generates all possible combinations."""
        attr = {"layers_per_block": [1, 2], "latent_channels": [4, 8]}
        combinations = self.generator._generate_all_combinations(attr)
        expected_combinations = [
            {"layers_per_block": 1, "latent_channels": 4},
            {"layers_per_block": 1, "latent_channels": 8},
            {"layers_per_block": 2, "latent_channels": 4},
            {"layers_per_block": 2, "latent_channels": 8},
        ]
        self.assertEqual(len(combinations), len(expected_combinations))
        for combo in expected_combinations:
            self.assertIn(combo, combinations)

    def test_assign_attributes(self):
        """Test that _assign_attributes correctly assigns attributes to the skeleton."""
        choice = {
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "block_out_channels": [64, 128, 256, 512],
            "layers_per_block": 2,
            "latent_channels": 16,
        }
        skeleton = self.generator._assign_attributes(choice)
        self.assertEqual(skeleton["down_block_types"], choice["down_block_types"])
        self.assertEqual(skeleton["up_block_types"], choice["up_block_types"])
        self.assertEqual(skeleton["block_out_channels"], choice["block_out_channels"])
        self.assertEqual(skeleton["layers_per_block"], choice["layers_per_block"])
        self.assertEqual(skeleton["latent_channels"], choice["latent_channels"])

    def test_search_space_16x1024(self):
        """Test that _search_space_16x1024 returns the correct search space."""
        search_space = self.generator._search_space_16x1024()
        expected_keys = {
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "layers_per_block",
            "latent_channels",
        }
        self.assertEqual(set(search_space.keys()), expected_keys)
        self.assertTrue(all(isinstance(v, list) for v in search_space.values()))

    def test_sort_data_in_place(self):
        """Test that _sort_data_in_place correctly sorts data based on the specified mode."""
        data = [
            {"param_diff": 10, "cuda_mem_diff": 100},
            {"param_diff": 5, "cuda_mem_diff": 50},
            {"param_diff": -3, "cuda_mem_diff": 30},
            {"param_diff": 7, "cuda_mem_diff": 70},
        ]
        # Test sorting by absolute parameter difference
        self.generator._sort_data_in_place(data, mode="abs_param_diff")
        expected_order_param = [-3, 5, 7, 10]
        actual_order_param = [item["param_diff"] for item in data]
        self.assertEqual(actual_order_param, expected_order_param)

        # Test sorting by absolute CUDA memory difference
        self.generator._sort_data_in_place(data, mode="abs_cuda_mem_diff")
        expected_order_mem = [30, 50, 70, 100]
        actual_order_mem = [item["cuda_mem_diff"] for item in data]
        self.assertEqual(actual_order_mem, expected_order_mem)

        # Test sorting by mean squared error (MSE)
        self.generator._sort_data_in_place(data, mode="mse")
        expected_order_mse = [-3, 5, 7, 10]  # Computed based on MSE values
        actual_order_mse = [item["param_diff"] for item in data]
        self.assertEqual(actual_order_mse, expected_order_mse)

    def test_search_for_target_vae_invalid(self):
        """Test that search_for_target_vae raises an error when no budget is specified."""
        with self.assertRaises(ValueError):
            self.generator.search_for_target_vae(parameters_budget=0, cuda_max_mem=0)


if __name__ == "__main__":
    unittest.main()
