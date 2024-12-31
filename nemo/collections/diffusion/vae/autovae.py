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

import itertools
import time
from typing import Dict, List

import torch
import torch.profiler
from diffusers import AutoencoderKL
from torch import nn


class VAEGenerator:
    """
    A class for generating and searching different Variational Autoencoder (VAE) configurations.

    This class provides functionality to generate various VAE architecture configurations
    given a specific input resolution and compression ratio. It allows searching through a
    design space to find configurations that match given parameter and memory budgets.
    """

    def __init__(self, input_resolution: int = 1024, compression_ratio: int = 16) -> None:
        if input_resolution == 1024:
            assert compression_ratio in [8, 16]
        elif input_resolution == 2048:
            assert compression_ratio in [8, 16, 32]
        else:
            raise NotImplementedError("Higher resolution than 2028 is not implemented yet!")

        self._input_resolution = input_resolution
        self._compression_ratio = compression_ratio

    def _generate_input(self):
        """
        Generate a random input tensor with the specified input resolution.

        The tensor is placed on the GPU in half-precision (float16).
        """
        random_tensor = torch.rand(1, 3, self.input_resolution, self.input_resolution)
        random_tensor = random_tensor.to(dtype=torch.float16, device="cuda")
        return random_tensor

    def _count_parameters(self, model: nn.Module = None):
        """
        Count the number of trainable parameters in a given model.

        Args:
            model (nn.Module): The model for which to count parameters.

        Returns:
            int: The number of trainable parameters.
        """
        assert model is not None, "Please provide a nn.Module to count the parameters."
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _load_base_json_skeleton(self):
        """
        Load a base configuration skeleton for the VAE.

        Returns:
            dict: A dictionary representing the base configuration JSON skeleton.
        """
        skeleton = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.20.0.dev0",
            "_name_or_path": "../sdxl-vae/",
            "act_fn": "silu",
            "block_out_channels": [],
            "down_block_types": [],
            "force_upcast": False,
            "in_channels": 3,
            "latent_channels": -1,  # 16
            "layers_per_block": -1,  # 2
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,  # resolution size
            "scaling_factor": 0.13025,
            "up_block_types": [],
        }
        return skeleton

    def _generate_all_combinations(self, attr):
        """
        Generates all possible combinations from a search space dictionary.

        Args:
            attr (dict): A dictionary where each key has a list of possible values.

        Returns:
            List[Dict]: A list of dictionaries, each representing a unique combination of attributes.
        """
        keys = list(attr.keys())
        choices = [attr[key] for key in keys]
        all_combinations = list(itertools.product(*choices))

        combination_dicts = []
        for combination in all_combinations:
            combination_dict = {key: value for key, value in zip(keys, combination)}
            combination_dicts.append(combination_dict)

        return combination_dicts

    def _assign_attributes(self, choice):
        """
        Assign a chosen set of attributes to the base VAE configuration skeleton.

        Args:
            choice (dict): A dictionary of attributes to assign to the skeleton.

        Returns:
            dict: A dictionary representing the updated VAE configuration.
        """
        search_space_skleton = self._load_base_json_skeleton()
        search_space_skleton["down_block_types"] = choice["down_block_types"]
        search_space_skleton["up_block_types"] = choice["up_block_types"]
        search_space_skleton["block_out_channels"] = choice["block_out_channels"]
        search_space_skleton["layers_per_block"] = choice["layers_per_block"]
        search_space_skleton["latent_channels"] = choice["latent_channels"]
        return search_space_skleton

    def _search_space_16x1024(self):
        """
        Define the search space for a 16x compression ratio at 1024 resolution.

        Returns:
            dict: A dictionary defining lists of possible attribute values.
        """
        attr = {}
        attr["down_block_types"] = [["DownEncoderBlock2D"] * 5]
        attr["up_block_types"] = [["UpDecoderBlock2D"] * 5]
        attr["block_out_channels"] = [
            [128, 256, 512, 512, 512],
            [128, 256, 512, 512, 1024],
            [128, 256, 512, 1024, 2048],
            [64, 128, 256, 512, 512],
        ]
        attr["layers_per_block"] = [1, 2, 3]
        attr["latent_channels"] = [4, 16, 32, 64]
        return attr

    def _search_space_8x1024(self):
        """
        Define the search space for an 8x compression ratio at 1024 resolution.

        Returns:
            dict: A dictionary defining lists of possible attribute values.
        """
        attr = {}
        attr["down_block_types"] = [["DownEncoderBlock2D"] * 4]
        attr["up_block_types"] = [["UpDecoderBlock2D"] * 4]
        attr["block_out_channels"] = [[128, 256, 512, 512], [128, 256, 512, 1024], [64, 128, 256, 512]]
        attr["layers_per_block"] = [1, 2, 3]
        attr["latent_channels"] = [4, 16, 32, 64]
        return attr

    def _sort_data_in_place(self, data: List[Dict], mode: str) -> None:
        """
        Sort the list of design configurations in place based on a chosen mode.

        Args:
            data (List[Dict]): A list of dictionaries representing design configurations.
            mode (str): The sorting criterion. Can be 'abs_param_diff', 'abs_cuda_mem_diff', or 'mse'.
        """
        if mode == 'abs_param_diff':
            data.sort(key=lambda x: abs(x['param_diff']))
        elif mode == 'abs_cuda_mem_diff':
            data.sort(key=lambda x: abs(x['cuda_mem_diff']))
        elif mode == 'mse':
            data.sort(key=lambda x: (x['param_diff'] ** 2 + x['cuda_mem_diff'] ** 2) / 2)
        else:
            raise ValueError("Invalid mode. Choose from 'abs_param_diff', 'abs_cuda_mem_diff', 'mse'.")

    def _print_table(self, data, headers, col_widths):
        """
        Print a formatted table of the design choices.

        Args:
            data (List[Dict]): The data to print, each entry a design configuration.
            headers (List[str]): Column headers.
            col_widths (List[int]): Widths for each column.
        """
        # Create header row
        header_row = ""
        for header, width in zip(headers, col_widths):
            header_row += f"{header:<{width}}"
        print(header_row)
        print("-" * sum(col_widths))

        # Print each data row
        for item in data:
            row = f"{item['param_diff']:<{col_widths[0]}}"
            row += f"{item['cuda_mem_diff']:<{col_widths[1]}}"
            print(row)

    def search_for_target_vae(self, parameters_budget=0, cuda_max_mem=0):
        """
        Search through available VAE design choices to find one that best matches
        the given parameter and memory budgets.

        Args:
            parameters_budget (float, optional): The target number of parameters (in millions).
            cuda_max_mem (float, optional): The target maximum GPU memory usage (in MB).

        Returns:
            AutoencoderKL: The chosen VAE configuration that best matches the provided budgets.
        """
        if parameters_budget <= 0 and cuda_max_mem <= 0:
            raise ValueError("Please specify a valid parameter budget or cuda max memory budget")

        search_space_choices = []
        if self.input_resolution == 1024 and self.compression_ratio == 8:
            search_space = self._search_space_8x1024()
            search_space_choices = self._generate_all_combinations(search_space)
        elif self.input_resolution == 1024 and self.compression_ratio == 16:
            search_space = self._search_space_16x1024()
            search_space_choices = self._generate_all_combinations(search_space)

        inp_tensor = self._generate_input()
        inp_tensor = inp_tensor.to(dtype=torch.float16, device="cuda")
        design_choices = []

        for choice in search_space_choices:
            parameters_budget_diff = 0
            cuda_max_mem_diff = 0

            curt_design_json = self._assign_attributes(choice)
            print("-" * 20)
            print(choice)
            vae = AutoencoderKL.from_config(curt_design_json)
            vae = vae.to(dtype=torch.float16, device="cuda")
            total_params = self._count_parameters(vae)
            total_params /= 10**6
            # Reset peak memory statistics
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,  # Enables memory profiling
                record_shapes=True,  # Records tensor shapes
                with_stack=True,  # Records stack traces
            ) as prof:
                # Perform forward pass
                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = vae.encode(inp_tensor).latent_dist.sample()
                torch.cuda.synchronize()
                end_time = time.perf_counter()

            total_execution_time_ms = (end_time - start_time) * 1000

            # Get maximum memory allocated
            max_memory_allocated = torch.cuda.max_memory_allocated()
            max_memory_allocated = max_memory_allocated / (1024**2)

            parameters_budget_diff = parameters_budget - total_params
            cuda_max_mem_diff = cuda_max_mem - max_memory_allocated
            design_choices.append(
                {"param_diff": parameters_budget_diff, "cuda_mem_diff": cuda_max_mem_diff, "design": curt_design_json}
            )

            print(f"  Total params: {total_params}")
            print(f"  Max GPU Memory Usage: {max_memory_allocated} MB")
            print(f"  Total Execution Time: {total_execution_time_ms:.2f} ms")

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            print("-" * 20)
        sort_mode = "abs_param_diff"
        if parameters_budget == 0:
            sort_mode = "abs_cuda_mem_diff"
        elif cuda_max_mem == 0:
            sort_mode = "abs_param_diff"
        else:
            sort_mode = "mse"

        print("#" * 20)
        self._sort_data_in_place(design_choices, sort_mode)
        headers = ["param_diff (M)", "cuda_mem_diff (MB)"]
        col_widths = [12, 15]
        self._print_table(design_choices, headers, col_widths)

        vae = AutoencoderKL.from_config(design_choices[0]["design"])
        return vae

    @property
    def input_resolution(self) -> int:
        """
        Get the input resolution for the VAE.

        Returns:
            int: The input resolution.
        """
        return self._input_resolution

    @property
    def compression_ratio(self) -> float:
        """
        Get the compression ratio for the VAE.

        Returns:
            float: The compression ratio.
        """
        return self._compression_ratio
