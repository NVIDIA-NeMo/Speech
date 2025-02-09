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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch.nn as nn
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from torch import nn

from nemo.collections.llm.peft.utils import wildcard_match
from nemo.utils.import_utils import safe_import_from

TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
HAVE_TE = all((HAVE_TE_COL_LINEAR, HAVE_TE_LN_COL_LINEAR, HAVE_TE_ROW_LINEAR))


@dataclass
class ModuleMatcher:
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    exclude_modules: List[str] = field(default_factory=list)
    canonical_mapping: Dict[str, Set] = field(default_factory=lambda: defaultdict(set))

    def match(self, m: nn.Module, name=None, prefix=None):
        """
        Determines whether a given module matches specified target patterns.

        This function checks if the provided module `m` should be included based on predefined
        mapping rules (`canonical_mapping`, `target_modules`, and `exclude_modules`). It returns
        the matching pattern if a match is found; otherwise, it returns `None`.

        Args:
            m (nn.Module): The module being checked.
            name (str, optional): The module's name.
            prefix (str, optional): A prefix to be used in constructing `full_name`.

        Returns:
            str or None: The matching pattern if a match is found; otherwise, `None`.

        Matching Logic:
        1) If `canonical_mapping` is defined, it checks:
        - Whether `name` exactly matches a pattern.
        - Whether `full_name` matches any regex pattern in `canonical_mapping`.
        2) If `target_modules` is defined, it follows the same logic as `canonical_mapping`.
        3) If neither `canonical_mapping` nor `target_modules` are defined, it ensures:
        - `name` is not in `exclude_modules`.
        - `full_name` does not match any `target_modules` patterns.
        - `m` is an instance of `nn.Linear`.

        Notes:
        - `exclude_modules` should only be non-empty if neither `canonical_mapping` nor `target_modules` are set.
        - The function asserts that `exclude_modules` is empty when using `canonical_mapping` or `target_modules`.
        """

        full_name = f"{prefix}.{name}" if prefix else name
        if len(self.canonical_mapping or []) > 0:
            """
            Find the element in canonical_mapping which
            1) matches the current `name` exactly, OR
            2) matches the current `full_name` with regex
            match is None if current module name doesn't match the specified targets.
            """
            assert len(self.exclude_modules) == 0
            for pattern in self.canonical_mapping:
                if name == pattern or wildcard_match(pattern, full_name):
                    return (pattern, full_name)
        elif len(self.target_modules or []) > 0:
            assert len(self.exclude_modules) == 0
            for pattern in self.target_modules:
                if name == pattern or wildcard_match(pattern, full_name):
                    return (pattern, full_name)
        else:
            linear_types = [ColumnParallelLinear, RowParallelLinear, nn.Linear]
            if HAVE_TE_COL_LINEAR:
                linear_types.append(TEColumnParallelLinear)
            if HAVE_TE_LN_COL_LINEAR:
                linear_types.append(TELayerNormColumnParallelLinear)
            if HAVE_TE_ROW_LINEAR:
                linear_types.append(TERowParallelLinear)
            linear_types = tuple(linear_types)

            if (
                not name in self.exclude_modules
                and not any(wildcard_match(pattern, full_name) for pattern in self.exclude_modules)
                and isinstance(m, linear_types)
            ):
                return (name, full_name)

        return None
