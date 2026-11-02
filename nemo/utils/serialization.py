# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration types supported by the weights-only loader."""

from collections import defaultdict
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import (
    AnyNode,
    BooleanNode,
    BytesNode,
    EnumNode,
    FloatNode,
    IntegerNode,
    InterpolationResultNode,
    PathNode,
    StringNode,
    ValueNode,
)


def register_checkpoint_config_types() -> None:
    """Allow OmegaConf's data containers when Lightning reads existing checkpoints.

    Lightning owns the load calls during resume, so registration is process-wide.
    Keep this list explicit: checkpoint contents must never add their own types.
    This does not enable full pickle loading or resolve configuration interpolations.
    """
    torch.serialization.add_safe_globals(
        [
            DictConfig,
            ListConfig,
            ContainerMetadata,
            Metadata,
            AnyNode,
            BooleanNode,
            BytesNode,
            EnumNode,
            FloatNode,
            IntegerNode,
            InterpolationResultNode,
            PathNode,
            StringNode,
            ValueNode,
            Any,
            defaultdict,
            dict,
            list,
            tuple,
            int,
            float,
            str,
            bool,
            bytes,
        ]
    )
