# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration types supported by the weights-only loader."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import is_dataclass
from enum import Enum
from pickle import UnpicklingError
from typing import Any, Dict, List, Set, Tuple, Union, get_args, get_origin, get_type_hints

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata, UnionNode
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


def checkpoint_config_types() -> list:
    """Return the fixed data types and restricted annotation loader used by checkpoints."""
    return [
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        UnionNode,
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
        Dict,
        List,
        Set,
        Tuple,
        Union,
        (_restore_checkpoint_annotation, "_operator.getitem"),
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


@contextmanager
def checkpoint_config_loading():
    """Add scoped configuration types while retaining existing process registrations."""
    existing = set(torch.serialization.get_safe_globals())
    additions = [entry for entry in checkpoint_config_types() if entry not in existing]
    with torch.serialization.safe_globals(additions):
        yield


def register_checkpoint_config_types() -> None:
    """Register the fixed configuration container types used by Lightning checkpoints."""
    torch.serialization.add_safe_globals(checkpoint_config_types())


def register_checkpoint_schema_types(*schemas: type) -> None:
    """Register source-defined dataclass schemas, nested schemas, and annotated enums.

    Call from the schema's module after its definitions. Only trusted source types
    are traversed; checkpoint contents never select imports or additional globals.
    """
    pending = list(schemas)
    registered = set()
    while pending:
        schema = pending.pop()
        arguments = get_args(schema)
        if arguments:
            pending.extend(arguments)
        elif isinstance(schema, type) and schema not in registered:
            if is_dataclass(schema):
                registered.add(schema)
                pending.extend(get_type_hints(schema).values())
            elif issubclass(schema, Enum):
                registered.add(schema)
    torch.serialization.add_safe_globals(list(registered))


def _restore_checkpoint_annotation(origin, arguments):
    """Rebuild typing annotations without permitting general object item access."""
    if not any(origin is candidate for candidate in (Dict, List, Set, Tuple, Union)):
        raise UnpicklingError('Checkpoint item access is restricted to typing annotations.')
    items = arguments if isinstance(arguments, tuple) else (arguments,)
    if not all(_is_checkpoint_annotation(item) for item in items):
        raise UnpicklingError('Checkpoint annotation arguments must be types.')
    return origin[arguments]


def _is_checkpoint_annotation(value):
    if value is None or value is Ellipsis or isinstance(value, type):
        return True
    origin = get_origin(value)
    return any(origin is candidate for candidate in (dict, list, set, tuple, Union)) and all(
        _is_checkpoint_annotation(argument) for argument in get_args(value)
    )
