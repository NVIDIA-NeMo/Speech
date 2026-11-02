# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Optimizer parameter groups selected by the owning module type or by parameter name."""

import fnmatch
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Type, Union

import torch

from nemo.utils.model_utils import import_class_by_path

__all__ = ['apply_param_group_rules']

SELECTOR_KEYS = ('module_types', 'param_names')


def apply_param_group_rules(
    module: torch.nn.Module,
    param_groups: Union[Sequence[Dict[str, Any]], Iterable[torch.Tensor]],
    rules: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Split optimizer parameter groups by rules that select parameters and set their group options.

    A rule selects parameters by the types of the modules that own them
    (``module_types``: classes or import paths; a parameter shared between modules matches when any owner matches)
    or by ``fnmatch`` patterns on their names as :meth:`torch.nn.Module.named_parameters` of ``module`` reports them
    (``param_names``; every name of a shared parameter counts, a second name within one module as well),
    and its remaining keys are the options for the selected parameters,
    such as ``weight_decay: 0.0`` or an optimizer-specific flag.
    Each group of ``param_groups`` is split into the parameters no rule selects, which keep the group's options,
    and one sub-group per combination of matching rules,
    which carries the group's options overridden by the options of those rules in the order of ``rules``.
    Parameters keep their order, sub-groups appear in the order of their first parameter,
    and empty groups are dropped.
    The rules decide the optimizer's parameter groups:
    a changed rule set changes their membership and order,
    so an optimizer state saved under other rules (a training checkpoint) fails to load,
    or loads onto other parameters when the group counts and sizes still match;
    keep the rules unchanged when resuming.

    Example::

        rules = {
            'no_weight_decay': {
                'module_types': ['torch.nn.Embedding', 'torch.nn.LayerNorm'],
                'param_names': ['*.bias'],
                'weight_decay': 0.0,
            },
        }
        groups = apply_param_group_rules(model, [{'params': list(model.parameters())}], rules)

    Args:
        module: the module whose parameters the groups hold; it resolves owners and names
        param_groups: parameter groups as the optimizer takes them,
            dicts with ``params`` (parameters or a single parameter) or plain parameters
        rules: rule name to rule; the name labels the rule in error messages and documentation only

    Returns:
        the split parameter groups
    """
    param_groups = list(param_groups)
    if param_groups and not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    owners: Dict[int, List[torch.nn.Module]] = {}
    names: Dict[int, List[str]] = {}
    for module_name, owner in module.named_modules(remove_duplicate=False):
        for param_name, param in owner.named_parameters(recurse=False, remove_duplicate=False):
            owners.setdefault(id(param), []).append(owner)
            names.setdefault(id(param), []).append(f"{module_name}.{param_name}" if module_name else param_name)

    compiled = [_compile_rule(name, rule) for name, rule in rules.items()]

    def matching_rules(param: torch.Tensor) -> Tuple[str, ...]:
        return tuple(
            name
            for name, types, patterns, _ in compiled
            if (types and any(isinstance(owner, types) for owner in owners.get(id(param), ())))
            or any(fnmatch.fnmatchcase(alias, pattern) for alias in names.get(id(param), ()) for pattern in patterns)
        )

    result: List[Dict[str, Any]] = []
    for group in param_groups:
        group_params = group['params']
        if isinstance(group_params, torch.Tensor):
            group_params = [group_params]
        buckets: Dict[Tuple[str, ...], List[torch.Tensor]] = {}
        for param in group_params:
            buckets.setdefault(matching_rules(param), []).append(param)
        for matched, params in buckets.items():
            new_group = {key: value for key, value in group.items() if key != 'params'}
            for name, _, _, options in compiled:
                if name in matched:
                    new_group.update(options)
            new_group['params'] = params
            result.append(new_group)
    return result


def _compile_rule(
    name: str, rule: Mapping[str, Any]
) -> Tuple[str, Tuple[Type[torch.nn.Module], ...], Tuple[str, ...], Dict[str, Any]]:
    types = tuple(_resolve_module_type(name, entry) for entry in rule.get('module_types') or ())
    patterns = tuple(rule.get('param_names') or ())
    options = {key: value for key, value in rule.items() if key not in SELECTOR_KEYS}
    if not types and not patterns:
        raise ValueError(f"Parameter group rule {name!r} selects nothing: set `module_types` or `param_names`")
    if not options:
        raise ValueError(f"Parameter group rule {name!r} sets no parameter group options")
    if 'params' in options:
        raise ValueError(f"Parameter group rule {name!r} must not set `params`")
    return name, types, patterns, options


def _resolve_module_type(name: str, entry: Union[str, Type[torch.nn.Module]]) -> Type[torch.nn.Module]:
    module_type = import_class_by_path(entry) if isinstance(entry, str) else entry
    if not (isinstance(module_type, type) and issubclass(module_type, torch.nn.Module)):
        raise TypeError(f"Parameter group rule {name!r}: {entry!r} is not a torch.nn.Module class")
    return module_type
