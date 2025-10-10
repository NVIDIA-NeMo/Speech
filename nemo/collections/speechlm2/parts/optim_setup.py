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
import re
from typing import Generator, Iterable

import hydra
import torch
from lightning import LightningModule

from nemo.utils import logging


def configure_optimizers(model: LightningModule):
    """
    Re-usable optimizer configuration function for top-level PyTorch Lightning modules in this collection.
    It sets up parameter freezing, optimizer, and LR scheduler.

    The ``model`` object is expected to have a ``model.cfg`` attribute with OmegaConf configuration.
    The following fields are expected:

    * ``optimizer`` with hydra-style ``_target_`` pointing to optimizer class, and the remaining options
        passed directly to its ``__init__`` method.

    * (optional) ``freeze_params`` with a list of regex pattern for identifying frozen parameters.

    * (optional) ``prevent_freeze_params`` with a list of regex pattern for keeping specific parameters trainable
        (overrides ``freeze_params``).

    * (optional) ``lr_scheduler`` with hydra-style ``_target_`` pointing to LR scheduler class,
        and the remaining options passed directly to its ``__init__`` method.

    Returns:
        PyTorch Lightning Trainer-compatible dict with structure::

            {
                "optimizer": <optimizer>,
                "lr_scheduler": {"scheduler": <lr_scheduler>, "interval": "step", "frequency": 1}
            }

    """
    assert hasattr(model, "cfg"), "Expected `model.cfg` attribute to exist."
    assert "optimizer" in model.cfg, "Expected `model.cfg` to contain 'optimizer' configuration."
    parameters = freeze_and_subset(
        model.named_parameters(),
        exclude_patterns=model.cfg.get("freeze_params", []),
        keep_patterns=model.cfg.get("prevent_freeze_params", []),
    )
    optimizer = hydra.utils.instantiate(model.cfg.optimizer, parameters, _convert_='all')
    ans = {"optimizer": optimizer}
    if "lr_scheduler" in model.cfg:
        lr_scheduler = hydra.utils.instantiate(model.cfg.lr_scheduler, optimizer)
        ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    return ans


def freeze_and_subset(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    exclude_patterns: list[str],
    keep_patterns: list[str] = None,
) -> Generator[torch.nn.Parameter, None, None]:
    """
    Utility used to freeze select model parameters, and skip them for the purpose
    of initializing an optimizer's parameter group.

    Args:
        named_parameters: The output of `torch.nn.Module.named_parameters()`
        exclude_patterns: A list of regex patterns matching parameter names to be frozen
            and excluded from optimization.
        keep_patterns: A list of regex patterns matching parameter names to be trained.
            This list overrides all matches to `exclude_patterns`.

    Returns:
        A generator over parameters, equivalent to calling `torch.nn.Module.parameters()`,
            that will be passed to the optimizer and trained.

    Example:

        >>> model = MyModel()
        ... # freeze all LLM parameters in "model.llm"
        ... params = freeze_and_subset(model.named_parameters(), [r'^llm\\.\\..+$'])
        ... optimizer = torch.optim.AdamW(params, lr=1e-3)

    """
    exclude_counter = {p: 0 for p in exclude_patterns}

    if not keep_patterns:
        keep_counter = {}

        def _must_keep(_) -> bool:
            return False

    else:
        keep_counter = {p: 0 for p in keep_patterns}
        compiled_keep_patterns = [re.compile(p) for p in keep_patterns]

        def _must_keep(name: str) -> bool:
            for p in compiled_keep_patterns:
                if p.match(name) is not None:
                    keep_counter[p.pattern] += 1
                    return True
            return False

    compiled_exclude_patterns = [re.compile(p) for p in exclude_patterns]

    def _exclude(name: str) -> bool:
        for p in compiled_exclude_patterns:
            if p.match(name) is not None:
                exclude_counter[p.pattern] += 1
                return True
        return False

    trainable, nontrainable = 0, 0
    for name, param in named_parameters:
        discard = False
        if _exclude(name) and not _must_keep(name):
            param.requires_grad = False
            discard = True
        if not discard:
            yield param
            trainable += param.numel()
        else:
            nontrainable += param.numel()
    total = trainable + nontrainable

    logging.info(f"Parameters | trainable={trainable} ({trainable / total:.2%}) | total={total}")

    if unused_excluded_patterns := [k for k, v in exclude_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_excluded_patterns) + "']"
        logging.warning(f"Parameter freezing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")

    if unused_keep_patterns := [k for k, v in keep_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_keep_patterns) + "']"
        logging.warning(f"Parameter freeze-preventing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")


def is_frozen(module: torch.nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())
