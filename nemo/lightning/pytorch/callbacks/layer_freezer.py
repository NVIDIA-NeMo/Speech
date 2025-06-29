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
import math
from typing import Dict, List, Tuple, Union

from lightning.pytorch.callbacks.callback import Callback

from nemo.lightning.io.mixin import IOMixin

ScheduleValue = Union[int, List[int], Tuple[int, int]]  # -1, N, [start, end], [start, -1]


def _resolve_attr(root, path: str):
    """
    Traverse dotted attribute path (“encoder.layer1”) from root.
    """
    m = root
    for part in path.split('.'):
        m = getattr(m, part)
    return m


def make_start_end(name: str, spec: Union[int, list[int]]):
    """Translates spec to start/end steps, for example,
    spec = -1           -> (0, inf)
    spec = N (int>0)    -> (N, int)
    spec = [start, end] -> (start, end)


    Args:
        name (str): name layer
        spec (Union[int, list[int]]): spec.

    Raises:
        ValueError: if spec is not int/list/tuple

    Returns:
        tuple(int, int): returns start/end
    """
    # Normalize to (start, end) where end==inf means “forever”
    if isinstance(spec, int):
        if spec == -1:  # forever
            start, end = 0, math.inf
        else:  # first N steps
            start, end = 0, spec - 1
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        start, end = spec
        start = max(start, 0)
        if end < 0:
            end = math.inf
    else:
        raise ValueError(f"Invalid schedule for '{name}': {spec}")
    return start, end


class LayerFreezer(Callback, IOMixin):
    """
    Freezes sub-modules of a LightningModule based on the list provided. The list of layers should
    be the full FQN.

    Instantiate
    -----------
    # to keep layers frozen for all training
    callback = LayerFreezer(['layer1', 'layer2',])
    # for some steps
    callback = LayerFreezer({'layer1': 10, 'layer2': (10, 100)})

    trainer  = pl.Trainer(callbacks=[callback], ...)
    """

    def __init__(self, schedule: Union[List[str], Dict[str, ScheduleValue]]):
        """
        Args
        ----
        schedule: Union[list, dict]
        - dict
            key   = attribute path of sub-module inside LightningModule
            value = one of
                    : -1                -> frozen for entire run
                    :  N (int>0)        -> frozen for first N steps (0..N-1)
                    : [start, end]      -> frozen if start <= step <= end
                    use -1 for "until end of training"
        - list:
            key   = attribute path of sub-module inside LightningModule
            value = -1 (hardcoded; use a dict if you want to specify manually).
        """
        super().__init__()
        assert isinstance(schedule, (list, dict)), type(schedule)
        if isinstance(schedule, list):
            schedule = {item: -1 for item in schedule}

        self.schedule: Dict[str, Tuple[int, float]] = {}
        self.frozen_state: Dict[str, bool] = {}  # last applied state

        for name, spec in schedule.items():
            self.schedule[name] = make_start_end(name, spec)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _resolve_attr(root, path: str):
        """
        Traverse dotted attribute path (“encoder.layer1”) from root.
        """
        m = root
        for part in path.split('.'):
            m = getattr(m, part)
        return m

    def _apply_freeze(self, module, freeze: bool):
        """
        Enable/disable gradients + switch (eval/train) mode.
        """
        for p in module.parameters():
            p.requires_grad = not freeze
        # Optional: also flip training mode so dropout / BN are disabled.
        module.eval() if freeze else module.train()

    # --------------------------------------------------------------------- #
    # Lightning hooks
    # --------------------------------------------------------------------- #
    def on_train_batch_start(self, trainer, pl_module, *_):
        """
        freezes layers listed on frozen_layers
        Args:
            trainer (Trainer): the trainer
            pl_module (LightningModule): model
        """
        step = trainer.global_step

        for name, (start, end) in self.schedule.items():
            should_be_frozen = start <= step <= end
            # skip if status unchanged since last check
            if self.frozen_state.get(name, None) == should_be_frozen:
                continue

            submod = self._resolve_attr(pl_module, name)
            self._apply_freeze(submod, should_be_frozen)
            self.frozen_state[name] = should_be_frozen

    def on_train_start(self, trainer, pl_module):
        """
        on_train_start
        In case we resume from checkpoint, re-establish correct state
        Args:
            trainer (Trainer): the trainer
            pl_module (LightningModule): model
        """
        self.on_train_batch_start(trainer, pl_module, None, 0)
