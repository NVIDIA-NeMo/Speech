# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""eva_airline scenario package.

Layout:
    base.py        — ``EvaAirlineBaseScenario`` + 5 hand-authored seed scenarios.
    group_Nx.py    — auto-scaffolded scenarios for eva sub-flow ``N.*``
                     (generated from ``eva_airline_dataset.jsonl`` via
                     ``nemo_experiments/generate_eva_airline_scaffolds.py``).

This ``__init__`` re-exports ``EvaAirlineBaseScenario`` so the group submodules
can keep their canonical import path (``from ...eva_airline import EvaAirlineBaseScenario``)
and triggers the group submodule imports so their ``@register_eval_scenario``
decorators fire.
"""

# Trigger @register_eval_scenario decorators in each group module. Order matches
# the eva taxonomy (1.x = voluntary, 2.x = IRROPS, etc.).
from nemo.agents.voice_agent.evaluation.scenarios.data.eva_airline import (  # noqa: F401, E402
    group_1x,
    group_2x,
    group_3x,
    group_4x,
    group_5x,
    group_6x,
    group_7x,
)
from nemo.agents.voice_agent.evaluation.scenarios.data.eva_airline.base import EvaAirlineBaseScenario

__all__ = ["EvaAirlineBaseScenario"]
