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

from typing import Dict, List, Optional
from nemo.agents.voice_agent.evaluation.scenarios.classes import Scenario
ALL_EVAL_SCENARIOS: Dict[str, Scenario] = {}


def register_eval_scenario(cls):
    """Class decorator that registers a scenario class into ALL_EVAL_SCENARIOS."""
    if not issubclass(cls, Scenario):
        raise ValueError(f"Class {cls.__name__} is not a subclass of Scenario")
    key = getattr(cls, "name", cls.__name__)
    ALL_EVAL_SCENARIOS[key] = cls
    return cls

def get_eval_scenario(name: str, **kwargs) -> Optional[Scenario]:
    """
    Get an evaluation scenario by name.
    """
    if name not in ALL_EVAL_SCENARIOS:
        return None
    return ALL_EVAL_SCENARIOS[name](**kwargs)


def list_eval_scenarios() -> List[str]:
    """
    List all evaluation scenarios.
    """
    return list(ALL_EVAL_SCENARIOS.keys())

# Import data subpackage to trigger @register_eval_scenario decorators.
# Must be at the end to avoid circular imports (data modules import register_eval_scenario).
import nemo.agents.voice_agent.evaluation.scenarios.data  # noqa: E402, F401
