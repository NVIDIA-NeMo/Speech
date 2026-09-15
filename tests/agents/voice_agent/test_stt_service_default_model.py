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
import ast
from pathlib import Path

# `nemo.agents.voice_agent` depends on the optional `pipecat-ai` stack (see
# examples/voice_agent/environment.yaml), which isn't part of this repo's test
# environment, so this test reads the source with `ast` instead of importing it.
REPO_ROOT = Path(__file__).parents[3]
STT_PATH = REPO_ROOT / "nemo/agents/voice_agent/pipecat/services/nemo/stt.py"


def _module_level_list(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"module-level assignment `{name}` not found in {STT_PATH}")


def _init_kwonly_default(tree: ast.Module, class_name: str, arg_name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for arg, default in zip(item.args.kwonlyargs, item.args.kw_defaults):
                        if arg.arg == arg_name and default is not None:
                            return ast.literal_eval(default)
    raise AssertionError(f"could not find a default for {class_name}.__init__(..., {arg_name}=...)")


def test_nemo_stt_service_default_model_is_an_eou_model():
    """NemoSTTService.__init__ auto-detects `has_turn_taking` from whether `model`
    is a member of ASR_EOU_MODELS:

        if has_turn_taking is None:
            has_turn_taking = True if model in ASR_EOU_MODELS else False

    The service's own out-of-the-box default for `model` must therefore itself be
    a member of ASR_EOU_MODELS, or `NemoSTTService()` silently disables EOU-based
    turn taking for its own default model.
    """
    tree = ast.parse(STT_PATH.read_text())
    eou_models = _module_level_list(tree, "ASR_EOU_MODELS")
    default_model = _init_kwonly_default(tree, "NemoSTTService", "model")

    assert default_model in eou_models, (
        f"NemoSTTService's default `model` value {default_model!r} is not a member of "
        f"ASR_EOU_MODELS {eou_models!r} -- has_turn_taking auto-detection silently "
        "defaults to False for the service's own default model."
    )
