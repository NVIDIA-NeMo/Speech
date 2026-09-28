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

"""CPU regression for actual target/draft compute_logits methods without vLLM installed."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    'module,class_name', [('model', 'NeMoSpeechLMForConditionalGeneration'), ('mtp', 'NeMoSpeechLMMTP')]
)
@pytest.mark.parametrize('shape', [(1, 16), (4, 16)])
@pytest.mark.parametrize(
    'audio_token_id,runtime_added_ids',
    [(None, []), (0, []), (4, []), (0, [0]), (4, [4]), (6, [6]), (15, [15]), (4, [0, 4])],
)
def test_untrained_rows_cannot_win(module, class_name, shape, audio_token_id, runtime_added_ids):
    path = ROOT / 'nemo/collections/speechlm2/vllm/salm' / f'{module}.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_logits']
    node = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id='Base', ctx=ast.Load())],
        keywords=[],
        body=methods or [ast.Pass()],
        decorator_list=[],
    )

    class Base:
        def compute_logits(self, hidden_states, *args, **kwargs):
            return hidden_states

    ns = {'torch': torch, 'Base': Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    obj = ns[class_name]()
    obj.config = SimpleNamespace(
        speechlm_output_vocab_size=6,
        audio_token_id=audio_token_id,
        speechlm_runtime_added_token_ids=runtime_added_ids,
    )
    obj.language_model = Base()
    logits = torch.full(shape, -3.0)
    logits[..., 2] = -1.0  # correct EOS, despite negative absolute score
    logits[..., 6:] = 0.0  # zero-padded runtime-only output rows
    if audio_token_id is not None:
        logits[..., audio_token_id] = 1.0  # make the preservation/suppression policy observable
    valid = logits[..., :6].clone()
    for token_id in runtime_added_ids:
        if token_id < 6:
            valid[..., token_id] = -torch.inf
    result = obj.compute_logits(logits)
    assert torch.equal(result[..., :6], valid)
    assert torch.isneginf(result[..., 6:]).all()
    expected_winner = (
        audio_token_id
        if audio_token_id is not None and audio_token_id < 6 and audio_token_id not in runtime_added_ids
        else 2
    )
    assert (result.argmax(dim=-1) == expected_winner).all()
    assert obj.compute_logits(None) is None
