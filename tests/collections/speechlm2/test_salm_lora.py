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
from types import SimpleNamespace

from omegaconf import DictConfig
from peft import PeftModel
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo.collections.speechlm2.parts.lora import maybe_install_lora


def _make_qwen3_stub():
    llm = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    return SimpleNamespace(
        cfg=DictConfig(
            {
                "prevent_freeze_params": [],
                "lora": {
                    "r": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                    "task_type": "CAUSAL_LM",
                },
            }
        ),
        llm=llm,
        embed_tokens=llm.model.embed_tokens,
    )


def test_maybe_install_lora_restores_qwen3_input_embeddings_temporarily():
    model = _make_qwen3_stub()
    del model.llm.model.embed_tokens

    maybe_install_lora(model)

    assert isinstance(model.llm, PeftModel)
    assert hasattr(model.llm.base_model.model.model.layers[0].self_attn.q_proj, "lora_A")
    assert model.cfg.prevent_freeze_params == [r"^.+\.lora_.+$"]
    assert not hasattr(model.llm.base_model.model.model, "embed_tokens")
