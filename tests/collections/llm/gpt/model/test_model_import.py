# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os

import torch

torch.set_grad_enabled(False)


config_name_to_hf_id = {
    'MistralConfig7B': 'mistralai/Mistral-7B-v0.1',
    # 'Nemotron3Config4B': 'nvidia/Minitron-4B-Base',
    'Llama2Config7B': 'meta-llama/Llama-2-7b-hf',
    'Llama3Config8B': 'meta-llama/Meta-Llama-3-8B',
    # 'MixtralConfig8x7B': 'mistralai/Mixtral-8x7B-v0.1',
    # 'ChatGLM2Config6B': 'THUDM/chatglm2-6b',
    'GemmaConfig2B': 'google/gemma-2b',
    # 'Baichuan2Config7B': 'baichuan-inc/Baichuan2-7B-Base',
}


def strip_digits_from_end(s):
    s = list(s)
    while s and s[-1].isdigit():
        s = s[:-1]
    return ''.join(s)


def get_modulename_from_config_name(config_name):
    # Finds name of model class from config class name.
    # Llama2Config7B -> Llama2Model (fail) -> LlamaModel
    import nemo.collections.llm.gpt.model as nemo_ux_llms

    assert 'Config' in config_name, 'Expected config_name to contain "Config".'
    module_name = config_name.split('Config')[0] + "Model"
    if not hasattr(nemo_ux_llms, module_name):
        module_name = strip_digits_from_end(config_name.split('Config')[0]) + "Model"
    if not hasattr(nemo_ux_llms, module_name):
        raise ValueError("Failed to get modulename")
    return module_name


def generate_twolayer_checkpoints(config_name, hf_id):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    # Reduce number of layers to two.
    if hasattr(config, 'num_hidden_layers'):
        print(config.num_hidden_layers)
        config.num_hidden_layers = 2
    elif hasattr(config, 'num_layers'):
        print(config.num_layers)
        config.num_layers = 2
    else:
        print(config)
        raise ValueError("HF config has neither num_hidden_layers nor num_layers")

    # Calling random init is slow.
    with torch.device('meta'):
        model_2l = AutoModel.from_config(config, trust_remote_code=True)

    model_2l = model_2l.to_empty(device='cpu')
    state = model_2l.state_dict()
    # Fill state-dict with i/n
    n = len(state.items())
    for i, key in enumerate(state.keys()):
        value = torch.empty_like(state[key]).fill_(i / n)
        state[key] = value
    model_2l.load_state_dict(state)
    model_2l.save_pretrained(f'hf_ckpts/{config_name}/', safe_serialization=False)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    hf_tokenizer.save_pretrained(f'hf_ckpts/{config_name}/', trust_remote_code=True)


def import_from_hf(config_name, hf_path):
    import nemo.collections.llm.gpt.model as nemo_ux_llms
    from nemo.collections.llm import import_ckpt

    module_name = get_modulename_from_config_name(config_name)
    config_cls = getattr(nemo_ux_llms, config_name)
    model_cls = getattr(nemo_ux_llms, module_name)
    model = model_cls(config_cls())
    import_ckpt(model=model, source=hf_path)


if __name__ == '__main__':
    for config_name, hf_id in config_name_to_hf_id.items():
        for env_var in ['NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN']:
            if env_var in os.environ:
                del os.environ[env_var]
        src = f'hf:///home/TestData/nemo2_ckpt/{config_name}'
        import_from_hf(config_name, src)
