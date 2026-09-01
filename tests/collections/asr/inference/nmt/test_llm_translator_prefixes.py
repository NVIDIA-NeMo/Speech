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

from types import SimpleNamespace

from nemo.collections.asr.inference.nmt import llm_translator
from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator
from nemo.collections.asr.inference.nmt.prompts import RivaV2TranslatorPromptTemplate


class _PairTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [(start, min(start + 2, len(text))) for start in range(0, len(text), 2)]
        result = {"input_ids": [hash(text[start:end]) for start, end in offsets]}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]


def test_token_lcp_retreats_to_a_shared_model_token_boundary():
    translator = LLMTranslator.__new__(LLMTranslator)
    translator.prefix_tokenizer = _PairTokenizer()
    translator.prefix_boundary_mode = "token"
    translator.waitk = -1

    prefix = translator.get_prefixes(["source"], ["会议开幕"], ["会议开始"])[0]

    assert prefix == "会议"


def test_local_model_resolution_honors_revision(monkeypatch):
    calls = []

    def _snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/cached/revision"

    monkeypatch.setattr(llm_translator, "snapshot_download", _snapshot_download)
    translator = LLMTranslator.__new__(LLMTranslator)

    path = translator._get_local_model_path("nvidia/model", revision="abc123")

    assert path == "/cached/revision"
    assert calls == [{"repo_id": "nvidia/model", "revision": "abc123", "local_files_only": True}]


def test_riva_v2_uses_model_chat_template():
    chat_calls = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            chat_calls.append((messages, kwargs))
            return "tokenizer-rendered-prompt"

    class _Model:
        def generate(self, prompts, sampling_params, use_tqdm):
            assert prompts == ["tokenizer-rendered-prompt"]
            return [SimpleNamespace(outputs=[SimpleNamespace(text="Uebersetzung")])]

    translator = LLMTranslator.__new__(LLMTranslator)
    translator.prompt_template = RivaV2TranslatorPromptTemplate
    translator.prefix_tokenizer = _Tokenizer()
    translator.nmt_model = _Model()
    translator.sampling_params = object()
    translator.prefix_boundary_mode = "whitespace"

    result = translator.translate_batch(
        ["Current source"],
        [""],
        ["en"],
        ["de"],
        [["Earlier source"]],
        [["Fruehere Ausgabe"]],
    )

    assert result == ["Uebersetzung"]
    assert chat_calls == [
        (
            [
                {"role": "system", "content": "en-de"},
                {"role": "user", "content": "Earlier source"},
                {"role": "assistant", "content": "Fruehere Ausgabe"},
                {"role": "user", "content": "Current source"},
            ],
            {"tokenize": False, "add_generation_prompt": True, "continue_final_message": False},
        )
    ]


def test_model_loading_preserves_process_level_gpu_isolation(monkeypatch):
    loaded = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(llm_translator, "LLM", lambda **kwargs: loaded.append(kwargs) or object())

    translator = LLMTranslator.__new__(LLMTranslator)
    translator.model_name = "nvidia/model"
    translator.device_id = 0
    monkeypatch.setattr(translator, "_get_local_model_path", lambda *args, **kwargs: None)

    translator.load_model({})

    assert llm_translator.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert loaded == [{"model": "nvidia/model"}]
