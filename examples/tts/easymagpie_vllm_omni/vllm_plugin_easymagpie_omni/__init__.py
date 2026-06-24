# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""vLLM plugin: register ``EasyMagpieTTS`` as a model architecture for vLLM-Omni.

Loaded by vLLM in the parent process and each EngineCore subprocess via the
``vllm.general_plugins`` entry point. The lazy ``<module>:<class>`` target means
the (NeMo-free) model module is only imported when vLLM resolves the
architecture, keeping heavy imports out of the parent process.
"""

_TARGET = "easymagpie_vllm_omni.easymagpie:EasyMagpieTTSForConditionalGeneration"
_ARCHS = ("EasyMagpieTTS", "EasyMagpieTTSForConditionalGeneration")


def register() -> None:
    """Register the model class under all supported arch names.

    The architecture must be registered in **both** registries:

    * ``vllm.ModelRegistry`` — the stock vLLM global registry.
    * ``vllm_omni``'s ``OmniModelRegistry`` — a *separate* ``_ModelRegistry``
      instance that the vLLM-Omni engine actually consults when resolving a
      model architecture. Registering only in the stock registry leaves the
      omni engine reporting ``Model architectures [...] are not supported``.
    """
    from vllm import ModelRegistry

    registries = [ModelRegistry]
    try:
        from vllm_omni.model_executor.models import OmniModelRegistry

        registries.append(OmniModelRegistry)
    except Exception:
        # vllm_omni not installed — stock vLLM registration is enough.
        pass

    for registry in registries:
        for arch in _ARCHS:
            if arch not in registry.get_supported_archs():
                registry.register_model(arch, _TARGET)
