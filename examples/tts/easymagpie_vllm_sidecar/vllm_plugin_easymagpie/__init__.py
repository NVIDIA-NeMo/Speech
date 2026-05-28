"""vLLM plugin: register ``EasyMagpieSmallMamba`` as a model architecture.

Loaded by vLLM in the parent process and each EngineCore subprocess via the
entry-point group ``vllm.general_plugins``.

The legacy arch name ``EasyMagpieSmallMambaV2`` is also registered (alias) so
existing checkpoints whose ``config.json`` lists it still resolve. The lazy
``<module>:<class>`` form means the model module is only imported when vLLM
resolves the architecture — keeping nemo imports out of the parent process.
"""

_TARGET = "easymagpie_vllm.easymagpie_smallmamba:EasyMagpieSmallMamba"
_ARCHS = ("EasyMagpieSmallMamba", "EasyMagpieSmallMambaV2")


def register() -> None:
    """Register the model class under all supported arch names."""
    from vllm import ModelRegistry

    for arch in _ARCHS:
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, _TARGET)
