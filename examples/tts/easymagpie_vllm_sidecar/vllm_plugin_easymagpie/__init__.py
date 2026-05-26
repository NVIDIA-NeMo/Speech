"""vLLM plugin: register ``EasyMagpieSmallMambaV2`` so the LLM API
instantiates our composition-based TTS class when ``config.json`` sets
``architectures: ["EasyMagpieSmallMambaV2"]``.

Loaded by vLLM in both the parent process and each EngineCore subprocess via
the entry-point group ``vllm.general_plugins``.

Uses the lazy ``<module>:<class>`` form so registration only stores a path
string — the model module itself isn't imported until vLLM resolves the
architecture. The class module is structured so its top-level imports are
nemo-free (nemo imports are deferred to method bodies), preventing the
parent's resolve_model_cls from poisoning the spawn-child's CUDA init.
"""


def register() -> None:
    from vllm import ModelRegistry

    arch = "EasyMagpieSmallMambaV2"
    if arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            arch,
            "easymagpie_vllm.easymagpie_smallmamba_v2"
            ":EasyMagpieSmallMambaV2",
        )
