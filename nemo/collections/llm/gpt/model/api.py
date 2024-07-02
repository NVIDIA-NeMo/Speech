import pytorch_lightning as pl

from nemo.collections.llm.gpt.model.gemma import (
    CodeGemmaConfig2B,
    CodeGemmaConfig7B,
    GemmaConfig,
    GemmaConfig2B,
    GemmaConfig7B,
    GemmaModel,
)
from nemo.collections.llm.gpt.model.llama import (
    CodeLlamaConfig7B,
    CodeLlamaConfig13B,
    CodeLlamaConfig34B,
    CodeLlamaConfig70B,
    Llama2Config7B,
    Llama2Config13B,
    Llama2Config70B,
    Llama3Config8B,
    Llama3Config70B,
    LlamaModel,
)
from nemo.collections.llm.gpt.model.mistral import MistralConfig7B, MistralModel
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B, MixtralModel
from nemo.collections.llm.utils import factory


@factory
def mistral() -> pl.LightningModule:
    return MistralModel(MistralConfig7B())


@factory
def mixtral() -> pl.LightningModule:
    return MixtralModel(MixtralConfig8x7B())


@factory
def llama2_7b() -> pl.LightningModule:
    return LlamaModel(Llama2Config7B())


@factory
def llama3_8b() -> pl.LightningModule:
    return LlamaModel(Llama3Config8B())


@factory
def llama2_13b() -> pl.LightningModule:
    return LlamaModel(Llama2Config13B())


@factory
def llama2_70b() -> pl.LightningModule:
    return LlamaModel(Llama2Config70B())


@factory
def llama3_70b() -> pl.LightningModule:
    return LlamaModel(Llama3Config70B())


@factory
def code_llama_7b() -> pl.LightningModule:
    return LlamaModel(CodeLlamaConfig7B())


@factory
def code_llama_13b() -> pl.LightningModule:
    return LlamaModel(CodeLlamaConfig13B())


@factory
def code_llama_34b() -> pl.LightningModule:
    return LlamaModel(CodeLlamaConfig34B())


@factory
def code_llama_70b() -> pl.LightningModule:
    return LlamaModel(CodeLlamaConfig70B())


@factory
def gemma() -> pl.LightningModule:
    return GemmaModel(GemmaConfig())


@factory
def gemma_2b() -> pl.LightningModule:
    return GemmaModel(GemmaConfig2B())


@factory
def gemma_7b() -> pl.LightningModule:
    return GemmaModel(GemmaConfig7B())


@factory
def code_gemma_2b() -> pl.LightningModule:
    return GemmaModel(CodeGemmaConfig2B())


@factory
def code_gemma_7b() -> pl.LightningModule:
    return GemmaModel(CodeGemmaConfig7B())


__all__ = [
    "mistral",
    "mixtral",
    "llama2_7b",
    "llama3_8b",
    "llama2_13b",
    "llama2_70b",
    "llama3_70b",
    "code_llama_7b",
    "code_llama_13b",
    "code_llama_34b",
    "code_llama_70b",
    "gemma",
    "gemma_2b",
    "gemma_7b",
    "code_gemma_2b",
    "code_gemma_7b",
]
