from nemo.collections.llm.recipes import (
    llama3_8b,
    llama3_8b_16k,
    llama3_8b_64k,
    llama3_70b,
    llama3_70b_16k,
    llama3_70b_64k,
    mistral,
    mixtral_8x3b,
    mixtral_8x3b_16k,
    mixtral_8x3b_64k,
    mixtral_8x7b,
    mixtral_8x7b_16k,
    mixtral_8x7b_64k,
    mixtral_8x22b,
)
from nemo.collections.llm.recipes.log.default import default_log, default_resume
from nemo.collections.llm.recipes.optim import adam

__all__ = [
    "llama3_8b",
    "llama3_8b_16k",
    "llama3_8b_64k",
    "llama3_70b",
    "llama3_70b_16k",
    "llama3_70b_64k",
    "mistral",
    "mixtral_8x3b",
    "mixtral_8x3b_16k",
    "mixtral_8x3b_64k",
    "mixtral_8x7b",
    "mixtral_8x7b_16k",
    "mixtral_8x7b_64k",
    "mixtral_8x22b",
    "adam",
    "default_log",
    "default_resume",
]
