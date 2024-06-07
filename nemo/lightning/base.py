import gc
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed
from pytorch_lightning import Trainer
from torch import nn


DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
NEMO_CACHE_HOME = Path(os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME))
DEFAULT_NEMO_DATASETS_CACHE = NEMO_CACHE_HOME / "datasets"
NEMO_DATASETS_CACHE = Path(os.getenv("NEMO_DATASETS_CACHE", DEFAULT_NEMO_DATASETS_CACHE))
DEFAULT_NEMO_MODELS_CACHE = NEMO_CACHE_HOME / "models"
NEMO_MODELS_CACHE = Path(os.getenv("NEMO_MODELS_CACHE", DEFAULT_NEMO_MODELS_CACHE))


def get_vocab_size(
    config,
    vocab_size: int,
    make_vocab_size_divisible_by: int = 128,
) -> int:
    from nemo.utils import logging

    after = vocab_size
    multiple = make_vocab_size_divisible_by * config.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    logging.info(
        f"Padded vocab_size: {after}, original vocab_size: {vocab_size}, dummy tokens:" f" {after - vocab_size}."
    )

    return after


def teardown(trainer: Trainer, model: Optional[nn.Module] = None) -> None:
    # Destroy torch distributed
    if torch.distributed.is_initialized():
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    trainer._teardown()  # noqa: SLF001
    if model is not None:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj

    gc.collect()
    torch.cuda.empty_cache()


__all__ = ["get_vocab_size", "teardown"]
