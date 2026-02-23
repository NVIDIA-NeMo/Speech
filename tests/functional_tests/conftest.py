# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Shared fixtures and utilities for per-model functional tests."""

import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataloader import default_collate


def run_training_step(model, batch_tensors, collate_fn=None):
    """Run one real training step via Lightning Trainer.fit().

    This uses the full Lightning training loop (no mocking), calling the
    model's actual training_step(), optimizer, and loss.

    For RNNT/TDT/Hybrid models whose loss uses numba CUDA kernels, callers
    should swap model.loss._loss to RNNTLossPytorch before calling this.

    Args:
        model: A NeMo model (must inherit from LightningModule).
        batch_tensors: Either a tuple of tensors (for TensorDataset) or
            a custom Dataset instance.
        collate_fn: Optional collate function for the DataLoader.
    """
    if isinstance(batch_tensors, tuple):
        dataset = TensorDataset(*batch_tensors)
        dl = DataLoader(dataset, batch_size=len(batch_tensors[0]), collate_fn=collate_fn)
    else:
        # Assume it's already a Dataset
        dl = DataLoader(batch_tensors, batch_size=2, collate_fn=collate_fn)

    original_device = next(model.parameters()).device

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=1,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=1,
    )
    trainer.fit(model, train_dataloaders=dl)

    # Trainer.fit() may move the model; restore it to the original device.
    model.to(original_device)


def swap_rnnt_loss_to_pytorch(model):
    """Replace RNNTLossNumba with RNNTLossPytorch for environments where
    numba CUDA kernels are unavailable.

    This swaps only the backend implementation; the loss function
    (RNNT/TDT transducer loss) remains mathematically identical.
    """
    from nemo.collections.asr.losses.rnnt_pytorch import RNNTLossPytorch

    if hasattr(model, 'loss') and hasattr(model.loss, '_loss'):
        loss_mod = model.loss._loss
        cls_name = type(loss_mod).__name__
        if 'Numba' in cls_name:
            model.loss._loss = RNNTLossPytorch(
                blank=model.loss._blank,
                reduction=model.loss.reduction,
            )


def dict_collate_fn(batch):
    """Collate for AudioCodecModel: returns {"audio": ..., "audio_lens": ...}."""
    tensors = default_collate(batch)
    return {"audio": tensors[0], "audio_lens": tensors[1]}


def ssl_collate_fn(batch):
    """Collate for EncDecDenoiseMaskedTokenPredModel: returns AudioNoiseBatch."""
    from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch

    tensors = default_collate(batch)
    return AudioNoiseBatch(
        audio=tensors[0],
        audio_len=tensors[1],
        noise=tensors[2],
        noise_len=tensors[3],
        noisy_audio=tensors[4],
        noisy_audio_len=tensors[5],
    )


def list_collate_fn(batch):
    """Collate for SortformerEncLabelModel: returns list of tensors."""
    return list(default_collate(batch))


def prompted_collate_fn(batch):
    """Collate for EncDecMultiTaskModel: returns PromptedAudioToTextMiniBatch."""
    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch

    tensors = default_collate(batch)
    return PromptedAudioToTextMiniBatch(
        audio=tensors[0],
        audio_lens=tensors[1],
        transcript=tensors[2],
        transcript_lens=tensors[3],
        prompt=tensors[4],
        prompt_lens=tensors[5],
        prompted_transcript=tensors[6],
        prompted_transcript_lens=tensors[7],
        cuts=None,
    )
