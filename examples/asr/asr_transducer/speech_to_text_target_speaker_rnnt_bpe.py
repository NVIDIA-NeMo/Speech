# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune a cache-aware RNN-T for enrollment-conditioned target-speaker ASR.

Start from a cache-aware FastConformer RNN-T config and add the following model
and train-dataset sections (paths are intentionally deployment-specific)::

    model:
      spk_kernel_type: ff
      spk_kernel_layers: [0]
      add_bg_spk_kernel: true
      target_speaker:
        enrollment_dim: 192
        activity_loss_weight: 0.2
        freeze_backbones_until_step: 2000
        sortformer_model_path: /models/diar_streaming_sortformer_4spk-v2.nemo
      # The new conditioning head and kernels use model.optim.lr; pretrained
      # ASR/Sortformer branches use one-tenth of it after warmup.
      optim_param_groups:
        encoder: {lr: 1.0e-4}
        target_sortformer: {lr: 1.0e-4}
      train_ds:
        use_lhotse: true
        background_cuts_path: /data/train_cuts.jsonl.gz
        enrollment_cuts_path: /data/train_cuts.jsonl.gz
        enrollment_embeddings_path: /data/titanet_large_embeddings.pt
        num_background_clips: [1, 2]
        background_snr: [-5, 5]
        absent_target_prob: 0.1
        noise_path: null

``noise_path`` must be null: this dataset owns background mixing so it can
remove background supervisions before audio is constructed.
"""

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecTargetSpeakerRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(
    config_path="../conf/fastconformer/cache_aware_streaming",
    config_name="fastconformer_transducer_bpe_streaming",
)
def main(cfg):
    logging.info(f'Target-speaker Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecTargetSpeakerRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
