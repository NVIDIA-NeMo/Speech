# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python run_speech_intent_slot_train.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_intent_slot/results.html

"""

from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel, SLUIntentSlotBPEModel, SpeechEncDecSelfSupervisedModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="./configs/", config_name="conformer_transformer_large_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = SLUIntentSlotBPEModel(cfg=cfg.model, trainer=trainer)

    # Init encoder from pretrained model
    pretrained_encoder_name = cfg.pretrained_encoder.name
    if pretrained_encoder_name is not None:
        if Path(pretrained_encoder_name).is_file():
            if not pretrained_encoder_name.endswith(".nemo"):
                logging.info(f"Loading encoder from PyTorch Lightning checkpoint: {pretrained_encoder_name}")
                state_dict = torch.load(pretrained_encoder_name, map_location='cpu')['state_dict']
                pretraind_model = None
            else:
                logging.info(f"Loading pretrained encoder from NeMo file: {pretrained_encoder_name}")
                pretraind_model = ASRModel.restore_from(
                    restore_path=pretrained_encoder_name, map_location=torch.device("cpu")
                )
                state_dict = pretraind_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
            del pretraind_model
        else:
            logging.info(f"Loading pretrained encoder from NGC: {pretrained_encoder_name}")
            if pretrained_encoder_name.startswith("ssl_"):
                model_cls = SpeechEncDecSelfSupervisedModel
            elif pretrained_encoder_name.startswith("stt_"):
                model_cls = ASRModel
            else:
                raise ValueError(f"Unknown pretrained encoder: {pretrained_encoder_name}")
            pretraind_model = model_cls.from_pretrained(
                model_name=pretrained_encoder_name, map_location=torch.device("cpu")
            )
            model.encoder.load_state_dict(pretraind_model.encoder.state_dict(), strict=False)
            del pretraind_model
    else:
        logging.info("Not using pretrained encoder.")

    if cfg.pretrained_encoder.freeze:
        logging.info("Freezing encoder...")
        model.encoder.freeze()
    else:
        model.encoder.unfreeze()

    trainer.fit(model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)


if __name__ == '__main__':
    main()
