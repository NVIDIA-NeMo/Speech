# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import asdict

import lightning.pytorch as pl

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel, configs
from nemo.utils.exp_manager import exp_manager

"""
python speech_to_text_structured.py
"""

# Generate default asr model config
cfg = configs.EncDecCTCModelConfig()

# set global values
cfg.model.repeat = 5
cfg.model.separable = True

# fmt: off
LABELS = [
    " ", "a", "b", "c", "d", "e",
    "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "'",
]
# fmt: on

qn_15x5 = [
    nemo_asr.modules.conv_asr.JasperEncoderConfig(
        filters=256,
        repeat=1,
        kernel=[33],
        stride=[2],
        separable=cfg.model.separable,
        dilation=[1],
        dropout=cfg.model.dropout,
        residual=False,
    ),
    nemo_asr.modules.conv_asr.JasperEncoderConfig(
        filters=256,
        repeat=1,
        kernel=[33],
        stride=[1],
        separable=cfg.model.separable,
        dilation=[1],
        dropout=cfg.model.dropout,
        residual=True,
    ),
    # ... repeat 14 more times
    nemo_asr.modules.conv_asr.JasperEncoderConfig(
        filters=1024,
        repeat=1,
        kernel=[1],
        stride=[1],
        dilation=[1],
        dropout=cfg.model.dropout,
        residual=False,
    ),
]


def main():
    # Update values
    # MODEL UPDATES
    cfg.name = "Mini QuartzNet"
    cfg.model.labels = LABELS

    # train ds
    cfg.model.train_ds.manifest_filepath = "<path to train dataset>"
    cfg.model.train_ds.labels = LABELS
    cfg.model.train_ds.sample_rate = cfg.model.sample_rate

    # validation ds
    cfg.model.validation_ds.manifest_filepath = "<path to test dataset>"
    cfg.model.validation_ds.labels = LABELS
    cfg.model.validation_ds.sample_rate = cfg.model.sample_rate

    # del `test_ds` does not work!
    # Refer - https://stackoverflow.com/questions/58119758/how-to-remove-dataclass-attributes
    # Hydra/OmegaConf dont allow custom .asdict() methods either
    # For now, explicitly set parameters
    cfg.model.test_ds.sample_rate = cfg.model.sample_rate
    cfg.model.test_ds.labels = cfg.model.labels

    # preprocessor
    cfg.model.preprocessor.sample_rate = cfg.model.sample_rate

    # spec aug
    cfg.model.spec_augment.rect_masks = 5
    cfg.model.spec_augment.rect_freq = 50
    cfg.model.spec_augment.rect_time = 120

    # encoder
    cfg.model.encoder.feat_in = cfg.model.preprocessor.features
    cfg.model.encoder.activation = 'relu'
    cfg.model.encoder.jasper = qn_15x5

    # decoder
    cfg.model.decoder.feat_in = qn_15x5[-1].filters
    cfg.model.decoder.num_classes = len(LABELS)
    cfg.model.decoder.vocabulary = LABELS

    # optim
    cfg.model.optim.name = 'novograd'
    cfg.model.optim.lr = 0.01

    # `betas` dont exist inside the base config,
    # so they cannot be added as such!
    # Same for `weight_decay`.
    cfg.model.optim.betas = [0.8, 0.5]
    cfg.model.optim.weight_decay = 0.001

    # sched
    # As parameters such as warmup_steps and warmup_ratio
    # dont exist inside the shell config, these values are not added!
    cfg.model.optim.sched.name = "CosineAnnealing"
    cfg.model.optim.sched.warmup_steps = None
    cfg.model.optim.sched.warmup_ratio = 0.01

    # Trainer config
    cfg.trainer.devices = 1
    cfg.trainer.max_epochs = 5

    # Exp Manager config
    cfg.exp_manager.name = cfg.name

    # Note usage of asdict
    trainer = pl.Trainer(**asdict(cfg.trainer))
    exp_manager(trainer, asdict(cfg.exp_manager))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
