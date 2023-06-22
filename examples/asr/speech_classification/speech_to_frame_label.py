# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
The script trains a model that peforms classification on each frame of the input audio. 
The default config (i.e., marblenet_3x2x64_20ms.yaml) outputs 20ms frames.

## Training
```sh
python speech_to_label.py \
    --config-path=<path to dir of configs e.g. "../conf/marblenet">
    --config-name=<name of config without .yaml e.g. "marblenet_3x2x64_20ms"> \
    model.train_ds.manifest_filepath="<path to train manifest>" \
    model.validation_ds.manifest_filepath=["<path to val manifest>","<path to test manifest>"] \
    trainer.devices=2 \
    trainer.accelerator="gpu" \
    strategy="ddp" \
    trainer.max_epochs=200
```

The input manifest must be a manifest json file, where each line is a Python dictionary. The fields ["audio_filepath", "offset", "duration",  "label"] are required. An example of a manifest file is:
```
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000,  "label": "0 1 0 0 1"}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000,  "label": "0 0 0 1 1 1 1 0 0"}
```
For example, if you have a 1s audio file, you'll need to have 50 frame labels in the manifest entry like "0 0 0 0 1 1 0 1 .... 0 1".
However, shorter label strings are also supported for smaller file sizes. For example, you can prepare the `label` in 40ms frame, and the model will properly repeat the label for each 20ms frame.

"""

import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models.classification_models import EncDecFrameClassificationModel

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/marblenet", config_name="marblenet_3x2x64_20ms")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncDecFrameClassificationModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
