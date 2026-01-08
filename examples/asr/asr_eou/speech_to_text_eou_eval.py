# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Example usage:

```bash
NEMO_PATH=/home/heh/codes/nemo-eou
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

TEST_MANIFEST="[/path/to/your/test_manifest.json,/path/to/your/test_manifest2.json,...]"
TEST_NAME="[test_name1,test_name2,...]"
TEST_BATCH=32
NUM_WORKERS=8

PRETRAINED_NEMO=/path/to/EOU/model.nemo
SCRIPT=${NEMO_PATH}/examples/asr/asr_eou/speech_to_text_eou_eval.py
CONFIG_PATH=${NEMO_PATH}/examples/asr/conf/asr_eou
CONFIG_NAME=fastconformer_transducer_bpe_streaming

export CUDA_VISIBLE_DEVICES=0 && \
python $SCRIPT \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++init_from_nemo_model=$PRETRAINED_NEMO \
    ~model.train_ds \
    ~model.validation_ds \
    ++model.test_ds.defer_setup=true \
    ++model.test_ds.sample_rate=16000 \
    ++model.test_ds.manifest_filepath=$TEST_MANIFEST \
    ++model.test_ds.name=$TEST_NAME \
    ++model.test_ds.batch_size=$TEST_BATCH \
    ++model.test_ds.num_workers=$NUM_WORKERS \
    ++model.test_ds.drop_last=false \
    ++model.test_ds.force_finite=true \
    ++model.test_ds.shuffle=false \
    ++model.test_ds.pin_memory=true \
    exp_manager.name=$EXP_NAME-eval \
    exp_manager.create_wandb_logger=false \
```

"""


import lightning.pytorch as pl
import torch

torch.set_float32_matmul_precision("highest")
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core.classes import typecheck
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

typecheck.set_typecheck_enabled(False)


def load_model(cfg: DictConfig, trainer: pl.Trainer) -> ASRModel:
    if "init_from_nemo_model" in cfg:
        logging.info(f"Loading model from local file: {cfg.init_from_nemo_model}")
        model = ASRModel.restore_from(cfg.init_from_nemo_model, trainer=trainer)
    elif "init_from_pretrained_model" in cfg:
        logging.info(f"Loading model from remote: {cfg.init_from_pretrained_model}")
        model = ASRModel.from_pretrained(cfg.init_from_pretrained_model, trainer=trainer)
    else:
        raise ValueError(
            "Please provide either 'init_from_nemo_model' or 'init_from_pretrained_model' in the config file."
        )
    if cfg.get("init_from_ptl_ckpt", None):
        logging.info(f"Loading weights from checkpoint: {cfg.init_from_ptl_ckpt}")
        state_dict = torch.load(cfg.init_from_ptl_ckpt, map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state_dict, strict=True)
    return model


@hydra_runner(config_path="../conf/asr_eou", config_name="fastconformer_transducer_bpe_streaming")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = load_model(cfg, trainer)
    asr_model = asr_model.eval()  # Set the model to evaluation mode
    if hasattr(asr_model, 'wer'):
        asr_model.wer.log_prediction = False

    with open_dict(asr_model.cfg):
        if "save_pred_to_file" in cfg:
            asr_model.cfg.save_pred_to_file = cfg.save_pred_to_file
        if "calclate_eou_metrics" in cfg:
            asr_model.cfg.calclate_eou_metrics = cfg.calclate_eou_metrics
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        with open_dict(cfg.model.test_ds):
            cfg.model.test_ds.pad_eou_label_secs = asr_model.cfg.get('pad_eou_label_secs', 0.0)
        asr_model.setup_multiple_test_data(test_data_config=cfg.model.test_ds)
        trainer.test(asr_model)
    else:
        raise ValueError(
            "No test dataset provided. Please provide a test dataset in the config file under model.test_ds."
        )
    logging.info("Test completed.")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
