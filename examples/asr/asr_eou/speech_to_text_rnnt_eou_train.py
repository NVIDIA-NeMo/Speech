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

1. Prepare dataset based on <NeMo Root>/nemo/collections/asr/data/audio_to_eou_label_lhotse.py
  Specifically, each sample in the jsonl manifest should have the following fields:
  {
    "audio_filepath": "/path/to/audio.wav",
    "text": "The text of the audio."
    "offset": 0.0,  # offset of the audio, in seconds
    "duration": 3.0,  # duration of the audio, in seconds
    "sou_time": 0.2,  # start of utterance time, in seconds
    "eou_time": 1.5,  # end of utterance time, in seconds
  }

2. If using a normal ASR model as initialization:
    -  Add special tokens <EOU> and <EOB> to the tokenizer of pretrained model, by refering to the script
        <NeMo Root>/scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py
    - If pretrained model is HybridRNNTCTCBPEModel, convert it to RNNT using the script
        <NeMo Root>/examples/asr/asr_hybrid_transducer_ctc/helpers/convert_nemo_asr_hybrid_to_ctc.py

3. Run the following command to train the ASR-EOU model:
```bash
#!/bin/bash

TRAIN_MANIFEST=/path/to/train_manifest.json
VAL_MANIFEST=/path/to/val_manifest.json
NOISE_MANIFEST=/path/to/noise_manifest.json

PRETRAINED_NEMO=/path/to/pretrained_model.nemo
TOKENIZER_DIR=/path/to/tokenizer_dir

BATCH_SIZE=16
NUM_WORKERS=8
LIMIT_TRAIN_BATCHES=1000
VAL_CHECK_INTERVAL=1000
MAX_STEPS=1000000

EXP_NAME=fastconformer_transducer_bpe_streaming_eou
SCRIPT=${NEMO_PATH}/examples/asr/asr_eou/speech_to_text_rnnt_eou_train.py
CONFIG_PATH=${NEMO_PATH}/examples/asr/conf/asr_eou
CONFIG_NAME=fastconformer_transducer_bpe_streaming

CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++init_from_nemo_model=$PRETRAINED_NEMO \
    model.encoder.att_context_size="[70,1]" \
    model.tokenizer.dir=$TOKENIZER_DIR \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.train_ds.augmentor.noise.manifest_path=$NOISE_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.train_ds.num_workers=$NUM_WORKERS \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.num_workers=$NUM_WORKERS \
    ~model.test_ds \
    trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    exp_manager.name=$EXP_NAME
```

"""

from dataclasses import is_dataclass
from typing import Optional

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.models.asr_eou_models import EncDecRNNTBPEEOUModel
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def add_global_adapter_cfg(model, global_adapter_cfg):
    # Convert to DictConfig from dict or Dataclass
    if is_dataclass(global_adapter_cfg):
        global_adapter_cfg = OmegaConf.structured(global_adapter_cfg)

    if not isinstance(global_adapter_cfg, DictConfig):
        global_adapter_cfg = DictConfig(global_adapter_cfg)

    # Update the model.cfg with information about the new adapter global cfg
    with open_dict(global_adapter_cfg), open_dict(model.cfg):
        if 'adapters' not in model.cfg:
            model.cfg.adapters = OmegaConf.create({})

        # Add the global config for adapters to the model's internal config
        model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg

        # Update all adapter modules (that already exist) with this global adapter config
        model.update_adapter_cfg(model.cfg.adapters)


def update_model_config_to_support_adapter(model_cfg):
    with open_dict(model_cfg):
        # Update encoder adapter compatible config
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


def setup_adapters(cfg: DictConfig, model: ASRModel):
    # Setup adapters
    with open_dict(cfg.model.adapter):
        # Extract the name of the adapter (must be give for training)
        adapter_name = cfg.model.adapter.pop("adapter_name")
        adapter_type = cfg.model.adapter.pop("adapter_type")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)

        # Resolve the config of the specified `adapter_type`
        if adapter_type not in cfg.model.adapter.keys():
            raise ValueError(
                f"Adapter type ({adapter_type}) config could not be found. Adapter setup config - \n"
                f"{OmegaConf.to_yaml(cfg.model.adapter)}"
            )

        adapter_type_cfg = cfg.model.adapter[adapter_type]
        print(f"Found `{adapter_type}` config :\n" f"{OmegaConf.to_yaml(adapter_type_cfg)}")

        # Augment adapter name with module name, if not provided by user
        if adapter_module_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_module_name}:{adapter_name}'

        # Extract the global adapter config, if provided
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
        if adapter_global_cfg is not None:
            add_global_adapter_cfg(model, adapter_global_cfg)

    model.add_adapter(adapter_name, cfg=adapter_type_cfg)
    assert model.is_adapter_available()

    # Disable all other adapters, enable just the current adapter.
    model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training
    model.set_enabled_adapters(adapter_name, enabled=True)  # enable just one adapter by name

    model.freeze()  # freeze whole model by default
    if not cfg.model.get("freeze_decoder", True):
        logging.info("Unfreezing decoder weights.")
        model.decoder.unfreeze()
    if hasattr(model, 'joint') and not cfg.model.get(f"freeze_joint", True):
        logging.info("Unfreezing joint network weights.")
        model.joint.unfreeze()

    # Activate dropout() and other modules that depend on train mode.
    model = model.train()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of encoder/decoder/joint/etc)
    model.unfreeze_enabled_adapters()
    return model


def get_pretrained_model_name(cfg: DictConfig) -> Optional[str]:
    if hasattr(cfg, 'init_from_ptl_ckpt') and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError(
            "Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`"
        )
    nemo_model_path = cfg.get('init_from_nemo_model', None)
    pretrained_name = cfg.get('init_from_pretrained_model', None)
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
    elif nemo_model_path is None and pretrained_name is None:
        return None

    if nemo_model_path:
        return nemo_model_path
    if pretrained_name:
        return pretrained_name
    return None


def init_from_pretrained_nemo(model: EncDecRNNTBPEEOUModel, pretrained_model_path: str, cfg: DictConfig):
    """
    Load the pretrained model from a .nemo file or remote checkpoint. If the pretrained model has exactly
    the same vocabulary size as the current model, the whole model will be loaded directly. Otherwise,
    the encoder and decoder weights will be loaded separately and the EOU/EOB classes will be handled separately.
    """
    if pretrained_model_path.endswith('.nemo'):
        pretrained_model = ASRModel.restore_from(restore_path=pretrained_model_path)  # type: EncDecRNNTBPEModel
    else:
        pretrained_model = ASRModel.from_pretrained(pretrained_model_path)  # type: EncDecRNNTBPEModel

    if not isinstance(pretrained_model, (EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel)):
        raise TypeError(
            f"Pretrained model {pretrained_model.__class__} is not EncDecRNNTBPEModel or EncDecHybridRNNTCTCBPEModel."
        )

    try:
        model.load_state_dict(pretrained_model.state_dict(), strict=True)
        logging.info(
            f"Pretrained model from {pretrained_model_path} has exactly the same model structure, skip further loading."
        )
        return
    except Exception:
        logging.warning(
            f"Pretrained model {pretrained_model_path} has different model structure, try loading weights separately and add EOU/EOB classes."
        )

    # Load encoder state dict into the model
    model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)
    logging.info(f"Encoder weights loaded from {pretrained_model_path}.")

    # Load decoder state dict into the model
    decoder = model.decoder  # type: RNNTDecoder
    pretrained_decoder = pretrained_model.decoder  # type: RNNTDecoder
    if not isinstance(decoder, RNNTDecoder) or not isinstance(pretrained_decoder, RNNTDecoder):
        raise TypeError(
            f"Decoder {decoder.__class__} is not RNNTDecoder or pretrained decoder {pretrained_decoder.__class__} is not RNNTDecoder."
        )

    decoder.prediction["dec_rnn"].load_state_dict(pretrained_decoder.prediction["dec_rnn"].state_dict(), strict=True)

    decoder_embed_states = decoder.prediction["embed"].state_dict()['weight']  # shape: [num_classes+2, hid_dim]
    pretrained_decoder_embed_states = pretrained_decoder.prediction["embed"].state_dict()[
        'weight'
    ]  # shape: [num_classes, hid_dim]
    if decoder_embed_states.shape[0] != pretrained_decoder_embed_states.shape[0] + 2:
        raise ValueError(
            f"Size mismatched between pretrained ({pretrained_decoder_embed_states.shape[0]}+2) and current model ({decoder_embed_states.shape[0]}), skip loading decoder embedding."
        )

    decoder_embed_states[:-3, :] = pretrained_decoder_embed_states[:-1, :]  # everything except EOU, EOB and blank
    decoder_embed_states[-1, :] = pretrained_decoder_embed_states[-1, :]  # blank class
    decoder.prediction["embed"].load_state_dict({"weight": decoder_embed_states}, strict=True)
    logging.info(f"Decoder weights loaded from {pretrained_model_path}.")

    # Load joint network weights if new model's joint network has two more classes than the pretrained model
    joint_network = model.joint  # type: RNNTJoint
    pretrained_joint_network = pretrained_model.joint  # type: RNNTJoint
    assert isinstance(joint_network, RNNTJoint), f"Joint network {joint_network.__class__} is not RNNTJoint."
    assert isinstance(
        pretrained_joint_network, RNNTJoint
    ), f"Pretrained joint network {pretrained_joint_network.__class__} is not RNNTJoint."
    joint_network.pred.load_state_dict(pretrained_joint_network.pred.state_dict(), strict=True)
    joint_network.enc.load_state_dict(pretrained_joint_network.enc.state_dict(), strict=True)

    if joint_network.num_classes_with_blank != pretrained_joint_network.num_classes_with_blank + 2:
        raise ValueError(
            f"Size mismatched between pretrained ({pretrained_joint_network.num_classes_with_blank}+2) and current model ({joint_network.num_classes_with_blank}), skip loading joint network."
        )

    # Load the joint network weights
    pretrained_joint_state = pretrained_joint_network.joint_net.state_dict()
    joint_state = joint_network.joint_net.state_dict()
    pretrained_joint_clf_weight = pretrained_joint_state['2.weight']  # shape: [num_classes, hid_dim]
    pretrained_joint_clf_bias = pretrained_joint_state['2.bias'] if '2.bias' in pretrained_joint_state else None

    token_init_method = cfg.model.get('token_init_method', 'constant')
    # Copy the weights and biases from the pretrained model to the new model
    # shape: [num_classes+2, hid_dim]
    joint_state['2.weight'][:-3, :] = pretrained_joint_clf_weight[:-1, :]  # everything except EOU, EOB and blank
    joint_state['2.weight'][-1, :] = pretrained_joint_clf_weight[-1, :]  # blank class

    value = None
    if token_init_method == 'min':
        # set the EOU and EOB class to the minimum value of the pretrained model
        value = pretrained_joint_clf_weight.min(dim=0)[0]
    elif token_init_method == 'max':
        # set the EOU and EOB class to the maximum value of the pretrained model
        value = pretrained_joint_clf_weight.max(dim=0)[0]
    elif token_init_method == 'mean':
        # set the EOU and EOB class to the mean value of the pretrained model
        value = pretrained_joint_clf_weight.mean(dim=0)
    elif token_init_method == 'constant':
        value = cfg.model.get('token_init_weight_value', None)
    elif token_init_method:
        raise ValueError(f"Unknown token_init_method: {token_init_method}.")

    if value is not None:
        joint_state['2.weight'][-2, :] = value  # EOB class
        joint_state['2.weight'][-3, :] = value  # EOU class

    if pretrained_joint_clf_bias is not None and '2.bias' in joint_state:
        joint_state['2.bias'][:-3] = pretrained_joint_clf_bias[:-1]  # everything except EOU, EOB and blank
        joint_state['2.bias'][-1] = pretrained_joint_clf_bias[-1]  # blank class
        value = None
        if token_init_method == 'constant':
            value = cfg.model.get('token_init_bias_value', None)
        elif token_init_method == 'min':
            # set the EOU and EOB class to the minimum value of the pretrained model
            value = pretrained_joint_clf_bias.min()
        elif token_init_method == 'max':
            # set the EOU and EOB class to the maximum value of the pretrained model
            value = pretrained_joint_clf_bias.max()
        elif token_init_method == 'mean':
            # set the EOU and EOB class to the mean value of the pretrained model
            value = pretrained_joint_clf_bias.mean()
        elif token_init_method:
            raise ValueError(f"Unknown token_init_method: {token_init_method}.")

        if value is not None:
            joint_state['2.bias'][-2] = value  # EOB class
            joint_state['2.bias'][-3] = value  # EOU class

    # Load the joint network weights
    joint_network.joint_net.load_state_dict(joint_state, strict=True)
    logging.info(f"Joint network weights loaded from {pretrained_model_path}.")


@hydra_runner(config_path="../conf/asr_eou", config_name="fastconformer_transducer_bpe_streaming")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.model.get("adapter", None) is not None:
        update_model_config_to_support_adapter(cfg.model)

    asr_model = EncDecRNNTBPEEOUModel(cfg=cfg.model, trainer=trainer)

    init_from_model = get_pretrained_model_name(cfg)
    if init_from_model:
        init_from_pretrained_nemo(asr_model, init_from_model, cfg)

    if cfg.model.get("freeze_encoder", False):
        logging.info("Freezing encoder weights.")
        asr_model.encoder.freeze()

    if cfg.model.get("adapter", None) is not None:
        asr_model = setup_adapters(cfg, asr_model)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
