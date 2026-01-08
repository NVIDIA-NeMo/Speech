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

0. Prepare dataset based on  <NeMo Root>/nemo/collections/asr/data/audio_to_eou_label_lhotse.py

1. Add special tokens <EOU> and <EOB> to the tokenizer of pretrained model, by refering to the script
    <NeMo Root>/scripts/asr_end_of_utterance/tokenizers/add_special_tokens_to_sentencepiece.py

2. If pretrained model is HybridRNNTCTCBPEModel, convert it to RNNT using the script
   <NeMo Root>/examples/asr/asr_hybrid_transducer_ctc/helpers/convert_nemo_asr_hybrid_to_ctc.py

3. Run the following command to train the ASR-EOU model:
```bash
#!/bin/bash

NEMO_PATH=/home/heh/codes/nemo-eou
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

TRAIN_MANIFEST=/home/heh/codes/nemo-eou/nemo_experiments/turnGPT_TTS_data/daily_dialogue_test_tts.json
VAL_MANIFEST=/home/heh/codes/nemo-eou/nemo_experiments/turnGPT_TTS_data/daily_dialogue_test_tts.json
NOISE_MANIFEST=/home/heh/codes/nemo-eou/nemo_experiments/noise_manifest.json

PRETRAINED_NEMO=/media/data3/pretrained_models/nemo_asr/stt_en_fastconformer_hybrid_large_streaming_80ms_rnnt.nemo
TOKENIZER_DIR=/media/data3/pretrained_models/nemo_asr/tokenizers/stt_en_fastconformer_hybrid_large_streaming_80ms_eou

BATCH_DURATION=30
NUM_WORKERS=0
LIMIT_TRAIN_BATCHES=100
VAL_CHECK_INTERVAL=100
MAX_STEPS=1000000

EXP_NAME=fastconformer_transducer_bpe_streaming_eou_debug

SCRIPT=${NEMO_PATH}/examples/asr/asr_eou/speech_to_text_hybrid_eou_train.py
CONFIG_PATH=${NEMO_PATH}/examples/asr/conf/fastconformer/cache_aware_streaming
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
    model.train_ds.batch_duration=$BATCH_DURATION \
    model.train_ds.num_workers=$NUM_WORKERS \
    model.validation_ds.batch_duration=$BATCH_DURATION \
    model.validation_ds.num_workers=$NUM_WORKERS \
    ~model.test_ds \
    trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    exp_manager.name=$EXP_NAME
```

"""


from typing import Optional

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.models.asr_eou_models import EncDecHybridRNNTCTCBPEEOUModel, EncDecRNNTBPEEOUModel
from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


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


def init_from_pretrained_nemo(model: EncDecHybridRNNTCTCBPEModel, pretrained_model_path: str, cfg: DictConfig):
    """
    load the pretrained model from a .nemo file, taking into account the joint network
    """
    if pretrained_model_path.endswith('.nemo'):
        pretrained_model = ASRModel.restore_from(restore_path=pretrained_model_path)  # type: EncDecRNNTBPEModel
    else:
        try:
            pretrained_model = ASRModel.from_pretrained(pretrained_model_path)  # type: EncDecRNNTBPEModel
        except Exception as e:
            raise ValueError(f"Could not load pretrained model from {pretrained_model_path}.") from e

    if not isinstance(pretrained_model, (EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel)):
        raise ValueError(
            f"Pretrained model {pretrained_model.__class__} is not EncDecRNNTBPEModel or EncDecHybridRNNTCTCBPEModel."
        )

    # Load encoder state dict into the model
    model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)
    logging.info(f"Encoder weights loaded from {pretrained_model_path}.")

    # Load decoder state dict into the model
    decoder = model.decoder  # type: RNNTDecoder
    pretrained_decoder = pretrained_model.decoder  # type: RNNTDecoder
    if not isinstance(decoder, RNNTDecoder) or not isinstance(pretrained_decoder, RNNTDecoder):
        raise ValueError(
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

    # Load the CTC decoder weights if the model is EncDecHybridRNNTCTCBPEEOUModel
    if not hasattr(model, 'ctc_decoder') or not isinstance(model, EncDecHybridRNNTCTCBPEEOUModel):
        return
    if not hasattr(pretrained_model, 'ctc_decoder') or not isinstance(pretrained_model, EncDecHybridRNNTCTCBPEModel):
        raise ValueError(
            f"CTC decoder {model.ctc_decoder.__class__} is not EncDecHybridRNNTCTCBPEEOUModel or pretrained CTC decoder {pretrained_model.ctc_decoder.__class__} is not EncDecHybridRNNTCTCBPEModel."
        )

    ctc_decoder = model.ctc_decoder  # type: ConvASRDecoder
    pretrained_ctc_decoder = pretrained_model.ctc_decoder  # type: ConvASRDecoder
    assert isinstance(ctc_decoder, ConvASRDecoder), f"CTC decoder {ctc_decoder.__class__} is not ConvASRDecoder."
    assert isinstance(
        pretrained_ctc_decoder, ConvASRDecoder
    ), f"Pretrained CTC decoder {pretrained_ctc_decoder.__class__} is not ConvASRDecoder."

    ctc_decoder_state = ctc_decoder.state_dict()
    pretrained_ctc_decoder_state = pretrained_ctc_decoder.state_dict()

    if ctc_decoder._num_classes == pretrained_ctc_decoder._num_classes:
        logging.info("CTC decoder weights loaded from pretrained model with same shape.")
        ctc_decoder.load_state_dict(pretrained_ctc_decoder_state, strict=True)
        return
    elif ctc_decoder._num_classes != pretrained_ctc_decoder._num_classes + 2:
        raise ValueError(
            f"Size mismatched between pretrained ({pretrained_ctc_decoder._num_classes}+2) and current model ({ctc_decoder._num_classes}), skip loading CTC decoder."
        )

    pretrained_weight = pretrained_ctc_decoder_state['decoder_layers.0.weight']  # shape: [num_classes, hid_dim, 1]
    pretrained_bias = (
        pretrained_ctc_decoder_state['decoder_layers.0.bias']
        if 'decoder_layers.0.bias' in pretrained_ctc_decoder_state
        else None
    )  # shape: [num_classes]

    # Copy the weights and biases from the pretrained model to the new model
    ctc_decoder_state['decoder_layers.0.weight'][:-3, :, :] = pretrained_weight[
        :-1, :, :
    ]  # everything except EOU, EOB and blank
    ctc_decoder_state['decoder_layers.0.weight'][-1, :, :] = pretrained_weight[-1, :, :]  # blank class
    value = None
    if token_init_method == 'min':
        # set the EOU and EOB class to the minimum value of the pretrained model
        value = pretrained_weight.min(dim=0)[0]
    elif token_init_method == 'max':
        # set the EOU and EOB class to the maximum value of the pretrained model
        value = pretrained_weight.max(dim=0)[0]
    elif token_init_method == 'mean':
        # set the EOU and EOB class to the mean value of the pretrained model
        value = pretrained_weight.mean(dim=0)
    elif token_init_method == 'constant':
        value = cfg.model.get('token_init_weight_value', None)
    elif token_init_method:
        raise ValueError(f"Unknown token_init_method: {token_init_method}.")

    if value is not None:
        ctc_decoder_state['decoder_layers.0.weight'][-2, :] = value  # EOB class
        ctc_decoder_state['decoder_layers.0.weight'][-3, :] = value  # EOU class

    if pretrained_bias is not None and 'decoder_layers.0.bias' in ctc_decoder_state:
        ctc_decoder_state['decoder_layers.0.bias'][:-3] = pretrained_bias[:-1]  # everything except EOU, EOB and blank
        ctc_decoder_state['decoder_layers.0.bias'][-1] = pretrained_bias[-1]  # blank class
        value = None
        if token_init_method == 'constant':
            value = cfg.model.get('token_init_bias_value', None)
        elif token_init_method == 'min':
            # set the EOU and EOB class to the minimum value of the pretrained model
            value = pretrained_bias.min()
        elif token_init_method == 'max':
            # set the EOU and EOB class to the maximum value of the pretrained model
            value = pretrained_bias.max()
        elif token_init_method == 'mean':
            # set the EOU and EOB class to the mean value of the pretrained model
            value = pretrained_bias.mean()
        elif token_init_method:
            raise ValueError(f"Unknown token_init_method: {token_init_method}.")
        if value is not None:
            ctc_decoder_state['decoder_layers.0.bias'][-2] = value
            ctc_decoder_state['decoder_layers.0.bias'][-3] = value

    # Load the CTC decoder weights
    model.ctc_decoder.load_state_dict(ctc_decoder_state, strict=True)
    logging.info(f"CTC decoder weights loaded from {pretrained_model_path}.")
    return


@hydra_runner(config_path="../conf/asr_eou", config_name="fastconformer_hybrid_transducer_ctc_bpe_streaming")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecHybridRNNTCTCBPEEOUModel(cfg=cfg.model, trainer=trainer)

    init_from_model = get_pretrained_model_name(cfg)
    if init_from_model:
        init_from_pretrained_nemo(asr_model, init_from_model, cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
