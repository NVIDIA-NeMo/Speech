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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm.api import _setup, finetune
from nemo.collections.speechlm.data.audio_to_text_module import AudioToTextDataModule
from nemo.collections.speechlm.models.speech_to_text_llm_model import SpeechToTextLLM, SpeechToTextLLMConfig
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils import SpeechToTextLLMPEFT, get_object_list_from_config
from nemo.collections.speechlm.utils.io import prepare_pretrained_llm_dist_ckpt
from nemo.core.classes.common import Serialization, typecheck
from nemo.lightning import AutoResume
from nemo.lightning.pytorch.callbacks import PreemptionCallback
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, TimingCallback


@dataclass
class PipelineComponents:
    model: Optional[SpeechToTextLLM] = None
    data: Optional[AudioToTextDataModule] = None
    trainer: Optional[nl.Trainer] = None
    optim: Optional[nl.MegatronOptimizerModule] = None
    peft: Optional[SpeechToTextLLMPEFT] = None
    resume: Optional[AutoResume] = None
    logger: Optional[nl.NeMoLogger] = None
    cfg: Optional[dict] = None


def dump_config(cfg: dict, logger: nl.NeMoLogger):
    log_dir = logger.explicit_log_dir if logger.explicit_log_dir else logger.log_dir
    if log_dir is None:
        log_dir = str(Path.cwd() / "nemo_experiments")

    name = logger.name if logger.name else "default"
    output_dir = Path(log_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def build_components(cfg: DictConfig, tokenizer: Optional[AutoTokenizer] = None) -> PipelineComponents:
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 1. build the model
    if tokenizer is not None:
        logging.info(f"Using provided tokenizer: {tokenizer}")
    elif 'pretrained_model' in cfg['model']['llm']:
        logging.info(f"Using tokenizer from pretrained model: {cfg['model']['llm']['pretrained_model']}")
        tokenizer = AutoTokenizer(cfg['model']['llm']['pretrained_model'])
    else:
        raise ValueError(
            "Tokenizer is not provided, please pass `tokenizer` to `speech_to_text_llm_train`",
            "or specify `pretrained_model` in the config.",
        )

    model_config = SpeechToTextLLMConfig(
        language_model_class=cfg['model']['llm']['_target_'],
        language_model_config=Serialization.from_config_dict(cfg['model']['llm']['config']),
        speech_model_config=ASRModuleConfig(**cfg['model']['speech_encoder']),
        modality_adapter_config=ModalityAdapterConfig(**cfg['model']['modality_adapter']),
        language_model_from_pretrained=cfg['model']['llm'].get('pretrained_model', None),
        freeze_language_model=cfg['model']['freeze_language_model'],
        freeze_speech_model=cfg['model']['freeze_speech_model'],
        freeze_modality_adapter=cfg['model']['freeze_modality_adapter'],
        data_config=cfg['data']['common'],
        resume_speech_model_from_path=cfg['model'].get('resume_speech_model_from_path', None),
        resume_modality_adapter_from_path=cfg['model'].get('resume_modality_adapter_from_path', None),
        inference_config=cfg['model'].get('inference_config', None),
    )

    if model_config.language_model_from_pretrained:
        prepare_pretrained_llm_dist_ckpt(model_config)

    model = SpeechToTextLLM(config=model_config, tokenizer=tokenizer)

    # 2. build dataset
    data = AudioToTextDataModule(cfg['data'], tokenizer=tokenizer)

    # 3. setup the optimizer
    optim = Serialization.from_config_dict(cfg['optim'])

    # 4. setup trainer
    callbacks = get_object_list_from_config(cfg['callbacks'])
    if cfg.get('max_time_per_run', None):
        if cfg['strategy'].get('ckpt_async_save', True):
            raise ValueError(
                f"`strategy.ckpt_async_save` must be `False` to save ckpt when `max_time_per_run` is set,",
                f"got {cfg['strategy']['ckpt_async_save']}. `max_time_per_run` will not work in this case!",
            )
        else:
            # ckpt_async_save must be False to save ckpt when training is interrupted by max_time_per_run
            logging.info(f"Setting max_time_per_run={cfg['max_time_per_run']} for the training job.")
            callbacks.append(StatelessTimer(cfg['max_time_per_run']))
    else:
        callbacks.append(PreemptionCallback())
    callbacks.append(TimingCallback())

    trainer = nl.Trainer(
        strategy=Serialization.from_config_dict(cfg['strategy']),
        plugins=get_object_list_from_config(cfg['plugins']),
        callbacks=callbacks,
        **cfg['trainer'],
    )

    # 5. setup PEFT
    peft = None
    if cfg['model'].get('peft', None):
        peft = SpeechToTextLLMPEFT(peft=Serialization.from_config_dict(cfg['model']['peft']))

    # 6. setup logger and auto-resume
    resume = Serialization.from_config_dict(cfg['resume'])
    logger = Serialization.from_config_dict(cfg['logger'])

    return PipelineComponents(
        model=model,
        data=data,
        trainer=trainer,
        optim=optim,
        peft=peft,
        resume=resume,
        logger=logger,
        cfg=cfg,
    )


def speech_to_text_llm_train(cfg: DictConfig, tokenizer: Optional[AutoTokenizer] = None):
    """Train the model using provided config."""

    components = build_components(cfg, tokenizer)

    # save config to log_dir
    dump_config(components.cfg, components.logger)

    finetune(
        model=components.model,
        data=components.data,
        trainer=components.trainer,
        optim=components.optim,
        log=components.logger,
        peft=components.peft,
        resume=components.resume,
    )

    return components.logger.log_dir


def speech_to_text_llm_validate(cfg: DictConfig, tokenizer: Optional[AutoTokenizer] = None):
    """
    Validate the model using provided config, groundtruth required.

    NOTE: Can use dummy groundtruth (e.g., answer='-') for inference
    when speech_to_text_llm_generate is not implemented yet.
    """

    components = build_components(cfg, tokenizer)

    # 7. run the inference
    app_state = _setup(
        model=components.model,
        data=components.data,
        trainer=components.trainer,
        log=components.logger,
        resume=components.resume,
        optim=components.optim,
        tokenizer=tokenizer,
        model_transform=components.peft,
    )

    components.trainer.validate(components.model, datamodule=components.data)

    return app_state.log_dir


def speech_to_text_llm_generate(cfg: DictConfig, tokenizer: Optional[AutoTokenizer] = None):
    """Running inference using provided config without groundtruth."""
    # TODO: implement this based on llm.generate()
    raise NotImplementedError("This function is not implemented yet.")
