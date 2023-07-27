# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import hashlib
import json
import os
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from transformers import TRANSFORMERS_CACHE

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.modules import BertModule
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import VOCAB_FILE_NAME
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.megatron.megatron_utils import (
    MEGATRON_CONFIG_MAP,
    get_megatron_pretrained_bert_models,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.utils import AppState, logging

__all__ = ['NLPModel']

NEMO_NLP_TMP = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), "nemo_nlp_tmp")

os.makedirs(NEMO_NLP_TMP, exist_ok=True)


class NLPModel(ModelPT, Exportable):
    """Base class for NLP Models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, no_lm_init=False):

        self.hidden_size = None
        self.bert_model = None
        vocab_file = None
        nemo_file = None
        config_dict = None
        config_file = None

        # tokenizer needs to get initialized before the super.__init__()
        # as dataloaders and datasets need it to process the data
        pretrain_model_name = ''
        if cfg.get('language_model') and cfg.language_model.get('pretrained_model_name', ''):
            pretrain_model_name = cfg.language_model.get('pretrained_model_name', '')
        all_pretrained_megatron_bert_models = get_megatron_pretrained_bert_models()

        if cfg.get('tokenizer'):
            # Some models have their own tokenizer setup
            if (
                not hasattr(self, 'tokenizer')
                and cfg.tokenizer.get('tokenizer_name')
                and pretrain_model_name not in all_pretrained_megatron_bert_models
            ):
                self.setup_tokenizer(cfg.tokenizer)
            elif pretrain_model_name in all_pretrained_megatron_bert_models:
                copy_cfg = copy.deepcopy(cfg)
                bert_model = get_lm_model(
                    config_file=config_file,
                    config_dict=config_dict,
                    vocab_file=vocab_file,
                    trainer=trainer,
                    cfg=copy_cfg,
                )
                # set the tokenizer if it is not initialized explicitly
                if (
                    (hasattr(self, 'tokenizer') and self.tokenizer is None) or not hasattr(self, 'tokenizer')
                ) and hasattr(bert_model, 'tokenizer'):
                    self.tokenizer = bert_model.tokenizer
            if (
                cfg.get('tokenizer')
                and hasattr(cfg.get('tokenizer'), 'vocab_file')
                and cfg.get('tokenizer').get('vocab_file')
            ):
                vocab_file = self.register_artifact('tokenizer.vocab_file', cfg.tokenizer.vocab_file)
        super().__init__(cfg, trainer)

        # handles model parallel save and restore logic
        self._save_restore_connector = NLPSaveRestoreConnector()

        if cfg.get('language_model') and not no_lm_init:
            if cfg.get('language_model').get('nemo_file'):
                nemo_file = self.register_artifact('language_model.nemo_file', cfg.language_model.nemo_file)
            if cfg.get('language_model').get('config'):
                config_dict = OmegaConf.to_container(cfg.language_model.config)
            if cfg.get('language_model').get('config_file'):
                config_file = self.register_artifact('language_model.config_file', cfg.language_model.config_file)
            bert_model = get_lm_model(
                config_file=config_file, config_dict=config_dict, vocab_file=vocab_file, trainer=trainer, cfg=cfg,
            )
            # set the tokenizer if it is not initialized explicitly
            if ((hasattr(self, 'tokenizer') and self.tokenizer is None) or not hasattr(self, 'tokenizer')) and hasattr(
                bert_model, 'tokenizer'
            ):
                self.tokenizer = bert_model.tokenizer

            # Required to pull up the config for MegatronBert models
            self.pretrained_model_name = cfg.language_model.pretrained_model_name

            if (
                cfg.tokenizer is not None
                and cfg.tokenizer.get("tokenizer_name", "") is not None
                and "megatron" in cfg.tokenizer.get("tokenizer_name", "")
            ) or pretrain_model_name in all_pretrained_megatron_bert_models:
                self.hidden_size = bert_model.cfg.hidden_size
            else:
                self.hidden_size = bert_model.config.hidden_size

        if cfg.get('language_model') and not no_lm_init:
            self.bert_model = bert_model
            # register encoder config
            self.register_bert_model()

    def register_artifact(
        self, config_path: str, src: str, verify_src_exists: bool = False,
    ):
        """ Overrides ModelPT register_artifact default behavior.
        NLP models usually need artifacts that are optional."""
        return super().register_artifact(config_path, src, verify_src_exists=verify_src_exists)

    @rank_zero_only
    def register_bert_model(self):
        """Adds encoder config to .nemo archive for Jarvis.
        """
        # check if there is an encoder, warn if not
        if self.bert_model is not None:
            # get encoder config and create source for artifact
            if isinstance(self.bert_model, BertModule):
                # HuggingFace Transformer Config
                pretrained_model_name = self.bert_model.name_or_path
                # Some HF names have "/" in them so we replace with _
                pretrained_model_name = pretrained_model_name.replace("/", "_")
                encoder_config_path = pretrained_model_name + '_encoder_config'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path + '.json')
                self.bert_model.config.to_json_file(encoder_config_src)  # name requested by jarvis team
                self.register_artifact('language_model.config_file', encoder_config_src)  # for .nemo
            # MegatronBertModel's superclass is NLP model, hence can't check for isinstance of self.bert_modelel
            elif hasattr(self, 'pretrained_model_name') and 'megatron' in self.pretrained_model_name:
                if self.pretrained_model_name in MEGATRON_CONFIG_MAP:
                    output_config = MEGATRON_CONFIG_MAP[self.pretrained_model_name]["config"]
                    if output_config is not None:
                        encoder_config_path = self.pretrained_model_name + '_encoder_config'
                        encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path + '.json')
                        with open(encoder_config_src, 'w', encoding='utf-8') as f:
                            f.write(json.dumps(output_config, indent=2, sort_keys=True) + '\n')
                        self.register_artifact('language_model.config_file', encoder_config_src)  # for .nemo
                    else:
                        # No defaults as this case can be any possible hyper-parameter combination of MegatronBert config
                        logging.info(f'For {self.pretrained_model_name}, set the config_file in the YAML file')
                else:
                    logging.info(
                        f'Registering MegatronBERT model config for {self.pretrained_model_name} is not yet supported. \
                        Please override this method if needed.'
                    )
            else:
                logging.info(
                    f'Registering BERT model config for {self.bert_model} is not yet supported. Please override this method if needed.'
                )

    def setup_tokenizer(self, cfg: DictConfig):
        """Instantiates tokenizer based on config and registers tokenizer artifacts.

           If model is being restored from .nemo file then the tokenizer.vocab_file will
           be used (if it exists).

           Otherwise, we will use the vocab file provided in the config (if it exists).

           Finally, if no vocab file is given (this happens frequently when using HF),
           we will attempt to extract the vocab from the tokenizer object and then register it.

        Args:
            cfg (DictConfig): Tokenizer config
        """
        vocab_file = None
        if cfg.get('vocab_file'):
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
        # only load tokenizer if vocab_file and tokenizer_model is not None
        if cfg.tokenizer_name or vocab_file or cfg.tokenizer_model:
            self.tokenizer = get_tokenizer(
                tokenizer_name=cfg.tokenizer_name,
                vocab_file=vocab_file,
                special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
                tokenizer_model=self.register_artifact(
                    config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model
                ),
            )

        if vocab_file is None:
            # when there is no vocab file we try to get the vocab from the tokenizer and register it
            self._register_vocab_from_tokenizer(vocab_file_config_path='tokenizer.vocab_file', cfg=cfg)

    @rank_zero_only
    def _register_vocab_from_tokenizer(
        self,
        vocab_file_config_path: str = 'tokenizer.vocab_file',
        vocab_dict_config_path: str = 'tokenizer_vocab_dict',
        cfg: DictConfig = None,
    ):
        """Creates vocab file from tokenizer if vocab file is None.

        Args:
            vocab_file_config_path: path to the vocab_file in the config
            vocab_dict_config_path: path to the vocab_dict in the config
            cfg: tokenizer config
        """
        if self.tokenizer is None:
            raise ValueError('Instantiate self.tokenizer before registering vocab from it.')
        else:
            if isinstance(self.tokenizer, AutoTokenizer):
                # extract vocab from tokenizer
                vocab_dict = self.tokenizer.tokenizer.get_vocab()

                # for fast and slow tokenizer vocabularies compatibility
                vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))

                # get hash of vocab_dict to create a unique directory to write vocab_dict and vocab_file
                m = hashlib.md5()
                if 'tokenizer_name' in cfg:
                    if cfg.tokenizer_name is not None:
                        # different pretrained models with the same vocab will have different hash
                        m.update(cfg.tokenizer_name.encode())
                # get string representation of vocab_dict
                vocab_dict_str = json.dumps(vocab_dict, sort_keys=True).encode()
                m.update(vocab_dict_str)
                vocab_dict_hash = m.hexdigest()

                hash_path = os.path.join(NEMO_NLP_TMP, vocab_dict_hash)
                os.makedirs(hash_path, exist_ok=True)

                vocab_json_src = os.path.join(hash_path, vocab_dict_config_path)

                with open(vocab_json_src, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(vocab_dict, indent=2, sort_keys=True) + '\n')
                self.register_artifact(config_path=vocab_dict_config_path, src=vocab_json_src)

                tokenizer_name = self.tokenizer.tokenizer.__class__.__name__
                # save vocab file
                # depending on the HuggingFace model, vocab file could mean different things, see VOCAB_FILE_NAME
                self.tokenizer.save_vocabulary(hash_path)

                # create vocab file
                vocab_file_src = os.path.join(hash_path, VOCAB_FILE_NAME[tokenizer_name])
                cfg.vocab_file = vocab_file_src
                self.register_artifact(config_path=vocab_file_config_path, src=vocab_file_src)
            else:
                logging.info(
                    f'Registering tokenizer vocab for {self.tokenizer} is not yet supported. Please override this method if needed.'
                )

    @staticmethod
    def _unpack_nemo_file(path2file: str, out_folder: str) -> str:
        return super(NLPModel, NLPModel)._unpack_nemo_file(path2file, out_folder)

    @staticmethod
    def _make_nemo_file_from_folder(filename, source_dir):
        return super(NLPModel, NLPModel)._make_nemo_file_from_folder(filename, source_dir)

    @property
    def input_module(self):
        return self.bert_model

    @property
    def output_module(self):
        return self.classifier

    @property
    def is_model_parallel_initialized(self):
        app_state = AppState()
        if app_state.model_parallel_group is not None:
            return True
        else:
            return False

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Any = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoin() documentation.
        """
        checkpoint = None
        try:
            cls._set_model_restore_state(is_being_restored=True)
            # TODO: replace with proper PTL API
            with pl_legacy_patch():
                if map_location is not None:
                    checkpoint = pl_load(checkpoint_path, map_location=map_location)
                else:
                    checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

            if hparams_file is not None:
                extension = hparams_file.split(".")[-1]
                if extension.lower() == "csv":
                    hparams = load_hparams_from_tags_csv(hparams_file)
                elif extension.lower() in ("yml", "yaml"):
                    hparams = load_hparams_from_yaml(hparams_file)
                else:
                    raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

                hparams["on_gpu"] = False

                # overwrite hparams by the given file
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

            # for past checkpoint need to add the new key
            if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
            # override the hparams with values that were passed in
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].get('cfg', checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])
            # TODO: can we do this without overriding?
            config_kwargs = kwargs.copy()
            if 'trainer' in config_kwargs:
                config_kwargs.pop('trainer')
            cfg.update(config_kwargs)

            if cfg.get('megatron_amp_O2', False):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('model.', 'model.module.', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            if 'cfg' in kwargs:
                model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
            else:
                model = ptl_load_state(cls, checkpoint, strict=strict, cfg=cfg, **kwargs)
                # cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg

            # NMT models do not have a `tokenizer` attribute, they instead have an encoder_tokenizer and decoder_tokenizer attribute.
            if hasattr(cfg, "tokenizer"):
                if cfg.tokenizer.get("tokenizer_model") is not None:
                    model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.tokenizer_model)
                if cfg.tokenizer.get("vocab_file") is not None:
                    model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
                if cfg.tokenizer.get("merge_file") is not None:
                    model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)

            if hasattr(cfg, "encoder_tokenizer"):
                if cfg.encoder_tokenizer.get("tokenizer_model") is not None:
                    model.register_artifact("encoder_tokenizer.tokenizer_model", cfg.encoder_tokenizer.tokenizer_model)
                if cfg.encoder_tokenizer.get("vocab_file") is not None:
                    model.register_artifact("encoder_tokenizer.vocab_file", cfg.encoder_tokenizer.vocab_file)
                if cfg.encoder_tokenizer.get("merge_file") is not None:
                    model.register_artifact("encoder_tokenizer.merge_file", cfg.encoder_tokenizer.merge_file)

            if hasattr(cfg, "decoder_tokenizer"):
                if cfg.decoder_tokenizer.get("tokenizer_model") is not None:
                    model.register_artifact("decoder_tokenizer.tokenizer_model", cfg.decoder_tokenizer.tokenizer_model)
                if cfg.decoder_tokenizer.get("vocab_file") is not None:
                    model.register_artifact("decoder_tokenizer.vocab_file", cfg.decoder_tokenizer.vocab_file)
                if cfg.decoder_tokenizer.get("merge_file") is not None:
                    model.register_artifact("decoder_tokenizer.merge_file", cfg.decoder_tokenizer.merge_file)

            checkpoint = model

        finally:
            cls._set_model_restore_state(is_being_restored=False)
        return checkpoint

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # starting with trasformers v4.31.0, buffer for position_ids is persistent=False
        if (
            self.bert_model is not None
            and "position_ids" not in self.bert_model.embeddings._modules
            and "bert_model.embeddings.position_ids" in state_dict
        ):
            del state_dict["bert_model.embeddings.position_ids"]
        results = super(NLPModel, self).load_state_dict(state_dict, strict=strict)
        return results
