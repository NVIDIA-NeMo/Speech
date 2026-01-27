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
from pathlib import Path
from typing import Any, Dict, Optional, Union

from huggingface_hub import CONFIG_NAME, PyTorchModelHubMixin
from huggingface_hub.hub_mixin import DataclassInstance
from omegaconf import DictConfig, OmegaConf
from transformers.utils import cached_file


class HFHubMixin(
    PyTorchModelHubMixin,
    library_name="NeMo",
    repo_url="https://github.com/NVIDIA/NeMo",
    docs_url="https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit",
):
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """
        Load Pytorch pretrained weights and return the loaded model.
        Wrapper over PyTorchModelHubMixin that auto-handles config in **model_kwargs.
        """
        resolved_config_file = cached_file(
            model_id,
            CONFIG_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved_config_file is None:
            raise RuntimeError(f"Missing {CONFIG_NAME} file for {model_id=}")
        model_kwargs['cfg'] = OmegaConf.to_container(OmegaConf.load(resolved_config_file))
        # The setting below tells the model's __init__ not to load the original pretrained weights
        # for individual children modules.
        # To illustrate: if you trained a new model M using a pretrained ASR and a pretrained LLM,
        # this setting skips loading the original pretrained ASR and LLM weights, and loads the
        # final trained model weights directly.
        model_kwargs['cfg']['pretrained_weights'] = False
        # When loading from HF checkpoint, set init_from_path to the HF checkpoint directory
        # so it can load the model weights from the correct location.
        if 'init_from_path' in model_kwargs['cfg']:
            model_kwargs['cfg']['init_from_path'] = model_id

        return super()._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            **model_kwargs,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        model_card_kwargs: Optional[Dict[str, Any]] = None,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
                If not provided, we will automatically serialize attribute ``model.cfg``.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            model_card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the model card template to customize the model card.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        if config is None:
            config = getattr(self, "cfg")
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(self.cfg)
        return super().save_pretrained(
            save_directory=save_directory,
            config=config,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            model_card_kwargs=model_card_kwargs,
            **push_to_hub_kwargs,
        )
