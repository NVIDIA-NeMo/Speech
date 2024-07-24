# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from huggingface_hub import HfApi, ModelCard, ModelCardData
from huggingface_hub import get_token as get_hf_token
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import SoftTemporaryDirectory

from nemo.utils import logging


class HuggingFaceFileIO(ABC):
    """
    Mixin that provides Hugging Face file IO functionality for NeMo models.
    It is usually implemented as a mixin to `ModelPT`.

    This mixin provides the following functionality:
    - `search_huggingface_models()`: Search the hub programmatically via some model filter.
    - `push_to_hf_hub()`: Push a model to the hub.
    """

    @classmethod
    def get_hf_model_filter(cls) -> Dict[str, Any]:
        """
        Generates a filter for HuggingFace models.

        Additionaly includes default values of some metadata about results returned by the Hub.

        Metadata:
            resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
            limit_results: Optional int, limits the number of results returned.

        Returns:
            A dict representing the arguments passable to huggingface list_models().
        """
        model_filter = dict(
            author=None,
            library='nemo',
            language=None,
            model_name=None,
            task=None,
            tags=None,
            limit=None,
            full=None,
            cardData=False,
        )

        return model_filter

    @classmethod
    def search_huggingface_models(cls, model_filter: Optional[Dict[str, Any]] = None) -> Iterable['ModelInfo']:
        """
        Should list all pre-trained models available via Hugging Face Hub.

        The following metadata can be passed via the `model_filter` for additional results.
        Metadata:

            resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.

            limit_results: Optional int, limits the number of results returned.

        .. code-block:: python

            # You can replace <DomainSubclass> with any subclass of ModelPT.
            from nemo.core import ModelPT

            # Get default filter dict
            filt = <DomainSubclass>.get_hf_model_filter()

            # Make any modifications to the filter as necessary
            filt['language'] = [...]
            filt['task'] = ...
            filt['tags'] = [...]

            # Add any metadata to the filter as needed (kwargs to list_models)
            filt['limit'] = 5

            # Obtain model info
            model_infos = <DomainSubclass>.search_huggingface_models(model_filter=filt)

            # Browse through cards and select an appropriate one
            card = model_infos[0]

            # Restore model using `modelId` of the card.
            model = ModelPT.from_pretrained(card.modelId)

        Args:
            model_filter: Optional Dictionary (for Hugging Face Hub kwargs)
                that filters the returned list of compatible model cards, and selects all results from each filter.
                Users can then use `model_card.modelId` in `from_pretrained()` to restore a NeMo Model.

        Returns:
            A list of ModelInfo entries.
        """
        # Resolve model filter if not provided as argument
        if model_filter is None:
            model_filter = cls.get_hf_model_filter()

        # Check if api token exists, use if it does
        hf_token = get_hf_token()

        # Search for all valid models after filtering
        api = HfApi()

        # Setup extra arguments for model filtering
        all_results = []  # type: List[ModelInfo]

        results = api.list_models(
            token=hf_token, sort="lastModified", direction=-1, **model_filter
        )  # type: Iterable[ModelInfo]

        return results

    def push_to_hf_hub(
        self,
        repo_id: str,
        *,
        pack_nemo_file: bool = True,
        model_card: Optional['ModelCard'] | object | str = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
    ):
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            pack_nemo_file (`bool`, *optional*, defaults to `True`): Whether to pack the model checkpoint and
                configuration into a single `.nemo` file. If set to false, uploads the contents of the directory
                containing the model checkpoint and configuration plus additional artifacts.
            model_card (`ModelCard`, *optional*): Model card to upload with the model. If None, will use the model
                card template provided by the class itself via `generate_model_card()`. Any object that implements
                str(obj) can be passed here. Two keyword replacements are passed to `generate_model_card()`:
                `model_name` and `repo_id`. If the model card generates a string, and it contains `{model_name}` or
                `{repo_id}`, they will be replaced with the actual values.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.

        Returns:
            The url of the uploaded HF repo.
        """
        if "/" not in repo_id or len(repo_id.split("/")) != 2:
            raise ValueError("Invalid repo_id provided. Please provide a repo_id of the form `username/repo-name`.")

        domain_name, model_name = repo_id.split("/")

        if token is None:
            token = get_hf_token()

        api = HfApi(endpoint=api_endpoint, token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            saved_path.mkdir(parents=True, exist_ok=True)

            # Save nemo file in temp dir
            # Get SaveRestoreConnector from subclass implementation
            if not hasattr(self, '_save_restore_connector'):
                raise NotImplementedError(
                    "Model must implement a `_save_restore_connector` property to push to the HuggingFace Hub."
                )

            # We want to save a NeMo file, but not pack its contents into a tarfile by default
            save_restore_connector = self._save_restore_connector
            save_restore_connector.pack_nemo_file = pack_nemo_file

            nemo_filepath = saved_path / f"{model_name}.nemo"
            self.save_to(nemo_filepath)

            # Save model card in temp dir
            if model_card is None:
                card_model_name = model_name.replace("_", " ").split(" ")
                card_model_name = " ".join([word.capitalize() for word in card_model_name])
                template_kwargs = {
                    'model_name': card_model_name,
                    'repo_id': repo_id,
                }
                # Generate model card from subclass that implements this method
                model_card = self.generate_model_card(type='hf', template_kwargs=template_kwargs)

            # Convert model card to str
            model_card = str(model_card)

            # Write model card to temp dir
            model_card_filepath = saved_path / f"README.md"
            model_card_filepath.write_text(str(model_card), encoding='utf-8', errors='ignore')

            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

            if branch is None:
                branch = "main"

            return f"https://huggingface.co/{repo_id}/tree/{branch}"

    def _get_hf_model_card(self, template: str, template_kwargs: Optional[Dict[str, str]] = None):
        """
        Generate a HuggingFace ModelCard from a str template. The template may have markers with `{key}` that will be
        populated by values from `template_kwargs` if provided.

        Args:
            template: Str template for the model card.
            template_kwargs (optional): Dict of key-value pairs to populate the template with.

        Returns:
            A HuggingFace ModelCard object that can be converted to a model card string.
        """
        card_data = ModelCardData(
            library_name='nemo',
            tags=['pytorch', 'NeMo'],
            license='cc-by-4.0',
            ignore_metadata_errors=True,
        )

        if 'card_data' not in template_kwargs:
            template_kwargs['card_data'] = card_data.to_yaml()

        # Update template with kwargs
        # We need to do a manual replace because not all keys may be provided in the kwargs
        for key, val in template_kwargs.items():
            template = template.replace("{" + key.strip() + "}", val)

        hf_model_card = ModelCard(template)
        return hf_model_card
