# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

SAFETENSORS_SINGLE_FILE = "model.safetensors"
LLM_BACKBONE_DIR = "llm_backbone"
# Distributed loading defaults to ``strict=False``. Allow small forward-
# compatible gaps, but reject checkpoints that would leave a substantial
# fraction of the model randomly initialized.
_MIN_CHECKPOINT_PARAMETER_COVERAGE = 0.9


class HFHubMixin(
    PyTorchModelHubMixin,
    library_name="NeMo",
    repo_url="https://github.com/NVIDIA-NeMo/Speech",
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
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        trust_remote_code: bool = False,
        **model_kwargs,
    ):
        """
        Load Pytorch pretrained weights and return the loaded model.
        Wrapper over PyTorchModelHubMixin that auto-handles config in **model_kwargs.

        Supports distributed model-parallel loading via ``distributed_setup``:

            >>> strategy = setup_distributed(tp_size=2)
            >>> model = SALM.from_pretrained(
            ...     "nvidia/salm-model", distributed_setup=strategy.distributed_setup
            ... )

        ``trust_remote_code`` is a runtime security decision. It is deliberately
        taken from the caller and overwrites any value stored in the downloaded
        checkpoint config so that a model repository cannot opt itself into
        executing remote code.

        When ``strict=True``, distributed loading rejects missing model state
        and unused checkpoint tensors. The default ``strict=False`` permits
        small compatibility gaps with warnings, while still rejecting
        catastrophic parameter mismatches.
        """
        if not isinstance(trust_remote_code, bool):
            raise TypeError(f"trust_remote_code must be a bool, got {type(trust_remote_code).__name__}")

        distributed_setup = model_kwargs.pop("distributed_setup", None)
        device_mesh = distributed_setup.mesh_context.device_mesh if distributed_setup is not None else None
        torch_dtype = model_kwargs.pop("torch_dtype", None)

        _cached_file_kwargs = dict(
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

        resolved_config_file = cached_file(model_id, CONFIG_NAME, **_cached_file_kwargs)
        if resolved_config_file is None:
            raise RuntimeError(f"Missing {CONFIG_NAME} file for {model_id=}")
        model_kwargs['cfg'] = OmegaConf.to_container(OmegaConf.load(resolved_config_file))
        model_kwargs['cfg']['trust_remote_code'] = trust_remote_code
        _inject_local_artifact_paths(model_kwargs['cfg'], model_id, _cached_file_kwargs)
        # The setting below tells the model's __init__ not to load the original pretrained weights
        # for individual children modules.
        # To illustrate: if you trained a new model M using a pretrained ASR and a pretrained LLM,
        # this setting skips loading the original pretrained ASR and LLM weights, and loads the
        # final trained model weights directly.
        model_kwargs['cfg']['pretrained_weights'] = False

        if device_mesh is None:
            # Non-distributed: for Automodel checkpoints we must build the modules
            # before ``PyTorchModelHubMixin`` applies checkpoint weights.
            if model_kwargs['cfg'].get("use_nemo_automodel", False):
                model_kwargs['cfg']['init_configure_model'] = True
            if torch_dtype is not None:
                model_kwargs['cfg']['torch_dtype'] = (
                    torch_dtype if isinstance(torch_dtype, str) else str(torch_dtype).replace("torch.", "")
                )
            return super()._from_pretrained(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                map_location=map_location,
                strict=strict,
                **model_kwargs,
            )

        # --- Distributed flow ---
        # Delegate to a module-level function so that ``cls(...)`` is not called
        # from a frame that has ``__class__`` in its closure (which our classmethod
        # has due to the ``super()`` call above).  Lightning's
        # ``save_hyperparameters()`` walks the call stack and mistakes such frames
        # for ``__init__`` frames, causing a ``KeyError: 'self'``.
        return _distributed_from_pretrained(
            cls=cls,
            model_id=model_id,
            model_kwargs=model_kwargs,
            torch_dtype=torch_dtype,
            strict=strict,
            distributed_setup=distributed_setup,
            cached_file_kwargs=_cached_file_kwargs,
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
        # Ensure HF-compatible fields are present so vLLM / transformers can identify the model.
        if isinstance(config, dict):
            config = dict(config)
            # Remote-code trust is a runtime choice and must never be persisted
            # in a checkpoint that can be loaded by another user.
            config.pop("trust_remote_code", None)
            config.setdefault("model_type", "nemo_speechlm")
            config.setdefault("architectures", ["NeMoSpeechLMForConditionalGeneration"])
        return super().save_pretrained(
            save_directory=save_directory,
            config=config,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            model_card_kwargs=model_card_kwargs,
            **push_to_hub_kwargs,
        )


def _distributed_from_pretrained(
    cls,
    model_id,
    model_kwargs,
    torch_dtype,
    strict,
    distributed_setup,
    cached_file_kwargs,
):
    """Create a distributed model instance outside of a classmethod frame.

    Lightning's ``save_hyperparameters()`` walks the call stack looking for
    ``__init__`` frames.  Our ``_from_pretrained`` classmethod has ``__class__``
    in its closure (due to a ``super()`` call), which Lightning mistakes for an
    ``__init__`` frame, causing ``KeyError: 'self'``.  By moving the constructor
    call here (a plain module-level function), the problematic frame is avoided.
    """
    model_kwargs['cfg']['init_configure_model'] = False
    if torch_dtype is not None:
        model_kwargs['cfg']['torch_dtype'] = (
            torch_dtype if isinstance(torch_dtype, str) else str(torch_dtype).replace("torch.", "")
        )

    # 1. Create instance (tokenizer only; llm=None, perception=None)
    instance = cls(**model_kwargs)

    # 2. Build parallelized architecture
    instance.configure_model(distributed_setup=distributed_setup)

    # 3. Load weights
    weight_file = cached_file(model_id, SAFETENSORS_SINGLE_FILE, **cached_file_kwargs)
    if weight_file is None:
        raise RuntimeError(f"Missing {SAFETENSORS_SINGLE_FILE} file for {model_id=}")
    _load_state_dict_with_dtensors(instance, str(Path(weight_file).parent), strict=strict)

    return instance


def _load_state_dict_with_dtensors(model, weight_dir, strict=False):
    """Load safetensors weights into a model with DTensor parameters using DCP.

    Uses ``torch.distributed.checkpoint`` with ``_HuggingFaceStorageReader``
    to load weights directly into model parameters in-place.  This mirrors
    the loading path used by ``NeMoAutoModelForCausalLM``. Model names are
    canonicalized before matching, and catastrophic parameter-coverage gaps
    are rejected before any checkpoint data is copied.

    Args:
        model: The model with DTensor parameters (after ``configure_model``).
        weight_dir: Directory containing ``.safetensors`` file(s).
        strict: Whether to reject every missing model tensor or unused
            checkpoint tensor.
    """
    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader

    # DCP is strict by default — it errors on model keys missing from the
    # checkpoint (e.g. positional-encoding buffers computed at init).
    # Read the checkpoint metadata first and keep only matching model state.
    reader = _HuggingFaceStorageReader(path=weight_dir)
    checkpoint_metadata = reader.read_metadata().state_dict_metadata
    state_dict = _checkpoint_state_dict(model, checkpoint_metadata.keys(), strict=strict)

    # DCP + HF storage reader: parses safetensors header for byte offsets,
    # the planner narrows each tensor to the local DTensor shard,
    # and copies directly into model parameter storage.
    # DefaultLoadPlanner rejects checkpoint/model shape mismatches before scheduling reads.
    dcp.load(state_dict, storage_reader=reader)


def _inject_local_artifact_paths(cfg: dict, model_id: str, cached_file_kwargs: dict) -> None:
    """
    Redirect a loaded SpeechLM2 checkpoint config to artifacts saved beside it.

    The root checkpoint directory keeps NeMo's wrapper ``config.json``. When it
    also contains a root tokenizer and ``llm_backbone/config.json``, point
    tokenizer construction to the root directory and LLM config construction to
    ``llm_backbone`` by mutating ``tokenizer_path`` plus ``pretrained_llm`` or
    ``pretrained_lm_name`` in-place.
    """
    resolved_tokenizer_file = cached_file(model_id, "tokenizer_config.json", **cached_file_kwargs)
    if resolved_tokenizer_file is not None and ("pretrained_llm" in cfg or "pretrained_lm_name" in cfg):
        cfg["tokenizer_path"] = str(Path(resolved_tokenizer_file).parent)

    resolved_llm_config_file = cached_file(model_id, f"{LLM_BACKBONE_DIR}/{CONFIG_NAME}", **cached_file_kwargs)
    if resolved_llm_config_file is None:
        return

    llm_backbone_path = str(Path(resolved_llm_config_file).parent)
    if "pretrained_llm" in cfg:
        cfg["pretrained_llm"] = llm_backbone_path
    if "pretrained_lm_name" in cfg:
        cfg["pretrained_lm_name"] = llm_backbone_path


def _checkpoint_state_dict(model, checkpoint_keys, strict=False):
    """Map unique model tensors to stable checkpoint names and validate coverage."""
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    model_state = {}
    state_groups = {}
    parameter_group_ids = set()
    parameter_groups_by_top_level = {}
    persistent_buffer_group_ids = set()

    def add_model_state(raw_name, value):
        name = canonical_parameter_fqn(raw_name)
        if name in model_state and model_state[name] is not value:
            raise RuntimeError(f"Multiple model tensors map to checkpoint key {name!r}")
        model_state[name] = value
        group_id = id(value)
        if group_id not in state_groups:
            state_groups[group_id] = (value, [])
        _, aliases = state_groups[group_id]
        if name not in aliases:
            aliases.append(name)
        return name, group_id

    # Safetensors keeps one state-dict name for tied parameters. Preserve every
    # alias here so whichever name was saved can target the shared tensor.
    for raw_name, value in model.named_parameters(remove_duplicate=False):
        canonical_name, group_id = add_model_state(raw_name, value)
        parameter_group_ids.add(group_id)
        if "." in canonical_name:
            top_level = canonical_name.partition(".")[0]
            parameter_groups_by_top_level.setdefault(top_level, set()).add(group_id)

    # Walk direct module buffers to preserve every alias while distinguishing
    # persistent state from runtime-only buffers without invoking state_dict()
    # and its FSDP2 hooks.
    for module_name, module in model.named_modules(remove_duplicate=False):
        for buffer_name, value in module._buffers.items():
            if value is None:
                continue
            if buffer_name in module._non_persistent_buffers_set:
                continue
            raw_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            _, group_id = add_model_state(raw_name, value)
            persistent_buffer_group_ids.add(group_id)

    checkpoint_keys = set(checkpoint_keys)
    matched_parameters = []
    missing_parameters = []
    missing_persistent_buffers = []
    matched_parameter_group_ids = set()
    matched_checkpoint_keys = set()
    duplicate_tied_alias_groups = []
    state_dict = {}
    for group_id, (value, aliases) in state_groups.items():
        matched_aliases = [name for name in aliases if name in checkpoint_keys]
        if matched_aliases:
            # DCP should copy into each shared tensor only once, even when a
            # checkpoint contains more than one tied alias.
            matched_checkpoint_key = matched_aliases[0]
            matched_checkpoint_keys.update(matched_aliases)
            state_dict[matched_checkpoint_key] = value
            if len(matched_aliases) > 1:
                duplicate_tied_alias_groups.append(matched_aliases)

        if group_id in parameter_group_ids:
            if matched_aliases:
                matched_parameters.append(value)
                matched_parameter_group_ids.add(group_id)
            else:
                missing_parameters.append(aliases[0])
        elif group_id in persistent_buffer_group_ids and not matched_aliases:
            missing_persistent_buffers.append(aliases[0])

    matched_numel = sum(value.numel() for value in matched_parameters)
    model_numel = sum(state_groups[group_id][0].numel() for group_id in parameter_group_ids)
    coverage = matched_numel / model_numel if model_numel else 1.0
    undercovered_top_levels = []
    for name, group_ids in parameter_groups_by_top_level.items():
        component_numel = sum(state_groups[group_id][0].numel() for group_id in group_ids)
        matched_component_numel = sum(
            state_groups[group_id][0].numel() for group_id in group_ids & matched_parameter_group_ids
        )
        component_coverage = matched_component_numel / component_numel if component_numel else 1.0
        if component_coverage < _MIN_CHECKPOINT_PARAMETER_COVERAGE:
            undercovered_top_levels.append(
                f"{name} ({component_coverage:.1%}, {matched_component_numel}/{component_numel})"
            )
    undercovered_top_levels.sort()
    unmatched_checkpoint_keys = checkpoint_keys - matched_checkpoint_keys
    ignored_extra_state_keys = sorted(key for key in unmatched_checkpoint_keys if key.endswith("_extra_state"))
    unexpected_checkpoint_keys = sorted(key for key in unmatched_checkpoint_keys if not key.endswith("_extra_state"))

    diagnostics = []
    if missing_parameters:
        diagnostics.append(
            f"Safetensors checkpoint matched only {coverage:.1%} of model parameter elements "
            f"({matched_numel}/{model_numel}); missing model parameters "
            f"({len(missing_parameters)} groups, first 8): {sorted(missing_parameters)[:8]}"
        )
    if undercovered_top_levels:
        diagnostics.append(
            f"top-level model components below {_MIN_CHECKPOINT_PARAMETER_COVERAGE:.0%} parameter coverage "
            f"(first 8): {undercovered_top_levels[:8]}"
        )
    if missing_persistent_buffers:
        diagnostics.append(
            f"missing persistent model buffers ({len(missing_persistent_buffers)}, first 8): "
            f"{sorted(missing_persistent_buffers)[:8]}"
        )
    if unexpected_checkpoint_keys:
        diagnostics.append(
            f"unused checkpoint tensors ({len(unexpected_checkpoint_keys)}, first 8): "
            f"{unexpected_checkpoint_keys[:8]}"
        )

    advisory_warnings = []
    if ignored_extra_state_keys:
        advisory_warnings.append(
            f"Ignoring checkpoint extra-state tensors ({len(ignored_extra_state_keys)}, first 8): "
            f"{ignored_extra_state_keys[:8]}. These tensors have no direct parameter or buffer load destination."
        )
    if duplicate_tied_alias_groups:
        advisory_warnings.append(
            f"Checkpoint stores multiple aliases for {len(duplicate_tied_alias_groups)} tied model tensors "
            f"(first 8 groups): {duplicate_tied_alias_groups[:8]}. Each shared tensor is loaded once."
        )

    if advisory_warnings or diagnostics:
        from nemo.utils import logging

    if advisory_warnings:
        for warning in advisory_warnings:
            logging.warning(warning)

    diagnostic_message = "; ".join(diagnostics)
    if strict and diagnostics:
        raise RuntimeError(f"Strict checkpoint loading failed: {diagnostic_message}")
    if coverage < _MIN_CHECKPOINT_PARAMETER_COVERAGE or undercovered_top_levels:
        raise RuntimeError(
            f"Refusing to load checkpoint: global or per-top-level-component parameter coverage is below "
            f"{_MIN_CHECKPOINT_PARAMETER_COVERAGE:.0%}: {diagnostic_message}"
        )

    if diagnostics:
        logging.warning(f"{diagnostic_message}. Continuing because strict checkpoint loading is disabled.")

    return state_dict
