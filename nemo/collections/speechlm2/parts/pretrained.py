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
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf, open_dict
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging
from nemo.utils.compat import python313_pathlib_pickle_compat


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        # Local .nemo restore_from() doesn't resolve the config's `target` (instantiates
        # the abstract base). Resolve the concrete class first, like from_pretrained().
        cfg = cls.restore_from(model_path_or_name, return_config=True)
        target = cfg.get("target", None) if hasattr(cfg, "get") else None
        if target is not None:
            from nemo.core.classes.common import _get_allowed_target_class

            resolved_cls = _get_allowed_target_class(target)
            concrete_cls = resolved_cls
            while hasattr(concrete_cls, "__wrapped__"):
                concrete_cls = concrete_cls.__wrapped__
            if not isinstance(concrete_cls, type) or not issubclass(concrete_cls, cls):
                raise TypeError(f"Checkpoint target {target!r} is not a subclass of {cls.__name__}.")
            cls = resolved_cls
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_nemo_config(cls, model_path_or_name: str):
    """Load a NeMo model config without loading model weights."""
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name, return_config=True)
    return cls.from_pretrained(model_path_or_name, return_config=True)


def load_pretrained_hf(
    model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32, trust_remote_code: bool = False
):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Args:
        model_path_or_name: Path or name of the model to load
        pretrained_weights: Whether to load pretrained weights (True) or random init (False)
        dtype: Data type for the model
        trust_remote_code: Whether to trust remote code when loading model (needed for some models like Nemotron)
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)


def load_pretrained_automodel_llm(
    model_path_or_name: str,
    pretrained_weights: bool = True,
    dtype=torch.float32,
    trust_remote_code: bool = False,
    mtp_config_overrides: dict | None = None,
    replace_mtp_config: bool = False,
    **kwargs,
):
    """
    Load a causal LM using NeMo Automodel (``NeMoAutoModelForCausalLM``).

    Automodel is a drop-in HuggingFace replacement that provides Liger kernel +
    SDPA attention optimizations and model-type-aware parallelization.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture
    with the checkpoint, but is randomly initialized.

    The checkpoint config is inspected when ``mtp_config_overrides`` is provided
    or repeated-layer MTP is requested. Overrides are applied only when the
    checkpoint has no enabled MTP head, or unconditionally when
    ``replace_mtp_config=True``. A preserved native head must have physical depth
    one when repeated-layer MTP is requested. When overrides apply, the resulting
    model is constructed through ``from_config`` before its base checkpoint is
    loaded, so Automodel applies its normal initialization and parallelization to
    a new MTP head. All checkpoint ``mtp.*`` tensors are excluded from that base
    load so the configured head keeps its fresh initialization. Extra ``kwargs``
    (including ``distributed_setup``) are forwarded so parallelization happens
    during construction.
    """
    if replace_mtp_config and mtp_config_overrides is None:
        raise ValueError("replace_mtp_config=True requires mtp_config_overrides to define the replacement head.")

    from nemo_automodel import NeMoAutoModelForCausalLM

    from nemo.collections.speechlm2.parts.automodel_compat import remove_automodel_backend_for_hf_fallback

    remove_automodel_backend_for_hf_fallback(
        model_path_or_name,
        kwargs,
        trust_remote_code=trust_remote_code,
    )

    use_repeated_mtp = bool(kwargs.get("mtp_use_repeated_layer", False))
    if mtp_config_overrides is not None or use_repeated_mtp:
        config_kwargs, automodel_kwargs = _split_automodel_hf_resolution_kwargs(kwargs)
        if pretrained_weights:
            checkpoint_path = _resolve_automodel_checkpoint_path(model_path_or_name, config_kwargs)
        else:
            checkpoint_path = _resolve_automodel_checkpoint_path(
                model_path_or_name, config_kwargs, include_weights=False
            )
        config = AutoConfig.from_pretrained(
            checkpoint_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        checkpoint_physical_depth = _automodel_config_mtp_depth(config)
        checkpoint_has_mtp = checkpoint_physical_depth > 0
        if use_repeated_mtp and not replace_mtp_config:
            if not checkpoint_has_mtp and mtp_config_overrides is None:
                raise ValueError(
                    "MTP use_repeated_layer=True requires either a checkpoint with a native MTP head or "
                    "mtp_config_overrides that define a new head."
                )
            if checkpoint_has_mtp and checkpoint_physical_depth != 1:
                raise ValueError(
                    "MTP use_repeated_layer=True can preserve only a checkpoint with one physical MTP depth; "
                    f"checkpoint declares {checkpoint_physical_depth}. Set use_repeated_layer=false to preserve "
                    "the native independent heads, or set replace_existing_head=true to initialize a fresh "
                    "repeated head."
                )
        initialize_fresh_mtp = replace_mtp_config or not checkpoint_has_mtp
        if not initialize_fresh_mtp and pretrained_weights:
            return NeMoAutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **automodel_kwargs,
            )
        if initialize_fresh_mtp:
            for name, value in mtp_config_overrides.items():
                setattr(config, name, value)
        model = NeMoAutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype,
            load_base_model=False,
            trust_remote_code=trust_remote_code,
            **automodel_kwargs,
        )
        if pretrained_weights and initialize_fresh_mtp:
            _load_automodel_base_checkpoint_without_mtp(model, checkpoint_path, automodel_kwargs)
        return model

    if pretrained_weights:
        return NeMoAutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return NeMoAutoModelForCausalLM.from_config(config, torch_dtype=dtype, **kwargs)


def _load_automodel_base_checkpoint_without_mtp(model, model_path_or_name: str, automodel_kwargs: dict) -> None:
    """Load an Automodel base checkpoint without overwriting a freshly configured MTP head."""
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

    distributed_setup = automodel_kwargs.get("distributed_setup")
    mesh_context = getattr(distributed_setup, "mesh_context", None)
    cache_dir = automodel_kwargs.get("cache_dir")
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir="",
            model_save_format="safetensors",
            model_cache_dir=cache_dir,
            model_repo_id=model_path_or_name,
            save_consolidated=False,
            is_peft=automodel_kwargs.get("peft_config") is not None,
            skip_task_head_prefixes_for_base_model=["mtp."],
        ),
        0,
        0,
        0,
        getattr(mesh_context, "moe_mesh", None),
        process_group=getattr(mesh_context, "process_group", None),
    )

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    with _exclude_mtp_checkpoint_state(model):
        checkpointer.load_base_model(model, device, cache_dir, model_path_or_name, load_base_model=True)


def update_perception_output_dim(model):
    """
    Align the perception module's output projection with the actual LLM hidden size.

    When the LLM is loaded after the perception module (deferred init in
    ``configure_model``), the projection layer may have been created with an
    ``output_dim`` from the YAML config that doesn't match the LLM.  This
    helper replaces ``perception.proj`` with a correctly-sized ``nn.Linear``
    when the dimensions disagree.
    """
    hidden_size = model.llm.config.hidden_size
    proj = model.perception.proj
    if isinstance(proj, torch.nn.Linear) and proj.out_features != hidden_size:
        model.perception.proj = torch.nn.Linear(proj.in_features, hidden_size, bias=proj.bias is not None)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.

    If user config specifies encoder parameters, they will override the pretrained model's config.
    """
    from nemo.collections.speechlm2.modules.perception import MultiLayerProjectionConnector, QformerConnector

    # Save user-specified encoder config before filling missing architecture fields.
    user_encoder_config = {}
    if "encoder" in model.cfg.perception:
        user_encoder_config = OmegaConf.to_container(model.cfg.perception.encoder, resolve=True)

    # Training configs normally omit these fields and get them from the ASR model.
    # Do the same for architecture-only initialization, without loading ASR weights.
    needs_asr_config = pretrained_weights or any(
        key not in model.cfg.perception for key in ("preprocessor", "encoder")
    )
    asr = None
    if needs_asr_config:
        asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval() if pretrained_weights else None
        asr_cfg = asr.cfg if asr is not None else load_pretrained_nemo_config(ASRModel, model.cfg.pretrained_asr)

        with open_dict(model.cfg):
            if pretrained_weights or "preprocessor" not in model.cfg.perception:
                model.cfg.perception.preprocessor = asr_cfg.preprocessor
            if pretrained_weights or "encoder" not in model.cfg.perception:
                model.cfg.perception.encoder = asr_cfg.encoder
            if model.llm is not None:
                hidden_size = model.llm.config.hidden_size
                model.cfg.perception.output_dim = hidden_size
                # Connectors like MultiLayerProjectionConnector carry their own
                # output projection via ``modality_adapter.output_dim``; keep it
                # in sync with the LLM so the inner Linear matches.
                adapter_cfg = model.cfg.perception.get("modality_adapter", None)
                if adapter_cfg is not None and "output_dim" in adapter_cfg:
                    adapter_cfg.output_dim = hidden_size
            # Override user-specified encoder parameters, e.g. for causal setup.
            if user_encoder_config:
                for key, value in user_encoder_config.items():
                    if value is not None:  # Only override explicitly set values.
                        model.cfg.perception.encoder[key] = value

    model.perception = AudioPerceptionModule(model.cfg.perception).train()
    if asr is not None:
        asr_sd = asr.state_dict()
        # When a multilayer/Qformer connector is used, the encoder lives at
        # ``encoder_multilayer.encoder.*`` rather than ``encoder.*``; remap ASR
        # state-dict keys so pretrained encoder weights actually load.
        if isinstance(model.perception.modality_adapter, (QformerConnector, MultiLayerProjectionConnector)):
            asr_sd = {("encoder_multilayer." + k if k.startswith("encoder.") else k): v for k, v in asr_sd.items()}
        model.perception.load_state_dict(asr_sd, strict=False)

    if model.cfg.get("pe_encoder_path", None) not in (None, "", False):
        setup_parallel_expert_encoder(model)


def setup_parallel_expert_encoder(model: torch.nn.Module):
    """Mount a ParallelExpertEncoder bundle from ``model.pe_encoder_path``.

    This is an encoder replacement, not a training-checkpoint restore. It keeps
    the existing SALM perception path intact:

        preprocessor -> ParallelExpertEncoder -> modality_adapter -> proj

    The PE encoder expects un-normalised mels for its Sortformer branch and
    replays ASR normalisation internally, so the outer perception preprocessor
    normalisation is disabled when the bundle is mounted.
    """
    pe_encoder_path = model.cfg.get("pe_encoder_path", None)
    if pe_encoder_path in (None, "", False):
        return

    if not (hasattr(model, "perception") and model.perception is not None):
        raise RuntimeError(
            f"model.pe_encoder_path='{pe_encoder_path}' is set but the model has no "
            "`perception` module to mount it onto. Call setup_speech_encoder() first."
        )
    if not isinstance(pe_encoder_path, str) or not pe_encoder_path:
        raise ValueError(
            "model.pe_encoder_path must be a local ParallelExpertEncoderPT .nemo bundle path or a "
            f"pretrained model id (HuggingFace '{{repo}}/{{name}}' or NGC alias), got {pe_encoder_path!r}."
        )
    if not hasattr(model.perception, "encoder"):
        raise RuntimeError(
            "model.pe_encoder_path requires a direct `model.perception.encoder` to replace. "
            "Adapters that wrap the encoder at construction time (for example multi-layer "
            "feature extractors) need a separate implementation."
        )

    # Fail fast when a local .nemo is given but isn't a PE bundle. HuggingFace Hub /
    # NGC ids are resolved + validated by ParallelExpertEncoderPT.load_from_nemo
    # (from_pretrained -> restore_from, which checks the bundle target class).
    if pe_encoder_path.endswith(".nemo") and Path(pe_encoder_path).is_file():
        if not ParallelExpertEncoderPT.is_pe_nemo(pe_encoder_path):
            raise ValueError(
                f"model.pe_encoder_path={pe_encoder_path!r} is not a ParallelExpertEncoderPT .nemo bundle."
            )

    pe_encoder = ParallelExpertEncoderPT.load_from_nemo(
        pe_encoder_path,
        map_location="cpu",
        strict=True,
    )

    existing_encoder = model.perception.encoder
    existing_d_model = int(getattr(existing_encoder, "d_model", -1))
    if existing_d_model > 0 and int(pe_encoder.d_model) != existing_d_model:
        raise ValueError(
            f"ParallelExpertEncoder d_model={pe_encoder.d_model} does not match the "
            f"existing perception encoder d_model={existing_d_model}. Re-export the "
            "PE bundle with a matching ASR encoder or use a matching perception config."
        )

    adapter_cfg = model.cfg.get("perception", {}).get("modality_adapter", {})
    adapter_d_model = adapter_cfg.get("d_model", None)
    if adapter_d_model is not None and int(adapter_d_model) != int(pe_encoder.d_model):
        raise ValueError(
            f"ParallelExpertEncoder d_model={pe_encoder.d_model} does not match "
            f"model.perception.modality_adapter.d_model={adapter_d_model}."
        )

    proj = getattr(model.perception, "proj", None)
    if isinstance(proj, torch.nn.Linear) and int(proj.in_features) != int(pe_encoder.d_model):
        raise ValueError(
            f"ParallelExpertEncoder d_model={pe_encoder.d_model} does not match "
            f"model.perception.proj.in_features={proj.in_features}."
        )

    prev_normalize = None
    try:
        prev_normalize = model.perception.preprocessor.featurizer.normalize
        model.perception.preprocessor.featurizer.normalize = None
    except AttributeError:
        logging.warning(
            "Could not disable perception preprocessor featurizer.normalize while mounting "
            "ParallelExpertEncoder from %s.",
            pe_encoder_path,
        )
    try:
        with open_dict(model.cfg):
            if "perception" in model.cfg and "preprocessor" in model.cfg.perception:
                model.cfg.perception.preprocessor.normalize = None
    except (AttributeError, TypeError):
        # cfg may lack this nested key or be a non-editable structure; the runtime
        # featurizer.normalize disabling above is the functional change that matters.
        pass

    model.perception.encoder = pe_encoder
    logging.info(
        "Mounted ParallelExpertEncoder from %s onto model.perception.encoder "
        "(d_model=%d, n_spk=%d, freeze_diar=%s, freeze_asr=%s); "
        "perception preprocessor normalization disabled (was %r).",
        pe_encoder_path,
        int(pe_encoder.d_model),
        int(pe_encoder.n_spk),
        bool(pe_encoder.freeze_diar),
        bool(pe_encoder.freeze_asr),
        prev_normalize,
    )


def set_model_dict_for_partial_init(
    pretrained_dict: Dict[str, torch.Tensor],
    model_dict: Dict[str, torch.Tensor],
    allow_partial_copy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Partially initialize a model's state dictionary with a pretrained state dictionary.

    This function safely copies compatible layers from a pretrained model into a new model,
    ignoring layers with missing keys or incompatible shapes.

    By default, only tensors with exactly matching shapes are restored.

    If ``allow_partial_copy=True``, tensors whose shapes differ only in the first
    dimension are partially restored by copying the overlapping rows from the
    pretrained tensor into the target tensor. The remaining rows keep their
    model-initialized values. This is useful when adding new vocabulary rows or
    special tokens, e.g. adding an interruption token to an embedding table.

    Args:
        pretrained_dict:
            State dictionary from the pretrained checkpoint.

        model_dict:
            State dictionary of the target model.

        allow_partial_copy:
            If True, allow partial row-wise restore for tensors where only
            dimension 0 differs and all trailing dimensions match. Defaults to False.

    Returns:
        Dict[str, torch.Tensor]:
            The updated model state dictionary with compatible pretrained weights loaded.

    Example:
        >>> model_dict = model.state_dict()
        >>> pretrained_dict = load_checkpoint("pretrained_model.ckpt")
        >>> model_dict = set_model_dict_for_partial_init(
        ...     pretrained_dict,
        ...     model_dict,
        ...     allow_partial_copy=True,
        ... )
        >>> model.load_state_dict(model_dict)
    """
    restored_dict = {}
    exact_restored = 0
    partial_restored = 0
    skipped_mismatch = 0

    for key, pretrained_value in pretrained_dict.items():
        if key not in model_dict:
            continue

        model_value = model_dict[key]

        if not hasattr(pretrained_value, "shape") or not hasattr(model_value, "shape"):
            continue

        if pretrained_value.shape == model_value.shape:
            restored_dict[key] = pretrained_value
            exact_restored += 1
            continue

        can_partial_copy = (
            allow_partial_copy
            and pretrained_value.ndim == model_value.ndim
            and pretrained_value.ndim > 0
            and pretrained_value.shape[1:] == model_value.shape[1:]
        )

        if can_partial_copy:
            merged_value = model_value.clone()
            rows_to_copy = min(pretrained_value.shape[0], model_value.shape[0])

            merged_value[:rows_to_copy].copy_(
                pretrained_value[:rows_to_copy].to(
                    device=merged_value.device,
                    dtype=merged_value.dtype,
                )
            )

            restored_dict[key] = merged_value
            partial_restored += 1

            logging.info(
                f" | > Partially restored resized tensor: {key} "
                f"pretrained={tuple(pretrained_value.shape)} "
                f"model={tuple(model_value.shape)} "
                f"copied_rows={rows_to_copy}"
            )
            continue

        skipped_mismatch += 1
        logging.info(
            f" | > Layer with shape mismatch in the model definition: {key} "
            f"pretrained={tuple(pretrained_value.shape)} "
            f"model={tuple(model_value.shape)}"
        )

    model_dict.update(restored_dict)

    logging.info(
        f" | > {len(restored_dict)} / {len(model_dict)} layers are restored "
        f"({exact_restored} exact, {partial_restored} partial, "
        f"{skipped_mismatch} skipped due to incompatible shape)."
    )

    return model_dict


def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from disk.

    Supports loading checkpoints stored in either PyTorch (`.ckpt`, `.pt`) or
    SafeTensors (`.safetensors`) formats. All parameters are loaded onto CPU
    regardless of the original device.

    Args:
        checkpoint_path (str):
            Path to the checkpoint file. If the filename contains `.safetensors`,
            it is loaded using the SafeTensors backend; otherwise, it is assumed
            to be a PyTorch checkpoint containing a `state_dict` field.

    Returns:
        dict:
            A state dictionary mapping parameter names to tensors.
    """
    if ".safetensors" in checkpoint_path:
        checkpoint_state = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    return checkpoint_state


def _load_checkpoint_state(checkpoint_path: str) -> dict:
    """Load checkpoint state dict from a file or HF directory.

    Args:
        checkpoint_path: Path to checkpoint file or HF directory with model.safetensors
    """
    import os

    if os.path.isdir(checkpoint_path):
        from safetensors.torch import load_file

        return load_file(os.path.join(checkpoint_path, "model.safetensors"))
    else:
        return torch.load(checkpoint_path, map_location='cpu')['state_dict']


def init_perception_from_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load perception module from another STT/S2S checkpoint.

    Args:
        model: The model whose perception module will be initialized
        checkpoint_path: Path to checkpoint file or HF directory
    """
    if checkpoint_path is None:
        return

    from nemo.utils import logging

    logging.info(f"Loading perception from checkpoint: {checkpoint_path}")
    checkpoint_state = _load_checkpoint_state(checkpoint_path)

    checkpoint_state = {k.replace("perception.", ""): v for k, v in checkpoint_state.items() if "perception." in k}
    checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.perception.state_dict())
    model.perception.load_state_dict(checkpoint_state, strict=True)


def init_model_from_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load full model state from a checkpoint.

    Args:
        model: The model to initialize
        checkpoint_path: Path to checkpoint file or HF directory
    """
    if checkpoint_path is None:
        return

    from nemo.utils import logging

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint_state = _load_checkpoint_state(checkpoint_path)

    checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.state_dict())
    model.load_state_dict(checkpoint_state, strict=True)


def load_pretrained_model(model: torch.nn.Module, checkpoint_path: str):
    """Load a pretrained S2S model from a checkpoint path.

    Supports both incremental loading from safetensors (for large models to avoid OOM)
    and standard loading from various checkpoint formats.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint file or HF directory
    """
    if checkpoint_path is None:
        return

    import gc
    import os
    from nemo.utils import logging

    logging.info(f"Loading pretrained s2s model from {checkpoint_path}")

    if os.path.isdir(checkpoint_path) and model.cfg.get("incremental_loading", False):
        # Hugging Face format with incremental loading
        from safetensors import safe_open

        # Load tensors incrementally to avoid OOM
        model_state_dict = model.state_dict()
        loaded_keys = []
        missing_keys = []

        with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt", device="cpu") as f:
            available_keys = f.keys()
            for key in available_keys:
                if key in model_state_dict:
                    # Load tensor and copy to model parameter
                    tensor = f.get_tensor(key)
                    model_state_dict[key].copy_(tensor)
                    loaded_keys.append(key)
                    del tensor  # Free memory immediately
                else:
                    missing_keys.append(key)

                # Periodic garbage collection for very large models
                if len(loaded_keys) % 100 == 0:
                    gc.collect()

        logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
        if missing_keys:
            logging.warning(f"Keys in checkpoint but not in model: {len(missing_keys)} keys")

        del model_state_dict
        gc.collect()
    else:
        init_model_from_checkpoint(model, checkpoint_path)


def _is_dcp_checkpoint(path: str) -> bool:
    """Check if a path is a distributed checkpoint (DCP) directory."""
    import os

    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


def init_from_training_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Initialize model weights from a previous training checkpoint.

    Only model weights are loaded — optimizer state, LR scheduler, and training
    step are NOT restored, enabling a fresh fine-tuning start from the checkpoint.

    Supports three checkpoint formats:
    - **Distributed checkpoints** (DCP): directories with a ``.metadata`` file,
      produced by ``ModelParallelStrategy`` / ``AutomodelParallelStrategy``.
      Handles resharding when parallelism differs between the source and target runs.
      Works with both FSDP2-wrapped (DTensor) and regular parameters.
    - **HuggingFace model directories**: contain ``model.safetensors``
      (e.g. output of ``to_hf.py``).
    - **Single-file checkpoints**: ``.ckpt`` or ``.pt`` files with a
      ``state_dict`` key.

    Args:
        model: The model to initialize.
        checkpoint_path: Path to the checkpoint (directory or file).
    """
    if checkpoint_path is None:
        return

    logging.info(f"Initializing model weights from training checkpoint: {checkpoint_path}")

    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT

    if ParallelExpertEncoderPT.is_pe_nemo(checkpoint_path):
        raise ValueError(
            f"init_from_checkpoint={checkpoint_path!r} points to a ParallelExpertEncoderPT bundle. "
            "Use model.pe_encoder_path for PE encoder bundles."
        )

    if _is_dcp_checkpoint(checkpoint_path):
        import torch.distributed.checkpoint as dcp

        # Lightning saves model weights under the "state_dict" key in DCP.
        # Wrapping with the same structure lets DCP match keys correctly.
        # Optimizer states and other trainer state are ignored automatically
        # because we only provide the model's state_dict.
        state_dict = {"state_dict": model.state_dict()}
        with python313_pathlib_pickle_compat():
            dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
        model.load_state_dict(state_dict["state_dict"])
        logging.info(f"Loaded distributed checkpoint from {checkpoint_path}")
    else:
        init_model_from_checkpoint(model, checkpoint_path)


def maybe_load_pretrained_models(model: torch.nn.Module):
    """
    Optionally load pretrained model weights based on configuration.

    Checks for and loads (in order):
    - ``pretrained_perception_from_s2s``: Perception module weights from another S2S checkpoint
    - ``pretrained_s2s_model``: Full S2S model weights from a checkpoint (supports incremental loading)
    - ``init_from_checkpoint``: Full model weights from a training checkpoint
      (DCP, HuggingFace directory, or single-file format). Only model weights
      are loaded; optimizer/scheduler state is discarded for a fresh fine-tuning start.
    """
    if model.cfg.get("pretrained_perception_from_s2s", None):
        init_perception_from_checkpoint(model, model.cfg.pretrained_perception_from_s2s)

    if model.cfg.get("pretrained_s2s_model", None):
        load_pretrained_model(model, model.cfg.pretrained_s2s_model)

    if model.cfg.get("init_from_checkpoint", None):
        init_from_training_checkpoint(model, model.cfg.init_from_checkpoint)


def _automodel_config_mtp_depth(config) -> int:
    """Return the physical MTP depth declared by an HF config."""
    depth = getattr(config, "num_nextn_predict_layers", None)
    if depth is None:
        depth = getattr(config, "mtp_num_hidden_layers", 0)
    return int(depth or 0)


_AUTOMODEL_HF_RESOLUTION_KWARGS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "resume_download",
    "revision",
    "subfolder",
    "token",
    "use_auth_token",
}


def _split_automodel_hf_resolution_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Separate Hugging Face repository resolution options from model-construction options."""
    config_kwargs = {name: value for name, value in kwargs.items() if name in _AUTOMODEL_HF_RESOLUTION_KWARGS}
    automodel_kwargs = {name: value for name, value in kwargs.items() if name not in _AUTOMODEL_HF_RESOLUTION_KWARGS}
    if "cache_dir" in kwargs:
        automodel_kwargs["cache_dir"] = kwargs["cache_dir"]
    return config_kwargs, automodel_kwargs


def _resolve_automodel_checkpoint_path(
    model_path_or_name: str, hf_kwargs: dict, *, include_weights: bool = True
) -> str:
    """Resolve one exact local checkpoint snapshot for config-first Automodel loading."""
    model_path = Path(model_path_or_name)
    subfolder = hf_kwargs.get("subfolder")
    if model_path.exists():
        resolved_path = model_path
    else:
        from huggingface_hub import snapshot_download

        snapshot_kwargs = {
            name: value
            for name, value in hf_kwargs.items()
            if name
            in {
                "cache_dir",
                "force_download",
                "local_files_only",
                "proxies",
                "resume_download",
                "revision",
                "token",
            }
        }
        if "token" not in snapshot_kwargs and "use_auth_token" in hf_kwargs:
            snapshot_kwargs["token"] = hf_kwargs["use_auth_token"]
        if not include_weights:
            snapshot_kwargs["allow_patterns"] = ["*.json", "*.py"]
        resolved_path = Path(snapshot_download(repo_id=model_path_or_name, **snapshot_kwargs))

    if subfolder:
        resolved_path = resolved_path / subfolder
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved Automodel checkpoint path does not exist: {resolved_path}")
    return str(resolved_path)


def _is_mtp_state_key(key: str) -> bool:
    return "mtp" in key.split(".")


@contextmanager
def _exclude_mtp_checkpoint_state(model):
    """Exclude MTP tensors even in Automodel's direct HF checkpoint fast paths."""
    load_hook = None
    if hasattr(model, "register_load_state_dict_pre_hook"):

        def remove_mtp_from_load(_module, state_dict, *_args):
            for key in list(state_dict):
                if _is_mtp_state_key(key):
                    state_dict.pop(key)

        load_hook = model.register_load_state_dict_pre_hook(remove_mtp_from_load)

    adapter = _find_automodel_state_dict_adapter(model)
    original_from_hf = getattr(adapter, "from_hf", None)
    adapter_attributes = getattr(adapter, "__dict__", {})
    had_instance_from_hf = "from_hf" in adapter_attributes
    instance_from_hf = adapter_attributes.get("from_hf")
    if original_from_hf is not None:

        def from_hf_without_mtp(state_dict, *args, **kwargs):
            filtered_state = {key: value for key, value in state_dict.items() if not _is_mtp_state_key(key)}
            return original_from_hf(filtered_state, *args, **kwargs)

        adapter.from_hf = from_hf_without_mtp
    try:
        yield
    finally:
        if load_hook is not None:
            load_hook.remove()
        if original_from_hf is not None:
            if had_instance_from_hf:
                adapter.from_hf = instance_from_hf
            else:
                del adapter.from_hf


def _find_automodel_state_dict_adapter(model):
    visited = set()
    current = model
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        adapter = getattr(current, "state_dict_adapter", None)
        if adapter is not None:
            return adapter
        current = getattr(current, "module", None) or getattr(current, "_orig_mod", None)
    return None
