# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize a training-only acoustic aligner without restoring the TTS model."""

import tarfile
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn


def resolve_aligner_config(config: Mapping | None = None) -> dict:
    """Resolve the ST-TTS flow and relative-transformer architecture configuration."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('aligned_text_aligner must be a configuration mapping')
    values = dict(config or {})
    defaults = {
        'flow_type': 'sttts',
        'hidden_channels': 128,
        'num_flows': 6,
        'num_text_layers': 2,
        'text_encoder_type': 'relative_transformer',
        'text_num_heads': 2,
        'text_filter_channels': None,
        'text_kernel_size': 3,
        'text_dropout': 0.1,
        'text_rel_attn_window_size': 4,
        'use_speaker_conditioning': False,
        'speaker_hidden_channels': 64,
        'speaker_embedding_dim': 64,
        'speaker_num_layers': 2,
        'coupling_num_layers': 4,
        'coupling_kernel_size': 5,
        'flow_n_split': 4,
        'max_log_scale': None,
        'dequantization_noise': 0.01,
        'mas_backend': 'auto',
    }
    unknown = values.keys() - defaults.keys() - {'audio_channels', 'text_channels'}
    if unknown:
        raise ValueError(f'Unknown aligned_text_aligner options: {sorted(unknown)}')
    defaults.update(values)
    for name, supported in (('flow_type', 'sttts'), ('text_encoder_type', 'relative_transformer')):
        if defaults[name] != supported:
            raise ValueError(f'aligned_text_aligner.{name} must be {supported!r}, got {defaults[name]!r}')
    return defaults


def load_pretrained_text_speech_aligner(
    aligner: nn.Module, checkpoint_path: str | Path | None, *, aligner_config: Mapping | None = None
) -> tuple[nn.Module, dict]:
    """Return a completely initialized aligner and its architecture configuration.

    Only ``text_aligner.*`` weights (including common model/DDP wrappers and the
    modular ``alignment.text_aligner.*`` namespace) are used. Bare aligner state
    dictionaries and Lightning ``state_dict`` checkpoints are also accepted.
    A ``.nemo`` donor supplies its own ``aligned_text_aligner`` architecture,
    including optional codec-reference speaker conditioning. Its audio/text
    feature dimensions must match those supplied by the receiving TTS model.

    No source TTS model is instantiated and no receiving module is modified.
    Loading into a separate candidate makes failures atomic. Call before the
    optimizer is created, then assign the returned module to ``text_aligner``.
    ``None`` preserves the receiving object. Non-NeMo checkpoint metadata uses
    pickle and must come from a trusted source.
    """
    config = (
        resolve_aligner_config(aligner_config)
        if aligner_config is not None
        else {key: getattr(aligner, key, default) for key, default in resolve_aligner_config().items()}
    )
    if checkpoint_path is None:
        return aligner, config
    source_path = str(checkpoint_path)
    state, source_config = _read_checkpoint(source_path)
    if source_config is not None:
        config = resolve_aligner_config(source_config)
    for name in ('audio_channels', 'text_channels'):
        supplied = getattr(aligner, name)
        requested = config.pop(name, supplied)
        if requested != supplied:
            raise ValueError(
                f'Checkpoint {source_path!r} requires aligner {name}={requested}, '
                f'but the receiving model provides {supplied}.'
            )

    # Candidate initialization must not advance TTS dropout/dequantization RNG.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        candidate = type(aligner)(audio_channels=aligner.audio_channels, text_channels=aligner.text_channels, **config)
    parameter = next(aligner.parameters(), None)
    if parameter is not None:
        candidate = candidate.to(device=parameter.device, dtype=parameter.dtype)
    candidate.train(aligner.training)
    trainable = {name: parameter.requires_grad for name, parameter in aligner.named_parameters()}
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(trainable.get(name, True))

    expected = candidate.state_dict()
    selected = _select_aligner_weights(state, expected, source_path)
    if any(isinstance(value, torch.Tensor) and value.is_meta for value in selected.values()):
        raise ValueError(f'Checkpoint {source_path!r} is incompatible with text_aligner: meta tensors have no values.')
    try:
        candidate.load_state_dict(selected, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f'Checkpoint {source_path!r} is incompatible with text_aligner: {error}') from error
    return candidate, config


def _read_checkpoint(source_path):
    config = None
    if Path(source_path).suffix.lower() == '.nemo':
        # Read only the two required members. Do not extract tokenizer/codec
        # assets or reconstruct the much larger source TTS model.
        with tarfile.open(source_path, 'r:*') as archive:
            members = archive.getmembers()
            weight_members = [
                member for member in members if member.isfile() and Path(member.name).name == 'model_weights.ckpt'
            ]
            config_members = [
                member for member in members if member.isfile() and Path(member.name).name == 'model_config.yaml'
            ]
            if len(weight_members) != 1 or len(config_members) > 1:
                raise ValueError(f'Checkpoint {source_path!r} must contain one unsharded model_weights.ckpt.')
            if config_members:
                content = archive.extractfile(config_members[0]).read().decode('utf-8')
                config = _find_aligner_config(OmegaConf.create(content))
            with archive.extractfile(weight_members[0]) as stream:
                state = torch.load(stream, map_location='cpu', weights_only=True)
    else:
        state = torch.load(source_path, map_location='cpu', weights_only=False)
        if isinstance(state, Mapping):
            for metadata_key in ('hyper_parameters', 'cfg', 'config', 'model_config'):
                candidate = _find_aligner_config(state.get(metadata_key))
                if candidate is not None:
                    config = candidate
                    break
    if isinstance(state, Mapping) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, Mapping):
        raise ValueError(f'Checkpoint {source_path!r} does not contain a model state dictionary.')
    return state, config


def _find_aligner_config(config):
    if not isinstance(config, Mapping):
        return None
    for wrapper in ('cfg', 'model'):
        if isinstance(config.get(wrapper), Mapping):
            config = config[wrapper]
    value = config.get('aligned_text_aligner')
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('The source aligned_text_aligner configuration must be a mapping.')
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def _select_aligner_weights(state, expected, source_path):
    normalized = {}
    wrappers = ('model.', 'module.', '_orig_mod.')
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f'Checkpoint {source_path!r} contains a non-string state dictionary key.')
        while key.startswith(wrappers):
            key = key.split('.', 1)[1]
        if key in normalized:
            raise ValueError(f'Checkpoint {source_path!r} contains duplicate unwrapped state key {key!r}.')
        normalized[key] = value
    namespaces = [
        {key[len(prefix) :]: value for key, value in normalized.items() if key.startswith(prefix)}
        for prefix in ('text_aligner.', 'alignment.text_aligner.')
    ]
    namespaces = [values for values in namespaces if values]
    if len(namespaces) > 1:
        raise ValueError(f'Checkpoint {source_path!r} contains multiple text_aligner namespaces.')
    if namespaces:
        return namespaces[0]
    if expected.keys() & normalized.keys():
        return normalized
    raise ValueError(f'Checkpoint {source_path!r} contains no text_aligner weights.')
