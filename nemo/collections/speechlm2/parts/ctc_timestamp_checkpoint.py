# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persist the recognized frozen timestamp artifact through native training checkpoints."""

import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    CTC_TIMESTAMP_ARTIFACT_FORMAT,
    CTC_TIMESTAMP_BUNDLE_PREFIX,
    TransformerCTCDecoder,
    _load_sentencepiece,
    _validate_vocabulary,
)


class CTCTimestampCheckpoint(torch.nn.Module):
    """Frozen decoder and serialized tokenizer with their original storage dtypes.

    Construction metadata belongs in the model config. Tensors use their public
    bundle names in both the native state dict and the exported safetensors file.
    The head is not called by the ASR training loss.
    """

    def __init__(self, config: Mapping):
        super().__init__()
        self.artifact_metadata = dict(config['metadata'])
        if self.artifact_metadata['ctc_timestamp_format'] != CTC_TIMESTAMP_ARTIFACT_FORMAT:
            raise ValueError('Unsupported bundled CTC timestamp format')
        decoder_config = json.loads(self.artifact_metadata['ctc_timestamp_decoder_config'])
        self.decoder = TransformerCTCDecoder(**decoder_config)
        self.tokenizer = torch.nn.Module()
        self.tokenizer.register_buffer('model_proto', torch.empty(config['tokenizer_num_bytes'], dtype=torch.uint8))
        state = self.state_dict()
        dtypes = dict(config['tensor_dtypes'])
        if set(state) != set(dtypes):
            raise ValueError('Bundled CTC timestamp tensor coverage does not match its decoder configuration')
        for name, tensor in state.items():
            dtype = getattr(torch, dtypes[name])
            if name == 'tokenizer.model_proto':
                if dtype != torch.uint8 or tensor.numel() == 0:
                    raise ValueError('CTC tokenizer must be a nonempty uint8 tensor')
            elif not dtype.is_floating_point:
                raise ValueError(f'CTC decoder tensor must be floating point: {name}')
            # Keep parameters registered; loading uses strict canonical coverage.
            module_path, _, leaf = name.rpartition('.')
            module = self.get_submodule(module_path)
            if leaf in module._parameters:
                module._parameters[leaf].data = tensor.to(dtype=dtype)
            else:
                module._buffers[leaf] = tensor.to(dtype=dtype)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def validate_tokenizer(self) -> None:
        payload = json.loads(self.artifact_metadata['ctc_timestamp_tokenizer_config'])
        payload['model_proto'] = self.tokenizer.model_proto.detach().cpu().numpy().tobytes()
        tokenizer = _load_sentencepiece(payload)
        _validate_vocabulary(tokenizer, json.loads(self.artifact_metadata['ctc_timestamp_decoder_config']))


def read_ctc_timestamp_config(weight_file: str | Path) -> dict | None:
    """Read only the header; reject unrecognized or incomplete timestamp bundles."""
    with safe_open(str(weight_file), framework='pt', device='cpu') as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = [key for key in checkpoint.keys() if key.startswith(CTC_TIMESTAMP_BUNDLE_PREFIX)]
        has_metadata = any(key.startswith('ctc_timestamp_') for key in metadata)
        if not keys and not has_metadata:
            return None
        required = {'ctc_timestamp_format', 'ctc_timestamp_decoder_config', 'ctc_timestamp_tokenizer_config'}
        ctc_metadata = {key: value for key, value in metadata.items() if key.startswith('ctc_timestamp_')}
        if set(ctc_metadata) != required or ctc_metadata['ctc_timestamp_format'] != CTC_TIMESTAMP_ARTIFACT_FORMAT:
            raise ValueError('Invalid CTC timestamp bundle metadata')
        tokenizer_key = CTC_TIMESTAMP_BUNDLE_PREFIX + 'tokenizer.model_proto'
        if tokenizer_key not in keys:
            raise ValueError('Missing bundled CTC tokenizer')
        tokenizer = checkpoint.get_slice(tokenizer_key)
        if tokenizer.get_dtype() != 'U8' or len(tokenizer.get_shape()) != 1 or not tokenizer.get_shape()[0]:
            raise ValueError('CTC tokenizer must be a nonempty one-dimensional uint8 tensor')
        dtype_names = {'F32': 'float32', 'BF16': 'bfloat16', 'F16': 'float16', 'U8': 'uint8'}
        return {
            'metadata': ctc_metadata,
            'tokenizer_num_bytes': tokenizer.get_shape()[0],
            'tensor_dtypes': {
                key.removeprefix(CTC_TIMESTAMP_BUNDLE_PREFIX): dtype_names[checkpoint.get_slice(key).get_dtype()]
                for key in keys
            },
        }
