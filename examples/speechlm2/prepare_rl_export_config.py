# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build canonical native-RL export configuration from the original local HF seed."""

import argparse
import json
from pathlib import Path

from nemo.collections.speechlm2.parts.ctc_timestamp_checkpoint import read_ctc_timestamp_config


def prepare_config(seed: Path, world_size: int, expert_parallel_size: int) -> dict:
    seed = seed.resolve()
    if world_size < 1 or expert_parallel_size < 1 or world_size % expert_parallel_size:
        raise ValueError('World size must be positive and divisible by expert parallel size')
    model = json.loads((seed / 'config.json').read_text())
    if not (seed / 'llm_backbone' / 'config.json').is_file():
        raise FileNotFoundError(seed / 'llm_backbone' / 'config.json')
    model['pretrained_llm'] = str(seed / 'llm_backbone')
    model['tokenizer_path'] = str(seed)
    ctc_config = read_ctc_timestamp_config(seed / 'model.safetensors')
    if ctc_config is not None:
        model['ctc_timestamp_artifact'] = ctc_config
    strategy = {
        '_target_': 'nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy',
        'dp_size': world_size,
        'dp_replicate_size': 1,
        'tp_size': 1,
        'pp_size': 1,
        'cp_size': 1,
        'ep_size': expert_parallel_size,
    }
    return {'model': model, 'trainer': {'strategy': strategy}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--expert-parallel-size', type=int, default=8)
    parser.add_argument('--strategy-overrides', type=json.loads, default={})
    parser.add_argument('--backend-config', type=json.loads)
    args = parser.parse_args()
    config = prepare_config(args.seed, args.world_size, args.expert_parallel_size)
    config['trainer']['strategy'].update(args.strategy_overrides)
    if args.backend_config is not None:
        config['model']['automodel_backend'] = args.backend_config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(config, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
