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
import argparse
import json
from pathlib import Path

from prettytable import PrettyTable

import nemo.lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, default="models/llama_31_toy")
    parser.add_argument('--collection', type=str, default='llm')
    parser.add_argument("--model-type", type=str, help="Name of the model", default="LlamaModel")
    parser.add_argument("--model-config", type=str, help="Model config", default="Llama31Config8B")
    parser.add_argument("--output-path", type=str, help="Output NeMo2 model")
    parser.add_argument("--ignore-keys", nargs='+', type=str, default=[], help="Ignore keys")
    return parser


def count_parameters(model, ignore_keys=None):
    table = PrettyTable(["NeMo Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ignore_keys is not None and name in ignore_keys:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"NeMo Total Trainable Params: {total_params}")
    return total_params


if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.collection == 'llm':
        collection = llm
    elif args.collection == 'vlm':
        collection = vlm
    else:
        raise ValueError(f'Unrecognized collection {args.collection}')
    config = getattr(collection, args.model_config)()
    model = getattr(collection, args.model_type)(config=config)
    output_path = llm.import_ckpt(
        model=model,
        source=f"hf://{args.hf_path}",
        output_path=args.output_path,
        overwrite=True,
    )

    trainer = nl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=1,
        strategy=nl.MegatronStrategy(ckpt_load_strictness=False),
    )
    path = Path(output_path)
    model_nemo: nl.io.TrainerContext = nl.io.load_context(
        path=nl.ckpt_utils.ckpt_to_context_subdir(path), subpath="model"
    )
    _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model_nemo)

    # Load HF Stats
    with open(f'{args.hf_path}/stats.json') as f:
        hf_stats = json.load(f)

    table = PrettyTable(["HuggingFace Modules", "Parameters"])
    for key, value in hf_stats.items():
        table.add_row([key, value])
    print(table)

    nemo_param_cnt = count_parameters(model_nemo, args.ignore_keys)
    hf_param_cnt = hf_stats['total_params']

    assert (
        hf_param_cnt == nemo_param_cnt
    ), f'Total converted params count does not match: NeMo model has {nemo_param_cnt} and HF model has {hf_param_cnt}.'
    print('HuggingFace -> NeMo conversion completed successfully.')
