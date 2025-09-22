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
"""Pruning example for Llama model.

Example usage to prune width automatically (you can skip parameters that you don't want to prune):
```python
    torchrun --nproc_per_node 8 scripts/llm/gpt_prune.py \
        --devices 8 \
        --tp_size 1 \
        --pp_size 8 \
        --restore_path <path/to/llama3.1-8b-nemo2> \
        --seq_length 8192 \
        --data_paths 1.0 path/to/tokenized/data \
        --index_mapping_dir path/to/index_mapping_dir \
        --target_ffn_hidden_size 9216 \
        --target_hidden_size 3072 \
        --target_num_attention_heads 32 \
        --target_num_query_groups 8 \
        --save_path llama3.1-8b-width-pruned
```

Example usage to prune depth automatically using cosine-similarity based importance metric:
```python
    torchrun --nproc_per_node 8 scripts/llm/gpt_prune.py \
        --devices 8 \
        --tp_size 1 \
        --pp_size 8 \
        --restore_path <path/to/llama3.1-8b-nemo2> \
        --seq_length 8192 \
        --data_paths 1.0 path/to/tokenized/data \
        --index_mapping_dir path/to/index_mapping_dir \
        --target_num_layers 16 \
        --save_path llama3.1-8b-depth-pruned
```

NOTE: for above usages, `--tp_size` must be 1 because of the current prune API limitation. If you
do not pass `--data_paths` and `--index_mapping_dir`, the script will use mock data for calibration which will
lead to randomly pruned model but helps in testing the pruning pipeline.

Example usage to prune depth by dropping specific model layers (1-indexed):
```python
    torchrun --nproc_per_node 8 scripts/llm/gpt_prune.py \
        --devices 8 \
        --tp_size 8 \
        --pp_size 1 \
        --restore_path <path/to/llama3.1-8b-nemo2> \
        --drop_layers 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --save_path llama3.1-8b-dropped-layers
```
"""

import argparse
import os

# isort: off
# Import modelopt first to avoid circular import in 0.35.0
import modelopt.torch.prune  # noqa: F401

# isort: on

from nemo.collections import llm
from nemo.collections.llm.modelopt.prune import PruningConfig
from nemo.utils import logging

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_data_module(args):
    """Get data module for running validation loop on."""
    assert args.num_train_samples % args.mbs == 0, "num_train_samples must be divisible by mbs"
    assert args.seq_length, "Sequence length must be provided for pruning"

    data_module_kwargs = {}
    if args.data_paths:
        logging.info(f"Loading pre-training data from {args.data_paths}")
        data_module_cls = llm.PreTrainingDataModule
        data_module_kwargs["paths"] = args.data_paths
        data_module_kwargs["split"] = args.split
        data_module_kwargs["index_mapping_dir"] = args.index_mapping_dir
    else:
        logging.warning("Using mock data since --data_paths is not provided.")
        data_module_cls = llm.MockDataModule
    data_module = data_module_cls(
        seq_length=args.seq_length,
        micro_batch_size=args.mbs,
        global_batch_size=args.mbs,  # global_batch_size is not used as there is no backward pass
        **data_module_kwargs,
    )

    return data_module


def main(args):
    """Main function for pruning Llama model."""
    pruning_config = PruningConfig(
        target_ffn_hidden_size=args.target_ffn_hidden_size,
        target_hidden_size=args.target_hidden_size,
        target_num_attention_heads=args.target_num_attention_heads,
        target_num_query_groups=args.target_num_query_groups,
        target_mamba_num_heads=args.target_mamba_num_heads,
        target_mamba_head_dim=args.target_mamba_head_dim,
        target_num_layers=args.target_num_layers,
        drop_layers=args.drop_layers,
    )

    data_module = get_data_module(args) if not args.drop_layers else None

    llm.prune(
        nemo_checkpoint=args.restore_path,
        save_path=args.save_path,
        pruning_config=pruning_config,
        devices=args.devices,
        num_nodes=args.num_nodes,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        num_train_samples=args.num_train_samples,
        data=data_module,
        tokenizer_path=args.tokenizer,
        legacy_ckpt=args.legacy_ckpt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Pruning Script")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use per node")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size. Can only be 1 if pruning and not dropping layers",
    )
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage. If None, pp will default to evenly split layers.",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage. If None, pp will default to evenly split layers.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        help="Sequence length. Only required if pruning and not dropping layers.",
    )
    parser.add_argument("--restore_path", type=str, required=True, help="Path to restore model checkpoint from")
    parser.add_argument(
        "--legacy_ckpt",
        action="store_true",
        help="Load ckpt saved with older TE versions. Use for missing state dict keys ending with `_extra_state`",
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save pruned model checkpoint to")
    parser.add_argument(
        "--tokenizer", type=str, help="Tokenizer to use for data module. If not provided, model tokenizer will be used"
    )
    # Calibration data parameters
    parser.add_argument(
        "--data_paths",
        type=str,
        metavar="N",
        nargs="*",
        help="""Paths of the data distributions. Only required if pruning and not dropping layers
            The given paths will be used to generate the train, validation and test datasets.
            The format can be either (1) a list of paths, e.g.
                ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"],
            or (2) a flattened, zipped list of weights and paths, e.g.
                ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]
            Check nemo.collections.llm.PreTrainingDataModule for more info and alternative formats.
        """,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="99,1,0",
        help="A string of 3 comma-separated integers denoting how much of the distribution to "
        "allocate to train, validation, and test sets, respectively",
    )
    parser.add_argument("--index_mapping_dir", type=str, help="Path to a directory to write index mapping files")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=1024,
        help="Number of training samples for importance estimation",
    )
    # Pruning parameters
    parser.add_argument("--target_ffn_hidden_size", type=int, help="Prune MLP FFN hidden size to this value")
    parser.add_argument("--target_hidden_size", type=int, help="Prune hidden size (embedding dim) to this value")
    parser.add_argument(
        "--target_num_attention_heads",
        type=int,
        help="Prune number of attention heads to this value. Must be supplied with --target_num_query_groups",
    )
    parser.add_argument(
        "--target_num_query_groups",
        type=int,
        help="Prune number of query groups to this value. Must be supplied with --target_num_attention_heads",
    )
    parser.add_argument(
        "--target_num_layers",
        type=int,
        help="Prune number of transformer layers to this value based on "
        "Block Influence metric (cosine similarity) as per https://arxiv.org/abs/2403.03853",
    )
    parser.add_argument(
        "--target_mamba_num_heads",
        type=int,
        help="Prune number of Mamba attention heads to this value",
    )
    parser.add_argument(
        "--target_mamba_head_dim",
        type=int,
        help="Prune dimension of Mamba attention heads to this value",
    )
    parser.add_argument(
        "--drop_layers",
        type=int,
        metavar="N",
        nargs="*",
        help="Drop specific model layers (1-indexed). Cannot be used with rest of the pruning options",
    )

    args = parser.parse_args()
    main(args)
