# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union


def add_optimizer_args(
    parent_parser: ArgumentParser,
    optimizer: str = 'adam',
    default_lr: float = None,
    default_opt_args: Optional[Union[Dict[str, Any], List[str]]] = None,
) -> ArgumentParser:
    """Extends existing argparse with support for optimizers.

    # Example of adding optimizer args to command line :
    python train_script.py ... --optimizer "novograd" --lr 0.01 \
        --opt_args betas=0.95,0.5 weight_decay=0.001

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.
        optimizer (str): Default optimizer required.
        default_lr (float): Default learning rate that should be overriden during training.
        default_opt_args (list(str)): List of overriding arguments for the instantiated optimizer.

    Returns:
        ArgumentParser: Parser extended by Optimizers arguments.
    """
    if default_opt_args is None:
        default_opt_args = []

    parser = ArgumentParser(parents=[parent_parser], add_help=True, conflict_handler='resolve')

    parser.add_argument('--optimizer', type=str, default=optimizer, help='Name of the optimizer. Defaults to Adam.')
    parser.add_argument('--lr', type=float, default=default_lr, help='Learning rate of the optimizer.')
    parser.add_argument(
        '--opt_args',
        default=default_opt_args,
        nargs='+',
        type=str,
        help='Overriding arguments for the optimizer. \n Must follow the pattern : \n name=value separated by spaces.'
        'Example: --opt_args weight_decay=0.001 eps=1e-8 betas=0.9,0.999',
    )

    return parser


def add_scheduler_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default LR scheduler args.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.

    Returns:
        ArgumentParser: Parser extended by LR Scheduler arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    parser.add_argument("--warmup_steps", type=int, required=False, default=None, help="Number of warmup steps")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=False,
        default=None,
        help="Number of warmup steps as a percentage of total training steps",
    )
    parser.add_argument("--hold_steps", type=int, required=False, default=None, help="Number of hold LR steps")
    parser.add_argument(
        "--hold_ratio",
        type=float,
        required=False,
        default=None,
        help="Number of hold LR steps as a percentage of total training steps",
    )
    parser.add_argument("--min_lr", type=float, required=False, default=0.0, help="Minimum learning rate")
    parser.add_argument(
        "--last_epoch", type=int, required=False, default=-1, help="Last epoch id. -1 indicates training from scratch"
    )
    return parser


def add_asr_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default ASR collection args.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.

    Returns:
        ArgumentParser: Parser extended by NeMo ASR Collection arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    parser.add_argument("--asr_model", type=str, required=True, default=None, help="Path to ASR model config yaml")
    parser.add_argument("--train_dataset", type=str, required=True, default=None, help="training dataset path")
    parser.add_argument("--eval_dataset", type=str, required=True, help="evaluation dataset path")
    return parser


def add_nlp_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default NLP collection args.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.

    Returns:
        ArgumentParser: Parser extended by NeMo NLP Collection arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    parser.add_argument(
        "--data_dir", type=str, required=False, help="data directory to training or/and evaluation dataset"
    )
    parser.add_argument(
        "--config_file", type=str, required=False, default=None, help="Huggingface model configuration file"
    )
    parser.add_argument(
        "--pretrained_model_name", default='bert-base-uncased', type=str, required=False, help="pretrained model name"
    )
    parser.add_argument(
        "--tokenizer_name", default='nemobert', type=str, choices=['sentencepiece', 'nemobert'], help="Tokenizer type"
    )
    parser.add_argument("--tokenizer_model", default=None, type=str, help="Tokenizer file for sentence piece")
    parser.add_argument("--do_lower_case", action='store_true', required=False, help="lower case data")
    return parser
