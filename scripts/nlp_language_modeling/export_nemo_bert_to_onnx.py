# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertTextEmbeddingModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--nemo_path", type=str, required=True)
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_bert_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument(
        "--onnx_path", type=str, default="bert.onnx", required=False, help="Path to output .nemo file."
    )
    parser.add_argument(
        "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )

    args = parser.parse_args()
    return args


def export(args):
    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision

    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronBertTextEmbeddingModel.restore_from(args.nemo_path, trainer=trainer)

    hf_tokenizer = model.tokenizer.tokenizer

    logging.info(f'=' * 50)
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
        'query: summit define',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # Tokenize the input texts
    batch_dict = hf_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    model = model.eval()

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["outputs"]
    export_input = tuple([batch_dict_cuda[name] for name in input_names])

    torch.onnx.export(
        model, export_input, args.onnx_path, verbose=False, input_names=input_names, output_names=output_names,
    )
    logging.info(f'NeMo model saved to: {args.onnx_path}')


if __name__ == '__main__':
    args = get_args()
    export(args)
