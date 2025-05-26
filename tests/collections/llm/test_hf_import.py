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
from pathlib import Path

from nemo import lightning as nl
from nemo.collections import llm

"""
Script to import a Hugging Face model checkpoint into NeMo 2.0 format.

Example usage:

python test_hf_import.py \
    --hf_model /path/to/hf/model \
    --model LlamaModel \
    --config Llama31Config8B \
    --output_path /path/to/nemo2/model

The source model can be a local directory or a model from HF Hub.
It may have different parameters than specified in the config. For example,
it be a small model with just two layers and lower hidden dimension size.
In this case, configuration will be overriden from the input HF config.

Finally, the output NeMo model is loaded using the Fabric API of pl.Trainer.
"""


def get_args():
    parser = argparse.ArgumentParser(description='Test Llama2 7B model model conversion from HF')
    parser.add_argument('--hf_model', type=str, help="Original HF model")
    parser.add_argument("--model", default="LlamaModel", help="Model class from nemo.collections.llm module")
    parser.add_argument("--config", default="Llama2Config7B", help="Config class from nemo.collections.llm module")
    parser.add_argument('--output_path', type=str, help="NeMo 2.0 export path")
    parser.add_argument('--overwrite', action="store_true", help="Overwrite the output model if exists")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    ModelClass = getattr(llm, args.model)
    ModelConfig = getattr(llm, args.config)
    model = ModelClass(config=ModelConfig)
    nemo2_path = llm.import_ckpt(
        model=model,
        source="hf://" + args.hf_model,
        output_path=Path(args.output_path),
        overwrite=args.overwrite,
    )

    trainer = nl.Trainer(
        devices=1,
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
        plugins=nl.MegatronMixedPrecision(precision='fp16'),
    )
    fabric = trainer.to_fabric()
    trainer.strategy.setup_environment()
    fabric.load_model(nemo2_path)
