# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.multiprocessing as mp
from datasets import load_dataset

from nemo.core.config import hydra_runner
from nemo.export.quantize import Quantizer

mp.set_start_method("spawn", force=True)

"""
Nemo quantization example script.

Please consult nemo.export.quantize.Quantizer class
and examples/nlp/language_modeling/conf/megatron_llama_quantization.yaml config on available quantization methods,
models supported as well as how to set up data and inference for calibration (with defaults recommended).

Example usage:
```
python examples/nlp/language_modeling/megatron_llama_quantization.py \
    model_file=llama2-7b-fp16.nemo \
    model_save=llama2-7b-fp8.qnemo \
    quantization.algorithm=fp8 \
    export.decoder_type=llama \
    export.inference_tensor_parallel=1
```
"""


def get_calib_dataloader(data="cnn_dailymail", batch_size=64, calib_size=512, max_sequence_length=512):
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    else:
        # Assume a local JSON dataset with a column named "text"
        dataset = load_dataset("json", data_files=data, split="train")
        text_column = "text"
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


@hydra_runner(config_path="conf", config_name="megatron_llama_quantization")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference.")

    quantizer = Quantizer(cfg.quantization, cfg.inference, cfg.export, cfg.trainer)

    # Quantization algorithm can be set to None. This is useful for baseline precision
    # accuracy validation. In this case only weights export step will be performed:
    if cfg.quantization.algorithm is not None:
        dataloader = get_calib_dataloader(
            cfg.quantization.calib_dataset,
            cfg.inference.batch_size,
            cfg.quantization.num_calib_size,
            cfg.inference.max_context_length,
        )
        dataloader = [data for data in dataloader]
    else:
        dataloader = None

    model = quantizer.quantize(
        cfg.model_file, dataloader, cfg.tensor_model_parallel_size, cfg.pipeline_model_parallel_size
    )

    quantizer.export(model, cfg.model_save)


if __name__ == '__main__':
    main()
