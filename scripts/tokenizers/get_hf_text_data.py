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
"""
This script is used to download text corpus from HuggingFace datasets,
where the saved corpus can be further used to train a tokenizer using `process_asr_text_tokenizer.py`.

Usage:
```
python get_hf_text_data.py --config-path="conf" --config-name="huggingface_data_tokenizer"
```

Please refer to "conf/huggingface_data_tokenizer.yaml" for more details.
"""


import os
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import datasets as hf_datasets
from omegaconf import OmegaConf, open_dict

from nemo.core.config import hydra_runner
from nemo.utils import logging


def clean_text(text: str, symbols_to_keep=None):
    symbols_to_keep = [x for x in symbols_to_keep] if symbols_to_keep is not None else []
    text = text.lower()
    # only keep alphanumeric characters, spaces and symbols defined in self.symbols_to_keep
    text = ''.join([c for c in text if c.isalnum() or c.isspace() or c in symbols_to_keep])
    return text


def get_nested_dict_value(dictionary: dict, key: str):
    """
    the key should be a string of nested keys separated by `.`, e.g. `key1.key2.key3`,
    then the returned value will be `dictionary[key1][key2][key3]`
    """
    nested_keys = key.split(".")
    result = dictionary
    for k in nested_keys:
        if k not in result:
            raise KeyError(
                f"Key `{key}` not found in [{result.keys()}], target is {nested_keys}, input is {dictionary}"
            )
        result = result[k]
    return result


def worker(x):
    sample, cfg = x
    text = get_nested_dict_value(sample, cfg.text_key)
    if cfg.normalize_text:
        text = clean_text(text, cfg.symbols_to_keep)
    return text


@hydra_runner(config_path="conf", config_name="huggingface_data_tokenizer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.output_file is None:
        cfg.output_file = 'huggingface_text_corpus.txt'

    if Path(cfg.output_file).exists():
        logging.info(f"Output file {cfg.output_file} already exists, removing it...")
        os.system(f"rm {cfg.output_file}")

    for data_cfg in cfg.hf_data_cfg:
        if 'num_proc' in data_cfg and data_cfg.get('streaming', False):
            logging.warning("num_proc is not supported for streaming datasets, removing it from config")
            with open_dict(data_cfg):
                data_cfg.pop('num_proc')
        logging.info(
            f"Loading from HuggingFace datasets library with config: {OmegaConf.to_container(data_cfg, resolve=True)}"
        )
        dataset = hf_datasets.load_dataset(**data_cfg)
        logging.info("Start extracting text from dataset...")
        with Pool(cfg.num_workers) as p:
            text_corpus = p.map(worker, zip(dataset, repeat(cfg)))
            with Path(cfg.output_file).open('a') as f:
                for line in text_corpus:
                    f.write(f"{line}\n")
            logging.info(f"Finished processing {len(text_corpus)} samples from {data_cfg}")
    logging.info("All Done!")


if __name__ == '__main__':
    main()
