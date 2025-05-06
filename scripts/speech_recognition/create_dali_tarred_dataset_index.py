# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import logging
import os
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from omegaconf import MISSING

try:
    from wds2idx import IndexCreator

    INDEX_CREATOR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    INDEX_CREATOR_AVAILABLE = False

"""
python create_dali_tarred_dataset_index.py \
    tar_dir=<path to the directory which contains tarred dataset> \
    workers=-1

"""

logging.basicConfig(level=logging.INFO)


@dataclass
class DALITarredIndexConfig:
    tar_dir: str = MISSING  # Path to the existing dataset's manifest
    workers: int = -1  # number of worker processes


def process_index_path(tar_paths, index_dir):
    """
    Appends the folder `{index_dir}` to the filepath of all tarfiles.
    Example:
         /X/Y/Z/audio_0.tar -> /X/Y/Z/{index_dir}/audio_0.index
    """
    index_paths = []
    for path in tar_paths:
        basepath, filename = os.path.split(path)
        path = filename.replace('.tar', '.index')
        path = os.path.join(basepath, path)
        base, name = os.path.split(path)
        index_path = os.path.join(index_dir, name)
        index_paths.append(index_path)

    return index_paths


def build_index(tarpath, indexfile):
    with IndexCreator(tarpath, indexfile) as index:
        index.create_index()


@hydra.main(config_path=None, config_name='index_config', version_base="1.1")
def main(cfg: DALITarredIndexConfig):
    if not INDEX_CREATOR_AVAILABLE:
        logging.error("`wds2idx` is not installed. Please install NVIDIA DALI >= 1.11")
        exit(1)

    tar_files = list(glob.glob(os.path.join(cfg.tar_dir, "*.tar")))

    index_dir = os.path.join(cfg.tar_dir, "dali_index")
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)

    index_paths = process_index_path(tar_files, index_dir)

    with Parallel(n_jobs=cfg.workers, verbose=len(tar_files)) as parallel:
        _ = parallel(delayed(build_index)(tarpath, indexfile) for tarpath, indexfile in zip(tar_files, index_paths))

    logging.info("Finished constructing index files !")


ConfigStore.instance().store(name='index_config', node=DALITarredIndexConfig)


if __name__ == '__main__':
    main()
