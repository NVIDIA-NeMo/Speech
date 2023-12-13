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
import io
import itertools
import json
import os
import pickle
import random
import re
from typing import Callable, List, Union

import boto3
import torch.distributed as dist
from botocore.config import Config
from PIL import Image

from nemo.collections.multimodal.data.common.data_samplers import SharedEpoch, WDSUrlsRandomSampler
from nemo.collections.multimodal.data.common.webdataset_s3 import WebDataset as WebDatasetS3
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.core.classes import IterableDataset as NeMoIterableDataset
from nemo.utils import logging

try:
    import webdataset as wds
    from webdataset import WebDataset, warn_and_continue
    from webdataset.filters import _shuffle
    from webdataset.utils import pytorch_worker_info

    HAVE_WEBDATASET = True

except (ImportError, AttributeError, ModuleNotFoundError):

    HAVE_WEBDATASET = False

    logging.warning("Webdataset import failed! We recommend use `webdataset==0.2.48`.")

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

Image.MAX_IMAGE_PIXELS = 933120000
_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()


def pil_loader(key, data):
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key: Image key.
        data: Image data stream.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert("RGB")

    return img


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


class WebDatasetCommon(NeMoIterableDataset):
    """
    A common dataset object shared by most of NeMo multimodal models.
    """

    def __init__(
        self,
        dataset_cfg,
        map_fn: Callable,
        compose_fn: Union[Callable, List[Callable]],
        consumed_samples: int,
        filter_fn: Callable = None,
        gen_cfg=None,
        decode_fn: Callable = None,
        is_train=True,
    ):

        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.num_workers = dataset_cfg.num_workers
        self.world_size = get_world_size()
        self.webdata_cfg = dataset_cfg.webdataset
        self.infinite_sampler = self.webdata_cfg.get("infinite_sampler", False)
        self.gen_cfg = gen_cfg
        self.consumed_samples = consumed_samples

        self.local_root_path = self.webdata_cfg.local_root_path
        if is_train:
            dataset_path = dataset_cfg.train.dataset_path
            self.augmentations = dataset_cfg.train.get("augmentations", None)
            self.filterings = dataset_cfg.train.get("filterings", None)
        else:
            dataset_path = dataset_cfg.validation.dataset_path
            self.augmentations = dataset_cfg.validation.get("augmentations", None)
            self.filterings = dataset_cfg.validation.get("filterings", None)

        if "boto3" in dataset_cfg:
            logging.info(f'Init boto3 using credentials file at {dataset_cfg.boto3.credentials_file}')
            self.use_boto3 = True
            assert dataset_cfg.boto3.credentials_file is not None
            with open(dataset_cfg.boto3.credentials_file) as fin:
                self.credentials = json.load(fin)
            config = Config(connect_timeout=30, signature_version="s3", retries={"max_attempts": 999999})
            self.s3 = boto3.client('s3', **self.credentials, config=config)
            self.bucket = dataset_cfg.boto3.bucket
            self.local_root_path = ""
        else:
            logging.info(f'Read Webdataset locally. Data stores at {self.local_root_path}')
            self.use_boto3 = False
            self.s3 = None
            self.bucket = None

        # wdinfo in a dict containing webdata information
        self.wdinfo = dict()
        if dataset_path[0].endswith(".pkl"):
            for dset_info_path in dataset_path:
                with open(dset_info_path, 'rb') as fp:
                    dset_info = pickle.load(fp)
                    if 'tar_files' not in self.wdinfo:
                        self.wdinfo['tar_files'] = dset_info['tar_files']
                        self.wdinfo['total_key_count'] = dset_info['total_key_count']
                        self.wdinfo['chunk_size'] = dset_info['chunk_size']
                    else:
                        self.wdinfo['tar_files'].extend(dset_info['tar_files'])
                        self.wdinfo['total_key_count'] += dset_info['total_key_count']
            train_info = self.wdinfo
        else:
            train_info = self.wdinfo
            train_info['tar_files'] = map(wds.shardlists.expand_urls, dataset_path)
            train_info['tar_files'] = list(itertools.chain.from_iterable(train_info['tar_files']))
            train_info['chunk_size'] = self.webdata_cfg.get("chunk_size", 1000)
            train_info['total_key_count'] = train_info['chunk_size'] * len(train_info['tar_files'])

        self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        chunk_size = train_info['chunk_size']

        num_workers = dataset_cfg.get("num_workers") or 1
        self.consumed_urls = (
            consumed_samples
            // (self.data_parallel_size * num_workers)
            // chunk_size
            * (self.data_parallel_size * num_workers)
        )
        self.consumed_samples = self.consumed_urls * chunk_size
        self.skip_ahead = consumed_samples - self.consumed_samples

        decode_fn = pil_loader if decode_fn is None else decode_fn
        shards_train_list = train_info["tar_files"]
        num_shards = len(shards_train_list)
        assert num_shards > 0, "Did not find any training data."

        # Shuffle buffer:
        shuffle_buffer_size = train_info["chunk_size"]

        if self.filterings is not None:
            # TODO : Not a good way of estimating filtering (We expect user to give estimated portion)
            # We should estimate in someway. This is anyway used only in progress bar
            logging.info(f'Estimated {self.filterings.estimated_portion} will be remaining after filtering')
            train_info["total_key_count"] = int(train_info["total_key_count"] * self.filterings.estimated_portion)

        # WDS Dataset Pipeline
        # DetShuffle -> Decode -> Filter -> Map -> Compose
        train_dataset, epoch = self._get_webdataset_and_epoch()
        train_dataset = train_dataset.compose(detshuffle2(bufsize=shuffle_buffer_size, epoch=epoch))
        train_dataset = train_dataset.decode(decode_fn, handler=warn_and_continue)

        if self.filterings is not None:
            if self.filterings.resolution is not None:
                train_dataset = train_dataset.select(filter_fn)

        train_dataset = train_dataset.map(map_fn, handler=warn_and_continue)
        if not isinstance(compose_fn, list):
            compose_fn = [compose_fn]
        for fn in compose_fn:
            train_dataset = train_dataset.compose(fn)
        train_dataset.total_images = train_info["total_key_count"]

        if train_info["total_key_count"] != train_info["chunk_size"] * len(train_info["tar_files"]):
            logging.warning("Total image count is not equal to chunk_size * number of tar files.")

        if self.infinite_sampler:
            rank, world_size, worker_id, num_workers = pytorch_worker_info()
            nbatches = train_dataset.total_images // world_size // self.num_workers
            logging.info(f'Setting nbatches={nbatches} for infinite sampler. world_size={world_size}')
            train_dataset = train_dataset.with_epoch(nbatches=nbatches)

        logging.info("Total number of training shards: %d", num_shards)
        logging.info("Total training key count: %d", train_dataset.total_images)

        self._dataset = train_dataset

    def _get_webdataset_and_epoch(self):
        train_info = self.wdinfo
        chunk_size = train_info["chunk_size"]
        shards_train_list = train_info["tar_files"]
        shards_train_list = [os.path.join(self.local_root_path, x) for x in shards_train_list]
        epoch = 0

        if not self.infinite_sampler:
            logging.info(f'Initiating Webdataset Random Sampler..')
            assert (
                self.filterings is None
            ), 'Webdataset Random Sampler should not be used with filters. Switch to infinite sampler'
            shards_train_list = WDSUrlsRandomSampler(
                urls=shards_train_list,
                total_urls=len(shards_train_list),
                chunk_size=chunk_size,
                consumed_samples=self.consumed_samples,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
                num_workers=self.dataset_cfg.get("num_workers") or 1,
                drop_last=True,
                data_sharding=self.dataset_cfg.train.get("data_sharding", True),
            )
            epoch = shards_train_list.epoch

        if self.use_boto3:
            train_dataset = WebDatasetS3(
                shards_train_list,
                handler=warn_and_continue,
                resampled=self.infinite_sampler or False,
                load_from_object_store=self.use_boto3,
                s3_client=self.s3,
                s3_bucket_name=self.bucket,
            )
        else:
            train_dataset = WebDataset(
                shards_train_list, handler=warn_and_continue, resampled=self.infinite_sampler or False,
            )

        return train_dataset, epoch

    def __iter__(self):
        ds_iter = self._dataset.__iter__()
        while self.skip_ahead > 0 and not self.infinite_sampler:
            try:
                _ = next(ds_iter)
                self.skip_ahead -= self.data_parallel_size * self.num_workers
            except StopIteration:
                self.skip_ahead = 0
        return ds_iter

    def __len__(self):
        return self._dataset.total_images


if HAVE_WEBDATASET:

    class detshuffle2(wds.PipelineStage):
        def __init__(
            self, bufsize=1000, initial=100, seed=0, epoch=-1,
        ):
            self.bufsize = bufsize
            self.initial = initial
            self.seed = seed
            self.epoch = epoch

        def run(self, src):
            if isinstance(self.epoch, SharedEpoch):
                epoch = self.epoch.get_value()
            else:
                # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
                # situation as different workers may wrap at different times (or not at all).
                self.epoch += 1
                epoch = self.epoch
            rng = random.Random()
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            if parallel_state.is_unitialized():
                seed = self.seed + epoch
            else:
                seed = self.seed + epoch + (100 * parallel_state.get_data_parallel_rank())
            rng.seed(seed)
            return _shuffle(src, self.bufsize, self.initial, rng)


else:

    class detshuffle2(ApexGuardDefaults):
        def __init__(self):
            super().__init__()
            logging.warning("Webdataset import failed! We recommend use `webdataset==0.2.48`.")
