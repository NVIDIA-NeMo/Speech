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
from functools import partial
from typing import Any, List, Union

import torch
from torch.utils.data import Dataset, default_collate

from nemo.collections.common.data.data_samplers import MegatronPretrainingSampler
from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.clip.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon
from nemo.collections.vision.data.megatron.image_folder import ImageFolder

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def tokenize(texts: Union[str, List[str]], tokenizer: Any, context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    tokenizer:
        Tokenizer loaded in NeMo
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    texts_is_str = False
    if isinstance(texts, str):
        texts = [texts]
        texts_is_str = True

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    all_tokens = [([bos_id] if bos_id is not None else []) + tokenizer.text_to_ids(text) + [eos_id] for text in texts]
    result = torch.ones(len(all_tokens), context_length, dtype=torch.long) * pad_id

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eos_id
        result[i, : len(tokens)] = torch.tensor(tokens)

    if texts_is_str:
        result = result[0]
    return result


# pylint: disable=C0116
def get_preprocess_fns_params(
    img_h, img_w, img_mean=None, img_std=None, is_train=True, max_position_embedding=None, tokenizer=None
):

    # This is equivalent to `get_preprocess_fns` but does not need the whole config to get the functions. This is
    # Particularly used in Nemo2
    # Define transforms
    img_size = (img_h, img_w)
    img_transform = image_transform(
        img_size,
        is_train=is_train,
        mean=img_mean,
        std=img_std,
    )
    text_transform = lambda x: x
    if tokenizer is not None:
        text_transform = partial(
            tokenize,
            tokenizer=tokenizer,
            context_length=max_position_embedding,
        )
    return img_transform, text_transform


def get_preprocess_fns(model_cfg, tokenizer=None, is_train=True):
    # Define transforms
    img_size = (model_cfg.vision.get("img_h"), model_cfg.vision.get("img_w"))
    img_mean = model_cfg.vision.get("img_mean")
    img_std = model_cfg.vision.get("img_std")
    img_transform = image_transform(
        img_size,
        is_train=is_train,
        mean=img_mean,
        std=img_std,
    )
    text_transform = lambda x: x
    if tokenizer is not None:
        text_transform = partial(
            tokenize,
            tokenizer=tokenizer,
            context_length=model_cfg.text.get("max_position_embeddings"),
        )
    return img_transform, text_transform


# This function maps data that are tuples to dictionary.
def tuple_to_dict(inp):
    for input in inp:
        out_dict = dict()
        out_dict['images'] = input[0]
        out_dict['captions'] = input[1]
        yield out_dict


def transform_fn(sample, img_transform, text_transform):
    image, text = sample["jpg"], sample["txt"]
    img_transformed, text_transformed = img_transform(image), text_transform(text)
    return img_transformed, text_transformed


def build_train_valid_datasets(
    model_cfg,
    consumed_samples,
    tokenizer=None,
):
    data_cfg = model_cfg.data

    train_img_transform, text_transform = get_preprocess_fns(model_cfg, tokenizer, is_train=True)
    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        map_fn=partial(transform_fn, img_transform=train_img_transform, text_transform=text_transform),
        compose_fn=tuple_to_dict,
        is_train=True,
    )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("dataset_path"):
        val_img_transform, text_transform = get_preprocess_fns(model_cfg, tokenizer, is_train=False)
        val_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=0,
            map_fn=partial(transform_fn, img_transform=val_img_transform, text_transform=text_transform),
            compose_fn=tuple_to_dict,
            is_train=False,
        )

    return train_data, val_data


def custom_collate(batch):
    if len(batch) == 0:
        return None, None
    else:
        return default_collate(batch)


def build_imagenet_validation_dataloader_params(
    imagenet_val,
    img_h,
    img_w,
    mbs,
    gbs,
    num_workers=0,
    pin_memory=True,
    img_mean=None,
    img_std=None,
    is_train=False,
    max_position_embedding=None,
    tokenizer=None,
):
    # This is equivalent to `build_imagenet_validation_dataloader` but does not need the whole config.
    # Particularly used in Nemo2
    val_image_transform, text_transform = get_preprocess_fns_params(
        img_h,
        img_w,
        img_mean,
        img_std,
        is_train=is_train,
        max_position_embedding=max_position_embedding,
        tokenizer=tokenizer,
    )

    imagenet_val_data = {}

    imagenet_path = imagenet_val
    if imagenet_path is None:
        return None

    image_dataset = ImageFolder(
        root=imagenet_path,
        transform=val_image_transform,
    )

    image_batch_sampler = MegatronPretrainingSampler(
        total_samples=len(image_dataset),
        consumed_samples=0,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        drop_last=False,
    )

    imagenet_val_data["images"] = torch.utils.data.DataLoader(
        image_dataset,
        batch_sampler=image_batch_sampler,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    text_dataset = ImagenetClassnameDataset(imagenet_classnames, openai_imagenet_template, text_transform)

    imagenet_val_data["texts"] = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=text_dataset.num_templates,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    return imagenet_val_data


# pylint: enable=C0116
# For zero-shot imagenet validation
def build_imagenet_validation_dataloader(model_cfg, tokenizer=None):
    """Build dataloaders"""
    val_image_transform, text_transform = get_preprocess_fns(model_cfg, tokenizer, is_train=False)
    data_cfg = model_cfg.data

    imagenet_val = {}

    imagenet_path = data_cfg.get("imagenet_val")
    if imagenet_path is None:
        return None

    image_dataset = ImageFolder(
        root=imagenet_path,
        transform=val_image_transform,
    )

    image_batch_sampler = MegatronPretrainingSampler(
        total_samples=len(image_dataset),
        consumed_samples=0,
        micro_batch_size=model_cfg.micro_batch_size,
        global_batch_size=model_cfg.global_batch_size,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        drop_last=False,
    )

    imagenet_val["images"] = torch.utils.data.DataLoader(
        image_dataset,
        batch_sampler=image_batch_sampler,
        num_workers=min(data_cfg.num_workers, 2),
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True,
    )

    text_dataset = ImagenetClassnameDataset(imagenet_classnames, openai_imagenet_template, text_transform)
    imagenet_val["texts"] = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=text_dataset.num_templates,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    return imagenet_val


class ImagenetClassnameDataset(Dataset):
    """Imagenet class dataset"""

    def __init__(self, classnames, templates, text_transform):
        # pylint: disable=C0116
        self.num_templates = len(templates)
        self.samples = []
        for classname in classnames:
            texts = [template(classname) for template in templates]
            self.samples.extend(text_transform(texts))

    def __getitem__(self, index):
        # pylint: disable=C0116
        return self.samples[index]

    def __len__(self):
        # pylint: disable=C0116
        return len(self.samples)
