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

import json
import random

import torch

from nemo.collections import llm, vlm
from nemo.collections.llm.gpt.data.hf_dataset import has_dist_env_init_or_rank_env_var


def mk_hf_vlm_dataset_fineweb_edu(data_path, processor, mbs, gbs):
    '''
    FineWeb-Edu dataset
    '''
    skipped_tokens = vlm.HFAutoModelForImageTextToText.extract_skipped_token_ids(processor)

    def collate_fn(examples, processor):
        # Simply tokenize the text directly
        texts = [example["text"] for example in examples]

        # Tokenize the text
        batch = processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )

        # Shift labels by 1 and truncate input_ids

        labels = batch["input_ids"].clone()[:, 1:]
        labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
        labels[torch.isin(labels, skipped_tokens)] = -100
        batch["labels"] = labels
        return batch

    return llm.HFDatasetDataModule(
        data_path,
        split="train",
        micro_batch_size=mbs,
        global_batch_size=gbs,
        collate_fn=lambda x: collate_fn(x, processor=processor),
        num_workers=4,
        persistent_workers=True,
        streaming=not has_dist_env_init_or_rank_env_var(),
    )


def mk_hf_vlm_dataset_rdr(data_path, processor, mbs, gbs):
    '''
    RDR dataset
    '''
    skipped_tokens = vlm.HFAutoModelForImageTextToText.extract_skipped_token_ids(processor)

    def collate_fn(examples, processor):
        def fmt(sample):
            instruction = "Describe accurately the given image."
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}, {"type": "image", "image": sample["image"]}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
            ]
            return {"conversation": conversation, "images": [sample['image'].convert("RGB")]}

        text = []
        images = []
        for example in map(fmt, examples):

            text.append(
                processor.apply_chat_template(
                    example["conversation"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images += example['images']

        # Tokenize the text and process the images
        batch = processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        labels = batch["input_ids"].clone()[:, 1:]  # shift labels by 1
        # Add a -100 to the end of the labels to make it same length as input_ids
        labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
        labels[torch.isin(labels, skipped_tokens)] = -100
        batch["labels"] = labels
        return batch

    return vlm.HFDatasetDataModule(
        data_path,
        split="train",
        micro_batch_size=mbs,
        global_batch_size=gbs,
        collate_fn=lambda x: collate_fn(x, processor=processor),
        num_workers=4,
        persistent_workers=True,
    )


def json2token(obj, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                output += fr"<s_{k}>" + json2token(obj[k], sort_json_key) + fr"</s_{k}>"
            return output
    elif type(obj) == list:
        return r"<sep/>".join([json2token(item, sort_json_key) for item in obj])
    else:
        obj = str(obj)
        return obj


def mk_hf_vlm_dataset_cord_v2(data_path, processor, mbs, gbs):
    '''
    CORD-V2 dataset
    '''
    skipped_tokens = vlm.HFAutoModelForImageTextToText.extract_skipped_token_ids(processor)

    def train_collate_fn(examples, processor):
        processed_examples = []
        for example in examples:
            ground_truth = json.loads(example["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            text = random.choice([json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons])
            processed_examples.append((example["image"], text))

        examples = processed_examples
        images = []
        texts = []

        for example in examples:
            image, ground_truth = example
            images.append(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract JSON"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ground_truth},
                    ],
                },
            ]
            text_prompt = processor.apply_chat_template(conversation)
            texts.append(text_prompt)

        batch = processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")

        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        labels = batch["input_ids"].clone()[:, 1:]  # shift labels by 1
        # Add a -100 to the end of the labels to make it same length as input_ids
        labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
        labels[torch.isin(labels, skipped_tokens)] = -100
        batch["labels"] = labels

        return batch

    return vlm.HFDatasetDataModule(
        data_path,
        split="train",
        micro_batch_size=mbs,
        global_batch_size=gbs,
        num_workers=4,
        persistent_workers=True,
        collate_fn=lambda x: train_collate_fn(x, processor=processor),
    )
