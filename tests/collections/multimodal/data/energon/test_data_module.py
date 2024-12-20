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

import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import webdataset as wds
from megatron.energon.flavors import BaseWebdataset
from PIL import Image
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule, ImageToken, MultiModalSampleConfig


class TestEnergonMultiModalDataModuleWithDummyData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        cls.tokenizer = cls.processor.tokenizer
        cls.image_processor = cls.processor.image_processor

    def setUp(self):
        self.config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200), ignore_place_holder=-100
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        self.dataset_path.mkdir(exist_ok=True, parents=True)

        self.create_vqa_test_dataset(self.dataset_path, 10)

        self.data_module = EnergonMultiModalDataModule(
            path=str(self.dataset_path),
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            num_workers=0,
            micro_batch_size=1,
            global_batch_size=2,
            multimodal_sample_config=self.config,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def decode_vqa_tokens_to_text(self, tokens):
        placeholder = self.data_module.multimodal_sample_config.image_token.token_id
        placeholder_text = self.data_module.multimodal_sample_config.image_token.token_str
        current_chunk = []
        decoded_text = []
        for token in tokens:
            if token == placeholder:
                # Decode the current chunk if there are any tokens in it
                if current_chunk:
                    decoded_text.append(self.tokenizer.decode(current_chunk, clean_up_tokenization_spaces=True))
                    current_chunk = []
                    # Append the placeholder text
                    decoded_text.append(placeholder_text)
            else:
                current_chunk.append(token)
        if current_chunk:
            decoded_text.append(self.tokenizer.decode(current_chunk, clean_up_tokenization_spaces=True))
        return "".join(decoded_text)

    def test_data_module(self):
        print(f"Data module initialized with path: {self.dataset_path}")

        # Test train dataloader
        train_loader = self.data_module.train_dataloader()
        for batch in train_loader:
            self.assertIsInstance(batch, dict)
            self.assertIn('position_ids', batch)
            self.assertIn('tokens', batch)
            self.assertIn('labels', batch)
            self.assertIn('loss_mask', batch)
            self.assertIn('attention_mask', batch)
            print(batch)
            decoded_text = self.decode_vqa_tokens_to_text(batch['tokens'][0].tolist())
            # system_message = re.escape(self.data_module.multimodal_sample_config.conversation_template_config.system)
            user_context = re.escape(self.vqa_json[0]['value'])
            assistant_answer = re.escape(self.vqa_json[1]['value'])
            # self.assertRegex(
            #     decoded_text,
            #     rf"{system_message}",
            #     msg="System message block does not match the expected format.",
            # )
            self.assertRegex(decoded_text, user_context, msg="User context did not match in decoded text")
            self.assertRegex(
                decoded_text, assistant_answer, msg="Assistant answer block did not match in decoded text"
            )
            break
        # Test val dataloader
        val_loader = self.data_module.val_dataloader()
        for batch in val_loader:
            self.assertIsInstance(batch, dict)
            self.assertIn('position_ids', batch)
            self.assertIn('tokens', batch)
            self.assertIn('labels', batch)
            self.assertIn('loss_mask', batch)
            self.assertIn('attention_mask', batch)
            print(batch)
            decoded_text = self.decode_vqa_tokens_to_text(batch['tokens'][0].tolist())
            # system_message = re.escape(self.data_module.multimodal_sample_config.conversation_template_config.system)
            user_context = re.escape(self.vqa_json[0]['value'])
            assistant_answer = re.escape(self.vqa_json[1]['value'])
            # self.assertRegex(
            #     decoded_text,
            #     rf"{system_message}",
            #     msg="System message block does not match the expected format.",
            # )
            self.assertRegex(decoded_text, user_context, msg="User context did not match in decoded text")
            self.assertRegex(
                decoded_text, assistant_answer, msg="Assistant answer block did not match in decoded text"
            )
            break

    def create_vqa_test_dataset(self, path: Path, num_samples: int):
        main_folder_name = ".nv-meta"
        main_folder_path = path / main_folder_name
        main_folder_path.mkdir(exist_ok=True, parents=True)
        self.vqa_json = [
            {"from": "human", "value": "<image>\nRender a clear and concise summary of the photo."},
            {"from": "gpt", "value": "a spartan helmet, laurels and laurel wreath, silhouette logo, emblem"},
        ]
        with wds.ShardWriter(f"{path}/data-%d.tar", maxcount=5) as shard_writer:
            for idx in range(num_samples):
                # Create a dummy image with random noise
                img_buf = io.BytesIO()
                randimg = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                image = Image.fromarray(randimg)
                image.save(img_buf, format="JPEG")
                img_bytes = img_buf.getvalue()

                json_data = json.dumps(self.vqa_json).encode("utf-8")

                shard_writer.write(
                    {
                        "__key__": f"{idx:06d}",
                        "jpg": img_bytes,
                        "json": json_data,
                    }
                )

            total_shards = shard_writer.shard
        BaseWebdataset.prepare_dataset(
            path,
            [f"data-{{0..{total_shards-1}}}.tar"],
            split_parts_ratio=[("train", 1.0), ("val", 1.0)],
        )
        # Create dataset.yaml inside the .nv-meta folder
        dataset_yaml_content = "\n".join(
            [
                "__class__: VQAWebdataset",
                "__module__: megatron.energon",
                "field_map:",
                "  answers: json[1][value]",
                "  context: json[0][value]",
                "  image: jpg",
            ]
        )
        with open(main_folder_path / "dataset.yaml", "w") as f:
            f.write(dataset_yaml_content)


if __name__ == '__main__':
    unittest.main()
