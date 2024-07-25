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

import numpy as np
from PIL import Image

from nemo.deploy.utils import str_list2numpy

use_pytriton = True
try:
    from pytriton.client import ModelClient
except Exception:
    use_pytriton = False

try:
    from decord import VideoReader
except Exception:
    import logging

    logging.warning("The package `decord` was not installed in this environment.")


class NemoQueryMultimodal:
    """
    Sends a query to Triton for Multimodal inference

    Example:
        from nemo.deploy.multimodal import NemoQueryMultimodal

        nq = NemoQueryMultimodal(url="localhost", model_name="neva", model_type="neva")

        input_text = "Hi! What is in this image?"
        output = nq.query(
            input_text=input_text,
            input_media="/path/to/image.jpg",
            max_output_len=30,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )
        print("prompts: ", prompts)
    """

    def __init__(self, url, model_name, model_type):
        self.url = url
        self.model_name = model_name
        self.model_type = model_type

    def setup_media(self, input_media):
        if self.model_type == "video-neva":
            vr = VideoReader(input_media)
            frames = [f.asnumpy() for f in vr]
            return np.array(frames)
        elif self.model_type == "lita" or self.model_type == "vita":
            vr = VideoReader(input_media)
            frames = [f.asnumpy() for f in vr]
            subsample_len = self.frame_len(frames)
            sub_frames = self.get_subsampled_frames(frames, subsample_len)
            return np.array(sub_frames)
        elif self.model_type == "neva" or self.model_type == "vila":
            media = Image.open(input_media).convert('RGB')
            return np.expand_dims(np.array(media), axis=0)
        else:
            raise RuntimeError(f"Invalid model type {self.model_type}")

    def frame_len(self, frames):
        max_frames = 256
        if len(frames) <= max_frames:
            return len(frames)
        else:
            subsample = int(np.ceil(float(len(frames)) / max_frames))
            return int(np.round(float(len(frames)) / subsample))

    def get_subsampled_frames(self, frames, subsample_len):
        idx = np.round(np.linspace(0, len(frames) - 1, subsample_len)).astype(int)
        sub_frames = [frames[i] for i in idx]
        return sub_frames

    def query(
        self,
        input_text,
        input_media,
        batch_size=1,
        max_output_len=30,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        init_timeout=60.0,
    ):

        prompts = str_list2numpy([input_text])
        inputs = {"input_text": prompts}

        media = self.setup_media(input_media)

        inputs["input_media"] = np.repeat(media[np.newaxis, :, :, :, :], prompts.shape[0], axis=0)

        if batch_size is not None:
            inputs["batch_size"] = np.full(prompts.shape, batch_size, dtype=np.int_)

        if max_output_len is not None:
            inputs["max_output_len"] = np.full(prompts.shape, max_output_len, dtype=np.int_)

        if top_k is not None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)

        if top_p is not None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)

        if temperature is not None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)

        if repetition_penalty is not None:
            inputs["repetition_penalty"] = np.full(prompts.shape, repetition_penalty, dtype=np.single)

        if num_beams is not None:
            inputs["num_beams"] = np.full(prompts.shape, num_beams, dtype=np.int_)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

            if output_type == np.bytes_:
                sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
                return sentences
            else:
                return result_dict["outputs"]
