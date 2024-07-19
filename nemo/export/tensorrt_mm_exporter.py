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

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import wrapt

from nemo.deploy import ITritonDeployable
from nemo.export.multimodal.build import build_trtllm_engine, build_visual_engine
from nemo.export.multimodal.run import MultimodalModelRunner

use_deploy = True
try:
    from nemo.deploy.utils import cast_output, ndarray2img, str_ndarray2list
except Exception:
    use_deploy = False


@wrapt.decorator
def noop_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False


LOGGER = logging.getLogger("NeMo")


class TensorRTMMExporter(ITritonDeployable):
    """
    Exports nemo checkpoints to TensorRT and run fast inference.

    Example:
        from nemo.export import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir="/path/for/model/files")
        exporter.export(
            visual_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="neva",
            tensor_parallel_size=1,
        )

        output = exporter.forward("Hi! What is in this image?", "/path/for/input_media")
        print("output: ", output)

    """

    def __init__(
        self,
        model_dir: str,
        load_model: bool = True,
    ):
        self.model_dir = model_dir
        self.runner = None

        if load_model:
            self._load()

    def export(
        self,
        visual_checkpoint_path: str,
        llm_checkpoint_path: str = None,
        model_type: str = "neva",
        llm_model_type: str = "llama",
        tensor_parallel_size: int = 1,
        max_input_len: int = 4096,
        max_output_len: int = 256,
        max_batch_size: int = 1,
        vision_max_batch_size: int = 1,
        max_multimodal_len: int = 3072,
        dtype: str = "bfloat16",
        delete_existing_files: bool = True,
        load_model: bool = True,
    ):
        if Path(self.model_dir).exists():
            if delete_existing_files and len(os.listdir(self.model_dir)) > 0:
                for files in os.listdir(self.model_dir):
                    path = os.path.join(self.model_dir, files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)

                if len(os.listdir(self.model_dir)) > 0:
                    raise Exception("Couldn't delete all files.")
            elif len(os.listdir(self.model_dir)) > 0:
                raise Exception("There are files in this folder. Try setting delete_existing_files=True.")
        else:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        llm_dir = os.path.join(self.model_dir, "llm_engine")
        build_trtllm_engine(
            model_dir=llm_dir,
            visual_checkpoint_path=visual_checkpoint_path,
            llm_checkpoint_path=llm_checkpoint_path,
            model_type=model_type,
            llm_model_type=llm_model_type,
            tensor_parallelism_size=tensor_parallel_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_multimodal_len=max_multimodal_len,
            dtype=dtype,
        )

        visual_dir = os.path.join(self.model_dir, "visual_engine")
        build_visual_engine(visual_dir, visual_checkpoint_path, model_type, vision_max_batch_size)

        if load_model:
            self._load()

    def forward(
        self,
        input_text: str,
        input_media: str,
        batch_size: int = 1,
        max_output_len: int = 30,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
    ):
        if self.runner is None:
            raise Exception(
                "A nemo checkpoint should be exported and " "then it should be loaded first to run inference."
            )

        input_media = self.runner.load_test_media(input_media)
        return self.runner.run(
            input_text,
            input_media,
            max_output_len,
            batch_size,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            num_beams,
        )

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="input_text", shape=(-1,), dtype=bytes),
            Tensor(name="input_media", shape=(-1, -1, -1, 3), dtype=np.uint8),
            Tensor(name="batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_output_len", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="repetition_penalty", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="num_beams", shape=(-1,), dtype=np.int_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        try:
            if self.runner is None:
                raise Exception(
                    "A nemo checkpoint should be exported and " "then it should be loaded first to run inference."
                )

            infer_input = {"input_text": str_ndarray2list(inputs.pop("input_text")[0])}
            video_model_list = ["video-neva", "lita", "vita"]
            if self.runner.model_type == "neva" or self.runner.model_type == "vila":
                infer_input["input_image"] = ndarray2img(inputs.pop("input_media")[0])[0]
            elif self.runner.model_type in video_model_list:
                infer_input["input_image"] = inputs.pop("input_media")[0]
            if "batch_size" in inputs:
                infer_input["batch_size"] = inputs.pop("batch_size")[0][0]
            if "max_output_len" in inputs:
                infer_input["max_new_tokens"] = inputs.pop("max_output_len")[0][0]
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")[0][0]
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")[0][0]
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")[0][0]
            if "repetition_penalty" in inputs:
                infer_input["repetition_penalty"] = inputs.pop("repetition_penalty")[0][0]
            if "num_beams" in inputs:
                infer_input["num_beams"] = inputs.pop("num_beams")[0][0]

            output_texts = self.runner.run(**infer_input)
            output = cast_output(output_texts, np.bytes_)
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)

        return {"outputs": output}

    def _load(self):
        llm_dir = os.path.join(self.model_dir, "llm_engine")
        visual_dir = os.path.join(self.model_dir, "visual_engine")
        self.runner = MultimodalModelRunner(visual_dir, llm_dir)
