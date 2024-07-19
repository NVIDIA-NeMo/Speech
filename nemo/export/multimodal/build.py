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


import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from time import time

import tensorrt as trt
import torch
import yaml
from tensorrt_llm.builder import Builder
from transformers import AutoModel

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model

logger = trt.Logger(trt.Logger.INFO)


def build_trtllm_engine(
    model_dir: str,
    visual_checkpoint_path: str,
    llm_checkpoint_path: str = None,
    model_type: str = "neva",
    llm_model_type: str = "llama",
    tensor_parallelism_size: int = 1,
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 1,
    max_multimodal_len: int = 1024,
    dtype: str = "bfloat16",
):
    trt_llm_exporter = TensorRTLLM(model_dir=model_dir, load_model=False)
    visual_checkpoint_model = ['neva', 'lita', 'vila', 'vita']
    trt_llm_exporter.export(
        nemo_checkpoint_path=visual_checkpoint_path if model_type in visual_checkpoint_model else llm_checkpoint_path,
        model_type=llm_model_type,
        tensor_parallelism_size=tensor_parallelism_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_prompt_embedding_table_size=max_multimodal_len,
        dtype=dtype,
        load_model=False,
    )


def export_visual_wrapper_onnx(
    visual_wrapper, input, output_dir, input_names=['input'], dynamic_axes={'input': {0: 'batch'}}
):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.makedirs(f'{output_dir}/onnx', exist_ok=True)
    torch.onnx.export(
        visual_wrapper,
        input,
        f'{output_dir}/onnx/visual_encoder.onnx',
        opset_version=17,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )


def build_trt_engine(
    model_type,
    input_sizes,
    output_dir,
    vision_max_batch_size,
    dtype=torch.bfloat16,
    image_size=None,
    num_frames=None,
    nemo_config=None,
):
    part_name = 'visual_encoder'
    onnx_file = '%s/onnx/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s.engine' % (output_dir, part_name)
    config_file = '%s/%s' % (output_dir, "config.json")
    nemo_config_file = '%s/%s' % (output_dir, "nemo_config.yaml")

    with open(nemo_config_file, 'w') as f:
        yaml.dump(nemo_config, f)

    logger.log(trt.Logger.INFO, "Building TRT engine for %s" % part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {"precision": str(dtype).split('.')[-1], "model_type": model_type}
    if image_size is not None:
        config_args["image_size"] = image_size
    if num_frames is not None:
        config_args["num_frames"] = num_frames

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    # Delete onnx files since we don't need them now
    shutil.rmtree(f'{output_dir}/onnx')

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(vision_max_batch_size / 2))
    nMaxBS = vision_max_batch_size

    inputT = network.get_input(0)

    # input sizes can be a list of ints (e.g., [3, H, W]) when inputs are images,
    # or a list of three int lists (e.g., [[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]]).
    assert isinstance(input_sizes, list), "input_sizes must be a list"
    if isinstance(input_sizes[0], int):
        logger.log(trt.Logger.INFO, f"Processed input sizes {input_sizes}")
        inputT.shape = [nBS, *input_sizes]
        min_size = opt_size = max_size = input_sizes
    elif len(input_sizes) == 3 and isinstance(input_sizes[0], list):
        min_size, opt_size, max_size = input_sizes
        logger.log(trt.Logger.INFO, f"Processed min/opt/max input sizes {min_size}/{opt_size}/{max_size}")
    else:
        raise ValueError(f"invalid input sizes: {input_sizes}")

    profile.set_shape(inputT.name, [nMinBS, *min_size], [nOptBS, *opt_size], [nMaxBS, *max_size])
    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO, "Succeeded building %s in %d s" % (engine_file, t1 - t0))
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

    Builder.save_config(config_wrapper, config_file)


def build_neva_engine(
    model_type: str,
    model_dir: str,
    visual_checkpoint_path: str,
    vision_max_batch_size: int = 1,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # extract NeMo checkpoint
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        mp0_weights, nemo_config, _ = load_nemo_model(visual_checkpoint_path, temp_path)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class DownSampleBlock(torch.nn.Module):
        def forward(self, x):
            vit_embeds = x
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.flat_square(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            return vit_embeds

        def flat_square(self, x):
            n, w, h, c = x.size()
            if w % 2 == 1:
                x = torch.cat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
                n, w, h, c = x.size()
            if h % 2 == 1:
                x = torch.cat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
                n, w, h, c = x.size()
            x = x.view(n, w, int(h / 2), int(c * 2))
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
            return x

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            vision_x = self.encoder(pixel_values=images, output_hidden_states=True)
            vision_x = vision_x.hidden_states[-2]
            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(
        vision_config["from_pretrained"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    if nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "mlp2x_gelu":
        vision_connector = torch.nn.Sequential(
            torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(nemo_config["hidden_size"], nemo_config["hidden_size"], bias=True),
        ).to(dtype=dtype)

        key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
        for layer in range(0, 3, 2):
            vision_connector[layer].load_state_dict(
                {
                    'weight': mp0_weights[f"{key_prefix}.{layer}.weight"].to(dtype),
                    'bias': mp0_weights[f"{key_prefix}.{layer}.bias"].to(dtype),
                }
            )
    elif nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "linear":
        vision_connector = torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True)
        key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
        vision_connector.load_state_dict(
            {
                'weight': mp0_weights[f"{key_prefix}.weight"].to(dtype),
                'bias': mp0_weights[f"{key_prefix}.bias"].to(dtype),
            }
        )
    elif nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "mlp_downsample":
        vision_connector = torch.nn.Sequential(
            DownSampleBlock(),
            torch.nn.LayerNorm(vision_config["hidden_size"] * 4),
            torch.nn.Linear(vision_config["hidden_size"] * 4, nemo_config["hidden_size"], bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(nemo_config["hidden_size"], nemo_config["hidden_size"], bias=True),
        ).to(dtype=dtype)
        key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
        for layer in [1, 2, 4]:
            vision_connector[layer].load_state_dict(
                {
                    'weight': mp0_weights[f"{key_prefix}.{layer}.weight"].to(dtype),
                    'bias': mp0_weights[f"{key_prefix}.{layer}.bias"].to(dtype),
                }
            )

    else:
        raise ValueError(f"Unknown projector type: {nemo_config['mm_cfg']['mm_mlp_adapter_type']}")

    # export the whole wrapper
    lita_num_frames = None
    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector).to(device, dtype)
    if model_type == "lita" or model_type == "vila":
        image_size = hf_config.image_size
        if model_type == "lita":
            lita_num_frames = nemo_config['mm_cfg']['lita']['sample_frames']
    else:
        image_size = hf_config.vision_config.image_size
        if model_type == "vita":
            lita_num_frames = nemo_config['mm_cfg']['lita']['sample_frames']
    dummy_image = torch.empty(
        1, 3, image_size, image_size, dtype=dtype, device=device
    )  # dummy image shape [B, C, H, W]

    export_visual_wrapper_onnx(wrapper, dummy_image, model_dir)
    build_trt_engine(
        model_type,
        [3, image_size, image_size],
        model_dir,
        vision_max_batch_size,
        dtype,
        image_size=image_size,
        num_frames=lita_num_frames if model_type == "lita" or model_type == 'vita' else None,
        nemo_config=nemo_config,
    )


def build_video_neva_engine(
    model_dir: str,
    visual_checkpoint_path: str,
    vision_max_batch_size: int = 1,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # extract NeMo checkpoint
    with tarfile.open(visual_checkpoint_path) as tar:
        nemo_config = yaml.safe_load(tar.extractfile("./model_config.yaml"))
        try:
            # trained without TP
            mp0_weights = torch.load(tar.extractfile("./model_weights.ckpt"), map_location=device)
        except KeyError:
            # trained with TP
            mp0_weights = torch.load(tar.extractfile("./mp_rank_00/model_weights.ckpt"), map_location=device)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            b, num_frames, c, h, w = images.shape
            images = images.view(b * num_frames, c, h, w)
            vision_x = self.encoder(pixel_values=images, output_hidden_states=True)  # [(B num_frames), C, H, W]
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]

            # reshape back to [B, num_frames, img_size, hidden_size]
            vision_x = vision_x.view(b, num_frames, -1, vision_x.shape[-1])

            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(
        vision_config["from_pretrained"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    assert nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "linear"
    vision_connector = torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True)

    key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
    vision_connector.load_state_dict(
        {
            'weight': mp0_weights[f"{key_prefix}.weight"].to(dtype),
            'bias': mp0_weights[f"{key_prefix}.bias"].to(dtype),
        }
    )

    # export the whole wrapper
    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector).to(device, dtype)
    image_size = hf_config.vision_config.image_size
    num_frames = nemo_config['data']['num_frames']
    dummy_video = torch.empty(1, num_frames, 3, image_size, image_size, dtype=dtype, device=device)  # dummy image
    export_visual_wrapper_onnx(wrapper, dummy_video, model_dir)
    build_trt_engine(
        "video-neva",
        [num_frames, 3, image_size, image_size],  # [num_frames, 3, H, W]
        model_dir,
        vision_max_batch_size,
        dtype,
        image_size=image_size,
        num_frames=num_frames,
    )


def build_visual_engine(
    model_dir: str,
    visual_checkpoint_path: str,
    model_type: str = "neva",
    vision_max_batch_size: int = 1,
):
    model_list = ['neva', 'lita', 'vila', 'vita']
    if model_type in model_list:
        build_neva_engine(model_type, model_dir, visual_checkpoint_path, vision_max_batch_size)
    elif model_type == "video-neva":
        build_video_neva_engine(model_dir, visual_checkpoint_path, vision_max_batch_size)
    else:
        raise RuntimeError(f"Invalid model type {model_type}")
