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
from typing import List

import tensorrt as trt
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm.builder import BuildConfig, Builder
from tensorrt_llm.commands.build import build as build_trtllm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import MLLaMAForCausalLM
from tensorrt_llm.plugin import PluginConfig
from transformers import AutoModel, AutoProcessor, MllamaForConditionalGeneration

from nemo.collections.multimodal.speech_llm.modules.perception_modules import AudioPerceptionModule
from nemo.core.classes.common import typecheck
from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model

from .converter import convert_mllama_nemo_to_hf

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
    use_lora_plugin: str = None,
    lora_target_modules: List[str] = None,
    max_lora_rank: int = 64,
    lora_ckpt_list: List[str] = None,
):
    """Build TRTLLM engine by nemo export"""
    trt_llm_exporter = TensorRTLLM(model_dir=model_dir, lora_ckpt_list=lora_ckpt_list, load_model=False)
    trt_llm_exporter.export(
        nemo_checkpoint_path=visual_checkpoint_path if llm_checkpoint_path is None else llm_checkpoint_path,
        model_type=llm_model_type,
        tensor_parallelism_size=tensor_parallelism_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_seq_len=max_input_len + max_output_len,
        max_batch_size=max_batch_size,
        max_prompt_embedding_table_size=max_multimodal_len,
        dtype=dtype,
        load_model=False,
        use_lora_plugin=use_lora_plugin,
        lora_target_modules=lora_target_modules,
        max_lora_rank=max_lora_rank,
        use_mcore_path=False,
    )


def build_mllama_trtllm_engine(
    model_dir: str,
    hf_model_path: str,
    tensor_parallelism_size: int = 1,
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 1,
    max_multimodal_len: int = 1024,
    dtype: str = "bfloat16",
    use_lora_plugin: str = None,
    lora_target_modules: List[str] = None,
    max_lora_rank: int = 64,
    lora_ckpt_list: List[str] = None,
):
    """Build mllama TRTLLM engine from HF"""
    if max_batch_size < 4:
        print(
            "TensorRT LLM may hit a runtime issue with batch size is smaller than 4 on some models." " Force set to 4"
        )
        max_batch_size = 4

    plugin_config = PluginConfig()
    plugin_config.gpt_attention_plugin = "auto"
    plugin_config.gemm_plugin = "auto"
    plugin_config.enable_paged_kv_cache(tokens_per_block=128)
    plugin_config.remove_input_padding = True
    plugin_config.use_paged_context_fmha = True

    max_seq_len = max_input_len + max_output_len
    max_num_tokens, opt_num_tokens = check_max_num_tokens(
        max_num_tokens=None,
        opt_num_tokens=None,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_beam_width=1,
        remove_input_padding=True,
        enable_context_fmha=plugin_config.context_fmha,
        tokens_per_block=128,
        multiple_profiles=False,
    )

    build_dict = {
        'max_input_len': max_input_len,
        'max_output_len': max_output_len,
        'max_encoder_input_len': max_multimodal_len,
        'max_batch_size': max_batch_size,
        'max_beam_width': 1,
        'max_seq_len': max_seq_len,
        'max_num_tokens': max_num_tokens,
        'opt_num_tokens': opt_num_tokens,
        'strongly_typed': True,
        'builder_opt': None,
    }
    build_config = BuildConfig.from_dict(build_dict, plugin_config=plugin_config)

    for rank in range(tensor_parallelism_size):
        mapping = Mapping(world_size=tensor_parallelism_size, rank=rank, tp_size=tensor_parallelism_size)
        model = MLLaMAForCausalLM.from_hugging_face(
            hf_model_path,
            dtype,
            mapping=mapping,
        )

        engine = build_trtllm(model, build_config)
        engine.save(model_dir)


def export_visual_wrapper_onnx(
    visual_wrapper, input, output_dir, input_names=['input'], dynamic_axes={'input': {0: 'batch'}}
):
    """Export visual wrapper to ONNX"""
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


def export_perception_wrapper_onnx(
    perception_wrapper,
    input,
    output_dir,
    input_names=['processed_signal', 'processed_signal_length'],
    output_names=['encoded', 'encoded_length'],
    dynamic_axes={
        'processed_signal': {0: 'batch', 2: 'time'},
        'processed_signal_length': {0: 'batch'},
        'encoded': {0: 'batch', 1: 'time'},
        'encoded_length': {0: 'batch'},
    },
):
    """Export perception wrapper to ONNX"""
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.makedirs(f'{output_dir}/onnx', exist_ok=True)
    torch.onnx.export(
        perception_wrapper,
        input,
        f'{output_dir}/onnx/perception_encoder.onnx',
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
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
    part_name='visual_encoder',
):
    """Build TRT engine from onnx"""
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
    # or a list of three list of lists
    # (e.g., [{input1: min_shape, input2: min_shape, }, \
    #     {input1: opt_shape, input2: opt_shape}, \
    # {input1: max_shape, input2: max_shape}] )
    assert isinstance(input_sizes, list), "input_sizes must be a list"
    if isinstance(input_sizes[0], int):
        logger.log(trt.Logger.INFO, f"Processed input sizes {input_sizes}")
        inputT.shape = [nBS, *input_sizes]
        min_size = opt_size = max_size = input_sizes
    elif len(input_sizes) == 3 and isinstance(input_sizes[0], list):
        min_size, opt_size, max_size = input_sizes
        logger.log(trt.Logger.INFO, f"Processed min/opt/max input sizes {min_size}/{opt_size}/{max_size}")
    elif len(input_sizes) == 3 and isinstance(input_sizes[0], dict):
        logger.log(trt.Logger.INFO, f"Processed min/opt/max input sizes {input_sizes}")
    else:
        raise ValueError(f"invalid input sizes: {input_sizes}")

    if isinstance(input_sizes[0], dict):
        for i in range(network.num_inputs):
            inputT = network.get_input(i)
            input_name = inputT.name
            min_size = input_sizes[0][input_name]
            opt_size = input_sizes[1][input_name]
            max_size = input_sizes[2][input_name]
            logger.log(trt.Logger.INFO, f"{input_name} min/opt/max input sizes {min_size}/{opt_size}/{max_size}")
            profile.set_shape(input_name, min_size, opt_size, max_size)
    else:
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
    """Build neva visual engine"""
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if os.path.isdir(visual_checkpoint_path):
        # load untar checkpoint
        config_path = os.path.join(visual_checkpoint_path, 'model_config.yaml')
        with open(config_path, 'r') as f:
            nemo_config = yaml.safe_load(f)
        try:
            weights_path = os.path.join(visual_checkpoint_path, 'model_weights.ckpt')
            mp0_weights = torch.load(weights_path, map_location=device)
        except FileNotFoundError:
            weights_path = os.path.join(visual_checkpoint_path, 'mp_rank_00/model_weights.ckpt')
            mp0_weights = torch.load(weights_path, map_location=device)
    else:
        # extract NeMo checkpoint
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            mp0_weights, nemo_config, _ = load_nemo_model(visual_checkpoint_path, temp_path)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class DownSampleBlock(torch.nn.Module):
        # pylint: disable=C0115,C0116
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
        # pylint: disable=C0115,C0116
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
        vision_config["from_pretrained"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='eager',
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
    """Build video neva visual engine"""
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
        # pylint: disable=C0115,C0116
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
        vision_config["from_pretrained"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='eager',
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


def build_perception_engine(
    model_dir: str,
    perception_checkpoint_path: str,
    model_type: str = "salm",
    max_batch_size: int = 1,
):
    """Build perception engine"""
    assert model_type == "salm", f"Invalid model type {model_type}"

    def load_perception_model(perception_checkpoint_path):
        weights = "model_weights.ckpt"
        perception_state_dict = torch.load(os.path.join(perception_checkpoint_path, weights))
        config = "model_config.yaml"
        config = OmegaConf.load(os.path.join(perception_checkpoint_path, config))
        perception = AudioPerceptionModule(cfg=config)
        perception.load_state_dict(perception_state_dict)
        perception.eval()
        return perception

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # load perception model
    perception_model = load_perception_model(perception_checkpoint_path)
    feature_extractor = perception_model.preprocessor
    input_signal = torch.randn(1, 1000, dtype=torch.float32)
    input_signal_length = torch.tensor([1000], dtype=torch.int32)

    processed_signal, processed_signal_length = feature_extractor(
        input_signal=input_signal, length=input_signal_length
    )
    processed_signal_length = processed_signal_length.to(torch.int32)
    dump_path = model_dir + "/feature_extractor.ts"  # dump the feature extractor as torchscript
    feature_extractor.export(dump_path, (input_signal, input_signal_length))

    class PerceptionWrapper(torch.nn.Module):
        # pylint: disable=C0115,C0116
        def __init__(self, encoder, modality_adapter, proj):
            super().__init__()
            self.encoder = encoder
            self.modality_adapter = modality_adapter
            self.proj = proj

        @typecheck.disable_checks()
        def forward(self, processed_signal, processed_signal_length):
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
            encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)
            # b, c, t -> b, t, c
            encoded = self.proj(encoded.transpose(1, 2))
            encoded_len = encoded_len.to(torch.int32)
            return encoded, encoded_len

    perception = PerceptionWrapper(perception_model.encoder, perception_model.modality_adapter, perception_model.proj)
    export_perception_wrapper_onnx(perception, (processed_signal, processed_signal_length), model_dir)
    # export the onnx perception model to tensorrt engine
    # 512 -> 5.12 sec, 3072 -> 30.72 sec
    opt_batch_size = max(1, max_batch_size // 2)
    shapes = [
        {"processed_signal": [1, 80, 64], "processed_signal_length": [1]},
        {"processed_signal": [opt_batch_size, 80, 512], "processed_signal_length": [opt_batch_size]},
        {"processed_signal": [max_batch_size, 80, 3072], "processed_signal_length": [max_batch_size]},
    ]
    build_trt_engine(
        model_type,
        shapes,
        model_dir,
        max_batch_size,
        dtype=torch.float16,
        nemo_config=None,
        part_name='perception_encoder',
    )


def build_mllama_visual_engine(
    model_dir: str,
    hf_model_path: str,
    processor_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    vision_max_batch_size: int = 1,
):
    """Build mllama visual engine"""
    hf_model = MllamaForConditionalGeneration.from_pretrained(hf_model_path, torch_dtype="auto", device_map="auto")
    model_dtype = hf_model.dtype

    class MLLaMAVisionWrapper(torch.nn.Module):
        # pylint: disable=C0115,C0116
        def __init__(self, vision_model, output_proj):
            super().__init__()
            self.vision_model = vision_model
            self.output_proj = output_proj

        def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
            out = self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask).last_hidden_state
            out = self.output_proj(out)
            return out

    wrapper = MLLaMAVisionWrapper(hf_model.vision_model, hf_model.multi_modal_projector)

    processor = AutoProcessor.from_pretrained(processor_name)
    image = Image.new('RGB', [2048, 2688])
    inputs = processor(images=image, return_tensors="pt").to(model_dtype)

    export_visual_wrapper_onnx(
        wrapper,
        tuple([value for _, value in inputs.items()]),
        model_dir,
        input_names=[key for key in inputs],
        dynamic_axes={key: {0: "batch"} for key in inputs},
    )
    shapes = [{k: list(v.shape) for k, v in inputs.items()}] * 3
    shapes[2] = shapes[0].copy()
    for k, v in shapes[2].items():
        shapes[2][k] = [vision_max_batch_size] + v[1:]
    build_trt_engine("mllama", shapes, model_dir, vision_max_batch_size, model_dtype)


def build_visual_engine(
    model_dir: str,
    visual_checkpoint_path: str,
    model_type: str = "neva",
    vision_max_batch_size: int = 1,
):
    """Build visual engine"""
    model_list = ['neva', 'lita', 'vila', 'vita']
    if model_type in model_list:
        build_neva_engine(model_type, model_dir, visual_checkpoint_path, vision_max_batch_size)
    elif model_type == "video-neva":
        build_video_neva_engine(model_dir, visual_checkpoint_path, vision_max_batch_size)
    else:
        raise RuntimeError(f"Invalid model type {model_type}")


def extract_lora_ckpt(
    lora_ckpt: str,
    output_dir: str,
):
    """Extrace lora from checkpoint"""
    if os.path.exists(os.path.join(lora_ckpt, "model_weights.ckpt")):
        model_weight = torch.load(os.path.join(lora_ckpt, "model_weights.ckpt"))
    elif os.path.exists(os.path.join(lora_ckpt, "mp_rank_00", "model_weights.ckpt")):
        model_weight = torch.load(os.path.join(lora_ckpt, "mp_rank_00", "model_weights.ckpt"))
    else:
        raise RuntimeError("Imcompatible lora checkpoint format")

    model_config = os.path.join(lora_ckpt, "model_config.yaml")

    if not os.path.exists(model_config):
        raise RuntimeError("Imcompatible lora checkpoint format")

    llm_lora_weight = {}

    for k, v in model_weight.items():
        if "mm_projector" not in k:
            llm_lora_weight[k] = v

    llm_lora_path = os.path.join(output_dir, "llm_lora.nemo")
    with tempfile.TemporaryDirectory() as tmp_dir:
        llm_weight_path = os.path.join(tmp_dir, "model_weights.ckpt")
        torch.save(llm_lora_weight, llm_weight_path)

        with tarfile.open(llm_lora_path, "w") as tar:
            tar.add(llm_weight_path, arcname="model_weights.ckpt")
            tar.add(model_config, arcname="model_config.yaml")

    return llm_lora_path


def build_mllama_engine(
    model_dir: str,
    checkpoint_path: str,
    processor_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    vision_max_batch_size: int = 1,
    tensor_parallelism_size: int = 1,
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 1,
    max_multimodal_len: int = 1024,
    dtype: str = "bfloat16",
    use_lora_plugin: str = None,
    lora_target_modules: List[str] = None,
    max_lora_rank: int = 64,
    lora_ckpt_list: List[str] = None,
):
    """Build mllama engine"""
    new_state_dict, config = convert_mllama_nemo_to_hf(checkpoint_path, processor_name)

    hf_model = MllamaForConditionalGeneration(config)
    hf_model = hf_model.to(torch.bfloat16)
    hf_model.load_state_dict(new_state_dict)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_model_path = os.path.join(tmp_dir, "hf_checkpoint")
        hf_model.save_pretrained(hf_model_path)
        del hf_model, new_state_dict

        build_mllama_visual_engine(
            os.path.join(model_dir, "visual_engine"),
            hf_model_path,
            vision_max_batch_size=vision_max_batch_size,
        )
        build_mllama_trtllm_engine(
            os.path.join(model_dir, "llm_engine"),
            hf_model_path,
            tensor_parallelism_size,
            max_input_len,
            max_output_len,
            max_batch_size,
            max_multimodal_len,
            dtype,
        )
