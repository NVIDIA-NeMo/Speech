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
Usage example:
  python /opt/NeMo/examples/multimodal/foundation/clip/convert_external_clip_to_nemo.py
    --arch=ViT-H-14
    --version=laion2b_s32b_b79k
    --hparams_file=path/to/saved.yaml
    --nemo_file_path=open_clip.nemo

If converting from OpenCLIP, specify the architecture (`arch`) and version (`version`) from the OpenCLIP model list (https://github.com/mlfoundations/open_clip#usage).

If converting from Hugging Face, set the version to `huggingface` and the architecture (`arch`) to the Hugging Face model name (e.g., `yuvalkirstain/PickScore_v1`).

Additionally, provide a NeMo hparams file with the correct model architecture arguments. Refer to examples/multimodal/foundation/clip/conf/megatron_clip_config.yaml.
"""

import os
from argparse import ArgumentParser

import einops
import open_clip
import torch
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer
from transformers import CLIPModel

from nemo.collections.multimodal.models.vision_language_foundation.clip import MegatronCLIPModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--arch", type=str, default="ViT-H-14")
    parser.add_argument("--version", type=str, default="laion2b_s32b_b79k")

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--gpus_per_node", type=int, required=False, default=1)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument(
        "--pipeline_model_parallel_split_rank",
        type=int,
        required=False,
        default=None,
        help="If pipeline parallel size > 1, this is the rank at which the encoder ends and the decoder begins.",
    )
    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")

    args = parser.parse_args()
    return args


def mapping_openclip_state_dict(open_model):
    open_state_dict = open_model.state_dict()
    key_mapping = {
        "positional_embedding": "text_encoder.language_model.embedding.position_embeddings",
        "token_embedding.weight": "text_encoder.language_model.embedding.word_embeddings.weight",
        "ln_final.weight": "text_encoder.language_model.encoder.final_layernorm.weight",
        "ln_final.bias": "text_encoder.language_model.encoder.final_layernorm.bias",
        "text_projection": "text_encoder.head.weight",
    }
    layer_mapping = {
        ".ln_1.weight": ".input_layernorm.weight",
        ".ln_1.bias": ".input_layernorm.bias",
        ".attn.in_proj_weight": ".self_attention.query_key_value.weight",
        ".attn.in_proj_bias": ".self_attention.query_key_value.bias",
        ".attn.out_proj.weight": ".self_attention.dense.weight",
        ".attn.out_proj.bias": ".self_attention.dense.bias",
        ".ln_2.weight": ".post_attention_layernorm.weight",
        ".ln_2.bias": ".post_attention_layernorm.bias",
        ".mlp.c_fc.weight": ".mlp.dense_h_to_4h.weight",
        ".mlp.c_fc.bias": ".mlp.dense_h_to_4h.bias",
        ".mlp.c_proj.weight": ".mlp.dense_4h_to_h.weight",
        ".mlp.c_proj.bias": ".mlp.dense_4h_to_h.bias",
        ".ln_pre.weight": ".preprocess_layernorm.weight",
        ".ln_pre.bias": ".preprocess_layernorm.bias",
        ".ln_post.weight": ".transformer.final_layernorm.weight",
        ".ln_post.bias": ".transformer.final_layernorm.bias",
        ".positional_embedding": ".position_embeddings",
        ".backbone.proj": ".head.weight",
        ".class_embedding": ".cls_token",
        ".backbone.conv1.weight": ".backbone.linear_encoder.weight",
    }

    nemo_state_dict = {}
    for key in open_state_dict.keys():
        if key.startswith("transformer.resblocks."):
            key_ = key.replace("transformer.resblocks.", "text_encoder.language_model.encoder.layers.")
        elif key.startswith("visual.transformer.resblocks."):
            key_ = key.replace("visual.transformer.resblocks.", "vision_encoder.backbone.transformer.layers.")
        elif key.startswith('visual.'):
            key_ = key.replace("visual.", "vision_encoder.backbone.")
        else:
            key_ = key
        for pat in key_mapping:
            if key_ == pat:
                key_ = key_.replace(pat, key_mapping[pat])
        for pat in layer_mapping:
            if key_.endswith(pat):
                key_ = key_[: -len(pat)] + layer_mapping[pat]
                break
        nemo_state_dict[key_] = open_state_dict[key]

    nemo_state_dict["text_encoder.head.weight"] = nemo_state_dict["text_encoder.head.weight"].T
    nemo_state_dict["vision_encoder.head.weight"] = nemo_state_dict["vision_encoder.head.weight"].T
    nemo_state_dict["vision_encoder.backbone.cls_token"] = nemo_state_dict[
        "vision_encoder.backbone.cls_token"
    ].reshape(1, 1, -1)
    w = nemo_state_dict["vision_encoder.backbone.linear_encoder.weight"]
    nemo_state_dict["vision_encoder.backbone.linear_encoder.weight"] = einops.rearrange(w, "b c p1 p2 -> b (p1 p2 c)",)
    nemo_state_dict["vision_encoder.backbone.linear_encoder.bias"] = torch.zeros(w.shape[0])

    return nemo_state_dict


def mapping_hf_state_dict(hf_model):
    hf_state_dict = hf_model.state_dict()
    key_mapping = {
        "text_projection.weight": "text_encoder.head.weight",
        "visual_projection.weight": "vision_encoder.head.weight",
    }

    layer_mapping = {
        ".layer_norm1.weight": ".input_layernorm.weight",
        ".layer_norm1.bias": ".input_layernorm.bias",
        ".self_attn.out_proj.weight": ".self_attention.dense.weight",
        ".self_attn.out_proj.bias": ".self_attention.dense.bias",
        ".layer_norm2.weight": ".post_attention_layernorm.weight",
        ".layer_norm2.bias": ".post_attention_layernorm.bias",
        ".mlp.fc1.weight": ".mlp.dense_h_to_4h.weight",
        ".mlp.fc1.bias": ".mlp.dense_h_to_4h.bias",
        ".mlp.fc2.weight": ".mlp.dense_4h_to_h.weight",
        ".mlp.fc2.bias": ".mlp.dense_4h_to_h.bias",
        ".pre_layrnorm.weight": ".preprocess_layernorm.weight",
        ".pre_layrnorm.bias": ".preprocess_layernorm.bias",
        ".post_layernorm.weight": ".transformer.final_layernorm.weight",
        ".post_layernorm.bias": ".transformer.final_layernorm.bias",
        ".backbone.embeddings.position_embedding.weight": ".backbone.position_embeddings",
        ".language_model.embeddings.position_embedding.weight": ".language_model.embedding.position_embeddings",
        ".embeddings.class_embedding": ".cls_token",
        ".backbone.embeddings.patch_embedding.weight": ".backbone.linear_encoder.weight",
        ".final_layer_norm.weight": ".encoder.final_layernorm.weight",
        ".final_layer_norm.bias": ".encoder.final_layernorm.bias",
        ".embeddings.token_embedding.weight": ".embedding.word_embeddings.weight",
    }

    nemo_state_dict = {}
    for key in hf_state_dict.keys():
        if key.startswith("text_model.encoder.layers"):
            key_ = key.replace("text_model.encoder.layers", "text_encoder.language_model.encoder.layers")
        elif key.startswith("vision_model.encoder.layers"):
            key_ = key.replace("vision_model.encoder.layers", "vision_encoder.backbone.transformer.layers")
        elif key.startswith('vision_model.'):
            key_ = key.replace("vision_model.", "vision_encoder.backbone.")
        elif key.startswith('text_model.'):
            key_ = key.replace('text_model.', 'text_encoder.language_model.')
        else:
            key_ = key
        for pat in key_mapping:
            if key_ == pat:
                key_ = key_.replace(pat, key_mapping[pat])
        for pat in layer_mapping:
            if key_.endswith(pat):
                key_ = key_[: -len(pat)] + layer_mapping[pat]
                break
        if 'q_proj' in key_:
            key_k = key.replace('q_proj', 'k_proj')
            key_v = key.replace('q_proj', 'v_proj')
            key_new = key_.replace('self_attn.q_proj', 'self_attention.query_key_value')
            value_new = torch.concat((hf_state_dict[key], hf_state_dict[key_k], hf_state_dict[key_v]), dim=0)
            nemo_state_dict[key_new] = value_new
        elif not ('k_proj' in key_ or 'v_proj' in key_ or 'position_ids' in key_):
            nemo_state_dict[key_] = hf_state_dict[key]

    nemo_state_dict["vision_encoder.backbone.cls_token"] = nemo_state_dict[
        "vision_encoder.backbone.cls_token"
    ].reshape(1, 1, -1)
    w = nemo_state_dict["vision_encoder.backbone.linear_encoder.weight"]
    nemo_state_dict["vision_encoder.backbone.linear_encoder.weight"] = einops.rearrange(w, "b c p1 p2 -> b (p1 p2 c)",)
    nemo_state_dict["vision_encoder.backbone.linear_encoder.bias"] = torch.zeros(w.shape[0])

    return nemo_state_dict


def convert(local_rank, rank, world_size, args):
    app_state = AppState()
    app_state.data_parallel_rank = 0
    num_nodes = world_size // args.gpus_per_node
    if args.bcp:
        trainer = Trainer(
            devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu', plugins=[TorchElasticEnvironment()]
        )
    else:
        trainer = Trainer(devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu')

    app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = args.tensor_model_parallel_size

    # no use atm, use to split ranks in encoder/decoder models.
    if args.pipeline_model_parallel_size > 1 and args.model_type in []:
        if args.pipeline_model_parallel_split_rank is not None:
            app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_split_rank
        else:
            if args.pipeline_model_parallel_size % 2 != 0:
                raise ValueError(
                    f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be even if split rank is not specified."
                )
            else:
                # If split rank is not set, then we set it to be pipeline_model_parallel_size // 2 - this is because in most cases we have the same number of enc/dec layers.
                app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_size // 2
    else:
        app_state.pipeline_model_parallel_split_rank = None

    app_state.model_parallel_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
    )

    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()

    cfg = OmegaConf.load(args.hparams_file)
    model = MegatronCLIPModel(cfg.model, trainer)

    if args.version == "huggingface":
        hf_model = CLIPModel.from_pretrained(args.arch)
        state_dict = mapping_hf_state_dict(hf_model)
    else:
        open_model, _, _ = open_clip.create_model_and_transforms(args.arch, pretrained=args.version)
        state_dict = mapping_openclip_state_dict(open_model)

    model.model.load_state_dict(state_dict)

    model._save_restore_connector = NLPSaveRestoreConnector()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(args.nemo_file_path)

    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


if __name__ == '__main__':
    args = get_args()
    local_rank, rank, world_size = initialize_distributed(args)
    convert(local_rank, rank, world_size, args)
