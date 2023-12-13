# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Vision Transformer(VIT) model."""

import math
from functools import partial

import einops
import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.vision.modules.common.megatron.vision_transformer import ParallelVisionTransformer


class DropPatch(MegatronModule):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, class_token_length=8, exclude_cls_tokens=True):
        assert 0 <= prob < 1.0
        super(DropPatch, self).__init__()
        self.prob = prob
        self.class_token_length = class_token_length
        self.exclude_cls_tokens = exclude_cls_tokens  # exclude CLS token

    def __call__(self, x):
        if self.prob == 0.0 or not self.training:
            return x

        class_token_length = self.class_token_length
        if self.exclude_cls_tokens:
            cls_tokens, x = x[:, :class_token_length], x[:, class_token_length:]

        batch, num_tokens, _ = x.shape
        device = x.device

        batch_indices = torch.arange(batch, device=device)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens, device=device)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_cls_tokens:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if x >= 0:
        sr = math.sqrt(x)
        return int(sr) * int(sr) == x
    return False


def twod_interpolate_position_embeddings_hook(
    model_cfg,
    class_token_present,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    num_patches_per_dim_h = model_cfg.img_h // model_cfg.patch_dim
    num_patches_per_dim_w = model_cfg.img_w // model_cfg.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = model_cfg.hidden_size
    class_token_length = model_cfg.get("class_token_length", 8)

    key = prefix + "weight"

    assert key in state_dict, f"{key} not in {state_dict.keys()}"
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - class_token_length)
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - class_token_length if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:class_token_length, :]
            input_param_grid = input_param[class_token_length:, :]
        else:
            input_param_tok = torch.zeros(class_token_length, hidden_size, device=input_param.device)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:
            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape((1, -1, gs_input, gs_input))
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(input_param_grid, scale_factor=scale_factor, mode="bilinear")

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert input_param.shape[0] == num_tok_output and input_param.shape[1] == hidden_size

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class VitBackbone(MegatronModule):
    """Vision Transformer Model."""

    def __init__(
        self,
        model_cfg,
        model_parallel_config,
        init_method=None,
        scaled_init_method=None,
        pre_process=True,
        post_process=True,
        class_token=True,
        single_token_output=False,
    ):
        super(VitBackbone, self).__init__(share_token_embeddings=False)

        self.fp16_lm_cross_entropy = model_cfg.fp16_lm_cross_entropy
        num_layers = model_cfg.num_layers
        init_method_std = model_cfg.init_method_std
        if init_method is None:
            init_method = init_method_normal(init_method_std)
        if scaled_init_method is None:
            scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.hidden_size = model_cfg.hidden_size
        self.patch_dim = model_cfg.patch_dim
        self.img_h = model_cfg.img_h
        self.img_w = model_cfg.img_w
        self.single_token_output = single_token_output
        self.drop_patch_rate = model_cfg.get("drop_patch_rate", 0.0)
        self.drop_path_rate = model_cfg.get("drop_path_rate", 0.0)
        preprocess_layernorm = model_cfg.get("preprocess_layernorm", False)

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        class_token_length = model_cfg.get("class_token_length", 8)
        self.seq_length = self.num_patches + (class_token_length if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * model_cfg.num_channels
        self.input_tensor = None
        self.position_ids = None
        self.preprocess_layernorm = None

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(torch.randn(1, class_token_length, self.hidden_size))
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(self.flatten_dim, self.hidden_size)

            # embedding
            self.position_embedding_type = model_cfg.get("position_embedding_type", "learned_absolute")

            if self.position_embedding_type == "learned_absolute":
                self.position_embeddings = torch.nn.Embedding(self.seq_length, self.hidden_size)
                init_method_normal(model_cfg.init_method_std)(self.position_embeddings.weight)

                class_token_present = self.class_token
                self.position_embeddings._register_load_state_dict_pre_hook(
                    partial(twod_interpolate_position_embeddings_hook, model_cfg, class_token_present)
                )
            elif self.position_embedding_type == "learned_parameters":
                self.position_embeddings = torch.nn.Parameter(torch.empty(self.seq_length, self.hidden_size))
                init_method_normal(model_cfg.init_method_std)(self.position_embeddings)
            else:
                raise ValueError(f"Unrecognized positional embedding type {self.position_embedding_type}!")

            self.embedding_dropout = torch.nn.Dropout(model_cfg.hidden_dropout)
            self.drop_patch = DropPatch(
                self.drop_patch_rate, class_token_length=class_token_length, exclude_cls_tokens=self.class_token
            )

            if preprocess_layernorm:
                self.preprocess_layernorm = get_layer_norm(
                    model_cfg.hidden_size,
                    model_cfg.layernorm_epsilon,
                    model_cfg.persist_layer_norm,
                    sequence_parallel=model_cfg.sequence_parallel,
                )

        self.transformer = ParallelVisionTransformer(
            config=model_parallel_config,
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            num_layers=model_cfg.num_layers,
            hidden_size=model_cfg.hidden_size,
            num_attention_heads=model_cfg.num_attention_heads,
            apply_query_key_layer_scaling=model_cfg.apply_query_key_layer_scaling,
            kv_channels=model_cfg.kv_channels,
            ffn_hidden_size=model_cfg.ffn_hidden_size,
            # self_attn_mask_type=self.encoder_attn_mask_type, # TODO (yuya)
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=model_cfg.precision,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            activations_checkpoint_method=model_cfg.activations_checkpoint_method,
            activations_checkpoint_num_layers=model_cfg.activations_checkpoint_num_layers,
            normalization=model_cfg.normalization,
            layernorm_epsilon=model_cfg.layernorm_epsilon,
            hidden_dropout=model_cfg.hidden_dropout,
            attention_dropout=model_cfg.attention_dropout,
            drop_path_rate=model_cfg.drop_path_rate,
            bias_activation_fusion=model_cfg.get("bias_activation_fusion", False),
            persist_layer_norm=model_cfg.persist_layer_norm,
            openai_gelu=model_cfg.openai_gelu,
            onnx_safe=model_cfg.onnx_safe,
            masked_softmax_fusion=model_cfg.masked_softmax_fusion,
            megatron_legacy=model_cfg.megatron_legacy,
            activations_checkpoint_granularity=model_cfg.activations_checkpoint_granularity,
            activation=model_cfg.get('activation', 'gelu'),
            ub_tp_comm_overlap=model_cfg.get('ub_tp_comm_overlap', False),
            use_flash_attention=model_cfg.get('use_flash_attention', False),
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):

        if self.pre_process:
            rearranged_input = einops.rearrange(
                input, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_dim, p2=self.patch_dim,
            )

            # [b num_patch patch_dim*patch_dim*c] ->  [b, s, h]; s:=num_patch, h:=hidden
            encoder_output = self.linear_encoder(rearranged_input)

            concatenated_tokens = encoder_output
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            if self.position_embedding_type == "learned_absolute":
                token_embeddings = concatenated_tokens + self.position_embeddings(
                    self.position_ids[:, : concatenated_tokens.shape[1]]
                )
            elif self.position_embedding_type == "learned_parameters":
                token_embeddings = concatenated_tokens + self.position_embeddings
            else:
                raise ValueError(f"Unrecognized position embedding type: {self.position_embedding_type}.")

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            token_embeddings = self.drop_patch(token_embeddings)

            if self.preprocess_layernorm is not None:
                token_embeddings = self.preprocess_layernorm(token_embeddings)

            # [b s h] => [s b h]
            token_embeddings = token_embeddings.transpose(0, 1).contiguous()
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        # 0 represents masking, 1 represents not masking
        # attention_mask = torch.zeros(
        #     [1, 1, hidden_states.shape[0], hidden_states.shape[0]],
        #     device=hidden_states.device,
        #     dtype=torch.bool,
        # )
        hidden_states = self.transformer(hidden_states, None)

        if self.post_process:
            # [s b h] => [b s h]
            if self.single_token_output:
                hidden_states = hidden_states[0]
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()

        return hidden_states
