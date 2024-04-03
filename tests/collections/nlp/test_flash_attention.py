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

import random

import pytest
import torch
from megatron.core import ModelParallelConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.attention import CoreAttention
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.utils import build_attention_mask_3d
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    import flash_attn

    HAVE_FA = True
except (ImportError, ModuleNotFoundError):
    HAVE_FA = False

try:
    import pkg_resources
    import triton

    HAVE_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAVE_TRITON = False

try:
    import pynvml

    HAVE_PYNVML = True
except (ImportError, ModuleNotFoundError):
    HAVE_PYNVML = False


def HAVE_AMPERE_GPU():
    if HAVE_PYNVML:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_arch = pynvml.nvmlDeviceGetArchitecture(handle)
        pynvml.nvmlShutdown()
        return device_arch == pynvml.NVML_DEVICE_ARCH_AMPERE
    else:
        return False


@pytest.mark.pleasefixme
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not HAVE_APEX, reason="apex is not installed")
class TestFlashAttention:
    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return

        GPUS = 1
        TP_SIZE = GPUS
        PP_SIZE = 1
        MB_SIZE = 4
        GB_SIZE = 8
        SEED = 1234
        trainer = Trainer(strategy=NLPDDPStrategy(), devices=GPUS, accelerator='gpu', num_nodes=1, logger=None,)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            micro_batch_size=MB_SIZE,
            global_batch_size=GB_SIZE,
            seed=SEED,
            apex_transformer_log_level=30,
        )

    @pytest.fixture()
    def cfg(self):
        cfg = {
            'bz': random.randint(1, 7),
            'sq': random.randint(2, 7),
            'sk': random.randint(2, 7),
            'head': random.randint(1, 7),
            'layer_number': random.randint(1, 7),
            'device': torch.cuda.current_device(),
        }
        # flash attention requires head dimensions are multiples of 8
        head_dim = random.randint(1, 7) * 8
        cfg['hidden'] = cfg['head'] * head_dim

        return cfg

    @pytest.fixture()
    def model_parallel_config(self, cfg):
        config = ModelParallelConfig()
        return config

    @pytest.mark.pleasefixme
    @pytest.mark.skipif(not HAVE_FA, reason="flash-attention is not installed")
    @pytest.mark.unit
    def test_flash_self_attention(self, cfg, model_parallel_config):
        device = cfg['device']
        layer_number = cfg['layer_number']
        bz, sl, np, h = cfg['bz'], cfg['sq'], cfg['head'], cfg['hidden']
        hn = h // np

        q = torch.rand(sl, bz, np, hn, device=device).half()
        k = torch.rand(sl, bz, np, hn, device=device).half()
        v = torch.rand(sl, bz, np, hn, device=device).half()

        attention_mask_2d = torch.arange(sl, device=device).unsqueeze(0) < torch.randint(
            1, sl, (bz,), device=device
        ).unsqueeze(1)

        attention_mask_padding_3d = build_attention_mask_3d(
            source_mask=attention_mask_2d, target_mask=attention_mask_2d, attn_mask_type=AttnMaskType.padding
        ).unsqueeze(1)

        attention_mask_causal_3d = build_attention_mask_3d(
            source_mask=attention_mask_2d, target_mask=attention_mask_2d, attn_mask_type=AttnMaskType.causal
        ).unsqueeze(1)

        # Non-causal
        attention = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
        )

        attention_fa = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_padding_3d)
        out_fa = attention_fa(q, k, v, attention_mask_padding_3d)
        torch.testing.assert_close(out, out_fa)
        out_fa = attention_fa(q, k, v, ~attention_mask_2d)
        torch.testing.assert_close(out, out_fa)

        # Causal
        attention = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
        )

        attention_fa = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_causal_3d)
        out_fa = attention_fa(q, k, v, attention_mask_causal_3d)
        torch.testing.assert_close(out, out_fa)
        out_fa = attention_fa(q, k, v, ~attention_mask_2d)
        torch.testing.assert_close(out, out_fa)

    @pytest.mark.pleasefixme
    @pytest.mark.skipif(not HAVE_FA, reason="flash-attention is not installed")
    @pytest.mark.unit
    def test_flash_cross_attention(self, cfg, model_parallel_config):
        device = cfg['device']
        layer_number = cfg['layer_number']
        bz, sq, sk, np, h = cfg['bz'], cfg['sq'], cfg['sk'], cfg['head'], cfg['hidden']
        hn = h // np

        q = torch.rand(sq, bz, np, hn, device=device).half()
        k = torch.rand(sk, bz, np, hn, device=device).half()
        v = torch.rand(sk, bz, np, hn, device=device).half()

        attention_mask_2d_q = torch.arange(sq, device=device).unsqueeze(0) < torch.randint(
            1, sq, (bz,), device=device
        ).unsqueeze(1)

        attention_mask_2d_k = torch.arange(sk, device=device).unsqueeze(0) < torch.randint(
            1, sk, (bz,), device=device
        ).unsqueeze(1)

        attention_mask_padding_3d = build_attention_mask_3d(
            source_mask=attention_mask_2d_q, target_mask=attention_mask_2d_k, attn_mask_type=AttnMaskType.padding
        ).unsqueeze(1)

        attention = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
        )

        attention_fa = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_padding_3d)
        out_fa = attention_fa(q, k, v, attention_mask_padding_3d)
        torch.testing.assert_close(out, out_fa)

    @pytest.mark.skipif(not HAVE_FA, reason="flash-attention is not installed")
    @pytest.mark.skipif(not HAVE_TRITON, reason="triton is not installed")
    @pytest.mark.skipif(
        not HAVE_AMPERE_GPU(),
        reason="should only run on AMPERE GPU. Please see https://github.com/HazyResearch/flash-attention/issues/245",
    )
    @pytest.mark.unit
    def test_flash_self_attention_triton(self, cfg, model_parallel_config):
        device = cfg['device']
        layer_number = cfg['layer_number']
        bz, sl, np, h = cfg['bz'], cfg['sq'], cfg['head'], cfg['hidden']
        hn = h // np

        q = torch.rand(sl, bz, np, hn, device=device).half()
        k = torch.rand(sl, bz, np, hn, device=device).half()
        v = torch.rand(sl, bz, np, hn, device=device).half()

        attention_mask_2d = torch.arange(sl, device=device).unsqueeze(0) < torch.randint(
            1, sl, (bz,), device=device
        ).unsqueeze(1)

        attention_mask_padding_3d = build_attention_mask_3d(
            source_mask=attention_mask_2d, target_mask=attention_mask_2d, attn_mask_type=AttnMaskType.padding
        ).unsqueeze(1)

        attention_mask_causal_3d = build_attention_mask_3d(
            source_mask=attention_mask_2d, target_mask=attention_mask_2d, attn_mask_type=AttnMaskType.causal
        ).unsqueeze(1)

        attention_bias = torch.rand(bz, np, sl, sl, device=device)

        # Non-causal
        attention = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
        )

        attention_fa = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_padding_3d, relative_position_bias=attention_bias)
        out_fa = attention_fa(q, k, v, attention_mask_padding_3d, relative_position_bias=attention_bias)
        torch.testing.assert_close(out, out_fa, rtol=1e-3, atol=1e-3)
        out_fa = attention_fa(q, k, v, ~attention_mask_2d, relative_position_bias=attention_bias)
        torch.testing.assert_close(out, out_fa, rtol=1e-3, atol=1e-3)

        # Causal
        attention = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
        )

        attention_fa = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_causal_3d, relative_position_bias=attention_bias)
        out_fa = attention_fa(q, k, v, attention_mask_causal_3d, relative_position_bias=attention_bias)
        torch.testing.assert_close(out, out_fa, rtol=1e-3, atol=1e-3)
        out_fa = attention_fa(q, k, v, ~attention_mask_2d, relative_position_bias=attention_bias)
        torch.testing.assert_close(out, out_fa, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not HAVE_FA, reason="flash-attention is not installed")
    @pytest.mark.skipif(not HAVE_TRITON, reason="triton is not installed")
    @pytest.mark.skipif(
        not HAVE_AMPERE_GPU(),
        reason="should only run on AMPERE GPU. Please see https://github.com/HazyResearch/flash-attention/issues/245",
    )
    @pytest.mark.unit
    def test_flash_cross_attention_triton(self, cfg, model_parallel_config):
        device = cfg['device']
        layer_number = cfg['layer_number']
        bz, sq, sk, np, h = cfg['bz'], cfg['sq'], cfg['sk'], cfg['head'], cfg['hidden']
        hn = h // np

        q = torch.rand(sq, bz, np, hn, device=device).half()
        k = torch.rand(sk, bz, np, hn, device=device).half()
        v = torch.rand(sk, bz, np, hn, device=device).half()

        attention_mask_2d_q = torch.arange(sq, device=device).unsqueeze(0) < torch.randint(
            1, sq, (bz,), device=device
        ).unsqueeze(1)

        attention_mask_2d_k = torch.arange(sk, device=device).unsqueeze(0) < torch.randint(
            1, sk, (bz,), device=device
        ).unsqueeze(1)

        attention_mask_padding_3d = build_attention_mask_3d(
            source_mask=attention_mask_2d_q, target_mask=attention_mask_2d_k, attn_mask_type=AttnMaskType.padding
        ).unsqueeze(1)

        attention_bias = torch.rand(bz, np, sq, sk, device=device)

        attention = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
        )

        attention_fa = CoreAttention(
            config=model_parallel_config,
            layer_number=layer_number,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_padding_3d, relative_position_bias=attention_bias)
        out_fa = attention_fa(q, k, v, attention_mask_padding_3d, relative_position_bias=attention_bias)
        torch.testing.assert_close(out, out_fa, rtol=1e-3, atol=1e-3)
