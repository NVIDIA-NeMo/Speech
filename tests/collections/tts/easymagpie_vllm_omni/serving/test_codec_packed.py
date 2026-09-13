# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packed import CODEC_STATE_ELEMENTS, CodecStateLayer, PackedEasyMagpieCodec
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata


def tiny_config() -> EasyMagpieCodecConfig:
    return EasyMagpieCodecConfig(
        input_dim=4,
        input_filters=8,
        hidden_filters=16,
        num_hidden_layers=2,
        pre_upsample_rates=[2],
        pre_upsample_filters=[8],
        resblock_upsample_rates=[2],
        resblock_upsample_filters=[4],
        num_codebooks=2,
        codebook_size=4,
        num_levels_per_group=[2, 2],
        frame_stacking_factor=2,
    )


def metadata(frames: int, *, has_initial: bool, device: torch.device | str = "cpu") -> Mamba1AttentionMetadata:
    result = Mamba1AttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=frames,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=1,
        has_initial_states_p=torch.tensor([has_initial], device=device),
        query_start_loc_p=torch.tensor([0, frames], dtype=torch.int32, device=device),
        num_computed_tokens_p=None,
        state_indices_tensor_p=torch.tensor([0], dtype=torch.int32, device=device),
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor([frames], dtype=torch.int32, device=device),
    )
    result.codec_max_query_len = frames
    result.codec_uniform = True
    return result


def decode_metadata(
    *,
    pages: tuple[int, ...] = (0,),
    device: torch.device | str = "cpu",
) -> Mamba1AttentionMetadata:
    num_decodes = len(pages)
    result = Mamba1AttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_reqs=num_decodes,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor(pages, dtype=torch.int32, device=device).unsqueeze(1),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.full((num_decodes,), 2, dtype=torch.int32, device=device),
    )
    result.codec_uniform = True
    return result


def mixed_metadata(prefill_frames: int, *, device: torch.device | str = "cpu") -> Mamba1AttentionMetadata:
    result = Mamba1AttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=prefill_frames,
        num_decodes=1,
        num_decode_tokens=1,
        num_reqs=2,
        has_initial_states_p=torch.tensor([False], device=device),
        query_start_loc_p=torch.tensor([0, prefill_frames], dtype=torch.int32, device=device),
        num_computed_tokens_p=None,
        state_indices_tensor_p=torch.tensor([1], dtype=torch.int32, device=device),
        state_indices_tensor_d=torch.tensor([[0]], dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor([2, prefill_frames], dtype=torch.int32, device=device),
    )
    result.codec_max_query_len = prefill_frames
    return result


def test_packed_profile_path_registers_state_layers() -> None:
    torch.manual_seed(11)
    config = tiny_config()
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        packed = PackedEasyMagpieCodec(config, dtype=torch.float32).eval()
    state_layers = [module for module in packed.modules() if isinstance(module, CodecStateLayer)]
    assert [layer.prefix for layer in state_layers] == [
        f"easymagpie_codec_state.{index}" for index in range(len(state_layers))
    ]

    codes = torch.randint(0, config.codebook_size, (1, 5, config.num_stacked_codebooks))
    with set_forward_context(None, vllm_config):
        actual = packed(codes.squeeze(0))
    assert actual.shape == (5 * config.samples_per_frame,)


def test_vllm_state_pages_match_full_decode() -> None:
    torch.manual_seed(17)
    config = tiny_config()
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        full = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="full").eval()
        packed = PackedEasyMagpieCodec(config, dtype=torch.float32).eval()
    packed.load_state_dict(full.state_dict(), strict=True)

    state_layers = [module for module in packed.modules() if isinstance(module, CodecStateLayer)]
    for layer in state_layers:
        layer.kv_cache = [torch.zeros((1, CODEC_STATE_ELEMENTS))]

    codes = torch.randint(0, config.codebook_size, (1, 6, config.num_stacked_codebooks))
    pieces = []
    offset = 0
    for chunk_index, chunk_size in enumerate((1, 2, 3)):
        layer_metadata = {layer.prefix: metadata(chunk_size, has_initial=chunk_index > 0) for layer in state_layers}
        with set_forward_context(layer_metadata, vllm_config):
            pieces.append(packed(codes[0, offset : offset + chunk_size]))
        offset += chunk_size

    actual = torch.cat(pieces)
    with set_forward_context(None, vllm_config):
        expected = full(codes.squeeze(0))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_one_frame_decode_metadata_matches_full_decode() -> None:
    torch.manual_seed(23)
    config = tiny_config()
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        full = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="full").eval()
        packed = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="one_frame").eval()
    packed.load_state_dict(full.state_dict(), strict=True)

    state_layers = [module for module in packed.modules() if isinstance(module, CodecStateLayer)]
    for layer in state_layers:
        layer.kv_cache = [torch.zeros((1, CODEC_STATE_ELEMENTS))]

    codes = torch.randint(0, config.codebook_size, (1, 5, config.num_stacked_codebooks))
    pieces = []
    for frame in range(codes.shape[1]):
        layer_metadata = {
            layer.prefix: (metadata(1, has_initial=False) if frame == 0 else decode_metadata())
            for layer in state_layers
        }
        with set_forward_context(layer_metadata, vllm_config):
            pieces.append(packed(codes[0, frame : frame + 1]))

    with set_forward_context(None, vllm_config):
        expected = full(codes.squeeze(0))
    torch.testing.assert_close(torch.cat(pieces), expected, atol=2e-5, rtol=2e-5)


def test_mixed_decode_and_prefill_batch() -> None:
    torch.manual_seed(27)
    config = tiny_config()
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        full = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="full").eval()
        packed = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="mixed").eval()
    packed.load_state_dict(full.state_dict(), strict=True)

    state_layers = [module for module in packed.modules() if isinstance(module, CodecStateLayer)]
    for layer in state_layers:
        layer.kv_cache = [torch.zeros((2, CODEC_STATE_ELEMENTS))]

    decode_codes = torch.randint(0, config.codebook_size, (1, 2, config.num_stacked_codebooks))
    prefill_codes = torch.randint(0, config.codebook_size, (1, 2, config.num_stacked_codebooks))
    initial_metadata = {layer.prefix: metadata(1, has_initial=False) for layer in state_layers}
    with set_forward_context(initial_metadata, vllm_config):
        packed(decode_codes[0, :1])

    batch_codes = torch.cat((decode_codes[0, 1:], prefill_codes[0]), dim=0)
    layer_metadata = {layer.prefix: mixed_metadata(2) for layer in state_layers}
    with set_forward_context(layer_metadata, vllm_config):
        actual = packed(batch_codes)

    with set_forward_context(None, vllm_config):
        decode_expected = full(decode_codes.squeeze(0))[config.samples_per_frame :]
        prefill_expected = full(prefill_codes.squeeze(0))
    torch.testing.assert_close(actual, torch.cat((decode_expected, prefill_expected)), atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton integration")
def test_cuda_packed_kernels_preserve_state_across_chunks() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    config = tiny_config()
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        full = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="full").to(device).eval()
        streamed = PackedEasyMagpieCodec(config, dtype=torch.float32, prefix="streamed").to(device).eval()
    streamed.load_state_dict(full.state_dict(), strict=True)

    full_layers = [module for module in full.modules() if isinstance(module, CodecStateLayer)]
    streamed_layers = [module for module in streamed.modules() if isinstance(module, CodecStateLayer)]
    for layer in full_layers + streamed_layers:
        layer.kv_cache = [torch.zeros((1, CODEC_STATE_ELEMENTS), device=device)]

    codes = torch.randint(
        0,
        config.codebook_size,
        (1, 5, config.num_stacked_codebooks),
        device=device,
    )
    full_metadata = {layer.prefix: metadata(5, has_initial=False, device=device) for layer in full_layers}
    with set_forward_context(full_metadata, vllm_config):
        expected = full(codes[0])

    pieces = []
    for frame in range(codes.shape[1]):
        chunk_metadata = {
            layer.prefix: (
                metadata(1, has_initial=False, device=device) if frame == 0 else decode_metadata(device=device)
            )
            for layer in streamed_layers
        }
        with set_forward_context(chunk_metadata, vllm_config):
            pieces.append(streamed(codes[0, frame : frame + 1]))

    torch.testing.assert_close(torch.cat(pieces), expected, atol=3e-5, rtol=3e-5)
