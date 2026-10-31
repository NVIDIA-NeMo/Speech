# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Transformer decoder modules: configurable blocks (embedding, norm, self- and cross-attention, feed-forward),
and the key-value cache for incremental decoding."""

import functools
import math

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_decoders import TransformerDecoderKeyValueCache
from nemo.collections.asr.modules.transformer.transformer_modules import (
    FixedPositionalEncoding,
    GatedPositionWiseFF,
    MultiHeadAttention,
    PositionWiseFF,
    RMSNorm,
    TransformerEmbedding,
)
from nemo.collections.asr.modules.transformer.transformer_utils import get_nemo_transformer


def _composed_decoder(**overrides):
    options = dict(
        vocab_size=11,
        hidden_size=16,
        num_layers=2,
        inner_size=24,
        num_attention_heads=2,
        max_sequence_length=32,
        num_token_types=0,
        pre_ln=True,
        embedding=functools.partial(
            TransformerEmbedding,
            scale_embeddings=True,
            positional_encoding_layout="tensor2tensor",
            use_layer_norm=False,
        ),
        norm=functools.partial(RMSNorm, eps=1e-6),
        self_attn=functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0),
        feed_forward=functools.partial(GatedPositionWiseFF, hidden_act="swish"),
    )
    options.update(overrides)
    return TransformerDecoderNM(**options).eval()


class TestTransformerDecoderBlockComposition:
    @pytest.mark.unit
    def test_blocks_from_partials(self):
        decoder = _composed_decoder()
        block = decoder.decoder.layers[0]
        assert isinstance(block.layer_norm_1, RMSNorm) and block.layer_norm_1.eps == 1e-6
        assert isinstance(decoder.decoder.final_layer_norm, RMSNorm)
        assert block.first_sub_layer.query_net.bias is None and block.first_sub_layer.rotary_pos_enc is not None
        # the cross-attention keeps the default block
        assert block.second_sub_layer.key_net.bias is not None and block.second_sub_layer.rotary_pos_enc is None
        assert block.third_sub_layer.dense_in.weight.shape == (48, 16) and block.third_sub_layer.dense_in.bias is None
        assert decoder.embedding.layer_norm is None and decoder.embedding.embedding_scale == math.sqrt(16)
        assert decoder.embedding.token_embedding.num_embeddings == 11

    @pytest.mark.unit
    @pytest.mark.parametrize("rotary", [True, False])
    def test_incremental_decoding_matches_full_pass(self, rotary):
        torch.manual_seed(0)
        overrides = {} if rotary else dict(self_attn=functools.partial(MultiHeadAttention, use_bias=False))
        decoder = _composed_decoder(**overrides)
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.3)
        input_ids = torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0]])
        encoder_states = torch.randn(2, 9, 16)
        encoder_mask = torch.ones(2, 9)
        with torch.no_grad():
            full, _ = decoder.decoder(
                decoder.embedding(input_ids), torch.ones_like(input_ids), encoder_states, encoder_mask
            )
            mems = None
            for step in range(input_ids.shape[1]):
                embedded = decoder.embedding(input_ids[:, step : step + 1], start_pos=step)
                mems, _ = decoder.decoder(
                    embedded, torch.ones(2, 1), encoder_states, encoder_mask, mems, return_mems=True
                )
        torch.testing.assert_close(mems[-1], full, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_blocks_from_hydra_sections(self):
        config = DictConfig(
            {
                "vocab_size": 11,
                "hidden_size": 16,
                "num_layers": 1,
                "inner_size": 24,
                "num_attention_heads": 2,
                "max_sequence_length": 32,
                "num_token_types": 0,
                "pre_ln": True,
                "norm": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.RMSNorm",
                    "_partial_": True,
                    "eps": 1e-6,
                },
                "self_attn": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.MultiHeadAttention",
                    "_partial_": True,
                    "use_bias": False,
                    "rotary_base": 10000.0,
                },
                "feed_forward": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.GatedPositionWiseFF",
                    "_partial_": True,
                    "hidden_act": "swish",
                },
            }
        )
        decoder = get_nemo_transformer(config_dict=config, encoder=False)
        block = decoder.decoder.layers[0]
        assert isinstance(block.layer_norm_1, RMSNorm) and block.layer_norm_1.eps == 1e-6
        assert block.first_sub_layer.query_net.bias is None and block.first_sub_layer.rotary_pos_enc is not None
        assert type(block.third_sub_layer) is GatedPositionWiseFF
        assert block.third_sub_layer.dense_in.weight.shape == (48, 16) and block.third_sub_layer.dense_in.bias is None
        assert isinstance(block.third_sub_layer.act_fn, torch.nn.SiLU)
        assert isinstance(decoder.embedding.layer_norm, torch.nn.LayerNorm)

    @pytest.mark.unit
    def test_rms_norm(self):
        torch.manual_seed(0)
        norm = RMSNorm(8, eps=1e-6)
        norm.weight.data.normal_()
        x = torch.randn(3, 8)
        torch.testing.assert_close(norm(x), torch.nn.functional.rms_norm(x, (8,), norm.weight, eps=1e-6))
        assert list(norm.state_dict()) == ["weight"]

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rms_norm_half_precision(self, dtype):
        # the mean square must not overflow: 1000 squared is outside the float16 range
        norm = RMSNorm(8, eps=1e-6).to(dtype)
        large = torch.full((2, 8), 1000.0, dtype=dtype)
        torch.testing.assert_close(norm(large), torch.ones_like(large))
        # forward and backward against the float32 computation; mixed signs, a zero row, a non-unit weight
        torch.manual_seed(0)
        x32 = torch.randn(5, 8) * 300.0
        x32[2] = 0.0
        w32 = torch.randn(8)
        norm.weight.data = w32.to(dtype)
        x = x32.to(dtype).requires_grad_()
        out = norm(x)
        assert out.dtype == dtype
        out.sum().backward()
        reference = RMSNorm(8, eps=1e-6)
        reference.weight.data = w32.clone()
        x_ref = x32.clone().requires_grad_()
        expected = reference(x_ref)
        expected.sum().backward()
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(x.grad.float(), x_ref.grad, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(norm.weight.grad.float(), reference.weight.grad, rtol=2e-2, atol=5e-2)

    @pytest.mark.unit
    def test_export_covers_rotary_positions_beyond_the_example(self, tmp_path):
        # the exported graph holds the rotary table as a constant,
        # so it must cover every position the decoder supports, not only the example's key length
        decoder = _composed_decoder(max_sequence_length=8)
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.3)

        def example(length):
            torch.manual_seed(length)
            input_ids = torch.randint(0, 11, (1, length))
            decoder_mask = torch.ones(1, length, dtype=torch.int64)
            encoder_states = torch.randn(1, 9, 16)
            encoder_mask = torch.ones(1, 9, dtype=torch.int64)
            decoder_mems = torch.randn(1, decoder.num_states, length - 1, 16)
            return input_ids, decoder_mask, encoder_states, encoder_mask, decoder_mems

        path = str(tmp_path / "decoder.ts")
        decoder.export(path, input_example=example(4), check_trace=False)
        exported = torch.jit.load(path)
        for length in (4, 5, 8):
            input_ids, decoder_mask, encoder_states, encoder_mask, decoder_mems = example(length)
            with torch.no_grad():
                # export switched the module to returning the states of every layer, as the graph does
                expected = decoder(
                    input_ids=input_ids,
                    decoder_mask=decoder_mask,
                    encoder_embeddings=encoder_states,
                    encoder_mask=encoder_mask,
                    decoder_mems=decoder_mems,
                )
            torch.testing.assert_close(
                exported(input_ids, decoder_mask, encoder_states, encoder_mask, decoder_mems), expected
            )

    @pytest.mark.unit
    def test_gated_feed_forward(self):
        torch.manual_seed(0)
        ff = GatedPositionWiseFF(16, 24)
        assert ff.dense_in.bias is None and ff.dense_in.weight.shape == (48, 16)
        x = torch.randn(3, 16)
        activated, gate = ff.dense_in(x).chunk(2, dim=-1)
        torch.testing.assert_close(ff(x), ff.dense_out(torch.nn.functional.silu(activated) * gate))
        # both activations are free: GEGLU-style first half with a gated second half, from registry names
        geglu = GatedPositionWiseFF(16, 24, hidden_act="gelu", gate_act="relu", use_bias=True)
        assert isinstance(geglu.act_fn, torch.nn.GELU) and isinstance(geglu.gate_act_fn, torch.nn.ReLU)
        assert geglu.dense_in.bias is not None
        # the plain network takes its activation from the registry as well
        assert isinstance(PositionWiseFF(16, 24, hidden_act="gelu").act_fn, torch.nn.GELU)
        with pytest.raises(ValueError, match="activation"):
            PositionWiseFF(16, 24, hidden_act="tanh")

    @pytest.mark.unit
    def test_tensor2tensor_positional_table(self):
        table = FixedPositionalEncoding(8, max_sequence_length=4, layout="tensor2tensor", scale=1.0).pos_enc
        positions = torch.arange(4.0)[:, None]
        frequencies = torch.exp(-math.log(10000.0) / 3 * torch.arange(4.0))
        torch.testing.assert_close(table[:, :4], torch.sin(positions * frequencies))
        torch.testing.assert_close(table[:, 4:], torch.cos(positions * frequencies))
        default = FixedPositionalEncoding(8, max_sequence_length=4).pos_enc
        torch.testing.assert_close(default[1, 0], torch.sin(torch.tensor(1.0)) / math.sqrt(8))

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        decoder = TransformerDecoderNM(
            vocab_size=8, hidden_size=16, num_layers=1, inner_size=24, num_attention_heads=2
        )
        block = decoder.decoder.layers[0]
        assert type(block.layer_norm_1) is torch.nn.LayerNorm and block.layer_norm_1.eps == 1e-5
        assert block.first_sub_layer.query_net.bias is not None and block.first_sub_layer.rotary_pos_enc is None
        assert type(block.third_sub_layer) is PositionWiseFF
        assert isinstance(block.third_sub_layer.act_fn, torch.nn.ReLU)
        assert block.third_sub_layer.dense_in.weight.shape == (24, 16)
        assert (
            isinstance(decoder.embedding.layer_norm, torch.nn.LayerNorm) and decoder.embedding.embedding_scale is None
        )
        assert decoder.decoder.final_layer_norm is None


HIDDEN, HEADS, VOCAB, FRAMES, BOS = 16, 2, 11, 9, 10
TOLERANCE = dict(atol=1e-5, rtol=1e-5)


def _cache_decoder(rotary=True, pre_ln=True, seed=0, **overrides):
    """A small random decoder: the composed form (rotary, RMSNorm, SwiGLU) or the default blocks."""
    torch.manual_seed(seed)
    blocks = {}
    if rotary:
        blocks["self_attn"] = functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0)
    if pre_ln:
        blocks.update(
            embedding=functools.partial(
                TransformerEmbedding,
                scale_embeddings=True,
                positional_encoding_layout="tensor2tensor",
                use_layer_norm=False,
            ),
            norm=functools.partial(RMSNorm, eps=1e-6),
            feed_forward=functools.partial(GatedPositionWiseFF, hidden_act="swish"),
        )
    blocks.update(overrides)
    decoder = TransformerDecoderNM(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=2,
        inner_size=24,
        num_attention_heads=HEADS,
        max_sequence_length=32,
        num_token_types=0,
        pre_ln=pre_ln,
        **blocks,
    ).eval()
    for parameter in decoder.parameters():
        parameter.data.normal_(0, 0.3)
    return decoder


def _encoder(lengths, seed=1):
    torch.manual_seed(seed)
    states = torch.randn(len(lengths), FRAMES, HIDDEN)
    mask = (torch.arange(FRAMES)[None, :] < torch.tensor(lengths)[:, None]).float()
    return states, mask


def _full_pass(decoder, input_ids, decoder_mask, encoder_states, encoder_mask):
    return decoder.decoder(decoder.embedding(input_ids), decoder_mask, encoder_states, encoder_mask)


class TestTransformerDecoderKeyValueCache:
    """The key-value cache reproduces the full pass and the memory-list path of the decoder."""

    @pytest.mark.unit
    @pytest.mark.parametrize("rotary,pre_ln", [(True, True), (False, True), (False, False)])
    def test_steps_match_full_pass(self, rotary, pre_ln):
        decoder = _cache_decoder(rotary, pre_ln)
        input_ids = torch.tensor([[BOS, 1, 5, 7, 2, 3], [BOS, 3, 4, 0, 0, 0]])  # pads after an ended hypothesis
        decoder_mask = (input_ids != 0).float()
        encoder_states, encoder_mask = _encoder([9, 6])
        with torch.no_grad():
            full, full_xatt = _full_pass(decoder, input_ids, decoder_mask, encoder_states, encoder_mask)
            cache = decoder.decoder.create_kv_cache(encoder_states, encoder_mask, capacity=8)
            outputs, xatts = [], []
            # a two-token prompt, then one token per step
            for start, end in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
                embedded = decoder.embedding(input_ids[:, start:end], start_pos=start)
                states, xatt = decoder.decoder(embedded, decoder_mask[:, start:end], kv_cache=cache)
                outputs.append(states)
                xatts.append(xatt)
        assert int(cache.length) == 6 and cache.capacity == 8 and cache.batch_size == 2
        torch.testing.assert_close(torch.cat(outputs, dim=1), full, **TOLERANCE)
        for layer in range(len(full_xatt)):
            torch.testing.assert_close(torch.cat([x[layer] for x in xatts], dim=2), full_xatt[layer], **TOLERANCE)

    @pytest.mark.unit
    def test_matches_memory_path(self):
        decoder = _cache_decoder()
        input_ids = torch.tensor([[BOS, 1, 5, 7, 2], [BOS, 3, 4, 6, 8]])
        encoder_states, encoder_mask = _encoder([9, 6])
        with torch.no_grad():
            cache = decoder.decoder.create_kv_cache(encoder_states, encoder_mask, capacity=5)
            mems = None
            for step in range(input_ids.shape[1]):
                embedded = decoder.embedding(input_ids[:, step : step + 1], start_pos=step)
                mask = torch.ones(2, 1)
                mems, mems_xatt = decoder.decoder(
                    embedded, mask, encoder_states, encoder_mask, mems, return_mems=True, return_mems_as_list=False
                )
                cached, cached_xatt = decoder.decoder(embedded, mask, kv_cache=cache, return_mems=True)
                assert len(cached) == mems.shape[0] == decoder.num_states
                torch.testing.assert_close(torch.stack(cached), mems[:, :, -1:], **TOLERANCE)
                for layer in range(len(mems_xatt)):
                    torch.testing.assert_close(cached_xatt[layer], mems_xatt[layer], **TOLERANCE)

    @pytest.mark.unit
    def test_hypotheses_repeat_reorder_and_select(self):
        decoder = _cache_decoder()
        encoder_states, encoder_mask = _encoder([9, 6])
        bos = torch.full((2, 1), BOS)
        first = torch.tensor([[1], [2], [3], [4], [5], [6]])
        backrefs = torch.tensor([1, 1, 0, 5, 3, 3])  # beam search back references within each source sequence
        second = torch.tensor([[7], [8], [9], [1], [2], [3]])
        with torch.no_grad():
            cache = decoder.decoder.create_kv_cache(encoder_states, encoder_mask, capacity=5)
            decoder.decoder(decoder.embedding(bos), None, kv_cache=cache)
            beams = cache.repeat_interleave(3)
            assert beams.batch_size == 6 and int(beams.length) == 1 and int(cache.length) == 1
            decoder.decoder(decoder.embedding(first, start_pos=1), None, kv_cache=beams)
            beams.reorder_hypotheses(backrefs)
            output, _ = decoder.decoder(decoder.embedding(second, start_pos=2), None, kv_cache=beams)
            prefixes = torch.cat([bos.repeat_interleave(3, dim=0), first[backrefs], second], dim=1)
            full, _ = _full_pass(
                decoder,
                prefixes,
                torch.ones(6, 3),
                encoder_states.repeat_interleave(3, dim=0),
                encoder_mask.repeat_interleave(3, dim=0),
            )
            torch.testing.assert_close(output, full[:, -1:], **TOLERANCE)

            chosen = torch.tensor([4, 0])
            selected = beams.index_select(chosen)
            third = torch.tensor([[2], [5]])
            output, _ = decoder.decoder(decoder.embedding(third, start_pos=3), None, kv_cache=selected)
            full, _ = _full_pass(
                decoder,
                torch.cat([prefixes[chosen], third], dim=1),
                torch.ones(2, 4),
                encoder_states[[1, 0]],
                encoder_mask[[1, 0]],
            )
            torch.testing.assert_close(output, full[:, -1:], **TOLERANCE)

    @pytest.mark.unit
    def test_bounded_buffers_reused_across_batches(self):
        decoder = _cache_decoder()
        cache = TransformerDecoderKeyValueCache(
            decoder.decoder, batch_size=8, frame_capacity=12, capacity=6, device=torch.device("cpu"), dtype=torch.float
        )
        assert cache.batch_size == 8 and cache.frame_capacity == 12 and cache.capacity == 6
        tokens = torch.tensor([[BOS, 3], [BOS, 4], [BOS, 5], [BOS, 6], [BOS, 7], [BOS, 8], [BOS, 1], [BOS, 2]])
        with torch.no_grad():
            for lengths, repeats, seed in [([9, 6], 3, 1), ([7], 2, 2)]:
                encoder_states, encoder_mask = _encoder(lengths, seed)
                rows = len(lengths) * repeats
                cache.load_encoder_states(decoder.decoder, encoder_states, encoder_mask, repeats=repeats)
                assert int(cache.length) == 0
                outputs = [
                    decoder.decoder(
                        decoder.embedding(tokens[:, step : step + 1], start_pos=step), None, kv_cache=cache
                    )[0]
                    for step in range(2)
                ]
                output = torch.cat(outputs, dim=1)
                assert torch.isfinite(output).all()
                exact = decoder.decoder.create_kv_cache(
                    encoder_states.repeat_interleave(repeats, dim=0),
                    encoder_mask.repeat_interleave(repeats, dim=0),
                    capacity=2,
                )
                expected, _ = decoder.decoder(decoder.embedding(tokens[:rows]), None, kv_cache=exact)
                torch.testing.assert_close(output[:rows], expected, **TOLERANCE)
        with pytest.raises(ValueError, match="exceed"):
            cache.load_encoder_states(decoder.decoder, torch.zeros(5, 3, HIDDEN), None, repeats=2)

    @pytest.mark.unit
    def test_rejects_conflicting_arguments(self):
        decoder = _cache_decoder()
        encoder_states, encoder_mask = _encoder([9, 6])
        cache = decoder.decoder.create_kv_cache(encoder_states, encoder_mask, capacity=4)
        embedded = decoder.embedding(torch.full((2, 1), BOS))
        with pytest.raises(ValueError, match="kv_cache carries"):
            decoder.decoder(embedded, None, encoder_states, encoder_mask, kv_cache=cache)
        rotary_cross = _cache_decoder(cross_attn=functools.partial(MultiHeadAttention, rotary_base=10000.0))
        with pytest.raises(ValueError, match="self-attention only"):
            rotary_cross.decoder.create_kv_cache(encoder_states, encoder_mask, capacity=4)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
    def test_captured_step_matches_eager(self):
        device = torch.device("cuda")
        decoder = _cache_decoder().to(device)
        encoder_states, encoder_mask = _encoder([9, 6])
        encoder_states, encoder_mask = encoder_states.to(device), encoder_mask.to(device)
        weight = decoder.embedding.token_embedding.weight
        steps = 6

        def decode(cache, tokens):
            """One step on static buffers: embed the last tokens at the cache length, decode, pick the next tokens."""
            embedded = decoder.embedding(tokens, start_pos=cache.length)
            output, _ = decoder.decoder(embedded, None, kv_cache=cache)
            tokens.copy_((output[:, -1] @ weight.T).argmax(dim=-1, keepdim=True))
            return output

        with torch.inference_mode():
            cache = decoder.decoder.create_kv_cache(encoder_states, encoder_mask, capacity=steps)
            tokens = torch.full((2, 1), BOS, device=device)
            expected = [decode(cache, tokens).clone() for _ in range(steps)]
            expected_tokens = tokens.clone()

            cache.reset()
            tokens.fill_(BOS)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                decode(cache, tokens)  # warm-up on the capture stream
            torch.cuda.current_stream().wait_stream(stream)
            cache.reset()
            tokens.fill_(BOS)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_output = decode(cache, tokens)
            cache.reset()
            tokens.fill_(BOS)
            outputs = []
            for _ in range(steps):
                graph.replay()
                outputs.append(static_output.clone())
            torch.cuda.synchronize()
        assert int(cache.length) == steps
        assert torch.equal(tokens, expected_tokens)
        for output, reference in zip(outputs, expected):
            torch.testing.assert_close(output, reference, **TOLERANCE)
