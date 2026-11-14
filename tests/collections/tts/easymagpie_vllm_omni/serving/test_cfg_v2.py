# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CFG contracts for compiled diagnostics and the v2 globally cached predictor."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def test_trace_mode_has_distinct_compiled_cache_identity(monkeypatch):
    from conftest import build_vllm_config
    from easymagpie_vllm_omni.local_transformer import EasyMagpieLocalTransformerPredictor
    from vllm.compilation.decorators import _model_hash_key

    monkeypatch.delenv('EASYMAGPIE_CFG_TRACE_PATH', raising=False)
    regular = EasyMagpieLocalTransformerPredictor(vllm_config=build_vllm_config())
    monkeypatch.setenv('EASYMAGPIE_CFG_TRACE_PATH', 'unused-test-trace.jsonl')
    traced = EasyMagpieLocalTransformerPredictor(vllm_config=build_vllm_config())
    assert _model_hash_key(regular._cfg_code_loop.forward) != _model_hash_key(traced._cfg_code_loop.forward)
    traced.load_state_dict(regular.state_dict())
    regular.init_forbidden_mask()
    traced.init_forbidden_mask()
    conditional, unconditional = torch.randn(2, 64), torch.randn(2, 64)
    torch.manual_seed(123)
    expected, production_diagnostics = regular.generate_codes_cfg(conditional, unconditional, torch.full((2, 1), 2.5))
    torch.manual_seed(123)
    actual, diagnostics = traced.generate_codes_cfg(conditional, unconditional, torch.full((2, 1), 2.5))
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(production_diagnostics) == 0
    assert (diagnostics[:, :2] > 0).all()


def test_v2_cfg_advances_all_cache_rows_and_shares_guided_history():
    predictor = _make_predictor()
    hidden = torch.tensor([[3.0, 2.0], [4.0, -2.0], [1.0, 1.0], [-2.0, 1.0]])
    # Pair rows are intentionally reordered. Rows 2 and 3 are prefill/padding.
    positions = torch.tensor([7, 7, 0, 0])
    conditional_rows = torch.tensor([0, 0, 2, 3])
    unconditional_rows = torch.tensor([1, 1, 2, 3])
    predict = torch.tensor([True, True, False, False])
    predictor.temperature = 0.0
    codes, diagnostics = predictor.generate_codes_cfg(
        hidden,
        positions=positions,
        conditional_rows=conditional_rows,
        unconditional_rows=unconditional_rows,
        scales=torch.full((4, 1), 2.5),
        predict=predict,
    )
    # Conditional argmax is zero, but guided logits [1.5, 8] select one.
    assert codes[:2].tolist() == [[1, 1], [1, 1]]
    assert codes[2:].eq(predictor.arch.mask_token_id).all()
    assert len(predictor.observed) == 2
    assert all(item[0].shape[0] == 4 for item in predictor.observed)
    assert all(torch.equal(item[1], positions) for item in predictor.observed)
    assert predictor.feedback[0].tolist() == [[1], [1], [99], [99]]
    assert diagnostics.shape == (4, 3)


def test_v2_cfg_scale_one_matches_native_sampler_with_shared_noise():
    predictor = _make_predictor()
    hidden = torch.tensor([[3.0, 2.0], [4.0, -2.0], [-1.0, 2.0], [-4.0, -2.0]])
    positions = torch.tensor([8, 8, 4, 4])
    # Two pairs, each with equal C/U hidden states to make native shared output testable.
    hidden[1] = hidden[0]
    hidden[3] = hidden[2]
    rows = torch.tensor([0, 0, 2, 2])
    torch.manual_seed(45)
    native = predictor.generate_codes(hidden, positions=positions)
    torch.manual_seed(45)
    actual, _ = predictor.generate_codes_cfg(
        hidden,
        positions=positions,
        conditional_rows=rows,
        unconditional_rows=torch.tensor([1, 1, 3, 3]),
        scales=torch.ones(4, 1),
        predict=torch.ones(4, dtype=torch.bool),
    )
    assert torch.equal(actual[rows.unique()], native[rows.unique()])
    assert torch.equal(actual[0], actual[1])
    assert torch.equal(actual[2], actual[3])


def test_v2_cfg_uses_each_pairs_scale_and_preserves_eos():
    predictor = _make_predictor()
    predictor.temperature = 0.0
    predictor.arch.codebook_size = 1
    hidden = torch.tensor([[3.0, 2.0], [4.0, -2.0], [3.0, 2.0], [4.0, -2.0]])
    codes, _ = predictor.generate_codes_cfg(
        hidden,
        positions=torch.arange(4),
        conditional_rows=torch.tensor([0, 0, 2, 2]),
        unconditional_rows=torch.tensor([1, 1, 3, 3]),
        scales=torch.tensor([[1.0], [1.0], [2.5], [2.5]]),
        predict=torch.ones(4, dtype=torch.bool),
    )
    assert codes[:2].eq(0).all()
    assert codes[2:].eq(predictor.arch.audio_eos_id).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph replay requires a GPU')
def test_v2_cfg_graph_replay_reads_updated_scales_and_prediction_mask():
    with torch.device('cuda'):
        predictor = _make_predictor()
        predictor.temperature = 0.0
        hidden = torch.tensor([[3.0, 2.0], [4.0, -2.0]])
        arguments = dict(
            positions=torch.tensor([7, 7]),
            conditional_rows=torch.tensor([0, 0]),
            unconditional_rows=torch.tensor([1, 1]),
            scales=torch.ones(2, 1),
            predict=torch.ones(2, dtype=torch.bool),
        )
        # Warm eager operations on a side stream before capturing the same loop.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            predictor.generate_codes_cfg(hidden, **arguments)
        torch.cuda.current_stream().wait_stream(stream)

        class Replay:
            def __init__(self, forward):
                self.forward = forward
                self.graph = torch.cuda.CUDAGraph()
                self.output = None

            def __call__(self, *args):
                if self.output is None:
                    with torch.cuda.graph(self.graph):
                        self.output = self.forward(*args)
                else:
                    self.graph.replay()
                return self.output

        predictor._cfg_forward = (Replay(predictor._generate_cfg),)
        codes, _ = predictor.generate_codes_cfg(hidden, **arguments)
        assert codes.eq(0).all()
        arguments['scales'].fill_(2.5)
        codes, _ = predictor.generate_codes_cfg(hidden, **arguments)
        assert codes.eq(1).all()
        arguments['predict'].zero_()
        codes, _ = predictor.generate_codes_cfg(hidden, **arguments)
        assert codes.eq(predictor.arch.mask_token_id).all()


def test_v2_cfg_pure_prefill_advances_caches_without_emitting_audio():
    from easymagpie_vllm_omni.config import NO_AUDIO_FRAME

    model = _make_model()
    model._forward_cfg(torch.zeros(4, dtype=torch.long), torch.arange(4), None, torch.randn(4, 2))
    assert len(model.code_predictor.observed) == 2
    assert model._out_codes[:4].eq(NO_AUDIO_FRAME).all()
    assert model.code_predictor.feedback[0].eq(99).all()


def test_v2_cfg_mixed_prefill_retains_every_cache_row_but_emits_only_decode():
    from easymagpie_vllm_omni.config import NO_AUDIO_FRAME

    model = _make_model()
    model._cfg_records.extend([('d', 'conditional', 4, 1, 2.5), ('d', 'unconditional', 5, 1, 2.5)])
    model._forward_cfg(torch.zeros(6, dtype=torch.long), torch.arange(6), None, torch.randn(6, 2))
    assert len(model.code_predictor.observed) == 2
    assert all(item[0].shape[0] == 6 for item in model.code_predictor.observed)
    assert model._out_codes[:4].eq(NO_AUDIO_FRAME).all()
    assert torch.equal(model._out_codes[4], model._out_codes[5])
    assert model.code_predictor.feedback[0][:4].eq(99).all()


def _make_model():
    from easymagpie_vllm_omni.cfg_model import CFGModelMixin

    class Model(CFGModelMixin):
        has_phoneme = False
        _cfg_trace_path = None

        def _assemble_decode_embeddings(self, combined, rows):
            pass

        def _flag_audio_eos(self, codes, rows):
            pass

    model = Model()
    model._cfg_records = [('p', 'conditional', 0, 2, 2.5), ('p', 'unconditional', 2, 2, 2.5)]
    model._cfg_prefill_keys = {'p'}
    model._combined_embeddings = torch.zeros(8, 2)
    model._out_codes = torch.zeros(8, 2, dtype=torch.long)
    model._dec_audio_valid = torch.ones(8)
    model._token_stop = torch.zeros(8)
    model._sample_stop = torch.zeros(8)
    model._cfg_backbone_forward = (lambda **kwargs: kwargs['inputs_embeds'].clone(),)
    model.code_predictor = _make_predictor()
    return model


def _make_predictor():
    from conftest import build_vllm_config
    from easymagpie_vllm_omni.acoustic_predictor import EasyMagpieAcousticCodesPredictor

    predictor = EasyMagpieAcousticCodesPredictor.__new__(EasyMagpieAcousticCodesPredictor)
    nn.Module.__init__(predictor)
    predictor.arch = SimpleNamespace(mask_token_id=99, codebook_size=32, audio_eos_id=33, embedding_dim=2)
    predictor.num_codebooks = 2
    predictor._sample_top_k = 2
    predictor.temperature = 0.7
    predictor.observed = []
    predictor.feedback = []

    class Block(nn.Module):
        num_codes = 1

        def __init__(self, index):
            super().__init__()
            self.codebook_indices = (index,)

        def forward(self, hidden, positions):
            predictor.observed.append((hidden.clone(), positions.clone()))
            return hidden

        def compute_logits(self, hidden):
            return hidden.unsqueeze(1)

    class Embedder(nn.Module):
        def embed_codebooks(self, codes, indices):
            predictor.feedback.append(codes.clone())
            return torch.zeros(codes.shape[0], 2)

    predictor.blocks = nn.ModuleList([Block(0), Block(1)])
    predictor.audio_embedder = Embedder()
    predictor._gumbel_buf = torch.zeros(128, 2, 2)
    predictor._init_cfg_buffers(build_vllm_config())
    return predictor
