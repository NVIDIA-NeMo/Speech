# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os.path

import pytest
import torch
from lightning.pytorch import Trainer
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)
from nemo.collections.asr.parts.context_biasing.context_graph_universal import ContextGraph
from nemo.collections.common.tokenizers import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenWithLength, VarBPERepresentation

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


def _case_variant_var_bpe_representation():
    return VarBPERepresentation(
        canonical_lengths=[1, 1],
        token_ids_with_merges=[
            [TokenWithLength(token_id=1), TokenWithLength(token_id=3)],
            [
                TokenWithLength(token_id=2),
                TokenWithLength(token_id=4),
                TokenWithLength(token_id=5, length=2),
                TokenWithLength(token_id=6, length=2),
            ],
        ],
    )


def _plain_var_bpe_representation(token_ids):
    return VarBPERepresentation(
        canonical_lengths=[1] * len(token_ids),
        token_ids_with_merges=[[TokenWithLength(token_id=token_id)] for token_id in token_ids],
    )


class _RecordingVarBPETokenizer:
    vocab_size = 7

    def __init__(self):
        self.var_bpe_calls = []

    def text_to_ids(self, text):
        raise AssertionError(f"Unexpected default tokenization for {text}")

    def text_to_ids_var_bpe(self, text, case_insensitive=True):
        self.var_bpe_calls.append((text, case_insensitive))
        return _case_variant_var_bpe_representation()


class _RecordingAggregateVarBPETokenizer:
    def __init__(self, vocab_size=10):
        self.vocab = [f"tok_{i}" for i in range(vocab_size)]
        self.vocab_size = vocab_size
        self.var_bpe_calls = []

    def text_to_ids_var_bpe(self, text, case_insensitive=True):
        self.var_bpe_calls.append((text, case_insensitive))
        return _case_variant_var_bpe_representation()


@pytest.mark.unit
def test_aggregate_tokenizer_offsets_var_bpe_representation():
    tokenizer1 = _RecordingAggregateVarBPETokenizer()
    tokenizer2 = _RecordingAggregateVarBPETokenizer()
    tokenizer = AggregateTokenizer({"en": tokenizer1, "es": tokenizer2})

    result = tokenizer.text_to_ids_var_bpe("Ab", lang_id="es", case_insensitive=False)

    assert tokenizer1.var_bpe_calls == []
    assert tokenizer2.var_bpe_calls == [("Ab", False)]
    assert result == VarBPERepresentation(
        canonical_lengths=[1, 1],
        token_ids_with_merges=[
            [TokenWithLength(token_id=11), TokenWithLength(token_id=13)],
            [
                TokenWithLength(token_id=12),
                TokenWithLength(token_id=14),
                TokenWithLength(token_id=15, length=2),
                TokenWithLength(token_id=16, length=2),
            ],
        ],
    )


@pytest.fixture(scope="module")
def test_context_graph():
    phrases = ["abc", "abd", "c"]
    phrases_ids = [[1, 2, 3], [1, 2, 4], [3]]
    scores = [0.0, 0.0, 0.0]
    context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
    context_graph.build(token_ids=phrases_ids, phrases=phrases, scores=scores, uniform_weights=False)
    return context_graph


@pytest.fixture(scope="module")
def test_boosting_tree(test_context_graph):
    boosting_tree = GPUBoostingTreeModel.from_context_graph(
        context_graph=test_context_graph,
        vocab_size=5,
        unk_score=0.0,
        final_eos_score=0.0,
        use_triton=True,
        uniform_weights=False,
    )
    return boosting_tree


@pytest.fixture(scope="module")
def conformer_ctc_bpe_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    model.set_trainer(Trainer(devices=1, accelerator="cpu"))
    model = model.eval()
    return model


class TestGPUBoostingTreeModel:
    @pytest.mark.unit
    def test_building_context_graph(self, test_context_graph):
        """Test initial python-based context graph"""
        context_graph = test_context_graph
        assert context_graph.num_nodes == 5
        # end nodes
        assert context_graph.root.next[1].next[2].next[3].is_end
        assert context_graph.root.next[1].next[2].next[4].is_end
        assert context_graph.root.next[3].is_end
        # words in the end nodes
        assert context_graph.root.next[1].next[2].next[3].phrase == "abc"
        assert context_graph.root.next[1].next[2].next[4].phrase == "abd"
        assert context_graph.root.next[3].phrase == "c"
        # fail links
        assert context_graph.root.next[1].next[2].next[3].fail.token == 3
        assert context_graph.root.next[1].next[2].next[4].fail.token == -1  # root
        assert context_graph.root.next[3].fail.token == -1  # root
        # node scores
        assert round(context_graph.root.next[1].next[2].next[3].node_score, 2) == 4.79
        assert round(context_graph.root.next[1].next[2].next[4].node_score, 2) == 4.79
        assert round(context_graph.root.next[3].node_score, 2) == 1.0

    @pytest.mark.unit
    def test_building_context_graph_from_var_bpe(self):
        """Test var-BPE graph aliases for case variants and merged tokens."""
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build_from_var_bpe(
            token_ids=[_case_variant_var_bpe_representation()],
            phrases=["ab"],
            scores=[0.0],
            uniform_weights=False,
            var_bpe_scoring_temp=0.0,
        )

        first_node = context_graph.root.next[1]
        final_node = first_node.next[2]

        assert context_graph.num_nodes == 2
        assert context_graph.root.next[3] is first_node
        assert first_node.next[4] is final_node
        assert context_graph.root.next[5] is final_node
        assert context_graph.root.next[6] is final_node
        assert final_node.is_end
        assert final_node.phrase == "ab"
        assert first_node.is_primary
        assert final_node.is_primary

        expected_final_score = 1.0 + 1.0 + torch.log(torch.tensor(2.0)).item()
        assert first_node.node_score == pytest.approx(1.0)
        assert final_node.node_score == pytest.approx(expected_final_score)

        order2cnt, tbranches = GPUBoostingTreeModel._read_context_graph(context_graph=context_graph)
        assert order2cnt == {1: 4, 2: 2}
        assert len(tbranches) == 6

    @pytest.mark.unit
    def test_var_bpe_overlapping_phrase_uses_primary_path_for_fail_link(self):
        """Test merged-token aliases do not hide canonical suffix fail links."""
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build_from_var_bpe(
            token_ids=[
                _case_variant_var_bpe_representation(),
                _plain_var_bpe_representation([2, 7]),
            ],
            phrases=["ab", "bc"],
            scores=[0.0, 0.0],
            uniform_weights=False,
            var_bpe_scoring_temp=0.0,
        )

        ab_final_node = context_graph.root.next[1].next[2]
        b_node = context_graph.root.next[2]

        assert ab_final_node.phrase == "ab"
        assert b_node.next[7].phrase == "bc"
        assert context_graph.root.next[5] is ab_final_node
        assert context_graph.root.next[6] is ab_final_node
        assert ab_final_node.fail is b_node

    @pytest.mark.unit
    def test_var_bpe_boosting_tree_scores_equivalent_paths(self):
        """Test split, merged, and case-variant paths receive the same total boost."""
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build_from_var_bpe(
            token_ids=[_case_variant_var_bpe_representation()],
            phrases=["ab"],
            scores=[0.0],
            uniform_weights=False,
            var_bpe_scoring_temp=0.0,
        )
        boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph,
            vocab_size=7,
            unk_score=0.0,
            final_eos_score=0.0,
            use_triton=False,
            uniform_weights=False,
        )

        sentences_ids = [[1, 2], [3, 2], [1, 4], [3, 4], [5], [6], [2]]
        boosting_scores = boosting_tree(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]),
            bos=False,
            eos=False,
        )

        expected_second_token_score = 1.0 + torch.log(torch.tensor(2.0)).item()
        expected_total_score = 1.0 + expected_second_token_score

        assert boosting_scores[0, 0].item() == pytest.approx(1.0)
        assert boosting_scores[0, 1].item() == pytest.approx(expected_second_token_score)
        assert boosting_scores[4, 0].item() == pytest.approx(expected_total_score)
        assert boosting_scores[5, 0].item() == pytest.approx(expected_total_score)
        assert boosting_scores.sum(dim=1)[:6].detach().tolist() == pytest.approx([expected_total_score] * 6)
        assert boosting_scores.sum(dim=1)[6].item() == pytest.approx(0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bpe_mode,phrase,expected_call",
        [("case_insensitive", "Ab", ("ab", True)), ("var_bpe", "Ab", ("Ab", False))],
    )
    def test_boosting_tree_from_config_uses_var_bpe_modes(self, bpe_mode, phrase, expected_call):
        """Test config routing for var-BPE and case-insensitive modes."""
        tokenizer = _RecordingVarBPETokenizer()
        boosting_tree = GPUBoostingTreeModel.from_config(
            BoostingTreeModelConfig(
                key_phrases_list=[phrase],
                bpe_mode=bpe_mode,
                depth_scaling=1.0,
                use_triton=False,
            ),
            tokenizer=tokenizer,
        )

        assert tokenizer.var_bpe_calls == [expected_call]

        boosting_scores = boosting_tree(
            labels=pad_sequence([torch.LongTensor([3, 4]), torch.LongTensor([5])], batch_first=True),
            labels_lengths=torch.LongTensor([2, 1]),
            bos=False,
            eos=False,
        )
        expected_total_score = 1.0 + 1.0 + torch.log(torch.tensor(2.0)).item()
        assert boosting_scores.sum(dim=1).detach().tolist() == pytest.approx([expected_total_score] * 2)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 3, 8])
    def test_advance_method(self, test_boosting_tree, device, batch_size):
        """Test advance method with different batch sizes"""
        test_boosting_tree.to(device)
        # Test with initial states
        init_states = test_boosting_tree.get_init_states(batch_size=batch_size, bos=True)
        scores, next_states = test_boosting_tree.advance(init_states)

        assert scores.shape == (batch_size, 5)  # vocab_size=5
        assert next_states.shape == (batch_size, 5)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_get_final_method(self, test_boosting_tree, device):
        """Test get_final method for EOS scoring"""
        test_boosting_tree.to(device)
        # Test with various states
        states = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        final_scores = test_boosting_tree.get_final(states)

        assert final_scores.shape == (3,)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_boosting_tree_inference(self, test_boosting_tree, device):
        """Test boosting tree inference with predefined sentences"""
        test_boosting_tree.to(device)

        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4], [3, 1, 2, 1], []]  # ['abcba', 'bbabd', 'caba', '']
        boosting_scores = test_boosting_tree(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True).to(
                device
            ),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]).to(device),
            bos=False,
            eos=False,
        )
        correct_answer = torch.tensor(
            [
                [1.0000, 1.6931, 2.0986, 0.0000, 1.0000],
                [0.0000, 0.0000, 1.0000, 1.6931, 2.0986],
                [1.0000, 1.0000, 1.6931, -1.6931, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            device=device,
        )
        assert torch.allclose(boosting_scores, correct_answer, atol=1e-4)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_vs_pytorch_consistency(self, test_context_graph):
        """Compare Triton vs PyTorch implementations"""
        device = torch.device("cuda")

        # Create two identical models with different implementations
        boosting_tree_triton = GPUBoostingTreeModel.from_context_graph(
            context_graph=test_context_graph, vocab_size=5, use_triton=True
        ).to(device)

        boosting_tree_pytorch = GPUBoostingTreeModel.from_context_graph(
            context_graph=test_context_graph, vocab_size=5, use_triton=False
        ).to(device)

        # Test with same input
        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4]]
        labels = pad_sequence([torch.LongTensor(s) for s in sentences_ids], batch_first=True).to(device)
        lengths = torch.LongTensor([len(s) for s in sentences_ids]).to(device)

        scores_triton = boosting_tree_triton(labels=labels, labels_lengths=lengths, bos=False, eos=False)
        scores_pytorch = boosting_tree_pytorch(labels=labels, labels_lengths=lengths, bos=False, eos=False)

        assert torch.allclose(scores_triton, scores_pytorch, atol=1e-5)

    @pytest.mark.unit
    def test_eos_handling(self, test_context_graph):
        """Test EOS token handling (important for AED models)"""
        boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=test_context_graph, vocab_size=5, unk_score=0.0, final_eos_score=1.0
        )

        # Test advance with EOS
        init_states = torch.tensor([1, 2], dtype=torch.long)
        scores, next_states = boosting_tree.advance(init_states, eos_id=0)

        # state 2 in the 1st batch should have final_eos_score value
        assert (
            round(scores[0, 0].item(), 2) == 1.69
        )  # (1.69+0): 1.69 as max score for state 1 and 0 because it is not final state
        assert scores[1, 0] == 2.0  # (1+1): 1 as max score for state 2 and 1 because it is final state

    @pytest.mark.unit
    # I need to test that the boosting tree model is built correctly from the config using model_path, key_phrases_file, key_phrases_list
    def test_boosting_tree_model_from_config(self, conformer_ctc_bpe_model, tmp_path):
        """Test that the boosting tree model is built correctly from the config using model_path, key_phrases_file, key_phrases_list"""

        # 1. build boosting tree model from model path
        boosting_tree_cfg = BoostingTreeModelConfig()
        phrases = ["abc", "abd", "c"]
        phrases_ids = [conformer_ctc_bpe_model.tokenizer.text_to_ids(phrase) for phrase in phrases]
        scores = [0.0, 0.0, 0.0]
        context_graph = ContextGraph(
            context_score=boosting_tree_cfg.context_score, depth_scaling=boosting_tree_cfg.depth_scaling
        )
        context_graph.build(
            token_ids=phrases_ids, phrases=phrases, scores=scores, uniform_weights=boosting_tree_cfg.uniform_weights
        )
        test_boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph,
            vocab_size=conformer_ctc_bpe_model.tokenizer.vocab_size,
            unk_score=boosting_tree_cfg.unk_score,
            final_eos_score=boosting_tree_cfg.final_eos_score,
            use_triton=boosting_tree_cfg.use_triton,
            uniform_weights=boosting_tree_cfg.uniform_weights,
        )

        test_boosting_tree.save_to(tmp_path / "test_boosting_tree.nemo")
        boosting_tree_cfg = BoostingTreeModelConfig(model_path=tmp_path / "test_boosting_tree.nemo")
        boosting_tree_from_model_path = GPUBoostingTreeModel.from_config(
            boosting_tree_cfg, tokenizer=conformer_ctc_bpe_model.tokenizer
        )

        # 2. build boosting tree model from key phrases file
        with open(tmp_path / "test_boosting_tree.txt", "w") as f:
            f.write("abc\nabd\nc")
        boosting_tree_cfg = BoostingTreeModelConfig(key_phrases_file=tmp_path / "test_boosting_tree.txt")
        boosting_tree_from_key_phrases_file = GPUBoostingTreeModel.from_config(
            boosting_tree_cfg, tokenizer=conformer_ctc_bpe_model.tokenizer
        )

        # 3. build boosting tree model from key phrases list
        boosting_tree_cfg = BoostingTreeModelConfig(key_phrases_list=["abc", "abd", "c"])
        boosting_tree_from_key_phrases_list = GPUBoostingTreeModel.from_config(
            boosting_tree_cfg, tokenizer=conformer_ctc_bpe_model.tokenizer
        )

        # check that the boosting tree models are the same
        assert torch.allclose(
            boosting_tree_from_model_path.arcs_weights, boosting_tree_from_key_phrases_file.arcs_weights
        )
        assert torch.allclose(
            boosting_tree_from_model_path.arcs_weights, boosting_tree_from_key_phrases_list.arcs_weights
        )


@pytest.fixture(scope="module")
def bpe_tokenizer():
    model_path = "/home/TestData/asr/stt_en_conformer_transducer_small.nemo"
    if os.path.exists(model_path):
        model = ASRModel.restore_from(model_path, map_location="cpu")
    else:
        model = ASRModel.from_pretrained("stt_en_conformer_transducer_small", map_location="cpu")
    return model.tokenizer


class TestVariativeBPE:
    @pytest.mark.unit
    @pytest.mark.with_downloads
    def test_tokenizer_var_bpe_representation_contains_split_and_merge(self, bpe_tokenizer):
        token_ids = bpe_tokenizer.text_to_ids("hello")
        split_ids = bpe_tokenizer.tokens_to_ids([bpe_tokenizer.spm_separator, "h", "e", "l", "l", "o"])

        assert bpe_tokenizer.ids_to_tokens(token_ids) == [f"{bpe_tokenizer.spm_separator}hello"]

        var_bpe = bpe_tokenizer.text_to_ids_var_bpe("hello", case_insensitive=False)

        assert var_bpe.canonical_lengths == [len(split_ids)]
        assert len(var_bpe.token_ids_with_merges) == len(split_ids)
        assert [token_group[0] for token_group in var_bpe.token_ids_with_merges] == [
            TokenWithLength(token_id=token_id) for token_id in split_ids
        ]
        assert TokenWithLength(token_id=token_ids[0], length=len(split_ids)) in var_bpe.token_ids_with_merges[-1]

    @pytest.mark.unit
    @pytest.mark.with_downloads
    @pytest.mark.parametrize(
        "bpe_mode,phrase",
        [("var_bpe", "hello"), ("case_insensitive", "Hello")],
    )
    @pytest.mark.parametrize("var_bpe_penalize_subsplits", [True, False])
    def test_boosting_tree_from_config_sentencepiece_var_bpe_scores_split_and_merged_paths(
        self,
        bpe_tokenizer,
        bpe_mode: str,
        phrase: str,
        var_bpe_penalize_subsplits: bool,
    ):
        boosting_tree = GPUBoostingTreeModel.from_config(
            BoostingTreeModelConfig(
                key_phrases_list=[phrase],
                bpe_mode=bpe_mode,
                depth_scaling=1.0,
                final_eos_score=0.0,
                use_triton=False,
                var_bpe_scoring_temp=0.0,
                var_bpe_penalize_subsplits=var_bpe_penalize_subsplits,
            ),
            tokenizer=bpe_tokenizer,
        )

        merged_ids = bpe_tokenizer.text_to_ids("hello")
        split_ids = bpe_tokenizer.tokens_to_ids([bpe_tokenizer.spm_separator, "h", "e", "l", "l", "o"])
        boosting_scores = boosting_tree(
            labels=pad_sequence(
                [torch.LongTensor(merged_ids), torch.LongTensor(split_ids)],
                batch_first=True,
            ),
            labels_lengths=torch.LongTensor([len(merged_ids), len(split_ids)]),
            bos=False,
            eos=False,
        )

        assert boosting_scores[0, 0].item() == pytest.approx(1.0)
        assert boosting_scores.sum(dim=1).detach().tolist() == pytest.approx([1.0, 1.0])

        if not var_bpe_penalize_subsplits:
            expected_split_scores = [1.0 / len(split_ids)] * len(split_ids)
            assert boosting_scores[1, : len(split_ids)].detach().tolist() == pytest.approx(expected_split_scores)
