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
import pytest
import torch

from nemo.collections.speechlm2.parts.metrics import BLEU, WER, Intelligibility
from nemo.collections.speechlm2.parts.metrics.empty_text import EmptyTextMetric
from nemo.collections.speechlm2.parts.metrics.perplexity import Perplexity


def test_bleu():
    metric = BLEU(verbose=False)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        hyps=["a b d"],
    )
    ans = metric.compute()
    assert ans["txt_bleu_dataset_1"] == 100.0
    assert ans["txt_bleu_dataset_2"] == 0.0
    assert ans["txt_bleu"] == 50.0  # average across datasets


def test_wer():
    metric = WER(verbose=False)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        hyps=["a b d"],
    )
    ans = metric.compute()
    assert ans["wer_dataset_1"] == 0.0
    assert ans["wer_dataset_2"] == 1 / 3
    assert ans["wer"] == 1 / 6  # average across datasets


def test_empty_text_metric():
    """Test EmptyTextMetric for detecting empty hypotheses"""
    metric = EmptyTextMetric(verbose=False)

    # Test with some empty and non-empty texts
    hyps = ["hello world", "", "  ", "test", "   \n  "]
    metric.update("test_batch", hyps)

    results = metric.compute()

    # Should detect 3 empty texts out of 5
    assert "empty_text_rate_test_batch" in results
    assert results["empty_text_rate_test_batch"].item() == pytest.approx(0.6, abs=0.01)  # 3/5 = 0.6


def test_empty_text_metric_reset():
    """Test EmptyTextMetric reset functionality"""
    metric = EmptyTextMetric(verbose=False)

    # Add some data
    metric.update("test", ["hello", ""])
    metric.reset()

    # After reset, should have no data
    results = metric.compute()
    assert len(results) == 0


def test_empty_text_all_valid():
    """Test EmptyTextMetric with no empty texts"""
    metric = EmptyTextMetric(verbose=False)

    metric.update("test", ["hello", "world", "test"])

    results = metric.compute()
    assert results["empty_text_rate_test"].item() == 0.0


def test_empty_text_all_empty():
    """Test EmptyTextMetric with all empty texts"""
    metric = EmptyTextMetric(verbose=False)

    metric.update("test", ["", "  ", "\n"])

    results = metric.compute()
    assert results["empty_text_rate_test"].item() == 1.0


def test_perplexity_basic():
    """Test basic perplexity calculation"""
    metric = Perplexity(ignore_index=-100, verbose=False)

    # Create simple logits and targets
    vocab_size = 10
    batch_size = 2
    seq_len = 3

    # Create perfect predictions (all correct)
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Set logits to have high probability for correct tokens
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, targets[b, s]] = 10.0  # High logit for correct token

    ppl = metric.update("test", logits, targets)

    # With perfect predictions, perplexity should be close to 1
    assert ppl < 2.0  # Should be very low


def test_perplexity_with_padding():
    """Test perplexity with padding tokens"""
    metric = Perplexity(ignore_index=-100, verbose=False)

    vocab_size = 10
    batch_size = 2
    seq_len = 4

    logits = torch.randn(batch_size, seq_len, vocab_size)
    # Include padding tokens (ignore_index = -100)
    targets = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]])

    ppl = metric.update("test", logits, targets)

    # Should not fail with padding
    assert ppl > 0
    assert not torch.isnan(torch.tensor(ppl))


def test_perplexity_compute():
    """Test perplexity compute aggregation"""
    metric = Perplexity(ignore_index=-100, verbose=False)

    vocab_size = 10
    logits = torch.randn(2, 3, vocab_size)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])

    metric.update("test", logits, targets)

    results = metric.compute()
    assert "perplexity_test" in results
    assert results["perplexity_test"] > 0


def test_turn_taking_import():
    """Test that turn taking metric function can be imported"""
    from nemo.collections.speechlm2.parts.metrics.turn_taking import compute_turn_taking_metrics

    # Test with dummy data
    source_tokens = torch.tensor([[1, 2, 3, 4]])  # dummy tokens
    pred_tokens = torch.tensor([[5, 6, 7, 8]])  # dummy tokens
    eos_token_id = 4
    bos_token_id = 5

    accuracy, latency = compute_turn_taking_metrics(source_tokens, pred_tokens, eos_token_id, bos_token_id)
    assert isinstance(accuracy, float)
    assert isinstance(latency, float)


def test_mcq_evaluator_import():
    """Test that MCQ evaluator can be imported and initialized"""
    import tempfile

    from nemo.collections.speechlm2.parts.metrics.mcq_evaluator import MCQEvaluator

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = MCQEvaluator(manifest_dir=tmpdir)
        assert evaluator is not None
        assert evaluator.manifest_dir == tmpdir


def test_results_logger_import():
    """Test that results logger can be imported"""
    import tempfile

    from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ResultsLogger(save_path=tmpdir)
        assert logger is not None


def test_results_logger_single_rank(tmp_path):
    """Test ResultsLogger with single rank (non-distributed)"""
    from unittest.mock import patch

    from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger

    save_path = str(tmp_path / "single_rank")

    # Mock distributed functions to simulate single rank
    with (
        patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_rank', return_value=0),
        patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_world_size', return_value=1),
        patch('torch.distributed.is_available', return_value=False),
    ):

        logger = ResultsLogger(save_path=save_path)

        # Add some test data
        logger.update(
            name="test_dataset",
            refs=["hello world", "goodbye"],
            hyps=["hello there", "goodbye"],
            asr_hyps=[None, None],
            samples_id=["sample1", "sample2"],
            pred_audio=None,
            pred_audio_sr=16000,
            user_audio=None,
            user_audio_sr=16000,
            src_refs=["user input 1", "user input 2"],
            src_hyps=["", ""],
        )

        # Compute and save
        metrics = logger.compute_and_save(special_subset_names=[], mcq_subset_names=[])

        # Check that files were created
        import os

        metadata_path = os.path.join(save_path, "metadatas", "test_dataset_rank0.json")
        assert os.path.exists(metadata_path)

        # Check final merged file (should be same as rank0 in single-rank case)
        final_path = os.path.join(save_path, "metadatas", "test_dataset.json")
        assert os.path.exists(final_path)


def test_results_logger_multi_rank(tmp_path):
    """Test ResultsLogger with multiple ranks (simulated distributed training)"""
    import json
    import os
    from unittest.mock import MagicMock, patch

    from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger

    save_path = str(tmp_path / "multi_rank")
    world_size = 4

    # Simulate each rank saving its own results
    for rank in range(world_size):
        with (
            patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_rank', return_value=rank),
            patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_world_size', return_value=world_size),
            patch('torch.distributed.is_available', return_value=True),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.barrier'),
        ):  # Mock barrier to avoid actual distributed ops

            logger = ResultsLogger(save_path=save_path)

            # Each rank processes different samples
            start_idx = rank * 2
            logger.update(
                name="test_dataset",
                refs=[f"ref_{start_idx}", f"ref_{start_idx+1}"],
                hyps=[f"hyp_{start_idx}", f"hyp_{start_idx+1}"],
                asr_hyps=[None, None],
                samples_id=[f"sample_{start_idx}", f"sample_{start_idx+1}"],
                pred_audio=None,
                pred_audio_sr=16000,
                user_audio=None,
                user_audio_sr=16000,
                src_refs=[f"src_{start_idx}", f"src_{start_idx+1}"],
                src_hyps=["", ""],
            )

            # Save rank-specific files (only this part, not the merge)
            rank_json_path = os.path.join(save_path, "metadatas", f"test_dataset_rank{rank}.json")
            os.makedirs(os.path.dirname(rank_json_path), exist_ok=True)
            with open(rank_json_path, 'w', encoding='utf-8') as fout:
                for item in logger.cached_results["test_dataset"]:
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Now simulate rank 0 merging all results
    with (
        patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_rank', return_value=0),
        patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_world_size', return_value=world_size),
        patch('torch.distributed.is_available', return_value=True),
        patch('torch.distributed.is_initialized', return_value=True),
        patch('torch.distributed.barrier'),
        patch('torch.distributed.broadcast_object_list'),
    ):  # Mock broadcast

        logger = ResultsLogger(save_path=save_path)
        # Manually set cached_results to simulate what rank 0 would have
        logger.cached_results["test_dataset"] = []

        # Call merge function directly
        merged_results = logger._merge_rank_files("test_dataset")

        # Verify all samples from all ranks are merged
        assert len(merged_results) == world_size * 2  # 4 ranks * 2 samples each

        # Verify sample IDs are from all ranks
        sample_ids = [item["id"] for item in merged_results]
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids
        assert "sample_6" in sample_ids  # Last rank's samples
        assert "sample_7" in sample_ids


def test_results_logger_rank_file_wait(tmp_path):
    """Test that rank 0 waits for other ranks' files"""
    import json
    import os
    import threading
    import time
    from unittest.mock import patch

    from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger

    save_path = str(tmp_path / "wait_test")
    os.makedirs(os.path.join(save_path, "metadatas"), exist_ok=True)
    world_size = 2

    # Create rank 0's file immediately
    rank0_file = os.path.join(save_path, "metadatas", "test_dataset_rank0.json")
    with open(rank0_file, 'w') as f:
        json.dump({"id": "sample_0", "pred_text": "test0"}, f)
        f.write('\n')

    # Simulate rank 1's file appearing after a delay
    def create_rank1_file_delayed():
        time.sleep(1)  # 1 second delay
        rank1_file = os.path.join(save_path, "metadatas", "test_dataset_rank1.json")
        with open(rank1_file, 'w') as f:
            json.dump({"id": "sample_1", "pred_text": "test1"}, f)
            f.write('\n')

    thread = threading.Thread(target=create_rank1_file_delayed)
    thread.start()

    # Rank 0 tries to merge - should wait for rank 1's file
    with (
        patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_rank', return_value=0),
        patch('nemo.collections.speechlm2.parts.metrics.results_logger.get_world_size', return_value=world_size),
    ):

        logger = ResultsLogger(save_path=save_path)
        merged_results = logger._merge_rank_files("test_dataset")

        # Should have results from both ranks
        assert len(merged_results) == 2
        sample_ids = [item["id"] for item in merged_results]
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids

    thread.join()


def test_intelligibility():
    metric = Intelligibility(pretrained_asr=None, verbose=False, reuse_asr_hyps=True)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        asr_hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
        pred_audio=None,
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        asr_hyps=["a b d"],
        pred_audio=None,
    )
    ans = metric.compute()
    # wer
    assert ans["wer_dataset_1"] == 0.0
    assert ans["wer_dataset_2"] == 1 / 3
    assert ans["wer"] == 1 / 6  # average across datasets
    # cer
    assert ans["cer_dataset_1"] == 0.0
    assert ans["cer_dataset_2"] == 1 / 5
