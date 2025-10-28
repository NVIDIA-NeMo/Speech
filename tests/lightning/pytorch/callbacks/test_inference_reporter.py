"""Unit tests for InferenceReporter callback."""

from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.inference.inference_request import Status
from megatron.core.inference.sampling_params import SamplingParams

from nemo.lightning.pytorch.callbacks import inference_reporter
from nemo.lightning.pytorch.callbacks.inference_reporter import InferenceReporter


@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.global_step = 100
    trainer.logger = Mock()
    trainer.logger.experiment = Mock()
    return trainer


@pytest.fixture
def mock_pl_module():
    module = Mock()
    module.config.hidden_size = 768
    module.dtype = torch.float32
    module.module.module.vocab_size = 50000
    module.tokenizer.ids_to_text = lambda ids: f"text_{ids[0]}" if ids else ""
    module.tokenizer.detokenize = module.tokenizer.ids_to_text
    return module


@pytest.fixture
def callback():
    return InferenceReporter(
        checkpoint_name="test_ckpt",
        dataset_name="test_data",
        inference_batch_times_seqlen_threshold=2048,
        inference_max_seq_length=2048,
        sampling_params={"num_tokens_to_generate": 10, "top_k": 1},
    )


def test_init(callback):
    assert callback.checkpoint_name == "test_ckpt"
    assert callback.dataset_name == "test_data"
    assert callback.sample_idx == 0
    assert isinstance(callback.sampling_params, SamplingParams)
    assert callback.text_generation_controller is None


def test_setup(callback, mock_trainer, mock_pl_module):
    callback.setup(mock_trainer, mock_pl_module, "fit")
    assert hasattr(mock_pl_module.tokenizer, "detokenize")
    assert hasattr(mock_pl_module.tokenizer, "offsets")


@patch("torch.distributed.get_rank", return_value=0)
@patch("torch.distributed.broadcast")
def test_get_prompt_tokens_rank0(mock_broadcast, mock_rank, callback):
    batch = {
        "tokens": [torch.tensor([1, 2, 3])],
        "labels": [torch.tensor([4, 5, 6])],
    }

    with patch("torch.cuda.is_available", return_value=False):
        tokens = callback._get_prompt_tokens(batch)

    assert tokens == [1, 2, 3, 6]
    assert mock_broadcast.call_count == 2


@patch("torch.distributed.get_rank", return_value=1)
@patch("torch.distributed.broadcast")
def test_get_prompt_tokens_rank1(mock_broadcast, mock_rank, callback):
    batch = {"tokens": [torch.tensor([1, 2, 3])]}

    def broadcast_side_effect(tensor, src):
        if tensor.numel() == 1:
            tensor.fill_(4)
        else:
            tensor.copy_(torch.tensor([1, 2, 3, 6]))

    mock_broadcast.side_effect = broadcast_side_effect

    with patch("torch.cuda.is_available", return_value=False):
        tokens = callback._get_prompt_tokens(batch)

    assert len(tokens) == 4


def test_run_inference(callback, mock_pl_module):
    mock_result = Mock()
    mock_result.generated_tokens = torch.tensor([10, 11, 12])
    mock_result.prompt_log_probs = [0.1, 0.2, 0.3]
    mock_result.logits = None

    mock_controller = Mock()
    mock_controller.generate_all_output_tokens_static_batch.return_value = {"0": mock_result}
    callback.text_generation_controller = mock_controller

    tokens, logprobs, logits = callback._run_inference(mock_pl_module, [1, 2, 3])

    assert tokens == [10, 11, 12]
    assert logprobs == [0.1, 0.2, 0.3]
    assert logits is None

    call_args = mock_controller.generate_all_output_tokens_static_batch.call_args[0][0]
    request = call_args["0"]
    assert request.prompt_tokens == [1, 2, 3]
    assert request.status == Status.ACTIVE_BUT_NOT_GENERATING_TOKENS


@patch(f"{inference_reporter.__name__}.parallel_state")
def test_log_artifacts_skips_non_primary_ranks(mock_parallel_state, callback, mock_trainer):
    mock_parallel_state.get_tensor_model_parallel_rank.return_value = 1
    mock_parallel_state.get_data_parallel_rank.return_value = 0

    callback._log_artifacts(mock_trainer, 0, [1, 2], None, None, [1], "input", "output")

    mock_trainer.logger.experiment.log_artifact.assert_not_called()


@patch(f"{inference_reporter.__name__}.parallel_state")
def test_log_artifacts_logs_on_primary_rank(mock_parallel_state, callback, mock_trainer):
    mock_parallel_state.get_tensor_model_parallel_rank.return_value = 0
    mock_parallel_state.get_data_parallel_rank.return_value = 0
    callback.sample_idx = 5

    callback._log_artifacts(
        mock_trainer,
        batch_idx=2,
        generated_tokens=[10, 11],
        prompt_logprobs=[0.1, 0.2],
        prompt_logits=None,
        prompt_tokens=[1, 2],
        input_text="input",
        generated_text="output",
    )

    assert mock_trainer.logger.experiment.log_artifact.call_count == 5

    for call in mock_trainer.logger.experiment.log_artifact.call_args_list:
        file_path, artifact_path = call[0]
        assert "inference/validation/step_100/batch_2" in artifact_path
        assert "_5.json" in file_path


@patch(f"{inference_reporter.__name__}.parallel_state")
def test_log_artifacts_saves_to_disk_without_logger(mock_parallel_state, callback, mock_trainer, tmp_path):
    mock_parallel_state.get_tensor_model_parallel_rank.return_value = 0
    mock_parallel_state.get_data_parallel_rank.return_value = 0
    mock_trainer.logger = None
    callback.output_dir = str(tmp_path)
    callback.sample_idx = 3

    callback._log_artifacts(
        mock_trainer,
        batch_idx=1,
        generated_tokens=[10, 11],
        prompt_logprobs=[0.1, 0.2],
        prompt_logits=None,
        prompt_tokens=[1, 2],
        input_text="input",
        generated_text="output",
    )

    expected_path = tmp_path / "inference" / "validation" / "step_100" / "batch_1"
    assert (expected_path / "token_ids" / "token_ids_3.json").exists()
    assert (expected_path / "prompt_tokens" / "prompt_tokens_3.json").exists()


@patch(f"{inference_reporter.__name__}.parallel_state")
@patch("torch.distributed.get_rank", return_value=0)
@patch("torch.distributed.broadcast")
def test_on_validation_batch_end_integration(
    mock_broadcast,
    mock_rank,
    mock_parallel_state,
    callback,
    mock_trainer,
    mock_pl_module,
):
    mock_parallel_state.get_tensor_model_parallel_rank.return_value = 0
    mock_parallel_state.get_data_parallel_rank.return_value = 0

    mock_result = Mock()
    mock_result.generated_tokens = torch.tensor([10, 11])
    mock_result.prompt_log_probs = [0.1, 0.2]
    mock_result.logits = None

    mock_controller = Mock()
    mock_controller.generate_all_output_tokens_static_batch.return_value = {"0": mock_result}
    callback.text_generation_controller = mock_controller

    batch = {"tokens": [torch.tensor([1, 2, 3])], "labels": [torch.tensor([4, 5, 6])]}

    with patch("torch.cuda.is_available", return_value=False):
        with patch("lightning.seed_everything"):
            callback.on_validation_batch_end(mock_trainer, mock_pl_module, None, batch, 0, 0)

    assert callback.sample_idx == 1
    assert mock_trainer.logger.experiment.log_artifact.call_count > 0
