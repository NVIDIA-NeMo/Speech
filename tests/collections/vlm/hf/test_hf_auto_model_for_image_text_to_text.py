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

from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor

from nemo.collections.vlm import HFAutoModelForImageTextToText
from nemo.lightning.io.hf import HFCheckpointIO


@pytest.fixture
def mock_hf_calls():
    """
    Patches out all Hugging Face calls so no network or file I/O happens.
    """
    with (
        patch("transformers.AutoModelForImageTextToText.from_pretrained") as mock_from_pretrained,
        patch("transformers.AutoModelForImageTextToText.from_config") as mock_from_config,
        patch("transformers.AutoConfig.from_pretrained") as mock_from_auto_config,
        patch("transformers.AutoProcessor.from_pretrained") as mock_proc,
    ):
        # Return mock objects instead of real HF classes
        mock_from_pretrained.return_value = MagicMock()
        mock_from_config.return_value = MagicMock()
        mock_proc.return_value = MagicMock()
        mock_from_auto_config.return_value = MagicMock()

        yield  # Once we exit, the patches are removed


@pytest.fixture
def dummy_model(mock_hf_calls):
    """
    Creates a HFAutoModelForImageTextToText instance with all HF calls mocked out.
    """
    model = HFAutoModelForImageTextToText(
        model_name="mock-model",
        load_pretrained_weights=False,
        default_dtype=torch.float32,
    )
    return model


def test_configure_model(dummy_model):
    """
    Calls configure_model() to ensure it doesn't trigger real HF calls.
    """
    dummy_model.configure_model()
    # We expect dummy_model.model to be the MagicMock from our `mock_from_config.return_value`
    assert dummy_model.model is not None


def test_forward_pass(dummy_model):
    """
    Verifies forward() returns a mocked output.
    """
    # Configure with mocked HF object
    dummy_model.configure_model()

    batch = {"input_ids": torch.randint(0, 10, (2, 3))}
    # The mocked HF model should return another MagicMock
    dummy_model.model.return_value = MagicMock(logits=torch.randn(2, 3, 10))

    output = dummy_model.forward(batch)
    assert hasattr(output, "logits"), "Forward pass output should have 'logits'."


# def test_processor_property(dummy_model):
#     """Ensure calling `processor` property doesn't cause network download."""
#     with patch("nemo.collections.vlm.AutoProcessor.from_pretrained", return_value="MOCK_PROCESSOR"):
#         # Accessing property should invoke the static configure_processor,
#         # but we mock it to avoid any real Hugging Face call
#         processor = dummy_model.processor
#         assert processor == "MOCK_PROCESSOR"


def test_configure_model(dummy_model):
    """Ensure configure_model runs without performing external calls."""
    # We already patched from_pretrained/from_config, just call the method:
    dummy_model.configure_model()
    assert dummy_model.model is not None, "Expected model to be set after configure_model()"


def test_forward(dummy_model):
    """Test forward pass with a mocked HF model."""
    dummy_model.configure_model()

    # Create a mock input batch
    batch = {"input_ids": torch.randint(0, 10, (2, 5))}

    # We simulate the HF model output
    dummy_model.model.return_value = MagicMock(logits=torch.randn(2, 5, 10))

    output = dummy_model.forward(batch)
    # Ensure output contains 'logits' just like an HF model
    assert hasattr(output, "logits"), "Forward output should have 'logits' attribute."


@pytest.mark.parametrize("load_pretrained", [True, False])
def test_loading_flags(dummy_model, load_pretrained):
    """
    Quick check that toggling load_pretrained_weights won't break
    configure_model calls with the mocking in place.
    """
    dummy_model.load_pretrained_weights = load_pretrained
    try:
        dummy_model.configure_model()
    except MisconfigurationException:
        pytest.fail("Model configuration should not raise with mocking.")


def test_training_step(dummy_model):
    """
    Verifies training_step runs without requiring a real Trainer or Hugging Face model downloads.
    """
    # 1) Configure the HF mock model
    dummy_model.configure_model()

    # 2) Provide a valid torch.device to the mock so we avoid .to(MagicMock()) issues
    dummy_model.model.device = torch.device("cpu")

    # 3) Mock out the Trainer references used in your code
    dummy_model.trainer = MagicMock()
    dummy_model.trainer.strategy = MagicMock()
    dummy_model.trainer.strategy.checkpoint_io = MagicMock()

    # 4) Create a small training batch
    batch = {
        "input_ids": torch.randint(0, 10, (2, 5)),
        "labels": torch.randint(0, 10, (2, 5)),
        "loss_mask": torch.ones_like(torch.randint(0, 1, (2, 5)), dtype=torch.float32),
    }

    # 5) The model returns logits (e.g., shape [batch, seq, vocab_size])
    dummy_model.model.return_value = MagicMock(logits=torch.randn(2, 5, 10))

    # 6) Invoke training_step
    loss = dummy_model.training_step(batch, batch_idx=0)

    assert loss is not None, "Expected a loss from training_step."
    assert torch.is_tensor(loss), "Loss should be a PyTorch tensor."
    # If needed, you can also check shape, dtype, etc.


def test_validation_step(dummy_model):
    """Test validation_step with a real-tensor logits to avoid MagicMock shape issues."""
    # 1) Configure the model (mocks out HF calls).
    dummy_model.configure_model()

    # 2) Set the mock model's device so .to(self.model.device) is valid:
    dummy_model.model.device = torch.device("cpu")

    # 3) Mock the trainer references (if needed by your code).
    dummy_model.trainer = MagicMock()
    dummy_model.trainer.strategy = MagicMock()
    dummy_model.trainer.strategy.checkpoint_io = MagicMock()
    dummy_model.forward = MagicMock()
    dummy_model.forward.return_value.logits = torch.randn(2, 4, 12)

    # 4) Prepare a validation batch that has a shape consistent with the logits.
    # E.g., we want 2 batches of length 4, so flattening yields 8 total tokens.
    batch = {
        "input_ids": torch.randint(0, 10, (2, 4)),
        "labels": torch.randint(0, 10, (2, 4)),
        "loss_mask": torch.ones((2, 4), dtype=torch.float32),
    }

    # 5) Return a *real* tensor for logits instead of a MagicMock.
    # Suppose we want shape [2, 4, 12] so flattening yields (8, 12).
    # Then labels.view(-1) is size (8), so no dimension mismatch.
    mock_output = MagicMock()
    mock_output.logits = torch.randn(2, 4, 12)
    dummy_model.model.return_value = mock_output

    # 6) Now run validation_step. It should no longer fail on the shape assertion.
    dummy_model.validation_step(batch, batch_idx=0)


def test_save_pretrained(dummy_model, tmp_path):
    """Test saving logic; ensures no real HF endpoint is called."""
    dummy_model.configure_model()

    # Mock the model's state_dict
    dummy_model.model.state_dict = MagicMock(return_value={})
    # Actually call the method with a temp dir
    dummy_model.save_pretrained(tmp_path, state_dict={})

    # Check that model's save_pretrained was indeed called
    dummy_model.model.save_pretrained.assert_called_once()
