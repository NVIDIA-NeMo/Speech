from unittest.mock import MagicMock, call, patch

import torch.nn as nn
from pytorch_lightning.trainer.states import TrainerFn
from nemo.collections.llm import fn
from nemo.lightning.pytorch.callbacks.peft import PEFT, WrappedAdapterIO


class TestPEFT:
    class DummyPEFT(PEFT):
        def transform(self, module, name=None, prefix=None):
            return module  # No-op transform for testing

    class DummyModel(nn.Module, fn.FNMixin):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.conv = nn.Conv2d(3, 3, 3)

    def test_peft_call(self):
        model = self.DummyModel()
        peft = self.DummyPEFT()

        transformed_model = peft(model)

        assert transformed_model.linear.weight.requires_grad == False
        assert transformed_model.conv.weight.requires_grad == False

    def test_peft_setup(self):
        peft = self.DummyPEFT()
        trainer = MagicMock()
        pl_module = MagicMock()

        pl_module.model_transform = peft
        peft.setup(trainer, pl_module, "fit")

        assert isinstance(trainer.strategy._checkpoint_io, WrappedAdapterIO)
        assert peft.model_transform is not None
        assert peft._needs_to_call is True

    @patch('nemo.lightning.pytorch.callbacks.peft.logging')
    def test_peft_on_train_epoch_start_with_adapter(self, mock_logging):
        peft = self.DummyPEFT()
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.model_transform = peft
        trainer.state.fn = TrainerFn.FITTING  # Mock the trainer to be in FITTING state

        peft.setup(trainer, pl_module, "fit")

        assert peft.model_transform is not None
        assert peft._needs_to_call is True

        peft.wrapped_io = MagicMock()
        peft.wrapped_io.adapter_ckpt_path = "dummy_path"
        peft.wrapped_io.load_checkpoint.return_value = {"dummy_state": "dummy_value"}
        peft.on_train_epoch_start(trainer, pl_module)

        # Check for all expected log messages
        mock_logging.info.assert_has_calls(
            [
                call("Loading adapters from dummy_path"),
                call("Initializing model parallel"),
                call("Setting up optimizers"),
            ],
            any_order=True,
        )

        # Verify the number of calls
        assert mock_logging.info.call_count == 3

        trainer.strategy.load_model_state_dict.assert_called_once_with({"dummy_state": "dummy_value"}, strict=False)
        trainer.strategy.init_model_parallel.assert_called_once()
        trainer.strategy.setup_optimizers.assert_called_once_with(trainer)
