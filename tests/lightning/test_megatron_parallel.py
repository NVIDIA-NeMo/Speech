from collections import defaultdict

import pytest
from megatron.core import parallel_state
from torch import nn

from nemo import lightning as nl
from nemo.lightning import megatron_parallel as mp


class TestMegatronParallel:
    """Unit tests for the MegatronParallel class."""

    @pytest.fixture
    def mock_pipeline(self, mocker):
        """Fixture to create a mock pipeline."""

        class DummyModule(nn.Module):
            def __init__(self, dummy_arg=None):
                self.dummy_arg = dummy_arg
                super().__init__()

            def forward(self, x):
                return x

        return DummyModule()

    @pytest.fixture
    def mock_precision_plugin(self, mocker):
        """Fixture to create a mock precision plugin."""
        return nl.MegatronMixedPrecision(precision="bf16-mixed")

    @pytest.fixture
    def mock_callbacks(self, mocker):
        """Fixture to create a mock callback connector."""
        return mocker.MagicMock(spec=mp.CallbackConnector)

    @pytest.fixture
    def mock_data_step(self, mocker):
        """Fixture to create a mock data step function."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_forward_step(self, mocker):
        """Fixture to create a mock forward step function."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_loss_reduction(self, mocker):
        """Fixture to create a mock loss reduction function."""
        return mocker.MagicMock()

    def test_init_with_defaults(self, mocker, mock_pipeline):
        """Test __init__ with default parameters."""
        mocker.patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', return_value=1)
        mocker.patch('megatron.core.parallel_state.model_parallel_is_initialized', return_value=False)

        megatron_parallel = mp.MegatronParallel(pipeline=mock_pipeline)

        assert megatron_parallel.pipeline == mock_pipeline
        assert megatron_parallel.precision_plugin is None
        assert isinstance(megatron_parallel.callbacks, mp.CallbackConnector)
        assert megatron_parallel.data_step == mp.default_data_step
        assert megatron_parallel.forward_step == mp.default_forward_step
        assert megatron_parallel.loss_reduction is None

    def test_init_with_custom_parameters(
        self,
        mocker,
        mock_pipeline,
        mock_precision_plugin,
        mock_callbacks,
        mock_data_step,
        mock_forward_step,
        mock_loss_reduction,
    ):
        """Test __init__ with custom parameters."""
        mocker.patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', return_value=1)
        mocker.patch('megatron.core.parallel_state.model_parallel_is_initialized', return_value=False)

        megatron_parallel = mp.MegatronParallel(
            pipeline=mock_pipeline,
            precision_plugin=mock_precision_plugin,
            callbacks=mock_callbacks,
            data_step=mock_data_step,
            forward_step=mock_forward_step,
            loss_reduction=mock_loss_reduction,
        )

        assert megatron_parallel.pipeline == mock_pipeline
        assert megatron_parallel.precision_plugin == mock_precision_plugin
        assert megatron_parallel.callbacks == mock_callbacks
        assert megatron_parallel.data_step == mock_data_step
        assert megatron_parallel.forward_step == mock_forward_step
        assert megatron_parallel.loss_reduction == mock_loss_reduction

    def test_init_with_virtual_pipeline(self, mocker, mock_pipeline):
        """Test __init__ with virtual pipeline model parallel world size."""
        mocker.patch('torch.distributed.get_rank', return_value=1)
        mocker.patch('megatron.core.parallel_state.get_tensor_model_parallel_group', return_value=1)
        mocker.patch('megatron.core.parallel_state.get_pipeline_model_parallel_group', return_value=1)
        mocker.patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', return_value=2)
        mocker.patch('megatron.core.parallel_state.model_parallel_is_initialized', return_value=True)
        mocker.patch('megatron.core.parallel_state.set_virtual_pipeline_model_parallel_world_size')
        mocker.patch('megatron.core.parallel_state.set_virtual_pipeline_model_parallel_rank')
        mocker.patch('nemo.lightning.io.reinit', return_value=mock_pipeline)

        megatron_parallel = mp.MegatronParallel(mock_pipeline, vp_size=2, cpu=True)

        assert len(megatron_parallel.pipeline) == 2
        assert all(isinstance(mod, nn.Module) for mod in megatron_parallel.pipeline)
        parallel_state.set_virtual_pipeline_model_parallel_world_size.assert_called_once_with(2)
        assert parallel_state.set_virtual_pipeline_model_parallel_rank.call_count == 1


class TestCallbackConnector:
    def test_add_callbacks(self) -> None:
        callback_connector = mp.CallbackConnector()
        callback = TestCallback()
        callback_connector.add(callback)

        assert callback in callback_connector.callbacks["on_megatron_step_start"]
        assert callback in callback_connector.callbacks["on_megatron_microbatch_start"]

    def test_event(self, mocker) -> None:
        callback_connector = mp.CallbackConnector()
        callback = TestCallback()
        callback_connector.add(callback)

        mocker.spy(callback, "on_megatron_step_start")
        mocker.spy(callback, "on_megatron_microbatch_start")

        callback_connector.event("on_megatron_step_start")
        callback_connector.event("on_megatron_microbatch_start")

        assert callback.on_megatron_step_start.call_count == 1
        assert callback.on_megatron_microbatch_start.call_count == 1

    def test_add_connector(self) -> None:
        callback_connector1 = mp.CallbackConnector()
        callback_connector2 = mp.CallbackConnector()
        callback1 = TestCallback()
        callback2 = TestCallback()

        callback_connector1.add(callback1)
        callback_connector2.add(callback2)

        callback_connector1 += callback_connector2

        assert callback1 in callback_connector1.callbacks["on_megatron_step_start"]
        assert callback2 in callback_connector1.callbacks["on_megatron_step_start"]

    def test_contains(self):
        callback_connector = mp.CallbackConnector()
        callback = TestCallback()
        callback_connector.add(callback)

        assert callback in callback_connector

    def test_add_count_callback(self):
        """Test adding a CountCallback to the CallbackConnector."""
        connector = mp.CallbackConnector()
        count_callback = CountCallback()
        connector.add(count_callback)

        # Check if the CountCallback has been added correctly
        assert count_callback in connector, "CountCallback should be in the CallbackConnector"

    def test_event_trigger_with_count_callback(self):
        """Test if the event triggers the method in CountCallback."""
        connector = mp.CallbackConnector()
        count_callback = CountCallback()
        connector.add(count_callback)

        # Simulate an event that CountCallback listens to
        connector.event('on_megatron_step_start')

        # Check if the CountCallback's method was called
        assert (
            count_callback.counts["on_megatron_step_start"] == 1
        ), "CountCallback's method should have been triggered once"


class TestCallback:
    def on_megatron_step_start(self):
        pass

    def on_megatron_microbatch_start(self):
        pass


class CountCallback:
    def __init__(self) -> None:
        self.counts = defaultdict(int)

    def on_megatron_step_start(self, *args, **kwargs) -> None:
        # assert len(kwargs) == 12
        self.counts["on_megatron_step_start"] += 1

    def on_megatron_microbatch_start(self, *args, **kwargs) -> None:
        # assert len(kwargs) == 14
        self.counts["on_megatron_microbatch_start"] += 1

    def on_megatron_microbatch_callback(self, *args, **kwargs) -> None:
        self.counts["on_megatron_microbatches_callback"] += 1

    def on_megatron_microbatch_end(self, *args, **kwargs) -> None:
        self.counts["on_megatron_microbatches_end"] += 1

    def on_megatron_reduce_microbatches_start(self, *args, **kwargs) -> None:
        self.counts["on_megatron_reduce_microbatches_start"] += 1

    def on_megatron_reduce_microbatches_end(self, *args, **kwargs) -> None:
        self.counts["on_megatron_reduce_microbatches_end"] += 1

    def on_megatron_log_step_end(self, *args, **kwargs) -> None:
        self.counts["on_megatron_log_step_end"] += 1

    def on_megatron_step_end(self, *args, **kwargs) -> None:
        self.counts["on_megatron_step_end"] += 1
