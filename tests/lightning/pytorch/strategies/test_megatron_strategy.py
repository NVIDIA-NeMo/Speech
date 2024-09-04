from unittest.mock import patch

from nemo.lightning.pytorch.strategies import MegatronStrategy


class TestMegatronStrategy:
    @patch('nemo.lightning.pytorch.strategies.megatron_strategy.create_checkpoint_io')
    def test_checkpoint_io(self, mock_create_checkpoint_io):
        class Dummy: ...

        mock_create_checkpoint_io.side_effect = lambda *args, **kwargs: Dummy()
        strategy = MegatronStrategy()

        first_io = strategy.checkpoint_io
        mock_create_checkpoint_io.assert_called_once()

        assert first_io == strategy.checkpoint_io

        new_io = object()
        strategy.checkpoint_io = new_io
        assert new_io == strategy.checkpoint_io

        strategy2 = MegatronStrategy()
        second_io = strategy2.checkpoint_io
        mock_create_checkpoint_io.assert_called()

        assert first_io != second_io
        assert second_io == strategy2.checkpoint_io
