from nemo import lightning as nl


class TestFabricConversion:
    def test_simple_conversion(self):
        trainer = nl.Trainer(
            devices=1,
            accelerator="cpu",
            strategy=nl.MegatronStrategy(tensor_model_parallel_size=2),
            plugins=nl.MegatronMixedPrecision(precision='16-mixed'),
        )

        fabric = trainer.to_fabric()

        assert isinstance(fabric.strategy, nl.FabricMegatronStrategy)
        assert fabric.strategy.tensor_model_parallel_size == 2
        assert isinstance(fabric._precision, nl.FabricMegatronMixedPrecision)
        assert fabric._precision.precision == '16-mixed'
