# NeMo Lightning

The NeMo Lightning directory provides custom PyTorch Lightning-compatible objects for seamlessly training NeMo 2.0 models using PTL. NeMo 2.0 models
are implemented using [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). NeMo Lightning provides the bridge between higher-level, object-oriented PTL APIs and lower-level Megatron APIs.
For detailed tutorials and documentation on NeMo 2.0, refer to the [docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo_2.0/index.html).

Some of the helpful classes provided here include:
- [`Trainer`](./pytorch/trainer.py): A lightweight wrapper around PTL's `Trainer` object which provides some additional support for capturing the arguments used to initialized the trainer. More information on NeMo 2's serialization mechanisms is available [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo_2.0/design/serialization.html).
- [`MegatronStrategy`](./pytorch/strategies.py): A PTL strategy that enables training of Megatron models on NVIDIA GPUs.
- [`MegatronParallel`](./megatron_parallel.py): Class which sets up and manages Megatron's distributed model parallelism.
- [`MegatronMixedPrecision`](./pytorch/plugins/mixed_precision.py): A specialized precision plugin for training Megatron-based models in PTL.

More information on `MegatronStrategy`, `MegatronParallel`, and `MegatronMixedPrecision` can be found in [this document](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo_2.0/design/megatron.html).
