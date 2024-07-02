from functools import singledispatch
from typing import Any, TypeVar

from lightning_fabric import plugins as fl_plugins
from lightning_fabric import strategies as fl_strategies
from pytorch_lightning import plugins as pl_plugins
from pytorch_lightning import strategies as pl_strategies

T = TypeVar('T')
FabricT = TypeVar('FabricT')


@singledispatch
def to_fabric(obj: Any) -> Any:
    """
    Convert a PyTorch Lightning object to its Fabric equivalent.

    Args:
        obj: The object to convert.

    Returns:
        The Fabric equivalent of the input object.

    Raises:
        NotImplementedError: If no converter is registered for the object's type.

    Example:
        >>> from pytorch_lightning.strategies import Strategy as PLStrategy
        >>> from lightning_fabric.strategies import Strategy as FabricStrategy
        >>> from nemo.lightning.fabric.conversion import to_fabric
        >>>
        >>> # Define a custom PyTorch Lightning strategy
        >>> class CustomPLStrategy(PLStrategy):
        ...     def __init__(self, custom_param: str):
        ...         super().__init__()
        ...         self.custom_param = custom_param
        >>>
        >>> # Define a custom Fabric strategy
        >>> class CustomFabricStrategy(FabricStrategy):
        ...     def __init__(self, custom_param: str):
        ...         super().__init__()
        ...         self.custom_param = custom_param
        >>>
        >>> # Register a custom conversion
        >>> @to_fabric.register(CustomPLStrategy)
        ... def _custom_converter(strategy: CustomPLStrategy) -> CustomFabricStrategy:
        ...     return CustomFabricStrategy(custom_param=strategy.custom_param)
        >>>
        >>> # Use the custom conversion
        >>> pl_strategy = CustomPLStrategy(custom_param="test")
        >>> fabric_strategy = to_fabric(pl_strategy)
        >>> assert isinstance(fabric_strategy, CustomFabricStrategy)
        >>> assert fabric_strategy.custom_param == "test"
    """
    raise NotImplementedError(
        f"No Fabric converter registered for {type(obj).__name__}. "
        f"To register a new conversion, use the @to_fabric.register decorator:\n\n"
        f"from nemo.lightning.fabric.conversion import to_fabric\n"
        f"from lightning_fabric import strategies as fl_strategies\n\n"
        f"@to_fabric.register({type(obj).__name__})\n"
        f"def _{type(obj).__name__.lower()}_converter(obj: {type(obj).__name__}) -> fl_strategies.Strategy:\n"
        f"    return fl_strategies.SomeStrategy(\n"
        f"        # Map relevant attributes from 'obj' to Fabric equivalent\n"
        f"        param1=obj.param1,\n"
        f"        param2=obj.param2,\n"
        f"        # ... other parameters ...\n"
        f"    )\n\n"
        f"Add this code to the appropriate module (e.g., nemo/lightning/fabric/conversion.py)."
    )


@to_fabric.register(pl_strategies.DDPStrategy)
def _ddp_converter(strategy: pl_strategies.DDPStrategy) -> fl_strategies.DDPStrategy:
    return fl_strategies.DDPStrategy(
        accelerator=strategy.accelerator,
        parallel_devices=strategy.parallel_devices,
        cluster_environment=strategy.cluster_environment,
        process_group_backend=strategy.process_group_backend,
        timeout=strategy._timeout,
        start_method=strategy._start_method,
        **strategy._ddp_kwargs,
    )


@to_fabric.register(pl_strategies.FSDPStrategy)
def _fsdp_converter(strategy: pl_strategies.FSDPStrategy) -> fl_strategies.FSDPStrategy:
    return fl_strategies.FSDPStrategy(
        cpu_offload=strategy.cpu_offload,
        parallel_devices=strategy.parallel_devices,
        cluster_environment=strategy.cluster_environment,
        process_group_backend=strategy.process_group_backend,
        timeout=strategy._timeout,
        **strategy.kwargs,
    )


@to_fabric.register(pl_plugins.MixedPrecision)
def _mixed_precision_converter(plugin: pl_plugins.MixedPrecision) -> fl_plugins.MixedPrecision:
    return fl_plugins.MixedPrecision(
        precision=plugin.precision,
        device=plugin.device,
        scaler=plugin.scaler,
    )


@to_fabric.register(pl_plugins.FSDPPrecision)
def _fsdp_precision_converter(plugin: pl_plugins.FSDPPrecision) -> fl_plugins.FSDPPrecision:
    return fl_plugins.FSDPPrecision(
        precision=plugin.precision,
    )
