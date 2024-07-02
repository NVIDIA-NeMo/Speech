from copy import deepcopy
from pathlib import Path
from typing import Optional, Protocol, Type, TypeVar, Union, runtime_checkable

import fiddle as fdl
import lightning_fabric as lb
from torch import nn
from typing_extensions import Self, override

from nemo.lightning.io.mixin import IOMixin, serialization, track_io

ModelT = TypeVar("ModelT", bound=nn.Module)


class Fabric(lb.Fabric, IOMixin):
    def io_init(self, **kwargs) -> fdl.Config[Self]:
        # Each argument of the trainer can be stateful so we copy them
        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        for val in cfg_kwargs.values():
            if not serialization.find_node_traverser(type(val)):
                track_io(type(val))

        return fdl.Config(type(self), **cfg_kwargs)

    def load_model(
        self,
        path: Union[str, Path],
        model: Optional[ModelT] = None,
    ) -> "DistributedModel[ModelT]":
        """Load and set up a model for distributed training.

        This method loads a model from the given path, sets it up for distributed training
        using the current Fabric instance, and returns a DistributedModel.

        Args:
            path (Union[str, Path]): The path to the saved model checkpoint.
            model (Optional[ModelT], optional): An optional pre-instantiated model. If not
            provided, the model will be loaded from the checkpoint. Defaults to None.

        Returns:
            DistributedModel[ModelT]: The loaded and distributed model.

        Example:
            >>> from nemo import lightning as nl
            >>>
            >>> trainer = nl.Trainer(
            ...     devices=2,
            ...     strategy=nl.MegatronStrategy(tensor_model_parallel_size=2),
            ...     plugins=nl.MegatronMixedPrecision(precision='16-mixed')
            ... )
            >>> fabric = trainer.to_fabric()
            >>> distributed_model = fabric.load_model("path/to/checkpoint/dir")
            >>>
            >>> # You can now interact with the parallel model
        """
        self.launch()

        from nemo.lightning.io import load_context

        if model is None:
            context = load_context(path)
            model = context.model

        dist_model = self.setup_module(model)
        self.load(path, {"state_dict": dist_model})

        return dist_model

    def import_model(
        self,
        path: Union[str, Path],
        model_type: Type[ModelT],
    ) -> "DistributedModel[ModelT]":
        """
        Import a model from a given path and set it up for distributed training.

        This method imports a model of the specified type from the given path, loads it,
        and sets it up for distributed training using the current Fabric instance.

        Args:
            path (Union[str, Path]): The path to the model. Can be a local path or a
                Hugging Face model identifier.
            model_type (Type[ModelT]): The type of the model to import. Must be a subclass
                of ConnectorMixin.

        Returns:
            DistributedModel[ModelT]: The imported and distributed model.

        Raises:
            TypeError: If the provided model_type is not a subclass of ConnectorMixin.

        Example:
            >>> from nemo import lightning as nl
            >>> from nemo.collections.llm import MistralModel
            >>>
            >>> trainer = nl.Trainer(
            ...     devices=2,
            ...     strategy=nl.MegatronStrategy(tensor_model_parallel_size=2),
            ...     plugins=nl.MegatronMixedPrecision(precision='16-mixed')
            ... )
            >>> fabric = trainer.to_fabric()
            >>> model = fabric.import_model("hf://mistralai/Mistral-7B-v0.1", MistralModel)
            >>>
            >>> # You can now interact with the parallel model
        """
        from nemo.lightning.io import ConnectorMixin

        if not issubclass(model_type, ConnectorMixin):
            raise TypeError("The provided model class must be a subclass of ConnectorMixin")

        model: ModelT = model_type.import_from(path)

        return self.load_model(model.ckpt_path, model)

    @override
    def setup_module(self, module: nn.Module, move_to_device: bool = True, _reapply_compile: bool = True):
        from nemo.lightning.fabric.strategies import FabricMegatronStrategy

        out = super().setup_module(module, move_to_device=move_to_device, _reapply_compile=_reapply_compile)

        # We don't want to return a _FabricModule for megatron since we only want to precision convert
        # at the beginning and end of the pipeline
        if isinstance(self.strategy, FabricMegatronStrategy):
            return out._forward_module

        return out


@runtime_checkable
class DistributedModel(Protocol[ModelT]):
    module: ModelT
