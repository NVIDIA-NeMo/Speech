from torch import nn
from typing_extensions import Self

from nemo.collections.llm.fn import base as fn
from nemo.utils import logging


class FNMixin:
    """
    A mixin class providing utility methods for operating on PyTorch modules.

    This mixin class offers methods to apply functions, check predicates, and modify
    the state (freeze/unfreeze) of PyTorch modules within a container. It is designed
    to be used with classes that are composed of multiple PyTorch modules, facilitating
    operations that affect all contained modules either directly or recursively.

    Methods
    -------
        forall: Checks if a predicate holds for all modules.
        map: Applies a function to each module.
        walk: Traverses each module, applying a function.
        freeze: Freezes the parameters of all modules.
        unfreeze: Unfreezes the parameters of all modules.

    Examples
    --------
        >>> class MyModel(nn.Module, FNMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer1 = nn.Linear(10, 10)
        ...         self.layer2 = nn.Linear(10, 10)
        ...
        >>> model = MyModel()
        >>> model.freeze()  # Freezes all parameters in the model
        >>> model.forall(lambda module: not module.parameters().requires_grad, recurse=True)
        True
    """

    def forall(self, func: fn.ModulePredicate, recurse: bool = False) -> bool:
        """
        Evaluates a predicate for all modules in the container, optionally recursively.

        This method checks if a given predicate holds for all modules in the container.
        If `recurse` is True, it also checks all submodules recursively.

        Args:
            func (fn.ModulePredicate): A predicate function to apply to each module.
            recurse (bool, optional): Whether to apply the predicate recursively. Defaults to False.

        Returns
        -------
            bool: True if the predicate holds for all modules, False otherwise.

        Example:
            >>> model = MyModel()
            >>> model.forall(lambda module: isinstance(module, nn.Linear), recurse=True)
            True
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        return fn.forall(self, func, recurse=recurse)

    def map(self, func: fn.ModuleFunc, leaf_only: bool = False) -> Self:
        """
        Applies a function to each module in the container, optionally to leaf modules only.

        This method applies a given function to each module in the container. If `leaf_only`
        is True, the function is applied to leaf modules only.

        Args:
            func (fn.ModuleFunc): A function to apply to each module.
            leaf_only (bool, optional): Whether to apply the function to leaf modules only. Defaults to False.

        Returns
        -------
            Self: The container itself after applying the function.

        Example:
            >>> model = MyModel()
            >>> model.map(lambda module: module.double() if isinstance(module, nn.Linear) else module)
            <MyModel object>
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        return fn.map(self, func, leaf_only=leaf_only, _skip_map=True)

    def walk(self, func: fn.ModuleFunc, leaf_only: bool = False) -> Self:
        """
        Traverses each module in the container, applying a function, optionally to leaf modules only.

        This method is similar to `map`, but it is typically used for operations that do not
        modify the modules but instead collect information or perform checks.

        Args:
            func (fn.ModuleFunc): A function to apply to each module.
            leaf_only (bool, optional): Whether to traverse leaf modules only. Defaults to False.

        Returns
        -------
            Self: The container itself after the traversal.

        Example:
            >>> model = MyModel()
            >>> model.walk(print, leaf_only=True)
            <MyModel object>
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        return fn.walk(self, func, leaf_only=leaf_only, _skip_map=True)

    def freeze(self) -> None:
        """
        Freezes the parameters of all modules in the container
        by setting `requires_grad` to False.
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        params = list(self.parameters())
        if not params:
            logging.info(f"No parameters found in module {self.__class__.__name__}")
        else:
            for param in params:
                param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Unfreezes the parameters of all modules in the container
        by setting `requires_grad` to True.
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        params = list(self.parameters())
        if not params:
            logging.info(f"No parameters found in module {self.__class__.__name__}")
        else:
            for param in params:
                param.requires_grad = True
