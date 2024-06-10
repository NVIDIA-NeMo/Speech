import inspect
from typing import Callable, Iterable, Protocol, TypeVar, Union, runtime_checkable

from torch import nn


@runtime_checkable
class HasBool(Protocol):
    def __bool__(self) -> bool: ...


_TModule = TypeVar("_TModule", bound=nn.Module)
ModuleFunc = Callable[[nn.Module], nn.Module]
ModulePredicate = Callable[[nn.Module], Union[bool, HasBool]]


def map(  # noqa: A001
    module: _TModule,
    func: ModuleFunc,
    leaf_only: bool = False,
    **kwargs,
) -> _TModule:
    """Applies a function to a PyTorch module or a collection of modules.

    This function can be used to modify modules in place, such as changing their attributes,
    applying normalization, or any other custom transformations. It supports individual modules,
    lists of modules, and dictionaries of modules. The function can be applied selectively to
    modules that do not have parameters if `leaf_only` is set to True.

    Args:
        module: The module or collection of modules to which the function will be applied.
        func: A callable that takes a module (and optionally additional keyword arguments) and
              returns a transformed module. The signature should be `func(module, **kwargs)`.
        leaf_only: If True, the function will only be applied to modules that
                                    do not have any parameters. Defaults to False.
        **kwargs: Additional keyword arguments that will be passed to `func`.

    Returns
    -------
        The transformed module or collection of modules.

    Examples
    --------
        >>> import torch
        >>> import torch.nn as nn
        >>> from nemo.collections.llm import fn

        # Example: Doubling the weights of all Linear layers in a model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        def double_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data *= 2
            return m
        model = fn.map(model, double_weights)
        print(model)

    """
    if not kwargs.pop("_skip_map", False) and hasattr(module, "map"):
        return module.map(func, leaf_only=leaf_only, **kwargs)

    elif isinstance(module, Iterable):
        if all(hasattr(module, key) for key in ["items", "values", "keys"]):
            return _map_module_dict(module, func, leaf_only=leaf_only, **kwargs)

        return _map_module_list(module, func, leaf_only=leaf_only, **kwargs)
    else:
        return _map_module(module, func, leaf_only=leaf_only, **kwargs)


def walk(
    module: _TModule,
    func: ModuleFunc,
    leaf_only: bool = False,
    **kwargs,
) -> _TModule:
    """Recursively apply a function to a module or collection.

    This function is similar to `map`, but it applies the function recursively to all child
    modules as well. This is useful for applying transformations that need to consider the
    module hierarchy.

    Args:
        module: The module or collection to recursively apply to.
        func: The function to apply.
        leaf_only: If True, only apply to modules without parameters. Defaults to False.
        **kwargs: Additional kwargs to pass to the function.

    Returns
    -------
        The transformed module or collection.

    Examples
    --------
        >>> import torch
        >>> import torch.nn as nn
        >>> from nemo.collections.llm import fn

        # Example: Setting the bias of all Conv2d layers to False
        model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 10, 5))
        def remove_bias(m):
            if isinstance(m, nn.Conv2d):
                m.bias = None
            return m
        model = fn.walk(model, remove_bias)
        print(model)
    """
    return map(
        module,
        func,
        recurse=True,
        leaf_only=leaf_only,
        **kwargs,
    )


def forall(module: nn.Module, func: ModulePredicate, recurse: bool = False) -> bool:
    """
    Checks if a predicate holds for all modules in a given module or its children, optionally
    recursively.

    This function iterates over all modules and applies a predicate function to determine if
    all modules satisfy a certain condition. If `recurse` is True, it checks all child modules
    recursively.

    Args:
        module (nn.Module): The root module to check.
        func (ModulePredicate): A predicate function that takes a module as input and returns
                                a boolean or an object that can be evaluated as a boolean.
        recurse (bool): If True, applies the predicate recursively to all child modules.
                        Defaults to False.

    Returns
    -------
        bool: True if all modules satisfy the predicate, False otherwise.

    Examples
    --------
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        >>> predicate = lambda m: isinstance(m, nn.Linear)
        >>> print(forall(model, predicate))
        False
        >>> print(forall(model, predicate, recurse=True))
        True
    """

    def apply_predicate(m):
        result = func(m)
        # Convert result to bool if it's not already a boolean (e.g., if it's an instance of HasBool)
        return bool(result)

    if recurse:
        # Apply the predicate to all modules recursively
        results = [apply_predicate(m) for m in module.modules()]
    else:
        # Apply the predicate only to the top-level module
        results = [apply_predicate(module)]

    return all(results)


def _map_module(
    module: _TModule, func: ModuleFunc, recurse=False, leaf_only=False, transformed_modules=None, **kwargs
) -> _TModule:
    """
    Applies a transformation function to a module and optionally to its child modules.

    Parameters
    ----------
    module : nn.Module
        The module to which the function will be applied.
    func : ModuleFunc
        The function that will be applied to the module.
    recurse : bool, optional
        Whether to apply the function recursively to child modules.
    leaf_only : bool, optional
        Whether to apply the function only to modules without parameters.
    transformed_modules : set, optional
        A set to keep track of modules that have already been transformed.
    **kwargs : dict
        Additional keyword arguments that will be passed to the transformation function.

    Returns
    -------
    nn.Module
        The transformed module.
    """
    if transformed_modules is None:
        transformed_modules = set()

    if id(module) in transformed_modules:
        return module

    new_module = module
    f_kwargs = _get_func_kwargs(func, **kwargs)

    if not leaf_only or list(module.parameters(recurse=False)):
        new_module = func(new_module, **f_kwargs)

    prefix = kwargs.get("name", "") if not kwargs.get("prefix", "") else f"{kwargs['prefix']}.{kwargs['name']}"
    kwargs.pop('i', None)
    kwargs.pop('name', None)
    kwargs.pop('prefix', None)

    for i, (name, child) in enumerate(module.named_children()):
        setattr(
            new_module,
            name,
            map(
                child,
                func,
                recurse=recurse,
                leaf_only=leaf_only,
                transformed_modules=transformed_modules,
                i=i,
                name=name,
                prefix=prefix,
                **kwargs,
            ),
        )

    transformed_modules.add(id(new_module))

    return new_module


def _map_module_list(
    module_list: _TModule, func: ModuleFunc, recurse=False, leaf_only=False, transformed_modules=None, **kwargs
) -> _TModule:
    if transformed_modules is None:
        transformed_modules = set()

    f_kwargs = _get_func_kwargs(func, **kwargs)
    if not leaf_only:
        module_list = func(module_list, **f_kwargs)

    mapped_modules = []
    prefix = kwargs.get("name", "") if not kwargs.get('prefix', "") else f"{kwargs['prefix']}.{kwargs['name']}"
    kwargs.pop('i', None)
    kwargs.pop('name', None)
    kwargs.pop('prefix', None)
    for i, module in enumerate(module_list):
        new_module = map(
            module,
            func,
            recurse=recurse,
            leaf_only=leaf_only,
            transformed_modules=transformed_modules,
            i=i,
            name=str(i),
            prefix=prefix,
            **kwargs,
        )
        mapped_modules.append(new_module)

    return _create_list_wrapper(module_list, mapped_modules)


def _map_module_dict(
    module_dict: _TModule,
    func: ModuleFunc,
    recurse: bool = False,
    leaf_only: bool = False,
    transformed_modules=None,
    **kwargs,
) -> _TModule:
    """
    Applies a transformation function to a ModuleDict of modules.

    Parameters
    ----------
    module_dict : nn.ModuleDict
        The ModuleDict of modules to which the function will be applied.
    func : ModuleFunc
        The function that will be applied to the modules.
    recurse : bool, optional
        Whether to apply the function recursively to child modules.
    parameterless_modules_only : bool, optional
        Whether to apply the function only to modules without parameters.
    **kwargs : dict
        Additional keyword arguments that will be passed to the transformation function.

    Returns
    -------
    nn.ModuleDict
        The ModuleDict of transformed modules.
    """
    if transformed_modules is None:
        transformed_modules = set()

    f_kwargs = _get_func_kwargs(func, **kwargs)
    if not leaf_only:
        module_dict = func(module_dict, **f_kwargs)

    mapped_modules = {}
    for i, (name, module) in enumerate(module_dict.items()):
        kwargs["i"] = i
        kwargs["name"] = name

        mapped_modules[name] = map(
            module,
            func,
            recurse=recurse,
            leaf_only=leaf_only,
            transformed_modules=transformed_modules,
            **kwargs,
        )

    return type(module_dict)(mapped_modules)


def _create_list_wrapper(module_list, to_add):
    # Check the signature of the type constructor
    sig = inspect.signature(type(module_list).__init__)
    if "args" in sig.parameters:
        return type(module_list)(*to_add)  # Unpack new_modules

    return type(module_list)(to_add)  # Don't unpack new_modules


def _get_func_kwargs(func, **kwargs):
    sig = inspect.signature(func)
    return {kwarg: value for kwarg, value in kwargs.items() if kwarg in sig.parameters}
