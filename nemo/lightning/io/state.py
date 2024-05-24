import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union, overload

import numpy as np
from torch import nn

SourceModuleT = TypeVar("SourceModuleT", bound=nn.Module)
TargetModuleT = TypeVar("TargetModuleT", bound=nn.Module)
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TransformCTX:
    source: nn.Module
    source_state: dict
    target: nn.Module
    target_state: dict


def apply_transforms(
    source: nn.Module,
    target: TargetModuleT,
    mapping: Dict[str, str],
    transforms: Optional[List[Callable[[TransformCTX], TransformCTX]]] = None,
) -> TargetModuleT:
    """
    Applies a series of transformations to adapt the state dictionary of a source module to
    match the structure of a target module's state dictionary.

    This function renames keys according to a provided mapping and modifies values using a list
    of transformation functions. Each transformation function typically is decorated
    with `io.state_transform`.

    Args:
        source (nn.Module): The source module from which parameters and buffers are taken.
        target (TargetModuleT): The target module to which parameters and buffers are adapted.
        mapping (Dict[str, str]): Key-value pairs where each key from the source state dictionary
            is mapped to a corresponding key in the target state dictionary.
        transforms (Optional[List[Callable[[TransformCTX], TransformCTX]]]): A list of functions
            that modify the `TransformCTX` object. If None, no transformations beyond key renaming
            are applied. Defaults to None.

    Returns
    -------
        TargetModuleT: The modified target module with its state dictionary adjusted according to
        the specified mappings and transformations.

    Raises
    ------
        ValueError: If there's a mismatch in shape between corresponding source and target parameters
            or buffers.
        RuntimeError: If the target state dictionary contains keys that are not present in the source
            state dictionary after all transformations.

    Examples
    --------
        >>> source_module = nn.Linear(10, 5)
        >>> target_module = nn.Linear(10, 5)
        >>> mapping = {'weight': 'weights', 'bias': 'biases'}
        @io.state_transform(
            source_key="weight",
            target_key="weights"
        )
        def scale_weights(ctx):
            ctx.target_state['weights'] = ctx.source_state['weight'] * 2
            return ctx
        >>> transformed_target = apply_transforms(
        ...     source_module, target_module, mapping, [scale_weights]
        ... )
        >>> print(transformed_target.state_dict()['weights'])

    See Also
    --------
        - `TransformCTX`: For more details on the context object used in transformations.
        - `StateDictTransform`: For creating complex transformations.

    Note:
        This function is particularly useful when adapting models from different frameworks or
        when consolidating models with different architectural changes.
    """
    from megatron.core.transformer.module import MegatronModule

    # TODO: How can we improve this?
    _source = source
    if hasattr(source, "module") and isinstance(source.module, MegatronModule):
        _source = source.module
    _target = target
    if hasattr(target, "module") and isinstance(target.module, MegatronModule):
        _target = target.module

    target_state = _target.state_dict()
    ctx = TransformCTX(
        source=_source,
        source_state=_source.state_dict(),
        target=_target,
        target_state=target_state,
    )

    for key, val in mapping.items():
        ctx = StateDictTransform(key, val)(ctx)

    if transforms:
        for transform in transforms:
            ctx = transform(ctx)

    _params: Dict[str, nn.Parameter] = {}
    for name, param in _target.named_parameters():
        if name in target_state:
            target_param = target_state[name]
            if param.data.shape != target_param.shape:
                raise ValueError(f"Shape mismatch for parameter {name}: {param.shape} vs {target_param.shape}")

            _params[name] = nn.Parameter(target_param, requires_grad=param.requires_grad)
            target_state.pop(name)
        else:
            print(f"Unexpected key: {name} not in checkpoint but in model.")

    for key, val in _params.items():
        _module, _key = _target, key
        if "." in key:
            for part in key.split(".")[:-1]:
                _module = getattr(_module, part)
            _key = key.split(".")[-1]

        _module.register_parameter(_key, val)

    _buffers = {}
    for name, buffer in _target.named_buffers():
        if name in target_state:
            if buffer.shape != target_state[name].shape:
                raise ValueError(f"Shape mismatch for buffer {name}: {buffer.shape} vs {target_state[name].shape}")

            _buffers[name] = nn.Parameter(target_state[name], requires_grad=False)
            target_state.pop(name)

    for key, val in _buffers.items():
        _module, _key = _target, key
        if "." in key:
            for part in key.split(".")[:-1]:
                _module = getattr(_module, part)
            _key = key.split(".")[-1]

        _module.register_buffer(_key, val)

    keys = [name for name in list(target_state.keys()) if not name.endswith("_extra_state")]
    if len(keys) != 0:
        raise RuntimeError(f"Additional keys: {target_state.keys()} in checkpoint but not in model.")

    # TODO: Is this correct?
    # for key in target.state_dict():
    #     if key.endswith("_extra_state"):
    #         del target.state_dict()[key]

    """finally:
        cls._set_model_restore_state(is_being_restored=False)"""

    if hasattr(target, "module") and isinstance(target.module, MegatronModule):
        target.module = _target

        return target

    return _target


def _default_transform(inp):
    return inp.float()


class StateDictTransform(Generic[F]):
    """
    A transformation class for state dictionaries, allowing for flexible key matching and
    transformation of values between source and target state dictionaries.

    Attributes
    ----------
        source_key: A string, tuple of strings, or a dictionary specifying the keys in the source
            state dictionary to match. Wildcards (*) are supported.
        target_key: A string or tuple of strings specifying the keys in the target state dictionary
            to match. Wildcards (*) are supported.
        transform: A callable that performs the transformation on matched keys' values.

    Examples
    --------
        >>> def example_transform(ctx, *args):
        ...     return sum(args)
        >>> transform = StateDictTransform(
        ...     source_key="model.layers.*.self_attn.*_proj.weight",
        ...     target_key="decoder.layers.*.self_attention.linear_qkv.weight",
        ...     transform=example_transform
        ... )
    """

    def __init__(
        self,
        source_key: Union[str, Tuple[str, ...], Dict[str, str]],
        target_key: Union[str, Tuple[str, ...]],
        transform: F = _default_transform,
    ):
        self.source_key = source_key
        self.target_key = target_key
        self.transform = transform

    def __call__(self, ctx: TransformCTX) -> TransformCTX:
        source_key = self.source_key
        target_key = self.target_key
        source_dict, target_dict = ctx.source_state, ctx.target_state

        fn_params = dict(inspect.signature(self.transform).parameters)
        fn_params.pop("ctx", None)

        if isinstance(source_key, (dict, tuple)):
            if isinstance(source_key, tuple):
                source_key_dict = {param: source_key[i] for i, param in enumerate(fn_params)}
            else:
                source_key_dict = source_key
            source_matches_dict = {k: _match_keys(list(source_dict.keys()), v) for k, v in source_key_dict.items()}
            target_matches = _match_keys(list(target_dict.keys()), target_key)

            for target_index, target_match in np.ndenumerate(target_matches):
                kwargs = {}
                for param in fn_params:
                    if param in source_matches_dict:
                        source_match = source_matches_dict[param][target_index[:-1]]
                        kwargs[param] = source_dict[source_match[target_index]]

                target_dict[target_match] = self.call_transform(ctx, **kwargs)
        else:
            source_keys = list(source_dict.keys())
            target_keys = list(target_dict.keys())

            source_matches = _match_keys(source_keys, source_key)
            if source_matches.size == 1 and source_matches == np.array(None):
                raise ValueError(f"No matches found for source key: {source_key}")

            if isinstance(target_key, str):
                target_matches = _match_keys(target_keys, target_key)
                if target_matches.size < 1:
                    raise ValueError(f"No matches found for target key: {target_key}")
            else:
                if isinstance(target_key, dict):
                    raise ValueError("Target key must be a string or a tuple of strings.")

                _matches = np.vstack([_match_keys(target_keys, key) for key in target_key])
                target_matches = np.transpose(_matches)

            # Determine if we are dealing with multiple source matches or multiple target matches
            multiple_sources = source_matches.ndim >= target_matches.ndim
            accepts_var_args = any(
                param.kind == param.VAR_POSITIONAL for param in inspect.signature(self.transform).parameters.values()
            )

            if multiple_sources:
                for target_index, target_match in np.ndenumerate(target_matches):
                    source_match = source_matches[target_index]

                    if accepts_var_args:
                        source_values = [source_dict[k] for k in source_match]
                        target_dict[target_match] = self.call_transform(ctx, *source_values)
                    else:
                        _source_match_list = [source_match] if isinstance(source_match, str) else list(source_match)
                        if len(fn_params) != len(_source_match_list):
                            raise ValueError(
                                f"Mismatch between source and target keys: {source_match} vs {target_match}"
                            )

                        kwargs = {param: source_dict[k] for param, k in zip(fn_params, _source_match_list)}
                        target_dict[target_match] = self.call_transform(ctx, **kwargs)
            else:
                if source_matches.ndim == 0:
                    source_matches_list = [source_matches.item()]
                    source_matches = np.array(source_matches_list, dtype=object)
                else:
                    source_matches_list = list(source_matches)

                if source_matches.shape[0] != target_matches.shape[0]:
                    if target_matches.shape[0] == 1 and source_matches.shape[0] == target_matches.shape[1]:
                        source_matches_list = [source_matches_list]
                    else:
                        raise ValueError(
                            "Mismatch between source and target keys: {source_matches} vs {target_matches}"
                        )

                for source_index, source_match in enumerate(source_matches_list):
                    target_match = target_matches[source_index]
                    source_values = (
                        [source_dict[source_match]]
                        if np.isscalar(source_match)
                        else [source_dict[k] for k in source_match]
                    )
                    if accepts_var_args:
                        outputs = self.call_transform(ctx, *source_values)
                    else:
                        kwargs = {param: val for param, val in zip(fn_params, source_values)}
                        outputs = self.call_transform(ctx, **kwargs)

                    if isinstance(target_match, str):
                        target_dict[target_match] = outputs
                    else:
                        for i, t in enumerate(outputs):
                            target_dict[target_match[i]] = t

        return ctx

    def call_transform(self, ctx: TransformCTX, *args, **kwargs):
        func_params = inspect.signature(self.transform).parameters
        expected_num_args = len([p for p in func_params if p not in ['self', 'ctx']])
        provided_num_args = len(args) + len(kwargs)
        accepts_var_args = any(param.kind == param.VAR_POSITIONAL for param in func_params.values())

        if not accepts_var_args and provided_num_args != expected_num_args:
            raise ValueError(
                f"Expected {expected_num_args} arguments for the transformation function, but got {provided_num_args}."
            )

        if 'ctx' in func_params:
            return self.transform(ctx, *args, **kwargs)

        return self.transform(*args, **kwargs)


def _match_keys(keys: List[str], pattern: str) -> np.ndarray:
    regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
    wildcard_matches = [[] for _ in range(pattern.count("*"))]

    for key in keys:
        match = regex_pattern.match(key)
        if match:
            for i, group in enumerate(match.groups()):
                if group not in wildcard_matches[i]:
                    wildcard_matches[i].append(group)

    # Sort the wildcard matches to maintain consistent ordering
    for i in range(len(wildcard_matches)):
        wildcard_matches[i].sort(key=lambda x: int(x) if x.isdigit() else x)

    # Determine the shape of the output array based on the unique matches for each wildcard
    shape = [len(matches) for matches in wildcard_matches]

    # Initialize an empty array with the determined shape
    output_array = np.empty(shape, dtype=object)

    # Populate the array with the keys, now that we have the correct shape and ordering
    for key in keys:
        match = regex_pattern.match(key)
        if match:
            # Convert match groups to indices based on their position in wildcard_matches
            indices = [wildcard_matches[i].index(group) for i, group in enumerate(match.groups())]
            output_array[tuple(indices)] = key  # Place the key in the array based on the indices

    return output_array


@overload
def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]],
    target_key: Union[str, Tuple[str, ...]],
) -> Callable[[F], StateDictTransform[F]]: ...


@overload
def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]], target_key: Union[str, Tuple[str, ...]], fn: F
) -> StateDictTransform[F]: ...


def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]],
    target_key: Union[str, Tuple[str, ...]],
    fn: Optional[F] = None,
):
    """
    A decorator for creating StateDictTransform instances with specified source and target keys,
    and a transformation function. This allows for concise definition of state dictionary
    transformations.

    Args:
        source_key: A string, tuple of strings, or a dictionary specifying the keys in the source
            state dictionary to match. Wildcards (*) are supported.
        target_key: A string or tuple of strings specifying the keys in the target state dictionary
            to match. Wildcards (*) are supported.
        fn: An optional callable that performs the transformation on matched keys' values. If not
            provided, the decorator can be used to wrap a function definition.

    Returns
    -------
        A StateDictTransform instance if `fn` is provided, otherwise returns a decorator that
        takes a function and returns a StateDictTransform instance.

    Examples
    --------
        >>> @state_transform(
        ...     source_key="model.layers.*.self_attn.*_proj.weight",
        ...     target_key="decoder.layers.*.self_attention.linear_qkv.weight"
        ... )
        ... def sum_transform(ctx, *args):
        ...     return sum(args)
    """

    def wrapper(fn) -> StateDictTransform:
        return StateDictTransform(source_key, target_key, fn)

    if fn is None:
        return wrapper

    return wrapper(fn)
