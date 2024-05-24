import functools
import logging
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

import fiddle as fdl

log = logging.getLogger(__name__)


def capture(to_capture: Optional[Callable] = None):
    if to_capture is None:
        return lambda f: capture(f)

    @functools.wraps(to_capture)
    def wrapper(*args, **kwargs):
        if isinstance(to_capture, IOProtocol):
            return to_capture(*args, **kwargs)

        output = to_capture(*args, **kwargs)
        if not hasattr(output, '__dict__'):
            try:
                if isinstance(output, (int, float, str, tuple)):
                    new_output = type_factory(type(output), base_value=output)
                else:
                    NewType = type_factory(type(output))
                    new_output = NewType(output)
                new_output.__io__ = fdl.Partial(to_capture, *args, **kwargs)
                output = new_output
            except Exception as e:
                logging.error(f"Error creating configurable type: {e}")
        else:
            output.__io__ = fdl.Partial(to_capture, *args, **kwargs)

        return output

    return wrapper


SelfT = TypeVar("SelfT", covariant=True)


@runtime_checkable
class IOProtocol(Protocol, Generic[SelfT]):
    @property
    def __io__(self) -> fdl.Config[SelfT]: ...


@runtime_checkable
class ReInitProtocol(Protocol, Generic[SelfT]):
    def reinit(self) -> SelfT: ...


def reinit(configurable: IOProtocol[SelfT]) -> SelfT:
    if isinstance(configurable, ReInitProtocol):
        return configurable.reinit()

    if not hasattr(configurable, '__io__'):
        raise ValueError(f"Cannot reinit {configurable} because it does not have a __io__ attribute")

    return fdl.build(configurable.__io__)


# Global cache for dynamically created types
type_cache = {}


def type_factory(original_type, base_value=None):
    """
    Factory function to create or retrieve from cache a new type that can have additional attributes,
    even if the original type is immutable.

    Args:
        original_type: The type of the original output value.
        base_value: The base value to use for immutable types, if applicable.

    Returns
    -------
        A new type that inherits from the original type and can have additional attributes,
        or an instance of this new type if base_value is provided.
    """
    type_name = f"Configurable{original_type.__name__}"
    if type_name in type_cache:
        NewType = type_cache[type_name]
    else:
        NewType = type(f"Configurable{original_type.__name__}", (original_type,), {})
        type_cache[type_name] = NewType

    if base_value is not None:
        try:
            instance = NewType(base_value)
        except TypeError:
            logging.warning(f"Could not instantiate type {NewType.__name__} with base value.")
            instance = NewType()
        return instance

    return NewType
