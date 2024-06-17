from typing import Any, Callable, Generic, TypeVar

T = TypeVar('T', bound=Callable[..., Any])

try:
    import nemo_sdk as sdk

    Config = sdk.Config
    Partial = sdk.Partial
except ImportError:
    _T = TypeVar('_T')

    class Config(Generic[_T]):
        pass

    class Partial(Generic[_T]):
        pass


def task(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    try:
        import nemo_sdk as sdk

        return sdk.task(*args, **kwargs)
    except ImportError:
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        return noop_decorator
