from typing import Any, Callable, TypeVar

T = TypeVar('T', bound=Callable[..., Any])


def task(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    try:
        import nemo_sdk as sdk

        return sdk.task(*args, **kwargs)
    except ImportError:
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        return noop_decorator
