import logging
from typing import Any, Callable, Generic, TypeVar, Union, overload

T = TypeVar("T", bound=Callable[..., Any])

try:
    import nemo_run as run

    Config = run.Config
    Partial = run.Partial
except ImportError:
    logging.warning(
        "Trying to use Config or Partial, but NeMo-Run is not installed. Please install NeMo-Run before proceeding."
    )

    _T = TypeVar("_T")

    class Config(Generic[_T]):
        pass

    class Partial(Generic[_T]):
        pass


def task(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    try:
        import nemo_run as run

        return run.task(*args, **kwargs)
    except (ImportError, AttributeError):
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        return noop_decorator


@overload
def factory() -> Callable[[T], T]: ...


@overload
def factory(*args: Any, **kwargs: Any) -> Callable[[T], T]: ...


def factory(*args: Any, **kwargs: Any) -> Union[Callable[[T], T], T]:
    try:
        import nemo_run as run

        if not args:
            return run.factory(**kwargs)
        else:
            # Used as @factory(*args, **kwargs)
            return run.factory(*args, **kwargs)
    except (ImportError, AttributeError):
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        if not args and not kwargs:
            return noop_decorator
        else:
            return noop_decorator
