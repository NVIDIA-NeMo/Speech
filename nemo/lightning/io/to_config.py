# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Callable, TypeVar

import fiddle as fdl

T = TypeVar("T")


class PredicateDispatch:
    """A dispatcher that routes values to handlers based on predicates.

    This class implements a predicate-based dispatch system where handlers are registered
    with conditions that determine when they should be used. When called, it tries each
    predicate in order until it finds a match.

    Example:
        ```python
        dispatcher = PredicateDispatch()

        @dispatcher.register(lambda x: isinstance(x, str))
        def handle_strings(s):
            return f"String: {s}"

        result = dispatcher("hello")  # Returns "String: hello"
        ```
    """

    def __init__(self):
        """Initialize an empty handler registry."""
        self.handlers = []

    def register(self, predicate: Callable[[Any], bool]):
        """Register a new handler with a predicate.

        Args:
            predicate: A callable that takes a value and returns True if the handler
                should process this value.

        Returns:
            A decorator function that registers the handler.
        """

        def decorator(func: Callable[[T], Any]):
            self.handlers.append((predicate, func))
            return func

        return decorator

    def __call__(self, value):
        """Process a value through the registered handlers.

        Args:
            value: The value to be processed.

        Returns:
            The processed value from the first matching handler, or the original value
            if no handlers match.
        """
        for predicate, handler in self.handlers:
            if predicate(value):
                return handler(value)
        return value  # default case: return unchanged

    def register_class(self, cls: type):
        """Register a handler for instances of a specific class.

        A convenience method that automatically creates an isinstance predicate.

        Args:
            cls: The class to check instances against.

        Returns:
            A decorator function that registers the handler.
        """
        return self.register(lambda v: isinstance(v, cls))


"""Global dispatcher for converting Python objects to Fiddle configurations.

This dispatcher is used by Fiddle's serialization system to handle special cases
during configuration serialization. When Fiddle encounters an object it doesn't
know how to serialize, it will pass it through this dispatcher to convert it
into a serializable Fiddle configuration.

Example use cases:
    - Converting functools.partial to fdl.Partial
    - Converting HuggingFace models to their from_pretrained configurations
    - Handling custom classes with special serialization needs

The dispatcher is extended by registering new handlers with predicates that
determine when they should be used. See PredicateDispatch for more details.
"""
to_config = PredicateDispatch()


@to_config.register_class(partial)
def handle_partial(value: partial):
    """Convert functools.partial objects to Fiddle Partial configurations.

    This handler enables serialization of partial function applications by converting
    them to Fiddle's equivalent representation.

    Args:
        value: A functools.partial object.

    Returns:
        A Fiddle Partial configuration representing the same partial application.
    """
    return fdl.Partial(value.func, *value.args, **value.keywords)
