"""Fiddle extensions to handle PyTorch code more elegantly.

This module provides extensions for better handling of PyTorch types and functions
in codegen, graphviz, and other debugging functions.
"""

import types
from functools import partial

import fiddle as fdl
import libcst as cst
import torch
import torch.nn as nn
from fiddle._src import daglish_extensions
from fiddle._src.codegen import import_manager, py_val_to_cst_converter, special_value_codegen
from fiddle._src.experimental import serialization


def _make_torch_importable(name: str) -> special_value_codegen.Importable:
    return special_value_codegen.SingleImportable("torch", lambda torch_name: f"{torch_name}.{name}")


_torch_type_importables = (
    (torch.bool, _make_torch_importable("bool")),
    (torch.uint8, _make_torch_importable("uint8")),
    (torch.int8, _make_torch_importable("int8")),
    (torch.int16, _make_torch_importable("int16")),
    (torch.int32, _make_torch_importable("int32")),
    (torch.int64, _make_torch_importable("int64")),
    (torch.float16, _make_torch_importable("float16")),
    (torch.bfloat16, _make_torch_importable("bfloat16")),
    (torch.float32, _make_torch_importable("float32")),
    (torch.float64, _make_torch_importable("float64")),
    (torch.complex64, _make_torch_importable("complex64")),
    (torch.complex128, _make_torch_importable("complex128")),
)

_torch_initializers = (
    nn.init.constant_,
    nn.init.dirac_,
    nn.init.xavier_normal_,
    nn.init.xavier_uniform_,
    nn.init.kaiming_normal_,
    nn.init.kaiming_uniform_,
    nn.init.normal_,
    nn.init.ones_,
    nn.init.orthogonal_,
    nn.init.uniform_,
    nn.init.zeros_,
)

_import_aliases = (("torch.nn.init", "from torch.nn import init"),)


def _make_torch_nn_importable(name: str) -> special_value_codegen.Importable:
    return special_value_codegen.SingleImportable("torch", lambda torch_mod_name: f"{torch_mod_name}.nn.{name}")


_nn_type_importables = (
    (nn.ReLU, _make_torch_nn_importable("ReLU")),
    (nn.GELU, _make_torch_nn_importable("GELU")),
    (nn.ReLU6, _make_torch_nn_importable("ReLU6")),
    (nn.SiLU, _make_torch_nn_importable("SiLU")),
    (nn.Sigmoid, _make_torch_nn_importable("Sigmoid")),
    (nn.SELU, _make_torch_nn_importable("SELU")),
    (nn.Hardtanh, _make_torch_nn_importable("Hardtanh")),
    (nn.Tanh, _make_torch_nn_importable("Tanh")),
)


def is_torch_tensor(value):
    """Returns true if `value` is a PyTorch Tensor."""
    return isinstance(value, torch.Tensor)


def convert_torch_tensor_to_cst(value, convert_child):
    return cst.Call(
        func=cst.Attribute(value=convert_child(torch), attr=cst.Name("tensor")),
        args=[
            cst.Arg(convert_child(value.tolist())),
            py_val_to_cst_converter.kwarg_to_cst("dtype", convert_child(value.dtype)),
        ],
    )


def enable():
    """Registers PyTorch fiddle extensions.

    This allows for things like nicer handling of torch dtypes.
    """
    for value, importable in _torch_type_importables:
        special_value_codegen.register_exact_value(value, importable)

    for value, importable in _nn_type_importables:
        special_value_codegen.register_exact_value(value, importable)

    for module_str, import_stmt in _import_aliases:
        import_manager.register_import_alias(module_str, import_stmt)

    py_val_to_cst_converter.register_py_val_to_cst_converter(is_torch_tensor)(convert_torch_tensor_to_cst)

    for dtype, _ in _torch_type_importables:
        daglish_extensions.register_immutable(dtype)
        lib, symbol = str(dtype).split(".")
        serialization.register_constant(lib, symbol, compare_by_identity=True)

    for init in _torch_initializers:
        daglish_extensions.register_immutable(init)
        daglish_extensions.register_function_with_immutable_return_value(init)

    # Monkey-patch the Serialization class to handle things like activation-functions
    def _modified_serialize(self, value, current_path, all_paths=None):
        if isinstance(value, types.BuiltinFunctionType):
            return self._pyref(value, current_path)
        if isinstance(value, partial):
            value = fdl.Partial(value.func, *value.args, **value.keywords)
        return self._original_serialize(value, current_path, all_paths)

    serialization.Serialization._original_serialize = serialization.Serialization._serialize
    serialization.Serialization._serialize = _modified_serialize
