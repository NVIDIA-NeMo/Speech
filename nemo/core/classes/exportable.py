# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC
from typing import List, Union

import torch
from pytorch_lightning.core.module import _jit_is_scripting

from nemo.core.classes import typecheck
from nemo.core.utils.neural_type_utils import get_dynamic_axes, get_io_names
from nemo.utils import logging
from nemo.utils.export_utils import (
    ExportFormat,
    augment_filename,
    get_export_format,
    parse_input_example,
    replace_for_export,
    verify_runtime,
    verify_torchscript,
    wrap_forward_method,
)

__all__ = ['ExportFormat', 'Exportable']


class Exportable(ABC):
    """
    This Interface should be implemented by particular classes derived from nemo.core.NeuralModule or nemo.core.ModelPT.
    It gives these entities ability to be exported for deployment to formats such as ONNX.

    Usage:
        # exporting pre-trained model to ONNX file for deployment.
        model.eval()
        model.to('cuda')  # or to('cpu') if you don't have GPU

        model.export('mymodel.onnx', [options])  # all arguments apart from `output` are optional.
    """

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def export(
        self,
        output: str,
        input_example=None,
        verbose=False,
        do_constant_folding=True,
        onnx_opset_version=None,
        check_trace: Union[bool, List[torch.Tensor]] = False,
        dynamic_axes=None,
        check_tolerance=0.01,
        export_modules_as_functions=False,
        keep_initializers_as_inputs=None,
    ):
        """
        Exports the model to the specified format. The format is inferred from the file extension of the output file.

        Args:
        output (str): Output file name. File extension be .onnx, .pt, or .ts, and is used to select export
            path of the model.
        input_example (list or dict): Example input to the model's forward function. This is used to
            trace the model and export it to ONNX/TorchScript. If the model takes multiple inputs, then input_example
            should be a list of input examples. If the model takes named inputs, then input_example
            should be a dictionary of input examples.
        verbose (bool): If True, will print out a detailed description of the model's export steps, along with
            the internal trace logs of the export process.
        do_constant_folding (bool): If True, will execute constant folding optimization on the model's graph
            before exporting. This is ONNX specific.
        onnx_opset_version (int): The ONNX opset version to export the model to. If None, will use a reasonable
            default version.
        check_trace (bool): If True, will verify that the model's output matches the output of the traced
            model, upto some tolerance.
        dynamic_axes (dict): A dictionary mapping input and output names to their dynamic axes. This is
            used to specify the dynamic axes of the model's inputs and outputs. If the model takes multiple inputs,
            then dynamic_axes should be a list of dictionaries. If the model takes named inputs, then dynamic_axes
            should be a dictionary of dictionaries. If None, will use the dynamic axes of the input_example
            derived from the NeuralType of the input and output of the model.
        check_tolerance (float): The tolerance to use when checking the model's output against the traced
            model's output. This is only used if check_trace is True. Note the high tolerance is used because
            the traced model is not guaranteed to be 100% accurate.
        export_modules_as_functions (bool): If True, will export the model's submodules as functions. This is
            ONNX specific.
        keep_initializers_as_inputs (bool): If True, will keep the model's initializers as inputs in the onnx graph.
            This is ONNX specific.

        Returns:
            A tuple of two outputs.
            Item 0 in the output is a list of outputs, the outputs of each subnet exported.
            Item 1 in the output is a list of string descriptions. The description of each subnet exported can be
            used for logging purposes.
        """
        all_out = []
        all_descr = []
        for subnet_name in self.list_export_subnets():
            model = self.get_export_subnet(subnet_name)
            out_name = augment_filename(output, subnet_name)
            out, descr, out_example = model._export(
                out_name,
                input_example=input_example,
                verbose=verbose,
                do_constant_folding=do_constant_folding,
                onnx_opset_version=onnx_opset_version,
                check_trace=check_trace,
                dynamic_axes=dynamic_axes,
                check_tolerance=check_tolerance,
                export_modules_as_functions=export_modules_as_functions,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
            )
            # Propagate input example (default scenario, may need to be overriden)
            if input_example is not None:
                input_example = out_example
            all_out.append(out)
            all_descr.append(descr)
            logging.info("Successfully exported {} to {}".format(model.__class__.__name__, out_name))
        return (all_out, all_descr)

    def _export(
        self,
        output: str,
        input_example=None,
        verbose=False,
        do_constant_folding=True,
        onnx_opset_version=None,
        check_trace: Union[bool, List[torch.Tensor]] = False,
        dynamic_axes=None,
        check_tolerance=0.01,
        export_modules_as_functions=False,
        keep_initializers_as_inputs=None,
    ):
        my_args = locals().copy()
        my_args.pop('self')

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        exportables = []
        for m in self.modules():
            if isinstance(m, Exportable):
                exportables.append(m)

        qual_name = self.__module__ + '.' + self.__class__.__qualname__
        format = get_export_format(output)
        output_descr = f"{qual_name} exported to {format}"

        # Pytorch's default opset version is too low, using reasonable latest one
        if onnx_opset_version is None:
            onnx_opset_version = 16

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Allow user to completely override forward method to export
            forward_method, old_forward_method = wrap_forward_method(self)

            # Set module mode
            with torch.inference_mode(), torch.no_grad(), torch.jit.optimized_execution(True), _jit_is_scripting():

                if input_example is None:
                    input_example = self.input_module.input_example()

                # Remove i/o examples from args we propagate to enclosed Exportables
                my_args.pop('output')
                my_args.pop('input_example')

                # Run (posibly overridden) prepare methods before calling forward()
                for ex in exportables:
                    ex._prepare_for_export(**my_args, noreplace=True)
                self._prepare_for_export(output=output, input_example=input_example, **my_args)

                input_list, input_dict = parse_input_example(input_example)
                input_names = self.input_names
                output_names = self.output_names
                output_example = tuple(self.forward(*input_list, **input_dict))

                if check_trace:
                    if isinstance(check_trace, bool):
                        check_trace_input = [input_example]
                    else:
                        check_trace_input = check_trace
                jitted_model = self
                if format == ExportFormat.TORCHSCRIPT:
                    jitted_model = torch.jit.trace_module(
                        self,
                        {"forward": tuple(input_list) + tuple(input_dict.values())},
                        strict=True,
                        check_trace=check_trace,
                        check_tolerance=check_tolerance,
                    )
                    jitted_model = torch.jit.freeze(jitted_model)
                    if verbose:
                        logging.info(f"JIT code:\n{jitted_model.code}")
                    jitted_model.save(output)
                    jitted_model = torch.jit.load(output)

                    if check_trace:
                        verify_torchscript(jitted_model, output, check_trace_input, check_tolerance)
                elif format == ExportFormat.ONNX:
                    # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
                    if dynamic_axes is None:
                        dynamic_axes = get_dynamic_axes(self.input_module.input_types_for_export, input_names)
                        dynamic_axes.update(get_dynamic_axes(self.output_module.output_types_for_export, output_names))
                    torch.onnx.export(
                        jitted_model,
                        input_example,
                        output,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        do_constant_folding=do_constant_folding,
                        dynamic_axes=dynamic_axes,
                        opset_version=onnx_opset_version,
                        keep_initializers_as_inputs=keep_initializers_as_inputs,
                        export_modules_as_functions=export_modules_as_functions,
                    )

                    if check_trace:
                        verify_runtime(self, output, check_trace_input, input_names, check_tolerance=check_tolerance)
                else:
                    raise ValueError(f'Encountered unknown export format {format}.')
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            if forward_method:
                type(self).forward = old_forward_method
            self._export_teardown()
        return (output, output_descr, output_example)

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set()

    @property
    def disabled_deployment_output_names(self):
        """Implement this method to return a set of output names disabled for export"""
        return set()

    @property
    def supported_export_formats(self):
        """Implement this method to return a set of export formats supported. Default is all types."""
        return set([ExportFormat.ONNX, ExportFormat.TORCHSCRIPT])

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        if not 'noreplace' in kwargs:
            replace_for_export(self)

    def _export_teardown(self):
        """
        Override this method for any teardown code after export.
        """
        pass

    @property
    def input_names(self):
        return get_io_names(self.input_module.input_types_for_export, self.disabled_deployment_input_names)

    @property
    def output_names(self):
        return get_io_names(self.output_module.output_types_for_export, self.disabled_deployment_output_names)

    @property
    def input_types_for_export(self):
        return self.input_types

    @property
    def output_types_for_export(self):
        return self.output_types

    def get_export_subnet(self, subnet=None):
        """
        Returns Exportable subnet model/module to export 
        """
        if subnet is None or subnet == 'self':
            return self
        else:
            return getattr(self, subnet)

    def list_export_subnets(self):
        """
        Returns default set of subnet names exported for this model
        First goes the one receiving input (input_example)
        """
        return ['self']

    def get_export_config(self):
        """
        Returns export_config dictionary
        """
        return getattr(self, 'export_config', {})

    def set_export_config(self, args):
        """
        Sets/updates export_config dictionary
        """
        ex_config = self.get_export_config()
        ex_config.update(args)
        self.export_config = ex_config
