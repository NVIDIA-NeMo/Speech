# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn


@pytest.mark.run_only_on('GPU')
class SimpleModel(nn.Module):
    @pytest.mark.run_only_on('GPU')
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    @pytest.mark.run_only_on('GPU')
    def forward(self, x):
        return self.relu(self.conv(x))


@pytest.mark.run_only_on('GPU')
class TestTensorRTLazyCompiler(unittest.TestCase):

    @pytest.mark.run_only_on('GPU')
    def setUp(self):
        self.model = SimpleModel()
        self.temp_dir = tempfile.mkdtemp()
        self.plan_path = os.path.join(self.temp_dir, "test_model.plan")

    @pytest.mark.run_only_on('GPU')
    def tearDown(self):
        if os.path.exists(self.plan_path):
            os.remove(self.plan_path)
        os.rmdir(self.temp_dir)

    @pytest.mark.run_only_on('GPU')
    def test_get_profile_shapes(self):
        from nemo.export.tensorrt_lazy_compiler import get_profile_shapes

        input_shape = [1, 3, 224, 224]
        dynamic_batchsize = [1, 4, 8]

        min_shape, opt_shape, max_shape = get_profile_shapes(input_shape, dynamic_batchsize)

        self.assertEqual(min_shape, [1, 3, 224, 224])
        self.assertEqual(opt_shape, [4, 3, 224, 224])
        self.assertEqual(max_shape, [8, 3, 224, 224])

        # Test with None dynamic_batchsize
        min_shape, opt_shape, max_shape = get_profile_shapes(input_shape, None)
        self.assertEqual(min_shape, input_shape)
        self.assertEqual(opt_shape, input_shape)
        self.assertEqual(max_shape, input_shape)

    @pytest.mark.run_only_on('GPU')
    def test_get_dynamic_axes(self):
        from nemo.export.tensorrt_lazy_compiler import get_dynamic_axes

        profiles = [{"input": [[1, 3, 224, 224], [4, 3, 224, 224], [8, 3, 224, 224]]}]

        dynamic_axes = get_dynamic_axes(profiles)
        self.assertEqual(dynamic_axes, {"input": [0]})

        # Test with empty profiles
        dynamic_axes = get_dynamic_axes([])
        self.assertEqual(dynamic_axes, {})

    @pytest.mark.run_only_on('GPU')
    @patch('nemo.export.tensorrt_lazy_compiler.trt_imported', True)
    @patch('nemo.export.tensorrt_lazy_compiler.polygraphy_imported', True)
    @patch('torch.cuda.is_available', return_value=True)
    def test_trt_compile_basic(self, mock_cuda_available):
        from nemo.export.tensorrt_lazy_compiler import trt_compile

        # Test basic compilation
        compiled_model = trt_compile(
            self.model,
            self.plan_path,
            args={"method": "onnx", "precision": "fp16", "build_args": {"builder_optimization_level": 5}},
        )

        self.assertEqual(compiled_model, self.model)
        self.assertTrue(hasattr(compiled_model, '_trt_compiler'))

    @pytest.mark.run_only_on('GPU')
    @patch('nemo.export.tensorrt_lazy_compiler.trt_imported', False)
    def test_trt_compile_no_tensorrt(self):
        from nemo.export.tensorrt_lazy_compiler import trt_compile

        # Test when TensorRT is not available
        compiled_model = trt_compile(self.model, self.plan_path)
        self.assertEqual(compiled_model, self.model)
        self.assertFalse(hasattr(compiled_model, '_trt_compiler'))

    @pytest.mark.run_only_on('GPU')
    def test_trt_compiler_initialization(self):
        from nemo.export.tensorrt_lazy_compiler import TrtCompiler

        compiler = TrtCompiler(
            self.model,
            self.plan_path,
            precision="fp16",
            method="onnx",
            input_names=["x"],
            output_names=["output"],
            logger=MagicMock(),
        )

        self.assertEqual(compiler.plan_path, self.plan_path)
        self.assertEqual(compiler.precision, "fp16")
        self.assertEqual(compiler.method, "onnx")
        self.assertEqual(compiler.input_names, ["x"])
        self.assertEqual(compiler.output_names, ["output"])

    @pytest.mark.run_only_on('GPU')
    def test_trt_compiler_invalid_precision(self):
        from nemo.export.tensorrt_lazy_compiler import TrtCompiler

        with self.assertRaises(ValueError):
            TrtCompiler(self.model, self.plan_path, precision="invalid_precision")

    @pytest.mark.run_only_on('GPU')
    def test_trt_compiler_invalid_method(self):
        from nemo.export.tensorrt_lazy_compiler import TrtCompiler

        with self.assertRaises(ValueError):
            TrtCompiler(self.model, self.plan_path, method="invalid_method")

    @pytest.mark.run_only_on('GPU')
    @patch('nemo.export.tensorrt_lazy_compiler.trt_imported', True)
    @patch('nemo.export.tensorrt_lazy_compiler.polygraphy_imported', True)
    @patch('torch.cuda.is_available', return_value=True)
    def test_trt_compile_with_submodule(self, mock_cuda_available):
        from nemo.export.tensorrt_lazy_compiler import trt_compile

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = SimpleModel()

        model = NestedModel()
        compiled_model = trt_compile(model, self.plan_path, submodule=["submodule"])

        self.assertEqual(compiled_model, model)
        self.assertTrue(hasattr(model.submodule, '_trt_compiler'))


if __name__ == '__main__':
    unittest.main()
