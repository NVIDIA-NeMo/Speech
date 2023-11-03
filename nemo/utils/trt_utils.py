# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

try:
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine

    HAVE_TRT = True

except (ImportError, ModuleNotFoundError):

    HAVE_TRT = False


def build_engine(
    onnx_path,
    output_path,
    fp16,
    input_profile=None,
    enable_refit=False,
    enable_preview=False,
    timing_cache=None,
    workspace_size=0,
):
    print(f"Building TensorRT engine for {onnx_path}: {output_path}")
    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    preview_features = None

    config_kwargs = {}
    if workspace_size > 0:
        config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
    engine = engine_from_network(
        network_from_onnx_path(onnx_path),
        config=CreateConfig(
            fp16=fp16,
            refittable=enable_refit,
            profiles=[p],
            preview_features=preview_features,
            load_timing_cache=timing_cache,
            **config_kwargs,
        ),
        save_timing_cache=timing_cache,
    )
    save_engine(engine, path=output_path)
