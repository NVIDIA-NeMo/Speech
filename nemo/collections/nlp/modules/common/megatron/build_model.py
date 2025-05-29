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


from typing import Any, Callable, Dict, List, Optional

import torch

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelType = ApexGuardDefaults()

    HAVE_MEGATRON_CORE = False

try:
    from apex.transformer.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


# Apex's `build model' refactored to call Megatron-Core classes
def build_model(
    model_provider_func: Callable[[Any, Dict[str, Any]], torch.nn.Module],
    wrap_with_ddp: bool = True,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    model_type: ModelType = ModelType.encoder_or_decoder,
    on_cpu: bool = False,
    *args: Any,
    **kwargs: Any,
) -> List[torch.nn.Module]:
    """Build the model satisfying pipeline model parallel requirements.
    This function sets `pre_process` and `post_process` to `**kwargs` and pass `*args` and `**kwargs` to
    `model_provider_func`.
    Args:
        model_provider_func: A function which takes `*args` and `**kwargs` and returns a `nn.Module`.
        wrap_with_ddp: If :obj:`True`, wrap the instantiated model
            with `torch.nn.parallel.distributed.DistributedDataParallel`, a.k.a. `DDP`.
        virtual_pipeline_model_parallel_size: Specify when using interleaving scheduling pipeline model parallel.
        model_type:
        *args: arguments for model provider func
        **kwargs: Keyword arguments for model provider func
    Returns:
        a list of `nn.Module`(s). If `virtual_pipeline_model_parallel_size` is not None,
        the list has multiple models, otherwise one.
    """
    if model_type is None:
        model_type = ModelType.encoder_or_decoder

    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and virtual_pipeline_model_parallel_size is not None
    ):
        raise ValueError("Virtual pipeline model parallel size is no longer supported for nemo 1.0")
        model = []
        for i in range(virtual_pipeline_model_parallel_size):
            model.append(
                model_provider_func(
                    *args,
                    **kwargs,
                    pre_process=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i),
                    post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i),
                    vp_stage=i,
                )
            )
    else:
        if model_type == ModelType.encoder_or_decoder:
            model = model_provider_func(
                *args,
                **kwargs,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            )
        elif model_type == ModelType.encoder_and_decoder:
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            # `add_encoder` & `add_decoder` logic.
            add_encoder, add_decoder = True, True
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                split_rank = parallel_state.get_pipeline_model_parallel_split_rank()
                if split_rank is None:
                    raise RuntimeError("Split rank needs to be specified for model with both encoder and decoder.")
                rank = parallel_state.get_pipeline_model_parallel_rank()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = rank == (split_rank - 1) or rank == (world_size - 1)
                add_encoder = parallel_state.is_pipeline_stage_before_split()
                add_decoder = parallel_state.is_pipeline_stage_after_split()
            model = model_provider_func(
                *args,
                **kwargs,
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            raise ValueError(f"Unrecognized ModelType '{model_type}'")

    if not isinstance(model, list):
        model = [model]

    for model_module in model:
        model_module.model_type = model_type

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if parallel_state.model_parallel_is_initialized() and parallel_state.get_data_parallel_rank() == 0:
        msg = " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            _calc_number_of_params(model),
        )
        logging.info(msg)

    # GPU allocation.
    if not on_cpu:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    if wrap_with_ddp:
        i = torch.cuda.current_device()
        model = [
            torch.nn.parallel.distributed.DistributedDataParallel(
                model_module,
                device_ids=[i],
                output_device=i,
                process_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            for model_module in model
        ]
    return model


def _calc_number_of_params(model: List[torch.nn.Module]) -> int:
    assert isinstance(model, list)
    return sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
