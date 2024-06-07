# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import itertools
from typing import Callable, Dict, Iterable, Optional, Union

import torch
from apex.contrib.optimizers.distributed_fused_adam import (
    DistributedFusedAdam,
    _disable_pre_forward_hook,
    _multi_tensor_copy,
)

try:
    import apex.contrib.nccl_allocator as nccl_allocator
except ImportError:
    nccl_allocator = None

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map, optim_state_to_sharding_state
from transformer_engine.pytorch.cpp_extensions import cast_to_fp8

from nemo.utils import logging, str_to_dtype
from nemo.utils.te_utils import is_float8tensor

_distribute_within_nodes_pgs = {}


def create_distribute_within_nodes_pgs():
    """Create process groups for distributing with nodes.

    User can reuse this function to reorder communicators for SHArP.
    """
    global _distribute_within_nodes_pgs
    assert torch.distributed.is_initialized()
    if _distribute_within_nodes_pgs:
        return _distribute_within_nodes_pgs

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    devices = torch.cuda.device_count()
    nodes = world_size // devices

    if nodes * devices != world_size:
        logging.warning("Expected all nodes have the same amout of devices, disable distribute_within_nodes.")
        return {}

    node_id = rank // devices
    device_id = rank % devices

    distributed_pgs = []
    for i in range(nodes):
        ranks = [i * devices + j for j in range(devices)]
        pg = torch.distributed.new_group(ranks=ranks)
        distributed_pgs.append(pg)

    redundant_pgs = []
    for i in range(devices):
        ranks = [i + j * devices for j in range(nodes)]
        pg = torch.distributed.new_group(ranks=ranks)
        redundant_pgs.append(pg)

    # To re-order SHArP communicator right after distributed init,
    # we have to expose redundant_process_group to user.
    # User has too invoke allreduce through redundant_process_group
    # before all other communicators to lock SHArP tree.
    _distribute_within_nodes_pgs = {
        'world_size': world_size,
        'rank': rank,
        'devices': devices,
        'nodes': nodes,
        'node_id': node_id,
        'device_id': device_id,
        'distributed_process_group': distributed_pgs[node_id],
        'redundant_process_group': redundant_pgs[device_id],
    }
    return _distribute_within_nodes_pgs


class MegatronDistributedFusedAdam(DistributedFusedAdam):
    """Adam optimizer with ZeRO algorithm

    Child class of Apex DistributedFusedAdam, with optimizations for
    NeMo-Megatron.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups.
        disable_distributed_parameters (bool, optional): use standard
            data-parallel communication instead of ZeRO.
            (default: False)
        distribute_within_nodes (bool, optional): distribute states
            within the same node, e.g. DGX. This can improve performance
            but requires larger memory than distributing within all
            ranks, especially for pure data parallel models.
            (default: False).
        **kwargs: keyword arguments to pass to Apex
            DistributedFusedAdam.

    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[dict]],
        disable_distributed_parameters: bool = False,
        distribute_within_nodes: bool = False,
        **kwargs,
    ):

        # Initialize process groups
        if 'process_group' not in kwargs and parallel_state.is_initialized():
            kwargs['process_group'] = parallel_state.get_data_parallel_group(with_context_parallel=True)
        if disable_distributed_parameters:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self_groups = [torch.distributed.new_group(ranks=[i]) for i in range(world_size)]
            kwargs['distributed_process_group'] = self_groups[rank]
            kwargs['redundant_process_group'] = kwargs['process_group']
        elif distribute_within_nodes:
            dist_pg_infos = create_distribute_within_nodes_pgs()
            if dist_pg_infos:
                kwargs['distributed_process_group'] = dist_pg_infos['distributed_process_group']
                kwargs['redundant_process_group'] = dist_pg_infos['redundant_process_group']
                global _distribute_within_nodes_pgs
                _distribute_within_nodes_pgs = {}

        # Make sure dtypes are in right type
        for keyword in ('dtype', 'grad_sync_dtype', 'param_sync_dtype'):
            if keyword in kwargs:
                kwargs[keyword] = str_to_dtype(kwargs[keyword])

        # Make sure params are in consistent format (list of param group dicts)
        param_groups = list(params)
        assert param_groups
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        # Construct distributed optimizer
        super().__init__(param_groups, **kwargs)

    def _broadcast_params(self) -> None:
        # Assume params have already been synchronized
        pass

    def _make_post_backward_hook(self, param: torch.nn.Parameter, param_group_id: int, param_id: int) -> Callable:
        def hook(*unused):
            if getattr(param, '_pre_forward_hook_is_enabled', False):
                raise RuntimeError(
                    'A parameter called its post-backward hook '
                    'before its pre-forward hook. '
                    'Please manually interact with the parameter '
                    'before the forward pass (e.g. by calling data_ptr) '
                    'or run DistributedFusedAdam with overlap_param_sync=False.'
                )
            with self._lock:
                need_to_initialize = 'fragments' not in self.state[param]
                if need_to_initialize:
                    self._init_param_state(param, param_group_id, param_id)
                if self.greedy_grad_copy and not getattr(param, '_disable_greedy_grad_copy', False):
                    self._grad_copy(param)
                    if self.overlap_grad_sync and not getattr(param, '_disable_overlap_grad_sync', False):
                        self._try_start_bucket_grad_sync(
                            params=[param],
                            ignore_last_bucket=need_to_initialize,
                        )

        return hook

    def init_params(
        self,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        """Initialize optimizer state for parameters

        Initializes FP8 and non-FP8 params separately.

        """

        # Default cases
        if params is None:
            params = self.parameters()
        elif isinstance(params, torch.Tensor):
            params = [params]

        # Ignore parameters that have already been initialized
        params = [param for param in params if "fragments" not in self.state[param]]
        if not params:
            return

        # Initialize FP8 and non-FP8 tensors separately
        if any(is_float8tensor(param) for param in params):
            super().init_params(
                filter(is_float8tensor, params),
                param_sync_dtype=torch.uint8,
                **kwargs,
            )
        super().init_params(
            params,
            param_sync_dtype=param_sync_dtype,
            **kwargs,
        )

    def init_params_bucket(
        self,
        params: Iterable[torch.nn.Parameter],
        grad_sync_dtype: Optional[torch.dtype] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        """Initialize optimizer state for parameters in one effective bucket"""

        # Ignore parameters that have already been initialized
        if isinstance(params, torch.Tensor):
            params = [params]
        params = [param for param in params if "fragments" not in self.state[param]]
        if not params:
            return

        # Initialize parameters with FP32 grads
        fp32_params = []
        remaining_params = []
        for param in params:
            if getattr(param, '_with_fp32_optimizer', False):
                fp32_params.append(param)
            else:
                remaining_params.append(param)
        params = remaining_params
        start_bucket_id = len(self.state["buckets"])
        super().init_params_bucket(
            fp32_params,
            grad_sync_dtype=torch.float32,
            param_sync_dtype=param_sync_dtype,
            **kwargs,
        )
        end_bucket_id = len(self.state["buckets"])
        fp32_buckets = self.state["buckets"][start_bucket_id:end_bucket_id]

        # Initialize FP8 parameters
        fp8_params = []
        remaining_params = []
        for param in params:
            if is_float8tensor(param):
                fp8_params.append(param)
            else:
                remaining_params.append(param)
        params = remaining_params
        start_bucket_id = len(self.state["buckets"])
        super().init_params_bucket(
            fp8_params,
            grad_sync_dtype=grad_sync_dtype,
            param_sync_dtype=torch.uint8,
            **kwargs,
        )
        end_bucket_id = len(self.state["buckets"])
        fp8_buckets = self.state["buckets"][start_bucket_id:end_bucket_id]

        # Initialize remaining parameters as usual
        normal_buckets = []
        start_bucket_id = len(self.state["buckets"])
        super().init_params_bucket(
            params,
            grad_sync_dtype=grad_sync_dtype,
            param_sync_dtype=param_sync_dtype,
            **kwargs,
        )
        end_bucket_id = len(self.state["buckets"])
        normal_buckets = self.state["buckets"][start_bucket_id:end_bucket_id]

        def add_param_to_bucket(
            param: torch.nn.Parameter,
            bucket: self.StateBucket,
        ) -> None:
            """Add trivial param fragment to bucket"""
            param_fragments = self.state[param]["fragments"]
            param_group_id = param_fragments[0].param_group_id
            param_id = param_fragments[0].param_id
            bucket_id = bucket.fragments[0].bucket_id
            param_size = param.numel()
            bucket_size = bucket.bucket_size
            fragment = self.ParameterFragment(
                param_group_id=param_group_id,
                param_id=param_id,
                bucket_id=bucket_id,
                param_range=(param_size, param_size),
                bucket_range=(bucket_size, bucket_size),
                in_local_shard=False,
                shard_range=None,
                shard_bucket_range=None,
                shard_param_range=None,
            )
            param_fragments.append(fragment)
            bucket.fragments.append(fragment)

        # Make sure all added buckets depend on provided params
        for bucket in fp32_buckets:
            for param in itertools.chain(fp8_params, params):
                add_param_to_bucket(param, bucket)
        for bucket in fp8_buckets:
            for param in itertools.chain(fp32_params, params):
                add_param_to_bucket(param, bucket)
        for bucket in normal_buckets:
            for param in itertools.chain(fp32_params, fp8_params):
                add_param_to_bucket(param, bucket)

    def _init_param_state(
        self,
        param: torch.nn.Parameter,
        param_group_id: int,
        param_id: int,
        param_sync_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        """Initialize optimizer state for a parameter

        Initializing the master weights requires slicing a flattened
        view of the param. FP8 tensors do not handle these operations
        gracefully, so we hack around it by explicitly casting to
        FP32.

        """

        # Initialize non-FP8 params as usual
        if not is_float8tensor(param):
            super()._init_param_state(
                param,
                param_group_id,
                param_id,
                param_sync_dtype=param_sync_dtype,
                **kwargs,
            )

        # Return immediately if already initialized
        if "fragments" in self.state[param]:
            return

        # Initialize with FP32 copy of param
        fp32_param = param.float()
        super()._init_param_state(
            fp32_param,
            param_group_id,
            param_id,
            param_sync_dtype=torch.uint8,
            **kwargs,
        )
        self.state[param].update(self.state[fp32_param])
        del self.state[fp32_param]

    @torch.no_grad()
    def init_param_buffer(self) -> None:
        """Allocate contiguous buffers for param buckets

        For FP8 params, the FP8 data buffer is made a view into a
        contiguous buffer.

        """

        # Make sure all params are initialized
        self.contiguous_param_buffer = True
        self.init_params()

        # Construct param buffers
        buffer_sizes = collections.defaultdict(lambda: 0)
        for bucket in self.state["buckets"]:
            dtypes = bucket.dtypes()
            buffer_sizes[dtypes] = max(bucket.contiguous_buffer_offset + bucket.bucket_size, buffer_sizes[dtypes])
        for dtypes, buffer_size in buffer_sizes.items():
            _, _, param_sync_dtype = dtypes
            if getattr(self, "nccl_ub", False):
                if not nccl_allocator:
                    raise RuntimeError("NCCL allocator importing failed but nccl ub is still requested")
                with nccl_allocator.nccl_mem():
                    self._param_buffers[dtypes] = torch.zeros(
                        [buffer_size], dtype=param_sync_dtype, device=self.device
                    )
            else:
                self._param_buffers[dtypes] = torch.zeros([buffer_size], dtype=param_sync_dtype, device=self.device)
        # Figure out corresponding positions in params and param buffer
        params = list(self.parameters())
        param_flat_views = []
        param_buffer_views = []
        for i, param in enumerate(params):
            fragment = self.state[param]["fragments"][0]
            bucket_id = fragment.bucket_id
            bucket = self.state["buckets"][bucket_id]
            param_size = param.numel()
            bucket_start, _ = fragment.bucket_range
            buffer_offset = bucket.contiguous_buffer_offset
            buffer_start = buffer_offset + bucket_start
            buffer_end = buffer_start + param_size
            param_buffer = self._param_buffers[bucket.dtypes()]
            param_buffer_view = param_buffer[buffer_start:buffer_end].detach()
            if param_buffer_view.device != param.device:
                raise RuntimeError(
                    "Attempted to change a parameter with device={param.device} "
                    f"into a buffer view with device={param_buffer_view.device}"
                )
            if is_float8tensor(param):
                param_flat_views.append(param._data.detach().view(-1))
            else:
                if param_buffer_view.dtype != param.dtype:
                    raise RuntimeError(
                        f"Attempted to change a parameter with dtype={param.dtype} "
                        f"into a buffer view with dtype={param_buffer_view.dtype}"
                    )
                if param.is_contiguous(memory_format=torch.channels_last):
                    param = param.permute(0, 2, 3, 1)
                param_flat_views.append(param.detach().view(-1))
            param_buffer_views.append(param_buffer_view)

        # Copy values into param buffer
        _multi_tensor_copy(
            param_flat_views,
            param_buffer_views,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

        # Make all params a view into the param buffer
        for param, buffer_view in zip(params, param_buffer_views):
            if is_float8tensor(param):
                param._data = buffer_view.view(param.size())
            else:
                # Preserve memory format for param here, i.e. NHWC tensors
                # `param.data.set_()` failed to change storage.
                # `param.set_()` invalidates bprop hook.
                param.data = torch.as_strided(
                    buffer_view,
                    param.size(),
                    param.stride(),
                    storage_offset=buffer_view.storage_offset(),
                )

    def try_grad_sync(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Attempt to launch gradient synchronization"""

        def is_grad_copy_enabled(param: torch.nn.Parameter) -> bool:
            return not getattr(param, '_disable_greedy_grad_copy', False) and not getattr(
                param, '_disable_overlap_grad_sync', False
            )

        params = list(filter(is_grad_copy_enabled, params))
        for p in params:
            self._grad_copy(p)
        self._try_start_bucket_grad_sync(params=params)

    def zero_grad(self, *args, **kwargs) -> None:
        super().zero_grad(*args, **kwargs)

        # Reset main grads
        if self.contiguous_grad_buffer:
            for param in self.parameters():
                with _disable_pre_forward_hook(param):
                    param.main_grad = self.grad_buffer_view(param)

    def grad_norm(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        norm_type: float = 2.0,
        force: bool = False,
    ) -> torch.Tensor:
        assert norm_type == 2

        if parameters is not None:
            # Make sure we can access iterable multiple times
            parameters = list(parameters)

        # Compute grad norm
        if force or self._grad_norm is None:

            # Compute norm of local gradients for distributed optimizer
            grad_norm_sq = self._local_grad_norm(parameters=parameters, norm_type=norm_type)
            if self.redundant_size > 1:
                grad_norm_sq /= self.redundant_size

            # Sum over all procs to get grad norm
            torch.distributed.all_reduce(
                grad_norm_sq,
                op=torch.distributed.ReduceOp.SUM,
            )
            self._grad_norm = grad_norm_sq.sqrt()

        # Use cached grad norm
        return super().grad_norm()

    @torch.no_grad()
    def _param_copy_fragments(self, fragments: Iterable[DistributedFusedAdam.ParameterFragment]) -> None:
        """Update parameter fragments with values from parameter buckets

        For FP8 params, values are copied directly into the FP8 data
        buffer.

        """

        # Figure out corresponding positions in param buckets and params
        buffers_in = []
        buffers_out = []
        fragments = list(fragments)
        for fragment in fragments:

            # Check if fragment needs to be updated
            bucket_id = fragment.bucket_id
            bucket_start, bucket_end = fragment.bucket_range
            param_start, param_end = fragment.param_range
            if param_end <= param_start or bucket_id not in self._params_buckets:
                continue

            # Corresponding positions in bucket and param
            param_bucket = self._params_buckets[bucket_id]
            param = self.parameter(fragment)
            buffer_in = param_bucket.params_bucket[bucket_start:bucket_end]
            if is_float8tensor(param):
                # Copy into FP8 params's data buffer
                assert (
                    param_bucket.params_bucket.dtype == torch.uint8
                ), "Expected FP8 params to perform param sync in UINT8"
                buffer_out = param._data.view(-1)[param_start:param_end]
                buffers_in.append(buffer_in)
                buffers_out.append(buffer_out)
            elif torch.is_floating_point(buffer_in) and torch.is_floating_point(param):
                # Conv with NHWC layout, i.e. shape (N, C, H, W) and stride
                # (HWC, 1, WC, C), can't `.view(-1)`. Here to turn it to
                # tensor with shape (N, H, W, C) and stride (HWC, WC, C, 1).
                # Note: https://github.com/NVIDIA/apex/pull/1794
                if param.is_contiguous(memory_format=torch.channels_last):
                    param = param.permute(0, 2, 3, 1)

                # Cast between floating-point dtypes
                buffer_out = param.detach().view(-1)[param_start:param_end]
                buffers_in.append(buffer_in)
                buffers_out.append(buffer_out)
            else:
                # Copy most significant bytes for non-floating-point
                # dtypes
                # Note: Assume dtypes are little-endian
                buffer_out = param.detach().view(-1)[param_start:param_end]
                in_bytes = buffer_in.unsqueeze(-1).view(torch.uint8)
                out_bytes = buffer_out.unsqueeze(-1).view(torch.uint8)
                copy_size = min(in_bytes.size(-1), out_bytes.size(-1))
                buffers_in.append(in_bytes[..., -copy_size:])
                buffers_out.append(out_bytes[..., -copy_size:])
                if copy_size < out_bytes.size(-1):
                    out_bytes[..., :-copy_size].zero_()

        # Copy data from parameter buckets to parameters
        _multi_tensor_copy(
            buffers_in,
            buffers_out,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

        # Update transpose caches
        params = set(self.parameter(fragment) for fragment in fragments)
        for param in params:
            if is_float8tensor(param):
                param._reset_caches()

    @torch.no_grad()
    def _check_params_shard_dtypes(self, params_buckets: Dict[int, DistributedFusedAdam.ParameterBucket]) -> None:
        """Make sure local shards of parameters are in expected datatypes

        For FP8 params, FP32 values are cast into FP8 using per-param
        scaling factors and per-param amaxes are computed and reduced.

        """

        # Just call base class function if there are no FP8 tensors
        num_fp8_params = sum(1 for param in self.parameters() if is_float8tensor(param))
        if num_fp8_params == 0:
            super()._check_params_shard_dtypes(params_buckets)
            return

        # Cast local data to FP8
        fp8_params_shards = dict()
        for bucket_id, param_bucket in params_buckets.items():
            state_bucket = self.state["buckets"][bucket_id]
            if state_bucket.param_sync_dtype != torch.uint8:
                continue

            # Initialize FP8 buffer for param sync
            params_shard = param_bucket.params_shard
            if self.contiguous_param_buffer:
                shard_size = state_bucket.shard_size
                buffer_offset = state_bucket.contiguous_buffer_offset
                buffer_start = buffer_offset + self.distributed_rank * shard_size
                buffer_end = buffer_start + shard_size
                param_buffer = self._param_buffers[state_bucket.dtypes()]
                fp8_params_shard = param_buffer[buffer_start:buffer_end]
            else:
                fp8_params_shard = torch.empty_like(params_shard, dtype=torch.uint8)
            param_bucket.params_shard = fp8_params_shard

            # Cast param fragments to FP8
            for fragment in self.state["buckets"][bucket_id].fragments:
                param = self.parameter(fragment)
                if not is_float8tensor(param):
                    continue
                if not fragment.in_local_shard:
                    continue
                shard_start, shard_end = fragment.shard_range
                if shard_end <= shard_start:
                    continue
                shard_range = slice(shard_start, shard_end)
                cast_to_fp8(
                    params_shard[shard_range].view(1, -1),
                    param._fp8_meta["scaling_fwd"],
                    param._fp8_meta_index,
                    param._fp8_dtype,
                    out=fp8_params_shard[shard_range].view(1, -1),
                )

        # Update FP8 scaling factors when all buckets have processed
        if getattr(self, "_check_params_shard_dtypes_progress", None) is None:
            self._check_params_shard_dtypes_progress = []
        self._check_params_shard_dtypes_progress.extend(params_buckets.keys())
        if len(self._check_params_shard_dtypes_progress) == len(self.state["buckets"]):
            assert len(set(self._check_params_shard_dtypes_progress)) == len(self.state["buckets"])

            # FP8 scaling factors
            amaxes = []
            scales = []
            scale_invs = []
            i = -1
            for param in self.parameters():
                if not is_float8tensor(param):
                    continue
                i += 1
                fp8_meta = param._fp8_meta["scaling_fwd"]
                fp8_meta_index = param._fp8_meta_index
                amaxes.append(fp8_meta.amax_history[0][fp8_meta_index].view(1))
                scales.append(fp8_meta.scale[fp8_meta_index].view(1))
                scale_invs.append(param._scale_inv.view(1))

            # Update cached scale-inverses
            packed_scales = torch.empty(num_fp8_params, dtype=torch.float32, device=self.device)
            packed_scale_views = [packed_scales[i].view(1) for i in range(num_fp8_params)]
            _multi_tensor_copy(
                scales,
                packed_scale_views,
                dummy_overflow_buf=self._dummy_overflow_buf,
            )
            torch.reciprocal(packed_scales, out=packed_scales)
            _multi_tensor_copy(
                packed_scale_views,
                scale_invs,
                dummy_overflow_buf=self._dummy_overflow_buf,
            )

            # Reduce amaxes
            # Note: Assume each param has a separate amax
            packed_amaxes = torch.empty(num_fp8_params, dtype=torch.float32, device=self.device)
            packed_amax_views = [packed_amaxes[i].view(1) for i in range(num_fp8_params)]
            _multi_tensor_copy(
                amaxes,
                packed_amax_views,
                dummy_overflow_buf=self._dummy_overflow_buf,
            )
            torch.distributed.all_reduce(
                packed_amaxes,
                op=torch.distributed.ReduceOp.MAX,
                group=self.distributed_process_group,
            )
            _multi_tensor_copy(
                packed_amax_views,
                amaxes,
                dummy_overflow_buf=self._dummy_overflow_buf,
            )

            # Reset
            self._check_params_shard_dtypes_progress = None

        # Handle any remaining dtype conversions
        super()._check_params_shard_dtypes(params_buckets)

    def sharded_state_dict(self, model_sharded_state_dict, optimizer_state_dict=None):
        if optimizer_state_dict is None:
            optimizer_state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict,
            optim_params_iter=self.parameters(),
        )
        # Convert state
        step = optimizer_state_dict['state'].pop('step')
        state_dict_format = optimizer_state_dict.pop('format', None)
        optim_state_to_sharding_state(optimizer_state_dict, id_to_sharded_param_map)
        optimizer_state_dict['state']['step'] = step
        if state_dict_format is not None:
            optimizer_state_dict['format'] = state_dict_format

        def rename_fp32_params(x):
            if isinstance(x, ShardedTensor) and x.key.startswith('optimizer.state.param'):
                x.key = x.key.replace('optimizer.state.param', 'optimizer.state.fp32_param')
            return x

        dict_list_map_inplace(rename_fp32_params, optimizer_state_dict)

        return optimizer_state_dict
