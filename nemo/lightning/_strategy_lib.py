import itertools
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, Mapping, Optional, Protocol, TypeVar

import torch
from torch import nn

NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE = "NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE"


if TYPE_CHECKING:
    from lightning_fabric.utilities.types import Optimizable
    from megatron.core.model_parallel_config import ModelParallelConfig


class SharedStateDictProtocol(Protocol):
    def sharded_state_dict(self, prefix=""): ...


def init_parallel_ranks(
    world_size: int,
    global_rank: int,
    local_rank: int,
    parallel_config: "ModelParallelConfig",
    seed=1234,
    fp8=False,
) -> None:
    """
    Initializes the parallel ranks for distributed training.

    This function sets up the parallel ranks based on the provided world size, global rank, local rank,
    and parallel configuration. It also sets the seed for random number generation and determines whether
    to use fp8 precision.

    Args:
        world_size (int): The total number of processes participating in the distributed training.
        global_rank (int): The rank of the current process in the distributed training setup.
        local_rank (int): The rank of the current process within its machine.
        parallel_config (ModelParallelConfig): The configuration object containing settings for model parallelism.
        seed (int, optional): The seed for random number generation. Defaults to 1234.
        fp8 (bool, optional): Whether to use fp8 precision for model parameters. Defaults to False.
    """
    from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
    from nemo.utils import AppState

    app_state = AppState()

    if os.environ.get(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, "false").lower() == "true":
        init_world_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size
        init_global_rank = app_state.global_rank
        init_local_rank = app_state.local_rank
    else:
        init_world_size = world_size
        init_global_rank = global_rank
        init_local_rank = local_rank

    initialize_model_parallel_for_nemo(
        world_size=init_world_size,
        global_rank=init_global_rank,
        local_rank=init_local_rank,
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        expert_model_parallel_size=parallel_config.expert_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
        context_parallel_size=parallel_config.context_parallel_size,
        seed=seed,
        pipeline_model_parallel_split_rank=getattr(parallel_config, "pipeline_model_parallel_split_rank", None),
        use_fp8=fp8,
        init_mpi_proc_group=getattr(parallel_config, "tp_comm_overlap", False),
        # apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
    )


def init_model_parallel(model: Optional[nn.Module] = None) -> None:
    """Initializes Megatron-LM model parallel if using model parallelism."""
    import torch.distributed
    from megatron.core import parallel_state

    from nemo.utils import AppState

    app_state = AppState()

    # we initialize megatron-lm model parallel and data parallel groups
    # after initializing DDP with PTL.
    if app_state.model_parallel_size is not None:
        # destroy groups in case they have already been created
        # this happens with multiple calls to trainer.test for example
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=app_state.tensor_model_parallel_size,
                pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=app_state.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
                context_parallel_size=app_state.context_parallel_size,
                expert_model_parallel_size=app_state.expert_model_parallel_size,
            )

            # assert that fake tp and pp rank match after model parallel init
            assert app_state.tensor_model_parallel_rank == parallel_state.get_tensor_model_parallel_rank()
            assert app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()

            app_state.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            app_state.data_parallel_group = parallel_state.get_data_parallel_group()
            app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
            app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
            app_state.pipeline_model_parallel_group = parallel_state.get_pipeline_model_parallel_group()

            # create MPI process group for UCX-based communication APIs
            if app_state.init_mpi_proc_group:
                torch.distributed.new_group(backend="mpi")

        if model:
            # Set TP group
            # Deep iterate but skip self to avoid infinite recursion.
            for index, child in enumerate(model.modules()):
                if index == 0:
                    continue
                if hasattr(child, "set_tensor_parallel_group"):
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    child.set_tensor_parallel_group(tp_group)


def set_model_parallel_attributes(model, parallelism):
    # Right now mcore sub-classes ModelParellelConfig, we should remove that
    # Given Lightning's structure it would be better if parallelism is a different object
    # Since then it can be passed to the Strategy

    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    assert isinstance(
        parallelism, ModelParallelConfig
    ), f"Expected parallelism config to be of type ModelParallelConfig, but got {type(parallelism)}"
    has_mcore_config = isinstance(getattr(model, "config", None), TransformerConfig)
    if has_mcore_config and hasattr(model, "configure_model"):
        config: TransformerConfig = model.config
        for attr_name in filter(lambda x: not x.startswith('__'), dir(parallelism)):
            if not hasattr(config, attr_name):
                continue
            setattr(config, attr_name, getattr(parallelism, attr_name))

        return config

    return None


@contextmanager
def megatron_lazy_init_context(config) -> Generator[None, None, None]:
    def monkey_patched(c):
        return {"device": "meta"}

    from megatron.core.transformer.custom_layers import transformer_engine as _te

    original = _te._get_extra_te_kwargs  # noqa: SLF001
    _te._get_extra_te_kwargs = monkey_patched  # noqa: SLF001

    _orig_perform_initialization = config.perform_initialization
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.perform_initialization = False
    config.use_cpu_initialization = True

    yield

    _te._get_extra_te_kwargs = original  # noqa: SLF001
    config.perform_initialization = _orig_perform_initialization
    config.use_cpu_initialization = _orig_use_cpu_initialization


@contextmanager
def megatron_cpu_init_context(config) -> Generator[None, None, None]:
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.use_cpu_initialization = True

    yield

    config.use_cpu_initialization = _orig_use_cpu_initialization


ModelT = TypeVar("ModelT", bound=nn.Module)


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self,
        init_scale=2.0**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        hysteresis=1,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.optimizer_update_skipped: Optional[bool] = None
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    def _unscale_grads_(self, optimizer, *args):
        if getattr(optimizer, "_custom_amp_unscale_grads", False):
            return optimizer.unscale_grads(*args)
        else:
            return super()._unscale_grads_(optimizer, *args)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        from megatron.core import parallel_state

        retval = None
        found_inf = torch.cuda.FloatTensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=parallel_state.get_model_parallel_group(),
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
            self.optimizer_update_skipped = False
        else:
            self.optimizer_update_skipped = True
        return retval

    def update(self, new_scale=None):
        """
        Updates to native grad scaler update function.
        1. Check inf across model-parallel ranks.
        2. Update hysteresis tracker.
        3. Apply hysteresis to grad scale update.
        """
        from megatron.core import parallel_state

        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = (
                    "new_scale should be a float or a 1-element torch.cuda.FloatTensor with" " requires_grad=False."
                )
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_model_parallel_group(),
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_model_parallel_group(),
                    )
                    found_inf_combined += found_inf

            if found_inf_combined > 0:
                self._hysteresis_tracker -= 1
                if self._hysteresis_tracker <= 0:
                    # When hysteresis becomes zero, follow the native grad scale update rule.
                    # Increase scale and reset growth tracker
                    torch._amp_update_scale_(  # noqa: SLF001
                        _scale,
                        _growth_tracker,
                        found_inf_combined,
                        self._growth_factor,
                        self._backoff_factor,
                        self._growth_interval,
                    )
                else:
                    # Only reset the growth tracker when hysteresis is larger than zero
                    _growth_tracker.fill_(0.0)
            else:
                # When no inf found, follow the native grad scale update rule.
                # Increment growth_tracker, update scale when growth tracker reaches the interval, and
                # reset the hysteresis tracker.
                torch._amp_update_scale_(  # noqa: SLF001
                    _scale,
                    _growth_tracker,
                    found_inf_combined,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
                self._hysteresis_tracker = self.hysteresis

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(
            torch.cuda.amp.grad_scaler._refresh_per_optimizer_state  # noqa: SLF001
        )

    def state_dict(self):
        """
        Add hysteresis_tracker to the native functions' state_dict.
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_hysteresis_tracker": self._hysteresis_tracker,
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Load hysteresis_tracker in addition to the state dict of the native function.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        if "_hysterisis_tracker" in state_dict:
            self._hysteresis_tracker = state_dict["_hysterisis_tracker"]
        else:
            self._hysteresis_tracker = 1


def enable_nvidia_optimizations() -> None:
    """These optimizations are present in NVIDIA NGC PyTorch Containers."""
    # NVIDIA container version check
    nvidia_torch_version = os.getenv("NVIDIA_PYTORCH_VERSION", None)
    if nvidia_torch_version is not None:
        try:
            NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split(".")[0])
        except Exception:
            NVIDIA_TORCH_MAJOR = 0
        try:
            NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split(".")[1])
        except Exception:
            NVIDIA_TORCH_MINOR = 0

        # NVFUSER available starting with 21.11
        if NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11):
            # NVFUSER
            torch._C._jit_set_profiling_executor(True)  # noqa: SLF001
            torch._C._jit_set_profiling_mode(True)  # noqa: SLF001
            torch._C._jit_override_can_fuse_on_cpu(False)  # noqa: SLF001
            torch._C._jit_override_can_fuse_on_gpu(False)  # noqa: SLF001
            torch._C._jit_set_texpr_fuser_enabled(False)  # noqa: SLF001
            # torch._C._jit_set_nvfuser_enabled(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)  # noqa: SLF001
    else:
        # Not a Nvidia container. NVFUSER Dependency check is on users
        pass


def optimizer_sharded_state_dict(
    model: SharedStateDictProtocol,
    optimizer: "Optimizable",
    is_loading=False,
    sharding_type='fully_sharded_model_space',
) -> Dict[str, torch.Tensor]:
    """
    Sharded state dictionary for an MainParamsOptimizerWrapper.
    Used to save and load the optimizer state when training with distributed_checkpoint.

    Returns
    -------
        dict: The sharded state dictionary for the optimizer
    Raises:
        ValueError: If a parameter ID does not match any model sharded parameter.
    """
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )

    from nemo.core.optim import MainParamsOptimizerWrapper
    from nemo.core.optim.optimizers import init_optimizer_states

    model_sharded_state_dict = model.sharded_state_dict()

    # remove _extra_state
    model_sharded_state_dict = {
        key: value for key, value in model_sharded_state_dict.items() if not key.endswith("_extra_state")
    }

    if hasattr(optimizer, "sharded_state_dict"):
        return optimizer.sharded_state_dict(
            model_sharded_state_dict, is_loading=is_loading, sharding_type=sharding_type
        )

    if not isinstance(optimizer, MainParamsOptimizerWrapper):
        # Regular optimizer, e.g. Adam or FusedAdam
        init_optimizer_states(optimizer)
        optimizer_state_dict = optimizer.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict,
            optim_params_iter=itertools.chain.from_iterable(g['params'] for g in optimizer.param_groups),
        )
        optim_state_to_sharding_state(optimizer_state_dict, id_to_sharded_param_map)
        return optimizer_state_dict

    optimizer_state_dict: Dict[str, Any] = optimizer.state_dict()

    id_to_sharded_param_map = get_param_id_to_sharded_param_map(
        model_sharded_state_dict=model_sharded_state_dict,
        optim_params_iter=itertools.chain.from_iterable(g for g in optimizer.float16_groups),
    )

    # Convert fp32_from_fp16_params
    assert len(optimizer_state_dict["fp32_from_fp16_params"]) == len(optimizer_state_dict["optimizer"]["param_groups"])

    def get_safe(param_id):
        try:
            return id_to_sharded_param_map[param_id]
        except KeyError as e:
            raise ValueError(f"Param id {param_id} does not match any model sharded param") from e

    optimizer_state_dict["fp32_from_fp16_params"] = [
        [
            make_sharded_optimizer_tensor(get_safe(param_id), fp32_param, prefix="optimizer.state.fp32_param")
            for param_id, fp32_param in zip(state_group["params"], fp32_group)
        ]
        for fp32_group, state_group in zip(
            optimizer_state_dict["fp32_from_fp16_params"],
            optimizer_state_dict["optimizer"]["param_groups"],
        )
    ]

    # Convert state
    optim_state_to_sharding_state(optimizer_state_dict["optimizer"], id_to_sharded_param_map)

    return optimizer_state_dict


def load_model_state_dict(megatron_parallel, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
    from megatron.core import parallel_state

    for index, module in enumerate(megatron_parallel):
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            if "state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["state_dict"][f"model_{index}"]
            else:
                checkpoint_state_dict = checkpoint[f"model_{index}"]
        else:
            if "state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["state_dict"]
            else:
                checkpoint_state_dict = checkpoint

        n_nesting = 0
        mcore_model = megatron_parallel.module
        while hasattr(mcore_model, "module"):
            mcore_model = mcore_model.module
            n_nesting += 1

        _state_dict = {}
        for key, value in checkpoint_state_dict.items():
            # Count the number of "module." at the start of the key
            count, _key = 0, key
            while _key.startswith("module."):
                _key = _key[len("module.") :]
                count += 1

            # Adjust the number of "module." prefixes
            if count < n_nesting:
                to_add = "module." * (n_nesting - count)
                _state_dict[f"{to_add}{key}"] = value
            elif count > n_nesting:
                to_remove = "module." * (count - n_nesting)
                _state_dict[key[len(to_remove) :]] = value
            else:
                _state_dict[key] = value

        module.load_state_dict(_state_dict, strict=strict)
