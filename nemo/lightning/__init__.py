from typing import Union

from lightning_fabric.plugins.environments import slurm
from pytorch_lightning import plugins as _pl_plugins

# This is here to import it once, which improves the speed of launch when in debug-mode
try:
    import transformer_engine  # noqa
except ImportError:
    pass

from nemo.lightning.base import get_vocab_size, teardown
from nemo.lightning.pytorch.plugins import MegatronDataSampler, MegatronMixedPrecision
from nemo.lightning.pytorch.plugins import data_sampler as _data_sampler
from nemo.lightning.pytorch.strategies import MegatronStrategy
from nemo.lightning.pytorch.trainer import Trainer


# We monkey patch because nvidia uses a naming convention for SLURM jobs
def _is_slurm_interactive_mode():
    job_name = slurm.SLURMEnvironment.job_name()
    return job_name is None or job_name.endswith("bash") or job_name.endswith("interactive")


slurm._is_slurm_interactive_mode = _is_slurm_interactive_mode  # noqa: SLF001


_pl_plugins._PLUGIN_INPUT = Union[_pl_plugins._PLUGIN_INPUT, _data_sampler.DataSampler]  # noqa: SLF001


__all__ = [
    "MegatronStrategy",
    "MegatronDataSampler",
    "MegatronMixedPrecision",
    "Trainer",
    "get_vocab_size",
    "teardown",
]
