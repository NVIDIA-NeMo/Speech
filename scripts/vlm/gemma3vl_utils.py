"""Utils for training recipes."""

import signal
import warnings
from nemo.utils import logging


def filter_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="typing_extensions")
    warnings.filterwarnings("ignore", module="megatron.core.distributed.param_and_grad_buffer")
    warnings.filterwarnings("ignore", message=r".*deprecated.*")


def filter_grad_bucket_logs():
    """Filter the noisy `Number of buckets...` log dumped by megatron."""

    def _filter(record):
        del record
        return False

    for handler in logging._logger.handlers:
        handler.addFilter(_filter)


def ignore_sigprof():
    signal.signal(signal.SIGPROF, signal.SIG_IGN)
