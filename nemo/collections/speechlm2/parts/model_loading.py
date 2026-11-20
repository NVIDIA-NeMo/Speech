# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Dependency-neutral helpers for loading pretrained NeMo models."""

from pathlib import Path


def load_pretrained_nemo(cls, model_path_or_name: str):
    """Load a pretrained NeMo model from a local archive or registered model name."""
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        # Local .nemo restore_from() does not resolve the config target, so resolve
        # the concrete class first, matching from_pretrained() behavior.
        cfg = cls.restore_from(model_path_or_name, return_config=True)
        target = cfg.get("target", None) if hasattr(cfg, "get") else None
        if target is not None:
            from nemo.core.classes.common import _get_allowed_target_class

            resolved_cls = _get_allowed_target_class(target)
            concrete_cls = resolved_cls
            while hasattr(concrete_cls, "__wrapped__"):
                concrete_cls = concrete_cls.__wrapped__
            if not isinstance(concrete_cls, type) or not issubclass(concrete_cls, cls):
                raise TypeError(f"Checkpoint target {target!r} is not a subclass of {cls.__name__}.")
            cls = resolved_cls
        return cls.restore_from(model_path_or_name)
    return cls.from_pretrained(model_path_or_name)


def load_pretrained_nemo_config(cls, model_path_or_name: str):
    """Load a NeMo model config without loading model weights."""
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name, return_config=True)
    return cls.from_pretrained(model_path_or_name, return_config=True)
