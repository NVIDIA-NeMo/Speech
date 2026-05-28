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


class NeMoBaseException(Exception):
    """NeMo Base Exception. All exceptions created in NeMo should inherit from this class"""


class LightningNotInstalledException(NeMoBaseException):
    def __init__(self, obj):
        message = (
            f" You are trying to use {obj} without installing all of pytorch_lightning, hydra, and "
            f"omegaconf. Please install those packages before trying to access {obj}."
        )
        super().__init__(message)


class CheckInstall:
    def __init__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __call__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __getattr__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)
