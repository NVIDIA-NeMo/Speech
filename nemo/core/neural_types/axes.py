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

from enum import Enum
from typing import Optional

__all__ = ['AxisKindAbstract', 'AxisKind', 'AxisType']


class AxisKindAbstract(Enum):
    """This is an abstract Enum to represents what does varying axis dimension mean.
    In practice, you will almost always use AxisKind Enum. This Enum should be inherited by
    your OWN Enum if you aren't satisfied with AxisKind. Then your own Enum can be used
    instead of AxisKind."""

    pass


class AxisKind(AxisKindAbstract):
    """This Enum represents what does varying axis dimension mean.
    For example, does this dimension correspond to width, batch, time, etc.
    The "Dimension" and "Channel" kinds are the same and used to represent
    a general axis. "Any" axis will accept any axis kind fed to it.
    """

    Batch = 0
    Time = 1
    Dimension = 2
    Channel = 2
    Width = 3
    Height = 4
    Any = 5
    Sequence = 6
    FlowGroup = 7
    Singleton = 8  # Used to represent a axis that has size 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.name).lower()

    def t_with_string(self, text):
        # it checks if text is "t_<any string>"
        return text.startswith("t_") and text.endswith("_") and text[2:-1] == self.__str__()

    @staticmethod
    def from_str(label):
        """Returns AxisKind instance based on short string representation"""
        _label = label.lower().strip()
        if _label == "b" or _label == "n" or _label == "batch":
            return AxisKind.Batch
        elif _label == "t" or _label == "time" or (len(_label) > 2 and _label.startswith("t_")):
            return AxisKind.Time
        elif _label == "d" or _label == "c" or _label == "channel":
            return AxisKind.Dimension
        elif _label == "w" or _label == "width":
            return AxisKind.Width
        elif _label == "h" or _label == "height":
            return AxisKind.Height
        elif _label == "s" or _label == "singleton":
            return AxisKind.Singleton
        elif _label == "seq" or _label == "sequence":
            return AxisKind.Sequence
        elif _label == "flowgroup":
            return AxisKind.FlowGroup
        elif _label == "any":
            return AxisKind.Any
        else:
            raise ValueError(f"Can't create AxisKind from {label}")


class AxisType(object):
    """This class represents axis semantics and (optionally) it's dimensionality
    Args:
        kind (AxisKindAbstract): what kind of axis it is? For example Batch, Height, etc.
        size (int, optional): specify if the axis should have a fixed size. By default it is set to None and you
        typically do not want to set it for Batch and Time
        is_list (bool, default=False): whether this is a list or a tensor axis
    """

    def __init__(self, kind: AxisKindAbstract, size: Optional[int] = None, is_list=False):
        if size is not None and is_list:
            raise ValueError("The axis can't be list and have a fixed size")
        self.kind = kind
        self.size = size
        self.is_list = is_list

    def __repr__(self):
        if self.size is None:
            representation = str(self.kind)
        else:
            representation = f"{str(self.kind)}:{self.size}"
        if self.is_list:
            representation += "_listdim"
        return representation
