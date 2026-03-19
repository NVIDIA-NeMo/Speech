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


class Stream:
    """
    Minimal interface for a stream
    """

    def __init__(self, stream_id: int):
        """
        Args:
            stream_id: (int) The id of the stream.
        """
        self._stream_id = stream_id
        self.frame_count = 0

    def __iter__(self):
        """Returns the iterator object"""
        return self

    def __next__(self):
        """Get the next frame in the stream"""
        raise NotImplementedError("Subclasses must implement __next__ method")

    @property
    def stream_id(self) -> int:
        """Get the stream id"""
        return self._stream_id
