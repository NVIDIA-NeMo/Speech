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

import os
import tempfile

try:
    import torchvision

    TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False


class VideoFeaturizer(object):
    def __init__(self):
        pass

    def process(self, video_file, offset, duration):

        # Load Video
        video = self.from_file(video_file, offset=offset, duration=duration)

        return video

    def from_file(self, video_file, offset, duration):

        if not TORCHVISION_AVAILABLE:
            raise Exception("Reading Video requires torchvision")

        # Load from filename
        if isinstance(video_file, str):
            video, audio, infos = torchvision.io.read_video(
                video_file, start_pts=offset, end_pts=offset + duration, pts_unit="sec"
            )

        # Load from bytes
        elif isinstance(video_file, bytes):

            # webdataset.torch_video
            with tempfile.TemporaryDirectory() as dirname:
                fname = os.path.join(dirname, f"file.mp4")
                with open(fname, "wb") as stream:
                    stream.write(video_file)
                    video, audio, infos = torchvision.io.read_video(
                        fname, start_pts=offset, end_pts=offset + duration, pts_unit="sec"
                    )
        else:
            raise Exception("Unknown video data format")

        return video
