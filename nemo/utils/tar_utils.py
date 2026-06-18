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

import logging
import os
import tarfile
from pathlib import PurePosixPath
from typing import Iterable, Optional, Union


class TarPathTraversalError(ValueError):
    """Raised when a tar member would extract outside the target directory."""


def is_safe_tar_member(member: tarfile.TarInfo, extract_to: str) -> bool:
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        return False

    if member.issym() or member.islnk():
        return False

    destination = os.path.realpath(os.path.join(extract_to, *member_path.parts))
    return os.path.commonpath([destination, extract_to]) == extract_to


def safe_extract(
    tar: tarfile.TarFile,
    path: str,
    members: Optional[Iterable[Union[tarfile.TarInfo, str]]] = None,
    *,
    skip_unsafe: bool = False,
) -> list[tarfile.TarInfo]:
    extract_to = os.path.realpath(path)
    os.makedirs(extract_to, exist_ok=True)
    if members is None:
        members = tar.getmembers()

    extracted_members = []
    for member in members:
        member = tar.getmember(member) if isinstance(member, str) else member
        if is_safe_tar_member(member, extract_to):
            tar.extract(member, extract_to, filter="data")
            extracted_members.append(member)
            continue

        message = f"Skipping potentially unsafe tar member: {member.name}"
        if skip_unsafe:
            logging.warning(message)
            continue
        raise TarPathTraversalError(message)

    return extracted_members
