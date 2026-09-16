# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

TEST_DATA_VERSION = "v1.0.0rc1"
TEST_DATA_FILENAME = "test_data.tar.gz"
TEST_DATA_SHA256 = "bcaf346953ddb7dbd73c67925230886546cfea7120fdab8b199f938ed5960c5d"
TEST_DATA_URL = f"https://github.com/NVIDIA-NeMo/Speech/releases/download/{TEST_DATA_VERSION}/{TEST_DATA_FILENAME}"
TEST_DATA_MARKER = ".test_data_sha256"


def get_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def archive_is_valid(path: Path) -> bool:
    return path.is_file() and get_sha256(path) == TEST_DATA_SHA256


def safe_extract(archive: tarfile.TarFile, output: Path, members: list[tarfile.TarInfo]) -> None:
    for member in members:
        source = archive.extractfile(member) if member.isfile() else None
        destination = output / Path(*PurePosixPath(member.name).parts)
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output_file:
                shutil.copyfileobj(source, output_file)


def safe_members(archive: tarfile.TarFile, output: Path) -> list[tarfile.TarInfo]:
    output = output.resolve()
    members = []
    for member in archive.getmembers():
        member_path = PurePosixPath(member.name)
        destination = (output / Path(*member_path.parts)).resolve()
        if member_path.is_absolute() or ".." in member_path.parts or not destination.is_relative_to(output):
            raise ValueError(f"Unsafe test-data archive path: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Test-data archive links are not allowed: {member.name}")
        if not member.isdir() and not member.isfile():
            raise ValueError(f"Unsupported test-data archive member: {member.name}")
        members.append(member)
    return members


def _extracted_data_is_valid(directory: Path) -> bool:
    archive_path = directory / TEST_DATA_FILENAME
    marker_path = directory / TEST_DATA_MARKER
    if not archive_is_valid(archive_path) or not marker_path.is_file():
        return False
    if marker_path.read_text(encoding="utf-8").strip() != TEST_DATA_SHA256:
        return False

    expected_files = {Path(TEST_DATA_FILENAME), Path(TEST_DATA_MARKER)}
    expected_directories = {Path(".")}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in safe_members(archive, directory):
            relative_path = Path(*PurePosixPath(member.name).parts)
            expected_directories.update(relative_path.parents)
            destination = directory / relative_path
            if member.isdir():
                expected_directories.add(relative_path)
                if not destination.is_dir():
                    return False
            elif member.isfile():
                expected_files.add(relative_path)
                if not destination.is_file() or destination.stat().st_size != member.size:
                    return False
                source = archive.extractfile(member)
                if source is None:
                    return False
                with source, destination.open("rb") as extracted_file:
                    while True:
                        expected = source.read(1024 * 1024)
                        actual = extracted_file.read(1024 * 1024)
                        if expected != actual:
                            return False
                        if not expected:
                            break

    for root, directories, files in os.walk(directory, followlinks=False):
        relative_root = Path(root).relative_to(directory)
        if relative_root not in expected_directories:
            return False
        for name in directories:
            path = Path(root, name)
            if path.is_symlink() or path.relative_to(directory) not in expected_directories:
                return False
        for name in files:
            path = Path(root, name)
            if path.is_symlink() or path.relative_to(directory) not in expected_files:
                return False
    return True


def extracted_data_is_valid(directory: Path) -> bool:
    try:
        return _extracted_data_is_valid(directory)
    except (OSError, tarfile.TarError, ValueError):
        return False


def download_archive(destination: Path, url: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Speech-CI-test-data-preparer"})
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
            return
        except (OSError, urllib.error.URLError):
            destination.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def stage_test_data(output: Path, source: Path | None, url: str) -> Path:
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=staging_parent))
    archive_path = staging / TEST_DATA_FILENAME

    if source:
        try:
            shutil.rmtree(staging)
            shutil.copytree(source, staging, symlinks=True)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
        else:
            if extracted_data_is_valid(staging):
                return staging
            shutil.rmtree(staging, ignore_errors=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=staging_parent))
        archive_path = staging / TEST_DATA_FILENAME

    archive_ready = False
    archive_sources = [output / TEST_DATA_FILENAME]
    if source:
        archive_sources.append(source / TEST_DATA_FILENAME)
    for archive_source in archive_sources:
        try:
            shutil.copy2(archive_source, archive_path)
        except OSError:
            continue
        if archive_is_valid(archive_path):
            archive_ready = True
            break
        archive_path.unlink(missing_ok=True)

    if not archive_ready:
        download_archive(archive_path, url)
    if not archive_is_valid(archive_path):
        raise ValueError(f"{TEST_DATA_FILENAME} does not match the pinned SHA-256")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, staging, safe_members(archive, staging))
    (staging / TEST_DATA_MARKER).write_text(f"{TEST_DATA_SHA256}\n", encoding="utf-8")
    if not extracted_data_is_valid(staging):
        raise ValueError("Prepared test-data directory failed validation")
    return staging


def replace_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(source, destination)


def prepare_test_data(output: Path, source: Path | None, url: str) -> None:
    if not extracted_data_is_valid(output):
        staging = stage_test_data(output, source, url)
        replace_directory(staging, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare immutable Speech test data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--url", default=TEST_DATA_URL)
    args = parser.parse_args()
    prepare_test_data(args.output, args.source, args.url)


if __name__ == "__main__":
    main()
