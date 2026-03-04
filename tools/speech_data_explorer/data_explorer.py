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

import argparse
import base64
import csv
import datetime
import difflib
import gzip
import io
import json
import logging
import math
import operator
import os
import pickle
import tarfile
from collections import defaultdict
from os.path import expanduser
from pathlib import Path
from urllib.parse import urlparse

import boto3
import dash
import dash_bootstrap_components as dbc
import diff_match_patch
import editdistance
import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
from botocore.config import Config
from botocore.exceptions import ClientError
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# number of items in a table per page
DATA_PAGE_SIZE = 10

# Global S3 client (initialized lazily)
_s3_client = None

import configparser
import os
from pathlib import Path


def parse_s3cfg(config_path='~/.s3cfg', section='default'):
    """
    Parse the .s3cfg file and extract configuration values.

    Args:
        config_path: Path to the s3cfg file (default: ~/.s3cfg)
        section: Section of the config file to parse (default: default)

    Returns:
        dict: Dictionary containing the parsed configuration
    """
    # Expand user path
    config_path = Path(config_path).expanduser()

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Create ConfigParser instance
    config = configparser.ConfigParser()

    # Read the config file
    config.read(config_path)

    # Extract values from [default] section
    if section in config:
        s3_config = {
            'use_https': config.getboolean(section, 'use_https', fallback=True),
            'access_key': config.get(section, 'access_key', fallback=None),
            'secret_key': config.get(section, 'secret_key', fallback=None),
            'bucket_location': config.get(section, 'bucket_location', fallback=None),
            'host_base': config.get(section, 'host_base', fallback=None),
        }
        return s3_config
    else:
        raise ValueError(f"No [{section}] section found in config file")


def get_s3_client(s3cfg):
    """Get or create a boto3 S3 client.
    Args:
        s3cfg: S3 configuration file
    Returns:
        boto3.client: S3 client
    """
    global _s3_client
    path, section = s3cfg.rsplit('[', 1)
    s3_config = parse_s3cfg(path, section.rstrip(']'))
    logging.debug(f"S3 config loaded: {s3_config}")
    if _s3_client is None:
        _s3_client = boto3.client(
            's3',
            endpoint_url="https://" + s3_config['host_base'],
            aws_access_key_id=s3_config['access_key'],
            aws_secret_access_key=s3_config['secret_key'],
            region_name=s3_config['bucket_location'],
            config=Config(connect_timeout=5),
        )
    return _s3_client


def is_s3_path(path):
    """Check if a path is an S3 URL."""
    if path is None:
        return False
    return str(path).startswith('s3://')


def is_sharded_path(path):
    """Check if a path contains a sharded range pattern like _OP_0..255_CL_."""
    if path is None:
        return False
    return '_OP_' in str(path) and '_CL_' in str(path)


def expand_sharded_path(path_pattern):
    """Expand a sharded path pattern into a list of individual paths.

    Supports patterns like:
        s3://ASR/sharded_manifests/manifest__OP_0..255_CL_.json
    which expands to:
        s3://ASR/sharded_manifests/manifest_0.json, manifest_1.json, ..., manifest_255.json

    Zero-padding: the replacement uses the same digit width as in the pattern (e.g.
        cuts._OP_000000..000098_CL_.jsonl.gz → cuts.000000.jsonl.gz, ..., cuts.000098.jsonl.gz),
    so Lhotse-style shard names (000000, 000001, ...) are correct.

    Args:
        path_pattern: Path string containing _OP_start..end_CL_ pattern

    Returns:
        list: List of expanded paths, or single-element list if no pattern found
    """
    import re

    pattern = r'_OP_(\d+)\.\.(\d+)_CL_'
    match = re.search(pattern, str(path_pattern))

    if not match:
        return [str(path_pattern)]

    start_str = match.group(1)
    end_str = match.group(2)
    start = int(start_str)
    end = int(end_str)
    # Use the max width of start/end for zero-padding (e.g. 000000..000098 → width 6)
    width = max(len(start_str), len(end_str))

    paths = []
    for i in range(start, end + 1):
        replacement = str(i).zfill(width)
        expanded_path = re.sub(pattern, replacement, str(path_pattern))
        paths.append(expanded_path)

    return paths


def get_shard_index_from_path(path_pattern, concrete_path):
    """Get the shard index from a concrete path given the pattern.

    Args:
        path_pattern: Pattern like s3://bucket/manifest__OP_0..255_CL_.json
        concrete_path: Concrete path like s3://bucket/manifest_5.json

    Returns:
        int: The shard index, or None if pattern doesn't match
    """
    import re

    pattern = r'_OP_(\d+)\.\.(\d+)_CL_'
    match = re.search(pattern, str(path_pattern))

    if not match:
        return None

    # Create a regex pattern to extract the index from the concrete path
    # Replace _OP_start..end_CL_ with a capture group for digits
    extraction_pattern = re.sub(pattern, r'(\\d+)', re.escape(str(path_pattern)))
    extraction_match = re.search(extraction_pattern, str(concrete_path))

    if extraction_match:
        return int(extraction_match.group(1))
    return None


def get_tar_path_for_shard(tar_pattern, shard_index):
    """Get the tar file path for a specific shard index.

    Args:
        tar_pattern: Tar pattern like s3://bucket/audio__OP_0..255_CL_.tar
        shard_index: The shard index to substitute

    Returns:
        str: The concrete tar path for this shard
    """
    import re

    if tar_pattern is None:
        return None

    pattern = r'_OP_\d+\.\.\d+_CL_'
    return re.sub(pattern, str(shard_index), str(tar_pattern))


def parse_s3_path(s3_path):
    """Parse an S3 URL into bucket and key components.

    Args:
        s3_path: S3 URL in format s3://bucket/key

    Returns:
        tuple: (bucket, key)
    """
    parsed = urlparse(str(s3_path))
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def read_s3_file(s3_path):
    """Read a file from S3 and return its contents as a string.

    Args:
        s3_path: S3 URL to the file
    Returns:
        str: File contents
    """
    global _s3_client
    try:
        bucket, key = parse_s3_path(s3_path)
        response = _s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    except ClientError as e:
        logging.error(f"Error reading S3 file {s3_path}: {e}")
        raise


def read_s3_file_bytes(s3_path):
    """Read a file from S3 and return its contents as bytes.

    Args:
        s3_path: S3 URL to the file

    Returns:
        bytes: File contents
    """
    global _s3_client
    try:
        bucket, key = parse_s3_path(s3_path)
        response = _s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except ClientError as e:
        logging.error(f"Error reading S3 file {s3_path}: {e}")
        raise


# Cache for tar file indexes (filename -> {offset, size}) to avoid repeated scans
_tar_index_cache = {}

# Cache for DALI index files
_dali_index_cache = {}


def parse_dali_index(index_content):
    """Parse a DALI index file content into a lookup dictionary.

    DALI index format:
        v1.2 64                           # header line
        <type> <offset> <size> <filename> # one line per file

    Args:
        index_content: String content of the DALI index file

    Returns:
        dict: Mapping of filename -> {'offset': int, 'size': int}
    """
    index = {}
    lines = index_content.strip().split('\n')

    for line in lines[1:]:  # Skip header line
        parts = line.split()
        if len(parts) >= 4:
            file_type = parts[0]
            offset = int(parts[1])
            size = int(parts[2])
            filename = parts[3]

            index[filename] = {'offset': offset, 'size': size}
            # Also index by basename for easier lookup
            basename = os.path.basename(filename)
            if basename and basename != filename:
                index[basename] = {'offset': offset, 'size': size}

    return index


def get_dali_index_path(tar_s3_path, dali_index_base=None):
    """Construct the DALI index file path for a given tar file.

    If dali_index_base is not provided, automatically derives it from tar path:
        s3://bucket/tarred/audio_0.tar -> s3://bucket/tarred/dali_index/audio_0.index

    Args:
        tar_s3_path: S3 path to the tar file (e.g., s3://bucket/tarred/audio_0.tar)
        dali_index_base: Optional base S3 path for DALI index files.
                         If None, auto-derives as tar_directory/dali_index/

    Returns:
        str: S3 path to the corresponding index file
    """
    # Extract the tar filename without extension
    tar_filename = os.path.basename(tar_s3_path)
    tar_name = tar_filename.rsplit('.', 1)[0]  # Remove .tar extension

    # Auto-derive dali_index_base if not provided
    if dali_index_base is None:
        # Get the directory containing the tar file
        tar_dir = tar_s3_path.rsplit('/', 1)[0]
        dali_index_base = f"{tar_dir}/dali_index"

    # Construct index path
    if dali_index_base.endswith('/'):
        return f"{dali_index_base}{tar_name}.index"
    else:
        return f"{dali_index_base}/{tar_name}.index"


def load_dali_index_from_s3(tar_s3_path, dali_index_base=None):
    """Load and cache a DALI index file from S3.

    If dali_index_base is not provided, automatically tries the standard location:
        s3://bucket/tarred/audio_0.tar -> s3://bucket/tarred/dali_index/audio_0.index

    Args:
        tar_s3_path: S3 path to the tar file
        dali_index_base: Optional base S3 path for DALI index files.
                         If None, auto-derives from tar path.

    Returns:
        dict: The parsed DALI index, or None if not found
    """
    global _dali_index_cache

    if tar_s3_path in _dali_index_cache:
        return _dali_index_cache[tar_s3_path]

    index_path = get_dali_index_path(tar_s3_path, dali_index_base)
    logging.info(f"Loading DALI index: {index_path}")

    try:
        index_content = read_s3_file(index_path)
        index = parse_dali_index(index_content)
        logging.info(f"DALI index loaded: {len(index) // 2} files, {len(index_content)/1024:.1f} KB")
        _dali_index_cache[tar_s3_path] = index
        return index
    except ClientError as e:
        logging.warning(f"DALI index not found at {index_path}: {e}")
        return None


def read_s3_range(s3_path, start_byte, end_byte):
    """Read a specific byte range from an S3 file.

    Args:
        s3_path: S3 URL to the file
        start_byte: Starting byte offset (inclusive)
        end_byte: Ending byte offset (inclusive)

    Returns:
        bytes: The requested byte range
    """
    global _s3_client
    try:
        bucket, key = parse_s3_path(s3_path)
        range_size = end_byte - start_byte + 1
        logging.info(f"S3 Range request: bytes={start_byte}-{end_byte} (size: {range_size} bytes) from {s3_path}")
        response = _s3_client.get_object(Bucket=bucket, Key=key, Range=f'bytes={start_byte}-{end_byte}')
        data = response['Body'].read()
        logging.info(f"S3 Range request completed: received {len(data)} bytes")
        return data
    except ClientError as e:
        logging.error(f"Error reading S3 file range {s3_path} [{start_byte}-{end_byte}]: {e}")
        raise


def build_tar_index_from_s3(tar_s3_path, chunk_size=512 * 1024):
    """Build an index of files in a tar archive on S3 by reading only headers.

    Uses S3 Range requests to read tar headers in large chunks,
    minimizing the number of HTTP requests while avoiding downloading the entire tar.

    Args:
        tar_s3_path: S3 URL to the tar file
        chunk_size: Size of chunks to read at a time (default 512KB)

    Returns:
        dict: Mapping of filename -> {'offset': data_start_byte, 'size': file_size}
    """
    logging.info(f"Building tar index by scanning headers: {tar_s3_path}")
    index = {}
    offset = 0
    total_bytes_downloaded = 0
    total_content_size = 0
    num_requests = 0

    # Buffer for reading tar data
    buffer = b''
    buffer_start_offset = 0

    while True:
        # Calculate position within our buffer
        buffer_offset = offset - buffer_start_offset

        # If we need more data, fetch a new chunk
        if buffer_offset >= len(buffer) or len(buffer) - buffer_offset < 512:
            try:
                # Read a large chunk starting from current offset
                chunk = read_s3_range(tar_s3_path, offset, offset + chunk_size - 1)
                num_requests += 1
                total_bytes_downloaded += len(chunk)
                buffer = chunk
                buffer_start_offset = offset
                buffer_offset = 0
            except ClientError as e:
                if 'InvalidRange' in str(e):
                    break
                raise

        if len(buffer) - buffer_offset < 512:
            break

        # Get the 512-byte header from buffer
        header = buffer[buffer_offset : buffer_offset + 512]

        # Check for end-of-archive marker (two consecutive zero blocks)
        if header[:100] == b'\x00' * 100:
            break

        # Parse tar header fields
        # Name: bytes 0-99, Size: bytes 124-135 (octal string)
        name = header[:100].rstrip(b'\x00').decode('utf-8', errors='replace')
        size_str = header[124:136].rstrip(b'\x00 ')
        if not size_str:
            break
        size = int(size_str, 8)
        total_content_size += size

        # Store the data offset (right after the header) and size
        if name:  # Skip empty names
            index[name] = {'offset': offset + 512, 'size': size}
            # Also index by basename for easier lookup
            basename = os.path.basename(name)
            if basename and basename != name:
                index[basename] = {'offset': offset + 512, 'size': size}

        # Move to next header: current header (512) + data (size rounded up to 512)
        offset += 512 + ((size + 511) // 512) * 512

    logging.info(
        f"Tar index built: {len(index) // 2} files, {num_requests} requests, {total_bytes_downloaded/1024:.1f} KB downloaded"
    )
    return index


def get_tar_index(tar_s3_path, dali_index_base=None):
    """Get or build the tar index for a given tar file.

    Automatically tries to load DALI index from standard location first (fast):
        s3://bucket/tarred/audio_0.tar -> s3://bucket/tarred/dali_index/audio_0.index

    Falls back to scanning tar headers if DALI index is not available.

    Args:
        tar_s3_path: S3 URL to the tar file
        dali_index_base: Optional base S3 path for DALI index files.
                         If None, auto-derives from tar path.

    Returns:
        dict: The tar index mapping filenames to offsets and sizes
    """
    global _tar_index_cache

    if tar_s3_path in _tar_index_cache:
        return _tar_index_cache[tar_s3_path]

    # Always try DALI index first (fast - single small file download)
    # It will auto-derive the path if dali_index_base is None
    dali_index = load_dali_index_from_s3(tar_s3_path, dali_index_base)
    if dali_index:
        _tar_index_cache[tar_s3_path] = dali_index
        return dali_index

    logging.warning("DALI index not found, falling back to tar header scanning")
    # Fall back to scanning tar headers (slow for large tars)
    _tar_index_cache[tar_s3_path] = build_tar_index_from_s3(tar_s3_path)
    return _tar_index_cache[tar_s3_path]


def get_audio_from_s3_tar(tar_s3_path, audio_filename, dali_index_base=None):
    """Extract an audio file from a tar archive stored on S3 using Range requests.

    This function only downloads the specific audio file bytes, not the entire tar.
    If dali_index_base is provided, uses DALI index for instant offset lookup.
    Otherwise falls back to scanning tar headers.

    Args:
        tar_s3_path: S3 URL to the tar file (e.g., s3://bucket/audio_0.tar)
        audio_filename: Name of the audio file within the tar (e.g., audio1.wav)
        dali_index_base: Optional base S3 path for DALI index files

    Returns:
        bytes: Audio file contents
    """
    # Get or build the tar index
    index = get_tar_index(tar_s3_path, dali_index_base)

    # Try to find the file in the index
    target_key = None

    # Try exact match first
    if audio_filename in index:
        target_key = audio_filename
    else:
        # Try basename match
        basename = os.path.basename(audio_filename)
        if basename in index:
            target_key = basename
        else:
            # Try to find by suffix
            for name in index:
                if name.endswith(audio_filename) or name.endswith('/' + audio_filename):
                    target_key = name
                    break

    if target_key is None:
        available = list(index.keys())[:10]
        raise FileNotFoundError(
            f"Audio file '{audio_filename}' not found in tar archive {tar_s3_path}. "
            f"Available files: {available}..."
        )

    file_info = index[target_key]
    offset = file_info['offset']
    size = file_info['size']

    # Fetch ONLY the audio file bytes using Range request
    logging.info(f"Fetching audio from tar: {target_key} (offset={offset}, size={size/1024:.1f} KB)")
    audio_bytes = read_s3_range(tar_s3_path, offset, offset + size - 1)
    logging.debug(f"Audio fetched: {len(audio_bytes)} bytes")
    return audio_bytes


def load_audio_from_s3(audio_filepath, tar_base_path=None, shard_index=None, dali_index_base=None):
    """Load audio data from S3, supporting both direct files and tarred audio.

    Args:
        audio_filepath: The audio file path (e.g., "audio1.wav" for tarred, or full S3 URL)
        tar_base_path: Base S3 path for tar files (e.g., "s3://ASR/tarred/audio_0.tar")
                       Can also be a sharded pattern like "s3://ASR/tarred/audio__OP_0..255_CL_.tar"
                       If provided, audio_filepath is treated as a file within this tar.
        shard_index: For sharded tar patterns, the index to use for this audio file
        dali_index_base: Optional base S3 path for DALI index files (for fast offset lookup)

    Returns:
        tuple: (audio_bytes, io.BytesIO) - BytesIO object for librosa to read
    """
    if tar_base_path and is_s3_path(tar_base_path):
        actual_tar_path = tar_base_path
        if is_sharded_path(tar_base_path) and shard_index is not None:
            actual_tar_path = get_tar_path_for_shard(tar_base_path, shard_index)
        logging.debug(f"Loading audio from S3 tar: {actual_tar_path}")

        # Audio is inside a tar file on S3
        audio_bytes = get_audio_from_s3_tar(actual_tar_path, audio_filepath, dali_index_base)
        return io.BytesIO(audio_bytes)
    elif is_s3_path(audio_filepath):
        # Direct S3 audio file
        audio_bytes = read_s3_file_bytes(audio_filepath)
        return io.BytesIO(audio_bytes)
    else:
        raise ValueError(f"Cannot load audio: {audio_filepath} (tar_base_path={tar_base_path})")


def open_manifest_file(manifest_path):
    """Open a manifest file, supporting both local and S3 paths.
    Supports plain text and gzip-compressed (.gz) manifests, including .jsonl.gz.

    Args:
        manifest_path: Path to the manifest file (local or S3 URL)

    Yields:
        str: Lines from the manifest file
    """
    is_gz = str(manifest_path).endswith('.gz')
    if is_s3_path(manifest_path):
        print("manifest_path", manifest_path)
        raw = read_s3_file_bytes(manifest_path)
        if is_gz:
            content = gzip.decompress(raw).decode('utf-8')
        else:
            content = raw.decode('utf-8')
        for line in content.splitlines():
            yield line
    else:
        if is_gz:
            with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    yield line.rstrip('\n')
        else:
            with open(manifest_path, 'r', encoding='utf8') as f:
                for line in f:
                    yield line.rstrip('\n')


# operators for filtering items
filter_operators = {
    '>=': 'ge',
    '<=': 'le',
    '<': 'lt',
    '>': 'gt',
    '!=': 'ne',
    '=': 'eq',
    'contains ': 'contains',
}
comparison_mode = False


# parse table filter queries
def split_filter_part(filter_part):
    for op in filter_operators:
        if op in filter_part:
            name_part, value_part = filter_part.split(op, 1)
            name = name_part[name_part.find('{') + 1 : name_part.rfind('}')]
            value_part = value_part.strip()
            v0 = value_part[0]
            if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                value = value_part[1:-1].replace('\\' + v0, v0)
            else:
                try:
                    value = float(value_part)
                except ValueError:
                    value = value_part
            return name, filter_operators[op], value
    return [None] * 3


# standard command-line arguments parser
def parse_args():
    parser = argparse.ArgumentParser(description='Speech Data Explorer')
    parser.add_argument(
        'manifest',
        help='Path to JSON manifest file. Supports S3 paths (s3://bucket/path) and '
        'sharded patterns using _OP_start..end_CL_ syntax '
        '(e.g., s3://ASR/manifests/manifest__OP_0..255_CL_.json)',
    )
    parser.add_argument('--vocab', help='optional vocabulary to highlight OOV words')
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument(
        '--disable-caching-metrics', action='store_true', help='disable caching metrics for errors analysis'
    )
    parser.add_argument(
        '--estimate-audio-metrics',
        '-a',
        action='store_true',
        help='estimate frequency bandwidth and signal level of audio recordings',
    )
    parser.add_argument(
        '--audio-base-path',
        default=None,
        type=str,
        help='A base path for the relative paths in manifest. It defaults to manifest path.',
    )
    parser.add_argument(
        '--tar-base-path',
        default=None,
        type=str,
        help='S3 path to tarred audio files (e.g., s3://ASR/tarred/audio_0.tar). '
        'Supports sharded patterns using _OP_start..end_CL_ syntax '
        '(e.g., s3://ASR/tarred/audio__OP_0..255_CL_.tar). '
        'When using sharded manifests, the tar shard index is automatically matched '
        'to the manifest shard index. '
        'When specified, audio_filepath values in the manifest are treated as '
        'filenames within the corresponding tar archive.',
    )
    parser.add_argument(
        '--dali-index-base',
        default=None,
        type=str,
        help='S3 path to DALI index files directory (e.g., s3://bucket/tarred/dali_index/). '
        'When provided, uses DALI index files for instant file offset lookup instead of '
        'scanning tar headers. This dramatically speeds up audio loading for large tar files. '
        'If not specified, automatically looks for index at <tar_dir>/dali_index/<tar_name>.index. '
        'Index files should be named audio_0.index, audio_1.index, etc. matching the tar files.',
    )
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')

    parser.add_argument(
        '--names_compared',
        '-nc',
        nargs=2,
        type=str,
        help='Names of the two fields that will be compared, example: pred_text_contextnet pred_text_conformer. "pred_text_" prefix IS IMPORTANT!',
    )
    parser.add_argument(
        '--show_statistics',
        '-shst',
        type=str,
        help='Field name for which you want to see statistics (optional). Example: pred_text_contextnet.',
    )
    parser.add_argument(
        '--s3cfg',
        '-s3c',
        type=str,
        default='',
        help='Path to the s3 credentials file and section. Example: ~/.s3cfg[default]. Set to "" to disable S3 support. Default is "". ',
    )
    args = parser.parse_args()

    # assume audio_filepath is relative to the directory where the manifest is stored
    # For S3 paths, we cannot use os.path.dirname, so leave it as None
    if args.audio_base_path is None:
        if is_s3_path(args.manifest):
            # For S3 manifests, audio_base_path should be explicitly provided
            # or tar_base_path should be used
            args.audio_base_path = None
        else:
            args.audio_base_path = os.path.dirname(args.manifest)

    # automaticly going in comparison mode, if there is names_compared argument
    if args.names_compared is not None:
        comparison_mode = True
        logging.info("comparison mod set to true")
    else:
        comparison_mode = False

    logging.debug(f"Parsed args: {args}, comparison_mode: {comparison_mode}")
    return args, comparison_mode


# estimate frequency bandwidth of signal
def eval_bandwidth(signal, sr, threshold=-50):
    time_stride = 0.01
    hop_length = int(sr * time_stride)
    n_fft = 512
    spectrogram = np.mean(
        np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window='blackmanharris')) ** 2, axis=1
    )
    power_spectrum = librosa.power_to_db(S=spectrogram, ref=np.max, top_db=100)
    freqband = 0
    for idx in range(len(power_spectrum) - 1, -1, -1):
        if power_spectrum[idx] > threshold:
            freqband = idx / n_fft * sr
            break
    return freqband


# load data from JSON manifest file
def load_data(
    data_filename,
    disable_caching=False,
    estimate_audio=False,
    vocab=None,
    audio_base_path=None,
    comparison_mode=False,
    names=None,
    tar_base_path=None,
    dali_index_base=None,
):
    if comparison_mode:
        if names is None:
            logging.error(f'Please, specify names of compared models')
        name_1, name_2 = names

    if not comparison_mode:
        if vocab is not None:
            # load external vocab
            vocabulary_ext = {}
            with open(vocab, 'r') as f:
                for line in f:
                    if '\t' in line:
                        # parse word from TSV file
                        word = line.split('\t')[0]
                    else:
                        # assume each line contains just a single word
                        word = line.strip()
                    vocabulary_ext[word] = 1

        # Disable caching for S3 manifests (cannot get mtime)
        if not disable_caching and not is_s3_path(data_filename):
            pickle_filename = data_filename.split('.json')[0]
            json_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(data_filename))
            timestamp = json_mtime.strftime('%Y%m%d_%H%M')
            pickle_filename += '_' + timestamp + '.pkl'
            if os.path.exists(pickle_filename):
                with open(pickle_filename, 'rb') as f:
                    data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available = pickle.load(f)
                if vocab is not None:
                    for item in vocabulary_data:
                        item['OOV'] = item['word'] not in vocabulary_ext
                if estimate_audio:
                    for item in data:
                        shard_index = item.get('_shard_index')
                        signal, sr = load_audio_data(
                            item['audio_filepath'], audio_base_path, tar_base_path, shard_index, dali_index_base
                        )
                        bw = eval_bandwidth(signal, sr)
                        item['freq_bandwidth'] = int(bw)
                        item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(
                        [data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available],
                        f,
                        pickle.HIGHEST_PROTOCOL,
                    )
                return data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available

    data = []
    wer_count = 0
    cer_count = 0
    wmr_count = 0
    wer = 0
    cer = 0
    wmr = 0
    mwa = 0
    num_hours = 0
    match_vocab_1 = defaultdict(lambda: 0)
    match_vocab_2 = defaultdict(lambda: 0)

    def append_data(
        data_filename,
        estimate_audio,
        field_name='pred_text',
    ):
        data = []
        wer_dist = 0.0
        wer_count = 0
        cer_dist = 0.0
        cer_count = 0
        wmr_count = 0
        wer = 0
        cer = 0
        wmr = 0
        mwa = 0
        num_hours = 0
        vocabulary = defaultdict(lambda: 0)
        alphabet = set()
        match_vocab = defaultdict(lambda: 0)

        sm = difflib.SequenceMatcher()
        metrics_available = False

        # Expand sharded manifest paths if pattern is present
        manifest_paths = expand_sharded_path(data_filename)
        is_sharded = len(manifest_paths) > 1

        logging.info(f"Loading {len(manifest_paths)} manifest file(s)")

        for manifest_idx, manifest_path in enumerate(manifest_paths):
            # Get shard index from the manifest path
            if is_sharded:
                shard_index = get_shard_index_from_path(data_filename, manifest_path)
                if shard_index is None:
                    shard_index = manifest_idx
            else:
                shard_index = None

            # Support both local files and S3 paths
            manifest_lines = list(open_manifest_file(manifest_path))
            desc = f"Shard {shard_index}" if shard_index is not None else manifest_path
            for line in tqdm.tqdm(manifest_lines, desc=desc):
                item = json.loads(line)
                # Lhotse shards: cuts manifest with "supervisions" and cut "id", "duration"
                is_lhotse_cut = (
                    isinstance(item.get('supervisions'), list)
                    and 'id' in item
                    and 'duration' in item
                )
                if is_lhotse_cut:
                    # Build entry from cut: use cut id as audio lookup, duration from cut
                    cut_duration = item['duration']
                    cut_id = item['id']
                    supervisions = item['supervisions']
                    sup = supervisions[0] if supervisions else {}
                    text = sup.get('text') or item.get('text') or ''
                    if not isinstance(text, str):
                        text = ''
                    num_chars = len(text)
                    orig = text.split()
                    num_words = len(orig)
                    for word in orig:
                        vocabulary[word] += 1
                    for char in text:
                        alphabet.add(char)
                    num_hours += cut_duration

                    entry = {
                        'audio_filepath': cut_id,
                        'id': cut_id,
                        'duration': round(cut_duration, 2),
                        'num_words': num_words,
                        'num_chars': num_chars,
                        'word_rate': round(num_words / cut_duration, 2) if cut_duration else 0,
                        'char_rate': round(num_chars / cut_duration, 2) if cut_duration else 0,
                        'text': text,
                    }
                    if item.get('start') is not None:
                        entry['offset'] = item['start']
                    # Add all supervision fields except 'alignment'
                    for k, v in sup.items():
                        if k == 'alignment':
                            continue
                        if k == 'duration':
                            entry['supervision_duration'] = v
                            continue
                        if k not in entry:
                            entry[k] = v
                    item_shard_index = shard_index
                    if item_shard_index is None and 'shard_id' in item:
                        item_shard_index = item['shard_id']
                    if item_shard_index is not None:
                        entry['_shard_index'] = item_shard_index
                    data.append(entry)
                    if field_name in item:
                        metrics_available = True
                        pred = item[field_name].split()
                        measures = jiwer.compute_measures(text, item[field_name])
                        word_dist = measures['substitutions'] + measures['insertions'] + measures['deletions']
                        char_dist = editdistance.eval(text, item[field_name])
                        wer_dist += word_dist
                        cer_dist += char_dist
                        wer_count += num_words
                        cer_count += num_chars
                        sm.set_seqs(orig, pred)
                        for m in sm.get_matching_blocks():
                            for word_idx in range(m[0], m[0] + m[2]):
                                match_vocab[orig[word_idx]] += 1
                        wmr_count += measures['hits']
                        nw, nc = max(num_words, 1e-9), max(num_chars, 1e-9)
                        data[-1][field_name] = item[field_name]
                        data[-1]['WER'] = round(word_dist / nw * 100.0, 2)
                        data[-1]['CER'] = round(char_dist / nc * 100.0, 2)
                        data[-1]['WMR'] = round(measures['hits'] / nw * 100.0, 2)
                        data[-1]['I'] = measures['insertions']
                        data[-1]['D'] = measures['deletions']
                        data[-1]['D-I'] = measures['deletions'] - measures['insertions']
                    if estimate_audio:
                        signal, sr = load_audio_data(
                            cut_id, audio_base_path, tar_base_path, shard_index, dali_index_base
                        )
                        if 'offset' in data[-1]:
                            offset = data[-1]['offset']
                            dur = data[-1]['duration']
                            signal = signal[
                                int(offset * sr) : int((offset + dur) * sr)
                            ]
                        bw = eval_bandwidth(signal, sr)
                        data[-1]['freq_bandwidth'] = int(bw)
                        data[-1]['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
                    for k in item:
                        if k not in data[-1] and k != 'supervisions':
                            data[-1][k] = item[k]
                    continue

                # NeMo-style manifest: audio_filepath, text, duration
                if not isinstance(item.get('text', ''), str):
                    item['text'] = ''
                item_text = item['text']
                num_chars = len(item_text)
                orig = item_text.split()
                num_words = len(orig)
                for word in orig:
                    vocabulary[word] += 1
                for char in item_text:
                    alphabet.add(char)
                num_hours += item['duration']

                if field_name in item:
                    metrics_available = True
                    pred = item[field_name].split()
                    measures = jiwer.compute_measures(item_text, item[field_name])
                    word_dist = measures['substitutions'] + measures['insertions'] + measures['deletions']
                    char_dist = editdistance.eval(item_text, item[field_name])
                    wer_dist += word_dist
                    cer_dist += char_dist
                    wer_count += num_words
                    cer_count += num_chars

                    sm.set_seqs(orig, pred)
                    for m in sm.get_matching_blocks():
                        for word_idx in range(m[0], m[0] + m[2]):
                            match_vocab[orig[word_idx]] += 1
                    wmr_count += measures['hits']
                else:
                    if comparison_mode:
                        if field_name != 'pred_text':
                            if field_name == name_1:
                                logging.error(f"The .json file has no field with name: {name_1}")
                                exit()
                            if field_name == name_2:
                                logging.error(f"The .json file has no field with name: {name_2}")
                                exit()
                entry = {
                    'audio_filepath': item['audio_filepath'],
                    'duration': round(item['duration'], 2),
                    'num_words': num_words,
                    'num_chars': num_chars,
                    'word_rate': round(num_words / item['duration'], 2),
                    'char_rate': round(num_chars / item['duration'], 2),
                    'text': item_text,
                }
                # Store shard index for sharded datasets (needed for tar file lookup)
                # Priority: 1) shard_index from manifest path, 2) shard_id from manifest entry
                item_shard_index = shard_index
                if item_shard_index is None and 'shard_id' in item:
                    item_shard_index = item['shard_id']
                if item_shard_index is not None:
                    entry['_shard_index'] = item_shard_index
                data.append(entry)
                if metrics_available:
                    data[-1][field_name] = item[field_name]
                    if num_words == 0:
                        num_words = 1e-9
                    if num_chars == 0:
                        num_chars = 1e-9
                    data[-1]['WER'] = round(word_dist / num_words * 100.0, 2)
                    data[-1]['CER'] = round(char_dist / num_chars * 100.0, 2)
                    data[-1]['WMR'] = round(measures['hits'] / num_words * 100.0, 2)
                    data[-1]['I'] = measures['insertions']
                    data[-1]['D'] = measures['deletions']
                    data[-1]['D-I'] = measures['deletions'] - measures['insertions']
                if estimate_audio:
                    signal, sr = load_audio_data(
                        item['audio_filepath'], audio_base_path, tar_base_path, shard_index, dali_index_base
                    )
                    bw = eval_bandwidth(signal, sr)
                    item['freq_bandwidth'] = int(bw)
                    item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
                for k in item:
                    if k not in data[-1]:
                        data[-1][k] = item[k]

        vocabulary_data = [{'word': word, 'count': vocabulary[word]} for word in vocabulary]
        return (
            vocabulary_data,
            metrics_available,
            data,
            wer_dist,
            wer_count,
            cer_dist,
            cer_count,
            wmr_count,
            wer,
            cer,
            wmr,
            mwa,
            num_hours,
            vocabulary,
            alphabet,
            match_vocab,
        )

    (
        vocabulary_data,
        metrics_available,
        data,
        wer_dist,
        wer_count,
        cer_dist,
        cer_count,
        wmr_count,
        wer,
        cer,
        wmr,
        mwa,
        num_hours,
        vocabulary,
        alphabet,
        match_vocab,
    ) = append_data(data_filename, estimate_audio, field_name=fld_nm)
    if comparison_mode:
        (
            vocabulary_data_1,
            metrics_available_1,
            data_1,
            wer_dist_1,
            wer_count_1,
            cer_dist_1,
            cer_count_1,
            wmr_count_1,
            wer_1,
            cer_1,
            wmr_1,
            mwa_1,
            num_hours_1,
            vocabulary_1,
            alphabet_1,
            match_vocab_1,
        ) = append_data(data_filename, estimate_audio, field_name=name_1)
        (
            vocabulary_data_2,
            metrics_available_2,
            data_2,
            wer_dist_2,
            wer_count_2,
            cer_dist_2,
            cer_count_2,
            wmr_count_2,
            wer_2,
            cer_2,
            wmr_2,
            mwa_2,
            num_hours_2,
            vocabulary_2,
            alphabet_2,
            match_vocab_2,
        ) = append_data(data_filename, estimate_audio, field_name=name_2)

    if not comparison_mode:
        if vocab is not None:
            for item in vocabulary_data:
                item['OOV'] = item['word'] not in vocabulary_ext

    if metrics_available or comparison_mode:
        if metrics_available:
            wer = wer_dist / wer_count * 100.0
            cer = cer_dist / cer_count * 100.0
            wmr = wmr_count / wer_count * 100.0
        if comparison_mode:
            if metrics_available_1 and metrics_available_2:
                wer_1 = wer_dist_1 / wer_count_1 * 100.0
                cer_1 = cer_dist_1 / cer_count_1 * 100.0
                wmr_1 = wmr_count_1 / wer_count_1 * 100.0

                wer = wer_dist_2 / wer_count_2 * 100.0
                cer = cer_dist_2 / cer_count_2 * 100.0
                wmr = wmr_count_2 / wer_count_2 * 100.0

                acc_sum_1 = 0
                acc_sum_2 = 0

                for item in vocabulary_data_1:
                    w = item['word']
                    word_accuracy_1 = match_vocab_1[w] / vocabulary_1[w] * 100.0
                    acc_sum_1 += word_accuracy_1
                    item['accuracy_1'] = round(word_accuracy_1, 1)
                mwa_1 = acc_sum_1 / len(vocabulary_data_1)

                for item in vocabulary_data_2:
                    w = item['word']
                    word_accuracy_2 = match_vocab_2[w] / vocabulary_2[w] * 100.0
                    acc_sum_2 += word_accuracy_2
                    item['accuracy_2'] = round(word_accuracy_2, 1)
                mwa_2 = acc_sum_2 / len(vocabulary_data_2)

        acc_sum = 0
        for item in vocabulary_data:
            w = item['word']
            word_accuracy = match_vocab[w] / vocabulary[w] * 100.0
            acc_sum += word_accuracy
            item['accuracy'] = round(word_accuracy, 1)
        mwa = acc_sum / len(vocabulary_data)

    num_hours /= 3600.0

    if not comparison_mode and not is_s3_path(data_filename):
        if not disable_caching:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(
                    [data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available],
                    f,
                    pickle.HIGHEST_PROTOCOL,
                )
    if comparison_mode:
        return (
            data,
            wer,
            cer,
            wmr,
            mwa,
            num_hours,
            vocabulary_data,
            alphabet,
            metrics_available,
            data_1,
            wer_1,
            cer_1,
            wmr_1,
            mwa_1,
            num_hours_1,
            vocabulary_data_1,
            alphabet_1,
            metrics_available_1,
            data_2,
            wer_2,
            cer_2,
            wmr_2,
            mwa_2,
            num_hours_2,
            vocabulary_data_2,
            alphabet_2,
            metrics_available_2,
        )

    return data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available


# plot histogram of specified field in data list
def plot_histogram(data, key, label):
    fig = px.histogram(
        data_frame=[item[key] for item in data],
        nbins=50,
        log_y=True,
        labels={'value': label},
        opacity=0.5,
        color_discrete_sequence=['green'],
        height=200,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
    return fig


def plot_word_accuracy(vocabulary_data):
    labels = ['Unrecognized', 'Sometimes recognized', 'Always recognized']
    counts = [0, 0, 0]
    for word in vocabulary_data:
        if word['accuracy'] == 0:
            counts[0] += 1
        elif word['accuracy'] < 100:
            counts[1] += 1
        else:
            counts[2] += 1
    colors = ['red', 'orange', 'green']

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=colors,
                text=['{:.2%}'.format(count / sum(counts)) for count in counts],
                textposition='auto',
            )
        ]
    )
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0), height=200, yaxis={'title_text': '#words'}
    )

    return fig


def absolute_audio_filepath(audio_filepath, audio_base_path, tar_base_path=None):
    """Return absolute path to an audio file.

    Check if a file exists at audio_filepath.
    If not, assume that the path is relative to audio_base_path.
    For S3 paths or tarred audio, returns the original path.

    Args:
        audio_filepath: Path to audio file (local, S3, or filename within tar)
        audio_base_path: Base path for relative audio files
        tar_base_path: S3 path to tar file containing audio (optional)

    Returns:
        str: The resolved audio filepath
    """
    # If using tarred audio from S3, just return the filename as-is
    # The actual loading will be handled by load_audio_data
    if tar_base_path and is_s3_path(tar_base_path):
        return str(audio_filepath)

    # If audio_filepath is already an S3 path, return as-is
    if is_s3_path(audio_filepath):
        return str(audio_filepath)

    audio_filepath = Path(audio_filepath)

    if not audio_filepath.is_file() and not audio_filepath.is_absolute():
        if audio_base_path:
            audio_filepath = Path(audio_base_path) / audio_filepath
        if audio_filepath.is_file():
            filename = str(audio_filepath)
        else:
            filename = expanduser(audio_filepath)
    else:
        filename = expanduser(audio_filepath)

    return filename


def load_audio_data(audio_filepath, audio_base_path=None, tar_base_path=None, shard_index=None, dali_index_base=None):
    """Load audio data from local file, S3, or tarred S3 archive.

    Args:
        audio_filepath: Path to audio file (local, S3, or filename within tar)
        audio_base_path: Base path for relative audio files
        tar_base_path: S3 path to tar file containing audio (optional).
                       Can be a sharded pattern like s3://bucket/audio__OP_0..255_CL_.tar
        shard_index: For sharded tar patterns, the shard index for this audio file
        dali_index_base: Optional base S3 path for DALI index files (for fast offset lookup)

    Returns:
        tuple: (audio_signal, sample_rate)
    """
    # Case 1: Tarred audio on S3
    if tar_base_path and is_s3_path(tar_base_path):
        audio_buffer = load_audio_from_s3(audio_filepath, tar_base_path, shard_index, dali_index_base)
        return librosa.load(audio_buffer, sr=None)

    # Case 2: Direct S3 audio file
    if is_s3_path(audio_filepath):
        audio_buffer = load_audio_from_s3(audio_filepath)
        return librosa.load(audio_buffer, sr=None)

    # Case 3: Local file
    filepath = absolute_audio_filepath(audio_filepath, audio_base_path, tar_base_path)
    return librosa.load(path=filepath, sr=None)


# parse the CLI arguments
args, comparison_mode = parse_args()

if args.s3cfg:
    _s3_client = get_s3_client(args.s3cfg)

if args.show_statistics is not None:
    fld_nm = args.show_statistics
else:
    fld_nm = 'pred_text'
# parse names of compared models, if any
if comparison_mode:
    name_1, name_2 = args.names_compared
    logging.debug(f"Comparing models: {name_1} vs {name_2}")


logging.info('Loading data')
if not comparison_mode:
    data, wer, cer, wmr, mwa, num_hours, vocabulary, alphabet, metrics_available = load_data(
        args.manifest,
        args.disable_caching_metrics,
        args.estimate_audio_metrics,
        args.vocab,
        args.audio_base_path,
        comparison_mode,
        args.names_compared,
        tar_base_path=args.tar_base_path,
        dali_index_base=args.dali_index_base,
    )
else:
    (
        data,
        wer,
        cer,
        wmr,
        mwa,
        num_hours,
        vocabulary,
        alphabet,
        metrics_available,
        data_1,
        wer_1,
        cer_1,
        wmr_1,
        mwa_1,
        num_hours_1,
        vocabulary_1,
        alphabet_1,
        metrics_available_1,
        data_2,
        wer_2,
        cer_2,
        wmr_2,
        mwa_2,
        num_hours_2,
        vocabulary_2,
        alphabet_2,
        metrics_available_2,
    ) = load_data(
        args.manifest,
        args.disable_caching_metrics,
        args.estimate_audio_metrics,
        args.vocab,
        args.audio_base_path,
        comparison_mode,
        args.names_compared,
        tar_base_path=args.tar_base_path,
        dali_index_base=args.dali_index_base,
    )

logging.info('Starting server')
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=os.path.basename(args.manifest),
)

figures_labels = {
    'duration': ['Duration', 'Duration, sec'],
    'num_words': ['Number of Words', '#words'],
    'num_chars': ['Number of Characters', '#chars'],
    'word_rate': ['Word Rate', '#words/sec'],
    'char_rate': ['Character Rate', '#chars/sec'],
    'WER': ['Word Error Rate', 'WER, %'],
    'CER': ['Character Error Rate', 'CER, %'],
    'WMR': ['Word Match Rate', 'WMR, %'],
    'I': ['# Insertions (I)', '#words'],
    'D': ['# Deletions (D)', '#words'],
    'D-I': ['# Deletions - # Insertions (D-I)', '#words'],
    'freq_bandwidth': ['Frequency Bandwidth', 'Bandwidth, Hz'],
    'level_db': ['Peak Level', 'Level, dB'],
}
figures_hist = {}
for k in data[0]:
    val = data[0][k]
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if k in figures_labels:
            ylabel = figures_labels[k][0]
            xlabel = figures_labels[k][1]
        else:
            title = k.replace('_', ' ')
            title = title[0].upper() + title[1:].lower()
            ylabel = title
            xlabel = title
        figures_hist[k] = [ylabel + ' (per utterance)', plot_histogram(data, k, xlabel)]

if metrics_available:
    figure_word_acc = plot_word_accuracy(vocabulary)

stats_layout = [
    dbc.Row(dbc.Col(html.H5(children='Global Statistics'), class_name='text-secondary'), class_name='mt-3'),
    dbc.Row(
        [
            dbc.Col(html.Div('Number of hours', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Number of utterances', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Vocabulary size', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Alphabet size', className='text-secondary'), width=3),
        ],
        class_name='bg-light mt-2 rounded-top border-top border-start border-end',
    ),
    dbc.Row(
        [
            dbc.Col(
                html.H5(
                    '{:.2f} hours'.format(num_hours),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                class_name='border-end',
            ),
            dbc.Col(
                html.H5(len(data), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}),
                width=3,
                class_name='border-end',
            ),
            dbc.Col(
                html.H5(
                    '{} words'.format(len(vocabulary)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                class_name='border-end',
            ),
            dbc.Col(
                html.H5(
                    '{} chars'.format(len(alphabet)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
            ),
        ],
        class_name='bg-light rounded-bottom border-bottom border-start border-end',
    ),
]
if metrics_available:
    stats_layout += [
        dbc.Row(
            [
                dbc.Col(
                    html.Div('Word Error Rate (WER), %', className='text-secondary'), width=3, class_name='border-end'
                ),
                dbc.Col(
                    html.Div('Character Error Rate (CER), %', className='text-secondary'),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.Div('Word Match Rate (WMR), %', className='text-secondary'),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(html.Div('Mean Word Accuracy, %', className='text-secondary'), width=3),
            ],
            class_name='bg-light mt-2 rounded-top border-top border-start border-end',
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(wer),
                        className='text-center p-1',
                        style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(cer), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(wmr),
                        className='text-center p-1',
                        style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(mwa),
                        className='text-center p-1',
                        style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                ),
            ],
            class_name='bg-light rounded-bottom border-bottom border-start border-end',
        ),
    ]
stats_layout += [
    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
    dbc.Row(
        dbc.Col(
            html.Div('{}'.format(sorted(alphabet))),
        ),
        class_name='mt-2 bg-light font-monospace rounded border',
    ),
]
for k in figures_hist:
    stats_layout += [
        dbc.Row(dbc.Col(html.H5(figures_hist[k][0]), class_name='text-secondary'), class_name='mt-3'),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='duration-graph', figure=figures_hist[k][1]),
            ),
        ),
    ]

if metrics_available:
    stats_layout += [
        dbc.Row(dbc.Col(html.H5('Word accuracy distribution'), class_name='text-secondary'), class_name='mt-3'),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='word-acc-graph', figure=figure_word_acc),
            ),
        ),
    ]

wordstable_columns = [{'name': 'Word', 'id': 'word'}, {'name': 'Count', 'id': 'count'}]
if 'OOV' in vocabulary[0]:
    wordstable_columns.append({'name': 'OOV', 'id': 'OOV'})
if metrics_available:
    wordstable_columns.append({'name': 'Accuracy, %', 'id': 'accuracy'})


stats_layout += [
    dbc.Row(dbc.Col(html.H5('Vocabulary'), class_name='text-secondary'), class_name='mt-3'),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='wordstable',
                columns=wordstable_columns,
                filter_action='custom',
                filter_query='',
                sort_action='custom',
                sort_mode='single',
                page_action='custom',
                page_current=0,
                page_size=DATA_PAGE_SIZE,
                cell_selectable=False,
                page_count=math.ceil(len(vocabulary) / DATA_PAGE_SIZE),
                sort_by=[{'column_id': 'word', 'direction': 'asc'}],
                style_cell={'maxWidth': 0, 'textAlign': 'left'},
                style_header={'color': 'text-primary'},
                css=[
                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                ],
            ),
        ),
        class_name='m-2',
    ),
    dbc.Row(
        dbc.Col(
            [
                html.Button('Download Vocabulary', id='btn_csv'),
                dcc.Download(id='download-vocab-csv'),
            ]
        ),
    ),
]


@app.callback(
    Output('download-vocab-csv', 'data'),
    [Input('btn_csv', 'n_clicks'), State('wordstable', 'sort_by'), State('wordstable', 'filter_query')],
    prevent_initial_call=True,
)
def download_vocabulary(n_clicks, sort_by, filter_query):
    vocabulary_view = vocabulary
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            vocabulary_view = [x for x in vocabulary_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            vocabulary_view = [x for x in vocabulary_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        vocabulary_view = sorted(vocabulary_view, key=lambda x: x[col], reverse=descending)

    with open('sde_vocab.csv', encoding='utf-8', mode='w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerow(vocabulary_view[0].keys())
        for item in vocabulary_view:
            writer.writerow([str(item[k]) for k in item])
    return dcc.send_file("sde_vocab.csv")


@app.callback(
    [Output('wordstable', 'data'), Output('wordstable', 'page_count')],
    [Input('wordstable', 'page_current'), Input('wordstable', 'sort_by'), Input('wordstable', 'filter_query')],
)
def update_wordstable(page_current, sort_by, filter_query):
    vocabulary_view = vocabulary
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            vocabulary_view = [x for x in vocabulary_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            vocabulary_view = [x for x in vocabulary_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        vocabulary_view = sorted(vocabulary_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(vocabulary_view):
        page_current = len(vocabulary_view) // DATA_PAGE_SIZE
    return [
        vocabulary_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(vocabulary_view) / DATA_PAGE_SIZE),
    ]


samples_layout = [
    dbc.Row(dbc.Col(html.H5('Data'), class_name='text-secondary'), class_name='mt-3'),
    html.Hr(),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='datatable',
                columns=[{'name': k.replace('_', ' '), 'id': k, 'hideable': True} for k in data[0]],
                filter_action='custom',
                filter_query='',
                sort_action='custom',
                sort_mode='single',
                sort_by=[],
                row_selectable='single',
                selected_rows=[0],
                page_action='custom',
                page_current=0,
                page_size=DATA_PAGE_SIZE,
                page_count=math.ceil(len(data) / DATA_PAGE_SIZE),
                style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                style_header={
                    'color': 'text-primary',
                    'text_align': 'center',
                    'height': 'auto',
                    'whiteSpace': 'normal',
                },
                css=[
                    {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                    {'selector': '.column-header--hide', 'rule': 'display: none'},
                ],
            ),
        )
    ),
] + [
    dbc.Row(
        [
            dbc.Col(
                html.Div(children=k.replace('_', ' ')),
                width=2,
                class_name='mt-1 bg-light font-monospace text-break small rounded border',
            ),
            dbc.Col(html.Div(id='_' + k), class_name='mt-1 bg-light font-monospace text-break small rounded border'),
        ]
    )
    for k in data[0]
]

if metrics_available:
    samples_layout += [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(children='text diff'),
                    width=2,
                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                ),
                dbc.Col(
                    html.Iframe(
                        id='_diff',
                        sandbox='',
                        srcDoc='',
                        style={'border': 'none', 'width': '100%', 'height': '100%'},
                        className='bg-light font-monospace text-break small',
                    ),
                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                ),
            ]
        )
    ]
samples_layout += [
    dbc.Row(
        dbc.Col(
            html.Audio(id='player', controls=True),
        ),
        class_name='mt-3 ',
    ),
    dbc.Row(dbc.Col(dcc.Graph(id='signal-graph')), class_name='mt-3'),
]


# updating vocabulary to show


wordstable_columns_tool = [{'name': 'Word', 'id': 'word'}, {'name': 'Count', 'id': 'count'}]
wordstable_columns_tool.append({'name': 'Accuracy_1, %', 'id': 'accuracy_1'})
wordstable_columns_tool.append({'name': 'Accuracy_2, %', 'id': 'accuracy_2'})


if comparison_mode:
    model_name_1, model_name_2 = name_1, name_2

    for i in range(len(vocabulary_1)):
        vocabulary_1[i].update(vocabulary_2[i])

    def _wer_(grnd, pred):
        grnd_words = grnd.split()
        pred_words = pred.split()
        edit_distance = editdistance.eval(grnd_words, pred_words)
        wer = edit_distance / len(grnd_words)
        return wer

    def metric(a, b, met=None):
        cer = editdistance.distance(a, b) / len(a)
        wer = _wer_(a, b)
        return round(float(wer) * 100, 2), round(float(cer) * 100, 2)

    def write_metrics(data, Ox, Oy):
        da = pd.DataFrame.from_records(data)
        gt = da['text']
        tt_1 = da[Ox]
        tt_2 = da[Oy]

        wer_tt1_c, cer_tt1_c = [], []
        wer_tt2_c, cer_tt2_c = [], []

        for j in range(len(gt)):
            wer_tt1, cer_tt1 = metric(gt[j], tt_1[j])  # first model
            wer_tt2, cer_tt2 = metric(gt[j], tt_2[j])  # second model
            wer_tt1_c.append(wer_tt1)
            cer_tt1_c.append(cer_tt1)
            wer_tt2_c.append(wer_tt2)
            cer_tt2_c.append(cer_tt2)

        da['wer_' + Ox] = pd.Series(wer_tt1_c, index=da.index)
        da['wer_' + Oy] = pd.Series(wer_tt2_c, index=da.index)
        da['cer_' + Ox] = pd.Series(cer_tt1_c, index=da.index)
        da['cer_' + Oy] = pd.Series(cer_tt2_c, index=da.index)
        return da.to_dict('records')

    data_with_metrics = write_metrics(data, model_name_1, model_name_2)
    if args.show_statistics is not None:
        textdiffstyle = {'border': 'none', 'width': '100%', 'height': '100%'}
    else:
        textdiffstyle = {'border': 'none', 'width': '1%', 'height': '1%', 'display': 'none'}

    def prepare_data(df, name1=model_name_1, name2=model_name_2):
        res = pd.DataFrame()
        tmp = df['word']
        res.insert(0, 'word', tmp)
        res.insert(1, 'count', [float(i) for i in df['count']])
        res.insert(2, 'accuracy_model_' + name1, df['accuracy_1'])
        res.insert(3, 'accuracy_model_' + name2, df['accuracy_2'])
        res.insert(4, 'accuracy_diff ' + '(' + name1 + ' - ' + name2 + ')', df['accuracy_1'] - df['accuracy_2'])
        res.insert(2, 'count^(-1)', 1 / df['count'])
        return res

    for_col_names = pd.DataFrame()
    for_col_names.insert(0, 'word', ['a'])
    for_col_names.insert(1, 'count', [0])
    for_col_names.insert(2, 'accuracy_model_' + model_name_1, [0])
    for_col_names.insert(3, 'accuracy_model_' + model_name_2, [0])
    for_col_names.insert(4, 'accuracy_diff ' + '(' + model_name_1 + ' - ' + model_name_2 + ')', [0])
    for_col_names.insert(5, 'count^(-1)', [0])

    @app.callback(
        Output('voc_graph', 'figure'),
        [
            Input('xaxis-column', 'value'),
            Input('yaxis-column', 'value'),
            Input('color-column', 'value'),
            Input('size-column', 'value'),
            Input("datatable-advanced-filtering", "derived_virtual_data"),
            Input("dot_spacing", 'value'),
            Input("radius", 'value'),
        ],
        prevent_initial_call=False,
    )
    def draw_vocab(Ox, Oy, color, size, data, dot_spacing='no', rad=0.01):
        import math
        import random

        import pandas as pd

        df = pd.DataFrame.from_records(data)

        res = prepare_data(df)
        res_spacing = res.copy(deep=True)

        if dot_spacing == 'yes':
            rad = float(rad)
            if Ox[0] == 'a' or 'c':
                tmp = []
                for i in range(len(res[Ox])):
                    tmp.append(
                        res[Ox][i]
                        + rad
                        * random.randrange(1, 10)
                        * math.cos(random.randrange(1, len(res[Ox])) * 2 * math.pi / len(res[Ox]))
                    )
                res_spacing[Ox] = tmp
            if Ox[0] == 'a' or 'c':
                tmp = []
                for i in range(len(res[Oy])):
                    tmp.append(
                        res[Oy][i]
                        + rad
                        * random.randrange(1, 10)
                        * math.sin(random.randrange(1, len(res[Oy])) * 2 * math.pi / len(res[Oy]))
                    )
                res_spacing[Oy] = tmp

            res = res_spacing

        fig = px.scatter(
            res,
            x=Ox,
            y=Oy,
            color=color,
            size=size,
            hover_data={'word': True, Ox: True, Oy: True, 'count': True},
            width=1300,
            height=1000,
        )
        if (Ox == 'accuracy_model_' + model_name_1 and Oy == 'accuracy_model_' + model_name_2) or (
            Oy == 'accuracy_model_' + model_name_1 and Ox == 'accuracy_model_' + model_name_2
        ):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=100,
                y1=100,
                line=dict(
                    color="MediumPurple",
                    width=1,
                    dash="dot",
                ),
            )

        return fig

    @app.callback(
        Output('filter-query-input', 'style'),
        Output('filter-query-output', 'style'),
        Input('filter-query-read-write', 'value'),
    )
    def query_input_output(val):
        input_style = {'width': '100%'}
        output_style = {}
        input_style.update(display='inline-block')
        output_style.update(display='none')
        return input_style, output_style

    @app.callback(Output('datatable-advanced-filtering', 'filter_query'), Input('filter-query-input', 'value'))
    def write_query(query):
        if query is None:
            return ''
        return query

    @app.callback(Output('filter-query-output', 'children'), Input('datatable-advanced-filtering', 'filter_query'))
    def read_query(query):
        if query is None:
            return "No filter query"
        return dcc.Markdown('`filter_query = "{}"`'.format(query))

    ############
    @app.callback(
        Output('filter-query-input-2', 'style'),
        Output('filter-query-output-2', 'style'),
        Input('filter-query-read-write', 'value'),
    )
    def query_input_output(val):
        input_style = {'width': '100%'}
        output_style = {}
        input_style.update(display='inline-block')
        output_style.update(display='none')
        return input_style, output_style

    @app.callback(Output('datatable-advanced-filtering-2', 'filter_query'), Input('filter-query-input-2', 'value'))
    def write_query(query):
        if query is None:
            return ''
        return query

    @app.callback(Output('filter-query-output-2', 'children'), Input('datatable-advanced-filtering-2', 'filter_query'))
    def read_query(query):
        if query is None:
            return "No filter query"
        return dcc.Markdown('`filter_query = "{}"`'.format(query))

    ############

    def display_query(query):
        if query is None:
            return ''
        return html.Details(
            [
                html.Summary('Derived filter query structure'),
                html.Div(
                    dcc.Markdown(
                        '''```json
    {}
    ```'''.format(
                            json.dumps(query, indent=4)
                        )
                    )
                ),
            ]
        )

    comparison_layout = [
        html.Div(
            [
                dcc.Markdown("model 1:" + ' ' + model_name_1[10:]),
                dcc.Markdown("model 2:" + ' ' + model_name_2[10:]),
                dcc.Dropdown(
                    ['word level', 'utterance level'],
                    'word level',
                    placeholder="choose comparison lvl",
                    id='lvl_choose',
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(for_col_names.columns[::], 'accuracy_model_' + model_name_1, id='xaxis-column'),
                        dcc.Dropdown(for_col_names.columns[::], 'accuracy_model_' + model_name_2, id='yaxis-column'),
                        dcc.Dropdown(
                            for_col_names.select_dtypes(include='number').columns[::],
                            placeholder='Select what will encode color of points',
                            id='color-column',
                        ),
                        dcc.Dropdown(
                            for_col_names.select_dtypes(include='number').columns[::],
                            placeholder='Select what will encode size of points',
                            id='size-column',
                        ),
                        dcc.Dropdown(
                            ['yes', 'no'],
                            placeholder='if you want to enable dot spacing',
                            id='dot_spacing',
                            style={'width': '200%'},
                        ),
                        dcc.Input(id='radius', placeholder='Enter radius of spacing (std is 0.01)'),
                        html.Hr(),
                        dcc.Input(
                            id='filter-query-input',
                            placeholder='Enter filter query',
                        ),
                    ],
                    style={'width': '200%', 'display': 'inline-block', 'float': 'middle'},
                ),
                html.Hr(),
                html.Div(id='filter-query-output'),
                dash_table.DataTable(
                    id='datatable-advanced-filtering',
                    columns=wordstable_columns_tool,
                    data=vocabulary_1,
                    editable=False,
                    page_action='native',
                    page_size=5,
                    filter_action="native",
                ),
                html.Hr(),
                html.Div(id='datatable-query-structure', style={'whitespace': 'pre'}),
                html.Hr(),
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(id='voc_graph'),
                    ),
                ),
                html.Hr(),
            ],
            id='wrd_lvl',
            style={'display': 'block'},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(['WER', 'CER'], 'WER', placeholder="Choose metric", id="choose_metric"),
                        dbc.Row(dbc.Col(html.H5('Data'), class_name='text-secondary'), class_name='mt-3'),
                        html.Hr(),
                        html.Hr(),
                        dcc.Input(
                            id='filter-query-input-2', placeholder='Enter filter query', style={'width': '100%'}
                        ),
                        html.Div(id='filter-query-output-2'),
                        dbc.Row(
                            dbc.Col(
                                [
                                    dash_table.DataTable(
                                        id='datatable-advanced-filtering-2',
                                        columns=[
                                            {'name': k.replace('_', ' '), 'id': k, 'hideable': True}
                                            for k in data_with_metrics[0]
                                        ],
                                        data=data_with_metrics,
                                        editable=False,
                                        page_action='native',
                                        page_size=5,
                                        row_selectable='single',
                                        selected_rows=[0],
                                        page_current=0,
                                        filter_action="native",
                                        style_cell={
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                            'maxWidth': 0,
                                            'textAlign': 'center',
                                        },
                                        style_header={
                                            'color': 'text-primary',
                                            'text_align': 'center',
                                            'height': 'auto',
                                            'whiteSpace': 'normal',
                                        },
                                        css=[
                                            {
                                                'selector': '.dash-spreadsheet-menu',
                                                'rule': 'position:absolute; bottom: 8px',
                                            },
                                            {'selector': '.dash-filter--case', 'rule': 'display: none'},
                                            {'selector': '.column-header--hide', 'rule': 'display: none'},
                                        ],
                                    ),
                                    dbc.Row(
                                        dbc.Col(
                                            html.Audio(id='player-1', controls=True),
                                        ),
                                        class_name='mt-3',
                                    ),
                                ]
                            )
                        ),
                    ]
                    + [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(children=k.replace('_', '-')),
                                    width=2,
                                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                                ),
                                dbc.Col(
                                    html.Div(id='__' + k),
                                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                                ),
                            ]
                        )
                        for k in data_with_metrics[0]
                    ]
                ),
            ],
            id='unt_lvl',
        ),
    ] + [
        html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            dbc.Col(
                                dcc.Graph(id='utt_graph'),
                            ),
                        ),
                        html.Hr(),
                        dcc.Input(id='clicked_aidopath', style={'width': '100%'}),
                        html.Hr(),
                        dcc.Input(id='my-output-1', style={'display': 'none'}),  # we do need this
                    ]
                ),
                html.Div(
                    [
                        dbc.Row(dbc.Col(dcc.Graph(id='signal-graph-1')), class_name='mt-3'),
                    ]
                ),
            ],
            id='down_thing',
            style={'display': 'block'},
        )
    ]

if args.show_statistics is not None:
    comparison_layout += [
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(children='text diff'),
                            width=2,
                            class_name='mt-1 bg-light font-monospace text-break small rounded border',
                        ),
                        dbc.Col(
                            html.Iframe(
                                id='__diff',
                                sandbox='',
                                srcDoc='',
                                style=textdiffstyle,
                                className='bg-light font-monospace text-break small',
                            ),
                            class_name='mt-1 bg-light font-monospace text-break small rounded border',
                        ),
                    ],
                    id="text_diff_div",
                )
            ],
            id='mid_thing',
            style={'display': 'block'},
        ),
    ]

    @app.callback(
        [
            Output(component_id='wrd_lvl', component_property='style'),
            Output(component_id='unt_lvl', component_property='style'),
            Output(component_id='mid_thing', component_property='style'),
            Output(component_id='down_thing', component_property='style'),
            Input(component_id='lvl_choose', component_property='value'),
        ]
    )
    def show_hide_element(visibility_state):
        if visibility_state == 'word level':
            return (
                {'width': '50%', 'display': 'inline-block', 'float': 'middle'},
                {'width': '50%', 'display': 'none', 'float': 'middle'},
                {'display': 'none'},
                {'display': 'none'},
            )
        else:
            return (
                {'width': '100%', 'display': 'none', 'float': 'middle'},
                {'width': '100%', 'display': 'inline-block', 'float': 'middle'},
                {'display': 'block'},
                {'display': 'block'},
            )


if args.show_statistics is None:

    @app.callback(
        [
            Output(component_id='wrd_lvl', component_property='style'),
            Output(component_id='unt_lvl', component_property='style'),
            Output(component_id='down_thing', component_property='style'),
            Input(component_id='lvl_choose', component_property='value'),
        ]
    )
    def show_hide_element(visibility_state):
        if args.show_statistics is not None:
            a = {'border': 'none', 'width': '100%', 'height': '100%', 'display': 'block'}
        else:
            a = {'border': 'none', 'width': '100%', 'height': '100%', 'display': 'none'}
        if visibility_state == 'word level':
            return (
                {'width': '50%', 'display': 'inline-block', 'float': 'middle'},
                {'width': '50%', 'display': 'none', 'float': 'middle'},
                {'display': 'none'},
            )
        else:
            return (
                {'width': '100%', 'display': 'none', 'float': 'middle'},
                {'width': '100%', 'display': 'inline-block', 'float': 'middle'},
                {'display': 'block'},
            )


store = []


@app.callback(
    [Output('datatable-advanced-filtering-2', 'page_current'), Output('my-output-1', 'value')],
    [
        Input('utt_graph', 'clickData'),
    ],
)
def real_select_click(hoverData):
    if hoverData is not None:
        path = str(hoverData['points'][0]['customdata'][-1])
        for t in range(len(data_with_metrics)):
            if data_with_metrics[t]['audio_filepath'] == path:
                ind = t
                s = t  # % 5
                sel = s
                pg = math.ceil(ind // 5)
        return pg, sel
    else:
        return 0, 0


@app.callback(
    [Output('datatable-advanced-filtering-2', 'selected_rows')],
    [Input('my-output-1', 'value')],
)
def real_select_click(num):
    s = num
    return [[s]]


CALCULATED_METRIC = [False, False]


@app.callback(
    [
        Output('utt_graph', 'figure'),
        Output('clicked_aidopath', 'value'),
        Input('choose_metric', 'value'),
        Input('utt_graph', 'clickData'),
        Input('datatable-advanced-filtering-2', 'derived_virtual_data'),
    ],
)
def draw_table_with_metrics(met, hoverData, data_virt):
    Ox = name_1
    Oy = name_2
    if met == "WER":
        cerower = 'wer_'
    else:
        cerower = 'cer_'
    da = pd.DataFrame.from_records(data_virt)

    c = da
    fig = px.scatter(
        c,
        x=cerower + Ox,
        y=cerower + Oy,
        width=1000,
        height=900,
        color='num_words',
        hover_data={
            'text': True,
            Ox: True,
            Oy: True,
            'wer_' + Ox: True,
            'wer_' + Oy: True,
            'cer_' + Ox: True,
            'cer_' + Oy: True,
            'audio_filepath': True,
        },
    )  #'numwords': True,
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        line=dict(
            color="Red",
            width=1,
            dash="dot",
        ),
    )
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=10)
    path = None

    if hoverData is not None:
        path = str(hoverData['points'][0]['customdata'][-1])

    return fig, path


@app.callback(
    [Output('datatable', 'data'), Output('datatable', 'page_count')],
    [Input('datatable', 'page_current'), Input('datatable', 'sort_by'), Input('datatable', 'filter_query')],
)
def update_datatable(page_current, sort_by, filter_query):
    data_view = data
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            data_view = [x for x in data_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            data_view = [x for x in data_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        data_view = sorted(data_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(data_view):
        page_current = len(data_view) // DATA_PAGE_SIZE
    return [
        data_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(data_view) / DATA_PAGE_SIZE),
    ]


if comparison_mode:
    app.layout = html.Div(
        [
            dcc.Location(id='url', refresh=False),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                    dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                    dbc.NavItem(dbc.NavLink('Comparison tool', id='comp_tool', href='/comparison')),
                ],
                brand='Speech Data Explorer',
                sticky='top',
                color='green',
                dark=True,
            ),
            dbc.Container(id='page-content'),
        ]
    )
else:
    app.layout = html.Div(
        [
            dcc.Location(id='url', refresh=False),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                    dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                ],
                brand='Speech Data Explorer',
                sticky='top',
                color='green',
                dark=True,
            ),
            dbc.Container(id='page-content'),
        ]
    )


if comparison_mode:

    @app.callback(
        [
            Output('page-content', 'children'),
            Output('stats_link', 'active'),
            Output('samples_link', 'active'),
            Output('comp_tool', 'active'),
        ],
        [Input('url', 'pathname')],
    )
    def nav_click(url):
        if url == '/samples':
            return [samples_layout, False, True, False]
        elif url == '/comparison':
            return [comparison_layout, False, False, True]
        else:
            return [stats_layout, True, False, False]

else:

    @app.callback(
        [
            Output('page-content', 'children'),
            Output('stats_link', 'active'),
            Output('samples_link', 'active'),
        ],
        [Input('url', 'pathname')],
    )
    def nav_click(url):
        if url == '/samples':
            return [samples_layout, False, True]
        else:
            return [stats_layout, True, False]


@app.callback(
    [Output('_' + k, 'children') for k in data[0]], [Input('datatable', 'selected_rows'), Input('datatable', 'data')]
)
def show_item(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    return [data[idx[0]][k] for k in data[0]]


if comparison_mode:

    @app.callback(
        [Output('__' + k, 'children') for k in data_with_metrics[0]],
        [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data')],
    )
    def show_item(idx, data):
        if len(idx) == 0:
            raise PreventUpdate
        return [data[idx[0]][k] for k in data_with_metrics[0]]


@app.callback(
    Output('_diff', 'srcDoc'),
    [
        Input('datatable', 'selected_rows'),
        Input('datatable', 'data'),
    ],
)
def show_diff(
    idx,
    data,
):
    if len(idx) == 0:
        raise PreventUpdate
    orig_words = data[idx[0]]['text']
    orig_words = '\n'.join(orig_words.split()) + '\n'

    pred_words = data[idx[0]][fld_nm]
    pred_words = '\n'.join(pred_words.split()) + '\n'

    diff = diff_match_patch.diff_match_patch()
    diff.Diff_Timeout = 0
    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    diffs_post = []
    for d in diffs:
        diffs_post.append((d[0], d[1].replace('\n', ' ')))

    diff_html = diff.diff_prettyHtml(diffs_post)

    return diff_html


@app.callback(
    Output('__diff', 'srcDoc'),
    [
        Input('datatable-advanced-filtering-2', 'selected_rows'),
        Input('datatable-advanced-filtering-2', 'data'),
    ],
)
def show_diff(
    idx,
    data,
):
    if len(idx) == 0:
        raise PreventUpdate
    orig_words = data[idx[0]]['text']
    orig_words = '\n'.join(orig_words.split()) + '\n'

    pred_words = data[idx[0]][fld_nm]
    pred_words = '\n'.join(pred_words.split()) + '\n'

    diff = diff_match_patch.diff_match_patch()
    diff.Diff_Timeout = 0
    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    diffs_post = []
    for d in diffs:
        diffs_post.append((d[0], d[1].replace('\n', ' ')))

    diff_html = diff.diff_prettyHtml(diffs_post)

    return diff_html


@app.callback(Output('signal-graph', 'figure'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def plot_signal(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    try:
        shard_index = data[idx[0]].get('_shard_index')
        audio, fs = load_audio_data(
            data[idx[0]]['audio_filepath'], args.audio_base_path, args.tar_base_path, shard_index, args.dali_index_base
        )
        if 'offset' in data[idx[0]]:
            audio = audio[
                int(data[idx[0]]['offset'] * fs) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * fs)
            ]
        time_stride = 0.01
        hop_length = int(fs * time_stride)
        n_fft = 512
        # linear scale spectrogram
        s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.power_to_db(S=np.abs(s) ** 2, ref=np.max, top_db=100)
        figs.add_trace(
            go.Scatter(
                x=np.arange(audio.shape[0]) / fs,
                y=audio,
                line={'color': 'green'},
                name='Waveform',
                hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>',
            ),
            row=1,
            col=1,
        )
        figs.add_trace(
            go.Heatmap(
                z=s_db,
                colorscale=[
                    [0, 'rgb(30,62,62)'],
                    [0.5, 'rgb(30,128,128)'],
                    [1, 'rgb(30,255,30)'],
                ],
                colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
                dx=time_stride,
                dy=fs / n_fft / 1000,
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>',
            ),
            row=2,
            col=1,
        )
        figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
        figs.update_xaxes(title_text='Time, s', row=1, col=1)
        figs.update_yaxes(title_text='Amplitude', row=1, col=1)
        figs.update_xaxes(title_text='Time, s', row=2, col=1)
        figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    except Exception as ex:
        app.logger.error(f'ERROR in plot signal: {ex}')

    return figs


@app.callback(
    Output('signal-graph-1', 'figure'),
    [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data')],
)
def plot_signal(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    try:
        shard_index = data[idx[0]].get('_shard_index')
        audio, fs = load_audio_data(
            data[idx[0]]['audio_filepath'], args.audio_base_path, args.tar_base_path, shard_index, args.dali_index_base
        )
        if 'offset' in data[idx[0]]:
            audio = audio[
                int(data[idx[0]]['offset'] * fs) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * fs)
            ]
        time_stride = 0.01
        hop_length = int(fs * time_stride)
        n_fft = 512
        # linear scale spectrogram
        s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.power_to_db(S=np.abs(s) ** 2, ref=np.max, top_db=100)
        figs.add_trace(
            go.Scatter(
                x=np.arange(audio.shape[0]) / fs,
                y=audio,
                line={'color': 'green'},
                name='Waveform',
                hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>',
            ),
            row=1,
            col=1,
        )
        figs.add_trace(
            go.Heatmap(
                z=s_db,
                colorscale=[
                    [0, 'rgb(30,62,62)'],
                    [0.5, 'rgb(30,128,128)'],
                    [1, 'rgb(30,255,30)'],
                ],
                colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
                dx=time_stride,
                dy=fs / n_fft / 1000,
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>',
            ),
            row=2,
            col=1,
        )
        figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
        figs.update_xaxes(title_text='Time, s', row=1, col=1)
        figs.update_yaxes(title_text='Amplitude', row=1, col=1)
        figs.update_xaxes(title_text='Time, s', row=2, col=1)
        figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    except Exception as ex:
        app.logger.error(f'ERROR in plot signal: {ex}')

    return figs


@app.callback(Output('player', 'src'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def update_player(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    try:
        shard_index = data[idx[0]].get('_shard_index')
        signal, sr = load_audio_data(
            data[idx[0]]['audio_filepath'], args.audio_base_path, args.tar_base_path, shard_index, args.dali_index_base
        )
        if 'offset' in data[idx[0]]:
            signal = signal[
                int(data[idx[0]]['offset'] * sr) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * sr)
            ]
        with io.BytesIO() as buf:
            # convert to PCM .wav
            sf.write(buf, signal, sr, format='WAV')
            buf.seek(0)
            encoded = base64.b64encode(buf.read())
        return 'data:audio/wav;base64,{}'.format(encoded.decode())
    except Exception as ex:
        app.logger.error(f'ERROR in audio player: {ex}')
        return ''


@app.callback(
    Output('player-1', 'src'),
    [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data')],
)
def update_player(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    try:
        shard_index = data[idx[0]].get('_shard_index')
        signal, sr = load_audio_data(
            data[idx[0]]['audio_filepath'], args.audio_base_path, args.tar_base_path, shard_index, args.dali_index_base
        )
        if 'offset' in data[idx[0]]:
            signal = signal[
                int(data[idx[0]]['offset'] * sr) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * sr)
            ]
        with io.BytesIO() as buf:
            # convert to PCM .wav
            sf.write(buf, signal, sr, format='WAV')
            buf.seek(0)
            encoded = base64.b64encode(buf.read())
        return 'data:audio/wav;base64,{}'.format(encoded.decode())
    except Exception as ex:
        app.logger.error(f'ERROR in audio player: {ex}')
        return ''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
