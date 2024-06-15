import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
import botocore
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_delay, wait_exponential

from nemo.utils import logging
from nemo.utils.s3_dirpath_utils import build_s3_url, is_s3_url

try:
    import awscrt
    import s3transfer.crt

    crt_available = True
except ImportError as e:
    crt_available = False

MB = 1024**2
GB = 1024**3

SHARED_MEM_DIR = '/dev/shm'
DEFAULT_CHUNK_SIZE_MB = 64
DEFAULT_MAX_READ_CONCURRENCY = 15
DEFAULT_MAX_WRITE_CONCURRENCY = 10


class S3Utils:
    """
    Utility class for interacting with S3. Handles downloading and uploading to S3, and parsing/formatting S3 urls.
    """

    '''
    Avoid caching boto3 client or resource as a class variable as it gets executed once during class construction.
    When the security token expires, the client or resouece will be no longer valid.
    Create a new resource as needed. To avoid multithreading errors, use different session for each thread.
    '''

    @staticmethod
    def s3_path_exists(s3_path: str, match_directory: bool = False) -> bool:
        """
        :s3_path: the path
        :match_directory: if the content is known to be a directory then set it to `True`. Since s3 isn't a file system, paths are funky and the concept of folders doesn't really exist.
        """
        bucket_name, prefix = S3Utils.parse_s3_url(s3_path)
        if not prefix:
            return False

        s3 = S3Utils._get_s3_resource()
        # bucket = s3.Bucket(bucket_name)
        s3_client = s3.meta.client

        try:
            objs = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1, Prefix=prefix).get('Contents', [])
        except s3_client.exceptions.NoSuchBucket:
            return False

        if prefix == '':  # bucket only
            return True

        return len(objs) > 0 and (match_directory or objs[0]['Key'].startswith(prefix))

    @staticmethod
    def remove_object(s3_path: str) -> None:
        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        s3_client.delete_object(Bucket=bucket, Key=key)

    @staticmethod
    def download_s3_file_to_stream(
        s3_path: str, chunk_size_MB: int = DEFAULT_CHUNK_SIZE_MB, max_concurrency: int = DEFAULT_MAX_READ_CONCURRENCY
    ) -> BytesIO:
        bytes_buffer = BytesIO()

        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        chunk_size = chunk_size_MB * MB
        config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)

        start_time = time.perf_counter()
        _download_fileobj_with_retry(s3_client, bucket, key, bytes_buffer, config)
        logging.info(
            f'Time elapsed downloading {s3_path} to file stream with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

        bytes_buffer.seek(0)
        return bytes_buffer

    @staticmethod
    def download_s3_file_to_path(
        s3_path: str,
        file_path: str,
        chunk_size_MB: int = DEFAULT_CHUNK_SIZE_MB,
        max_concurrency: int = DEFAULT_MAX_READ_CONCURRENCY,
    ) -> None:
        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        chunk_size = chunk_size_MB * MB
        config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)

        logging.info(
            f'Downloading {s3_path} to {file_path} with chunk_size={chunk_size_MB}MB and max_threads={max_concurrency}'
        )
        start_time = time.perf_counter()
        _download_file_with_retry(s3_client, bucket, key, file_path, config)
        logging.info(
            f'Time elapsed downloading {s3_path} to {file_path} with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

    @staticmethod
    def upload_file_stream_to_s3(
        bytes_buffer: BytesIO,
        s3_path: str,
        chunk_size_MB: int = DEFAULT_CHUNK_SIZE_MB,
        max_concurrency: int = DEFAULT_MAX_WRITE_CONCURRENCY,
    ) -> None:
        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        chunk_size = chunk_size_MB * MB
        config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        bytes_buffer.seek(0)

        start_time = time.perf_counter()
        _upload_fileobj_with_retry(s3_client, bytes_buffer, bucket, key, config)
        logging.info(
            f'Time elapsed uploading bytes buffer to {s3_path} with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

    @staticmethod
    def upload_file(
        file_path: str,
        s3_path: str,
        chunk_size_MB=DEFAULT_CHUNK_SIZE_MB,
        max_concurrency=DEFAULT_MAX_WRITE_CONCURRENCY,
        remove_file=False,
    ):
        total_size = os.path.getsize(file_path)
        assert total_size > 0, f"file size is zero, {file_path}"

        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)

        chunk_size = chunk_size_MB * MB
        config = TransferConfig(
            multipart_threshold=chunk_size, multipart_chunksize=chunk_size, max_concurrency=max_concurrency
        )

        start_time = time.perf_counter()
        _upload_file_with_retry(s3_client, file_path, bucket, key, config)
        if remove_file and os.path.exists(file_path):
            os.remove(file_path)
        logging.info(
            f'Time elapsed uploading file {file_path} of size {(total_size/GB):.1f}GB to {s3_path} with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

    @staticmethod
    def find_files_with_suffix(
        base_path: str,
        suffix: str = None,
        return_key_only: bool = True,
        profile: Optional[str] = None,
        creds: botocore.credentials.Credentials = None,
    ) -> List[str]:
        """
        Returns a list of keys that have the specified suffix
        :param base_path: the root of search
        :param suffix: the suffix to match, case sensitive
        :return: list of keys matching the suffix, relative to the base_path
        """
        s3 = S3Utils._get_s3_resource(profile, creds)
        bucket_name, prefix = S3Utils.parse_s3_url(base_path)

        start_time = time.perf_counter()
        bucket = s3.Bucket(bucket_name)
        objects_list = _scan_objects_with_retry(s3_bucket=bucket, s3_prefix=prefix)
        logging.info(
            f'Time elapsed reading all objects under path {base_path}: {(time.perf_counter() - start_time):.2f} seconds'
        )

        if suffix:
            objects_list = list(filter(lambda o: o.key.endswith(suffix), objects_list))

        if return_key_only:
            return [o.key for o in objects_list]
        else:
            return [S3Utils.build_s3_url(o.bucket_name, o.key) for o in objects_list]

    @staticmethod
    def _get_s3_resource(
        profile: str = None,
        creds: botocore.credentials.Credentials = None,
        get_client: bool = False,
        session=None,
        config={},
    ):
        config = botocore.config.Config(max_pool_connections=30, **config)

        if profile is not None and creds is not None:
            raise ValueError('Please provide profile or creds or neither, not both.')

        if profile is not None:
            s3 = boto3.Session(profile_name=profile).resource('s3', config=config)
        elif creds is not None:
            s3 = boto3.Session().resource(
                's3',
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                config=config,
            )
        else:
            s3 = (
                boto3.Session().resource('s3', config=config) if not session else session.resource('s3', config=config)
            )

        if get_client:
            return s3.meta.client
        else:
            return s3

    @staticmethod
    def parse_s3_url(s3_url: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"s3://([^/]+)/(.*)", s3_url, flags=re.UNICODE)

        if match is None:
            return None, None

        return match.groups()[0], match.groups()[1]

    @staticmethod
    def build_s3_url(bucket, key) -> str:
        return build_s3_url(bucket, key)

    @staticmethod
    def is_s3_url(path: Optional[str]) -> bool:
        return is_s3_url(path)

    @staticmethod
    def parse_prefix_with_step(path: str) -> str:
        """
        Use regex to find the pattern up to "-step=900-"
        s3://path/to/checkpoints/tp_rank_00_pp_rank_000/megatron_gpt--step=900-validation_loss=6.47-consumed_samples=35960.0-last.ckpt
        should return s3://path/to/checkpoints/tp_rank_00_pp_rank_000/megatron_gpt--step=900-
        """
        match = re.search(r'(.*step=\d+-)', path)

        if match:
            return match.group(1)

        return path


def _scan_objects_with_retry(s3_bucket, s3_prefix):
    # this returns a collection https://boto3.amazonaws.com/v1/documentation/api/latest/guide/collections.html
    # This collection acts as an iterable that automatically makes additional requests to retrieve more objects from S3 as needed
    objects = s3_bucket.objects.filter(Prefix=s3_prefix)
    return list(objects)


def is_slow_down_error(exception):
    """
    This function checks if the error is due to slowdown or is throttling related.
    If so, returns true to allow tenacity to retry the upload/download to S3.
    """
    class_name = exception.__class__.__name__
    module_name = exception.__class__.__module__
    full_class_name = f"{module_name}.{class_name}"
    logging.error(f'Caught exception of type {full_class_name}: {exception}')

    # 2023-12-07T05:59:25.913721576Z stdout F 2023-12-07 05:59:25,913 [ERROR] - s3_utils.py:354 - Caught exception:
    # AWS_ERROR_S3_INVALID_RESPONSE_STATUS: Invalid response status from request. Body from error request is: b'<?xml version="1.0" encoding="UTF-8"?>\n<Error><Code>RequestTimeout</Code><Message>Your socket connection to the server was not read from or written to within the timeout period. Idle connections will be closed.</Message><RequestId>XPHS9896G3RJE364</RequestId><HostId>ZAiF3HPpUD5IgSr/mfkP2QPs7ttuvY+uTRG9MET/jZZ45MJ6bVbnvSBQLggICvPCROPP/1k85p4=</HostId></Error>'
    message = str(exception)
    if (
        "<Code>SlowDown</Code>" in message
        or "<Code>RequestTimeout</Code>" in message
        or "<Code>InternalError</Code>" in message
    ):
        logging.info("Identified the Retriable Error retrying the job")
        return True

    if crt_available and isinstance(exception, awscrt.exceptions.AwsCrtError):
        logging.error(f'Caught awscrt.exceptions.AwsCrtError: {exception.__repr__()}')
        return True

    if isinstance(exception, ClientError):
        logging.error(f'Caught ClientError, response is: {exception.response}')
        error_code = exception.response['Error']['Code'] if exception.response else None
        return error_code in ['SlowDown', 'RequestTimeout', 'InternalError']
    logging.info("Non Retriable Error - Terminating the job")
    return False


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _download_fileobj_with_retry(
    s3_client, bucket: str, key: str, bytes_buffer: BytesIO, config: TransferConfig = None
):
    s3_client.download_fileobj(bucket, key, bytes_buffer, Config=config)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _download_file_with_retry(s3_client, bucket: str, key: str, file_path: str, config: TransferConfig = None):
    s3_client.download_file(bucket, key, file_path, Config=config)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _upload_fileobj_with_retry(s3_client, bytes_buffer: BytesIO, bucket: str, key: str, config: TransferConfig = None):
    s3_client.upload_fileobj(bytes_buffer, bucket, key, Config=config)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _upload_file_with_retry(s3_client, file_path: str, bucket: str, key: str, config: TransferConfig = None):
    s3_client.upload_file(file_path, bucket, key, Config=config)
