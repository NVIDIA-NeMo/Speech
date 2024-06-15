import os
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from multiprocessing import get_start_method
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Optional, Union

import torch
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO

from nemo.utils import logging
from nemo.utils.s3_utils import (
    DEFAULT_CHUNK_SIZE_MB,
    DEFAULT_MAX_READ_CONCURRENCY,
    DEFAULT_MAX_WRITE_CONCURRENCY,
    SHARED_MEM_DIR,
    S3Utils,
)


class S3CheckpointIO(CheckpointIO):
    """A custom S3CheckpointIO module that supports checkpoint reading/writing with s3 when filepath
    is a s3 url.
    """

    def __init__(
        self,
        dirpath: str,
        chunk_size_MB=DEFAULT_CHUNK_SIZE_MB,
        max_read_concurrency=DEFAULT_MAX_READ_CONCURRENCY,
        max_write_concurrency=DEFAULT_MAX_WRITE_CONCURRENCY,
        async_checkpointing=False,
    ):
        """
        Initialize the transfer configuration with custom values.

        This method overrides the default TransferConfig values in boto3.
        See https://boto3.amazonaws.com/v1/documentation/api/latest/_modules/boto3/s3/transfer.html#TransferConfig

        Args:
            chunk_size_MB (int, optional): The size of chunks to use when transferring files.
                Default is 64 (MB).
            max_read_concurrency (int, optional): The maximum number of threads that will be making
                requests to perform a download. Default is 15.
            max_write_concurrency (int, optional): The maximum number of threads that will be making
                requests to perform an upload. Default is 10.
            async_checkpointing (bool, optional): Uses a ProcessPoolExecutor to do the main saving logic.
                This feature should be used with save_top_k as it's possible a previous checkpoint is removed while
                the current checkpoint write fails.
        """
        if not S3Utils.is_s3_url(dirpath):
            raise AssertionError(
                f"Error attempting to initialize an S3CheckpointIO when {dirpath} is not an S3 url. Please use TorchCheckpointIO when using a non-S3 dirpath."
            )

        self.chunk_size_MB = chunk_size_MB
        self.max_read_concurrency = max_read_concurrency
        self.max_write_concurrency = max_write_concurrency
        self._async_checkpointing = async_checkpointing
        '''
        When using shared memory, we create a temporary file to hold the checkpoint before uploading to S3. 
        This list will track those temporary files, and clean up any leaked files that are still around during teardown. 
        '''
        self._temp_files = []

        if self.async_checkpointing:
            # create an executor that will asynchronously run functions
            self._executor = ProcessPoolExecutor(max_workers=1) if self.async_checkpointing else None

            # Eager creating a subprocess now so that forked subprocess does not inherit cuda context from parent
            if get_start_method() == 'fork' and torch.cuda.is_initialized() is True:
                raise Exception(
                    f'torch.cuda should not be initialized when checkpointing subprocess is created by fork method'
                )
            logging.info(f'Creating asynchronous checkpointing subprocess')
            future = self._executor.submit(dummy_func)
            try:
                future.result()
                logging.info(f'Asynchronous heckpointing subprocess created successfully')
            except Exception as e:
                logging.error(f'Failed to create asynchronous checkpointing subprocess, exception: {e}')
                raise e
            self._futures = []

        super().__init__()

    @property
    def async_checkpointing(self):
        return self._async_checkpointing

    def _serialize_checkpoint_to_shm(self, checkpoint: Dict, path: str) -> str:
        """
        Returns:
            filename of the temporary file in shared memory.
        """
        start_time = time.perf_counter()
        tempfile = NamedTemporaryFile(dir=SHARED_MEM_DIR, delete=False)
        torch.save(checkpoint, tempfile)
        logging.info(
            f'Time elapsed saving checkpoint dict to {tempfile.name} for {path}: {(time.perf_counter() - start_time):.2f} seconds, rank {torch.distributed.get_rank()}'
        )
        del checkpoint
        return tempfile.name

    def _serialize_checkpoint_to_bytes(self, checkpoint: Dict, path: str) -> BytesIO:
        """
        Returns:
            The bytestring of the checkpoint.
        """
        ss = time.perf_counter()
        bytes = BytesIO()
        torch.save(checkpoint, bytes)
        tt = time.perf_counter() - ss
        logging.info(
            f'Time elapsed saving checkpoint dict to bytes for {path}: {tt:.2f} seconds, rank {torch.distributed.get_rank()}'
        )
        del checkpoint
        return bytes

    def _check_uploading_results_so_far(self):
        """
        self._future is a list of tuples of form (future, destination path, source path)
        This function checks the result of all the futures, and updates the self._futures list appropriately.
        It also updates the list of self._temp_files, which is used to clean up leaked temporary files in SHARED_MEM during teardown.
        """
        if not self._futures:
            return
        start_time = time.perf_counter()
        done_futures = []
        in_progress_futures = []
        for item in self._futures:
            if item[0].done():
                done_futures.append(item)
            else:
                in_progress_futures.append(item)

        for item in done_futures:
            try:
                item[0].result()
            except Exception as e:
                logging.error(f'Failed to upload {item[2]} to {item[1]}, exception: {e}')
                raise e
            # If the future is complete, we can remove the temp file since we choose to clear the temp file when uploading.
            try:
                self._temp_files.remove(item[2])
            except:
                pass  # When not using shared memory, we do not append anything to the temp_files list, so remove will do nothing.
        self._futures = in_progress_futures
        logging.debug(
            f'Time elapsed checking uploading future results: {(time.perf_counter() - start_time):.2f} seconds'
        )

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        # if we have a shared memory directory, we can serialize as a file to shared memory instead of as bytes.
        if os.path.exists(SHARED_MEM_DIR):
            localfile = self._serialize_checkpoint_to_shm(checkpoint, path)
            self._temp_files.append(localfile)
            saved_as_file = True
        else:
            bytes = self._serialize_checkpoint_to_bytes(checkpoint, path)
            saved_as_file = False

        if self.async_checkpointing:
            self._check_uploading_results_so_far()
            logging.info(f'Uploading checkpoint to {path} in asynchronous mode, rank {torch.distributed.get_rank()}')
            if saved_as_file:
                future = self._executor.submit(
                    _upload_file_to_s3, localfile, path, self.chunk_size_MB, self.max_write_concurrency, True
                )
                self._futures.append((future, path, localfile))
            else:
                future = self._executor.submit(
                    _upload_bytes_to_s3, bytes, path, self.chunk_size_MB, self.max_write_concurrency
                )
                self._futures.append((future, path, 'bytes'))
        else:
            logging.info(f'Uploading checkpoint to {path} in synchronous mode, rank {torch.distributed.get_rank()}')
            if saved_as_file:
                _upload_file_to_s3(localfile, path, self.chunk_size_MB, self.max_write_concurrency, True)
                self._temp_files.remove(localfile)
            else:
                _upload_bytes_to_s3(bytes, path, self.chunk_size_MB, self.max_write_concurrency)

    def load_checkpoint(
        self, path: Union[str, Path], map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        if os.path.exists(SHARED_MEM_DIR):
            with NamedTemporaryFile(dir=SHARED_MEM_DIR, delete=True) as tempfile:
                logging.info(
                    f'Loading checkpoint {path} into a temp file in shared memory {tempfile.name}, rank {torch.distributed.get_rank()}'
                )
                S3Utils.download_s3_file_to_path(
                    s3_path=path,
                    file_path=tempfile.name,
                    chunk_size_MB=self.chunk_size_MB,
                    max_concurrency=self.max_read_concurrency,
                )
                checkpoint = torch.load(tempfile.name)
        else:
            file_stream: BytesIO = S3Utils.download_s3_file_to_stream(
                s3_path=path, chunk_size_MB=self.chunk_size_MB, max_concurrency=self.max_read_concurrency
            )
            checkpoint = torch.load(file_stream)
        return checkpoint

    def remove_checkpoint(self, path: Union[str, Path]) -> None:
        if S3Utils.is_s3_url(path):
            S3Utils.remove_object(path)
        else:
            super().remove_checkpoint(path)

    def teardown(self) -> None:
        # this ensure we wait for final checkpoint to finish uploading at train end.
        rank = torch.distributed.get_rank()
        if self.async_checkpointing:
            logging.info(f'Entering teardown, waiting for all jobs to finish, rank {rank}')
            start_time = time.perf_counter()
            self._executor.shutdown(wait=True)
            logging.info(f'executor shut down after {(time.perf_counter() - start_time):.2f} seconds, rank {rank}')

        '''
        this will be non-empty at the end of training if using asynchronous uploading since the futures are not processed with _check_uploading_results_so_far.
        therefore, we check that the path exists first before trying to delete. 
        '''
        if self._temp_files:
            for tfile in self._temp_files:
                if os.path.exists(tfile):
                    try:
                        os.remove(tfile)
                    except Exception as e:
                        logging.info(f"Error occurred while deleting file {tfile}: {e}")


def _clean_up_conflicting_checkpoint(filepath: str) -> None:
    '''
    before saving to s3, clean up any existing object with the same prefix megatron_gpt+step_count
    e.g. before we save "megatron_gpt--step=1400-validation_loss=6.32-consumed_samples=55920.0-last.ckpt"
    we need to clean up "megatron_gpt--step=1400-validation_loss=xxx-consumed_samples=yyy-last.ckpt"
    so that in case later we need to resume from step 1400, it has a single checkpoint file at step 1400
    '''

    if S3Utils.is_s3_url(filepath):
        prefix_with_step = S3Utils.parse_prefix_with_step(filepath)
        logging.info(f'Looking for conflicting checkpoint under prefix {prefix_with_step}')

        conflict_last_ckpts = S3Utils.find_files_with_suffix(
            base_path=prefix_with_step, suffix='last.ckpt', return_key_only=False
        )
        for last_ckpt in conflict_last_ckpts:
            logging.info(f'Cleaning up conflicting last ckpt {last_ckpt} before saving {filepath}')
            S3Utils.remove_object(last_ckpt)


def _upload_file_to_s3(localfile, path, chunk_size_MB, max_write_concurrency, remove_file):
    try:
        _clean_up_conflicting_checkpoint(path)
        S3Utils.upload_file(localfile, path, chunk_size_MB, max_write_concurrency, remove_file)
    except Exception as e:
        raise e


def _upload_bytes_to_s3(bytes, path, chunk_size_MB, max_write_concurrency):
    try:
        _clean_up_conflicting_checkpoint(path)
        S3Utils.upload_file_stream_to_s3(bytes, path, chunk_size_MB, max_write_concurrency)
    except Exception as e:
        raise e


def dummy_func():
    time.sleep(0.01)
