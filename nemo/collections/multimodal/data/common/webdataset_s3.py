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
import io
import os
import sys
from urllib.parse import urlparse

import yaml

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.utils import logging

try:
    import webdataset.gopen as gopen_webdata
    from webdataset import cache, filters, shardlists
    from webdataset.compat import FluidInterface
    from webdataset.handlers import reraise_exception
    from webdataset.pipeline import DataPipeline
    from webdataset.pytorch import IterableDataset
    from webdataset.tariterators import group_by_keys, tar_file_expander

    HAVE_WEBDATASET = True

except (ImportError, AttributeError, ModuleNotFoundError):

    HAVE_WEBDATASET = False

    logging.warning("Webdataset import failed! We recommend use `webdataset==0.2.48`.")

# Number of attempts to read aws objects.
_NUM_OBJECT_STORE_READ_ATTEMPTS = 10

if HAVE_WEBDATASET:

    def gopen(url, mode="rb", bufsize=8192, **kw):
        r"""Open the URL.
        This uses the `gopen_schemes` dispatch table to dispatch based
        on scheme.
        Support for the following schemes is built-in: pipe, file,
        http, https, sftp, ftps, scp.
        When no scheme is given the url is treated as a file.
        You can use the OPEN_VERBOSE argument to get info about
        files being opened.

        This implementation is based on webdataset's gopen,
        with the modification of supporting reading from s3 object_store:
            https://webdataset.github.io/webdataset/api/webdataset/gopen.html#gopen
        Args:
            url (list[str]): the source URL
            mode (str): the mode ("rb", "r")
            bufsize (int): the buffer size
        """
        global fallback_gopen
        verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
        if verbose:
            print("GOPEN", url, gopen_webdata.info, file=sys.stderr)

        assert mode in ["rb", "wb"], mode
        if url == "-":
            if mode == "rb":
                return sys.stdin.buffer
            elif mode == "wb":
                return sys.stdout.buffer
            else:
                raise ValueError(f"unknown mode {mode}")

        # If we specify 'object_store' in keyword arguments,
        # then we would load from AWS.
        # In this case, you also need to specify s3_client and s3_bucket_name
        # in arguments.
        if 'object_store' in kw and kw['object_store']:
            # Load from object store
            attempt = 0

            while attempt < _NUM_OBJECT_STORE_READ_ATTEMPTS:
                try:
                    s3_response_object = kw['s3_client'].get_object(Bucket=kw['s3_bucket_name'], Key=url)
                    object_content = s3_response_object['Body'].read()

                    # This is a check to verify is the object is fully read.
                    full_read = s3_response_object['ContentLength'] == len(object_content)
                    if full_read:
                        return io.BytesIO(object_content)
                    else:
                        attempt += 1
                except Exception as e:  # noqa
                    # If there is an exception (usually connectivity error or protocol error), read again
                    attempt += 1
                    print(e)
                    print('Retrying tar file download, attempt {}'.format(attempt))
                    continue
            raise ConnectionError('Unable to read {} from PBSS. {} attempts tried.'.format(url, attempt))

        # Append root path to the url if dataset is stored on local disk system
        elif 'local_root_path' in kw and kw['local_root_path'] is not None:
            url = os.path.join(kw['local_root_path'], url)

        # For all other gopen schemes, use the native webdataset gopen functions.
        pr = urlparse(url)
        if pr.scheme == "":
            bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
            return open(url, mode, buffering=bufsize)
        if pr.scheme == "file":
            bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
            return open(pr.path, mode, buffering=bufsize)
        handler = gopen_webdata.gopen_schemes["__default__"]
        handler = gopen_webdata.gopen_schemes.get(pr.scheme, handler)
        return handler(url, mode, bufsize, **kw)

    def url_opener(data, handler=reraise_exception, **kw):
        r"""Given a stream of url names (packaged in `dict(url=url)`), yield opened streams.

        Args:
            data: Iterator of dictionaires containing url paths.
            handler: Exception handler.
        """
        for sample in data:
            assert isinstance(sample, dict), sample
            assert "url" in sample
            url = sample["url"]
            try:
                stream = gopen(url, **kw)
                sample.update(stream=stream)
                yield sample
            except Exception as exn:
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                else:
                    break

    # Define a new tarfile_samples
    def tarfile_samples(
        src,
        handler=reraise_exception,
        load_from_object_store=False,
        s3_client=None,
        s3_bucket_name=None,
        local_root_path=None,
    ):
        r"""
        Given an iterator of filenames, this function opens the URL streams
        and groups data by keys.

        Args:
            src: Iterator of data dictionaires containing URL names.
            handler: Exception handler.
            load_from_object_store (bool): A boolean flag to specify whether to load from
                object store.
            s3_client: If loading from object store, specify S3 client.
            s3_bucket_name: If loading from object store, specify S3 bucket name.
            local_root_path: If loading from local (or mounted) disk system,
                    specify the root path of the dataset.
        """
        streams = url_opener(
            src,
            handler=handler,
            object_store=load_from_object_store,
            s3_client=s3_client,
            s3_bucket_name=s3_bucket_name,
            local_root_path=local_root_path,
        )
        files = tar_file_expander(streams, handler=handler)
        samples = group_by_keys(files, handler=handler)
        return samples

    tarfile_to_samples = filters.pipelinefilter(tarfile_samples)

    class WebDataset(DataPipeline, FluidInterface):
        r"""Webdataset class modified to support loading from object store."""

        def __init__(
            self,
            urls,
            handler=reraise_exception,
            resampled=False,
            shardshuffle=None,
            cache_size=-1,
            cache_dir=None,
            detshuffle=False,
            nodesplitter=shardlists.single_node_only,
            verbose=False,
            load_from_object_store=False,
            s3_client=None,
            s3_bucket_name=None,
            local_root_path=None,
        ):
            r"""
            Args:
                urls: An iterator containing a list of url names.
                handler: Exception handler.
                resampled: If true, sample shards from shard list with replacement.
                shardshuffle: If true, shuffles the entire shard list.
                cache_size: Size of cache.
                cache_dir: Path to store cache.
                detshuffle: Whether to use deterministic shuffling when shardshuffle is True.
                nodesplitter: Function for splitting urls among nodes.
                verbose: If True, prints logs.
                load_from_object_store (bool): A boolean flag to specify whether to load from
                    object store.
                s3_client: If loading from object store, specify S3 client.
                s3_bucket_name: If loading from object store, specify S3 bucket name.
                local_root_path: If loading from local (or mounted) disk system,
                    specify the root path of the dataset.
            """
            super().__init__()
            if isinstance(urls, IterableDataset):
                assert not resampled
                self.append(urls)
            elif isinstance(urls, str) and (urls.endswith(".yaml") or urls.endswith(".yml")):
                with (open(urls)) as stream:
                    spec = yaml.safe_load(stream)
                assert "datasets" in spec
                self.append(shardlists.MultiShardSample(spec))
            elif isinstance(urls, dict):
                assert "datasets" in urls
                self.append(shardlists.MultiShardSample(urls))
            elif resampled:
                self.append(shardlists.ResampledShards(urls))
            else:
                self.append(shardlists.SimpleShardList(urls))
                self.append(nodesplitter)
                self.append(shardlists.split_by_worker)
                if shardshuffle is True:
                    shardshuffle = 100
                if shardshuffle is not None:
                    if detshuffle:
                        self.append(filters.detshuffle(shardshuffle))
                    else:
                        self.append(filters.shuffle(shardshuffle))
            if cache_dir is None or cache_size == 0:
                self.append(
                    tarfile_to_samples(
                        handler=handler,
                        load_from_object_store=load_from_object_store,
                        s3_client=s3_client,
                        s3_bucket_name=s3_bucket_name,
                        local_root_path=local_root_path,
                    )
                )
            else:

                # We dont use cache.
                assert cache_size == -1 or cache_size > 0
                self.append(
                    cache.cached_tarfile_to_samples(
                        handler=handler, verbose=verbose, cache_size=cache_size, cache_dir=cache_dir,
                    )
                )


else:

    class WebDataset(ApexGuardDefaults):
        def __init__(self):
            super().__init__()
            logging.warning("Webdataset import failed! We recommend use `webdataset==0.2.48`.")
