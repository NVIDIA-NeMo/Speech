# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from dataclasses import dataclass
from typing import BinaryIO, Optional

import boto3
from botocore.config import Config


@dataclass
class S3Config:
    """Configuration required to connect to the target S3-compatible storage."""

    bucket: str
    endpoint_url: str
    region_name: str
    connect_timeout: int = 10


class S3Client:
    """Client for uploading report artifacts to S3-compatible object storage."""

    def __init__(
        self,
        cfg: S3Config,
        aws_access_key_id: str,
        aws_secret_access_key: str,
    ) -> None:
        self.cfg = cfg
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=cfg.region_name,
            config=Config(connect_timeout=cfg.connect_timeout),
        )

    def upload_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        expires_in: int,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload a binary file-like object and return a presigned download URL.

        Args:
            fileobj: File-like object to upload.
            key: S3 object key for the uploaded file.
            expires_in: Lifetime of the presigned URL in seconds.
            content_type: Optional content type stored with the uploaded object.

        Returns:
            Presigned URL for downloading the uploaded object.
        """
        kwargs = {
            "Fileobj": fileobj,
            "Bucket": self.cfg.bucket,
            "Key": key,
        }

        if content_type is not None:
            kwargs["ExtraArgs"] = {"ContentType": content_type}

        self.client.upload_fileobj(**kwargs)

        return self.get_presigned_url(key, expires_in)

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        expires_in: int,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload raw bytes and return a presigned download URL.

        Args:
            data: File content to upload.
            key: S3 object key for the uploaded file.
            expires_in: Lifetime of the presigned URL in seconds.
            content_type: Optional content type stored with the uploaded object.

        Returns:
            Presigned URL for downloading the uploaded object.
        """
        extra_args = {}

        if content_type is not None:
            extra_args["ContentType"] = content_type

        self.client.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=data,
            **extra_args,
        )

        return self.get_presigned_url(key, expires_in)

    def get_presigned_url(
        self,
        key: str,
        expires_in: int,
    ) -> str:
        """Generate a presigned download URL for an uploaded object.

        Args:
            key: S3 object key.
            expires_in: Lifetime of the presigned URL in seconds.

        Returns:
            Presigned URL for the requested object.
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.cfg.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def close(self) -> None:
        """Close the underlying S3 client."""
        self.client.close()
