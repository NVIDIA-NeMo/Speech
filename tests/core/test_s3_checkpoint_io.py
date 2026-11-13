# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from werkzeug.wrappers import Response


@pytest.mark.unit
def test_checkpoint_round_trip_with_s3_compatible_profile(tmp_path, monkeypatch, httpserver):
    pytest.importorskip('boto3')
    from nemo.utils.callbacks import s3_checkpoint_io
    from nemo.utils.s3_utils import S3Utils

    config = tmp_path / 'config'
    config.write_text(
        '[profile hf-checkpoints]\n'
        'region = us-east-1\n'
        f'endpoint_url = {httpserver.url_for("").rstrip("/")}\n'
        'request_checksum_calculation = when_required\n'
        'response_checksum_validation = when_required\n'
        's3 =\n'
        '    addressing_style = path\n',
        encoding='utf-8',
    )
    credentials = tmp_path / 'credentials'
    credentials.write_text(
        '[hf-checkpoints]\naws_access_key_id = test-key\naws_secret_access_key = test-secret\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('AWS_CONFIG_FILE', str(config))
    monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', str(credentials))
    monkeypatch.setenv('AWS_PROFILE', 'hf-checkpoints')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    for name in (
        'AWS_DEFAULT_PROFILE',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_SESSION_TOKEN',
        'AWS_SECURITY_TOKEN',
        'AWS_ENDPOINT_URL',
        'AWS_ENDPOINT_URL_S3',
        'AWS_IGNORE_CONFIGURED_ENDPOINT_URLS',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(s3_checkpoint_io, 'SHARED_MEM_DIR', str(tmp_path / 'missing'))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    checkpoint_data = {}
    key = 'run/model-last.ckpt'
    object_path = f'/checkpoints/{key}'

    def save_object(request):
        checkpoint_data[key] = request.get_data()
        return Response(status=200)

    def list_objects(request):
        assert request.args['list-type'] == '2'
        prefix = request.args['prefix']
        if prefix in (key, 'empty/'):
            return Response('<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>')

        assert prefix == 'run/'
        if 'continuation-token' not in request.args:
            return Response(
                '<ListBucketResult><IsTruncated>true</IsTruncated>'
                '<NextContinuationToken>next</NextContinuationToken>'
                '<Contents><Key>run/notes.txt</Key></Contents></ListBucketResult>'
            )

        assert request.args['continuation-token'] == 'next'
        assert key in checkpoint_data
        return Response(
            f'<ListBucketResult><IsTruncated>false</IsTruncated><Contents><Key>{key}</Key></Contents></ListBucketResult>'
        )

    httpserver.expect_request('/checkpoints', method='GET').respond_with_handler(list_objects)
    httpserver.expect_request(object_path, method='PUT').respond_with_handler(save_object)
    httpserver.expect_request(object_path, method='HEAD').respond_with_handler(
        lambda _: Response(headers={'Content-Length': str(len(checkpoint_data[key]))})
    )
    httpserver.expect_request(object_path, method='GET').respond_with_handler(lambda _: Response(checkpoint_data[key]))

    directory = 's3://checkpoints/run/'
    path = f's3://checkpoints/{key}'
    checkpoint_io = s3_checkpoint_io.S3CheckpointIO(directory)
    checkpoint = {'step': 4, 'weights': torch.arange(3)}
    checkpoint_io.save_checkpoint(checkpoint, path)
    assert S3Utils.s3_path_exists(directory, match_directory=True)
    assert S3Utils.find_files_with_suffix(directory) == ['run/notes.txt', key]
    assert S3Utils.find_files_with_suffix(directory, 'last.ckpt', return_key_only=False) == [path]
    assert S3Utils.find_files_with_suffix('s3://checkpoints/empty/') == []
    restored = checkpoint_io.load_checkpoint(path, weights_only=True)
    assert restored['step'] == 4
    torch.testing.assert_close(restored['weights'], checkpoint['weights'])
