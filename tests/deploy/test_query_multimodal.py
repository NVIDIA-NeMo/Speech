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

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from nemo.deploy.multimodal.query_multimodal import NemoQueryMultimodal


class TestNemoQueryMultimodal:
    @pytest.fixture
    def query_multimodal(self):
        return NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="neva")

    @pytest.fixture
    def mock_image(self):
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name)
            return tmp.name

    @pytest.fixture
    def mock_video(self):
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            # Just create an empty file for testing
            return tmp.name

    @pytest.fixture
    def mock_audio(self):
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Just create an empty file for testing
            return tmp.name

    def test_init(self):
        nq = NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="neva")
        assert nq.url == "localhost"
        assert nq.model_name == "test_model"
        assert nq.model_type == "neva"

    def test_setup_media_image_local(self, query_multimodal, mock_image):
        result = query_multimodal.setup_media(mock_image)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # Batch dimension
        os.unlink(mock_image)

    @patch('requests.get')
    def test_setup_media_image_url(self, mock_get, query_multimodal):
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        # Mock Image.open
        with patch('PIL.Image.open') as mock_image_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image

            result = query_multimodal.setup_media("http://example.com/image.jpg")
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 1

    def test_frame_len(self, query_multimodal):
        # Test with frames less than max_frames
        frames = [np.zeros((100, 100, 3)) for _ in range(100)]
        assert query_multimodal.frame_len(frames) == 100

        # Test with frames more than max_frames
        frames = [np.zeros((100, 100, 3)) for _ in range(300)]
        result = query_multimodal.frame_len(frames)
        assert result <= 256  # Should be less than or equal to max_frames

    def test_get_subsampled_frames(self, query_multimodal):
        frames = [np.zeros((100, 100, 3)) for _ in range(10)]
        subsample_len = 5
        result = query_multimodal.get_subsampled_frames(frames, subsample_len)
        assert len(result) == subsample_len

    @patch('nemo.deploy.multimodal.query_multimodal.ModelClient')
    def test_query(self, mock_model_client, query_multimodal, mock_image):
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"outputs": np.array(["test response"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal.query(
            input_text="test prompt",
            input_media=mock_image,
            max_output_len=30,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )

        assert isinstance(result, np.ndarray)
        assert result[0] == "test response"
        os.unlink(mock_image)

    @patch('nemo.deploy.multimodal.query_multimodal.VideoReader')
    def test_setup_media_video(self, mock_video_reader, mock_video):
        nq = NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="video-neva")

        # Mock VideoReader
        mock_frames = [MagicMock(asnumpy=lambda: np.zeros((100, 100, 3))) for _ in range(10)]
        mock_video_reader.return_value = mock_frames

        result = nq.setup_media(mock_video)
        assert isinstance(result, np.ndarray)
        os.unlink(mock_video)

    @patch('soundfile.read')
    def test_setup_media_audio(self, mock_sf_read, mock_audio):
        nq = NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="salm")

        # Mock soundfile.read
        mock_sf_read.return_value = (np.zeros(1000), 16000)

        result = nq.setup_media(mock_audio)
        assert isinstance(result, dict)
        assert "input_signal" in result
        assert "input_signal_length" in result
        os.unlink(mock_audio)
