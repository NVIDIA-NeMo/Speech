# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class StreamingEncoder(ABC):
    @abstractmethod
    def setup_streaming_params(
        self,
        max_look_ahead: int = 10000,
    ):
        """
        This function sets the needed values and parameters to perform streaming. The configuration (CacheAwareStreamingConfig) need to be stored in self.streaming_cfg.
        The streaming configuration is needed to simulate streaming inference. It would set the following
        """
        pass

    @abstractmethod
    def get_initial_cache_state(self, batch_size, dtype, device, max_dim):
        pass

    @staticmethod
    def to_numpy(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def cache_aware_stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        bypass_pre_encode=False,
    ):
        if self.streaming_cfg is None:
            self.setup_streaming_params()
        if drop_extra_pre_encoded is not None:
            prev_drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded
            self.streaming_cfg.drop_extra_pre_encoded = drop_extra_pre_encoded
        else:
            prev_drop_extra_pre_encoded = None

        if processed_signal_length is None:
            processed_signal_length = processed_signal.new_full(processed_signal.size(0), processed_signal.size(-1))

        encoder_output = self(
            audio_signal=processed_signal,
            length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

        encoder_output = self.streaming_post_process(encoder_output, keep_all_outputs=keep_all_outputs)

        if prev_drop_extra_pre_encoded is not None:
            self.streaming_cfg.drop_extra_pre_encoded = prev_drop_extra_pre_encoded

        return encoder_output


class _DummyStreamingCfg:
    """Minimal streaming config for stateless encoders.

    ``pre_encode_cache_size = 0`` tells the audio feature buffer that no
    pre-encode overlap frames are needed between chunks — the encoder
    processes each chunk independently with no temporal state.
    """

    pre_encode_cache_size: int = 0


class DummyCacheAwareStreamingMixin(StreamingEncoder):
    """Mixin for stateless encoders to satisfy the cache-aware streaming
    interface required by ``StreamingSTTModel`` inference.

    Encoders that process each audio chunk independently (no temporal KV cache,
    no recurrent state) inherit this mixin instead of implementing the full
    ``StreamingEncoder`` protocol.  All streaming cache arguments passed by the
    inference pipeline are silently ignored; ``get_initial_cache_state`` returns
    ``(None, None, None)`` so downstream code that guards on ``is not None``
    skips all cache-update logic.

    Usage::

        class MyEncoder(nn.Module, DummyCacheAwareStreamingMixin):
            ...
    """

    streaming_cfg: _DummyStreamingCfg = _DummyStreamingCfg()

    def setup_streaming_params(self, **kwargs) -> None:
        """No-op: stateless encoders have no streaming params to configure."""

    def get_initial_cache_state(
        self,
        batch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Tuple[None, None, None]:
        """Return empty cache — this encoder carries no state between chunks."""
        return None, None, None
