# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nemo.collections.asr.parts.submodules.streaming_encoder_cuda_graphs import CudaGraphsStreamingEncoderStep


class StreamingEncoder(ABC):
    # Optional CUDA-graph accelerator for `cache_aware_stream_step`, attached by
    # `set_streaming_cuda_graphs`. Kept as a plain (non-module) attribute.
    _stream_step_cuda_graphs = None

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

    def set_streaming_cuda_graphs(
        self, enabled: bool = True, warmup_steps: int = 3, max_graphs: int = 8
    ) -> CudaGraphsStreamingEncoderStep | None:
        """Enable or disable CUDA-graph replay for `cache_aware_stream_step` (inference only).

        When enabled, steady-state streaming steps are captured once into a
        :class:`torch.cuda.CUDAGraph` and replayed with a single kernel launch, removing the
        per-step host launch overhead of the eager encoder. Non-uniform steps (first step,
        final step with ``keep_all_outputs=True``) automatically run eager. For the covered
        non-autocast configurations the graph path is expected to preserve eager execution
        semantics (the tests assert ``torch.equal`` against eager). See
        :class:`~nemo.collections.asr.parts.submodules.streaming_encoder_cuda_graphs.CudaGraphsStreamingEncoderStep`.

        Args:
            enabled: attach (True) or remove (False) the CUDA-graph accelerator.
            warmup_steps: eager calls per unique step shape before capturing it.
            max_graphs: maximum number of distinct captured graphs kept alive.

        Returns:
            The attached ``CudaGraphsStreamingEncoderStep`` helper, or None when disabling.
        """
        if not enabled:
            helper = self._stream_step_cuda_graphs
            if helper is not None:
                helper.disable_cuda_graphs()
            self._stream_step_cuda_graphs = None
            return None
        self._stream_step_cuda_graphs = CudaGraphsStreamingEncoderStep(
            self, warmup_steps=warmup_steps, max_graphs=max_graphs
        )
        return self._stream_step_cuda_graphs

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
        if self._stream_step_cuda_graphs is not None:
            return self._stream_step_cuda_graphs.stream_step(
                processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=keep_all_outputs,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
                bypass_pre_encode=bypass_pre_encode,
            )
        return self._cache_aware_stream_step_impl(
            processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            bypass_pre_encode=bypass_pre_encode,
        )

    def _cache_aware_stream_step_impl(
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
