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

"""Retrieval Transformer."""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.position_embedding import RotaryEmbedding
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, build_attention_mask_3d

try:
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()
    HAVE_APEX = False

try:
    from megatron.core import ModelParallelConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

MIN_DIM_HEAD = 32


class MegatronRetrievalTransformerEncoderModule(MegatronModule):
    """Transformer encoder model.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        layer_type=[],
        pre_process=True,
        post_process=True,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_granularity=None,
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        bias=True,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        parent_model_type=ModelType.encoder_or_decoder,
        chunk_size=64,
        layer_number_offset=0,  # this is use only for attention norm_factor scaling
        normalize_attention_scores=True,
        megatron_legacy=False,
        turn_off_rop=False,
        version=1,  # model version
    ):
        super(MegatronRetrievalTransformerEncoderModule, self).__init__(config=config)

        self.transformer_block_type = transformer_block_type
        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_method = init_method
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.parent_model_type = parent_model_type
        self.turn_off_rop = turn_off_rop
        self.version = version

        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Transformer.
        self.model = ParallelTransformer(
            config=config,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            layer_type=layer_type,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            model_type=parent_model_type,
            chunk_size=chunk_size,
            layer_number_offset=layer_number_offset,
            normalize_attention_scores=normalize_attention_scores,
            megatron_legacy=megatron_legacy,
        )
        rot_dim = hidden_size // num_attention_heads if kv_channels is None else kv_channels
        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/
        if not turn_off_rop:
            self.rotary_pos_emb = RotaryEmbedding(min(rot_dim, MIN_DIM_HEAD))
        self.chunk_size = chunk_size
        self._model_key = 'model'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)

    def _allocate_memory(self, *shape, dtype):
        return torch.empty(*shape, dtype=dtype, device=torch.cuda.current_device())

    def forward(
        self,
        enc_input,
        enc_attn_mask,
        context_attn_mask=None,
        encoder_output=None,
        layer_past=None,
        get_key_value=False,
        set_inference_key_value_memory=False,  # when doing inference, set this to true to allocate all the cached matrix. later set false to do incremental inference
        inference_max_sequence_len=None,
        neighbors=2,
    ):
        # expected enc_input shape [batch, num_chunks, num_neighbors, retrieval_seq_len, dim]
        # expected enc_attn_mask shape [batch, num_chunks, num_neighbors, retrieval_seq_len]
        # expected encoder_output shape [batch, seq_len, dim]

        # batch, seq_len, dim
        b, n, dim = encoder_output.shape

        if set_inference_key_value_memory:
            # run once to setup the cache
            chunk_start = 0
            num_seq_chunks = n // self.chunk_size
            num_chunks = inference_max_sequence_len // self.chunk_size
            self.cache_output = self._allocate_memory(
                b, num_chunks, neighbors, self.chunk_size * 2, dim, dtype=encoder_output.dtype
            )
            self.seq_pos_in_chunk = n
            self.current_chunk = n // self.chunk_size
            self.encoder_output = self._allocate_memory(b, self.chunk_size, dim, dtype=encoder_output.dtype)
            self.context_attn_mask = self._allocate_memory(b, self.chunk_size, dtype=context_attn_mask.dtype)
            self.context_attn_mask
            chunk_beg = self.chunk_size * num_seq_chunks
            chunk_end = self.chunk_size * num_seq_chunks + self.seq_pos_in_chunk % self.chunk_size
            # store the remainders
            self.encoder_output[:, : self.seq_pos_in_chunk % self.chunk_size, :] = encoder_output[
                :, chunk_beg:chunk_end, :
            ]
            self.context_attn_mask[:, : self.seq_pos_in_chunk % self.chunk_size] = context_attn_mask[
                :, chunk_beg:chunk_end
            ]
        elif inference_max_sequence_len is not None:
            # second time of running
            # only support one token at a time
            assert n == 1
            self.seq_pos_in_chunk += n
            self.current_chunk = self.seq_pos_in_chunk // self.chunk_size
            # if exceed the chunk size
            pos_beg = (self.seq_pos_in_chunk - 1) % self.chunk_size
            # if self.seq_pos_in_chunk - 1 >= self.chunk_size:
            #     self.current_chunk += 1
            #     self.seq_pos_in_chunk -= self.chunk_size
            chunk_start = self.current_chunk - 1
            self.encoder_output[:, pos_beg : pos_beg + 1, :] = encoder_output
            self.context_attn_mask[:, pos_beg : pos_beg + 1] = context_attn_mask[
                :, self.seq_pos_in_chunk - 1 : self.seq_pos_in_chunk
            ]
            encoder_output = self.encoder_output[:, : pos_beg + 1, :]
            context_attn_mask = self.context_attn_mask[:, : pos_beg + 1]
            num_seq_chunks = 1
            if not self.seq_pos_in_chunk % self.chunk_size == 0:
                # still accumulate the encoder_output
                # return the cached results
                if self.current_chunk == 0:
                    return None
                return self.cache_output[:, : self.current_chunk]
            if enc_input is not None:
                # only need one chunk for the later calculation
                enc_input = enc_input[:, self.current_chunk - 1 : self.current_chunk]
                enc_attn_mask = enc_attn_mask[:, self.current_chunk - 1 : self.current_chunk]

        if enc_input is None:
            return None

        _, k, r, rn, _ = enc_input.shape

        assert r == neighbors
        if inference_max_sequence_len is None:
            num_seq_chunks = n // self.chunk_size
            assert k == num_seq_chunks, f'sequence requires {num_seq_chunks} retrieved chunks, but only {k} passed in'
        else:
            pass

        seq_index = num_seq_chunks * self.chunk_size

        retrieved = rearrange(enc_input, 'b k r n d -> n (b k r) d')
        enc_attn_mask = rearrange(enc_attn_mask, 'b k r n -> (b k r) n')
        # embed_as_context = repeat(encoder_output[:, :seq_index], 'b (k n) d -> (b k r) n d', n=self.chunk_size, r=r)
        # context_attn_mask = repeat(context_attn_mask[:, :seq_index], 'b (k n) -> (b k r) n', n=self.chunk_size, r=r)

        if inference_max_sequence_len is not None and not set_inference_key_value_memory:
            embed_as_context = repeat(encoder_output[:, :seq_index], 'b (k n) d -> n (b k r) d', n=pos_beg + 1, r=r)
            context_attn_mask = repeat(context_attn_mask[:, :seq_index], 'b (k n) -> (b k r) n', n=pos_beg + 1, r=r)
        else:
            embed_as_context = repeat(
                encoder_output[:, :seq_index], 'b (k n) d -> n (b k r) d', n=self.chunk_size, r=r
            )
            context_attn_mask = repeat(
                context_attn_mask[:, :seq_index], 'b (k n) -> (b k r) n', n=self.chunk_size, r=r
            )

        if not self.turn_off_rop:
            if inference_max_sequence_len is not None and not set_inference_key_value_memory:
                cross_attn_k_pos_emb = self.rotary_pos_emb(n % self.chunk_size, offset=pos_beg)
            else:
                cross_attn_k_pos_emb = self.rotary_pos_emb(self.chunk_size, offset=0)
            cross_attn_q_pos_emb = self.rotary_pos_emb(rn, offset=0)
            attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_q_pos_emb, cross_attn_k_pos_emb)
        else:
            attn_pos_emb = None

        # # convert to Megatron mask
        enc_attn_mask_3d = build_attention_mask_3d(
            source_mask=enc_attn_mask, target_mask=enc_attn_mask, attn_mask_type=AttnMaskType.padding,
        )
        enc_attn_mask_3d = enc_attn_mask_3d[:, None, :, :]

        enc_dec_attn_mask_3d = build_attention_mask_3d(
            source_mask=enc_attn_mask, target_mask=context_attn_mask, attn_mask_type=AttnMaskType.padding,
        )
        enc_dec_attn_mask_3d = enc_dec_attn_mask_3d[:, None, :, :]

        # transformer encoder
        enc_output = self.model(
            retrieved,
            enc_attn_mask_3d,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=embed_as_context,
            enc_dec_attn_mask=enc_dec_attn_mask_3d,
            rotary_pos_emb=attn_pos_emb,
        )
        # revert back to original retrieved shape
        enc_output = rearrange(enc_output, 'n (b k r) d -> b k r n d', b=b, k=k)

        if inference_max_sequence_len is not None:
            # update encoded for current chunk
            self.cache_output[:, chunk_start : self.current_chunk, :, :, :] = enc_output
            # read all encodings
            enc_output = self.cache_output[:, : self.current_chunk]
        return enc_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._model_key] = self.model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Encoder.
        if self._model_key in state_dict:
            state_dict_ = state_dict[self._model_key]
        self.model.load_state_dict(state_dict_, strict=strict)


class MegatronRetrievalTransformerDecoderModule(MegatronModule):
    """Transformer decoder model.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        layer_type=[],
        pre_process=True,
        post_process=True,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_granularity=None,
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        bias=True,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        parent_model_type=ModelType.encoder_or_decoder,
        chunk_size=64,
        layer_number_offset=0,  # this is use only for attention norm_factor scaling
        normalize_attention_scores=True,
        megatron_legacy=False,
        turn_off_rop=False,
        version=1,  # model version
    ):
        super(MegatronRetrievalTransformerDecoderModule, self).__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_method = init_method
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.parent_model_type = parent_model_type
        self.turn_off_rop = turn_off_rop
        self.version = version

        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Transformer.
        self.model = ParallelTransformer(
            config=config,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            layer_type=layer_type,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=AttnMaskType.padding,  # we use attention mask reset, enforce to use padding AttnMaskType, otherwise it has numeric issues
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            model_type=parent_model_type,
            chunk_size=chunk_size,
            layer_number_offset=layer_number_offset,
            normalize_attention_scores=normalize_attention_scores,
            megatron_legacy=megatron_legacy,
        )
        rot_dim = hidden_size // num_attention_heads if kv_channels is None else kv_channels
        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/
        if not turn_off_rop:
            self.rotary_pos_emb = RotaryEmbedding(min(rot_dim, MIN_DIM_HEAD))
        self.chunk_size = chunk_size
        self._model_key = 'model'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)

    def _calculate_dec_att_mask(self, dec_attn_mask, eod_positions):
        # # convert to Megatron mask

        # customized attention mask, starts with causal attention mask
        dec_attn_mask_3d = build_attention_mask_3d(
            source_mask=dec_attn_mask, target_mask=dec_attn_mask, attn_mask_type=AttnMaskType.causal,
        )
        # add the attention mask reset
        if eod_positions is not None:
            # to mask out the token ids [id, id,  eod, id, pad, eod, id, id]
            # so attention is not across eod, mask should be:
            # [false, true,  true, true,  true, true,  true,  true]
            # [false, false, true, true,  true, true,  true,  true]
            # [false, false, false,true,  true, true,  true,  true]
            # [true,  true,  true, false, true, true,  true,  true]
            # [true,  true,  true, true,  true, true,  true,  true]
            # [true,  true,  true, false, true, false, true,  true]
            # [true,  true,  true, true,  true, true,  false, true]
            # [true,  true,  true, true,  true, true,  false, false]
            for batch, eod_pos in zip(*eod_positions):
                eod_plus_one = eod_pos.item() + 1
                dec_attn_mask_3d[batch][eod_plus_one:, :eod_plus_one] = True
        dec_attn_mask_3d = dec_attn_mask_3d[:, None, :, :]
        return dec_attn_mask_3d

    def forward(
        self,
        dec_input,
        dec_attn_mask,
        retrieved_attn_mask=None,
        retrieved_emb=None,
        layer_past=None,
        get_key_value=False,
        eod_positions=None,  # this is a tuple of eod positions returned from tensor.where(tensor == eod_id)
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        # expected dec_input shape [batch, seq_len, dim]
        # expected dec_attn_mask shape [batch, seq_len]
        # expected retrieved_input shape [batch, num_chunks, num_neighbors, retrival_seq_len, dim]
        # expected retrieved_attn_mask shape [batch, num_chunks, num_neighbors, retrival_seq_len]

        # batch, seq_len, dim
        if isinstance(dec_input, tuple):
            n, _, _ = dec_input[1].shape
        else:
            _, n, _ = dec_input.shape

        if set_inference_key_value_memory:
            # seq_index = (n // chunk_size) * chunk_size
            self.current_len = n
            num_seq_chunks = self.current_len // self.chunk_size
        elif inference_max_sequence_len is not None:
            # only handles single token increment
            assert n == 1
            self.current_len += n
            num_seq_chunks = self.current_len // self.chunk_size
        else:
            # this is normal forward without inference
            num_seq_chunks = n // self.chunk_size

        if retrieved_emb is not None:
            b, k, r, rn, dim = retrieved_emb.shape
            assert (
                k == num_seq_chunks
            ), f'sequence requires {num_seq_chunks} retrieved chunks, but only {k} passed in'  # need to add extra chunk size, since it will be shifted

        if not self.turn_off_rop:
            if set_inference_key_value_memory:
                self_attn_emb = self.rotary_pos_emb(self.current_len)
            elif inference_max_sequence_len is not None:
                self_attn_emb = self.rotary_pos_emb(self.current_len)
            else:
                self_attn_emb = self.rotary_pos_emb(n)
            if retrieved_emb is not None:
                # -63, -62, ... 63  will be cut into -> [0, ... 63] in the chunk cross attention layer
                cross_attn_q_pos_emb = self.rotary_pos_emb(self.chunk_size * 2 - 1, offset=-self.chunk_size + 1)
                if self.version == 1:
                    cross_attn_k_pos_emb = self.rotary_pos_emb(rn, offset=0)
                elif self.version > 1:
                    # the first 64 tokens in retrieved is from the last chunk, align the continuation part with the query tokens
                    # use the following in the future. [-63, -62, ..., 63, 64]
                    cross_attn_k_pos_emb = self.rotary_pos_emb(rn, offset=-self.chunk_size + 1)
                else:
                    raise ValueError(f'incorrect version number {self.version}')
                attn_pos_emb = (self_attn_emb, cross_attn_q_pos_emb, cross_attn_k_pos_emb)
            else:
                attn_pos_emb = (self_attn_emb, None, None)
        else:
            attn_pos_emb = None

        dec_attn_mask_3d = self._calculate_dec_att_mask(dec_attn_mask, eod_positions)

        if retrieved_emb is not None:
            # need to shift the dec_attn_mask as first causal_padding elements are ignored
            # also pad it to be the multiple of self.chunk_size
            causal_padding = self.chunk_size - 1
            reminder = (self.chunk_size - (dec_attn_mask.shape[1] + 1)) % self.chunk_size
            dec_attn_mask = F.pad(dec_attn_mask, (-causal_padding, reminder), value=False)

            dec_attn_mask = rearrange(dec_attn_mask, 'b (k n) -> (b k) n', k=k)
            retrieved_attn_mask = rearrange(retrieved_attn_mask, 'b k r n -> (b k) (r n)')

            enc_dec_attn_mask_3d = build_attention_mask_3d(
                source_mask=dec_attn_mask, target_mask=retrieved_attn_mask, attn_mask_type=AttnMaskType.padding,
            )
            enc_dec_attn_mask_3d = enc_dec_attn_mask_3d[:, None, :, :]
        else:
            enc_dec_attn_mask_3d = None

        # transformer encoder
        if not isinstance(dec_input, tuple):
            dec_input = rearrange(dec_input, 'b s d -> s b d').contiguous()
        enc_output = self.model(
            dec_input,
            dec_attn_mask_3d,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=None,
            retrieved_emb=retrieved_emb,
            enc_dec_attn_mask=enc_dec_attn_mask_3d,
            rotary_pos_emb=attn_pos_emb,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
        )
        # enc_output = rearrange(dec_input, 's b d -> b s d')
        return enc_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._model_key] = self.model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Encoder.
        if self._model_key in state_dict:
            state_dict_ = state_dict[self._model_key]
        self.model.load_state_dict(state_dict_, strict=strict)
