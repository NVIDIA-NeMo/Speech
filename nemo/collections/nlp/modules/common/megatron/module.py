# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

from nemo.utils import logging

try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, config: ModelParallelConfig = None, share_token_embeddings=True):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super(MegatronModule, self).__init__()
        self.config = config
        self.share_token_embeddings = share_token_embeddings

    def word_embeddings_weight(self):
        if self.pre_process:
            if hasattr(self, 'language_model'):
                return self.language_model.embedding.word_embeddings.weight
            elif hasattr(self, 'encoder_embedding'):
                return self.encoder_embedding.word_embeddings.weight
            elif hasattr(self, 'decoder_embedding'):
                return self.decoder_embedding.word_embeddings.weight
            else:
                raise ValueError(
                    f"Pre_process is True, but no embedding is found on this rank. Looked for language_model.embedding, encoder_embedding, and decoder_embedding"
                )
        else:
            # This is the pipeline parallel last stage.
            if not self.share_token_embeddings:
                raise Exception(
                    'word_embeddings_weight() called for last ' 'stage, but share_token_embeddings is false'
                )
            return self.word_embeddings.weight

    def position_embeddings_weight(self):
        if self.pre_process:
            if hasattr(self, 'language_model'):
                return self.language_model.embedding.position_embeddings.weight
            elif hasattr(self, 'encoder_embedding'):
                return self.encoder_embedding.position_embeddings.weight
            elif hasattr(self, 'decoder_embedding'):
                return self.decoder_embedding.position_embeddings.weight
            else:
                raise ValueError(
                    f"Pre_process is True, but no embedding is found on this rank. Looked for language_model.embedding, encoder_embedding, and decoder_embedding"
                )
        else:
            # We only need position embeddings on the encoder and decoder first stages where pre_process=True
            raise ValueError(f"Pre_process is False, there is no position embedding on this rank.")

    def encoder_relative_position_embeddings_weight(self):
        if hasattr(self, 'encoder_relative_position_embedding'):
            return self.encoder_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                f"No encoder_relative_position_embedding found on this rank. Looking for encoder_relative_position_embedding.relative_position_embedding.weight"
            )

    def decoder_relative_position_embeddings_weight(self):
        if hasattr(self, 'decoder_relative_position_embedding'):
            return self.decoder_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                f"No decoder_relative_position_embedding found on this rank. Looking for decoder_relative_position_embedding.relative_position_embedding.weight"
            )

    def decoder_cross_attention_relative_position_embeddings_weight(self):
        if hasattr(self, 'decoder_cross_attention_relative_position_embedding'):
            return self.decoder_cross_attention_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                f"No decoder_cross_attention_relative_position_embedding found on this rank. Looking for decoder_cross_attention_relative_position_embedding.relative_position_embedding.weight"
            )

    def initialize_word_embeddings(self, init_method, vocab_size, hidden_size):
        if not self.share_token_embeddings:
            raise Exception('initialize_word_embeddings() was called but ' 'share_token_embeddings is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. If we aren't using pipeline
        # parallelism there is nothing to do.
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            return

        # Parameters are shared between the word embeddings layer, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if parallel_state.is_pipeline_last_stage() and not self.pre_process:
            # This is relevant for T5 when the decoder is only on a single rank. It is the last stage of the pipeline and also has embeddings on this rank already.
            assert not parallel_state.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                vocab_size, hidden_size, init_method=init_method, config=self.config,
            )
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        # This is the case where PP > 1 and we're on the decoder first stage.
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True) and self.pre_process:
            if hasattr(self, 'language_model'):
                # Zero params for GPT
                self.language_model.embedding.zero_parameters()
            else:
                # Zero decoder embeddings for T5
                assert hasattr(self, 'decoder_embedding')
                self.decoder_embedding.zero_parameters()

    def sync_initial_word_embeddings(self):

        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group() and self.share_token_embeddings:
                torch.distributed.all_reduce(
                    self.word_embeddings_weight().data, group=parallel_state.get_embedding_group()
                )
        else:
            logging.warning(
                "WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not synchronized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )

    def sync_initial_position_embeddings(self):
        # Ensure that the encoder first stage and decoder first have the same
        # initial position embedding parameter values.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if (
            parallel_state.is_rank_in_position_embedding_group()
            and parallel_state.get_pipeline_model_parallel_split_rank() is not None
        ):
            # TODO: Support tokentype embedding.
            # self.language_model.embedding.cuda()
            position_embeddings = self.position_embeddings_weight()
            torch.distributed.all_reduce(position_embeddings.data, group=parallel_state.get_position_embedding_group())

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def sync_initial_encoder_relative_position_embeddings(self):
        # Ensure that all encoder RPE stages have the same weights.
        if parallel_state.is_rank_in_encoder_relative_position_embedding_group():
            position_embeddings = self.encoder_relative_position_embeddings_weight()
            torch.distributed.all_reduce(
                position_embeddings.data, group=parallel_state.get_encoder_relative_position_embedding_group()
            )

    def sync_initial_decoder_relative_position_embeddings(self):
        if parallel_state.is_rank_in_decoder_relative_position_embedding_group():
            position_embeddings = self.decoder_relative_position_embeddings_weight()
            torch.distributed.all_reduce(
                position_embeddings.data, group=parallel_state.get_decoder_relative_position_embedding_group()
            )

    def sync_initial_decoder_cross_attention_relative_position_embeddings(self):
        if parallel_state.is_rank_in_decoder_relative_position_embedding_group():
            position_embeddings = self.decoder_cross_attention_relative_position_embeddings_weight()
            torch.distributed.all_reduce(
                position_embeddings.data, group=parallel_state.get_decoder_relative_position_embedding_group()
            )


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_converter):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_converter(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):
    def __init__(self, config: ModelParallelConfig, module, precision, share_token_embeddings=True):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "Megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(config=config, share_token_embeddings=share_token_embeddings)
        self.precision = precision

        if precision in ['bf16', 'bf16-mixed']:
            self.add_module('module', module.bfloat16())

            def float16_converter(val):
                return val.bfloat16()

        elif precision in [16, '16', '16-mixed']:
            self.add_module('module', module.half())

            def float16_converter(val):
                return val.half()

        else:
            raise Exception(
                f'precision {precision} is not supported. Float16Module (megatron_amp_O2) supports '
                'only fp16 and bf16.'
            )

        self.float16_converter = float16_converter

    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs, **kwargs):
        # Note: Legacy checkpoints didn't have pre-process.
        if getattr(self.module, 'pre_process', True):
            inputs = fp32_to_float16(inputs, self.float16_converter)
        outputs = self.module(*inputs, **kwargs)
        if parallel_state.is_pipeline_last_stage() and self.training:
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

    def word_embeddings_weight(self):
        if self.module.pre_process:
            if hasattr(self.module, 'language_model'):
                return self.module.language_model.embedding.word_embeddings.weight
            elif hasattr(self.module, 'encoder_embedding'):
                return self.module.encoder_embedding.word_embeddings.weight
            elif hasattr(self.module, 'decoder_embedding'):
                return self.module.decoder_embedding.word_embeddings.weight
            else:
                raise ValueError(
                    f"Pre_process is True, but no embedding is found on this rank. Looked for language_model.embedding, encoder_embedding, and decoder_embedding"
                )
        else:
            # This is the pipeline parallel last stage.
            if not self.share_token_embeddings:
                raise Exception(
                    'word_embeddings_weight() called for last ' 'stage, but share_token_embeddings is false'
                )
            return self.module.word_embeddings.weight

    def position_embeddings_weight(self):
        if self.module.pre_process:
            if hasattr(self.module, 'language_model'):
                return self.module.language_model.embedding.position_embeddings.weight
            elif hasattr(self.module, 'encoder_embedding'):
                return self.module.encoder_embedding.position_embeddings.weight
            elif hasattr(self.module, 'decoder_embedding'):
                return self.module.decoder_embedding.position_embeddings.weight
            else:
                raise ValueError(
                    f"Pre_process is True, but no embedding is found on this rank. Looked for language_model.position_embeddings, encoder_embedding.position_embedding_weight, and decoder_embedding.position_embedding_weight"
                )
        else:
            # We only need position embeddings on the encoder and decoder first stages where pre_process=True
            raise ValueError(f"Pre_process is False, there is no position embedding on this rank.")

    def encoder_relative_position_embeddings_weight(self):
        if hasattr(self.module, 'encoder_relative_position_embedding'):
            return self.module.encoder_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                f"No encoder_relative_position_embedding found on this rank. Looking for encoder_relative_position_embedding.relative_position_embedding.weight"
            )

    def decoder_relative_position_embeddings_weight(self):
        if hasattr(self.module, 'decoder_relative_position_embedding'):
            return self.module.decoder_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                f"No decoder_relative_position_embedding found on this rank. Looking for decoder_relative_position_embedding.relative_position_embedding.weight"
            )

    def decoder_cross_attention_relative_position_embeddings_weight(self):
        if hasattr(self.module, 'decoder_cross_attention_relative_position_embedding'):
            return self.module.decoder_cross_attention_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                f"No decoder_cross_attention_relative_position_embedding found on this rank. Looking for decoder_cross_attention_relative_position_embedding.relative_position_embedding.weight"
            )
