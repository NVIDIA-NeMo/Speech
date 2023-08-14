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


import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.attention import ParallelChunkedCrossAttention
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.position_embedding import RotaryEmbedding
from nemo.collections.nlp.modules.common.megatron.retrieval_token_level_encoder_decoder import (
    MegatronRetrievalTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.retrieval_transformer import (
    MegatronRetrievalTransformerDecoderModule,
    MegatronRetrievalTransformerEncoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    build_attention_mask_3d,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import ModelParallelConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


@pytest.fixture()
def model_parallel_config():
    config = ModelParallelConfig()
    return config


@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not HAVE_APEX or not HAVE_MEGATRON_CORE, reason="apex or megatron-core is not installed")
class TestRetrievalModuleInference:
    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        GPUS = 1
        TP_SIZE = GPUS
        PP_SIZE = 1
        MB_SIZE = 4
        GB_SIZE = 8
        SEED = 1234
        trainer = Trainer(strategy=NLPDDPStrategy(), devices=GPUS, accelerator='gpu', num_nodes=1, logger=None,)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            micro_batch_size=MB_SIZE,
            global_batch_size=GB_SIZE,
            seed=SEED,
            apex_transformer_log_level=30,
        )

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()
        torch.distributed.barrier()

    @pytest.mark.unit
    def test_retrieval_encoder_inference(self, model_parallel_config):

        init_method_std = 0.02

        batch = 2
        neighbors = 2
        # rotary pos emb dim
        dim = 128
        pad_id = 19999
        num_attention_heads = 8
        chunks = 32
        text_chunk_size = 64
        input_length = chunks * text_chunk_size
        vocab_size = 20000

        hidden = torch.randint(0, vocab_size, (batch, input_length)).cuda()  # (seq, batch, dim)
        hidden_mask = (hidden != pad_id).cuda()

        hidden_emb = torch.rand(batch, input_length, dim).cuda().half()  # (batch, seq, dim)
        retrieved = torch.randint(0, vocab_size, (batch, chunks, neighbors, 2 * text_chunk_size)).cuda()
        pad_id = vocab_size - 1
        context_mask = (retrieved != pad_id).cuda()
        retrieved_emb = torch.rand(batch, chunks, neighbors, 2 * text_chunk_size, dim).cuda().half()

        layer_type = [LayerType.encoder, LayerType.retrieval_encoder, LayerType.encoder, LayerType.retrieval_encoder]
        num_layers = len(layer_type)

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        encoder = (
            MegatronRetrievalTransformerEncoderModule(
                config=model_parallel_config,
                init_method=init_method,
                output_layer_init_method=scaled_init_method,
                hidden_size=dim,
                ffn_hidden_size=dim * 4,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                precision=16,
                chunk_size=text_chunk_size,
                layer_type=layer_type,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
            .cuda()
            .half()
        )
        out_gt = encoder(retrieved_emb, context_mask, context_attn_mask=hidden_mask, encoder_output=hidden_emb)
        assert out_gt.shape == torch.Size([batch, chunks, neighbors, 2 * text_chunk_size, dim])

        out_1 = encoder(
            None,
            None,
            context_attn_mask=hidden_mask[:, :62],
            encoder_output=hidden_emb[:, :62, :],
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert out_1 is None
        out_1 = encoder(
            None,
            None,
            context_attn_mask=hidden_mask[:, :63],
            encoder_output=hidden_emb[:, 62:63],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert out_1 is None
        out_2 = encoder(
            retrieved_emb[:, :1],
            context_mask[:, :1],
            context_attn_mask=hidden_mask[:, :64],
            encoder_output=hidden_emb[:, 63:64],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (encoder.encoder_output - hidden_emb[:, :64]).abs().max().item() < 1e-5
        assert (out_gt[:, 0,] - out_2[:, 0]).abs().max().item() < 1e-2
        out_test = encoder(
            retrieved_emb[:, :1],
            context_mask[:, :1],
            context_attn_mask=hidden_mask[:, :64],
            encoder_output=hidden_emb[:, :64],
        )
        assert (out_gt[:, 0,] - out_test[:, 0]).abs().max().item() < 1e-2
        assert (out_gt[:, 0,] - out_2[:, 0]).abs().max().item() < 1e-2

        for i in range(64, 127):
            out_3 = encoder(
                retrieved_emb[:, :1],
                context_mask[:, :1],
                context_attn_mask=hidden_mask[:, : i + 1],
                encoder_output=hidden_emb[:, i : i + 1],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
        i = 127
        out_3 = encoder(
            retrieved_emb[:, :2],
            context_mask[:, :2],
            context_attn_mask=hidden_mask[:, : i + 1],
            encoder_output=hidden_emb[:, i : i + 1],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (encoder.encoder_output - hidden_emb[:, 64:128]).abs().max().item() < 1e-5
        assert (out_gt[:, :2,] - out_3).abs().max().item() < 1e-2
        # test inference
        for i in range(128, 191):
            out_4 = encoder(
                retrieved_emb[:, :2],
                context_mask[:, :2],
                context_attn_mask=hidden_mask[:, : i + 1],
                encoder_output=hidden_emb[:, i : i + 1],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
        i = 191
        out_4 = encoder(
            retrieved_emb[:, :3],
            context_mask[:, :3],
            context_attn_mask=hidden_mask[:, : i + 1],
            encoder_output=hidden_emb[:, i : i + 1],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )

        assert (encoder.encoder_output - hidden_emb[:, 128:192]).abs().max().item() < 1e-5
        assert (out_gt[:, :3,] - out_4).abs().max().item() < 1e-2

        out_2 = encoder(
            retrieved_emb[:, :2],
            context_mask[:, :2],
            context_attn_mask=hidden_mask[:, :130],
            encoder_output=hidden_emb[:, :130, :],
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        for i in range(130, 191):
            out_2 = encoder(
                retrieved_emb[:, :2],
                context_mask[:, :2],
                context_attn_mask=hidden_mask[:, : i + 1],
                encoder_output=hidden_emb[:, i : i + 1],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
        i = 191
        out_4 = encoder(
            retrieved_emb[:, :3],
            context_mask[:, :3],
            context_attn_mask=hidden_mask[:, : i + 1],
            encoder_output=hidden_emb[:, i : i + 1],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (encoder.encoder_output - hidden_emb[:, 128:192]).abs().max().item() < 1e-5
        assert (out_gt[:, :3,] - out_4).abs().max().item() < 1e-2

    @pytest.mark.unit
    def test_cross_attn_inference(self, model_parallel_config):
        num_layers = 1
        init_method_std = 0.02
        batch = 2
        neighbors = 2
        # rotary pos emb dim
        dim = 128
        pad_id = 19999
        num_attention_heads = 8
        chunks = 32
        text_chunk_size = 64
        context_chunk_size = 2 * text_chunk_size
        input_length = chunks * text_chunk_size
        vocab_size = 20000

        rot_dim = dim // num_attention_heads
        rotary_pos_emb = RotaryEmbedding(rot_dim).cuda().half()

        hidden = torch.randint(0, vocab_size, (input_length, batch)).cuda()  # (seq, batch, dim)
        hidden_mask = (hidden != pad_id).cuda()
        hidden_emb = torch.rand(input_length, batch, dim).cuda().half()  # (seq, batch, dim)

        retrieved = torch.randint(0, vocab_size, (chunks, neighbors, context_chunk_size, batch)).cuda()
        # retrieved tokens - (num chunks, num retrieved neighbors, retrieved chunk with continuation, batch)

        # context attention mask [b, np, sq, sk]
        context_mask = (retrieved != pad_id).cuda()
        retrieved_emb = torch.rand(chunks, neighbors, context_chunk_size, batch, dim).cuda().half()
        # retrieved tokens - (num chunks, num retrieved neighbors, retrieved chunk with continuation, batch, hidden)

        # need to add extra chunk size, since it will be shifted
        cross_attn_q_pos_emb = rotary_pos_emb(text_chunk_size + text_chunk_size - 1, offset=-text_chunk_size + 1)
        cross_attn_k_pos_emb = rotary_pos_emb(context_chunk_size)
        cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)

        def get_attn_mask_3d(hidden_mask, context_mask, chunks):
            causal_padding = text_chunk_size - 1
            reminder = (text_chunk_size - (hidden_mask.shape[0] + 1)) % text_chunk_size
            hidden_mask = F.pad(hidden_mask, (0, 0, -causal_padding, reminder), value=False)

            dec_attn_mask = rearrange(hidden_mask, '(k n) b -> (b k) n', k=chunks)
            context_attn_mask = rearrange(context_mask, 'k r n b -> (b k) (r n)')
            enc_dec_attn_mask_3d = build_attention_mask_3d(
                source_mask=dec_attn_mask, target_mask=context_attn_mask, attn_mask_type=AttnMaskType.padding,
            )
            enc_dec_attn_mask_3d = enc_dec_attn_mask_3d[:, None, :, :]
            return enc_dec_attn_mask_3d

        enc_dec_attn_mask_3d = get_attn_mask_3d(hidden_mask, context_mask, chunks)

        init_method = init_method_normal(init_method_std)

        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        cross_attn = (
            ParallelChunkedCrossAttention(
                config=model_parallel_config,
                init_method=init_method,
                output_layer_init_method=scaled_init_method,
                layer_number=1,
                num_attention_heads=num_attention_heads,
                hidden_size=dim,
                precision=16,
                chunk_size=text_chunk_size,
                masked_softmax_fusion=False,
            )
            .cuda()
            .half()
        )

        out, bias = cross_attn(
            hidden_emb, enc_dec_attn_mask_3d, encoder_output=retrieved_emb, rotary_pos_emb=cross_attn_pos_emb
        )
        assert out.shape == torch.Size([input_length, batch, dim])
        assert bias.shape == torch.Size([dim])

        attn_mask_3d = None

        out_1, b = cross_attn(
            hidden_emb[:62],
            attn_mask_3d,
            encoder_output=None,
            rotary_pos_emb=cross_attn_pos_emb,
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
        )
        assert (out_1 - torch.zeros_like(hidden_emb[:62])).abs().max() == 0
        out_1, b = cross_attn(
            hidden_emb[62:63],
            attn_mask_3d,
            encoder_output=None,
            rotary_pos_emb=cross_attn_pos_emb,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
        )
        assert (out_1 - torch.zeros_like(hidden_emb[62:63])).abs().max() == 0

        attn_mask_3d = get_attn_mask_3d(hidden_mask[:64], context_mask[:1], 1)
        out_2, b = cross_attn(
            hidden_emb[63:64],
            attn_mask_3d,
            encoder_output=retrieved_emb[:1],
            rotary_pos_emb=cross_attn_pos_emb,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
        )
        assert (out[63] - out_2[0]).abs().max().item() < 1e-2

        for i in range(64, 127):
            attn_mask_3d = get_attn_mask_3d(hidden_mask[: i + 1], context_mask[:1], 1)
            out_2, b = cross_attn(
                hidden_emb[i : i + 1],
                attn_mask_3d,
                encoder_output=retrieved_emb[:1],
                rotary_pos_emb=cross_attn_pos_emb,
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
            )
        i = 127
        attn_mask_3d = get_attn_mask_3d(hidden_mask[: i + 1], context_mask[:2], 2)
        out_3, b = cross_attn(
            hidden_emb[i : i + 1],
            attn_mask_3d,
            encoder_output=retrieved_emb[:2],
            rotary_pos_emb=cross_attn_pos_emb,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
        )
        assert (out[i] - out_3[0]).abs().max().item() < 1e-2

        attn_mask_3d = get_attn_mask_3d(hidden_mask[:130], context_mask[:2], 2)

        out_1, b = cross_attn(
            hidden_emb[:130],
            attn_mask_3d,
            encoder_output=retrieved_emb[:2],
            rotary_pos_emb=cross_attn_pos_emb,
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
        )

        assert (out[:130] - out_1[:130]).abs().max().item() < 1e-2

        for i in range(130, 191):
            attn_mask_3d = get_attn_mask_3d(hidden_mask[: i + 1], context_mask[:2], 2)
            out_2, b = cross_attn(
                hidden_emb[i : i + 1],
                attn_mask_3d,
                encoder_output=retrieved_emb[:2],
                rotary_pos_emb=cross_attn_pos_emb,
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
            )
        i = 191
        attn_mask_3d = get_attn_mask_3d(hidden_mask[: i + 1], context_mask[:3], 3)
        out_4, b = cross_attn(
            hidden_emb[i : i + 1],
            attn_mask_3d,
            encoder_output=retrieved_emb[:3],
            rotary_pos_emb=cross_attn_pos_emb,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
        )
        assert (out[i] - out_4[0]).abs().max().item() < 1e-2

    @pytest.mark.unit
    def test_retrieval_decoder_inference(self, model_parallel_config):

        init_method_std = 0.02

        # rotary pos emb dim
        batch = 2
        neighbors = 2
        dim = 128
        pad_id = 19999
        num_attention_heads = 8
        chunks = 32
        text_chunk_size = 64
        input_length = chunks * text_chunk_size
        vocab_size = 20000
        # rot_dim = dim // num_attention_heads
        # rotary_pos_emb = RotaryEmbedding(rot_dim).cuda().half()
        hidden = torch.randint(0, vocab_size, (batch, input_length)).cuda()  # (seq, batch, dim)
        hidden_mask = (hidden != pad_id).cuda()

        hidden_emb = torch.rand(batch, input_length, dim).cuda().half()  # (batch, seq, dim)

        # context_chunk_size = 128
        retrieved = torch.randint(0, vocab_size, (batch, chunks, neighbors, 2 * text_chunk_size)).cuda()
        # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

        # context attention mask [b, np, sq, sk]
        pad_id = vocab_size - 1
        context_mask = (retrieved != pad_id).cuda()
        retrieved_emb = torch.rand(batch, chunks, neighbors, 2 * text_chunk_size, dim).cuda().half()
        # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation, hidden)

        layer_type = [LayerType.encoder, LayerType.retrieval_decoder, LayerType.encoder, LayerType.retrieval_decoder]
        num_layers = len(layer_type)

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        decoder = (
            MegatronRetrievalTransformerDecoderModule(
                config=model_parallel_config,
                init_method=init_method,
                output_layer_init_method=scaled_init_method,
                hidden_size=dim,
                ffn_hidden_size=dim * 4,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                precision=16,
                chunk_size=text_chunk_size,
                layer_type=layer_type,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
            .cuda()
            .half()
        )
        out = decoder(hidden_emb, hidden_mask, retrieved_attn_mask=context_mask, retrieved_emb=retrieved_emb)
        assert out.shape == torch.Size([input_length, batch, dim])

        out_1 = decoder(
            hidden_emb[:, :62],
            hidden_mask[:, :62],
            retrieved_attn_mask=None,
            retrieved_emb=None,
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
        )
        assert (out[:62] - out_1[:62]).abs().max().item() < 1e-2
        out_1 = decoder(
            hidden_emb[:, 62:63],
            hidden_mask[:, :63],
            retrieved_attn_mask=None,
            retrieved_emb=None,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
        )
        assert (out[62] - out_1[0]).abs().max().item() < 1e-2
        out_2 = decoder(
            hidden_emb[:, 63:64],
            hidden_mask[:, :64],
            retrieved_attn_mask=context_mask[:, :1],
            retrieved_emb=retrieved_emb[:, :1],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
        )
        assert (out[63] - out_2[0]).abs().max().item() < 1e-2
        for i in range(64, 127):
            out_2 = decoder(
                hidden_emb[:, i : i + 1],
                hidden_mask[:, : i + 1],
                retrieved_attn_mask=context_mask[:, :1],
                retrieved_emb=retrieved_emb[:, :1],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
            )
            assert (out[i] - out_2[0]).abs().max().item() < 1e-2
        for i in range(127, 191):
            out_3 = decoder(
                hidden_emb[:, i : i + 1],
                hidden_mask[:, : i + 1],
                retrieved_attn_mask=context_mask[:, :2],
                retrieved_emb=retrieved_emb[:, :2],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
            )
            assert (out[i] - out_3[0]).abs().max().item() < 1e-2

        out_1 = decoder(
            hidden_emb[:, :130],
            hidden_mask[:, :130],
            retrieved_attn_mask=context_mask[:, :2],
            retrieved_emb=retrieved_emb[:, :2],
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
        )
        assert (out[:130] - out_1[:130]).abs().max().item() < 1e-2
        for i in range(130, 191):
            out_3 = decoder(
                hidden_emb[:, i : i + 1],
                hidden_mask[:, : i + 1],
                retrieved_attn_mask=context_mask[:, :2],
                retrieved_emb=retrieved_emb[:, :2],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
            )
            assert (out[i] - out_3[0]).abs().max().item() < 1e-2

    @pytest.mark.unit
    def test_encoder_decoder_module_inference(self, model_parallel_config):
        # rotary pos emb dim
        batch = 2
        neighbors = 2
        dim = 128
        pad_id = 19999
        num_attention_heads = 8
        chunks = 32
        text_chunk_size = 64
        input_length = chunks * text_chunk_size
        vocab_size = 20000
        enc_num_layers = 4
        dec_num_layers = 6
        enc_cross_attention = [3]  # layer numbers for cross attention
        dec_cross_attention = [3, 5]  # layer numbers for cross attention

        all_tokens = torch.randint(0, vocab_size, (batch, input_length + 1)).cuda()  # (seq, batch, dim)
        hidden = all_tokens[:, :-1]
        labels = all_tokens[:, 1:]

        hidden_mask = (hidden != pad_id).cuda()
        retrieved = torch.randint(0, vocab_size, (batch, chunks, neighbors, 2 * text_chunk_size)).cuda()

        pad_id = vocab_size - 1
        context_mask = (retrieved != pad_id).cuda()

        class FakeTokenizer:
            eos_id = vocab_size - 2

        tokenizer = FakeTokenizer()

        encoder_decoder = (
            MegatronRetrievalTokenLevelEncoderDecoderModule(
                config=model_parallel_config,
                vocab_size=vocab_size,
                hidden_size=dim,
                max_position_embeddings=input_length,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=dim * 4,
                precision=16,
                chunk_size=text_chunk_size,
                enc_num_layers=enc_num_layers,
                dec_num_layers=dec_num_layers,
                enc_cross_attention=enc_cross_attention,
                dec_cross_attention=dec_cross_attention,
                add_position_embedding=False,
                tokenizer=tokenizer,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
            .cuda()
            .half()
        )

        out = encoder_decoder(hidden, hidden_mask, retrieved_ids=retrieved, retrieved_attn_mask=context_mask)

        out_1 = encoder_decoder(
            hidden[:, :62],
            hidden_mask[:, :62],
            retrieved_attn_mask=None,
            retrieved_ids=None,
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (out[:, :62] - out_1[:, :62]).abs().max().item() < 1e-2

        out_1 = encoder_decoder(
            hidden[:, 62:63],
            hidden_mask[:, :63],
            retrieved_attn_mask=None,
            retrieved_ids=None,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (out[:, 62] - out_1[:, 0]).abs().max().item() < 1e-2

        out_2 = encoder_decoder(
            hidden[:, 63:64],
            hidden_mask[:, :64],
            retrieved_ids=retrieved[:, :1],
            retrieved_attn_mask=context_mask[:, :1],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (out[:, 63] - out_2[:, 0]).abs().max().item() < 1e-2
        for i in range(64, 127):
            out_2 = encoder_decoder(
                hidden[:, i : i + 1],
                hidden_mask[:, : i + 1],
                retrieved_ids=retrieved[:, :1],
                retrieved_attn_mask=context_mask[:, :1],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
            assert (out[:, i] - out_2[:, 0]).abs().max().item() < 1e-2
        for i in range(127, 191):
            out_3 = encoder_decoder(
                hidden[:, i : i + 1],
                hidden_mask[:, : i + 1],
                retrieved_ids=retrieved[:, :2],
                retrieved_attn_mask=context_mask[:, :2],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
            assert (out[:, i] - out_3[:, 0]).abs().max().item() < 1e-2

        out_1 = encoder_decoder(
            hidden[:, :130],
            hidden_mask[:, :130],
            retrieved_ids=retrieved[:, :2],
            retrieved_attn_mask=context_mask[:, :2],
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        assert (out[:, :130] - out_1[:, :130]).abs().max().item() < 1e-2
        for i in range(130, 191):
            out_3 = encoder_decoder(
                hidden[:, i : i + 1],
                hidden_mask[:, : i + 1],
                retrieved_ids=retrieved[:, :2],
                retrieved_attn_mask=context_mask[:, :2],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
            assert (out[:, i] - out_3[:, 0]).abs().max().item() < 1e-2
