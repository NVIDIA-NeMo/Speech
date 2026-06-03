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
import shutil
import tempfile

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding as beam_decode
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.common import tokenizers
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def rnnt_asr_model_with_prompt(test_data_dir):
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    model_defaults = {
        'enc_hidden': 1024,
        'pred_hidden': 640,
        'initialize_prompt_feature': True,  # Enable prompt feature initialization
        'num_prompts': 128,
        'prompt_dictionary': {
            'en_US': 0,
            'es_ES': 1,
            'fr_FR': 2,
            'de_DE': 3,
        },
    }

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                }
            ],
        },
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {
            'pred_hidden': model_defaults['pred_hidden'],
            'pred_rnn_layers': 1,
        },
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {
            'joint_hidden': 640,
            'activation': 'relu',
        },
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    loss = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'tokenizer': DictConfig(tokenizer),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
        }
    )

    model_instance = EncDecRNNTBPEModelWithPrompt(cfg=modelConfig)
    return model_instance


class TestEncDecRNNTBPEModelWithPrompt:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_constructor(self, rnnt_asr_model_with_prompt):
        rnnt_asr_model_with_prompt.train()
        # Check to/from config_dict:
        confdict = rnnt_asr_model_with_prompt.to_config_dict()
        instance2 = EncDecRNNTBPEModelWithPrompt.from_config_dict(confdict)
        assert isinstance(instance2, EncDecRNNTBPEModelWithPrompt)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward(self, rnnt_asr_model_with_prompt):
        rnnt_asr_model_with_prompt = rnnt_asr_model_with_prompt.eval()

        rnnt_asr_model_with_prompt.preprocessor.featurizer.dither = 0.0
        rnnt_asr_model_with_prompt.preprocessor.featurizer.pad_to = 0

        rnnt_asr_model_with_prompt.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=321, high=500, size=[4])

        # 1D prompt indices: one language id per sample in the batch.
        prompt_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _ = rnnt_asr_model_with_prompt.forward(
                    input_signal=input_signal[i : i + 1],
                    input_signal_length=length[i : i + 1],
                    prompt_indices=prompt_indices[i : i + 1],
                )
                logprobs_instance.append(logprobs_ins)
            logits_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _ = rnnt_asr_model_with_prompt.forward(
                input_signal=input_signal, input_signal_length=length, prompt_indices=prompt_indices
            )

        assert logits_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_predict_step(self, rnnt_asr_model_with_prompt):
        rnnt_asr_model_with_prompt = rnnt_asr_model_with_prompt.eval()

        # Create a simple batch manually
        batch_size = 1
        seq_len = 1600

        # Create mock batch data
        audio_signal = torch.randn(batch_size, seq_len)
        audio_lengths = torch.tensor([seq_len])
        transcript = torch.randint(0, 10, (batch_size, 10))
        transcript_lengths = torch.tensor([10])
        prompt_indices = torch.tensor([0], dtype=torch.long)  # language id 0

        batch = (audio_signal, audio_lengths, transcript, transcript_lengths, prompt_indices)

        outputs = rnnt_asr_model_with_prompt.predict_step(batch, 0)
        assert len(outputs) == 1
        # predict_step returns list of (sample_id, hyp_or_text) pairs
        assert len(outputs[0]) == 2

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_save_restore_artifact(self, rnnt_asr_model_with_prompt):
        rnnt_asr_model_with_prompt.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'rnnt_bpe_prompt.nemo')
            rnnt_asr_model_with_prompt.save_to(path)

            # restore_from is intentionally delegated to the parent EncDecRNNTBPEModel
            # to avoid subclass-substitution that would hang on missing prompt_dictionary.
            new_model = EncDecRNNTBPEModelWithPrompt.restore_from(path)
            assert new_model.vocab_path.endswith('_vocab.txt')

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_save_restore_artifact_spe(self, rnnt_asr_model_with_prompt, test_data_dir):
        rnnt_asr_model_with_prompt.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
            rnnt_asr_model_with_prompt.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type='bpe')

            save_path = os.path.join(tmpdir, 'rnnt_bpe_prompt.nemo')
            rnnt_asr_model_with_prompt.train()
            rnnt_asr_model_with_prompt.save_to(save_path)

            new_model = EncDecRNNTBPEModelWithPrompt.restore_from(save_path)
            assert isinstance(new_model.tokenizer, tokenizers.SentencePieceTokenizer)
            assert new_model.model_path.endswith('_tokenizer.model')
            assert new_model.vocab_path.endswith('_vocab.txt')
            assert new_model.spe_vocab_path.endswith('_tokenizer.vocab')

    @pytest.mark.unit
    def test_save_restore_artifact_agg(self, rnnt_asr_model_with_prompt, test_data_dir):
        tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
        tok_en = {"dir": tokenizer_dir, "type": "wpe"}
        # the below is really an english tokenizer but we pretend it is spanish
        tok_es = {"dir": tokenizer_dir, "type": "wpe"}
        tcfg = DictConfig({"type": "agg", "langs": {"en": tok_en, "es": tok_es}})
        with tempfile.TemporaryDirectory() as tmpdir:
            rnnt_asr_model_with_prompt.change_vocabulary(new_tokenizer_dir=tcfg, new_tokenizer_type="agg")

            save_path = os.path.join(tmpdir, "rnnt_agg_prompt.nemo")
            rnnt_asr_model_with_prompt.train()
            rnnt_asr_model_with_prompt.save_to(save_path)

            new_model = EncDecRNNTBPEModelWithPrompt.restore_from(save_path)
            assert isinstance(new_model.tokenizer, tokenizers.AggregateTokenizer)

            # Both source tokenizers are the same 132-token vocab; the AggregateTokenizer
            # deduplicates 10 shared control tokens, so total = 132 + (132 - 10) = 254.
            assert new_model.tokenizer.tokenizer.vocab_size == 264
            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 264

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_vocab_change(self, test_data_dir, rnnt_asr_model_with_prompt):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
            new_tokenizer_dir = os.path.join(tmpdir, 'tokenizer')

            os.makedirs(new_tokenizer_dir, exist_ok=True)
            shutil.copy2(old_tokenizer_dir, new_tokenizer_dir)

            nw1 = rnnt_asr_model_with_prompt.num_weights
            rnnt_asr_model_with_prompt.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
            # No change
            assert nw1 == rnnt_asr_model_with_prompt.num_weights

            with open(os.path.join(new_tokenizer_dir, 'vocab.txt'), 'a+') as f:
                f.write("!\n")
                f.write('$\n')
                f.write('@\n')

            rnnt_asr_model_with_prompt.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')

            # rnn embedding + joint + bias (no CTC decoder in RNNT-only model)
            pred_embedding = 3 * (rnnt_asr_model_with_prompt.decoder.pred_hidden)
            joint_joint = 3 * (rnnt_asr_model_with_prompt.joint.joint_hidden + 1)
            assert rnnt_asr_model_with_prompt.num_weights == (nw1 + (pred_embedding + joint_joint))

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_decoding_change(self, rnnt_asr_model_with_prompt):
        assert isinstance(rnnt_asr_model_with_prompt.decoding.decoding, greedy_decode.GreedyBatchedRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'greedy'
        new_strategy.greedy = DictConfig({'max_symbols': 10})
        rnnt_asr_model_with_prompt.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(rnnt_asr_model_with_prompt.decoding.decoding, greedy_decode.GreedyRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 1})
        rnnt_asr_model_with_prompt.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(rnnt_asr_model_with_prompt.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert rnnt_asr_model_with_prompt.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 2})
        rnnt_asr_model_with_prompt.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(rnnt_asr_model_with_prompt.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert rnnt_asr_model_with_prompt.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'tsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        rnnt_asr_model_with_prompt.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(rnnt_asr_model_with_prompt.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert rnnt_asr_model_with_prompt.decoding.decoding.search_type == "tsd"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'alsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        rnnt_asr_model_with_prompt.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(rnnt_asr_model_with_prompt.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert rnnt_asr_model_with_prompt.decoding.decoding.search_type == "alsd"

    @pytest.mark.unit
    def test_input_output_types_with_prompt(self, rnnt_asr_model_with_prompt):
        """Test that input/output types include prompt-specific types."""
        input_types = rnnt_asr_model_with_prompt.input_types
        output_types = rnnt_asr_model_with_prompt.output_types

        # Check that prompt_indices is included in input types (1D, per-sample language id)
        assert 'prompt_indices' in input_types
        prompt_axes = input_types['prompt_indices'].axes
        assert len(prompt_axes) == 1  # 1D tensor [B]

        # Check standard input types are present
        assert 'input_signal' in input_types
        assert 'input_signal_length' in input_types

        # Check output types
        assert 'outputs' in output_types
        assert 'encoded_lengths' in output_types

    @pytest.mark.unit
    def test_prompt_feature_initialization(self, rnnt_asr_model_with_prompt):
        """Test that prompt feature initialization works correctly."""
        # Test that the model has prompt-related attributes
        assert hasattr(rnnt_asr_model_with_prompt, 'concat')
        assert hasattr(rnnt_asr_model_with_prompt, 'num_prompts')
        assert hasattr(rnnt_asr_model_with_prompt, 'prompt_kernel')

        # Test that concat is enabled
        assert rnnt_asr_model_with_prompt.concat == True

        # Test prompt kernel dimensions
        expected_input_size = (
            rnnt_asr_model_with_prompt.num_prompts + rnnt_asr_model_with_prompt._cfg.model_defaults.enc_hidden
        )
        expected_output_size = rnnt_asr_model_with_prompt._cfg.model_defaults.enc_hidden

        # Check first layer of prompt kernel
        first_layer = rnnt_asr_model_with_prompt.prompt_kernel[0]
        assert first_layer.in_features == expected_input_size
        assert first_layer.out_features == expected_output_size * 2

    @pytest.mark.unit
    def test_set_inference_prompt(self, rnnt_asr_model_with_prompt):
        """Test that set_inference_prompt accepts known languages and rejects unknown ones."""
        # Known language from the prompt_dictionary fixture
        rnnt_asr_model_with_prompt.set_inference_prompt('en_US')
        assert rnnt_asr_model_with_prompt._inference_prompt_index == 0

        rnnt_asr_model_with_prompt.set_inference_prompt('de_DE')
        assert rnnt_asr_model_with_prompt._inference_prompt_index == 3

        # Unknown language should raise
        with pytest.raises(ValueError):
            rnnt_asr_model_with_prompt.set_inference_prompt('zz_ZZ')

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward_different_prompts_produce_different_outputs(self, rnnt_asr_model_with_prompt):
        """Different prompt_indices should yield different encoded outputs (prompt actually conditions)."""
        rnnt_asr_model_with_prompt = rnnt_asr_model_with_prompt.eval()
        rnnt_asr_model_with_prompt.preprocessor.featurizer.dither = 0.0
        rnnt_asr_model_with_prompt.preprocessor.featurizer.pad_to = 0

        input_signal = torch.randn(size=(2, 512))
        length = torch.tensor([512, 512])

        with torch.no_grad():
            out_a, _ = rnnt_asr_model_with_prompt.forward(
                input_signal=input_signal,
                input_signal_length=length,
                prompt_indices=torch.tensor([0, 0], dtype=torch.long),
            )
            out_b, _ = rnnt_asr_model_with_prompt.forward(
                input_signal=input_signal,
                input_signal_length=length,
                prompt_indices=torch.tensor([1, 1], dtype=torch.long),
            )

        assert out_a.shape == out_b.shape
        # The prompt projection should make outputs differ for different language ids.
        assert torch.max(torch.abs(out_a - out_b)) > 0
