# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.parts.submodules import multitask_beam_decoding as beam_decode
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.prompts.canary import CanaryPromptFormatter
from nemo.collections.common.tokenizers import CanaryTokenizer


@pytest.fixture()
def asr_model(test_data_dir):
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    model_defaults = {'asr_enc_hidden': 128, 'lm_enc_hidden': 64, 'lm_dec_hidden': 64}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConformerEncoder',
        'params': {
            'feat_in': 64,
            'n_layers': 1,
            'd_model': model_defaults['asr_enc_hidden'],
            'subsampling': 'dw_striding',
            'subsampling_factor': 2,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 4,
            'conv_kernel_size': 9,
        },
    }

    transf_decoder = {
        '_target_': 'nemo.collections.asr.modules.transformer.get_nemo_transformer',
        'model_name': None,
        'pretrained': False,
        'encoder': None,
        'pre_ln_final_layer_norm': True,
        'config_dict': {
            'max_sequence_length': 512,
            'num_token_types': 0,
            'hidden_size': model_defaults['lm_dec_hidden'],
            'inner_size': 4 * model_defaults['lm_dec_hidden'],
            'num_layers': 1,
            'num_attention_heads': 2,
            'pre_ln': True,
            'vocab_size': None,
        },
    }

    head = {
        '_target_': 'nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier',
        'num_layers': 1,
        'activation': 'relu',
        'log_softmax': True,
        'hidden_size': model_defaults['lm_dec_hidden'],
        'num_classes': None,
    }

    decoding = {'strategy': 'beam', 'beam': {'beam_size': 1}}

    # os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128")
    tokenizer = {
        'dir': None,
        'type': 'agg',
        'langs': {
            'spl_tokens': {
                'dir': os.path.join(test_data_dir, "asr", "tokenizers", "canary"),
                'type': 'bpe',
            },
            'en': {
                'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"),
                'type': 'wpe',
            },
            'de': {
                'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"),
                'type': 'wpe',
            },
        },
        'custom_tokenizer': {
            '_target_': 'nemo.collections.common.tokenizers.canary_tokenizer.CanaryTokenizer',
            'tokenizers': None,
        },
    }

    loss = {
        '_target_': 'nemo.collections.common.losses.smoothed_cross_entropy.SmoothedCrossEntropyLoss',
        'label_smoothing': 0.0,
    }

    modelConfig = DictConfig(
        {
            'prompt_format': 'canary',
            'prompt_defaults': [
                {"role": "user", "slots": {"source_lang": "en", "target_lang": "en", "task": "asr", "pnc": "yes"}}
            ],
            'sample_rate': 16000,
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'transf_decoder': DictConfig(transf_decoder),
            'head': DictConfig(head),
            'tokenizer': DictConfig(tokenizer),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
        }
    )

    model_instance = EncDecMultiTaskModel(cfg=modelConfig)
    return model_instance


class TestEncDecMultiTaskModel:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecMultiTaskModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecMultiTaskModel)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self, asr_model):
        torch.manual_seed(0)
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        asr_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        targets = torch.randint(low=0, high=100, size=[4, 10])
        targets_len = torch.randint(low=1, high=10, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                log_probs, _, _, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1],
                    input_signal_length=length[i : i + 1],
                    transcript=targets[i : i + 1],
                    transcript_length=targets_len[i : i + 1],
                )
                print(log_probs.shape)
                logprobs_instance.append(log_probs)
            logits_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _, _, _ = asr_model.forward(
                input_signal=input_signal,
                input_signal_length=length,
                transcript=targets,
                transcript_length=targets_len,
            )

        assert logits_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-5
        diff = torch.max(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-5

    @pytest.mark.unit
    def test_save_restore_artifact(self, asr_model):
        asr_model.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'aed_bpe.nemo')
            asr_model.save_to(path)

            new_model = EncDecMultiTaskModel.restore_from(path)
            assert isinstance(new_model, type(asr_model))

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 32 + 128 + 128

    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_save_restore_artifact_change_vocab(self, asr_model, test_data_dir):
    #     asr_model.train()
    #
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
    #         asr_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type='bpe')
    #
    #         save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
    #         asr_model.train()
    #         asr_model.save_to(save_path)
    #
    #         new_model = EncDecMultiTaskModel.restore_from(save_path)
    #         assert isinstance(new_model, type(asr_model))
    #         assert isinstance(new_model.tokenizer, tokenizers.SentencePieceTokenizer)
    #         assert new_model.model_path.endswith('_tokenizer.model')
    #         assert new_model.vocab_path.endswith('_vocab.txt')
    #         assert new_model.spe_vocab_path.endswith('_tokenizer.vocab')

    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_save_restore_artifact_agg(self, asr_model, test_data_dir):
    #     tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
    #     tok_en = {"dir": tokenizer_dir, "type": "wpe"}
    #     # the below is really an english tokenizer but we pretend it is spanish
    #     tok_es = {"dir": tokenizer_dir, "type": "wpe"}
    #     tcfg = DictConfig({"type": "agg", "langs": {"en": tok_en, "es": tok_es}})
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         asr_model.change_vocabulary(new_tokenizer_dir=tcfg, new_tokenizer_type="agg")
    #
    #         save_path = os.path.join(tmpdir, "ctc_agg.nemo")
    #         asr_model.train()
    #         asr_model.save_to(save_path)
    #
    #         new_model = EncDecMultiTaskModel.restore_from(save_path)
    #         assert isinstance(new_model, type(asr_model))
    #         assert isinstance(new_model.tokenizer, tokenizers.AggregateTokenizer)
    #
    #         # should be double
    #         assert new_model.tokenizer.tokenizer.vocab_size == 254
    #         assert len(new_model.tokenizer.tokenizer.get_vocab()) == 254

    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_vocab_change(self, test_data_dir, asr_model):
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         old_tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
    #         new_tokenizer_dir = os.path.join(tmpdir, 'tokenizer')
    #
    #         os.makedirs(new_tokenizer_dir, exist_ok=True)
    #         shutil.copy2(old_tokenizer_dir, new_tokenizer_dir)
    #
    #         nw1 = asr_model.num_weights
    #         asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
    #         # No change
    #         assert nw1 == asr_model.num_weights
    #
    #         with open(os.path.join(new_tokenizer_dir, 'vocab.txt'), 'a+') as f:
    #             f.write("!\n")
    #             f.write('$\n')
    #             f.write('@\n')
    #
    #         asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
    #
    #         # rnn embedding + joint + bias
    #         pred_embedding = 3 * (asr_model.decoder.pred_hidden)
    #         joint_joint = 3 * (asr_model.joint.joint_hidden + 1)
    #         assert asr_model.num_weights == (nw1 + (pred_embedding + joint_joint))

    @pytest.mark.unit
    def test_decoding_change(self, asr_model):
        assert isinstance(asr_model.decoding.decoding, beam_decode.TransformerAEDBeamInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 5})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.TransformerAEDBeamInfer)
        assert asr_model.decoding.decoding.search_type == "default"

    @pytest.mark.unit
    def test_prompt_change(self, asr_model):
        assert asr_model.prompt_format == 'canary'
        assert isinstance(asr_model.prompt, CanaryPromptFormatter)

        # Default change prompt
        asr_model.change_prompt()
        assert asr_model.cfg.prompt_defaults is None

        prompt_defaults = asr_model.prompt.get_default_dialog_slots()
        prompt_defaults[0]['slots']['pnc'] = 'no'
        asr_model.change_prompt(prompt_defaults=prompt_defaults)

        assert asr_model.cfg.prompt_defaults[0]['slots']['pnc'] == 'no'

    @pytest.mark.unit
    def test_prompt_change_subclass(self, asr_model):
        assert asr_model.prompt_format == 'canary'
        assert isinstance(asr_model.prompt, CanaryPromptFormatter)

        class CanaryPromptFormatterSubclass(CanaryPromptFormatter):
            NAME = "canary2"

        # Default change prompt
        asr_model.change_prompt()
        assert asr_model.cfg.prompt_defaults is None

        prompt_defaults = asr_model.prompt.get_default_dialog_slots()
        prompt_defaults[0]['slots']['pnc'] = 'no'
        asr_model.change_prompt(prompt_format='canary2', prompt_defaults=prompt_defaults)

        assert asr_model.cfg.prompt_format == 'canary2'
        assert asr_model.cfg.prompt_defaults[0]['slots']['pnc'] == 'no'
        assert isinstance(asr_model.prompt, CanaryPromptFormatterSubclass)

        user_prompt = asr_model.prompt.get_default_dialog_slots()[0]
        slots = user_prompt['slots']
        slots['source_lang'] = 'en'
        slots['target_lang'] = 'en'
        slots['task'] = 'asr'
        slots['pnc'] = 'no'
        ans = asr_model.prompt.encode_dialog([user_prompt])
        recovered = asr_model.tokenizer.ids_to_text(ans["input_ids"])
        assert recovered == "<|startoftranscript|><|en|><|transcribe|><|en|><|nopnc|>"

    @pytest.mark.unit
    def test_transcribe_single_file(self, asr_model, test_data_dir):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = asr_model.transcribe(audio_file, batch_size=1)
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    @pytest.mark.unit
    def test_transcribe_single_file_translation(self, asr_model, test_data_dir):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = asr_model.transcribe(audio_file, batch_size=1, task="ast", source_lang='en', target_lang='de')
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    @pytest.mark.unit
    def test_transcribe_return_hypothesis(self, asr_model, test_data_dir):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = asr_model.transcribe(audio_file, batch_size=1, return_hypotheses=True)
        assert len(outputs) == 1
        assert isinstance(outputs[0], Hypothesis)

        hyp = outputs[0]
        assert isinstance(hyp.text, str)
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.alignments is None

    @pytest.mark.unit
    def test_transcribe_tensor(self, asr_model, test_data_dir):
        # Load audio file
        import soundfile as sf

        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
        audio, sr = sf.read(audio_file, dtype='float32')

        # Numpy array test
        outputs = asr_model.transcribe(audio, batch_size=1)
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    @pytest.mark.unit
    def test_build_tokenizer(self, asr_model, test_data_dir):
        # Load audio file
        task_tokens = ["ast", "asr"]
        lang_tokens = ["en", "es", "de", "fr"]
        tokens = task_tokens + lang_tokens
        spl_tokenizer_from_build = CanaryTokenizer.build_special_tokenizer(tokens, test_data_dir)

        tokenizer_cfg = {'dir': os.path.join(test_data_dir), 'type': 'bpe'}
        spl_tokenizer_from_load = asr_model._make_tokenizer(tokenizer_cfg, "spl_tokens")[0]

        tokens += ["<|nospeech|>", "<pad>", "<|endoftext|>", "<|startoftranscript|>", "<|pnc|>", "<|nopnc|>"]

        ids1 = [spl_tokenizer_from_build.tokens_to_ids(t)[0] for t in tokens]
        ids2 = [spl_tokenizer_from_load.tokens_to_ids(t)[0] for t in tokens]

        for i, j in zip(ids1, ids2):
            assert i == j
