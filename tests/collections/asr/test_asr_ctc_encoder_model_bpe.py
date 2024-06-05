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

import copy
import os
import shutil
import tempfile

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.models import configs
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.common import tokenizers
from nemo.utils.config_utils import assert_dataclass_signature_match


@pytest.fixture()
def asr_model(test_data_dir):
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 1024,
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
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 1024,
        'num_classes': -1,
        'vocabulary': None,
    }

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'tokenizer': DictConfig(tokenizer),
        }
    )

    model_instance = EncDecCTCModelBPE(cfg=modelConfig)
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecCTCModelBPE.from_config_dict(confdict)
        assert isinstance(instance2, EncDecCTCModelBPE)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self, asr_model):
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
                print(len(logprobs_ins))
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_save_restore_artifact(self, asr_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
            asr_model.train()
            asr_model.save_to(save_path)

            new_model = EncDecCTCModelBPE.restore_from(save_path)
            assert isinstance(new_model, type(asr_model))
            assert new_model.vocab_path.endswith('_vocab.txt')

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_save_restore_artifact_spe(self, asr_model, test_data_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
            asr_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type='bpe')

            save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
            asr_model.train()
            asr_model.save_to(save_path)

            new_model = EncDecCTCModelBPE.restore_from(save_path)
            assert isinstance(new_model, type(asr_model))
            assert isinstance(new_model.tokenizer, tokenizers.SentencePieceTokenizer)
            assert new_model.model_path.endswith('_tokenizer.model')
            assert new_model.vocab_path.endswith('_vocab.txt')
            assert new_model.spe_vocab_path.endswith('_tokenizer.vocab')

            assert new_model.tokenizer.tokenizer.vocab_size == 128
            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_save_restore_artifact_agg(self, asr_model, test_data_dir):
        tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
        tok_en = {"dir": tokenizer_dir, "type": "wpe"}
        # the below is really an english tokenizer but we pretend it is spanish
        tok_es = {"dir": tokenizer_dir, "type": "wpe"}
        tcfg = DictConfig({"type": "agg", "langs": {"en": tok_en, "es": tok_es}})
        with tempfile.TemporaryDirectory() as tmpdir:
            asr_model.change_vocabulary(new_tokenizer_dir=tcfg, new_tokenizer_type="agg")

            save_path = os.path.join(tmpdir, "ctc_agg.nemo")
            asr_model.train()
            asr_model.save_to(save_path)

            new_model = EncDecCTCModelBPE.restore_from(save_path)
            assert isinstance(new_model, type(asr_model))
            assert isinstance(new_model.tokenizer, tokenizers.AggregateTokenizer)

            # should be double
            assert new_model.tokenizer.tokenizer.vocab_size == 254
            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 254

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_vocab_change(self, test_data_dir, asr_model):
        old_vocab = copy.deepcopy(asr_model.decoder.vocabulary)

        with tempfile.TemporaryDirectory() as save_dir:
            save_path = os.path.join(save_dir, 'temp.nemo')

            with tempfile.TemporaryDirectory() as tmpdir:
                old_tmpdir_path = tmpdir

                old_tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
                new_tokenizer_dir = os.path.join(tmpdir, 'tokenizer')

                os.makedirs(new_tokenizer_dir, exist_ok=True)
                shutil.copy2(old_tokenizer_dir, new_tokenizer_dir)

                nw1 = asr_model.num_weights
                asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
                # No change
                assert nw1 == asr_model.num_weights

                with open(os.path.join(new_tokenizer_dir, 'vocab.txt'), 'a+') as f:
                    f.write("!\n")
                    f.write('$\n')
                    f.write('@\n')

                asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
                # fully connected + bias
                assert asr_model.num_weights == nw1 + 3 * (asr_model.decoder._feat_in + 1)

                new_vocab = copy.deepcopy(asr_model.decoder.vocabulary)
                assert len(old_vocab) != len(new_vocab)

                # save the model (after change of vocabulary)
                asr_model.save_to(save_path)
                assert os.path.exists(save_path)
                # delete copied version of the vocabulary from nested tmpdir (by scope)

            # assert copied vocab no longer exists
            assert not os.path.exists(os.path.join(old_tmpdir_path, 'tokenizer', 'vocab.txt'))

            # make a copy of the tokenizer before renaming
            try:
                os.rename(old_tokenizer_dir, old_tokenizer_dir + '.bkp')
                assert not os.path.exists(old_tokenizer_dir)

                # restore model from .nemo
                asr_model2 = EncDecCTCModelBPE.restore_from(save_path)
                assert isinstance(asr_model2, EncDecCTCModelBPE)

                # Check if vocabulary size is same
                assert asr_model.tokenizer.tokenizer.vocab_size == asr_model2.tokenizer.tokenizer.vocab_size

                # Make a copy of the tokenizer
                new_tokenizer_dir = os.path.join(save_dir, 'tokenizer')

                os.makedirs(new_tokenizer_dir, exist_ok=True)
                new_tokenizer_path = os.path.join(new_tokenizer_dir, 'vocab.txt')
                with open(new_tokenizer_path, 'w') as f:
                    for v in asr_model2.tokenizer.tokenizer.get_vocab():
                        f.write(f"{v}\n")

                    # Add some new tokens too
                    f.write("^\n")
                    f.write("^^\n")
                    f.write("^^^\n")

                assert os.path.exists(new_tokenizer_path)

                # change vocabulary
                asr_model2.change_vocabulary(new_tokenizer_dir, new_tokenizer_type='wpe')
                assert asr_model.tokenizer.vocab_size != asr_model2.tokenizer.vocab_size

                new_save_path = os.path.join(save_dir, 'temp2.nemo')
                asr_model2.save_to(new_save_path)

                asr_model3 = EncDecCTCModelBPE.restore_from(new_save_path)
                assert isinstance(asr_model3, EncDecCTCModelBPE)

                # Check if vocabulary size is same
                assert asr_model2.tokenizer.tokenizer.vocab_size == asr_model3.tokenizer.tokenizer.vocab_size
                assert asr_model2.vocab_path != asr_model3.vocab_path

                # Model PT level checks
                assert len(asr_model2.artifacts) == 1

            finally:
                os.rename(old_tokenizer_dir + '.bkp', old_tokenizer_dir)

    @pytest.mark.unit
    def test_decoding_change(self, asr_model):
        assert asr_model.decoding is not None
        assert isinstance(asr_model.decoding, CTCBPEDecoding)
        assert asr_model.decoding.cfg.strategy == "greedy_batch"
        assert asr_model.decoding.preserve_alignments is False
        assert asr_model.decoding.compute_timestamps is False

        cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        asr_model.change_decoding_strategy(cfg)

        assert asr_model.decoding.preserve_alignments is True
        assert asr_model.decoding.compute_timestamps is True

    @pytest.mark.unit
    def test_ASRDatasetConfig_for_AudioToBPEDataset(self):
        # ignore some additional arguments as dataclass is generic
        IGNORE_ARGS = [
            'is_tarred',
            'num_workers',
            'batch_size',
            'tarred_audio_filepaths',
            'shuffle',
            'pin_memory',
            'drop_last',
            'tarred_shard_strategy',
            'shard_manifests',
            'shuffle_n',
            'parser',
            'normalize',
            'unk_index',
            'pad_id',
            'bos_id',
            'eos_id',
            'blank_index',
            'bucketing_batch_size',
            'bucketing_strategy',
            'bucketing_weights',
            'channel_selector',
        ]

        REMAP_ARGS = {'trim_silence': 'trim', 'labels': 'tokenizer'}

        result = assert_dataclass_signature_match(
            audio_to_text.AudioToBPEDataset,
            configs.ASRDatasetConfig,
            ignore_args=IGNORE_ARGS,
            remap_args=REMAP_ARGS,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_ASRDatasetConfig_for_TarredAudioToBPEDataset(self):
        # ignore some additional arguments as dataclass is generic
        IGNORE_ARGS = [
            'is_tarred',
            'num_workers',
            'batch_size',
            'shuffle',
            'pin_memory',
            'drop_last',
            'parser',
            'normalize',
            'unk_index',
            'pad_id',
            'bos_id',
            'eos_id',
            'blank_index',
            'global_rank',
            'world_size',
            'bucketing_batch_size',
            'bucketing_strategy',
            'bucketing_weights',
            'max_utts',
        ]

        REMAP_ARGS = {
            'trim_silence': 'trim',
            'tarred_audio_filepaths': 'audio_tar_filepaths',
            'tarred_shard_strategy': 'shard_strategy',
            'shuffle_n': 'shuffle',
            'labels': 'tokenizer',
        }

        result = assert_dataclass_signature_match(
            audio_to_text.TarredAudioToBPEDataset,
            configs.ASRDatasetConfig,
            ignore_args=IGNORE_ARGS,
            remap_args=REMAP_ARGS,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None
