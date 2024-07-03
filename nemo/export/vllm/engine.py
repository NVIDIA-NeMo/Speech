# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from pathlib import Path

from vllm import LLMEngine
from vllm.transformers_utils.tokenizer_group.tokenizer_group import TokenizerGroup

from nemo.export.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.export.tarutils import TarPath
from nemo.export.vllm.tokenizer_group import NemoTokenizerGroup

LOGGER = logging.getLogger("NeMo")


class NemoLLMEngine(LLMEngine):
    """
    Overrides some functionality from vllm.LLMEngine to use our custom tokenizer
    instead of one from Transformers.
    """

    def _init_tokenizer(self, **tokenizer_init_kwargs):
        # Find the tokenizer file name in the Nemo checkpoint config
        tokenizer_config = self.model_config.nemo_model_config.get('tokenizer', {})
        tokenizer_model = tokenizer_config.get('model', tokenizer_config.get('tokenizer_model', None))

        # If there is no tokenizer file specified but there's a reference to an HF tokenizer, use that
        if tokenizer_model is None and tokenizer_config.get('library') == 'huggingface':
            tokenizer_type = tokenizer_config.get('type')
            if tokenizer_type is not None:
                tokenizer_group = TokenizerGroup(
                    tokenizer_id=tokenizer_type,
                    enable_lora=bool(self.lora_config),
                    max_num_seqs=self.scheduler_config.max_num_seqs,
                    max_input_length=None,
                )

                # Update the HF config fields that come from the tokenizer in NeMo
                self.model_config.hf_config.vocab_size = len(
                    tokenizer_group.tokenizer.vocab
                )  # this may be greater than vocab_size
                self.model_config.hf_config.bos_token_id = tokenizer_group.tokenizer.bos_token_id
                self.model_config.hf_config.eos_token_id = tokenizer_group.tokenizer.eos_token_id
                self.model_config.hf_config.pad_token_id = tokenizer_group.tokenizer.pad_token_id

                return tokenizer_group

        # Open the checkpoint archive
        with TarPath(self.model_config.nemo_checkpoint) as archive:
            tokenizer_model_file = None
            if isinstance(tokenizer_model, str) and tokenizer_model.startswith('nemo:'):
                tokenizer_model = tokenizer_model[len('nemo:') :]
                tokenizer_model_file = archive / tokenizer_model
                if not tokenizer_model_file.exists():
                    LOGGER.warn(
                        f'Tokenizer model file {tokenizer_model} specified in the model_config does not '
                        + 'exist in the checkpoint.'
                    )
                    tokenizer_model_file = None

            if tokenizer_model_file is None:
                for path in archive.glob('*tokenizer*.model'):
                    LOGGER.info(f'Found tokenizer model file {path}.')
                    tokenizer_model_file = path
                    break

            if tokenizer_model_file is None:
                raise RuntimeError('No tokenizer model file found, aborting.')

            # Extract the tokenizer model file into the model directory,
            # because sentencepiece cannot load it directly from TarPath.
            extracted_tokenizer_model = Path(self.model_config.model) / 'tokenizer.model'
            with tokenizer_model_file.open('rb') as infile:
                with extracted_tokenizer_model.open('wb') as outfile:
                    outfile.write(infile.read())

            # Construct the tokenizer object and wrapper
            tokenizer = SentencePieceTokenizer(str(extracted_tokenizer_model))

            # Determine if the model needs a bos token (which is not stored in Nemo checkpoints)
            add_bos_token = self.model_config.model_converter.requires_bos_token()

            tokenizer_group = NemoTokenizerGroup(tokenizer, add_bos_token=add_bos_token)

            # Update the HF config fields that come from the tokenizer in NeMo
            self.model_config.hf_config.vocab_size = tokenizer.vocab_size
            self.model_config.hf_config.bos_token_id = tokenizer.bos_token_id
            self.model_config.hf_config.eos_token_id = tokenizer.eos_token_id
            self.model_config.hf_config.pad_token_id = tokenizer.pad_id

            return tokenizer_group
