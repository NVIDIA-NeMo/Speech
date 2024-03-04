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


import glob
import json
import os
import pickle
import tarfile
import tempfile

from joblib import Parallel, delayed
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model
from nemo.collections.nlp.data.language_modeling.sentence_dataset import SentenceDataset
from nemo.collections.nlp.data.machine_translation.machine_translation_dataset import TranslationDataset
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer, get_tokenizer
from nemo.utils import logging


class MTDataPreproc:
    """ Automatically trains tokenizers and preprocesses machine translation data based on the MTEncDecModelConfig.
        For training NMT models with datasets larger than 5M sentence pairs, 
        it can be inefficient to train them without first creating a tarred dataset. 
        If the user wants to change the tokenizer, vocab size, or batch size, for example, 
        they must reprocess the data with the correct configuration. 
        With MTDataPreproc users can sweep through data configurations and the tarred dataset will 
        be automatically created according to the model configuration.
        To train tokenizer model and create tarred dataset specify in configuration:
            model.preproc_out_dir=/path/to/preproc_out
            model.encoder_tokenizer.vocab_size=32000
            model.decoder_tokenizer.vocab_size=32000 
            model.train_ds.use_tarred_dataset=True 
            model.train_ds.src_file_name=/path/to/src.txt
            model.train_ds.tgt_file_name=/path/to/tgt.txt
            model.train_ds.tokens_in_batch=16000 
        Once a dataset has been constructed based on this configuration, MTDataPreproc will not process it again.
        If a previously trained tokenizer model or tarred dataset is found, MTDataPreproc will not preprocess the data.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None) -> None:
        self._cfg = cfg
        self.global_rank = 0
        self.world_size = 1
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_devices) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_devices

        if hasattr(cfg, 'train_ds'):
            supported_tokenizers = ['huggingface', 'sentencepiece', 'megatron', 'byte-level']
            supported_multilingual_tokenizers = ['sentencepiece', 'byte-level']
            supported_train_tokenizers = ['sentencepiece']

            if (
                cfg.encoder_tokenizer.get('library') not in supported_tokenizers
                or cfg.decoder_tokenizer.get('library') not in supported_tokenizers
            ):
                raise NotImplementedError(f"Currently we only support {supported_tokenizers}.")

            if cfg.get('multilingual') and (
                cfg.encoder_tokenizer.get('library') not in supported_multilingual_tokenizers
                or cfg.decoder_tokenizer.get('library') not in supported_multilingual_tokenizers
            ):
                raise NotImplementedError(
                    f"Currently we only support {supported_multilingual_tokenizers} for multilingual models."
                )

            if cfg.get('shared_tokenizer') and cfg.encoder_tokenizer.get('library') != cfg.decoder_tokenizer.get(
                'library'
            ):
                raise ValueError("Shared tokenizers cannot be from different libraries.")

            # Prepare tokenizers
            if (
                cfg.encoder_tokenizer.get('library') in supported_train_tokenizers
                or cfg.decoder_tokenizer.get('library') in supported_train_tokenizers
            ):

                # Train tokenizer models if using sentencepiece and they don't exist
                if (
                    cfg.encoder_tokenizer.get('library') in supported_train_tokenizers
                    and cfg.encoder_tokenizer.get('tokenizer_model') is None
                ) or (
                    cfg.decoder_tokenizer.get('library') in supported_train_tokenizers
                    and cfg.decoder_tokenizer.get('tokenizer_model') is None
                ):
                    if cfg.get('preproc_out_dir') is None:
                        raise ValueError('Tokenizer model training required but cfg.preproc_out_dir is None.')
                    if cfg.train_ds.get('src_file_name') is None or cfg.train_ds.get('tgt_file_name') is None:
                        raise ValueError(
                            'src_file_name and tgt_file_name needed to train tokenizers but could not be found.'
                        )

                    src_fname = cfg.train_ds.get('src_file_name')
                    tgt_fname = cfg.train_ds.get('tgt_file_name')
                    src_language = cfg.get('src_language')
                    tgt_language = cfg.get('tgt_language')
                    spt_symbols = None
                    tempdir = tempfile.TemporaryDirectory()

                    if cfg.get('multilingual'):
                        spt_symbols = []
                        if isinstance(src_fname, ListConfig):
                            fnames = (" ").join(src_fname)
                            src_fname = os.path.join(tempdir.name, 'src.txt')
                            os.system('cat %s > %s' % (fnames, src_fname))

                        if isinstance(tgt_fname, ListConfig):
                            fnames = (" ").join(tgt_fname)
                            tgt_fname = os.path.join(tempdir.name, 'tgt.txt')
                            os.system('cat %s > %s' % (fnames, tgt_fname))

                        if isinstance(src_language, ListConfig):
                            for lng in src_language:
                                spt_symbols.append("<" + lng + ">")

                        if isinstance(tgt_language, ListConfig):
                            for lng in tgt_language:
                                spt_symbols.append("<" + lng + ">")

                    # train tokenizer model on training data
                    self.encoder_tokenizer_model, self.decoder_tokenizer_model = MTDataPreproc.train_tokenizers(
                        out_dir=cfg.get('preproc_out_dir'),
                        src_fname=src_fname,
                        tgt_fname=tgt_fname,
                        shared_tokenizer=cfg.get('shared_tokenizer'),
                        encoder_tokenizer_vocab_size=cfg.encoder_tokenizer.get('vocab_size'),
                        decoder_tokenizer_vocab_size=cfg.decoder_tokenizer.get('vocab_size'),
                        encoder_tokenizer_name=cfg.encoder_tokenizer.get('library'),
                        decoder_tokenizer_name=cfg.decoder_tokenizer.get('library'),
                        encoder_tokenizer_coverage=cfg.encoder_tokenizer.get('coverage', 0.999),
                        decoder_tokenizer_coverage=cfg.decoder_tokenizer.get('coverage', 0.999),
                        global_rank=self.global_rank,
                        encoder_training_sample_size=cfg.encoder_tokenizer.get('training_sample_size', -1),
                        decoder_training_sample_size=cfg.decoder_tokenizer.get('training_sample_size', -1),
                        encoder_special_tokens=OmegaConf.to_container(cfg.encoder_tokenizer.special_tokens)
                        if cfg.encoder_tokenizer.special_tokens
                        else None,
                        decoder_special_tokens=OmegaConf.to_container(cfg.decoder_tokenizer.special_tokens)
                        if cfg.decoder_tokenizer.special_tokens
                        else None,
                        spt_symbols=spt_symbols,
                    )
                    # update config
                    self._cfg.encoder_tokenizer.tokenizer_model = self.encoder_tokenizer_model
                    self._cfg.decoder_tokenizer.tokenizer_model = self.decoder_tokenizer_model

                    tempdir.cleanup()
                else:
                    self.encoder_tokenizer_model = cfg.encoder_tokenizer.get('tokenizer_model')
                    self.decoder_tokenizer_model = cfg.decoder_tokenizer.get('tokenizer_model')

            self.encoder_tokenizer, self.decoder_tokenizer = self.get_enc_dec_tokenizers(
                encoder_tokenizer_name=cfg.encoder_tokenizer.get('library'),
                encoder_model_name=cfg.encoder.get('model_name'),
                encoder_tokenizer_model=getattr(self, "encoder_tokenizer_model", None),
                encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
                encoder_r2l=cfg.encoder_tokenizer.get('r2l', False),
                decoder_tokenizer_name=cfg.decoder_tokenizer.get('library'),
                decoder_model_name=cfg.decoder.get('model_name'),
                decoder_tokenizer_model=getattr(self, "decoder_tokenizer_model", None),
                decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
                decoder_r2l=cfg.decoder_tokenizer.get('r2l', False),
                encoder_tokenizer_legacy=cfg.encoder_tokenizer.get('sentencepiece_legacy', False),
                decoder_tokenizer_legacy=cfg.decoder_tokenizer.get('sentencepiece_legacy', False),
            )

            # If using tarred dataset for training, automatically create it if needed
            if cfg.train_ds.get('use_tarred_dataset'):
                if cfg.train_ds.get('tar_files') is None and cfg.train_ds.get('metadata_file') is None:
                    if cfg.get('preproc_out_dir') is None:
                        raise ValueError('Data preprocessing required but cfg.preproc_out_dir is None.')
                    if cfg.train_ds.get('src_file_name') is None or cfg.train_ds.get('tgt_file_name') is None:
                        raise ValueError(
                            'src_file_name and tgt_file_name needed to create tarred dataset but could not be found.'
                        )
                    # Preprocess data and cache for use during training
                    if self.global_rank == 0:
                        logging.info(
                            f"Creating tarred dataset for src: {cfg.train_ds.get('src_file_name')} and tgt: {cfg.train_ds.get('tgt_file_name')}"
                        )

                    if isinstance(cfg.train_ds.get('src_file_name'), str):
                        src_file_list = [cfg.train_ds.get('src_file_name')]
                        tgt_file_list = [cfg.train_ds.get('tgt_file_name')]
                        outdir_list = [cfg.get('preproc_out_dir')]
                    else:
                        src_file_list = cfg.train_ds.get('src_file_name')
                        tgt_file_list = cfg.train_ds.get('tgt_file_name')
                        if isinstance(cfg.get('src_language'), ListConfig):
                            langs = cfg.get('src_language')
                        elif isinstance(cfg.get('tgt_language'), ListConfig):
                            langs = cfg.get('tgt_language')
                        outdir_list = []
                        for lang in langs:
                            outdir_list.append(os.path.join(cfg.get('preproc_out_dir'), lang))

                    if len(src_file_list) != len(tgt_file_list) or len(src_file_list) != len(outdir_list):
                        raise ValueError(
                            "Number of source files, target files, and multilingual language pairs must be the same."
                        )

                    # TODO: have to get tokenizers instide .preprocess_parallel because they can't be pickled
                    metadata_file_list = []
                    for idx, src_file in enumerate(src_file_list):
                        self.train_tar_files, self.train_metadata_file = MTDataPreproc.preprocess_parallel_dataset(
                            clean=cfg.train_ds.clean,
                            src_fname=src_file,
                            tgt_fname=tgt_file_list[idx],
                            out_dir=outdir_list[idx],
                            encoder_tokenizer_name=cfg.encoder_tokenizer.get('library'),
                            encoder_model_name=cfg.encoder.get('model_name'),
                            encoder_tokenizer_model=getattr(self, "encoder_tokenizer_model", None),
                            encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
                            encoder_tokenizer_r2l=cfg.encoder_tokenizer.get('r2l', False),
                            encoder_tokenizer_legacy=cfg.encoder_tokenizer.get('sentencepiece_legacy', False),
                            decoder_tokenizer_name=cfg.decoder_tokenizer.get('library'),
                            decoder_model_name=cfg.decoder.get('model_name'),
                            decoder_tokenizer_model=getattr(self, "decoder_tokenizer_model", None),
                            decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
                            decoder_tokenizer_r2l=cfg.decoder_tokenizer.get('r2l', False),
                            decoder_tokenizer_legacy=cfg.decoder_tokenizer.get('sentencepiece_legacy', False),
                            max_seq_length=cfg.train_ds.get('max_seq_length', 512),
                            tokens_in_batch=cfg.train_ds.get('tokens_in_batch', 8192),
                            lines_per_dataset_fragment=cfg.train_ds.get('lines_per_dataset_fragment', 1000000),
                            num_batches_per_tarfile=cfg.train_ds.get('num_batches_per_tarfile', 1000),
                            min_seq_length=1,
                            global_rank=self.global_rank,
                            world_size=self.world_size,
                            n_jobs=cfg.train_ds.get('n_preproc_jobs', -2),
                            tar_file_prefix=cfg.train_ds.get('tar_file_prefix', 'parallel'),
                        )
                        metadata_file_list.append(self.train_metadata_file)
                    # update config
                    # self._cfg.train_ds.tar_files = self.tar_files_to_string(self.train_tar_files)
                    # self._cfg.train_ds.tar_files = self.train_tar_files
                    if isinstance(cfg.train_ds.get('metadata_file'), str):
                        self._cfg.train_ds.metadata_file = metadata_file_list[0]
                    else:
                        self._cfg.train_ds.metadata_file = metadata_file_list

                    logging.info(
                        f"Using tarred dataset created in folder(s) {outdir_list} and metadata created at {self._cfg.train_ds.metadata_file}"
                    )

                elif cfg.train_ds.get('tar_files') is not None and cfg.train_ds.get('metadata_file') is None:
                    raise ValueError('A metadata file is required for tarred dataset but cfg.metadata_file is None.')
                elif cfg.train_ds.get('tar_files') is None and cfg.train_ds.get('metadata_file') is not None:
                    if isinstance(cfg.train_ds.get('metadata_file'), str):
                        metadata_file_list = [cfg.train_ds.get('metadata_file')]
                    else:
                        metadata_file_list = cfg.train_ds.get('metadata_file')

                    for metadata_file in metadata_file_list:
                        with open(metadata_file) as metadata_reader:
                            metadata = json.load(metadata_reader)
                        if metadata['tar_files']:
                            logging.info(f"Using tarred dataset: {metadata['tar_files']}")
                        else:
                            raise ValueError(f'tar_files not provided and metadata does not have tar files')
                else:
                    self.train_tar_files = cfg.train_ds.get('tar_files')
                    self.train_metadata_file = cfg.train_ds.get('metadata_file')
                    logging.info(
                        f"Using tarred dataset from config at {self.train_tar_files} and metadata from {self.train_metadata_file}"
                    )

    def tar_files_to_string(self, tar_files):
        """ Tar files are generated in the following format: basename.number.tar
            Where number is an integer from 1 to the number of tar files.
            We convert this list to a string that can be used in the model config to specify
            tarred datasets: basename_OP_1..num_tar_files_CL_.tar 

        Args:
            tar_files (List[str]): List of tar files generated by preprocess_parallel_dataset
        """
        num_tar_files = len(tar_files)
        split_on_dot = tar_files[0].split('.')
        basename = '.'.join(split_on_dot[0:-2])
        tar_file_string = f'{basename}._OP_1..{num_tar_files}_CL_.tar'
        return tar_file_string

    @staticmethod
    def get_enc_dec_tokenizers(
        encoder_tokenizer_name=None,
        encoder_tokenizer_model=None,
        encoder_bpe_dropout=0.0,
        encoder_model_name=None,
        encoder_r2l=False,
        decoder_tokenizer_name=None,
        decoder_tokenizer_model=None,
        decoder_bpe_dropout=0.0,
        decoder_model_name=None,
        decoder_r2l=False,
        encoder_tokenizer_legacy=False,
        decoder_tokenizer_legacy=False,
    ):

        if encoder_bpe_dropout is None:
            encoder_bpe_dropout = 0.0

        if decoder_bpe_dropout is None:
            decoder_bpe_dropout = 0.0

        encoder_tokenizer = get_nmt_tokenizer(
            library=encoder_tokenizer_name,
            model_name=encoder_model_name,
            tokenizer_model=encoder_tokenizer_model,
            bpe_dropout=encoder_bpe_dropout,
            r2l=encoder_r2l,
            legacy=encoder_tokenizer_legacy,
        )
        decoder_tokenizer = get_nmt_tokenizer(
            library=decoder_tokenizer_name,
            model_name=decoder_model_name,
            tokenizer_model=decoder_tokenizer_model,
            bpe_dropout=decoder_bpe_dropout,
            r2l=decoder_r2l,
            legacy=decoder_tokenizer_legacy,
        )

        return encoder_tokenizer, decoder_tokenizer

    @staticmethod
    def get_monolingual_tokenizer(
        tokenizer_name=None, tokenizer_model=None, bpe_dropout=0.0,
    ):
        if tokenizer_name == 'sentencepiece':
            tokenizer = SentencePieceTokenizer(model_path=tokenizer_model)
        else:
            try:
                tokenizer = get_tokenizer(tokenizer_name, special_tokens={"pad_token": "[PAD]"})
            except Exception as e:
                raise ValueError(f'{tokenizer_name} is not supported by either NeMo or HuggingFace. {e}')

        return tokenizer

    # TODO: add local or global rank 0 decorator
    @staticmethod
    def preprocess_parallel_dataset(
        clean,
        src_fname,
        tgt_fname,
        out_dir,
        encoder_tokenizer_name,
        encoder_tokenizer_model,
        encoder_tokenizer_r2l,
        encoder_bpe_dropout,
        encoder_model_name,
        decoder_tokenizer_name,
        decoder_tokenizer_model,
        decoder_bpe_dropout,
        decoder_model_name,
        decoder_tokenizer_r2l,
        max_seq_length,
        min_seq_length,
        tokens_in_batch,
        lines_per_dataset_fragment,
        num_batches_per_tarfile,
        global_rank,
        world_size,
        n_jobs=-2,
        tar_file_prefix='parallel',
        encoder_tokenizer_legacy=False,
        decoder_tokenizer_legacy=False,
    ):
        """Create tarred dataset from large paired translation data.

        Args:
            clean (str): Cleans source and target sentences to get rid of noisy data.
            src_fname (str): path to source text data
            tgt_fname (str): path to target text data
            out_dir (str): path to write tarred dataset
            encoder_tokenizer (Any): tokenizer for encoder 
            decoder_tokenizer (Any): tokenizer for decoder
            max_seq_length (int): maximum sequence length 
            min_seq_length (int): minimum sequence length 
            tokens_in_batch (int): tokens per batch per GPU, effectively batch size 
            lines_per_dataset_fragment (int): number of lines to consider for bucketing and padding
            num_batches_per_tarfile (int): number of batches (pickle files) within each tarfile
            tar_file_prefix (str) : add string prefix to tar files 
            n_jobs (int): number of processes to use for data processing (-2 to use all but 2)
        """

        os.makedirs(out_dir, exist_ok=True)

        metadata_path = os.path.join(out_dir, f'metadata.tokens.{tokens_in_batch}.json')

        if global_rank == 0:
            tar_files_in_out_dir = glob.glob(f'{out_dir}/*.tar')
            if tar_files_in_out_dir:
                logging.info(
                    f'Tarred dataset detected: {tar_files_in_out_dir} and will be used. Remove if reprocessing.'
                )
            else:
                filenames = [src_fname, tgt_fname]

                # get number of lines so that we can create a partition of the lines of the text file
                num_src_lines, num_tgt_lines = Parallel(n_jobs=2)(
                    delayed(MTDataPreproc._get_num_lines)(filename) for filename in filenames
                )
                logging.info(f'Found {num_src_lines} source lines and  {num_tgt_lines} target lines.')
                assert num_src_lines == num_tgt_lines, 'Number of source lines should equal number of target lines.'

                # create a partition of lines that we can parallelize over
                lines_partition = MTDataPreproc._get_lines_partition(num_src_lines, lines_per_dataset_fragment)
                logging.info(f"Found {len(lines_partition)} fragments to parallelize over.")

                # create tarfiles for each fragment in parallel
                results_list = Parallel(n_jobs=n_jobs)(
                    delayed(MTDataPreproc._process_fragment)(
                        src_filename=src_fname,
                        tgt_filename=tgt_fname,
                        lines_indices=lines_indices,
                        out_dir=out_dir,
                        num_batches_per_tarfile=num_batches_per_tarfile,
                        clean=clean,
                        max_seq_length=max_seq_length,
                        min_seq_length=min_seq_length,
                        tokens_in_batch=tokens_in_batch,
                        encoder_tokenizer_name=encoder_tokenizer_name,
                        encoder_tokenizer_model=encoder_tokenizer_model,
                        encoder_bpe_dropout=encoder_bpe_dropout,
                        encoder_model_name=encoder_model_name,
                        decoder_tokenizer_name=decoder_tokenizer_name,
                        decoder_tokenizer_model=decoder_tokenizer_model,
                        decoder_bpe_dropout=decoder_bpe_dropout,
                        decoder_model_name=decoder_model_name,
                        fragment_index=fragment_index,
                        encoder_tokenizer_r2l=encoder_tokenizer_r2l,
                        decoder_tokenizer_r2l=decoder_tokenizer_r2l,
                        encoder_tokenizer_legacy=encoder_tokenizer_legacy,
                        decoder_tokenizer_legacy=decoder_tokenizer_legacy,
                    )
                    for fragment_index, lines_indices in enumerate(lines_partition)
                )

                # compute total batches so far
                total_batches = sum([batch_count for batch_count, _ in results_list])

                # save batches from tar files containing the left over batches (if there's enough batches)
                remainder_tar_file_ctr = 0
                remainder_tar_file_path = os.path.join(
                    out_dir, f'remainder-batches.tokens.{tokens_in_batch}.tar_file_{remainder_tar_file_ctr}.tar'
                )
                remainder_tar_file_ptr = tarfile.open(remainder_tar_file_path, 'w')
                batch_in_tar_ctr = 0
                for _, tar_file_path in results_list:
                    tar_file_ptr = tarfile.open(tar_file_path, 'r')
                    for member in tar_file_ptr.getmembers():
                        remainder_tar_file_ptr.addfile(member, tar_file_ptr.extractfile(member.name))
                        batch_in_tar_ctr += 1
                        if batch_in_tar_ctr == num_batches_per_tarfile:
                            remainder_tar_file_ctr += 1
                            remainder_tar_file_ptr.close()
                            remainder_tar_file_path = os.path.join(
                                out_dir,
                                f'remainder-batches.tokens.{tokens_in_batch}.tar_file_{remainder_tar_file_ctr}.tar',
                            )
                            remainder_tar_file_ptr = tarfile.open(remainder_tar_file_path, 'w',)
                            batch_in_tar_ctr = 0
                    tar_file_ptr.close()
                    os.remove(tar_file_path)

                # log the number of batches remaining as they will be discarded
                num_batches_discarded = len(remainder_tar_file_ptr.getmembers())
                total_batches -= num_batches_discarded
                logging.info(
                    f'Number of batches discarded: {num_batches_discarded}, total batches kept: {total_batches}'
                )
                remainder_tar_file_ptr.close()
                os.remove(remainder_tar_file_path)

                # dump metadata to json
                metadata = {}
                metadata['num_batches'] = total_batches

                # rename tar files so they can be more easily used with CLI and YAML
                tar_file_paths = glob.glob(f'{out_dir}/*.tar')
                for index, path in enumerate(tar_file_paths):
                    os.rename(
                        path, os.path.join(out_dir, f'{tar_file_prefix}.batches.tokens.{tokens_in_batch}.{index}.tar')
                    )

                # add tar files to manifest
                tar_file_paths = glob.glob(f'{out_dir}/*.tar')
                metadata['tar_files'] = tar_file_paths
                json.dump(metadata, open(metadata_path, 'w'))

        tar_file_paths = glob.glob(f'{out_dir}/*.tar')

        num_tar_files = len(tar_file_paths)
        if num_tar_files < world_size:
            raise ValueError(
                (
                    f'Number of tar files found: {num_tar_files} is less than world size: {world_size}. '
                    f'There should be at least one tar file per GPU (ideally many tar files per GPU). '
                    f'This may be due to dataset size, it is advisable to use at least 5M sentence pairs for tarred datasets. '
                    f'Decrease num_batches_per_tarfile or num_tokens_per_batch to increase the number of tarfiles. '
                    f'Also using shard_strategy=replicate will use all available tarfiles for every GPU. '
                )
            )

        return tar_file_paths, metadata_path

    @staticmethod
    def _get_num_lines(filename):
        with open(filename) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    @staticmethod
    def _get_lines_partition(num_lines, lines_per_dataset_fragment):
        # create partition based on fragment size
        fragment_indices = []
        for i in range(0, num_lines, lines_per_dataset_fragment):
            fragment_indices.append([i, i + lines_per_dataset_fragment])
        # modify last indices
        last_indices = fragment_indices.pop()
        last_indices[1] = -1
        fragment_indices.append(last_indices)
        # if fragment_indices[-1][1] >= num_lines:
        #     fragment_indices.pop()
        return fragment_indices

    @staticmethod
    def _process_fragment(
        src_filename,
        tgt_filename,
        lines_indices,
        out_dir,
        num_batches_per_tarfile,
        clean,
        max_seq_length,
        min_seq_length,
        tokens_in_batch,
        encoder_tokenizer_name,
        encoder_tokenizer_model,
        encoder_bpe_dropout,
        encoder_model_name,
        encoder_tokenizer_r2l,
        decoder_tokenizer_name,
        decoder_tokenizer_model,
        decoder_bpe_dropout,
        decoder_model_name,
        decoder_tokenizer_r2l,
        fragment_index,
        encoder_tokenizer_legacy,
        decoder_tokenizer_legacy,
    ):
        start = lines_indices[0]
        stop = lines_indices[1]

        # write lines in partition to temporary files to be consumed by write_parallel_batches_to_tarfiles
        tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')

        with open(src_filename, 'r') as src_in, open(tgt_filename) as tgt_in:
            for line_number, (src_line, tgt_line) in enumerate(zip(src_in, tgt_in)):
                if start <= line_number and line_number < stop:
                    if src_line and tgt_line:
                        tmp_f_src.write(src_line)
                        tmp_f_tgt.write(tgt_line)

        tmp_f_src.close()
        tmp_f_tgt.close()

        num_batches_from_fragment, remainder_tar_file_path = MTDataPreproc.write_parallel_batches_to_tarfiles(
            out_dir=out_dir,
            num_batches_per_tarfile=num_batches_per_tarfile,
            clean=clean,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            src_fname=tmp_f_src.name,
            tgt_fname=tmp_f_tgt.name,
            num_tokens=tokens_in_batch,
            encoder_tokenizer_name=encoder_tokenizer_name,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=encoder_bpe_dropout,
            encoder_model_name=encoder_model_name,
            encoder_tokenizer_r2l=encoder_tokenizer_r2l,
            decoder_tokenizer_name=decoder_tokenizer_name,
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=decoder_bpe_dropout,
            decoder_model_name=decoder_model_name,
            decoder_tokenizer_r2l=decoder_tokenizer_r2l,
            fragment_index=fragment_index,
            encoder_tokenizer_legacy=encoder_tokenizer_legacy,
            decoder_tokenizer_legacy=decoder_tokenizer_legacy,
        )

        os.remove(tmp_f_src.name)
        os.remove(tmp_f_tgt.name)

        return num_batches_from_fragment, remainder_tar_file_path

    @staticmethod
    def preprocess_monolingual_dataset(
        clean,
        fname,
        out_dir,
        tokenizer,
        max_seq_length,
        min_seq_length,
        tokens_in_batch,
        lines_per_dataset_fragment,
        num_batches_per_tarfile,
        pkl_file_prefix,
        global_rank,
        world_size,
    ):
        """Create tarred dataset from a large monolingual corpus.

        Args:
            clean (str): Cleans sentences to get rid of very long or short sentences.
            fname (str): Path to source text data
            out_dir (str): Path to write tarred dataset
            tokenizer (Any): Path to tokenizer model
            max_seq_length (int): maximum sequence length 
            min_seq_length (int): minimum sequence length 
            tokens_in_batch (int): tokens per batch per GPU, effectively batch size 
            lines_per_dataset_fragment (int): number of lines to consider for bucketing and padding
            num_batches_per_tarfile (int): number of batches (pickle files) within each tarfile
            global_rank (int): if set to zero, data will be processed on this node
            world_size (int): total number of processes being run (for training only, set to 1 when preproc only)
        """
        os.makedirs(out_dir, exist_ok=True)

        tar_file_ctr = 1
        num_files_in_tar = 0
        num_lines = 0
        shard_num = 0
        global_batch_ctr = 0
        tmp_f = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tar_file_ptr = tarfile.open(
            os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, tokens_in_batch, 1)), 'w'
        )
        metadata_path = os.path.join(out_dir, f'metadata.tokens.{tokens_in_batch}.json')
        with open(fname, 'r') as f:
            for line in f:
                tmp_f.write(line)
                num_lines += 1

                if num_lines == lines_per_dataset_fragment:
                    tmp_f.close()
                    (
                        tar_file_ptr,
                        global_batch_ctr,
                        num_files_in_tar,
                        tar_file_ctr,
                    ) = MTDataPreproc.write_monolingual_batches_to_tarfiles(
                        out_dir,
                        num_batches_per_tarfile,
                        clean,
                        max_seq_length,
                        min_seq_length,
                        tmp_f.name,
                        tokens_in_batch,
                        tokenizer,
                        num_files_in_tar=num_files_in_tar,
                        tar_file_ptr=tar_file_ptr,
                        tar_file_ctr=tar_file_ctr,
                        global_batch_ctr=global_batch_ctr,
                        pkl_file_prefix=pkl_file_prefix,
                    )

                    num_lines = 0
                    shard_num += 1
                    os.remove(tmp_f.name)
                    tmp_f = tempfile.NamedTemporaryFile(delete=False, mode='w')

        tmp_f.close()
        (
            tar_file_ptr,
            global_batch_ctr,
            num_files_in_tar,
            tar_file_ctr,
        ) = MTDataPreproc.write_monolingual_batches_to_tarfiles(
            out_dir,
            num_batches_per_tarfile,
            clean,
            max_seq_length,
            min_seq_length,
            tmp_f.name,
            tokens_in_batch,
            tokenizer,
            num_files_in_tar=num_files_in_tar,
            tar_file_ptr=tar_file_ptr,
            tar_file_ctr=tar_file_ctr,
            global_batch_ctr=global_batch_ctr,
            pkl_file_prefix=pkl_file_prefix,
        )
        tar_file_ptr.close()
        os.remove(tmp_f.name)

        if num_files_in_tar != num_batches_per_tarfile:
            os.remove(
                os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, tokens_in_batch, tar_file_ctr))
            )
            global_batch_ctr -= num_files_in_tar
            print('Dropping %d batches because of overflow' % (num_files_in_tar))

        json.dump({'num_batches': global_batch_ctr}, open(os.path.join(out_dir, 'metadata.json'), 'w'))

        tar_file_paths = glob.glob(f'{out_dir}/{pkl_file_prefix}-batches.tokens.{tokens_in_batch}.*.tar')

        num_tar_files = len(tar_file_paths)
        if num_tar_files < world_size:
            raise ValueError(
                (
                    f'Number of tar files found: {num_tar_files} is less than world size: {world_size}. '
                    f'There should be at least one tar file per GPU (ideally many tar files per GPU). '
                    f'This may be due to dataset size, it is advisable to use at least 5M sentence pairs for tarred datasets. '
                    f'Decrease num_batches_per_tarfile or num_tokens_per_batch to increase the number of tarfiles. '
                    f'Also using shard_strategy=replicate will use all available tarfiles for every GPU. '
                )
            )

        return tar_file_paths, metadata_path

    @staticmethod
    def train_tokenizers(
        out_dir,
        src_fname,
        tgt_fname,
        shared_tokenizer,
        encoder_tokenizer_name,
        encoder_tokenizer_vocab_size,
        encoder_tokenizer_coverage,
        decoder_tokenizer_name,
        decoder_tokenizer_vocab_size,
        decoder_tokenizer_coverage,
        global_rank,
        encoder_training_sample_size=-1,
        decoder_training_sample_size=-1,
        encoder_special_tokens=None,
        decoder_special_tokens=None,
        spt_symbols=None,
        byte_fallback=False,
        split_digits=False,
        split_by_whitespace=True,
    ):
        """Trains a tokenizer with requested parameters, returns None if the tokenizer is not trainable"""

        encoder_tokenizer_model = None
        decoder_tokenizer_model = None
        os.makedirs(out_dir, exist_ok=True)

        supported_train_tokenizers = ['sentencepiece']

        if encoder_special_tokens:
            if isinstance(encoder_special_tokens, dict):
                encoder_special_tokens = list(encoder_special_tokens.values())
                print(encoder_special_tokens)

        if decoder_special_tokens:
            if isinstance(decoder_special_tokens, dict):
                decoder_special_tokens = list(decoder_special_tokens.values())

        if shared_tokenizer:
            if encoder_tokenizer_name in supported_train_tokenizers:
                encoder_tokenizer_model = os.path.join(
                    out_dir, 'shared_tokenizer.%d.BPE.model' % (encoder_tokenizer_vocab_size)
                )
                decoder_tokenizer_model = encoder_tokenizer_model
                if global_rank == 0:
                    if os.path.isfile(encoder_tokenizer_model):
                        logging.info(
                            f'Shared tokenizer model {encoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                        )
                    else:
                        logging.info(
                            f'Shared tokenizer model {encoder_tokenizer_model} not found. Training tokenizer model.'
                        )
                        with tempfile.TemporaryDirectory() as tmp:
                            concat_data_path = os.path.join(tmp, 'concat_dataset.txt')
                            os.system('cat %s %s > %s' % (src_fname, tgt_fname, concat_data_path))
                            create_spt_model(
                                data_file=concat_data_path,
                                vocab_size=encoder_tokenizer_vocab_size,
                                sample_size=encoder_training_sample_size,
                                do_lower_case=False,
                                tokenizer_type='bpe',
                                character_coverage=encoder_tokenizer_coverage,
                                output_dir=out_dir,
                                bos=True,
                                eos=True,
                                pad=True,
                                control_symbols=spt_symbols,
                                user_defined_symbols=encoder_special_tokens,
                                byte_fallback=byte_fallback,
                                split_digits=split_digits,
                                split_by_whitespace=split_by_whitespace,
                            )
                            os.rename(
                                os.path.join(out_dir, 'tokenizer.model'), encoder_tokenizer_model,
                            )
        else:
            if encoder_tokenizer_name in supported_train_tokenizers:
                encoder_tokenizer_model = os.path.join(
                    out_dir, 'tokenizer.encoder.%d.BPE.model' % (encoder_tokenizer_vocab_size)
                )
                if global_rank == 0:
                    if os.path.isfile(encoder_tokenizer_model):
                        logging.info(
                            f'Encoder tokenizer model {encoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                        )
                    else:
                        logging.info(
                            f'Encoder tokenizer model {encoder_tokenizer_model} not found. Training tokenizer model.'
                        )
                        dir_name = os.path.dirname(encoder_tokenizer_model)
                        create_spt_model(
                            data_file=src_fname,
                            vocab_size=encoder_tokenizer_vocab_size,
                            sample_size=encoder_training_sample_size,
                            do_lower_case=False,
                            tokenizer_type='bpe',
                            character_coverage=encoder_tokenizer_coverage,
                            output_dir=dir_name,
                            bos=True,
                            eos=True,
                            pad=True,
                            control_symbols=spt_symbols,
                            user_defined_symbols=encoder_special_tokens,
                            byte_fallback=byte_fallback,
                            split_digits=split_digits,
                            split_by_whitespace=split_by_whitespace,
                        )
                        os.rename(os.path.join(dir_name, 'tokenizer.model'), encoder_tokenizer_model)

            if decoder_tokenizer_name in supported_train_tokenizers:
                decoder_tokenizer_model = os.path.join(
                    out_dir, 'tokenizer.decoder.%d.BPE.model' % (decoder_tokenizer_vocab_size)
                )
                if global_rank == 0:
                    if os.path.isfile(decoder_tokenizer_model):
                        logging.info(
                            f'Decoder tokenizer model {decoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                        )
                    else:
                        logging.info(
                            f'Decoder tokenizer model {decoder_tokenizer_model} not found. Training tokenizer model.'
                        )
                        dir_name = os.path.dirname(decoder_tokenizer_model)
                        create_spt_model(
                            data_file=tgt_fname,
                            vocab_size=decoder_tokenizer_vocab_size,
                            sample_size=decoder_training_sample_size,
                            do_lower_case=False,
                            tokenizer_type='bpe',
                            character_coverage=decoder_tokenizer_coverage,
                            output_dir=dir_name,
                            bos=True,
                            eos=True,
                            pad=True,
                            control_symbols=spt_symbols,
                            user_defined_symbols=decoder_special_tokens,
                            byte_fallback=byte_fallback,
                            split_digits=split_digits,
                            split_by_whitespace=split_by_whitespace,
                        )
                        os.rename(os.path.join(dir_name, 'tokenizer.model'), decoder_tokenizer_model)

        return encoder_tokenizer_model, decoder_tokenizer_model

    @staticmethod
    def write_parallel_batches_to_tarfiles(
        out_dir,
        num_batches_per_tarfile,
        clean,
        max_seq_length,
        min_seq_length,
        src_fname,
        tgt_fname,
        num_tokens,
        encoder_tokenizer_name,
        encoder_tokenizer_model,
        encoder_tokenizer_r2l,
        encoder_bpe_dropout,
        encoder_model_name,
        decoder_tokenizer_name,
        decoder_tokenizer_model,
        decoder_bpe_dropout,
        decoder_model_name,
        decoder_tokenizer_r2l,
        fragment_index,
        encoder_tokenizer_legacy=False,
        decoder_tokenizer_legacy=False,
    ):
        """
        Writes current fragment of the overall parallel corpus to tarfiles by:
        (1) Creating a minibatches using a TranslationDataset object.
        (2) Writing each minibatch to a pickle file.
        (3) Adding pickle files to a tarfile until it reaches num_batches_per_tarfile.
        """

        dataset = TranslationDataset(
            dataset_src=src_fname,
            dataset_tgt=tgt_fname,
            tokens_in_batch=num_tokens,
            clean=clean,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            max_seq_length_diff=max_seq_length,
            max_seq_length_ratio=max_seq_length,
            cache_ids=False,
            cache_data_per_node=False,
            use_cache=False,
        )
        encoder_tokenizer, decoder_tokenizer = MTDataPreproc.get_enc_dec_tokenizers(
            encoder_tokenizer_name=encoder_tokenizer_name,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=encoder_bpe_dropout,
            encoder_model_name=encoder_model_name,
            encoder_r2l=encoder_tokenizer_r2l,
            decoder_tokenizer_name=decoder_tokenizer_name,
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=decoder_bpe_dropout,
            decoder_model_name=decoder_model_name,
            decoder_r2l=decoder_tokenizer_r2l,
            encoder_tokenizer_legacy=encoder_tokenizer_legacy,
            decoder_tokenizer_legacy=decoder_tokenizer_legacy,
        )

        # validate no token is negative for sentencepiece tokenizers and add missing special tokens.
        for tok_name, tok_library, tok_model, legacy in [
            ("encoder_tokenizer", encoder_tokenizer_name, encoder_tokenizer, encoder_tokenizer_legacy),
            ("decoder_tokenizer", decoder_tokenizer_name, decoder_tokenizer, decoder_tokenizer_legacy),
        ]:
            if tok_library == 'sentencepiece':
                negative_tokens = []
                for n in ["eos_id", "bos_id", "unk_id", "pad_id"]:
                    v = getattr(tok_model.tokenizer, n)()
                    if v < 0:
                        negative_tokens.append(f"{n}={v}")
                if negative_tokens and not legacy:
                    raise ValueError(
                        f"{tok_name}=sentencepiece has invalid negative special tokens = {negative_tokens}"
                    )
                # If using the legacy sentencepiece tokenizer, we can add the missing tokens as "special" tokens.
                else:
                    # If using sentencepiece legacy, eos, bos and pad need to be set/added differently.
                    if legacy:
                        # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
                        if not hasattr(tok_model, 'pad_token'):
                            if hasattr(tok_model.tokenizer, 'pad_id') and tok_model.tokenizer.pad_id() > 0:
                                tok_model.pad_token = tok_model.tokenizer.id_to_piece(tok_model.tokenizer.pad_id())
                            else:
                                tok_model.add_special_tokens({'pad_token': '<pad>'})
                        else:
                            tok_model.add_special_tokens({'pad_token': '<pad>'})

                        if not hasattr(tok_model, 'bos_token'):
                            if hasattr(tok_model.tokenizer, 'bos_id') and tok_model.tokenizer.bos_id() > 0:
                                tok_model.bos_token = tok_model.tokenizer.id_to_piece(tok_model.tokenizer.bos_id())
                            else:
                                tok_model.add_special_tokens({'bos_token': '<bos>'})
                        else:
                            tok_model.add_special_tokens({'bos_token': '<s>'})

                        if not hasattr(tok_model, 'eos_token'):
                            if hasattr(tok_model.tokenizer, 'eos_id') and tok_model.tokenizer.eos_id() > 0:
                                tok_model.eos_token = tok_model.tokenizer.id_to_piece(tok_model.tokenizer.eos_id())
                            else:
                                tok_model.add_special_tokens({'eos_token': '<eos>'})
                        else:
                            tok_model.add_special_tokens({'eos_token': '</s>'})

        dataset.batchify(encoder_tokenizer, decoder_tokenizer)

        tar_file_ctr = 0
        tar_file_path = os.path.join(
            out_dir, 'fragment-%s-batches.tokens.%d.%d.tar' % (fragment_index, num_tokens, tar_file_ctr)
        )
        tar_file_ptr = tarfile.open(tar_file_path, 'w')
        total_batch_ctr = 0
        batch_ctr = 0
        for _, batch in dataset.batches.items():
            total_batch_ctr += 1
            batch_ctr += 1
            pickle.dump(
                batch,
                open(os.path.join(out_dir, 'fragment-%s-batch-%d.pkl' % (fragment_index, total_batch_ctr)), 'wb'),
            )
            tar_file_ptr.add(os.path.join(out_dir, 'fragment-%s-batch-%d.pkl' % (fragment_index, total_batch_ctr)))
            os.remove(os.path.join(out_dir, 'fragment-%s-batch-%d.pkl' % (fragment_index, total_batch_ctr)))

            if batch_ctr == num_batches_per_tarfile:
                tar_file_ctr += 1
                tar_file_ptr.close()
                tar_file_path = os.path.join(
                    out_dir, 'fragment-%s-batches.tokens.%d.%d.tar' % (fragment_index, num_tokens, tar_file_ctr)
                )
                tar_file_ptr = tarfile.open(tar_file_path, 'w',)
                batch_ctr = 0

        # return tar files paths that have batches remaining
        remainder_tar_file_path = tar_file_ptr.name
        tar_file_ptr.close()

        return total_batch_ctr, remainder_tar_file_path

    @staticmethod
    def write_monolingual_batches_to_tarfiles(
        out_dir,
        num_batches_per_tarfile,
        clean,
        max_seq_length,
        min_seq_length,
        fname,
        num_tokens,
        tokenizer,
        num_files_in_tar,
        tar_file_ptr,
        tar_file_ctr,
        global_batch_ctr,
        pkl_file_prefix,
    ):
        """
        Writes current fragment of the overall parallel corpus to tarfiles by:
        (1) Creating a minibatches using a SentenceDataset object.
        (2) Writing each minibatch to a pickle file.
        (3) Adding pickle files to a tarfile until it reaches num_batches_per_tarfile.
        """

        dataset = SentenceDataset(
            tokenizer=tokenizer,
            dataset=fname,
            tokens_in_batch=num_tokens,
            clean=clean,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            cache_ids=False,
        )

        for batch in dataset.batches:
            global_batch_ctr += 1
            batch = {'src': batch}
            pickle.dump(
                batch, open(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)), 'wb')
            )

            if num_files_in_tar == num_batches_per_tarfile:
                tar_file_ctr += 1
                tar_file_ptr.close()
                tar_file_ptr = tarfile.open(
                    os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, num_tokens, tar_file_ctr)),
                    'w',
                )
                num_files_in_tar = 0

            tar_file_ptr.add(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)))
            num_files_in_tar += 1
            os.remove(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)))
        return tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr

    @property
    def cfg(self):
        return self._cfg
