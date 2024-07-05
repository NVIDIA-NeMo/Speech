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

import json
import os
import pickle as pkl
import shutil
import tarfile
import tempfile
from copy import deepcopy
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.mixins.mixins import DiarizationMixin
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    segments_manifest_to_subsegments_manifest,
    validate_vad_manifest,
    write_rttm2manifest,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.core.classes import Model
from nemo.utils import logging, model_utils

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['ClusteringDiarizer']

_MODEL_CONFIG_YAML = "model_config.yaml"
_VAD_MODEL = "vad_model.nemo"
_SPEAKER_MODEL = "speaker_model.nemo"


def get_available_model_names(class_name):
    "lists available pretrained model names from NGC"
    available_models = class_name.list_available_models()
    return list(map(lambda x: x.pretrained_model_name, available_models))


class ClusteringDiarizer(torch.nn.Module, Model, DiarizationMixin):
    """
    Inference model Class for offline speaker diarization.
    This class handles required functionality for diarization : Speech Activity Detection, Segmentation,
    Extract Embeddings, Clustering, Resegmentation and Scoring.
    All the parameters are passed through config file
    """

    def __init__(self, cfg: Union[DictConfig, Any], speaker_model=None):
        super().__init__()
        if isinstance(cfg, DictConfig):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            # Convert config to support Hydra 1.0+ instantiation
            cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        # Diarizer set up
        self._diarizer_params = self._cfg.diarizer

        # init vad model
        self.has_vad_model = False
        if not self._diarizer_params.oracle_vad:
            if self._cfg.diarizer.vad.model_path is not None:
                self._vad_params = self._cfg.diarizer.vad.parameters
                self._init_vad_model()

        # init speaker model
        self.multiscale_embeddings_and_timestamps = {}
        self._init_speaker_model(speaker_model)
        self._speaker_params = self._cfg.diarizer.speaker_embeddings.parameters

        # Clustering params
        self._cluster_params = self._diarizer_params.clustering.parameters

    @classmethod
    def list_available_models(cls):
        pass

    def _init_vad_model(self):
        """
        Initialize VAD model with model name or path passed through config
        """
        model_path = self._cfg.diarizer.vad.model_path
        if model_path.endswith('.nemo'):
            self._vad_model = EncDecClassificationModel.restore_from(model_path, map_location=self._cfg.device)
            logging.info("VAD model loaded locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecClassificationModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "vad_telephony_marblenet"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._vad_model = EncDecClassificationModel.from_pretrained(
                model_name=model_path, map_location=self._cfg.device
            )
        self._vad_window_length_in_sec = self._vad_params.window_length_in_sec
        self._vad_shift_length_in_sec = self._vad_params.shift_length_in_sec
        self.has_vad_model = True

    def _init_speaker_model(self, speaker_model=None):
        """
        Initialize speaker embedding model with model name or path passed through config
        """
        if speaker_model is not None:
            if self._cfg.device is None and torch.cuda.is_available():
                self._speaker_model = speaker_model.to(torch.device('cuda'))
            else:
                self._speaker_model = speaker_model
        else:
            model_path = self._cfg.diarizer.speaker_embeddings.model_path
            if model_path is not None and model_path.endswith('.nemo'):
                self._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=self._cfg.device)
                logging.info("Speaker Model restored locally from {}".format(model_path))
            elif model_path.endswith('.ckpt'):
                self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(
                    model_path, map_location=self._cfg.device
                )
                logging.info("Speaker Model restored locally from {}".format(model_path))
            else:
                if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                    logging.warning(
                        "requested {} model name not available in pretrained models, instead".format(model_path)
                    )
                    model_path = "ecapa_tdnn"
                logging.info("Loading pretrained {} model from NGC".format(model_path))
                self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                    model_name=model_path, map_location=self._cfg.device
                )
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _setup_vad_test_data(self, manifest_vad_input):
        vad_dl_config = {
            'manifest_filepath': manifest_vad_input,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'vad_stream': True,
            'labels': [
                'infer',
            ],
            'window_length_in_sec': self._vad_window_length_in_sec,
            'shift_length_in_sec': self._vad_shift_length_in_sec,
            'trim_silence': False,
            'num_workers': self._cfg.num_workers,
        }
        self._vad_model.setup_test_data(test_data_config=vad_dl_config)

    def _setup_spkr_test_data(self, manifest_file):
        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'trim_silence': False,
            'labels': None,
            'num_workers': self._cfg.num_workers,
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _run_vad(self, manifest_file):
        """
        Run voice activity detection.
        Get log probability of voice activity detection and smoothes using the post processing parameters.
        Using generated frame level predictions generated manifest file for later speaker embedding extraction.
        input:
        manifest_file (str) : Manifest file containing path to audio file and label as infer

        """

        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.makedirs(self._vad_dir)

        self._vad_model.eval()

        time_unit = int(self._vad_window_length_in_sec / self._vad_shift_length_in_sec)
        trunc = int(time_unit / 2)
        trunc_l = time_unit - trunc
        all_len = 0
        data = []
        for line in open(manifest_file, 'r', encoding='utf-8'):
            file = json.loads(line)['audio_filepath']
            data.append(get_uniqname_from_filepath(file))

        status = get_vad_stream_status(data)
        for i, test_batch in enumerate(
            tqdm(self._vad_model.test_dataloader(), desc='vad', leave=True, disable=not self.verbose)
        ):
            test_batch = [x.to(self._vad_model.device) for x in test_batch]
            with autocast():
                log_probs = self._vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
                probs = torch.softmax(log_probs, dim=-1)
                pred = probs[:, 1]
                if status[i] == 'start':
                    to_save = pred[:-trunc]
                elif status[i] == 'next':
                    to_save = pred[trunc:-trunc_l]
                elif status[i] == 'end':
                    to_save = pred[trunc_l:]
                else:
                    to_save = pred
                all_len += len(to_save)
                outpath = os.path.join(self._vad_dir, data[i] + ".frame")
                with open(outpath, "a", encoding='utf-8') as fout:
                    for f in range(len(to_save)):
                        fout.write('{0:0.4f}\n'.format(to_save[f]))
            del test_batch
            if status[i] == 'end' or status[i] == 'single':
                all_len = 0

        if not self._vad_params.smoothing:
            # Shift the window by 10ms to generate the frame and use the prediction of the window to represent the label for the frame;
            self.vad_pred_dir = self._vad_dir
            frame_length_in_sec = self._vad_shift_length_in_sec
        else:
            # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
            # smoothing_method would be either in majority vote (median) or average (mean)
            logging.info("Generating predictions with overlapping input segments")
            smoothing_pred_dir = generate_overlap_vad_seq(
                frame_pred_dir=self._vad_dir,
                smoothing_method=self._vad_params.smoothing,
                overlap=self._vad_params.overlap,
                window_length_in_sec=self._vad_window_length_in_sec,
                shift_length_in_sec=self._vad_shift_length_in_sec,
                num_workers=self._cfg.num_workers,
            )
            self.vad_pred_dir = smoothing_pred_dir
            frame_length_in_sec = 0.01

        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")

        vad_params = self._vad_params if isinstance(self._vad_params, (DictConfig, dict)) else self._vad_params.dict()
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.vad_pred_dir,
            postprocessing_params=vad_params,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=self._cfg.num_workers,
            out_dir=self._vad_dir,
        )

        AUDIO_VAD_RTTM_MAP = {}
        for key in self.AUDIO_RTTM_MAP:
            if os.path.exists(os.path.join(table_out_dir, key + ".txt")):
                AUDIO_VAD_RTTM_MAP[key] = deepcopy(self.AUDIO_RTTM_MAP[key])
                AUDIO_VAD_RTTM_MAP[key]['rttm_filepath'] = os.path.join(table_out_dir, key + ".txt")
            else:
                logging.warning(f"no vad file found for {key} due to zero or negative duration")

        write_rttm2manifest(AUDIO_VAD_RTTM_MAP, self._vad_out_file)
        self._speaker_manifest_path = self._vad_out_file

    def _run_segmentation(self, window: float, shift: float, scale_tag: str = ''):

        self.subsegments_manifest_path = os.path.join(self._speaker_dir, f'subsegments{scale_tag}.json')
        logging.info(
            f"Subsegmentation for embedding extraction:{scale_tag.replace('_',' ')}, {self.subsegments_manifest_path}"
        )
        self.subsegments_manifest_path = segments_manifest_to_subsegments_manifest(
            segments_manifest_file=self._speaker_manifest_path,
            subsegments_manifest_file=self.subsegments_manifest_path,
            window=window,
            shift=shift,
        )
        return None

    def _perform_speech_activity_detection(self):
        """
        Checks for type of speech activity detection from config. Choices are NeMo VAD,
        external vad manifest and oracle VAD (generates speech activity labels from provided RTTM files)
        """
        if self.has_vad_model:
            self._auto_split = True
            self._split_duration = 50
            manifest_vad_input = self._diarizer_params.manifest_filepath

            if self._auto_split:
                logging.info("Split long audio file to avoid CUDA memory issue")
                logging.debug("Try smaller split_duration if you still have CUDA memory issue")
                config = {
                    'input': manifest_vad_input,
                    'window_length_in_sec': self._vad_window_length_in_sec,
                    'split_duration': self._split_duration,
                    'num_workers': self._cfg.num_workers,
                    'out_dir': self._diarizer_params.out_dir,
                }
                manifest_vad_input = prepare_manifest(config)
            else:
                logging.warning(
                    "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
                )

            self._setup_vad_test_data(manifest_vad_input)
            self._run_vad(manifest_vad_input)

        elif self._diarizer_params.vad.external_vad_manifest is not None:
            self._speaker_manifest_path = self._diarizer_params.vad.external_vad_manifest
        elif self._diarizer_params.oracle_vad:
            self._speaker_manifest_path = os.path.join(self._speaker_dir, 'oracle_vad_manifest.json')
            self._speaker_manifest_path = write_rttm2manifest(self.AUDIO_RTTM_MAP, self._speaker_manifest_path)
        else:
            raise ValueError(
                "Only one of diarizer.oracle_vad, vad.model_path or vad.external_vad_manifest must be passed from config"
            )
        validate_vad_manifest(self.AUDIO_RTTM_MAP, vad_manifest=self._speaker_manifest_path)

    def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use.
        """
        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        self.embeddings = {}
        self._speaker_model.eval()
        self.time_stamps = {}

        all_embs = torch.empty([0])
        for test_batch in tqdm(
            self._speaker_model.test_dataloader(),
            desc=f'[{scale_idx+1}/{num_scales}] extract embeddings',
            leave=True,
            disable=not self.verbose,
        ):
            test_batch = [x.to(self._speaker_model.device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
            del test_batch

        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
                if uniq_name in self.embeddings:
                    self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
                else:
                    self.embeddings[uniq_name] = all_embs[i].view(1, -1)
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                self.time_stamps[uniq_name].append([start, end])

        if self._speaker_params.save_embeddings:
            embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir, exist_ok=True)

            prefix = get_uniqname_from_filepath(manifest_file)
            name = os.path.join(embedding_dir, prefix)
            self._embeddings_file = name + f'_embeddings.pkl'
            pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
            logging.info("Saved embedding files to {}".format(embedding_dir))

    def diarize(self, paths2audio_files: List[str] = None, batch_size: int = 0):
        """
        Diarize files provided through paths2audio_files or manifest file
        input:
        paths2audio_files (List[str]): list of paths to file containing audio file
        batch_size (int): batch_size considered for extraction of speaker embeddings and VAD computation
        """

        self._out_dir = self._diarizer_params.out_dir

        self._speaker_dir = os.path.join(self._diarizer_params.out_dir, 'speaker_outputs')

        if os.path.exists(self._speaker_dir):
            logging.warning("Deleting previous clustering diarizer outputs.")
            shutil.rmtree(self._speaker_dir, ignore_errors=True)
        os.makedirs(self._speaker_dir)

        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")

        if batch_size:
            self._cfg.batch_size = batch_size

        if paths2audio_files:
            if type(paths2audio_files) is list:
                self._diarizer_params.manifest_filepath = os.path.join(self._out_dir, 'paths2audio_filepath.json')
                self.path2audio_files_to_manifest(paths2audio_files, self._diarizer_params.manifest_filepath)
            else:
                raise ValueError("paths2audio_files must be of type list of paths to file containing audio file")

        self.AUDIO_RTTM_MAP = audio_rttm_map(self._diarizer_params.manifest_filepath)

        out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)

        # Speech Activity Detection
        self._perform_speech_activity_detection()

        # Segmentation
        scales = self.multiscale_args_dict['scale_dict'].items()
        for scale_idx, (window, shift) in scales:

            # Segmentation for the current scale (scale_idx)
            self._run_segmentation(window, shift, scale_tag=f'_scale{scale_idx}')

            # Embedding Extraction for the current scale (scale_idx)
            self._extract_embeddings(self.subsegments_manifest_path, scale_idx, len(scales))

            self.multiscale_embeddings_and_timestamps[scale_idx] = [self.embeddings, self.time_stamps]

        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )

        # Clustering
        all_reference, all_hypothesis = perform_clustering(
            embs_and_timestamps=embs_and_timestamps,
            AUDIO_RTTM_MAP=self.AUDIO_RTTM_MAP,
            out_rttm_dir=out_rttm_dir,
            clustering_params=self._cluster_params,
            device=self._speaker_model.device,
            verbose=self.verbose,
        )
        logging.info("Outputs are saved in {} directory".format(os.path.abspath(self._diarizer_params.out_dir)))

        # Scoring
        return score_labels(
            self.AUDIO_RTTM_MAP,
            all_reference,
            all_hypothesis,
            collar=self._diarizer_params.collar,
            ignore_overlap=self._diarizer_params.ignore_overlap,
            verbose=self.verbose,
        )

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            tar.add(source_dir, arcname="./")

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into EFF archive or .
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """

        # TODO: Why does this override the main save_to?

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)

            self.to_config_file(path2yaml_file=config_yaml)
            if self.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self._vad_model.save_to(vad_model)
            self._speaker_model.save_to(spkr_model)
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @staticmethod
    def __unpack_nemo_file(path2file: str, out_folder: str) -> str:
        if not os.path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
    ):
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
                else:
                    config_yaml = override_config_path
                conf = OmegaConf.load(config_yaml)
                if os.path.exists(os.path.join(tmpdir, _VAD_MODEL)):
                    conf.diarizer.vad.model_path = os.path.join(tmpdir, _VAD_MODEL)
                else:
                    logging.info(
                        f'Model {cls.__name__} does not contain a VAD model. A VAD model or manifest file with'
                        f'speech segments need for diarization with this model'
                    )

                conf.diarizer.speaker_embeddings.model_path = os.path.join(tmpdir, _SPEAKER_MODEL)
                conf.restore_map_location = map_location
                OmegaConf.set_struct(conf, True)
                instance = cls(cfg=conf)

                logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                os.chdir(cwd)

        return instance

    @property
    def verbose(self) -> bool:
        return self._cfg.verbose
