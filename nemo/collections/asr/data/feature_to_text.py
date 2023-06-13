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

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from nemo.collections.asr.data.feature_to_label import _audio_feature_collate_fn
from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.parts.utils.vad_utils import load_speech_segments_from_rttm
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset
from nemo.core.neural_types import AcousticEncodedRepresentation, LabelsType, LengthsType, NeuralType


class ASRFeatureManifestProcessor:
    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
    ):
        self.parser = parser
        self.collection = collections.ASRFeatureText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        sample = self.collection[index]
        return self.process_text_by_sample(sample)

    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)

    def process_text_by_sample(self, sample: collections.ASRAudioText.OUTPUT_TYPE) -> Tuple[List[int], int]:
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


class _FeatureTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio feature files, transcripts,
    durations (in seconds) and optional RTTM files. Each new line is a different sample. Example below:
    {"feature_filepath": "/path/to/audio_feature.pt", "text_filepath": "/path/to/audio.txt", 
    "rttm_filepath": "/path/to/audio_rttm.rttm", "duration": 23.147}
    ...
    {"feature_filepath": "/path/to/audio_feature.pt", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath (str): Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        normalize (bool): whether and where to normalize feature, must be one of [None, "post_norm", "pre_norm"]
        normalize_type (Union[str, dict]): how to normalize feature, see `nemo.collections.asr.parts.preprocessing.features.normalize_batch`
        use_rttm (bool): whether to use RTTM files if there is any, default to False
        rttm_mode (str): how to use RTTM files, must be one of ['mask', 'drop'], default to 'mask'
        feat_min_len (int): minimum length of feature when rttm_mode=deop, default to 4.
        feat_mask_val (Optional[float]): value used to mask features with RTTM files, default to None to use zero mel-spectralgram
        frame_unit_time_secs (float): time in seconds for each frame
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded audio
        max_duration (float): If audio exceeds this length, do not include in dataset
        min_duration (float): If audio is less than this length, do not include in dataset
        max_utts (int): Limit number of utterances
        trim (bool): whether or not to trim silence. Defaults to False
        bos_id (int): Id of beginning of sequence symbol to append if not None
        eos_id (int): Id of end of sequence symbol to append if not None
        pad_id (int): Id of pad symbol. Defaults to 0
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
    """

    ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
    NORM_MODES = ["pre_norm", "post_norm"]
    RTTM_MODES = ["mask", "drop"]

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'features': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'feature_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        normalize: Optional[str] = "post_norm",
        normalize_type: Union[str, dict] = "per_feature",
        use_rttm: bool = False,
        rttm_mode: str = "mask",
        feat_min_len: int = 4,
        feat_mask_val: Optional[float] = None,
        frame_unit_time_secs: float = 0.01,
        sample_rate: Optional[int] = 16000,
        augmentor: 'nemo.collections.asr.parts.perturb.FeatureAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.sample_rate = sample_rate
        self.normalize = normalize
        self.normalize_type = normalize_type
        self.use_rttm = use_rttm
        self.rttm_mode = rttm_mode
        if self.use_rttm and self.rttm_mode not in self.RTTM_MODES:
            raise ValueError(f"`rttm_mode` must be one of {self.RTTM_MODES}, got `{rttm_mode}` instead")

        self.feat_min_len = feat_min_len
        if feat_mask_val is not None:
            self.feat_mask_val = feat_mask_val
        elif normalize == "pre_norm":
            self.feat_mask_val = 0.0  # similar to SpectralAugmentation
        else:
            self.feat_mask_val = self.ZERO_LEVEL_SPEC_DB_VAL

        if normalize is not None and normalize not in self.NORM_MODES:
            raise ValueError(f"`normalize` must be one of {self.NORM_MODES}, got `{normalize}` instead")

        self.frame_unit_time_secs = frame_unit_time_secs

        self.manifest_processor = ASRFeatureManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = ExternalFeatureLoader(augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.feature_file)

        f, fl = features, torch.tensor(features.shape[1]).long()

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        # Feature normalization
        if self.normalize is None:
            if self.use_rttm and sample.rttm_file:
                f = self.process_features_with_rttm(f, offset, sample.rttm_file, self.feat_mask_val)
        elif self.normalize == "post_norm":
            # (Optional) Masking based on RTTM file
            if self.use_rttm and sample.rttm_file:
                f = self.process_features_with_rttm(f, offset, sample.rttm_file, self.feat_mask_val)

            f = self.normalize_feature(f)
        else:  # pre-norm
            f = self.normalize_feature(f)
            # (Optional) Masking based on RTTM file
            if self.use_rttm and sample.rttm_file:
                f = self.process_features_with_rttm(f, offset, sample.rttm_file, self.feat_mask_val)

        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output

    def process_features_with_rttm(self, features, offset, rttm_file, mask_val):
        segments = load_speech_segments_from_rttm(rttm_file)
        new_features = features.clone()
        sid, fid = 0, 0
        for i in range(features.size(1)):
            t = offset + i * self.frame_unit_time_secs
            while sid < len(segments) - 1 and segments[sid][1] < t:
                sid += 1
            if segments[sid][1] == 0 or t < segments[sid][0] or t > segments[sid][1]:
                # not in speech segment
                if self.rttm_mode == "drop":
                    # drop the frame
                    continue
                else:
                    # mask the frame with specified value
                    new_features[:, i] = mask_val
                    fid += 1
            else:
                # in speech segment
                new_features[:, fid] = features[:, i]
                fid += 1

        if fid < self.feat_min_len and self.rttm_mode == "drop":
            new_features[:, : self.feat_min_len] = mask_val
            return new_features[:, : self.feat_min_len]
        return new_features[:, :fid]

    def __len__(self):
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _audio_feature_collate_fn(
            batch, feat_pad_val=self.feat_mask_val, label_pad_id=self.manifest_processor.pad_id
        )

    def normalize_feature(self, feat):
        """
        Args:
            feat: feature tensor of shape [M, T]            
        """
        feat = feat.unsqueeze(0)  # add batch dim
        feat, _, _ = normalize_batch(feat, torch.tensor([feat.size(-1)]), self.normalize_type)
        return feat.squeeze(0)  # delete batch dim


class FeatureToCharDataset(_FeatureTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio feature
    files, transcripts, durations (in seconds) and optional RTTM files. Each new line is a
    different sample. Example below:
    {"feature_filepath": "/path/to/audio_feature.pt", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147, "rttm_filepath": "/path/to/audio_rttm.rttm",}
    ...
    {"feature_filepath": "/path/to/audio_feature.pt", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath (str): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (str): String containing all the possible characters to map to
        normalize (str): how to normalize feature, must be one of [None, "post_norm", "pre_norm"]
        normalize_type (Union[str, dict]): how to normalize feature, see `nemo.collections.asr.parts.preprocessing.features.normalize_batch`
        use_rttm (bool): whether to use RTTM files if there is any, default to False
        rttm_mode (str): how to use RTTM files, must be one of ['mask', 'drop'], default to 'mask'
        feat_min_len (int): minimum length of feature, default to 4
        feat_mask_val (Optional[float]): value used to mask features with RTTM files, default to None to use zero mel-spectralgram
        frame_unit_time_secs: time in seconds for each frame
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
    """

    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        normalize: Optional[str] = "post_norm",
        normalize_type: Union[str, dict] = "per_feature",
        use_rttm: bool = False,
        rttm_mode: str = "mask",
        feat_min_len: int = 4,
        feat_mask_val: Optional[float] = None,
        frame_unit_time_secs: float = 0.01,
        sample_rate: Optional[int] = 16000,
        augmentor: 'nemo.collections.asr.parts.perturb.FeatureAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = 'en',
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            normalize=normalize,
            normalize_type=normalize_type,
            use_rttm=use_rttm,
            rttm_mode=rttm_mode,
            feat_min_len=feat_min_len,
            feat_mask_val=feat_mask_val,
            frame_unit_time_secs=frame_unit_time_secs,
            sample_rate=sample_rate,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
        )


class FeatureToBPEDataset(_FeatureTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio feature
    files, transcripts, durations (in seconds) and optional RTTM files. Each new line is a different sample. 
    Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147, "rttm_filepath": "/path/to/audio_rttm.rttm",}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    In practice, the dataset and manifest used for character encoding and byte pair encoding
    are exactly the same. The only difference lies in how the dataset tokenizes the text in
    the manifest.

    Args:
        manifest_filepath (str): Path to manifest json as described above. Can
            be comma-separated paths.
        tokenizer: A subclass of the Tokenizer wrapper found in the common collection,
            nemo.collections.common.tokenizers.TokenizerSpec. ASR Models support a subset of
            all available tokenizers.
        normalize (str): how to normalize feature, must be one of [None, "post_norm", "pre_norm"]
        normalize_type (Union[str, dict]): how to normalize feature, see `nemo.collections.asr.parts.preprocessing.features.normalize_batch`
        use_rttm (bool): whether to use RTTM files if there is any, default to False
        rttm_mode (str): how to use RTTM files, must be one of ['mask', 'drop'], default to 'mask'
        feat_min_len (int): minimum length of feature, default to 4
        feat_mask_val (Optional[float]): value used to mask features with RTTM files, default to None to use zero mel-spectralgram
        frame_unit_time_secs: time in seconds for each frame
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        trim: Whether to trim silence segments
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        normalize: Optional[str] = "post_norm",
        normalize_type: Union[str, dict] = "per_feature",
        use_rttm: bool = False,
        rttm_mode: str = "mask",
        feat_min_len: int = 4,
        feat_mask_val: Optional[float] = None,
        frame_unit_time_secs: float = 0.01,
        sample_rate: Optional[int] = 16000,
        augmentor: 'nemo.collections.asr.parts.perturb.FeatureAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        use_start_end_token: bool = True,
        trim: bool = False,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
                    return t

                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            normalize=normalize,
            normalize_type=normalize_type,
            use_rttm=use_rttm,
            rttm_mode=rttm_mode,
            feat_min_len=feat_min_len,
            feat_mask_val=feat_mask_val,
            frame_unit_time_secs=frame_unit_time_secs,
            sample_rate=sample_rate,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
        )
