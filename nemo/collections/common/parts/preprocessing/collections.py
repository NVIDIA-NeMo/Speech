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

import collections
import json
import os
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.utils import logging


class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class Text(_Collection):
    """Simple list of preprocessed text entries, result in list of tokens."""

    OUTPUT_TYPE = collections.namedtuple('TextEntity', 'tokens')

    def __init__(self, texts: List[str], parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            texts: List of raw texts strings.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        data, output_type = [], self.OUTPUT_TYPE
        for text in texts:
            tokens = parser(text)

            if tokens is None:
                logging.warning("Fail to parse '%s' text line.", text)
                continue

            data.append(output_type(tokens))

        super().__init__(data)


class FromFileText(Text):
    """Another form of texts manifest with reading from file."""

    def __init__(self, file: str, parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            file: File path to read from.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        texts = self.__parse_texts(file)

        super().__init__(texts, parser)

    @staticmethod
    def __parse_texts(file: str) -> List[str]:
        if not os.path.exists(file):
            raise ValueError('Provided texts file does not exists!')

        _, ext = os.path.splitext(file)
        if ext == '.csv':
            texts = pd.read_csv(file)['transcript'].tolist()
        elif ext == '.json':  # Not really a correct json.
            texts = list(item['text'] for item in manifest.item_iter(file))
        else:
            with open(file, 'r') as f:
                texts = f.readlines()

        return texts


class AudioText(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='AudioTextEntity',
        field_names='id audio_file duration text_tokens offset text_raw speaker orig_sr lang',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        token_labels: List[Optional[int]],
        langs: List[Optional[str]],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.

        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            durations: List of float durations.
            texts: List of raw text transcripts.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            langs: List of language ids, one for eadh sample, or None.
            parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, duration, offset, text, speaker, orig_sr, token_labels, lang in zip(
            ids, audio_files, durations, offsets, texts, speakers, orig_sampling_rates, token_labels, langs
        ):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if token_labels is not None:
                text_tokens = token_labels
            else:
                if text != '':
                    if hasattr(parser, "is_aggregate") and parser.is_aggregate and isinstance(text, str):
                        if lang is not None:
                            text_tokens = parser(text, lang)
                        else:
                            raise ValueError("lang required in manifest when using aggregate tokenizers")
                    else:
                        text_tokens = parser(text)
                else:
                    text_tokens = []

                if text_tokens is None:
                    duration_filtered += duration
                    num_filtered += 1
                    continue

            total_duration += duration

            data.append(output_type(id_, audio_file, duration, text_tokens, offset, text, speaker, orig_sr, lang))
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                if file_id not in self.mapping:
                    self.mapping[file_id] = []
                self.mapping[file_id].append(len(data) - 1)

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class ASRAudioText(AudioText):
    """`AudioText` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """

        ids, audio_files, durations, texts, offsets, = (
            [],
            [],
            [],
            [],
            [],
        )
        speakers, orig_srs, token_labels, langs = [], [], [], []
        for item in manifest.item_iter(manifests_files):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            texts.append(item['text'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])
            token_labels.append(item['token_labels'])
            langs.append(item['lang'])
        super().__init__(
            ids, audio_files, durations, texts, offsets, speakers, orig_srs, token_labels, langs, *args, **kwargs
        )


class SpeechLabel(_Collection):
    """List of audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='SpeechLabelEntity', field_names='audio_file duration label offset',)

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        labels: List[Union[int, str]],
        offsets: List[Optional[float]],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-label manifest with filters and preprocessing.

        Args:
            audio_files: List of audio files.
            durations: List of float durations.
            labels: List of labels.
            offsets: List of offsets or None.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        for audio_file, duration, command, offset in zip(audio_files, durations, labels, offsets):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, command, offset))

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        logging.info("# {} files loaded accounting to # {} labels".format(len(data), len(self.uniq_labels)))

        super().__init__(data)


class ASRSpeechLabel(SpeechLabel):
    """`SpeechLabel` collector from structured json files."""

    def __init__(
        self,
        manifests_files: Union[str, List[str]],
        is_regression_task=False,
        cal_labels_occurrence=False,
        *args,
        **kwargs,
    ):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            is_regression_task: It's a regression task.
            cal_labels_occurrence: whether to calculate occurence of labels.
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        audio_files, durations, labels, offsets = [], [], [], []

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            if not is_regression_task:
                label = item['label']
            else:
                label = float(item['label'])

            labels.append(label)
            offsets.append(item['offset'])

        if cal_labels_occurrence:
            self.labels_occurrence = collections.Counter(labels)

        super().__init__(audio_files, durations, labels, offsets, *args, **kwargs)

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])

        # Duration.
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")

        # Label.
        if 'command' in item:
            item['label'] = item.pop('command')
        elif 'target' in item:
            item['label'] = item.pop('target')
        elif 'label' in item:
            pass
        else:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper label key.")

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            label=item['label'],
            offset=item.get('offset', None),
        )

        return item


class FeatureSequenceLabel(_Collection):
    """List of feature sequence of label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='FeatureSequenceLabelEntity', field_names='feature_file seq_label',)

    def __init__(
        self,
        feature_files: List[str],
        seq_labels: List[str],
        max_number: Optional[int] = None,
        index_by_file_id: bool = False,
    ):
        """Instantiates feature-SequenceLabel manifest with filters and preprocessing.

        Args:
            feature_files: List of feature files.
            seq_labels: List of sequences of abels.
            max_number: Maximum number of samples to collect.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, num_filtered = (
            [],
            0.0,
        )
        self.uniq_labels = set()

        if index_by_file_id:
            self.mapping = {}

        for feature_file, seq_label in zip(feature_files, seq_labels):

            label_tokens, uniq_labels_in_seq = self.relative_speaker_parser(seq_label)

            data.append(output_type(feature_file, label_tokens))
            self.uniq_labels |= uniq_labels_in_seq

            if label_tokens is None:
                num_filtered += 1
                continue

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(feature_file))
                self.mapping[feature_file] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        logging.info("# {} files loaded including # {} unique labels".format(len(data), len(self.uniq_labels)))
        super().__init__(data)

    def relative_speaker_parser(self, seq_label):
        """Convert sequence of speaker labels to relative labels.
        Convert sequence of absolute speaker to sequence of relative speaker [E A C A E E C] -> [0 1 2 1 0 0 2]
        In this seq of label , if label do not appear before, assign new relative labels len(pos); else reuse previous assigned relative labels.
        Args:
            seq_label (str): A string of a sequence of labels.

        Return:
            relative_seq_label (List) : A list of relative sequence of labels
            unique_labels_in_seq (Set): A set of unique labels in the sequence
        """
        seq = seq_label.split()
        conversion_dict = dict()
        relative_seq_label = []

        for seg in seq:
            if seg in conversion_dict:
                converted = conversion_dict[seg]
            else:
                converted = len(conversion_dict)
                conversion_dict[seg] = converted

            relative_seq_label.append(converted)

        unique_labels_in_seq = set(conversion_dict.keys())
        return relative_seq_label, unique_labels_in_seq


class ASRFeatureSequenceLabel(FeatureSequenceLabel):
    """`FeatureSequenceLabel` collector from asr structured json files."""

    def __init__(
        self, manifests_files: Union[str, List[str]], max_number: Optional[int] = None, index_by_file_id: bool = False,
    ):

        """Parse lists of feature files and sequences of labels.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            max_number:  Maximum number of samples to collect; pass to `FeatureSequenceLabel` constructor.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data; pass to `FeatureSequenceLabel` constructor.
        """

        feature_files, seq_labels = [], []
        for item in manifest.item_iter(manifests_files, parse_func=self._parse_item):
            feature_files.append(item['feature_file'])
            seq_labels.append(item['seq_label'])

        super().__init__(feature_files, seq_labels, max_number, index_by_file_id)

    def _parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Feature file
        if 'feature_filename' in item:
            item['feature_file'] = item.pop('feature_filename')
        elif 'feature_filepath' in item:
            item['feature_file'] = item.pop('feature_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper feature file key."
            )
        item['feature_file'] = os.path.expanduser(item['feature_file'])

        # Seq of Label.
        if 'seq_label' in item:
            item['seq_label'] = item.pop('seq_label')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper seq_label key."
            )

        item = dict(feature_file=item['feature_file'], seq_label=item['seq_label'],)

        return item


class DiarizationLabel(_Collection):
    """List of diarization audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='DiarizationLabelEntity',
        field_names='audio_file duration rttm_file offset target_spks sess_spk_dict clus_spk_digits rttm_spk_digits',
    )

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        rttm_files: List[str],
        offsets: List[float],
        target_spks_list: List[tuple],
        sess_spk_dicts: List[Dict],
        clus_spk_list: List[tuple],
        rttm_spk_list: List[tuple],
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-label manifest with filters and preprocessing.

        Args:
            audio_files:
                List of audio file paths.
            durations:
                List of float durations.
            rttm_files:
                List of RTTM files (Groundtruth diarization annotation file).
            offsets:
                List of offsets or None.
            target_spks (tuple):
                List of tuples containing the two indices of targeted speakers for evaluation.
                Example: [[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [(0, 1), (1, 2), (0, 2)], ...]
            sess_spk_dict (Dict):
                List of Mapping dictionaries between RTTM speakers and speaker labels in the clustering result.
            clus_spk_digits (tuple):
                List of Tuple containing all the speaker indices from the clustering result.
                Example: [(0, 1, 2, 3), (0, 1, 2), ...]
            rttm_spkr_digits (tuple):
                List of tuple containing all the speaker indices in the RTTM file.
                Example: (0, 1, 2), (0, 1), ...]
            max_number: Maximum number of samples to collect
            do_sort_by_duration: True if sort samples list by duration
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0

        zipped_items = zip(
            audio_files, durations, rttm_files, offsets, target_spks_list, sess_spk_dicts, clus_spk_list, rttm_spk_list
        )
        for (
            audio_file,
            duration,
            rttm_file,
            offset,
            target_spks,
            sess_spk_dict,
            clus_spk_digits,
            rttm_spk_digits,
        ) in zipped_items:

            if duration is None:
                duration = 0

            data.append(
                output_type(
                    audio_file,
                    duration,
                    rttm_file,
                    offset,
                    target_spks,
                    sess_spk_dict,
                    clus_spk_digits,
                    rttm_spk_digits,
                )
            )

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        logging.info(f"Total {len(data)} session files loaded accounting to # {len(audio_files)} audio clips")

        super().__init__(data)


class DiarizationSpeechLabel(DiarizationLabel):
    """`DiarizationLabel` diarization data sample collector from structured json files."""

    def __init__(
        self,
        manifests_files: Union[str, List[str]],
        emb_dict: Dict,
        clus_label_dict: Dict,
        round_digit=2,
        seq_eval_mode=False,
        pairwise_infer=False,
        *args,
        **kwargs,
    ):
        """
        Parse lists of audio files, durations, RTTM (Diarization annotation) files. Since diarization model infers only
        two speakers, speaker pairs are generated from the total number of speakers in the session.

        Args:
            manifest_filepath (str):
                Path to input manifest json files.
            emb_dict (Dict):
                Dictionary containing cluster-average embeddings and speaker mapping information.
            clus_label_dict (Dict):
                Segment-level speaker labels from clustering results.
            round_digit (int):
                Number of digits to be rounded.
            seq_eval_mode (bool):
                If True, F1 score will be calculated for each speaker pair during inference mode.
            pairwise_infer (bool):
                If True, this dataset class operates in inference mode. In inference mode, a set of speakers in the input audio
                is split into multiple pairs of speakers and speaker tuples (e.g. 3 speakers: [(0,1), (1,2), (0,2)]) and then
                fed into the diarization system to merge the individual results.
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        self.round_digit = round_digit
        self.emb_dict = emb_dict
        self.clus_label_dict = clus_label_dict
        self.seq_eval_mode = seq_eval_mode
        self.pairwise_infer = pairwise_infer
        audio_files, durations, rttm_files, offsets, target_spks_list, sess_spk_dicts, clus_spk_list, rttm_spk_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item_rttm):
            # Inference mode
            if self.pairwise_infer:
                clus_speaker_digits = sorted(list(set([x[2] for x in clus_label_dict[item['uniq_id']]])))
                if item['rttm_file']:
                    base_scale_index = max(self.emb_dict.keys())
                    _sess_spk_dict = self.emb_dict[base_scale_index][item['uniq_id']]['mapping']
                    sess_spk_dict = {int(v.split('_')[-1]): k for k, v in _sess_spk_dict.items()}
                    rttm_speaker_digits = [int(v.split('_')[1]) for k, v in _sess_spk_dict.items()]
                    if self.seq_eval_mode:
                        clus_speaker_digits = rttm_speaker_digits
                else:
                    sess_spk_dict = None
                    rttm_speaker_digits = None

            # Training mode
            else:
                rttm_labels = []
                with open(item['rttm_file'], 'r') as f:
                    for line in f.readlines():
                        start, end, speaker = self.split_rttm_line(line, decimals=3)
                        rttm_labels.append('{} {} {}'.format(start, end, speaker))
                speaker_set = set()
                for rttm_line in rttm_labels:
                    spk_str = rttm_line.split()[-1]
                    speaker_set.add(spk_str)
                speaker_list = sorted(list(speaker_set))
                sess_spk_dict = {key: val for key, val in enumerate(speaker_list)}
                target_spks = tuple(sess_spk_dict.keys())
                clus_speaker_digits = target_spks
                rttm_speaker_digits = target_spks

            if len(clus_speaker_digits) <= 2:
                spk_comb_list = [(0, 1)]
            else:
                spk_comb_list = [x for x in combinations(clus_speaker_digits, 2)]

            for target_spks in spk_comb_list:
                audio_files.append(item['audio_file'])
                durations.append(item['duration'])
                rttm_files.append(item['rttm_file'])
                offsets.append(item['offset'])
                target_spks_list.append(target_spks)
                sess_spk_dicts.append(sess_spk_dict)
                clus_spk_list.append(clus_speaker_digits)
                rttm_spk_list.append(rttm_speaker_digits)

        super().__init__(
            audio_files,
            durations,
            rttm_files,
            offsets,
            target_spks_list,
            sess_spk_dicts,
            clus_spk_list,
            rttm_spk_list,
            *args,
            **kwargs,
        )

    def split_rttm_line(self, rttm_line: str, decimals: int = 3):
        """
        Convert a line in RTTM file to speaker label, start and end timestamps.

        An example line of `rttm_line`:
            SPEAKER abc_dev_0123 1 146.903 1.860 <NA> <NA> speaker543 <NA> <NA>

        The above example RTTM line contains the following information:
            session name: abc_dev_0123
            segment start time: 146.903
            segment duration: 1.860
            speaker label: speaker543

        Args:
            rttm_line (str):
                A line in RTTM formatted file containing offset and duration of each segment.
            decimals (int):
                Number of digits to be rounded.

        Returns:
            start (float):
                Start timestamp in floating point number.
            end (float):
                End timestamp in floating point number.
            speaker (str):
                speaker string in RTTM lines.
        """
        rttm = rttm_line.strip().split()
        start = round(float(rttm[3]), decimals)
        end = round(float(rttm[4]), decimals) + round(float(rttm[3]), decimals)
        speaker = rttm[7]
        return start, end, speaker

    def __parse_item_rttm(self, line: str, manifest_file: str) -> Dict[str, Any]:
        """Parse each rttm file and save it to in Dict format"""
        item = json.loads(line)
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])
        item['uniq_id'] = os.path.splitext(os.path.basename(item['audio_file']))[0]
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")
        item = dict(
            audio_file=item['audio_file'],
            uniq_id=item['uniq_id'],
            duration=item['duration'],
            rttm_file=item['rttm_filepath'],
            offset=item.get('offset', None),
        )
        return item


class Audio(_Collection):
    """Prepare a list of all audio items, filtered by duration.
    """

    OUTPUT_TYPE = collections.namedtuple(typename='Audio', field_names='audio_files duration offset text')

    def __init__(
        self,
        audio_files_list: List[Dict[str, str]],
        duration_list: List[float],
        offset_list: List[float],
        text_list: List[str],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
    ):
        """Instantiantes an list of audio files.

        Args:
            audio_files_list: list of dictionaries with mapping from audio_key to audio_filepath
            duration_list: list of durations of input files
            offset_list: list of offsets
            text_list: list of texts
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration.
        """

        output_type = self.OUTPUT_TYPE
        data, total_duration = [], 0.0
        num_filtered, duration_filtered = 0, 0.0

        for audio_files, duration, offset, text in zip(audio_files_list, duration_list, offset_list, text_list):
            # Duration filters
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            total_duration += duration
            data.append(output_type(audio_files, duration, offset, text))

            # Max number of entities filter
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class AudioCollection(Audio):
    """List of audio files from a manifest file. 
    """

    def __init__(
        self, manifest_files: Union[str, List[str]], audio_to_manifest_key: Dict[str, str], *args, **kwargs,
    ):
        """Instantiates a list of audio files loaded from a manifest file.

        Args:
            manifest_files: path to a single manifest file or a list of paths
            audio_to_manifest_key: dictionary mapping audio signals to keys of the manifest
        """
        # Support for comma-separated manifests
        if type(manifest_files) == str:
            manifest_files = manifest_files.split(',')

        for audio_key, manifest_key in audio_to_manifest_key.items():
            # Support for comma-separated keys
            if type(manifest_key) == str and ',' in manifest_key:
                audio_to_manifest_key[audio_key] = manifest_key.split(',')

        # Keys from manifest which contain audio
        self.audio_to_manifest_key = audio_to_manifest_key

        # Initialize data
        audio_files_list, duration_list, offset_list, text_list = [], [], [], []

        # Parse manifest files
        for item in manifest.item_iter(manifest_files, parse_func=self.__parse_item):
            audio_files_list.append(item['audio_files'])
            duration_list.append(item['duration'])
            offset_list.append(item['offset'])
            text_list.append(item['text'])

        super().__init__(audio_files_list, duration_list, offset_list, text_list, *args, **kwargs)

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        """Parse a single line from a manifest file.

        Args:
            line: a string representing a line from a manifest file in JSON format
            manifest_file: path to the manifest file. Used to resolve relative paths.

        Returns:
            Dictionary with audio_files, duration, and offset.
        """
        # Local utility function
        def get_audio_file(item: Dict, manifest_key: Union[str, List[str]]):
            """Get item[key] if key is string, or a list
            of strings by combining item[key[0]], item[key[1]], etc.
            """
            # Prepare audio file(s)
            if manifest_key is None:
                # Support for inference, when a target key is None
                audio_file = None
            elif isinstance(manifest_key, str):
                # Load files from a single manifest key
                audio_file = item[manifest_key]
            elif isinstance(manifest_key, Iterable):
                # Load files from multiple manifest keys
                audio_file = []
                for key in manifest_key:
                    item_key = item[key]
                    if isinstance(item_key, str):
                        audio_file.append(item_key)
                    elif isinstance(item_key, list):
                        audio_file += item_key
                    else:
                        raise ValueError(f'Unexpected type {type(item_key)} of item for key {key}: {item_key}')
            else:
                raise ValueError(f'Unexpected type {type(manifest_key)} of manifest_key: {manifest_key}')

            return audio_file

        # Convert JSON line to a dictionary
        item = json.loads(line)

        # Handle all audio files
        audio_files = {}
        for audio_key, manifest_key in self.audio_to_manifest_key.items():

            audio_file = get_audio_file(item, manifest_key)

            # Get full path to audio file(s)
            if isinstance(audio_file, str):
                # This dictionary entry points to a single file
                audio_files[audio_key] = manifest.get_full_path(audio_file, manifest_file)
            elif isinstance(audio_file, Iterable):
                # This dictionary entry points to multiple files
                # Get the files and keep the list structure for this key
                audio_files[audio_key] = [manifest.get_full_path(f, manifest_file) for f in audio_file]
            elif audio_file is None and audio_key.startswith('target'):
                # For inference, we don't need the target
                audio_files[audio_key] = None
            else:
                raise ValueError(f'Unexpected type {type(audio_file)} of audio_file: {audio_file}')
        item['audio_files'] = audio_files

        # Handle duration
        if 'duration' not in item:
            raise ValueError(f'Duration not available in line: {line}. Manifest file: {manifest_file}')

        # Handle offset
        if 'offset' not in item:
            item['offset'] = 0.0

        # Handle text
        if 'text' not in item:
            item['text'] = None

        return dict(
            audio_files=item['audio_files'], duration=item['duration'], offset=item['offset'], text=item['text']
        )
