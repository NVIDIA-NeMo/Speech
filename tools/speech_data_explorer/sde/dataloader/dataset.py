import json
import logging
import multiprocessing as mp
import os
import pickle
from datetime import datetime

from tqdm import tqdm

from .sample import Sample


class Dataset:
    """
    A class for for processing and analyzing speech-related data.

    Parameters:
    - manifest_filepath (str): Path to the manifest file in .json format.
    - chunksize (int): Number of lines to process in each chunk (default is 10000).
    - data_engine (object): An optional object for additional data processing (default is None).
    - n_jobs (int): Number of parallel processes to use for data processing (default is -1, which uses all available CPU cores).
    - reference_field (str): The field in the manifest file representing reference text (default is "text").
    - hypothesis_fields (list[str]): List of fields in the manifest file representing hypothesis texts (default is ["pred_text"]).
    - hypothesis_labels (list[str]): List of labels for hypothesis fields (default is None, and labels are auto-assigned).
    - estimate_audio_metrics (bool): Flag indicating whether to estimate audio metrics (default is False).
    - enable_pkl (bool): Flag indicating whether to enable the use of pickle files for caching processed data (default is True).
    - plk_filepath (str): Path to the pickle file for caching processed data (default is None).
    """

    def __init__(
        self,
        manifest_filepath: str,
        chunksize: int = 10000,
        data_engine: object = None,
        n_jobs: int = -1,
        reference_field="text",
        hypothesis_fields: list[str] = None,
        hypothesis_labels: list[str] = None,
        estimate_audio_metrics: bool = False,
        enable_pkl: bool = True,
        plk_filepath: str = None,
    ):
        self.manifest_filepath = manifest_filepath
        self.chunksize = chunksize
        self.data_engine = data_engine
        self.n_jobs = n_jobs

        max_jobs = mp.cpu_count()
        if self.n_jobs == -1 or n_jobs > max_jobs:
            self.n_jobs = max_jobs

        self.reference_field = reference_field
        self.hypothesis_fields = hypothesis_fields
        self.hypothesis_labels = hypothesis_labels
        self.hypotheses = dict()
        self.estimate_audio_metrics = estimate_audio_metrics
        self.enable_pkl = enable_pkl
        self.plk_filepath = plk_filepath
        self.chunks = []

        self.num_words = 0
        self.num_chars = 0
        self.duration = 0
        self.charset = set()
        self.words_frequencies = dict()

        self.samples_data = []
        self.vocabulary_data = []

    def _check_hypotheses(self, manifest_line: str):
        """
        Internal method to check and validate hypothesis fields and labels based on the manifest file.

        Parameters:
        - manifest_line (str): A line from the manifest file.
        """

        if self.hypothesis_fields is None:
            self.hypothesis_fields = ["pred_text"]

        if self.hypothesis_labels is None:
            if len(self.hypothesis_fields) == 1:
                self.hypothesis_labels = [""]
            else:
                self.hypothesis_labels = list(range(1, len(self.hypothesis_fields) + 1))
        else:
            if len(hypothesis_labels) != len(self.hypothesis_fields):
                logging.error(
                    f"Amount of hypothesis_labels ({len(self.hypothesis_labels)}) is not equal to amount of hypothesis_fields ({len(self.hypothesis_fields)})."
                )
                raise

        sample_to_check = json.loads(manifest_line)

        checked_fields = []
        assigned_labels = []

        for field, label in zip(self.hypothesis_fields, self.hypothesis_labels):
            if field not in sample_to_check:
                logging.warning(f"Field '{field}' not found in sample.")
            else:
                logging.info(f"Field '{field}' was found (labeled as '{label}').")
                checked_fields.append(field)
                assigned_labels.append(label)
                self.hypotheses[field] = HypothesisMetrics(hypothesis_label=label)

        self.hypothesis_fields = checked_fields
        self.hypothesis_labels = assigned_labels

    def _read_manifest(self):
        """
        Internal method to read and process the manifest file, splitting it into chunks.
        """

        logging.info("Reading manifest..")
        with open(self.manifest_filepath, 'r', encoding="utf8") as manifest:
            lines = manifest.readlines()

        self._check_hypotheses(lines[0])

        lines_amount = len(lines)
        logging.info(f"Lines amount: {lines_amount}. Splitting to chunks ({self.chunksize} lines per chunk)..")

        start_chunk_indicies = list(range(0, lines_amount, self.chunksize))
        end_chunk_indicies = list(range(self.chunksize, lines_amount, self.chunksize)) + [lines_amount]

        for start_idx, end_idx in tqdm(zip(start_chunk_indicies, end_chunk_indicies), total=len(start_chunk_indicies)):
            chunk = DataChunk(
                manifest_lines=lines[start_idx:end_idx],
                data_engine=self.data_engine,
                reference_field=self.reference_field,
                hypothesis_fields=self.hypothesis_fields,
                hypothesis_labels=self.hypothesis_labels,
                estimate_audio_metrics=self.estimate_audio_metrics,
            )
            self.chunks.append(chunk)

    def _get_plk_filepath(self):
        """
        Internal method to generate a unique pickle file path based on the manifest file's timestamp.
        """

        timestamp = datetime.fromtimestamp(os.path.getmtime(self.manifest_filepath)).strftime('%Y-%m-%d_%H-%M-%S')
        return f"{self.manifest_filepath.replace('.json', '')}_{timestamp}.pkl"

    def _read_pickle(self):
        """
        Internal method to read data from a pickle file.
        """

        with open(self.plk_filepath, 'rb') as pkl:
            return pickle.load(pkl)

    def _write_pickle(self):
        """
        Internal method to save processed data to a pickle file.
        """

        logging.info(f'Saving .plk file..')
        with open(self.plk_filepath, 'wb') as pkl:
            pickle.dump(self, pkl, pickle.HIGHEST_PROTOCOL)

        logging.info(f'{self.plk_filepath} saved.')

    def process(self):
        """
        Method to process the dataset, including reading the manifest file, splitting into chunks, and computing global metrics.
        """

        if self.enable_pkl:
            logging.info(f'Looking for .plk file ({self.plk_filepath})')
            if self.plk_filepath is None:
                self.plk_filepath = self._get_plk_filepath()

            if os.path.exists(self.plk_filepath):
                logging.info(f'{self.plk_filepath} found.')
                return self._read_pickle()
            else:
                logging.info(f'{self.plk_filepath} not found. Loading from data from manifest..')

        self._read_manifest()

        processed_chunks = []
        logging.info(f'Samples processing ({self.n_jobs} processes)..')
        with mp.Pool(self.n_jobs) as pool:
            for processed_chunk in tqdm(pool.imap(DataChunk.process, self.chunks), total=len(self.chunks)):
                processed_chunks.append(processed_chunk)

        self.chunks = processed_chunks

        logging.info(f'Global metrics computing..')
        for chunk in tqdm(self.chunks):
            self.num_words += chunk.num_words
            self.num_chars += chunk.num_chars
            self.duration += chunk.duration
            self.charset.update(chunk.charset)

            for hypothesis_field in chunk.hypotheses:
                self.hypotheses[hypothesis_field].update(chunk.hypotheses[hypothesis_field])

            for word in chunk.words_frequencies:
                self.words_frequencies[word] = self.words_frequencies.get(word, 0) + chunk.words_frequencies[word]

            if self.data_engine is not None:
                self.samples_data.append(chunk.samples_data)

        for hypothesis_field in self.hypotheses:
            self.hypotheses[hypothesis_field].compute(
                dataset_num_words=self.num_words, dataset_num_chars=self.num_chars
            )

        self.duration = round(self.duration / 3600, 2)

        if self.data_engine is not None:
            logging.info(f'Samples datatable loading..')
            self.samples_data = self.data_engine.concat_samples_chunks(self.samples_data)
            self.vocabulary_data = self.data_engine.process_vocabulary(
                words_frequencies=self.words_frequencies, hypotheses_metrics=self.hypotheses.values()
            )

        if self.enable_pkl:
            self._write_pickle()

        return self


class DataChunk:
    """
    A class representing a chunk of data within a larger dataset (class Dataset) for processing and analyzing speech-related data.

    Parameters:
    - manifest_lines (list[str]): List of lines from the manifest file containing information about the chunk.
    - data_engine (object): An optional object for additional data processing (default is None).
    - reference_field (str): The field in the manifest file representing reference text (default is "text").
    - hypothesis_fields (list[str]): List of fields in the manifest file representing hypothesis texts (default is ["pred_text"]).
    - hypothesis_labels (list[str]): List of labels for hypothesis fields (default is None).
    - estimate_audio_metrics (bool): Flag indicating whether to estimate audio metrics (default is False).
    """

    def __init__(
        self,
        manifest_lines: list[str],
        data_engine: object = None,
        reference_field: str = "text",
        hypothesis_fields: list[str] = ["pred_text"],
        hypothesis_labels: list[str] = None,
        estimate_audio_metrics: bool = False,
    ):
        self.manifest_lines = manifest_lines
        self.reference_field = reference_field
        self.estimate_audio_metrics = estimate_audio_metrics
        self.samples_dicts = []
        self.num_words = 0
        self.num_chars = 0
        self.duration = 0
        self.charset = set()
        self.words_frequencies = dict()

        self.hypothesis_fields = hypothesis_fields
        self.hypothesis_labels = hypothesis_labels
        self.hypotheses = dict()

        for field, label in zip(hypothesis_fields, hypothesis_labels):
            self.hypotheses[field] = HypothesisMetrics(hypothesis_label=label)

        self.data_engine = data_engine
        self.samples_data = None

    def process(self):
        """
        Method to process the data chunk, including parsing manifest lines, computing metrics, and handling data engine interactions.
        """

        sample = Sample()
        for manifest_line in self.manifest_lines:
            sample.parse_line(
                manifest_line,
                reference_field=self.reference_field,
                hypothesis_fields=self.hypothesis_fields,
                hypothesis_labels=self.hypothesis_labels,
            )
            sample.compute(estimate_audio_metrics=self.estimate_audio_metrics)

            self.samples_dicts.append(sample.sample_dict)
            self.num_words += sample.num_words
            self.num_chars += sample.num_chars
            self.duration += sample.duration
            self.charset.update(sample.charset)

            for word in sample.words_frequencies:
                self.words_frequencies[word] = self.words_frequencies.get(word, 0) + sample.words_frequencies[word]

            for hypothesis_field in sample.hypotheses:
                self.hypotheses[hypothesis_field].update(sample.hypotheses[hypothesis_field])

            sample.reset()

        if self.data_engine is not None:
            self.samples_data = self.data_engine.load_samples_chunk(self.samples_dicts)
            self.samples_dicts = {}

        return self


class HypothesisMetrics:
    """
    A class representing metrics for evaluating hypotheses in speech-related data.

    Parameters:
    - hypothesis_label (str): Custom label associated with the hypothesis of some ASR-model (default is None).
    """

    def __init__(self, hypothesis_label: str = None):
        self.hypothesis_label = hypothesis_label
        self.word_distance = 0
        self.word_match = 0
        self.char_distance = 0

        self.wer = None
        self.wmr = None
        self.cer = None
        self.mwa = None

        self.match_words_frequencies = dict()

    def update(self, hypothesis: object):
        """
        Update the metrics based on object of HypothesisMetrics or Hypothesis classes.

        Parameters:
        - hypothesis (object): Another HypothesisMetrics object to update metrics.
        """

        assert self.hypothesis_label == hypothesis.hypothesis_label, "Hypothesis label mismatch!"

        self.word_distance += hypothesis.word_distance
        self.word_match += hypothesis.word_match
        self.char_distance += hypothesis.char_distance

        for word in hypothesis.match_words_frequencies:
            self.match_words_frequencies[word] = (
                self.match_words_frequencies.get(word, 0) + hypothesis.match_words_frequencies[word]
            )

    def compute(self, dataset_num_words: int, dataset_num_chars: int):
        """
        Compute final match metrics (WER, WMR, CER) based on the dataset's total number of words and characters.

        Parameters:
        - dataset_num_words (int): Total number of words in the dataset.
        - dataset_num_chars (int): Total number of characters in the dataset.
        """

        self.wer = round(self.word_distance / dataset_num_words * 100.0, 2)
        self.wmr = round(self.word_match / dataset_num_words * 100.0, 2)
        self.cer = round(self.char_distance / dataset_num_chars * 100.0, 2)
