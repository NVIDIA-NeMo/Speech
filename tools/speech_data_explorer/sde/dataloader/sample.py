import json
from collections import Counter
from difflib import SequenceMatcher

import editdistance
import jiwer
import librosa
import numpy as np


class Sample:
    """
    A class representing a sample of data, including reference and hypothesis texts, for processing and analysis.

    Attributes:
    - reference_text (str): The reference text associated with the sample.
    - num_chars (int): Number of characters in the reference text.
    - charset (set): Set of unique characters in the reference text.
    - words (list): List of words in the reference text.
    - num_words (int): Number of words in the reference text.
    - words_frequencies (dict): Dictionary containing word frequencies in the reference text.
    - duration (float): Duration of the audio in the sample.
    - frequency_bandwidth (float): Frequency bandwidth of the audio signal (computed if audio file provided).
    - level_db (float): Level of the audio signal in decibels (computed if audio file provided).
    - hypotheses (dict): Dictionary containing hypothesis objects for different fields.

    Methods:
    - reset():
        Resets the sample attributes to their initial state.

    - parse_line(manifest_line: str, reference_field: str = "text",
                 hypothesis_fields: list[str] = ["pred_text"],
                 hypothesis_labels: list[str] = None):
        Parses a line from the manifest file and updates the sample information.

    - compute(estimate_audio_metrics: bool = False):
        Computes metrics for the sample, including word frequencies and audio metrics if specified.

    - add_table_metrics_to_dict():
        Adds computed metrics to the sample dictionary.
    """

    def __init__(self):
        self.reference_text = None
        self.audio_filepath = None
        self.num_chars = None
        self.charset = set()
        self.words = None
        self.num_words = None
        self.words_frequencies = None
        self.duration = None
        self.frequency_bandwidth = None
        self.level_db = None
        self.hypotheses = {}

    def reset(self):
        """
        Resets the sample attributes to their initial state.
        """

        self.reference_text = None
        self.num_chars = None
        self.charset = set()
        self.words = None
        self.num_words = None
        self.words_frequencies = None
        self.duration = None
        self.frequency_bandwidth = None
        self.level_db = None
        self.hypotheses = {}

    def parse_line(
        self,
        manifest_line: str,
        reference_field: str = "text",
        hypothesis_fields: list[str] = ["pred_text"],
        hypothesis_labels: list[str] = None,
    ):
        """
        Parses a line from the manifest file and updates the sample information.
        """

        self.sample_dict = json.loads(manifest_line)
        self.reference_text = self.sample_dict.get(reference_field, None)
        self.duration = self.sample_dict.get("duration", None)
        self.audio_filepath = self.sample_dict.get("audio_filepath", None)

        if hypothesis_labels is None:
            hypothesis_labels = list(range(1, len(hypothesis_fields) + 1))

        for field, label in zip(hypothesis_fields, hypothesis_labels):
            hypothesis = Hypothesis(hypothesis_text=self.sample_dict[field], hypothesis_label=label)
            self.hypotheses[field] = hypothesis

    def _eval_signal_frequency_bandwidth(self, threshold=-50) -> float:
        time_stride = 0.01
        hop_length = int(self.sampling_rate * time_stride)
        n_fft = 512
        spectrogram = np.mean(
            np.abs(librosa.stft(y=self.signal, n_fft=n_fft, hop_length=hop_length, window='blackmanharris')) ** 2,
            axis=1,
        )
        power_spectrum = librosa.power_to_db(S=spectrogram, ref=np.max, top_db=100)
        frequency_bandwidth = 0
        for idx in range(len(power_spectrum) - 1, -1, -1):
            if power_spectrum[idx] > threshold:
                frequency_bandwidth = idx / n_fft * self.sampling_rate
                break

        self.frequency_bandwidth = frequency_bandwidth

    def _estimate_audio_metrics(self):
        self.signal, self.sampling_rate = librosa.load(path=self.audio_filepath, sr=None)
        self._eval_signal_frequency_bandwidth(threshold=-50)
        self.level_db = 20 * np.log10(np.max(np.abs(self.signal)))

    def compute(self, estimate_audio_metrics: bool = False):
        """
        Computes metrics for the sample, including word frequencies and audio metrics if specified.

        Parameters:
        - estimate_audio_metrics (bool): Flag indicating whether to estimate audio metrics (default is False).
        """

        self.num_chars = len(self.reference_text)
        self.words = self.reference_text.split()
        self.num_words = len(self.words)
        self.charset = set(self.reference_text)
        self.words_frequencies = dict(Counter(self.words))

        if self.duration is not None:
            self.char_rate = round(self.num_chars / self.duration, 2)
            self.word_rate = round(self.num_chars / self.duration, 2)

        if len(self.hypotheses) != 0:
            for label in self.hypotheses:
                self.hypotheses[label].compute(
                    reference_text=self.reference_text,
                    reference_words=self.words,
                    reference_num_words=self.num_words,
                    reference_num_chars=self.num_chars,
                )

        if estimate_audio_metrics and self.audio_filepath is not None:
            self._estimate_audio_metrics()

        self.add_table_metrics_to_dict()

    def add_table_metrics_to_dict(self):
        """
        Adds computed metrics to the sample dictionary.
        """

        metrics = {
            "num_chars": self.num_chars,
            "num_words": self.num_words,
        }

        if self.duration is not None:
            metrics["char_rate"] = self.char_rate
            metrics["word_rate"] = self.word_rate

        if len(self.hypotheses) != 0:
            for label in self.hypotheses:
                hypothesis_metrics = self.hypotheses[label].get_table_metrics()
                metrics.update(hypothesis_metrics)

        if self.frequency_bandwidth is not None:
            metrics["freq_bandwidth"] = self.frequency_bandwidth
            metrics["level_db"] = self.level_db

        self.sample_dict.update(metrics)


class Hypothesis:
    """
    A class representing a hypothesis for evaluating speech-related data.

    Parameters:
    - hypothesis_text (str): The text of the hypothesis.
    - hypothesis_label (str): Label associated with the hypothesis (default is None).

    Attributes:
    - hypothesis_text (str): The text of the hypothesis.
    - hypothesis_label (str): Label associated with the hypothesis.
    - hypothesis_words (list): List of words in the hypothesis text.
    - wer (float): Word Error Rate metric.
    - wmr (float): Word Match Rate metric.
    - num_insertions (int): Number of insertions in the hypothesis.
    - num_deletions (int): Number of deletions in the hypothesis.
    - deletions_insertions_diff (int): Difference between deletions and insertions.
    - word_match (int): Number of word matches in the hypothesis.
    - word_distance (int): Total word distance in the hypothesis.
    - match_words_frequencies (dict): Dictionary containing frequencies of matching words.
    - char_distance (int): Total character distance in the hypothesis.
    - cer (float): Character Error Rate metric.

    Methods:
    - compute(reference_text: str, reference_words: list[str], reference_num_words: int, reference_num_chars: int):
        Computes metrics for the hypothesis based on a reference text.

    - get_table_metrics() -> dict:
        Returns a dictionary containing computed metrics suitable for tabular presentation.

    """

    def __init__(self, hypothesis_text: str, hypothesis_label: str = None):
        self.hypothesis_text = hypothesis_text
        self.hypothesis_label = hypothesis_label
        self.hypothesis_words = None

        self.wer = None
        self.wmr = None
        self.num_insertions = None
        self.num_deletions = None
        self.deletions_insertions_diff = None
        self.word_match = None
        self.word_distance = None
        self.match_words_frequencies = dict()

        self.char_distance = None
        self.cer = None

    def compute(
        self,
        reference_text: str,
        reference_words: list[str] = None,
        reference_num_words: int = None,
        reference_num_chars: int = None,
    ):
        """
        Computes metrics for the hypothesis based on a reference text.

        Parameters:
        - reference_text (str): The reference text for comparison.
        - reference_words (list[str]): List of words in the reference text (default is None).
        - reference_num_words (int): Number of words in the reference text (default is None).
        - reference_num_chars (int): Number of characters in the reference text (default is None).
        """

        if reference_words is None:
            reference_words = reference_text.split()
        if reference_num_words is None:
            reference_num_words = len(reference_words)
        if reference_num_chars is None:
            reference_num_chars = len(reference_text)

        self.hypothesis_words = self.hypothesis_text.split()

        # word match metrics
        measures = jiwer.compute_measures(reference_text, self.hypothesis_text)

        self.wer = round(measures['wer'] * 100.0, 2)
        self.wmr = round(measures['hits'] / reference_num_words * 100.0, 2)
        self.num_insertions = measures['insertions']
        self.num_deletions = measures['deletions']
        self.deletions_insertions_diff = self.num_deletions - self.num_insertions
        self.word_match = measures['hits']
        self.word_distance = measures['substitutions'] + measures['insertions'] + measures['deletions']

        sm = SequenceMatcher()
        sm.set_seqs(reference_words, self.hypothesis_words)
        self.match_words_frequencies = dict(
            Counter(
                [
                    reference_words[word_idx]
                    for match in sm.get_matching_blocks()
                    for word_idx in range(match[0], match[0] + match[2])
                ]
            )
        )

        # char match metrics
        self.char_distance = editdistance.eval(reference_text, self.hypothesis_text)
        self.cer = round(self.char_distance / reference_num_chars * 100.0, 2)

    def get_table_metrics(self):
        """
        Returns a dictionary containing computed metrics.

        Returns:
        - dict: A dictionary containing computed metrics.
        """

        postfix = ""
        if self.hypothesis_label != "":
            postfix = f"_{self.hypothesis_label}"

        metrics = {
            f"WER{postfix}": self.wer,
            f"CER{postfix}": self.cer,
            f"WMR{postfix}": self.wmr,
            f"I{postfix}": self.num_insertions,
            f"D{postfix}": self.num_deletions,
            f"D-I{postfix}": self.deletions_insertions_diff,
        }
        return metrics
