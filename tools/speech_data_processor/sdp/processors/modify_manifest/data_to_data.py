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

import collections
import re
from typing import Dict, List

from sdp.processors.base_processor import DataEntry
from sdp.processors.modify_manifest.modify_manifest import ModifyManifestTextProcessor
from sdp.utils.edit_spaces import add_start_end_spaces
from sdp.utils.get_diff import get_diff_with_subs_grouped

from nemo.utils import logging

# TODO: some of the processors here can be replaced with generic regex processor
#     by substituiting different regex patterns


class SubSubstringToSpace(ModifyManifestTextProcessor):
    """
    Class for processor that converts substrings in "text" to spaces.

    Args:
        substrings: list of strings that will be replaced with spaces if
            they are contained in data["text"].
    """

    def __init__(self, substrings: List[str], **kwargs,) -> None:
        super().__init__(**kwargs)
        self.substrings = substrings

    def _process_dataset_entry(self, data_entry) -> List:
        remove_substring_counter = collections.defaultdict(int)
        for substring in self.substrings:
            while substring in data_entry["text"]:
                data_entry["text"] = data_entry["text"].replace(substring, " ")
                remove_substring_counter[substring] += 1
        return [DataEntry(data=data_entry, metrics=remove_substring_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for substring, value in counter.items():
                total_counter[substring] += value
        logging.info("Number of substrings that were converted to spaces")
        for substring, count in total_counter.items():
            logging.info(f"{substring}, {count}, \t\t(substring.encode(): {substring.encode()}")
        super().finalize(metrics)


class SubSubstringToSubstring(ModifyManifestTextProcessor):
    """
    Class for processor that converts substrings in "text" to other substrings.
    The before and after substring pairs are defined in 'substring_pairs'.

    Args:
        substring_pairs:  Dictonary where keys are the substrings you want to change and
            their values are the substrings you want to convert them to.
    """

    def __init__(
        self, substring_pairs: Dict, **kwargs,
    ):
        super().__init__(**kwargs)
        self.substring_pairs = substring_pairs

    def _process_dataset_entry(self, data_entry) -> List:
        replace_substring_counter = collections.defaultdict(int)
        for original_word, new_word in self.substring_pairs.items():
            while original_word in data_entry["text"]:
                data_entry["text"] = data_entry["text"].replace(original_word, new_word)
        return [DataEntry(data=data_entry, metrics=replace_substring_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for substring, value in counter.items():
                total_counter[substring] += value
        logging.info("Num of substrings that were substituted")
        for substring, count in total_counter.items():
            logging.info(f"{substring} {count}")
        super().finalize(metrics)


class InsIfASRInsertion(ModifyManifestTextProcessor):
    """
    Class for processor that adds a substring to data["text"] if it is
    present at that location in data["pred_text"].
    It is useful if words are systematically missing from ground truth
    transcriptions.

    Args:
        insert_words: list of strings that will be inserted into data['text'] if
            there is an insertion (containing only that string) in data['pred_text'].
            Note: because data_to_data looks for an exact match in the insertion,
            we recommend including variations with different spaces in 'insert_words',
            e.g. [' nemo', 'nemo ', ' nemo '].
    """

    def __init__(
        self, insert_words: List[str], **kwargs,
    ):
        super().__init__(**kwargs)
        self.insert_words = insert_words

    def _process_dataset_entry(self, data_entry) -> List:
        insert_word_counter = collections.defaultdict(int)
        for insert_word in self.insert_words:
            if not insert_word in data_entry["pred_text"]:
                break
            orig_words, pred_words = data_entry["text"], data_entry["pred_text"]
            diff = get_diff_with_subs_grouped(orig_words, pred_words)

            if len(diff) > 0:  # ie if there are differences between text and pred_text

                new_sent = ""

                for diff_entry in diff:
                    if diff_entry[0] == 0:  # no change
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == -1:  # deletion in original string
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == 1:  # insertion in original string
                        if diff_entry[1] == insert_word:
                            new_sent += insert_word
                            insert_word_counter[insert_word] += 1

                    elif isinstance(diff_entry, tuple):  # i.e. diff is a substitution
                        new_sent += diff_entry[0][1]
                    else:
                        raise ValueError(f"unexpected item in diff_entry: {diff_entry}")

                new_sent = " ".join(new_sent.split())  # remove any extra spaces
                data_entry["text"] = new_sent

        return [DataEntry(data=data_entry, metrics=insert_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logging.info("Num of words that were inserted")
        for word, count in total_counter.items():
            logging.info(f"{word} {count}")
        super().finalize(metrics)


class SubIfASRSubstitution(ModifyManifestTextProcessor):
    """
    Class for processor that converts a substring in data['text'] to a
    substring in data['pred_text'] if both are located in the same place
    (ie are part of a 'substitution' operation) and if the substrings
    correspond to key-value pairs in 'sub_words'.
    This is useful if words are systematically incorrect in ground truth
    transcriptions.

    Args:
        sub_words: dictionary where a key is a string that might be in data['text']
            and the value is the string that might be in data['pred_text']. If both
            are located in the same place (ie are part of a 'substitution' operation)
            then the key string will be converted to the value string in data['text'].

            .. note::
                data_to_data looks for exact string matches of substitutions, so
                you may need to be careful with spaces in 'sub_words', e.g.
                recommended to do sub_words = {"nmo ": "nemo "} instead of
                sub_words = {"nmo" : "nemo"}
    """

    def __init__(
        self, sub_words: Dict, **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_words = sub_words

    def _process_dataset_entry(self, data_entry) -> List:
        sub_word_counter = collections.defaultdict(int)
        for original_word, new_word in self.sub_words.items():
            if not original_word in data_entry["text"]:
                break
            orig_words, pred_words = data_entry["text"], data_entry["pred_text"]
            diff = get_diff_with_subs_grouped(orig_words, pred_words)

            if len(diff) > 0:  # ie if there are differences between text and pred_text

                new_sent = ""

                for diff_entry in diff:
                    if diff_entry[0] == 0:  # no change
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == -1:  # deletion in original string
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == 1:  # insertion in original string
                        # don't make changes
                        pass

                    elif isinstance(diff_entry, tuple):  # substitution
                        if diff_entry[0][1] == original_word and diff_entry[1][1] == new_word:
                            # ie. substitution is one we want to use to change the original text
                            new_sent += new_word
                            sub_word_counter[original_word] += 1

                        else:
                            # ie. substitution is one we want to ignore
                            new_sent += diff_entry[0][1]
                    else:
                        raise ValueError(f"unexpected item in diff_entry: {diff_entry}")

                new_sent = add_start_end_spaces(new_sent)
                data_entry["text"] = new_sent

        return [DataEntry(data=data_entry, metrics=sub_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logging.info("Num of words that were substituted")
        for word, count in total_counter.items():
            logging.info(f"{word} {count}")
        super().finalize(metrics)


class SubMakeLowercase(ModifyManifestTextProcessor):
    """
    Class to convert data['text'] to lowercase by calling '.lower()' on it.
    """

    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

    def _process_dataset_entry(self, data_entry) -> List:
        data_entry["text"] = data_entry["text"].lower()
        return [DataEntry(data=data_entry)]

    def finalize(self, metrics):
        logging.info("Made all letters lowercase")
        super().finalize(metrics)


class SubRegex(ModifyManifestTextProcessor):
    """
    Class for processor that converts a regex match to a string, as defined
    by key-value pairs in regex_to_sub.

    Args:
        regex_to_sub: dictionary where the keys are regex patterns that might
            be in data['text'], and the values are the strings that will replace
            the regex matches if they are found.
    """

    def __init__(
        self, regex_to_sub: Dict, **kwargs,
    ):
        super().__init__(**kwargs)
        self.regex_to_sub = regex_to_sub

    def _process_dataset_entry(self, data_entry) -> List:
        replace_word_counter = collections.defaultdict(int)
        for regex, sub in self.regex_to_sub.items():
            while re.search(regex, data_entry["text"]):
                for match in re.finditer(regex, data_entry["text"]):
                    replace_word_counter[match.group(0)] += 1
                    data_entry["text"] = re.sub(regex, sub, data_entry["text"])
        return [DataEntry(data=data_entry, metrics=replace_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logging.info("Some of the words matching the regex that were substituted")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True))
        for word, count in total_counter_sorted.items():
            if count > 1:
                logging.info(f"{word} {count}")
        super().finalize(metrics)
