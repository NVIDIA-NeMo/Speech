# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import soundfile as sf
import torch
from tqdm.auto import tqdm
from utils.constants import BLANK_TOKEN, SPACE_TOKEN, V_NEGATIVE_NUM

from nemo.utils import logging


def _get_utt_id(audio_filepath, audio_filepath_parts_in_utt_id):
    fp_parts = Path(audio_filepath).parts[-audio_filepath_parts_in_utt_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


def get_batch_starts_ends(manifest_filepath, batch_size):
    """
    Get the start and end ids of the lines we will use for each 'batch'.
    """

    with open(manifest_filepath, 'r') as f:
        num_lines_in_manifest = sum(1 for _ in f)

    starts = [x for x in range(0, num_lines_in_manifest, batch_size)]
    ends = [x - 1 for x in starts]
    ends.pop(0)
    ends.append(num_lines_in_manifest)

    return starts, ends


def is_entry_in_any_lines(manifest_filepath, entry):
    """
    Returns True if entry is a key in any of the JSON lines in manifest_filepath
    """

    entry_in_manifest = False

    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry in data:
                entry_in_manifest = True

    return entry_in_manifest


def is_entry_in_all_lines(manifest_filepath, entry):
    """
    Returns True is entry is a key in all of the JSON lines in manifest_filepath.
    """
    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry not in data:
                return False

    return True


def get_manifest_lines_batch(manifest_filepath, start, end):
    manifest_lines_batch = []
    with open(manifest_filepath, "r", encoding="utf-8-sig") as f:
        for line_i, line in enumerate(f):
            if line_i >= start and line_i <= end:
                data = json.loads(line)
                if "text" in data:
                    # remove any BOM, any duplicated spaces, convert any
                    # newline chars to spaces
                    data["text"] = data["text"].replace("\ufeff", "")
                    data["text"] = " ".join(data["text"].split())
                manifest_lines_batch.append(data)

            if line_i == end:
                break
    return manifest_lines_batch


def get_char_tokens(text, model):
    tokens = []
    for character in text:
        if character in model.decoder.vocabulary:
            tokens.append(model.decoder.vocabulary.index(character))
        else:
            tokens.append(len(model.decoder.vocabulary))  # return unk token (same as blank token)

    return tokens


def is_sub_or_superscript_pair(ref_text, text):
    """returns True if ref_text is a subscript or superscript version of text"""
    sub_or_superscript_to_num = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }

    if text in sub_or_superscript_to_num:
        if sub_or_superscript_to_num[text] == ref_text:
            return True
    return False


def restore_token_case(word, word_tokens):

    # remove repeated "▁" and "_" from word as that is what the tokenizer will do
    while "▁▁" in word:
        word = word.replace("▁▁", "▁")

    while "__" in word:
        word = word.repalce("__", "_")

    word_tokens_cased = []
    word_char_pointer = 0

    for token in word_tokens:
        token_cased = ""

        for token_char in token:
            if token_char == word[word_char_pointer]:
                token_cased += token_char
                word_char_pointer += 1

            else:
                if token_char.upper() == word[word_char_pointer] or is_sub_or_superscript_pair(
                    token_char, word[word_char_pointer]
                ):
                    token_cased += token_char.upper()
                    word_char_pointer += 1
                else:
                    if token_char == "▁" or token_char == "_":
                        if word[word_char_pointer] == "▁" or word[word_char_pointer] == "_":
                            token_cased += token_char
                            word_char_pointer += 1
                        elif word_char_pointer == 0:
                            token_cased += token_char

                    else:
                        raise RuntimeError(
                            f"Unexpected error - failed to recover capitalization of tokens for word {word}"
                        )

        word_tokens_cased.append(token_cased)

    return word_tokens_cased


@dataclass
class Token:
    text: str = None
    text_cased: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None


@dataclass
class Word:
    text: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None
    tokens: List[Token] = field(default_factory=list)


@dataclass
class Segment:
    text: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None
    words_and_tokens: List[Union[Word, Token]] = field(default_factory=list)


@dataclass
class Utterance:
    token_ids_with_blanks: List[int] = field(default_factory=list)
    segments_and_tokens: List[Union[Segment, Token]] = field(default_factory=list)
    text: str = None
    pred_text: str = None
    audio_filepath: str = None
    utt_id: str = None
    saved_output_files: dict = field(default_factory=dict)


def get_utt_obj(
    text, model, separator, T, audio_filepath, utt_id,
):
    """
    Function to create an Utterance object and add all necessary information to it except
        for timings of the segments / words / tokens according to the alignment - that will
        be done later in a different function, after the alignment is done.

        The Utterance object has a list segments_and_tokens which contains Segment objects and
        Token objects (for blank tokens in between segments).
        Within the Segment objects, there is a list words_and_tokens which contains Word objects and
        Token objects (for blank tokens in between words).
        Within the Word objects, there is a list tokens tokens which contains Token objects for
        blank and non-blank tokens.
        We will be building up these lists in this function. This data structure will then be useful for
        generating the various output files that we wish to save.
    """

    if not separator:  # if separator is not defined - treat the whole text as one segment
        segments = [text]
    else:
        segments = text.split(separator)

    # remove any spaces at start and end of segments
    segments = [seg.strip() for seg in segments]
    # remove any empty segments
    segments = [seg for seg in segments if len(seg) > 0]

    utt = Utterance(text=text, audio_filepath=audio_filepath, utt_id=utt_id,)

    # build up lists: token_ids_with_blanks, segments_and_tokens.
    # The code for these is different depending on whether we use char-based tokens or not
    if hasattr(model, 'tokenizer'):
        if hasattr(model, 'blank_id'):
            BLANK_ID = model.blank_id
        else:
            BLANK_ID = len(model.tokenizer.vocab)  # TODO: check

        utt.token_ids_with_blanks = [BLANK_ID]

        # check for text being 0 length
        if len(text) == 0:
            return utt

        # check for # tokens + token repetitions being > T
        all_tokens = model.tokenizer.text_to_ids(text)
        n_token_repetitions = 0
        for i_tok in range(1, len(all_tokens)):
            if all_tokens[i_tok] == all_tokens[i_tok - 1]:
                n_token_repetitions += 1

        if len(all_tokens) + n_token_repetitions > T:
            logging.info(
                f"Utterance {utt_id} has too many tokens compared to the audio file duration."
                " Will not generate output alignment files for this utterance."
            )
            return utt

        # build up data structures containing segments/words/tokens
        utt.segments_and_tokens.append(Token(text=BLANK_TOKEN, text_cased=BLANK_TOKEN, s_start=0, s_end=0,))

        segment_s_pointer = 1  # first segment will start at s=1 because s=0 is a blank
        word_s_pointer = 1  # first word will start at s=1 because s=0 is a blank

        for segment in segments:
            # add the segment to segment_info and increment the segment_s_pointer
            segment_tokens = model.tokenizer.text_to_tokens(segment)
            utt.segments_and_tokens.append(
                Segment(
                    text=segment,
                    s_start=segment_s_pointer,
                    # segment_tokens do not contain blanks => need to muliply by 2
                    # s_end needs to be the index of the final token (including blanks) of the current segment:
                    # segment_s_pointer + len(segment_tokens) * 2 is the index of the first token of the next segment =>
                    # => need to subtract 2
                    s_end=segment_s_pointer + len(segment_tokens) * 2 - 2,
                )
            )
            segment_s_pointer += (
                len(segment_tokens) * 2
            )  # multiply by 2 to account for blanks (which are not present in segment_tokens)

            words = segment.split(" ")  # we define words to be space-separated sub-strings
            for word_i, word in enumerate(words):

                word_tokens = model.tokenizer.text_to_tokens(word)
                word_token_ids = model.tokenizer.text_to_ids(word)
                word_tokens_cased = restore_token_case(word, word_tokens)

                # add the word to word_info and increment the word_s_pointer
                utt.segments_and_tokens[-1].words_and_tokens.append(
                    # word_tokens do not contain blanks => need to muliply by 2
                    # s_end needs to be the index of the final token (including blanks) of the current word:
                    # word_s_pointer + len(word_tokens) * 2 is the index of the first token of the next word =>
                    # => need to subtract 2
                    Word(text=word, s_start=word_s_pointer, s_end=word_s_pointer + len(word_tokens) * 2 - 2)
                )
                word_s_pointer += (
                    len(word_tokens) * 2
                )  # multiply by 2 to account for blanks (which are not present in word_tokens)

                for token_i, (token, token_id, token_cased) in enumerate(
                    zip(word_tokens, word_token_ids, word_tokens_cased)
                ):
                    # add the text tokens and the blanks in between them
                    # to our token-based variables
                    utt.token_ids_with_blanks.extend([token_id, BLANK_ID])
                    # adding Token object for non-blank token
                    utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                        Token(
                            text=token,
                            text_cased=token_cased,
                            # utt.token_ids_with_blanks has the form [...., <this non-blank token>, <blank>] =>
                            # => if do len(utt.token_ids_with_blanks) - 1 you get the index of the final <blank>
                            # => we want to do len(utt.token_ids_with_blanks) - 2 to get the index of <this non-blank token>
                            s_start=len(utt.token_ids_with_blanks) - 2,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 2,
                        )
                    )

                    # adding Token object for blank tokens in between the tokens of the word
                    # (ie do not add another blank if you have reached the end)
                    if token_i < len(word_tokens) - 1:
                        utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                            Token(
                                text=BLANK_TOKEN,
                                text_cased=BLANK_TOKEN,
                                # utt.token_ids_with_blanks has the form [...., <this blank token>] =>
                                # => if do len(utt.token_ids_with_blanks) -1 you get the index of this <blank>
                                s_start=len(utt.token_ids_with_blanks) - 1,
                                # s_end is same as s_start since the token only occupies one element in the list
                                s_end=len(utt.token_ids_with_blanks) - 1,
                            )
                        )

                # add a Token object for blanks in between words in this segment
                # (but only *in between* - do not add the token if it is after the final word)
                if word_i < len(words) - 1:
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            # utt.token_ids_with_blanks has the form [...., <this blank token>] =>
                            # => if do len(utt.token_ids_with_blanks) -1 you get the index of this <blank>
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

            # add the blank token in between segments/after the final segment
            utt.segments_and_tokens.append(
                Token(
                    text=BLANK_TOKEN,
                    text_cased=BLANK_TOKEN,
                    # utt.token_ids_with_blanks has the form [...., <this blank token>] =>
                    # => if do len(utt.token_ids_with_blanks) -1 you get the index of this <blank>
                    s_start=len(utt.token_ids_with_blanks) - 1,
                    # s_end is same as s_start since the token only occupies one element in the list
                    s_end=len(utt.token_ids_with_blanks) - 1,
                )
            )

        return utt

    elif hasattr(model.decoder, "vocabulary"):  # i.e. tokenization is simply character-based

        BLANK_ID = len(model.decoder.vocabulary)  # TODO: check this is correct
        SPACE_ID = model.decoder.vocabulary.index(" ")

        utt.token_ids_with_blanks = [BLANK_ID]

        # check for text being 0 length
        if len(text) == 0:
            return utt

        # check for # tokens + token repetitions being > T
        all_tokens = get_char_tokens(text, model)
        n_token_repetitions = 0
        for i_tok in range(1, len(all_tokens)):
            if all_tokens[i_tok] == all_tokens[i_tok - 1]:
                n_token_repetitions += 1

        if len(all_tokens) + n_token_repetitions > T:
            logging.info(
                f"Utterance {utt_id} has too many tokens compared to the audio file duration."
                " Will not generate output alignment files for this utterance."
            )
            return utt

        # build up data structures containing segments/words/tokens
        utt.segments_and_tokens.append(Token(text=BLANK_TOKEN, text_cased=BLANK_TOKEN, s_start=0, s_end=0,))

        segment_s_pointer = 1  # first segment will start at s=1 because s=0 is a blank
        word_s_pointer = 1  # first word will start at s=1 because s=0 is a blank

        for i_segment, segment in enumerate(segments):
            # add the segment to segment_info and increment the segment_s_pointer
            segment_tokens = get_char_tokens(segment, model)
            utt.segments_and_tokens.append(
                Segment(
                    text=segment,
                    s_start=segment_s_pointer,
                    # segment_tokens do not contain blanks => need to muliply by 2
                    # s_end needs to be the index of the final token (including blanks) of the current segment:
                    # segment_s_pointer + len(segment_tokens) * 2 is the index of the first token of the next segment =>
                    # => need to subtract 2
                    s_end=segment_s_pointer + len(segment_tokens) * 2 - 2,
                )
            )

            # for correct calculation: multiply len(segment_tokens) by 2 to account for blanks (which are not present in segment_tokens)
            # and + 2 to account for [<token for space in between segments>, <blank token after that space token>]
            segment_s_pointer += len(segment_tokens) * 2 + 2

            words = segment.split(" ")  # we define words to be space-separated substrings
            for i_word, word in enumerate(words):

                # convert string to list of characters
                word_tokens = list(word)
                # convert list of characters to list of their ids in the vocabulary
                word_token_ids = get_char_tokens(word, model)

                # add the word to word_info and increment the word_s_pointer
                utt.segments_and_tokens[-1].words_and_tokens.append(
                    # note for s_end:
                    # word_tokens do not contain blanks => need to muliply by 2
                    # s_end needs to be the index of the final token (including blanks) of the current word:
                    # word_s_pointer + len(word_tokens) * 2 is the index of the first token of the next word =>
                    # => need to subtract 2
                    Word(text=word, s_start=word_s_pointer, s_end=word_s_pointer + len(word_tokens) * 2 - 2)
                )

                # for correct calculation: multiply len(word_tokens) by 2 to account for blanks (which are not present in word_tokens)
                # and + 2 to account for [<token for space in between words>, <blank token after that space token>]
                word_s_pointer += len(word_tokens) * 2 + 2

                for token_i, (token, token_id) in enumerate(zip(word_tokens, word_token_ids)):
                    # add the text tokens and the blanks in between them
                    # to our token-based variables
                    utt.token_ids_with_blanks.extend([token_id])
                    utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                        Token(
                            text=token,
                            text_cased=token,
                            # utt.token_ids_with_blanks has the form [..., <this non-blank token>]
                            # => do len(utt.token_ids_with_blanks) - 1 to get the index of this non-blank token
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

                    if token_i < len(word_tokens) - 1:  # only add blank tokens that are in the middle of words
                        utt.token_ids_with_blanks.extend([BLANK_ID])
                        utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                            Token(
                                text=BLANK_TOKEN,
                                text_cased=BLANK_TOKEN,
                                # utt.token_ids_with_blanks has the form [..., <this blank token>]
                                # => do len(utt.token_ids_with_blanks) - 1 to get the index of this blank token
                                s_start=len(utt.token_ids_with_blanks) - 1,
                                # s_end is same as s_start since the token only occupies one element in the list
                                s_end=len(utt.token_ids_with_blanks) - 1,
                            )
                        )

                # add space token (and the blanks around it) unless this is the final word in a segment
                if i_word < len(words) - 1:
                    utt.token_ids_with_blanks.extend([BLANK_ID, SPACE_ID, BLANK_ID])
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            # utt.token_ids_with_blanks has the form
                            # [..., <final token of previous word>, <blank token>, <space token>, <blank token>]
                            # => do len(utt.token_ids_with_blanks) - 3 to get the index of the blank token before the space token
                            s_start=len(utt.token_ids_with_blanks) - 3,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 3,
                        )
                    )
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=SPACE_TOKEN,
                            text_cased=SPACE_TOKEN,
                            # utt.token_ids_with_blanks has the form
                            # [..., <final token of previous word>, <blank token>, <space token>, <blank token>]
                            # => do len(utt.token_ids_with_blanks) - 2 to get the index of the space token
                            s_start=len(utt.token_ids_with_blanks) - 2,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 2,
                        )
                    )
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            # utt.token_ids_with_blanks has the form
                            # [..., <final token of previous word>, <blank token>, <space token>, <blank token>]
                            # => do len(utt.token_ids_with_blanks) - 1 to get the index of the blank token after the space token
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

            # add a blank to the segment, and add a space after if this is not the final segment
            utt.token_ids_with_blanks.extend([BLANK_ID])
            utt.segments_and_tokens.append(
                Token(
                    text=BLANK_TOKEN,
                    text_cased=BLANK_TOKEN,
                    # utt.token_ids_with_blanks has the form [..., <this blank token>]
                    # => do len(utt.token_ids_with_blanks) - 1 to get the index of this blank token
                    s_start=len(utt.token_ids_with_blanks) - 1,
                    # s_end is same as s_start since the token only occupies one element in the list
                    s_end=len(utt.token_ids_with_blanks) - 1,
                )
            )

            if i_segment < len(segments) - 1:
                utt.token_ids_with_blanks.extend([SPACE_ID, BLANK_ID])
                utt.segments_and_tokens.append(
                    Token(
                        text=SPACE_TOKEN,
                        text_cased=SPACE_TOKEN,
                        # utt.token_ids_with_blanks has the form
                        # [..., <space token>, <blank token>]
                        # => do len(utt.token_ids_with_blanks) - 2 to get the index of the space token
                        s_start=len(utt.token_ids_with_blanks) - 2,
                        # s_end is same as s_start since the token only occupies one element in the list
                        s_end=len(utt.token_ids_with_blanks) - 2,
                    )
                )
                utt.segments_and_tokens.append(
                    Token(
                        text=BLANK_TOKEN,
                        text_cased=BLANK_TOKEN,
                        # utt.token_ids_with_blanks has the form
                        # [..., <space token>, <blank token>]
                        # => do len(utt.token_ids_with_blanks) - 1 to get the index of the blank token
                        s_start=len(utt.token_ids_with_blanks) - 1,
                        # s_end is same as s_start since the token only occupies one element in the list
                        s_end=len(utt.token_ids_with_blanks) - 1,
                    )
                )

        return utt

    else:
        raise RuntimeError("Cannot get tokens of this model.")


def add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration):
    """
    Function to add t_start and t_end (representing time in seconds) to the Utterance object utt_obj.
    Args:
        utt_obj: Utterance object to which we will add t_start and t_end for its 
            constituent segments/words/tokens.
        alignment_utt: a list of ints indicating which token does the alignment pass through at each 
            timestep (will take the form [0, 0, 1, 1, ..., <num of tokens including blanks in uterance>]).
        output_timestep_duration: a float indicating the duration of a single output timestep from
            the ASR Model.

    Returns:
        utt_obj: updated Utterance object.
    """

    # General idea for the algorithm of how we add t_start and t_end
    # the timestep where a token s starts is the location of the first appearance of s_start in alignment_utt
    # the timestep where a token s ends is the location of the final appearance of s_end in alignment_utt
    # We will make dictionaries num_to_first_alignment_appearance and
    # num_to_last_appearance and use that to update all of
    # the t_start and t_end values in utt_obj.
    # We will put t_start = t_end = -1 for tokens that are skipped (should only be blanks)

    num_to_first_alignment_appearance = dict()
    num_to_last_alignment_appearance = dict()

    prev_s = -1  # use prev_s to keep track of when the s changes
    for t, s in enumerate(alignment_utt):
        if s > prev_s:
            num_to_first_alignment_appearance[s] = t

            if prev_s >= 0:  # dont record prev_s = -1
                num_to_last_alignment_appearance[prev_s] = t - 1
        prev_s = s
    # add last appearance of the final s
    num_to_last_alignment_appearance[prev_s] = len(alignment_utt) - 1

    # update all the t_start and t_end in utt_obj
    for segment_or_token in utt_obj.segments_and_tokens:
        if type(segment_or_token) is Segment:
            segment = segment_or_token
            segment.t_start = num_to_first_alignment_appearance[segment.s_start] * output_timestep_duration
            segment.t_end = (num_to_last_alignment_appearance[segment.s_end] + 1) * output_timestep_duration

            for word_or_token in segment.words_and_tokens:
                if type(word_or_token) is Word:
                    word = word_or_token
                    word.t_start = num_to_first_alignment_appearance[word.s_start] * output_timestep_duration
                    word.t_end = (num_to_last_alignment_appearance[word.s_end] + 1) * output_timestep_duration

                    for token in word.tokens:
                        if token.s_start in num_to_first_alignment_appearance:
                            token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                        else:
                            token.t_start = -1

                        if token.s_end in num_to_last_alignment_appearance:
                            token.t_end = (
                                num_to_last_alignment_appearance[token.s_end] + 1
                            ) * output_timestep_duration
                        else:
                            token.t_end = -1
                else:
                    token = word_or_token
                    if token.s_start in num_to_first_alignment_appearance:
                        token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                    else:
                        token.t_start = -1

                    if token.s_end in num_to_last_alignment_appearance:
                        token.t_end = (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                    else:
                        token.t_end = -1

        else:
            token = segment_or_token
            if token.s_start in num_to_first_alignment_appearance:
                token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
            else:
                token.t_start = -1

            if token.s_end in num_to_last_alignment_appearance:
                token.t_end = (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
            else:
                token.t_end = -1

    return utt_obj


def get_batch_variables(
    manifest_lines_batch,
    model,
    separator,
    align_using_pred_text,
    audio_filepath_parts_in_utt_id,
    output_timestep_duration,
    simulate_cache_aware_streaming=False,
    use_buffered_chunked_streaming=False,
    buffered_chunk_params={},
):
    """
    Returns:
        log_probs, y, T, U (y and U are s.t. every other token is a blank) - these are the tensors we will need
            during Viterbi decoding.
        utt_obj_batch: a list of Utterance objects for every utterance in the batch.
        output_timestep_duration: a float indicating the duration of a single output timestep from
            the ASR Model.
    """

    # get hypotheses by calling 'transcribe'
    # we will use the output log_probs, the duration of the log_probs,
    # and (optionally) the predicted ASR text from the hypotheses
    audio_filepaths_batch = [line["audio_filepath"] for line in manifest_lines_batch]
    B = len(audio_filepaths_batch)
    log_probs_list_batch = []
    T_list_batch = []
    pred_text_batch = []

    if not use_buffered_chunked_streaming:
        if not simulate_cache_aware_streaming:
            with torch.no_grad():
                hypotheses = model.transcribe(audio_filepaths_batch, return_hypotheses=True, batch_size=B)
        else:
            with torch.no_grad():
                hypotheses = model.transcribe_simulate_cache_aware_streaming(
                    audio_filepaths_batch, return_hypotheses=True, batch_size=B
                )

        # if hypotheses form a tuple (from Hybrid model), extract just "best" hypothesis
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        for hypothesis in hypotheses:
            log_probs_list_batch.append(hypothesis.y_sequence)
            T_list_batch.append(hypothesis.y_sequence.shape[0])
            pred_text_batch.append(hypothesis.text)
    else:
        delay = buffered_chunk_params["delay"]
        model_stride_in_secs = buffered_chunk_params["model_stride_in_secs"]
        tokens_per_chunk = buffered_chunk_params["tokens_per_chunk"]
        for l in tqdm(audio_filepaths_batch, desc="Sample:"):
            model.reset()
            model.read_audio_file(l, delay, model_stride_in_secs)
            hyp, logits = model.transcribe(tokens_per_chunk, delay, keep_logits=True)
            log_probs_list_batch.append(logits)
            T_list_batch.append(logits.shape[0])
            pred_text_batch.append(hyp)

    # we loop over every line in the manifest that is in our current batch,
    # and record the y (list of tokens, including blanks), U (list of lengths of y) and
    # token_info_batch, word_info_batch, segment_info_batch
    y_list_batch = []
    U_list_batch = []
    utt_obj_batch = []

    for i_line, line in enumerate(manifest_lines_batch):
        if align_using_pred_text:
            gt_text_for_alignment = " ".join(pred_text_batch[i_line].split())
        else:
            gt_text_for_alignment = line["text"]
        utt_obj = get_utt_obj(
            gt_text_for_alignment,
            model,
            separator,
            T_list_batch[i_line],
            audio_filepaths_batch[i_line],
            _get_utt_id(audio_filepaths_batch[i_line], audio_filepath_parts_in_utt_id),
        )

        # update utt_obj.pred_text or utt_obj.text
        if align_using_pred_text:
            utt_obj.pred_text = pred_text_batch[i_line]
            if len(utt_obj.pred_text) == 0:
                logging.info(
                    f"'pred_text' of utterance {utt_obj.utt_id} is empty - we will not generate"
                    " any output alignment files for this utterance"
                )
            if "text" in line:
                utt_obj.text = line["text"]  # keep the text as we will save it in the output manifest
        else:
            utt_obj.text = line["text"]
            if len(utt_obj.text) == 0:
                logging.info(
                    f"'text' of utterance {utt_obj.utt_id} is empty - we will not generate"
                    " any output alignment files for this utterance"
                )

        y_list_batch.append(utt_obj.token_ids_with_blanks)
        U_list_batch.append(len(utt_obj.token_ids_with_blanks))
        utt_obj_batch.append(utt_obj)

    # turn log_probs, y, T, U into dense tensors for fast computation during Viterbi decoding
    T_max = max(T_list_batch)
    U_max = max(U_list_batch)
    #  V = the number of tokens in the vocabulary + 1 for the blank token.
    if hasattr(model, 'tokenizer'):
        V = len(model.tokenizer.vocab) + 1
    else:
        V = len(model.decoder.vocabulary) + 1
    T_batch = torch.tensor(T_list_batch)
    U_batch = torch.tensor(U_list_batch)

    # make log_probs_batch tensor of shape (B x T_max x V)
    log_probs_batch = V_NEGATIVE_NUM * torch.ones((B, T_max, V))
    for b, log_probs_utt in enumerate(log_probs_list_batch):
        t = log_probs_utt.shape[0]
        log_probs_batch[b, :t, :] = log_probs_utt

    # make y tensor of shape (B x U_max)
    # populate it initially with all 'V' numbers so that the 'V's will remain in the areas that
    # are 'padding'. This will be useful for when we make 'log_probs_reorderd' during Viterbi decoding
    # in a different function.
    y_batch = V * torch.ones((B, U_max), dtype=torch.int64)
    for b, y_utt in enumerate(y_list_batch):
        U_utt = U_batch[b]
        y_batch[b, :U_utt] = torch.tensor(y_utt)

    # calculate output_timestep_duration if it is None
    if output_timestep_duration is None:
        if not 'window_stride' in model.cfg.preprocessor:
            raise ValueError(
                "Don't have attribute 'window_stride' in 'model.cfg.preprocessor' => cannot calculate "
                " model_downsample_factor => stopping process"
            )

        if not 'sample_rate' in model.cfg.preprocessor:
            raise ValueError(
                "Don't have attribute 'sample_rate' in 'model.cfg.preprocessor' => cannot calculate start "
                " and end time of segments => stopping process"
            )

        with sf.SoundFile(audio_filepaths_batch[0]) as f:
            audio_dur = f.frames / f.samplerate
        n_input_frames = audio_dur / model.cfg.preprocessor.window_stride
        model_downsample_factor = round(n_input_frames / int(T_batch[0]))

        output_timestep_duration = (
            model.preprocessor.featurizer.hop_length * model_downsample_factor / model.cfg.preprocessor.sample_rate
        )

        logging.info(
            f"Calculated that the model downsample factor is {model_downsample_factor}"
            f" and therefore the ASR model output timestep duration is {output_timestep_duration}"
            " -- will use this for all batches"
        )

    return (
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        utt_obj_batch,
        output_timestep_duration,
    )
