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
import os

import soundfile as sf
import torch
from tqdm.auto import tqdm
from utils.constants import BLANK_TOKEN, SPACE_TOKEN, V_NEGATIVE_NUM


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
    with open(manifest_filepath, "r") as f:
        for line_i, line in enumerate(f):
            if line_i >= start and line_i <= end:
                manifest_lines_batch.append(json.loads(line))

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


def get_y_and_boundary_info_for_utt(text, model, separator):
    """
    Get y_token_ids_with_blanks, token_info, word_info and segment_info for the text provided, tokenized 
    by the model provided.
    y_token_ids_with_blanks is a list of the indices of the text tokens with the blank token id in between every
    text token.
    token_info, word_info and segment_info are lists of dictionaries containing information about 
    where the tokens/words/segments start and end.
    For example, 'hi world | hey ' with separator = '|' and tokenized by a BPE tokenizer can have token_info like:
    token_info = [
        {'text': '<b>', 's_start': 0, 's_end': 0},
        {'text': '▁hi', 's_start': 1, 's_end': 1},
        {'text': '<b>', 's_start': 2, 's_end': 2},
        {'text': '▁world', 's_start': 3, 's_end': 3},
        {'text': '<b>', 's_start': 4, 's_end': 4},
        {'text': '▁he', 's_start': 5, 's_end': 5},
        {'text': '<b>', 's_start': 6, 's_end': 6},
        {'text': 'y', 's_start': 7, 's_end': 7},
        {'text': '<b>', 's_start': 8, 's_end': 8},    
    ]
    's_start' and 's_end' indicate where in the sequence of tokens does each token start and end.

    The word_info will be as follows:
    word_info = [
        {'text': 'hi', 's_start': 1, 's_end': 1},
        {'text': 'world', 's_start': 3, 's_end': 3},
        {'text': 'hey', 's_start': 5, 's_end': 7},
    ]
    's_start' and 's_end' indicate where in the sequence of tokens does each word start and end.

    segment_info will be as follows:
    segment_info = [
        {'text': 'hi world', 's_start': 1, 's_end': 3},
        {'text': 'hey', 's_start': 5, 's_end': 7},
    ]
    's_start' and 's_end' indicate where in the sequence of tokens does each segment start and end.
    """

    if not separator:  # if separator is not defined - treat the whole text as one segment
        segments = [text]
    else:
        segments = text.split(separator)

    # remove any spaces at start and end of segments
    segments = [seg.strip() for seg in segments]

    if hasattr(model, 'tokenizer'):
        if hasattr(model, 'blank_id'):
            BLANK_ID = model.blank_id
        else:
            BLANK_ID = len(model.decoder.vocabulary)  # TODO: check

        y_token_ids_with_blanks = [BLANK_ID]
        token_info = [{"text": BLANK_TOKEN, "s_start": 0, "s_end": 0,}]
        word_info = []
        segment_info = []

        segment_s_pointer = 1  # first segment will start at s=1 because s=0 is a blank
        word_s_pointer = 1  # first word will start at s=1 because s=0 is a blank

        for segment in segments:
            words = segment.split(" ")  # we define words to be space-separated sub-strings
            for word in words:

                word_tokens = model.tokenizer.text_to_tokens(word)
                word_ids = model.tokenizer.text_to_ids(word)
                for token, id_ in zip(word_tokens, word_ids):
                    # add the text token and the blank that follows it
                    # to our token-based variables
                    y_token_ids_with_blanks.extend([id_, BLANK_ID])
                    token_info.extend(
                        [
                            {
                                "text": token,
                                "s_start": len(y_token_ids_with_blanks) - 2,
                                "s_end": len(y_token_ids_with_blanks) - 2,
                            },
                            {
                                "text": BLANK_TOKEN,
                                "s_start": len(y_token_ids_with_blanks) - 1,
                                "s_end": len(y_token_ids_with_blanks) - 1,
                            },
                        ]
                    )

                # add the word to word_info and increment the word_s_pointer
                word_info.append(
                    {
                        "text": word,
                        "s_start": word_s_pointer,
                        "s_end": word_s_pointer + (len(word_tokens) - 1) * 2,  # TODO check this,
                    }
                )
                word_s_pointer += len(word_tokens) * 2  # TODO check this

            # add the segment to segment_info and increment the segment_s_pointer
            segment_tokens = model.tokenizer.text_to_tokens(segment)
            segment_info.append(
                {
                    "text": segment,
                    "s_start": segment_s_pointer,
                    "s_end": segment_s_pointer + (len(segment_tokens) - 1) * 2,
                }
            )
            segment_s_pointer += len(segment_tokens) * 2

        return y_token_ids_with_blanks, token_info, word_info, segment_info

    elif hasattr(model.decoder, "vocabulary"):  # i.e. tokenization is simply character-based

        BLANK_ID = len(model.decoder.vocabulary)  # TODO: check this is correct
        SPACE_ID = model.decoder.vocabulary.index(" ")

        y_token_ids_with_blanks = [BLANK_ID]
        token_info = [{"text": BLANK_TOKEN, "s_start": 0, "s_end": 0,}]
        word_info = []
        segment_info = []

        segment_s_pointer = 1  # first segment will start at s=1 because s=0 is a blank
        word_s_pointer = 1  # first word will start at s=1 because s=0 is a blank

        for i_segment, segment in enumerate(segments):
            words = segment.split(" ")  # we define words to be space-separated characters
            for i_word, word in enumerate(words):

                # convert string to list of characters
                word_tokens = list(word)
                # convert list of characters to list of their ids in the vocabulary
                word_ids = get_char_tokens(word, model)
                for token, id_ in zip(word_tokens, word_ids):
                    # add the text token and the blank that follows it
                    # to our token-based variables
                    y_token_ids_with_blanks.extend([id_, BLANK_ID])
                    token_info.extend(
                        [
                            {
                                "text": token,
                                "s_start": len(y_token_ids_with_blanks) - 2,
                                "s_end": len(y_token_ids_with_blanks) - 2,
                            },
                            {
                                "text": BLANK_TOKEN,
                                "s_start": len(y_token_ids_with_blanks) - 1,
                                "s_end": len(y_token_ids_with_blanks) - 1,
                            },
                        ]
                    )

                # add space token (and the blank after it) unless this is the final word in the final segment
                if not (i_segment == len(segments) - 1 and i_word == len(words) - 1):
                    y_token_ids_with_blanks.extend([SPACE_ID, BLANK_ID])
                    token_info.extend(
                        (
                            {
                                "text": SPACE_TOKEN,
                                "s_start": len(y_token_ids_with_blanks) - 2,
                                "s_end": len(y_token_ids_with_blanks) - 2,
                            },
                            {
                                "text": BLANK_TOKEN,
                                "s_start": len(y_token_ids_with_blanks) - 1,
                                "s_end": len(y_token_ids_with_blanks) - 1,
                            },
                        )
                    )
                # add the word to word_info and increment the word_s_pointer
                word_info.append(
                    {
                        "text": word,
                        "s_start": word_s_pointer,
                        "s_end": word_s_pointer + len(word_tokens) * 2 - 2,  # TODO check this,
                    }
                )
                word_s_pointer += len(word_tokens) * 2 + 2  # TODO check this

            # add the segment to segment_info and increment the segment_s_pointer
            segment_tokens = get_char_tokens(segment, model)
            segment_info.append(
                {
                    "text": segment,
                    "s_start": segment_s_pointer,
                    "s_end": segment_s_pointer + (len(segment_tokens) - 1) * 2,
                }
            )
            segment_s_pointer += len(segment_tokens) * 2 + 2

        return y_token_ids_with_blanks, token_info, word_info, segment_info

    else:
        raise RuntimeError("Cannot get tokens of this model.")


def get_batch_tensors_and_boundary_info(
    manifest_lines_batch,
    model,
    separator,
    align_using_pred_text,
    simulate_cache_aware_streaming=False,
    use_buffered_chunked_streaming=False,
    buffered_chunk_params={},
):
    """
    Returns:
        log_probs, y, T, U (y and U are s.t. every other token is a blank) - these are the tensors we will need
            during Viterbi decoding.
        token_info_list, word_info_list, segment_info_list - these are lists of dictionaries which we will need
            for writing the CTM files with the human-readable alignments.
        pred_text_list - this is a list of the transcriptions from our model which we will save to our output JSON
            file if align_using_pred_text is True.
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
    token_info_batch = []
    word_info_batch = []
    segment_info_batch = []

    for i_line, line in enumerate(manifest_lines_batch):
        if align_using_pred_text:
            gt_text_for_alignment = pred_text_batch[i_line]
        else:
            gt_text_for_alignment = line["text"]
        y_utt, token_info_utt, word_info_utt, segment_info_utt = get_y_and_boundary_info_for_utt(
            gt_text_for_alignment, model, separator
        )

        y_list_batch.append(y_utt)
        U_list_batch.append(len(y_utt))
        token_info_batch.append(token_info_utt)
        word_info_batch.append(word_info_utt)
        segment_info_batch.append(segment_info_utt)

    # turn log_probs, y, T, U into dense tensors for fast computation during Viterbi decoding
    T_max = max(T_list_batch)
    U_max = max(U_list_batch)
    #  V = the number of tokens in the vocabulary + 1 for the blank token.
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

    return (
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        token_info_batch,
        word_info_batch,
        segment_info_batch,
        pred_text_batch,
    )
