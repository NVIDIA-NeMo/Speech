# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import argparse
import json
import random


def write_manifest(fp, records):
    """
    Writes a list of records to a JSON file, where each record is written as a new line.

    Args:
        fp (str): File path where the records should be written.
        records (list): List of records (dictionaries) to write.
    """
    with open(fp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} records to: {}".format(len(records), fp))


def main():
    """
    Processes text and audio context data to create text-context pairs.
    The resulting dataset is saved as a JSON manifest file.

    Example usage:
    python scripts/magpietts/dpo/create_text_contextpairs.py \
    --challenging_texts /Data/DPOPairsInputDatav2/challenging_with_short.txt \
    --regular_texts_for_audiocontext /Data/DPOPairsInputDatav2/regular_texts_for_audiocontext.txt \
    --regular_texts_for_textcontext /Data/DPOPairsInputDatav2/regular_texts_for_textcontext.txt \
    --audio_contexts /Data/DPOPairsInputDatav2/audio_context_list.json \
    --text_contexts /Data/DPOPairsInputDatav2/text_context_list_with_audio.txt \
    --output_manifest /Data/DPOPairsInputDatav2/grpo_train_with_short.json \
    --n_audio_contexts_per_challenging_text 2 \
    --n_text_contexts_per_challenging_text 2 \
    --n_audio_contexts_per_regular_text 1 \
    --n_text_contexts_per_regular_text 1 \
    --nsamples_perpair 1 ;
    """
    parser = argparse.ArgumentParser(description='Create text-context pairs for DPO')
    parser.add_argument("--challenging_texts", type=str, help="Text file containing challenging texts")
    parser.add_argument("--regular_texts_for_audiocontext", type=str, help="Text file containing regular texts")
    parser.add_argument("--regular_texts_for_textcontext", type=str, help="Text file containing regular texts")
    parser.add_argument(
        "--audio_contexts", type=str, help="Manifest containing audio contexts"
    )  # This manifest should contain 'context_audio_filepath', 'context_audio_duration' and (optionally) 'context_audio_codes_path'
    parser.add_argument("--text_contexts", type=str, help="Text file containing text contexts")
    parser.add_argument("--n_audio_contexts_per_challenging_text", type=int, default=10)
    parser.add_argument("--n_audio_contexts_per_regular_text", type=int, default=1)
    parser.add_argument("--n_text_contexts_per_challenging_text", type=int, default=3)
    parser.add_argument("--n_text_contexts_per_regular_text", type=int, default=1)
    parser.add_argument("--nsamples_perpair", type=int, default=6)
    parser.add_argument("--output_manifest", type=str)
    args = parser.parse_args()

    with open(args.challenging_texts, 'r') as f:
        challenging_texts = f.readlines()
        challenging_texts = [text.strip() for text in challenging_texts if text.strip() != '']

    with open(args.regular_texts_for_audiocontext, 'r') as f:
        regular_texts_for_audiocontext = f.readlines()
        regular_texts_for_audiocontext = [
            text.strip() for text in regular_texts_for_audiocontext if text.strip() != ''
        ]

    with open(args.regular_texts_for_textcontext, 'r') as f:
        regular_texts_for_textcontext = f.readlines()
        regular_texts_for_textcontext = [text.strip() for text in regular_texts_for_textcontext if text.strip() != '']

    with open(args.audio_contexts, 'r') as f:
        audio_contexts = f.readlines()
        audio_contexts = [json.loads(context.strip()) for context in audio_contexts if context.strip() != '']

    with open(args.text_contexts, 'r') as f:
        text_contexts = f.readlines()
        text_contexts = [text for text in text_contexts if text.strip() != '']

    all_records = []
    for challenging_text in challenging_texts:
        for _ in range(args.n_audio_contexts_per_challenging_text):
            audio_context = random.choice(audio_contexts)
            record = create_audio_context_record(challenging_text, audio_context, 'challenging')
            all_records.append(record)

        for _ in range(args.n_text_contexts_per_challenging_text):
            text_context = random.choice(text_contexts)
            record = create_text_context_record(challenging_text, text_context, 'challenging')
            all_records.append(record)

    for regular_text in regular_texts_for_audiocontext:
        for _ in range(args.n_audio_contexts_per_regular_text):
            audio_context = random.choice(audio_contexts)
            record = create_audio_context_record(regular_text, audio_context, 'regular')
            all_records.append(record)

    for regular_text in regular_texts_for_textcontext:
        for _ in range(args.n_text_contexts_per_regular_text):
            text_context = random.choice(text_contexts)
            record = create_text_context_record(regular_text, text_context, 'regular')
            all_records.append(record)

    random.shuffle(all_records)
    repeated_records = []
    for record in all_records:
        for _ in range(args.nsamples_perpair):
            repeated_records.append(record)

    write_manifest(args.output_manifest, repeated_records)
    write_manifest(
        args.output_manifest.replace(".json", "_tinysubset.json"), repeated_records[: 100 * args.nsamples_perpair]
    )


def create_audio_context_record(text, audio_context, record_type):
    """
    Creates a record for a text-context pair with audio context.

    Args:
        text (str): The main text content.
        audio_context (dict): Dictionary containing audio context information.
        record_type (str): Type of record ('challenging' or 'regular').

    Returns:
        dict: A dictionary representing the audio context record.
    """
    record = {
        'text': text,
        'duration': 6.0,  # Does not matter, avoids filtering out in DPO,
        'audio_filepath': audio_context['context_audio_filepath'],
        'context_audio_filepath': audio_context['context_audio_filepath'],
        'context_audio_duration': audio_context['context_audio_duration'],
        'record_type': record_type,  # challenging or regular
    }
    if 'context_audio_codes_path' in audio_context:
        record['context_audio_codes_path'] = audio_context['context_audio_codes_path']
        record['target_audio_codes_path'] = audio_context['context_audio_codes_path']

    return record


def create_text_context_record(text, text_context, record_type):
    """
    Creates a record for a text-context pair with text context.

    Args:
        text (str): The main text content.
        text_context (str): The associated text context.
        record_type (str): Type of record ('challenging' or 'regular').

    Returns:
        dict: A dictionary representing the text context record.
    """
    if text_context.endswith("\n"):
        text_context = text_context[:-1]
    record = {
        'text': text,
        'duration': 6.0,  # Does not matter, avoids filtering out in DPO,
        'audio_filepath': text_context.split(",")[1],
        'context_text': text_context.split(",")[0],
        'record_type': record_type,  # challenging or regular
    }
    if text_context.split(",")[-1].endswith(".pt"):
        record['target_audio_codes_path'] = text_context.split(",")[-1]
    return record


if __name__ == '__main__':
    main()
