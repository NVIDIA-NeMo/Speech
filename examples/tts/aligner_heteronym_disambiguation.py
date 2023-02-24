# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os
import re

import librosa
import soundfile as sf
import torch

from nemo.collections.tts.models import AlignerModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import general_padding


"""
G2P disambiguation using an Aligner model's input embedding distances.

Does not handle OOV and leaves them as graphemes.

The output will have each token's phonemes (or graphemes) bracketed, e.g.
<\"><M AH1 L ER0><, ><M AH1 L ER0><, ><HH IY1 Z>< ><DH AH0>< ><M AE1 N><.\">

Example:
python aligner_heteronym_disambiguation.py \
    --model=<model_path> \
    --manifest=<manifest_path> \
    --out=<output_json_path> \
    --confidence=0.02 \
    --verbose
"""


def get_args():
    """Retrieve arguments for disambiguation.
    """
    parser = argparse.ArgumentParser("G2P disambiguation using Aligner input embedding distances.")
    # TODO(jocelynh): Make this required=False with default download from NGC once ckpt uploaded
    parser.add_argument('--model', required=True, type=str, help="Path to Aligner model checkpoint (.nemo file).")
    parser.add_argument(
        '--manifest',
        required=True,
        type=str,
        help="Path to data manifest. Each entry should contain the path to the audio file as well as the text in graphemes.",
    )
    parser.add_argument(
        '--out', required=True, type=str, help="Path to output file where disambiguations will be written."
    )
    parser.add_argument(
        '--sr',
        required=False,
        default=22050,
        type=int,
        help="Target sample rate to load the dataset. Should match what the model was trained on.",
    )
    parser.add_argument(
        '--heteronyms',
        required=False,
        type=str,
        default='../../scripts/tts_dataset_files/heteronyms-052722',
        help="Heteronyms file to specify which words should be disambiguated. All others will use default pron.",
    )
    parser.add_argument(
        '--confidence', required=False, type=float, default=0.0, help="Confidence threshold to keep a disambiguation."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="If set to True, logs scores for each disambiguated word in disambiguation_logs.txt.",
    )
    args = parser.parse_args()
    return args


def load_and_prepare_audio(aligner, audio_path, target_sr, device):
    """Loads and resamples audio to target sample rate (if necessary), and preprocesses for Aligner input.
    """
    # Load audio and get length for preprocessing
    audio_data, orig_sr = sf.read(audio_path)
    if orig_sr != target_sr:
        audio_data = librosa.core.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)

    audio = torch.tensor(audio_data, dtype=torch.float, device=device).unsqueeze(0)
    audio_len = torch.tensor(audio_data.shape[0], device=device).long().unsqueeze(0)

    # Generate spectrogram
    spec, spec_len = aligner.preprocessor(input_signal=audio, length=audio_len)

    return spec, spec_len


def disambiguate_candidates(aligner, text, spec, spec_len, confidence, device, heteronyms, log_file=None):
    """Retrieves and disambiguate all candidate sentences for disambiguation of a given some text.

    Assumes that the max number of candidates per word is a reasonable batch size.

    Note: This could be sped up if multiple words' candidates were batched, but this is conceptually easier.
    """
    # Grab original G2P result
    aligner_g2p = aligner.tokenizer.g2p
    base_g2p = aligner_g2p(text)

    # Tokenize text
    words = [word for word, _ in aligner_g2p.word_tokenize_func(text)]

    ### Loop Through Words ###
    result_g2p = []
    word_start_idx = 0

    has_heteronym = False

    for word in words:
        # Retrieve the length of the word in the default G2P conversion
        g2p_default_len = len(aligner_g2p(word))

        # Check if word needs to be disambiguated
        if word in heteronyms:
            has_heteronym = True

            # Add candidate for each ambiguous pronunciation
            word_candidates = []
            candidate_prons_and_lengths = []

            for pron in aligner_g2p.phoneme_dict[word]:
                # Replace graphemes in the base G2P result with the current variant
                candidate = base_g2p[:word_start_idx] + pron + base_g2p[word_start_idx + g2p_default_len :]
                candidate_tokens = aligner.tokenizer.encode_from_g2p(candidate)

                word_candidates.append(candidate_tokens)
                candidate_prons_and_lengths.append((pron, len(pron)))

            ### Inference ###
            num_candidates = len(word_candidates)

            # If only one candidate, just convert and continue
            if num_candidates == 1:
                has_heteronym = False
                result_g2p.append(f"<{' '.join(candidate_prons_and_lengths[0][0])}>")
                word_start_idx += g2p_default_len
                continue

            text_len = [len(toks) for toks in word_candidates]
            text_len_in = torch.tensor(text_len, device=device).long()

            # Have to pad text tokens in case different pronunciations have different lengths
            max_text_len = max(text_len)
            text_stack = []
            for i in range(num_candidates):
                padded_tokens = general_padding(
                    torch.tensor(word_candidates[i], device=device).long(), text_len[i], max_text_len
                )
                text_stack.append(padded_tokens)
            text_in = torch.stack(text_stack)

            # Repeat spectrogram and spec_len tensors to match batch size
            spec_in = spec.repeat([num_candidates, 1, 1])
            spec_len_in = spec_len.repeat([num_candidates])

            with torch.no_grad():
                soft_attn, _ = aligner(spec=spec_in, spec_len=spec_len_in, text=text_in, text_len=text_len_in)

            # Need embedding distances and duration preds to calculate mean distance for just the one word
            text_embeddings = aligner.embed(text_in).transpose(1, 2)
            l2_dists = aligner.alignment_encoder.get_dist(keys=text_embeddings, queries=spec_in).sqrt()

            durations = aligner.alignment_encoder.get_durations(soft_attn, text_len_in, spec_len_in).int()

            # Retrieve average embedding distances
            min_dist = float('inf')
            max_dist = 0.0
            best_candidate = None
            for i in range(num_candidates):
                candidate_mean_dist = aligner.alignment_encoder.get_mean_distance_for_word(
                    l2_dists=l2_dists[i],
                    durs=durations[i],
                    start_token=word_start_idx + (1 if aligner.tokenizer.pad_with_space else 0),
                    num_tokens=candidate_prons_and_lengths[i][1],
                )
                if log_file:
                    log_file.write(f"{candidate_prons_and_lengths[i][0]} -- {candidate_mean_dist}\n")

                if candidate_mean_dist < min_dist:
                    min_dist = candidate_mean_dist
                    best_candidate = candidate_prons_and_lengths[i][0]
                if candidate_mean_dist > max_dist:
                    max_dist = candidate_mean_dist

            # Calculate confidence score. If below threshold, skip and use graphemes.
            disamb_conf = (max_dist - min_dist) / ((max_dist + min_dist) / 2.0)
            if disamb_conf < confidence:
                if log_file:
                    log_file.write(f"Below confidence threshold: {best_candidate} ({disamb_conf})\n")

                has_heteronym = False
                result_g2p.append(f"<{' '.join(aligner_g2p(word))}>")
                word_start_idx += g2p_default_len
                continue

            # Otherwise, can write disambiguated word
            if log_file:
                log_file.write(f"best candidate: {best_candidate} (confidence: {disamb_conf})\n")

            result_g2p.append(f"<{' '.join(best_candidate)}>")
        else:
            if re.search("[a-zA-Z]", word) is None:
                # Punctuation or space
                result_g2p.append(f"<{word}>")
            elif word in aligner_g2p.phoneme_dict:
                # Take default pronunciation for everything else in the dictionary
                result_g2p.append(f"<{' '.join(aligner_g2p.phoneme_dict[word][0])}>")
            else:
                # OOV
                result_g2p.append(f"<{' '.join(aligner_g2p(word))}>")

        # Advance to phoneme index of next word
        word_start_idx += g2p_default_len

    if log_file and has_heteronym:
        log_file.write(f"{text}\n")
        log_file.write(f"===\n{''.join(result_g2p)}\n===\n")
        log_file.write(f"===============================\n")

    return result_g2p, has_heteronym


def disambiguate_dataset(
    aligner, manifest_path, out_path, sr, heteronyms, confidence, device, verbose, heteronyms_only=True
):
    """Disambiguates the phonemes for all words with ambiguous pronunciations in the given manifest.
    """
    log_file = open('disambiguation_logs.txt', 'w') if verbose else None

    with open(out_path, 'w') as f_out:
        with open(manifest_path, 'r') as f_in:
            count = 0

            for line in f_in:
                # Retrieve entry and base G2P conversion for full text
                entry = json.loads(line)
                # Set punct_post_process=True in order to preserve words with apostrophes
                text = aligner.normalizer.normalize(entry['text'], punct_post_process=True)
                text = aligner.tokenizer.text_preprocessing_func(text)

                # Load and preprocess audio
                audio_path = entry['audio_filepath']
                spec, spec_len = load_and_prepare_audio(aligner, audio_path, sr, device)

                # Get pronunciation candidates and disambiguate
                disambiguated_text, has_heteronym = disambiguate_candidates(
                    aligner, text, spec, spec_len, confidence, device, heteronyms, log_file
                )

                # Skip writing entry if user only wants samples with heteronyms
                if heteronyms_only and not has_heteronym:
                    continue

                # Save entry with disambiguation
                entry['disambiguated_text'] = ''.join(disambiguated_text)
                f_out.write(f"{json.dumps(entry)}\n")

                count += 1
                if count % 100 == 0:
                    print(f"Finished {count} entries.")

    print(f"Finished all entries, with a total of {count}.")
    if log_file:
        log_file.close()


def main():
    args = get_args()

    # Check file paths from arguments
    if not os.path.exists(args.model):
        print("Could not find model checkpoint file: ", args.model)
    if not os.path.exists(args.manifest):
        print("Could not find data manifest file: ", args.manifest)
    if os.path.exists(args.out):
        print("Output file already exists: ", args.out)
        overwrite = input("Is it okay to overwrite it? (Y/N): ")
        if overwrite.lower() != 'y':
            print("Not overwriting output file, quitting.")
            quit()
    if not os.path.exists(args.heteronyms):
        print("Could not find heteronyms list: ", args.heteronyms)

    # Read heteronyms list, one per line
    heteronyms = set()
    with open(args.heteronyms, 'r') as f_het:
        for line in f_het:
            heteronyms.add(line.strip().lower())

    # Load model
    print("Restoring Aligner model from checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = AlignerModel.restore_from(args.model, map_location=device)

    # Disambiguation
    print("Beginning disambiguation...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    disambiguate_dataset(aligner, args.manifest, args.out, args.sr, heteronyms, args.confidence, device, args.verbose)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
