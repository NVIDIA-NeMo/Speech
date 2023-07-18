# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


"""
This script is used to create an index of custom vocabulary and save it to file.
See "examples/nlp/spellchecking_asr_customization/run_infer.sh" for the whole inference pipeline.
"""

from argparse import ArgumentParser

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import get_index, load_ngram_mappings

parser = ArgumentParser(description="Create an index of custom vocabulary and save it to file")

parser.add_argument(
    "--input_name", required=True, type=str, help="Path to input file with custom vocabulary (plain text)"
)
parser.add_argument(
    "--ngram_mappings", required=True, type=str, help="Path to input file with n-gram mapping vocabulary"
)
parser.add_argument("--output_name", required=True, type=str, help="Path to output file with custom vocabulary index")
parser.add_argument("--min_log_prob", default=-4.0, type=float, help="Threshold on log probability")
parser.add_argument(
    "--max_phrases_per_ngram",
    default=500,
    type=int,
    help="Threshold on number of phrases that can be stored for one n-gram key in index. Keys with more phrases are discarded.",
)
parser.add_argument(
    "--max_misspelled_freq", default=125000, type=int, help="Threshold on maximum frequency of misspelled n-gram"
)

args = parser.parse_args()

# Load custom vocabulary
custom_phrases = set()
with open(args.input_name, "r", encoding="utf-8") as f:
    for line in f:
        phrase = line.strip()
        custom_phrases.add(" ".join(list(phrase.replace(" ", "_"))))
print("Size of customization vocabulary:", len(custom_phrases))

# Load n-gram mappings vocabulary
ngram_mapping_vocab, ban_ngram = load_ngram_mappings(args.ngram_mappings, max_misspelled_freq=args.max_misspelled_freq)

# Generate index of custom phrases
phrases, ngram2phrases = get_index(
    custom_phrases,
    ngram_mapping_vocab,
    ban_ngram,
    min_log_prob=args.min_log_prob,
    max_phrases_per_ngram=args.max_phrases_per_ngram,
)

# Save index to file
with open(args.output_name, "w", encoding="utf-8") as out:
    for ngram in ngram2phrases:
        for phrase_id, begin, size, logprob in ngram2phrases[ngram]:
            phrase = phrases[phrase_id]
            out.write(ngram + "\t" + phrase + "\t" + str(begin) + "\t" + str(size) + "\t" + str(logprob) + "\n")
