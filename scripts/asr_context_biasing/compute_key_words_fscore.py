# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.asr.parts import context_biasing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="manifest with recognition results",
    )
    parser.add_argument(
        "--key_words_file",
        type=str,
        default=None,
        help="file of key words for fscore calculation (global mode)",
    )
    parser.add_argument(
        "--key_words_field",
        type=str,
        nargs='?',
        const='dictionary',
        default=None,
        help="manifest field with comma-separated per-sample keywords (default field name: 'dictionary')",
    )
    parser.add_argument(
        "--ctcws-mode",
        action='store_true',
        help="whether to use ctcws mode to split the key words from transcriptions",
    )

    args = parser.parse_args()

    if args.key_words_file is None and args.key_words_field is None:
        parser.error("Either --key_words_file or --key_words_field must be provided")

    if args.key_words_field is not None:
        context_biasing.compute_fscore(args.input_manifest, key_words_field=args.key_words_field, print_stats=True)
    else:
        key_words_list = []
        with open(args.key_words_file, encoding='utf-8') as f:
            for line in f.readlines():
                if args.ctcws_mode:
                    item = line.strip().split("_")[0].lower()
                else:
                    item = line.strip().lower()
                if item not in key_words_list:
                    key_words_list.append(item)

        context_biasing.compute_fscore(args.input_manifest, key_words_list=key_words_list, print_stats=True)


if __name__ == '__main__':
    main()
