# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nemo.collections.audio.data.audio_to_audio_lhotse import convert_manifest_nemo_to_lhotse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an audio-to-audio manifest from NeMo format to Lhotse format. "
        "This step enables the use of Lhotse datasets for audio-to-audio processing. "
    )
    parser.add_argument("input", help='Path to the input NeMo manifest.')
    parser.add_argument(
        "output", help="Path where we'll write the output Lhotse manifest (supported extensions: .jsonl.gz and .jsonl)"
    )
    parser.add_argument(
        "-i",
        "--input_key",
        default="audio_filepath",
        help="Key of the input recording, mapped to Lhotse's 'Cut.recording'.",
    )
    parser.add_argument(
        "-t",
        "--target_key",
        default="target_filepath",
        help="Key of the target recording, mapped to Lhotse's 'Cut.target_recording'.",
    )
    parser.add_argument(
        "-r",
        "--reference_key",
        default="reference_filepath",
        help="Key of the reference recording, mapped to Lhotse's 'Cut.reference_recording'.",
    )
    parser.add_argument(
        "-e",
        "--embedding_key",
        default="embedding_filepath",
        help="Key of the embedding, mapped to Lhotse's 'Cut.embedding_vector'.",
    )
    parser.add_argument(
        "-a",
        "--force_absolute_paths",
        action='store_true',
        default=False,
        help="Force absolute paths in the generated manifests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_manifest_nemo_to_lhotse(
        input_manifest=args.input,
        output_manifest=args.output,
        input_key=args.input_key,
        target_key=args.target_key,
        reference_key=args.reference_key,
        embedding_key=args.embedding_key,
        force_absolute_paths=args.force_absolute_paths,
    )


if __name__ == "__main__":
    main()
