#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import math

import click

from nemo.collections.common.tokenizers import CanaryTokenizer


@click.command()
@click.argument("output_dir", type=click.Path())
def main(output_dir: str) -> None:
    """
    Builds the special tokens tokenizer for NVIDIA Canary-2.0 model.
    It's intended to be used with CanaryTokenizer (a specialized AggregateTokenizer)
    under name ``spl_tokens``.
    """

    tokens = (
        [
            # Generic special tokens
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|nopnc|>",
            "<|pnc|>",
            "<|nospeech|>",
            "<|startofcontext|>",
            "<|itn|>",
            "<|noitn|>",
            "<|timestamp|>",
            "<|notimestamp|>",
            "<|diarize|>",
            "<|nodiarize|>",
            "<|spkchange|>",
            "<|audioseparator|>",
            "<|emo:undefined|>",
            "<|emo:neutral|>",
            "<|emo:happy|>",
            "<|emo:sad|>",
            "<|emo:angry|>",
        ]
        # Language special tokens
        + [
            "<|unklang|>",
        ]
        + ISO_LANGS
        # Timestamp frame special tokens
        + [f"<|{i}|>" for i in range(900)]
        # Speaker indicator special tokens
        + [f"<|spk{i}|>" for i in range(16)]
    )

    num_tokens = len(tokens) + 3  # count "<pad>", "<unk>", "_" too
    print(f"We have {num_tokens} special tokens.")
    final_num_tokens = next_multiple_of_64(num_tokens)
    num_extra_tokens = final_num_tokens - num_tokens
    print(f"Adding extra {num_extra_tokens} unused special tokens for a total vocab size of {final_num_tokens}")

    tokens += [
        # Timestamp related special tokens
        f"<|spltoken{i}|>"
        for i in range(num_extra_tokens)
    ]

    tokenizer = CanaryTokenizer.build_special_tokenizer(
        tokens=tokens,
        model_dir=output_dir,
        force_rebuild=True,
    )

    assert tokenizer.vocab_size == 1152, tokenizer.vocab_size


def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64


ISO_LANGS = [
    "<|aa|>",
    "<|ab|>",
    "<|af|>",
    "<|ak|>",
    "<|sq|>",
    "<|am|>",
    "<|ar|>",
    "<|an|>",
    "<|hy|>",
    "<|as|>",
    "<|av|>",
    "<|ae|>",
    "<|ay|>",
    "<|az|>",
    "<|bm|>",
    "<|ba|>",
    "<|eu|>",
    "<|be|>",
    "<|bn|>",
    "<|bi|>",
    "<|bs|>",
    "<|br|>",
    "<|bg|>",
    "<|my|>",
    "<|ca|>",
    "<|ch|>",
    "<|ce|>",
    "<|ny|>",
    "<|zh|>",
    "<|cu|>",
    "<|cv|>",
    "<|kw|>",
    "<|co|>",
    "<|cr|>",
    "<|hr|>",
    "<|cs|>",
    "<|da|>",
    "<|dv|>",
    "<|nl|>",
    "<|dz|>",
    "<|en|>",
    "<|eo|>",
    "<|et|>",
    "<|ee|>",
    "<|fo|>",
    "<|fj|>",
    "<|fi|>",
    "<|fr|>",
    "<|fy|>",
    "<|ff|>",
    "<|gd|>",
    "<|gl|>",
    "<|lg|>",
    "<|ka|>",
    "<|de|>",
    "<|el|>",
    "<|kl|>",
    "<|gn|>",
    "<|gu|>",
    "<|ht|>",
    "<|ha|>",
    "<|he|>",
    "<|hz|>",
    "<|hi|>",
    "<|ho|>",
    "<|hu|>",
    "<|is|>",
    "<|io|>",
    "<|ig|>",
    "<|id|>",
    "<|ia|>",
    "<|ie|>",
    "<|iu|>",
    "<|ik|>",
    "<|ga|>",
    "<|it|>",
    "<|ja|>",
    "<|jv|>",
    "<|kn|>",
    "<|kr|>",
    "<|ks|>",
    "<|kk|>",
    "<|km|>",
    "<|ki|>",
    "<|rw|>",
    "<|ky|>",
    "<|kv|>",
    "<|kg|>",
    "<|ko|>",
    "<|kj|>",
    "<|ku|>",
    "<|lo|>",
    "<|la|>",
    "<|lv|>",
    "<|li|>",
    "<|ln|>",
    "<|lt|>",
    "<|lu|>",
    "<|lb|>",
    "<|mk|>",
    "<|mg|>",
    "<|ms|>",
    "<|ml|>",
    "<|mt|>",
    "<|gv|>",
    "<|mi|>",
    "<|mr|>",
    "<|mh|>",
    "<|mn|>",
    "<|na|>",
    "<|nv|>",
    "<|nd|>",
    "<|nr|>",
    "<|ng|>",
    "<|ne|>",
    "<|no|>",
    "<|nb|>",
    "<|nn|>",
    "<|oc|>",
    "<|oj|>",
    "<|or|>",
    "<|om|>",
    "<|os|>",
    "<|pi|>",
    "<|ps|>",
    "<|fa|>",
    "<|pl|>",
    "<|pt|>",
    "<|pa|>",
    "<|qu|>",
    "<|ro|>",
    "<|rm|>",
    "<|rn|>",
    "<|ru|>",
    "<|se|>",
    "<|sm|>",
    "<|sg|>",
    "<|sa|>",
    "<|sc|>",
    "<|sr|>",
    "<|sn|>",
    "<|sd|>",
    "<|si|>",
    "<|sk|>",
    "<|sl|>",
    "<|so|>",
    "<|st|>",
    "<|es|>",
    "<|su|>",
    "<|sw|>",
    "<|ss|>",
    "<|sv|>",
    "<|tl|>",
    "<|ty|>",
    "<|tg|>",
    "<|ta|>",
    "<|tt|>",
    "<|te|>",
    "<|th|>",
    "<|bo|>",
    "<|ti|>",
    "<|to|>",
    "<|ts|>",
    "<|tn|>",
    "<|tr|>",
    "<|tk|>",
    "<|tw|>",
    "<|ug|>",
    "<|uk|>",
    "<|ur|>",
    "<|uz|>",
    "<|ve|>",
    "<|vi|>",
    "<|vo|>",
    "<|wa|>",
    "<|cy|>",
    "<|wo|>",
    "<|xh|>",
    "<|ii|>",
    "<|yi|>",
    "<|yo|>",
    "<|za|>",
    "<|zu|>",
]


if __name__ == "__main__":
    main()
