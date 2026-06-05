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
from random import Random

import click
from lhotse import CutSet

from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationTarWriter,
)


@click.command()
@click.argument("manifest", type=click.Path())
@click.argument("output_dir", type=click.Path())
@click.option("-n", "--shard_size", type=int, default=100, help="Number of conversations per shard.")
@click.option("--shuffle/--no-shuffle", default=False, help="Shuffle conversations.")
@click.option("-s", "--seed", type=int, default=42, help="Random seed.")
def export(manifest: str, output_dir: str, shard_size: int, shuffle: bool, seed: int):
    with NeMoMultimodalConversationTarWriter(output_dir, shard_size=shard_size) as writer:
        source = NeMoMultimodalConversationJsonlAdapter(manifest, audio_locator_tag="<dummy>")
        if shuffle:
            source = CutSet(source).shuffle(buffer_size=50000, rng=Random(seed))
        for item in source:
            writer.write(item)


if __name__ == '__main__':
    export()
