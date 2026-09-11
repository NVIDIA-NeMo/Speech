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

from functools import partial
from io import BytesIO
from pathlib import Path

import click
import lhotse
import torch.utils.data
from lhotse import CutSet, MonoCut
from lhotse.audio.backend import LibsndfileBackend
from lhotse.dataset import DynamicCutSampler, IterableDatasetWrapper
from lhotse.shar import JsonlShardWriter, TarWriter
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig


@click.command()
@click.argument("manifest_filepath")
@click.argument("tarred_audio_filepaths")
@click.argument("filtered_manifest_filepath")
@click.argument("output_dir", type=click.Path())
@click.option(
    "-f",
    "--output-format",
    type=click.Choice(["lhotse_shar", "nemo_tarred"]),
    default="lhotse_shar",
    help="Which format should we use to save the filtered tarred data.",
)
@click.option("-s", "--shard-size", type=int, default=1000, help="Desired number of examples per output shard.")
def filter_tarred(
    manifest_filepath: str,
    tarred_audio_filepaths: str,
    filtered_manifest_filepath: str,
    output_dir: str,
    output_format: str,
    shard_size: int,
):
    """
    Given an existing tarred dataset and manifests that point to a subset of examples,
    create a new tarred dataset corresponding to the subset.

    This is useful if you want to "re-tar" an existing tarred dataset in order to efficiently
    read some subset of it.
    """
    lhotse.set_dill_enabled(True)
    all_cuts = read_cutset(manifest_filepath, tarred_audio_filepaths)
    keep_cuts = {cut.id: cut for cut in read_cutset(filtered_manifest_filepath)}
    filtered_cuts = bg_load(
        all_cuts.filter(lambda c: c.id in keep_cuts).map(partial(attach_custom, cuts_with_custom=keep_cuts))
    )
    if not '://' in output_dir:  # we support object store writing too
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    if output_format == "lhotse_shar":
        filtered_cuts.to_shar(output_dir=output_dir, fields={"recording": "flac"}, shard_size=shard_size)
    elif output_format == "nemo_tarred":
        export_to_nemo_tarred(cuts=filtered_cuts, output_dir=output_dir, shard_size=shard_size)
    else:
        raise RuntimeError(f"Unsupported output format: '{output_format}'")


def read_cutset(src: str, tar: str | None = None) -> CutSet:
    inp_arg = ["force_finite=true"]
    if tar is not None:
        inp_arg += [f"manifest_filepath={src}", f"tarred_audio_filepaths={tar}"]
    else:
        inp_arg += ["metadata_only=true"]
        if src.endswith(".yaml"):
            inp_arg += [f"input_cfg={src}"]
        elif Path(src).is_dir():
            inp_arg += [f"shar_path={src}"]
        else:
            inp_arg += [f"manifest_filepath={src}"]
    config = OmegaConf.merge(
        OmegaConf.structured(LhotseDataLoadingConfig),
        OmegaConf.from_dotlist(inp_arg),
    )
    cuts, _ = read_cutset_from_config(config)
    return cuts


def export_to_nemo_tarred(cuts: CutSet, output_dir: str, shard_size: int) -> None:
    with (
        TarWriter(pattern=f"{output_dir}/audio_%d.tar", shard_size=shard_size) as aw,
        JsonlShardWriter(pattern=f"{output_dir}/manifest_%d.jsonl", shard_size=shard_size) as mw,
    ):
        for cut in cuts:
            assert (
                isinstance(cut, MonoCut) and len(cut.supervisions) == 1
            ), f"Export to nemo_tarred format is possible only for mono cuts with a single supervision, but we got: {cut}"
            # Prepare audio for writing.
            audio_name = f"{cut.id}.flac"
            audio = BytesIO()
            LibsndfileBackend().save_audio(audio, cut.load_audio(), sampling_rate=cut.sampling_rate, format="flac")
            audio.seek(0)
            # Prepare manifest for writing.
            ans = {"audio_filepath": audio_name, "duration": cut.duration}
            if cut.supervisions[0].text:
                ans["text"] = cut.supervisions[0].text
            if cut.supervisions[0].language:
                ans["lang"] = cut.supervisions[0].language
            if cut.custom is not None:
                # Ensure if we export anything custom, these are only simple built-in types compatible with JSON.
                ans.update({k: v for k, v in cut.custom.items() if isinstance(v, (int, float, str, list, dict))})
            # Set the right shard_id.
            shard_id = max(0, mw.num_shards - 1)
            if mw.num_items > 0 and mw.num_items % mw.shard_size == 0:
                shard_id += 1
            ans["shard_id"] = shard_id
            # Write both items.
            aw.write(audio_name, audio)
            mw.write(ans)


def attach_custom(cut, cuts_with_custom):
    custom = cuts_with_custom[cut.id].custom
    if custom is not None:
        cut.custom.update(custom)
    return cut


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, x):
        cut = x[0]
        for k in ["dataloading_info", "shard_id"]:
            cut.custom.pop(k, None)
        return cut


def bg_load(cuts: CutSet) -> CutSet:
    return CutSet(
        torch.utils.data.DataLoader(
            IterableDatasetWrapper(Identity(), DynamicCutSampler(cuts, max_cuts=1)),
            batch_size=None,
            num_workers=1,
            prefetch_factor=10,
        )
    )


if __name__ == '__main__':
    filter_tarred()
