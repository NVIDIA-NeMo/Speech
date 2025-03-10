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
from contextlib import contextmanager
from typing import Sequence

import click
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.common.data.lhotse.cutset import get_parser_fn


@click.command()
@click.argument("input_cfgs", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.argument("output_cfg", type=click.Path())
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=None,
    multiple=True,
    help="Temperature for re-weighting datasets. 1 is a neutral value. "
    "Lower temperature over-samples smaller datasets, and vice versa. "
    "Can be specified multiple times to apply a different temperature to each group level in the YAML config.",
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["num_hours", "num_examples"]),
    default="num_hours",
    help="Strategy for choosing weights for each dataset.",
)
def estimate_data_weights(input_cfgs: str, output_cfg: str, temperature: list[float], strategy: str):
    """
    Read a YAML specification of datasets from INPUT_CFGS, compute their weights, and save the result in OUTPUT_CFG.
    The weight for each entry is determined by the number of hours in a given dataset.

    If more than one config is provided as input, we will concatenate them and output a single merged config.

    Optionally, apply temperature re-weighting to balance the datasets (specify TEMPERATURE lesser than 1).
    """
    data = ListConfig([])
    for icfg in input_cfgs:
        data.extend(OmegaConf.load(icfg))
    temperature = parse_temperature(temperature)
    validate(data)
    count(data, weight_key=strategy)
    aggregate_group_weights(data)
    reweight(data, temperature=temperature)
    OmegaConf.save(data, output_cfg)


def validate(entry: DictConfig | ListConfig, _level: int = 0):
    if isinstance(entry, ListConfig):
        for subentry in entry:
            validate(subentry, _level + 1)
        return

    assert "type" in entry, f"Invalid YAML data config at nesting level {_level}: missing key 'type' in entry={entry}"

    if entry.type == "group":
        for subentry in entry["input_cfg"]:
            validate(subentry, _level + 1)


def count(entry: DictConfig | ListConfig, weight_key: str) -> None:
    if isinstance(entry, ListConfig):
        for subentry in entry:
            count(subentry, weight_key=weight_key)
        return
    if entry.type == "group":
        for subentry in entry["input_cfg"]:
            count(subentry, weight_key=weight_key)
        return

    with quick_iter_options(entry):
        iterable, is_tarred = get_parser_fn(entry.type)(entry)
        stats = {"num_hours": 0.0, "num_examples": 0}
        for example in iterable:
            if hasattr(example, "duration"):
                stats["num_hours"] += example.duration
            stats["num_examples"] += 1
        stats["num_hours"] /= 3600.0

    if weight_key == "num_hours" and stats[weight_key] == 0.0:
        raise RuntimeError(
            f"Cannot set weights based on 'num_hours': at least one dataset has examples without 'duration' property. "
            f"Details: {entry=}"
        )

    entry["weight"] = stats[weight_key]


def aggregate_group_weights(entry: DictConfig | ListConfig) -> None:
    if isinstance(entry, ListConfig):
        for subentry in entry:
            aggregate_group_weights(subentry)
        return

    if entry.type != "group":
        return

    for subentry in entry["input_cfg"]:
        if "weight" not in subentry:
            aggregate_group_weights(subentry)

    entry.weight = sum(subentry["weight"] for subentry in entry["input_cfg"])


def reweight(entry: DictConfig | ListConfig, temperature: None | float | list[float]) -> None:
    if not temperature or (isinstance(entry, DictConfig) and entry.type != "group"):
        return

    if isinstance(temperature, Sequence):
        temperature, *next_temperatures = temperature
    else:
        next_temperatures = temperature

    if isinstance(entry, ListConfig):
        for subentry in entry:
            reweight(subentry, temperature=next_temperatures)
        new_weights = temperature_reweighting([se.weight for se in entry], temperature=temperature)
        for se, nw in zip(entry, new_weights):
            se.weight = nw
        return

    for subentry in entry["input_cfg"]:
        reweight(subentry, temperature=next_temperatures)

    new_weights = temperature_reweighting([se.weight for se in entry["input_cfg"]], temperature=temperature)
    for se, nw in zip(entry["input_cfg"], new_weights):
        se.weight = nw


def temperature_reweighting(weights: list[float], temperature: float = 1.0):
    """(w_i ^ alpha / sum(w_i ^ alpha))"""
    weights = np.asarray(weights) ** temperature
    return (weights / weights.sum()).tolist()


@contextmanager
def quick_iter_options(entry: DictConfig):
    entry.metadata_only = True
    entry.force_finite = True
    yield entry
    del entry["metadata_only"]
    del entry["force_finite"]


def parse_temperature(value: list[float]) -> float | list[float] | None:
    match value:
        case 0:
            return None
        case 1:
            return value[0]
        case _:
            return value


if __name__ == '__main__':
    estimate_data_weights()
