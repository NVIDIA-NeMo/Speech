# -*- coding: utf-8 -*-
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

""" Script responsible for generation of a JSON file containing list of modules of a given collection. """

import argparse
import importlib
import inspect
import json
import os

import nemo
from nemo.utils import logging


def process_member(name, obj, module_list):
    """Helper function processing the passed object and, if ok, adding a record to the module list.

    Args:
        name: name of the member
        obj: member (class/function etc.)
        module_list: list of modules that (probably) will be expanded.
    """
    # It is not a class - skip it.
    if not inspect.isclass(obj):
        return

    # Check inheritance - we know that all our datasets/modules/losses inherit from Serialization,
    # Btw. Serialization is also required by this script.
    if not issubclass(obj, nemo.core.Serialization):
        return

    logging.info("  * Processing `{}`".format(str(obj)))

    module_list.append(
        {
            "name": name,
            "cls": str(obj),
            # Temporary solution: mockup arguments.
            "arguments": [
                "jasper",
                "activation",
                "feat_in",
                "normalization_mode",
                "residual_mode",
                "norm_groups",
                "conv_mask",
                "frame_splicing",
                "init_mode",
            ],
            # Temporary solution: mockup input types.
            "input_types": {
                "audio_signal": "axes: (batch, dimension, time); elements_type: MelSpectrogramType",
                "length": "axes: (batch,); elements_type: LengthType",
            },
            # Temporary solution: mockup output types.
            "output_types": {
                "encoder_output": "axes: (batch, dimension, time); elements_type: AcousticEncodedRepresentation"
            },
        }
    )


def main():
    """Main function analysing the indicated NeMo collection and generating a JSON file with module descriptions."""
    # Parse filename.
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', help='ID of the collection', type=str)
    parser.add_argument('--filename', help='Name of the output JSON file', type=str, default="modules.json")
    args = parser.parse_args()

    # Get collections directory.
    colletions_dir = os.path.dirname(nemo.collections.__file__)
    logging.info('Analysing collections in `{}`'.format(colletions_dir))

    # Generate list of NeMo collections - from the list of collection subfolders.
    collections = {}
    for sub_dir in os.listdir(colletions_dir):
        # Skip cache.
        if sub_dir == "__pycache__":
            continue
        # Check if it is a directory.
        if os.path.isdir(os.path.join(colletions_dir, sub_dir)):
            collections[sub_dir] = "nemo.collections." + sub_dir

    # Check the collection.
    if args.collection not in collections.keys():
        logging.error("Coudn't process the incidated `{}` collection".format(args.collection))
        logging.info(
            "Please select one of the existing collections using `--collection [{}]`".format("|".join(collections))
        )
        exit(-1)

    # Load the collection specification.
    collection_spec = importlib.util.find_spec(collections[args.collection])
    if collection_spec is None:
        logging.error("Failed to load the `{}` collection".format(val))

    # Import the module from the module specification.
    collection = importlib.util.module_from_spec(collection_spec)
    collection_spec.loader.exec_module(collection)

    module_list = []
    # Iterate over the packages in the indicated collection.
    logging.info("Analysing the `{}` collection".format(args.collection))

    try:  # Datasets in dataset folder
        logging.info("Analysing the 'data' package")
        for name, obj in inspect.getmembers(collection.data):
            process_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No datasets found")

    try:  # Datasets in dataset folder
        logging.info("Analysing the 'datasets' package")
        for name, obj in inspect.getmembers(collection.datasets):
            process_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No datasets found")

    try:  # Modules
        logging.info("Analysing the 'modules' package")
        for name, obj in inspect.getmembers(collection.modules):
            process_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No modules found")

    try:  # Losses
        logging.info("Analysing the 'losses' package")
        for name, obj in inspect.getmembers(collection.losses):
            process_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No losses found")

    # Add prefix - only for default name.
    filename = args.filename if args.filename != "modules.json" else args.collection + "_" + args.filename
    # Export to JSON.
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(module_list, outfile)

    logging.info(
        'Finished analysis of the `{}` collection, results exported to `{}`.'.format(args.collection, filename)
    )


if __name__ == '__main__':
    main()
