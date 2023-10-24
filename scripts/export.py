# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
import sys

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

import nemo
from nemo.core import ModelPT
from nemo.core.classes import Exportable
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Export NeMo models to ONNX/Torchscript",
    )
    parser.add_argument("source", help="Source .nemo file")
    parser.add_argument("out", help="Location to write result to")
    parser.add_argument("--autocast", action="store_true", help="Use autocast when exporting")
    parser.add_argument("--runtime-check", action="store_true", help="Runtime check of exported net result")
    parser.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")
    parser.add_argument("--max-batch", type=int, default=None, help="Max batch size for model export")
    parser.add_argument("--max-dim", type=int, default=None, help="Max dimension(s) for model export")
    parser.add_argument("--onnx-opset", type=int, default=None, help="ONNX opset for model export")
    parser.add_argument(
        "--cache_support", action="store_true", help="enables caching inputs for the models support it."
    )
    parser.add_argument("--device", default="cuda", help="Device to export for")
    parser.add_argument("--check-tolerance", type=float, default=0.01, help="tolerance for verification")
    parser.add_argument(
        "--export-config",
        metavar="KEY=VALUE",
        nargs='+',
        help="Set a number of key-value pairs to model.export_config dictionary "
        "(do not put spaces before or after the = sign). "
        "Note that values are always treated as strings.",
    )

    args = parser.parse_args(argv)
    return args


def nemo_export(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))

    """Convert a .nemo saved model into .riva Riva input format."""
    nemo_in = args.source
    out = args.out

    # Create a PL trainer object which is required for restoring Megatron models
    cfg_trainer = TrainerConfig(
        accelerator='gpu',
        strategy="ddp",
        num_nodes=1,
        devices=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
        enable_checkpointing=False,
    )
    cfg_trainer = OmegaConf.to_container(OmegaConf.create(cfg_trainer))
    trainer = Trainer(**cfg_trainer)

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        with torch.inference_mode():
            # Restore instance from .nemo file using generic model restore_from
            model = ModelPT.restore_from(restore_path=nemo_in, trainer=trainer)
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.__class__.__name__, nemo_in))

    if not isinstance(model, Exportable):
        logging.error("Your NeMo model class ({}) is not Exportable.".format(model.__class__.__name__))
        sys.exit(1)

    #
    #  Add custom export parameters here
    #
    check_trace = args.runtime_check

    in_args = {}
    max_batch = 1
    max_dim = None
    if args.max_batch is not None:
        in_args["max_batch"] = args.max_batch
        max_batch = args.max_batch
    if args.max_dim is not None:
        in_args["max_dim"] = args.max_dim
        max_dim = args.max_dim

    if args.cache_support:
        model.set_export_config({"cache_support": "True"})

    if args.export_config:
        kv = {}
        for key_value in args.export_config:
            lst = key_value.split("=")
            if len(lst) != 2:
                raise Exception("Use correct format for --export_config: k=v")
            k, v = lst
            kv[k] = v
        model.set_export_config(kv)

    autocast = nullcontext
    if args.autocast:
        autocast = torch.cuda.amp.autocast
    try:
        with autocast(), torch.no_grad(), torch.inference_mode():
            model.to(device=args.device).freeze()
            model.eval()
            input_example = None
            if check_trace and len(in_args) > 0:
                input_example = model.input_module.input_example(**in_args)
                check_trace = [input_example]
                for key, arg in in_args.items():
                    in_args[key] = (arg + 1) // 2
                input_example2 = model.input_module.input_example(**in_args)
                check_trace.append(input_example2)
                logging.info(f"Using additional check args: {in_args}")

            _, descriptions = model.export(
                out,
                input_example=input_example,
                check_trace=check_trace,
                check_tolerance=args.check_tolerance,
                onnx_opset_version=args.onnx_opset,
                verbose=bool(args.verbose),
            )

    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.__class__
            )
        )
        raise e


if __name__ == '__main__':
    nemo_export(sys.argv[1:])
