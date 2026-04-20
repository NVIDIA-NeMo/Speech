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

"""
Script for calibrating a pretrained ASR model for quantization
"""

from argparse import ArgumentParser

import torch
from omegaconf import open_dict

from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

try:
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    from pytorch_quantization.tensor_quant import QuantDescriptor
except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )

can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model",
        type=str,
        default="stt_en_fastconformer_ctc_large",
        required=True,
        help="Pass: 'stt_en_fastconformer_ctc_large'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--dont_normalize_text",
        default=False,
        action='store_false',
        help="Turn off trasnscript normalization. Recommended for non-English.",
    )
    parser.add_argument('--num_calib_batch', default=1, type=int, help="Number of batches for calibration.")
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument("--amp", action="store_true", help="Use AMP in calibration.")
    parser.set_defaults(amp=False)

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    # Initialize quantization
    quant_desc_input = QuantDescriptor(calib_method=args.calibrator)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model_cfg = EncDecCTCModel.restore_from(restore_path=args.asr_model, return_config=True)
        with open_dict(asr_model_cfg):
            asr_model_cfg.encoder.quantize = True
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model, override_config_path=asr_model_cfg)

    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model_cfg = EncDecCTCModel.from_pretrained(model_name=args.asr_model, return_config=True)
        with open_dict(asr_model_cfg):
            asr_model_cfg.encoder.quantize = True
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model, override_config_path=asr_model_cfg)

    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model.decoder.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': args.dont_normalize_text,
            'shuffle': True,
        }
    )
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()

    # Enable calibrators
    for name, module in asr_model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, test_batch in enumerate(asr_model.test_dataloader()):
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
            with torch.amp.autocast(asr_model.device.type, enabled=args.amp):
                _ = asr_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
        if i >= args.num_calib_batch:
            break

    # Save calibrated model(s)
    model_name = args.asr_model.replace(".nemo", "") if args.asr_model.endswith(".nemo") else args.asr_model
    if not args.calibrator == "histogram":
        compute_amax(asr_model, method="max")
        asr_model.save_to(F"{model_name}-max-{args.num_calib_batch*args.batch_size}.nemo")
    else:
        for percentile in args.percentile:
            print(F"{percentile} percentile calibration")
            compute_amax(asr_model, method="percentile")
            asr_model.save_to(F"{model_name}-percentile-{percentile}-{args.num_calib_batch*args.batch_size}.nemo")

        for method in ["mse", "entropy"]:
            print(F"{method} calibration")
            compute_amax(asr_model, method=method)
            asr_model.save_to(F"{model_name}-{method}-{args.num_calib_batch*args.batch_size}.nemo")


def compute_amax(model, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    if can_gpu:
        model.cuda()


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
