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
Script for post training quantization of ASR models
"""

import collections
from argparse import ArgumentParser
from pprint import pprint

import torch
from omegaconf import open_dict

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.utils import logging

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
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
    parser.add_argument("--wer_target", type=float, default=None, help="used by test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
    parser.add_argument(
        "--dont_normalize_text",
        default=False,
        action='store_false',
        help="Turn off trasnscript normalization. Recommended for non-English.",
    )
    parser.add_argument(
        "--use_cer", default=False, action='store_true', help="Use Character Error Rate as the evaluation metric"
    )
    parser.add_argument('--sensitivity', action="store_true", help="Perform sensitivity analysis")
    parser.add_argument('--onnx', action="store_true", help="Export to ONNX")
    parser.add_argument('--quant-disable-keyword', type=str, nargs='+', help='disable quantizers by keyword')
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    quant_modules.initialize()

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
        }
    )
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()

    if args.quant_disable_keyword:
        for name, module in asr_model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                for keyword in args.quant_disable_keyword:
                    if keyword in name:
                        logging.warning(F"Disable {name}")
                        module.disable()

    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
    decoding_cfg = CTCDecodingConfig()
    char_decoding = CTCDecoding(decoding_cfg, vocabulary=labels_map)
    wer = WER(char_decoding, use_cer=args.use_cer)
    wer_quant = evaluate(asr_model, labels_map, wer)
    logging.info(f'Got WER of {wer_quant}. Tolerance was {args.wer_tolerance}')

    if args.sensitivity:
        if wer_quant < args.wer_tolerance:
            logging.info("Tolerance is already met. Skip sensitivity analyasis.")
            return
        quant_layer_names = []
        for name, module in asr_model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module.disable()
                layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
                if layer_name not in quant_layer_names:
                    quant_layer_names.append(layer_name)
        logging.info(F"{len(quant_layer_names)} quantized layers found.")

        # Build sensitivity profile
        quant_layer_sensitivity = {}
        for i, quant_layer in enumerate(quant_layer_names):
            logging.info(F"Enable {quant_layer}")
            for name, module in asr_model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                    module.enable()
                    logging.info(F"{name:40}: {module}")

            # Eval the model
            wer_value = evaluate(asr_model, labels_map, wer)
            logging.info(F"WER: {wer_value}")
            quant_layer_sensitivity[quant_layer] = args.wer_tolerance - wer_value

            for name, module in asr_model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                    module.disable()
                    logging.info(F"{name:40}: {module}")

        # Skip most sensitive layers until WER target is met
        for name, module in asr_model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module.enable()
        quant_layer_sensitivity = collections.OrderedDict(sorted(quant_layer_sensitivity.items(), key=lambda x: x[1]))
        pprint(quant_layer_sensitivity)
        skipped_layers = []
        for quant_layer, _ in quant_layer_sensitivity.items():
            for name, module in asr_model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if quant_layer in name:
                        logging.info(F"Disable {name}")
                        if not quant_layer in skipped_layers:
                            skipped_layers.append(quant_layer)
                        module.disable()
            wer_value = evaluate(asr_model, labels_map, wer)
            if wer_value <= args.wer_tolerance:
                logging.info(
                    F"WER tolerance {args.wer_tolerance} is met by skipping {len(skipped_layers)} sensitive layers."
                )
                print(skipped_layers)
                export_onnx(args, asr_model)
                return
        raise ValueError(f"WER tolerance {args.wer_tolerance} can not be met with any layer quantized!")

    export_onnx(args, asr_model)


def export_onnx(args, asr_model):
    if args.onnx:
        if args.asr_model.endswith("nemo"):
            onnx_name = args.asr_model.replace(".nemo", ".onnx")
        else:
            onnx_name = args.asr_model
        logging.info(F"Export to {onnx_name}")
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        asr_model.export(onnx_name, onnx_opset_version=13)
        quant_nn.TensorQuantizer.use_fb_fake_quant = False


def evaluate(asr_model, labels_map, wer):
    # Eval the model
    hypotheses = []
    references = []
    for test_batch in asr_model.test_dataloader():
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with torch.amp.autocast(asr_model.device.type):
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        hypotheses += wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)[0]
        for batch_ind in range(greedy_predictions.shape[0]):
            seq_len = test_batch[3][batch_ind].cpu().detach().numpy()
            seq_ids = test_batch[2][batch_ind].cpu().detach().numpy()
            reference = ''.join([labels_map[c] for c in seq_ids[0:seq_len]])
            references.append(reference)
        del test_batch
    wer_value = word_error_rate(hypotheses=hypotheses, references=references, use_cer=wer.use_cer)

    return wer_value


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
