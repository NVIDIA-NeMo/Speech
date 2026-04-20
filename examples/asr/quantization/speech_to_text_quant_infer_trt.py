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
Script for inference ASR models using TensorRT
"""

import os
from argparse import ArgumentParser

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
from omegaconf import open_dict

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.utils import logging

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

TRT_LOGGER = trt.Logger()


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
    parser.add_argument(
        "--asr_onnx",
        type=str,
        default="./asr_model.onnx",
        help="Pass path to exported ONNX model",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dont_normalize_text",
        default=False,
        action='store_false',
        help="Turn off trasnscript normalization. Recommended for non-English.",
    )
    parser.add_argument(
        "--use_cer", default=False, action='store_true', help="Use Character Error Rate as the evaluation metric"
    )
    parser.add_argument('--qat', action="store_true", help="Use onnx file exported from QAT tools")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

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
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
    decoding_cfg = CTCDecodingConfig()
    char_decoding = CTCDecoding(decoding_cfg, vocabulary=labels_map)
    wer = WER(char_decoding, use_cer=args.use_cer)
    wer_result = evaluate(asr_model, args.asr_onnx, labels_map, wer, args.qat)
    logging.info(f'Got WER of {wer_result}.')


def get_min_max_input_shape(asr_model):
    max_shape = (1, 64, 1)
    min_shape = (64, 64, 99999)
    for test_batch in asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        processed_signal, processed_signal_length = asr_model.preprocessor(
            input_signal=test_batch[0], length=test_batch[1]
        )
        shape = processed_signal.cpu().numpy().shape
        if shape[0] > max_shape[0]:
            max_shape = (shape[0], *max_shape[1:])
        if shape[0] < min_shape[0]:
            min_shape = (shape[0], *min_shape[1:])
        if shape[2] > max_shape[2]:
            max_shape = (*max_shape[0:2], shape[2])
        if shape[2] < min_shape[2]:
            min_shape = (*min_shape[0:2], shape[2])
    return min_shape, max_shape


def build_trt_engine(asr_model, onnx_path, qat):
    trt_engine_path = "{}.trt".format(onnx_path)
    if os.path.exists(trt_engine_path):
        return trt_engine_path

    min_input_shape, max_input_shape = get_min_max_input_shape(asr_model)
    workspace_size = 512
    with trt.Builder(TRT_LOGGER) as builder:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if qat:
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        with (
            builder.create_network(flags=network_flags) as network,
            trt.OnnxParser(network, TRT_LOGGER) as parser,
            builder.create_builder_config() as builder_config,
        ):
            parser.parse_from_file(onnx_path)
            builder_config.max_workspace_size = workspace_size * (1024 * 1024)
            if qat:
                builder_config.set_flag(trt.BuilderFlag.INT8)

            profile = builder.create_optimization_profile()
            profile.set_shape("audio_signal", min=min_input_shape, opt=max_input_shape, max=max_input_shape)
            builder_config.add_optimization_profile(profile)

            engine = builder.build_engine(network, builder_config)
            serialized_engine = engine.serialize()
            with open(trt_engine_path, "wb") as fout:
                fout.write(serialized_engine)
    return trt_engine_path


def trt_inference(stream, trt_ctx, d_input, d_output, input_signal, input_signal_length):
    print("infer with shape: {}".format(input_signal.shape))

    trt_ctx.set_binding_shape(0, input_signal.shape)
    assert trt_ctx.all_binding_shapes_specified

    h_output = cuda.pagelocked_empty(tuple(trt_ctx.get_binding_shape(1)), dtype=np.float32)

    h_input_signal = cuda.register_host_memory(np.ascontiguousarray(input_signal.cpu().numpy().ravel()))
    cuda.memcpy_htod_async(d_input, h_input_signal, stream)
    trt_ctx.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    greedy_predictions = torch.tensor(h_output).argmax(dim=-1, keepdim=False)
    return greedy_predictions


def evaluate(asr_model, asr_onnx, labels_map, wer, qat):
    # Eval the model
    hypotheses = []
    references = []
    stream = cuda.Stream()
    vocabulary_size = len(labels_map) + 1
    engine_file_path = build_trt_engine(asr_model, asr_onnx, qat)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        trt_engine = runtime.deserialize_cuda_engine(f.read())
        trt_ctx = trt_engine.create_execution_context()

        profile_shape = trt_engine.get_profile_shape(profile_index=0, binding=0)
        print("profile shape min:{}, opt:{}, max:{}".format(profile_shape[0], profile_shape[1], profile_shape[2]))
        max_input_shape = profile_shape[2]
        input_nbytes = trt.volume(max_input_shape) * trt.float32.itemsize
        d_input = cuda.mem_alloc(input_nbytes)
        max_output_shape = [max_input_shape[0], vocabulary_size, (max_input_shape[-1] + 1) // 2]
        output_nbytes = trt.volume(max_output_shape) * trt.float32.itemsize
        d_output = cuda.mem_alloc(output_nbytes)

        for test_batch in asr_model.test_dataloader():
            if can_gpu:
                test_batch = [x.cuda() for x in test_batch]
            processed_signal, processed_signal_length = asr_model.preprocessor(
                input_signal=test_batch[0], length=test_batch[1]
            )

            greedy_predictions = trt_inference(
                stream,
                trt_ctx,
                d_input,
                d_output,
                input_signal=processed_signal,
                input_signal_length=processed_signal_length,
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
