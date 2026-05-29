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

import argparse
import os
from concurrent import futures

import api.nmt_pb2 as nmt
import api.nmt_pb2_grpc as nmtsrv
import grpc
import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Path to a folder containing .nemo translation model files.",
    )
    parser.add_argument(
        "--punctuation_model",
        default="",
        type=str,
        help="Optionally provide a path a .nemo file for punctation and capitalization (recommend if working with Riva speech recognition outputs)",
    )
    parser.add_argument("--port", default=50052, type=int, required=False)
    parser.add_argument("--batch_size", type=int, default=256, help="Maximum number of batches to process")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam Size")
    parser.add_argument("--len_pen", type=float, default=0.6, help="Length Penalty")
    parser.add_argument("--max_delta_length", type=int, default=5, help="Max Delta Generation Length.")

    args = parser.parse_args()
    return args


def batches(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class RivaTranslateServicer(nmtsrv.RivaTranslateServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(
        self,
        model_dir,
        punctuation_model_path,
        beam_size=1,
        len_pen=0.6,
        max_delta_length=5,
        batch_size=256,
    ):
        self._models = {}
        self._beam_size = beam_size
        self._len_pen = len_pen
        self._max_delta_length = max_delta_length
        self._batch_size = batch_size
        self._punctuation_model_path = punctuation_model_path
        self._model_dir = model_dir

        model_paths = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname.endswith('.nemo')]

        for idx, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            logging.info(f"Loading model {model_path}")
            self._load_model(model_path)

        if self._punctuation_model_path != "":
            assert os.path.exists(punctuation_model_path)
            logging.info(f"Loading punctuation model {model_path}")
            self._load_puncutation_model(punctuation_model_path)

        logging.info("Models loaded. Ready for inference requests.")

    def _load_puncutation_model(self, punctuation_model_path):
        if punctuation_model_path.endswith(".nemo"):
            self.punctuation_model = nemo_nlp.models.PunctuationCapitalizationModel.restore_from(
                restore_path=punctuation_model_path
            )
            self.punctuation_model.eval()
        else:
            raise NotImplemented(f"Only support .nemo files, but got: {punctuation_model_path}")

        if torch.cuda.is_available():
            self.punctuation_model = self.punctuation_model.cuda()

    def _load_model(self, model_path):
        if model_path.endswith(".nemo"):
            logging.info("Attempting to initialize from .nemo file")
            model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=model_path)
            model = model.eval()
            model.beam_search.beam_size = self._beam_size
            model.beam_search.len_pen = self._len_pen
            model.beam_search.max_delta_length = self._max_delta_length
            if torch.cuda.is_available():
                model = model.cuda()
        else:
            raise NotImplemented(f"Only support .nemo files, but got: {model_path}")

        if not hasattr(model, "src_language") or not hasattr(model, "tgt_language"):
            raise ValueError(
                f"Could not find src_language and tgt_language in model attributes. If using NeMo rc1 checkpoints, please edit the config files to add model.src_language and model.tgt_language"
            )

        src_language = model.src_language
        tgt_language = model.tgt_language

        if src_language not in self._models:
            self._models[src_language] = {}

        if tgt_language not in self._models[src_language]:
            self._models[src_language][tgt_language] = model
            if torch.cuda.is_available():
                self._models[src_language][tgt_language] = self._models[src_language][tgt_language].cuda()
        else:
            raise ValueError(f"Already found model for language pair {src_language}-{tgt_language}")

    def TranslateText(self, request, context):
        logging.info(f"Request received w/ {len(request.texts)} utterances")
        results = []

        if request.source_language not in self._models:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Could not find source-target language pair {request.source_language}-{request.target_language} in list of models."
            )
            return nmt.TranslateTextResponse()

        if request.target_language not in self._models[request.source_language]:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Could not find source-target language pair {request.source_language}-{request.target_language} in list of models."
            )
            return nmt.TranslateTextResponse()

        request_strings = [x for x in request.texts]

        for batch in batches(request_strings, self._batch_size):
            if self._punctuation_model_path != "":
                batch = self.punctuation_model.add_punctuation_capitalization(batch)
            batch_results = self._models[request.source_language][request.target_language].translate(text=batch)
            translations = [nmt.Translation(translation=x) for x in batch_results]
            results.extend(translations)

        return nmt.TranslateTextResponse(translations=results)


def serve():
    args = get_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = RivaTranslateServicer(
        model_dir=args.model_dir,
        punctuation_model_path=args.punctuation_model,
        beam_size=args.beam_size,
        len_pen=args.len_pen,
        batch_size=args.batch_size,
        max_delta_length=args.max_delta_length,
    )
    nmtsrv.add_RivaTranslateServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:' + str(args.port))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
