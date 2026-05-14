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
This is a helper script to extract speaker embeddings based on manifest file
Usage:
python extract_speaker_embeddings.py --manifest=/path/to/manifest/file'
--model_path='/path/to/.nemo/file'(optional)
--embedding_dir='/path/to/embedding/directory'

Args:
--manifest: path to manifest file containing audio_file paths for which embeddings need to be extracted
--model_path(optional): path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would
    be downloaded from NGC and used to extract embeddings
--embeddings_dir(optional): path to directory where embeddings need to stored default:'./'


"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.utils import logging


def get_embeddings(speaker_model, manifest_file, batch_size=1, embedding_dir='./', device='cuda'):
    """
    save embeddings to cached file
    Args:
        speaker_model: NeMo <EncDecSpeakerLabel> model
        manifest_file: path to the manifest file containing the audio file path from which the
                       embeddings should be extracted
        batch_size: batch_size for inference
        embedding_dir: path to directory to store embeddings file
        device: compute device to perform operations
    """

    all_embs, _, _, _ = speaker_model.batch_inference(manifest_file, batch_size=batch_size, device=device)
    all_embs = np.asarray(all_embs)
    all_embs = embedding_normalize(all_embs)
    out_embeddings = {}

    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_name = '@'.join(dic['audio_filepath'].split('/')[-3:])
            out_embeddings[uniq_name] = all_embs[i]

    embedding_dir = os.path.join(embedding_dir, 'embeddings')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir, exist_ok=True)

    prefix = manifest_file.split('/')[-1].rsplit('.', 1)[-2]

    name = os.path.join(embedding_dir, prefix)
    embeddings_file = name + '_embeddings.pt'
    torch.save(out_embeddings, embeddings_file)
    logging.info("Saved embedding files to {}".format(embedding_dir))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='titanet_large',
        required=False,
        help="path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would be downloaded from NGC and used to extract embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="batch size",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default='./',
        required=False,
        help="path to directory where embeddings need to stored default:'./'",
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.model_path.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.model_path}")
        speaker_model = EncDecSpeakerLabelModel.restore_from(restore_path=args.model_path)
    elif args.model_path.endswith('.ckpt'):
        speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(checkpoint_path=args.model_path)
    else:
        speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
        logging.info(f"using pretrained titanet_large speaker model from NGC")

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
        logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")

    get_embeddings(
        speaker_model, args.manifest, batch_size=args.batch_size, embedding_dir=args.embedding_dir, device=device
    )


if __name__ == '__main__':
    main()
