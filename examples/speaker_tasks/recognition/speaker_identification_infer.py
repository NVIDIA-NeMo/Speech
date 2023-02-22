# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.core.config import hydra_runner
from nemo.utils import logging

seed_everything(42)


@hydra_runner(config_path="conf", config_name="speaker_identification_infer")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    enrollment_manifest = cfg.data.enrollment_manifest
    test_manifest = cfg.data.test_manifest
    out_manifest = cfg.data.out_manifest
    sample_rate = cfg.data.sample_rate

    backend = cfg.backend.backend_model.lower()

    featurizer = WaveformFeaturizer(sample_rate=sample_rate)
    dataset = AudioToSpeechLabelDataset(manifest_filepath=enrollment_manifest, labels=None, featurizer=featurizer)
    enroll_id2label = dataset.id2label

    if backend == 'cosine_similarity':
        model_path = cfg.backend.cosine_similarity.model_path
        batch_size = cfg.backend.cosine_similarity.batch_size
        if model_path.endswith('.nemo'):
            speaker_model = EncDecSpeakerLabelModel.restore_from(model_path)
        else:
            speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_path)

        enroll_embs, _, enroll_truelabels, _ = speaker_model.batch_inference(
            enrollment_manifest, batch_size, sample_rate, device=device,
        )

        test_embs, _, _, _ = speaker_model.batch_inference(test_manifest, batch_size, sample_rate, device=device,)

        # length normalize
        enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
        test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))

        # reference embedding
        reference_embs = []
        keyslist = list(enroll_id2label.values())
        for label_id in keyslist:
            indices = np.where(enroll_truelabels == label_id)
            embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices)
            reference_embs.append(embedding)

        reference_embs = np.asarray(reference_embs)

        scores = np.matmul(test_embs, reference_embs.T)
        matched_labels = scores.argmax(axis=-1)

    elif backend == 'neural_classifier':
        model_path = cfg.backend.neural_classifier.model_path
        batch_size = cfg.backend.neural_classifier.batch_size

        if model_path.endswith('.nemo'):
            speaker_model = EncDecSpeakerLabelModel.restore_from(model_path)
        else:
            speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_path)

        if speaker_model.decoder.final.out_features != len(enroll_id2label):
            raise ValueError(
                "number of labels mis match. Make sure you trained or finetuned neural classifier with labels from enrollement manifest_filepath"
            )

        _, test_logits, _, _ = speaker_model.batch_inference(test_manifest, batch_size, sample_rate, device=device,)
        matched_labels = test_logits.argmax(axis=-1)

    with open(test_manifest, 'rb') as f1, open(out_manifest, 'w', encoding='utf-8') as f2:
        lines = f1.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            item = json.loads(line)
            item['infer'] = enroll_id2label[matched_labels[idx]]
            json.dump(item, f2)
            f2.write('\n')

    logging.info("Inference labels have been written to {} manifest file".format(out_manifest))


if __name__ == '__main__':
    main()
