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

import argparse
import os
import pickle as pkl
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm


"""
This script faciliates to get EER % based on cosine-smilarity 
for Voxceleb dataset.

Args:
    trial_file str: path to voxceleb trial file
    emb : path to pickle file of embeddings dictionary (generated from spkr_get_emb.py)
    save_kaldi_emb: if required pass this argument to save kaldi embeddings for KALDI PLDA training later
    Note: order of audio files in manifest file should match the embeddings
"""


def get_acc(trial_file='', emb='', save_kaldi_emb=False):

    trial_score = open('trial_score.txt', 'w')
    dirname = os.path.dirname(trial_file)
    with open(emb, 'rb') as f:
        emb = pkl.load(f)
    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []

    # for each trials in trial file
    with open(trial_file, 'r') as f:
        tmp_file = f.readlines()
        for line in tqdm(tmp_file):
            line = line.strip()
            truth, x_speaker, y_speaker = line.split()

            x_speaker = x_speaker.split('/')
            x_speaker = '@'.join(x_speaker)

            y_speaker = y_speaker.split('/')
            y_speaker = '@'.join(y_speaker)

            X = emb[x_speaker]
            Y = emb[y_speaker]

            if save_kaldi_emb and x_speaker not in keys:
                keys.append(x_speaker)
                trial_embs.extend([X])

            if save_kaldi_emb and y_speaker not in keys:
                keys.append(y_speaker)
                trial_embs.extend([Y])

            score = np.dot(X, Y) / ((np.dot(X, X) * np.dot(Y, Y)) ** 0.5)
            score = (score + 1) / 2

            all_scores.append(score)
            trial_score.write(str(score) + "\t" + truth)
            truth = int(truth)
            all_keys.append(truth)

            trial_score.write('\n')
    trial_score.close()

    if save_kaldi_emb:
        np.save(dirname + '/all_embs_voxceleb.npy', np.asarray(trial_embs))
        np.save(dirname + '/all_ids_voxceleb.npy', np.asarray(keys))
        print("Saved KALDI PLDA related embeddings to {}".format(dirname))

    return np.asarray(all_scores), np.asarray(all_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_file", help="path to voxceleb trial file", type=str, required=True)
    parser.add_argument("--emb", help="path to numpy file of embeddings", type=str, required=True)
    parser.add_argument(
        "--save_kaldi_emb",
        help=":save kaldi embeddings for KALDI PLDA training later",
        required=False,
        action='store_true',
    )

    args = parser.parse_args()
    trial_file, emb, save_kaldi_emb = args.trial_file, args.emb, args.save_kaldi_emb

    y_score, y = get_acc(trial_file=trial_file, emb=emb, save_kaldi_emb=save_kaldi_emb)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    sys.stdout.write("{0:.2f}\n".format(eer * 100))
