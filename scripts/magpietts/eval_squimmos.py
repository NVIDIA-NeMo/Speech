# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
# limitations under the License.from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import argparse
import os

import librosa
import numpy as np
import scipy.stats as stats
import torch


def find_sample_audios(audio_dir):
    file_list = []
    for f in os.listdir(audio_dir):
        if "predicted_audio" in f and f.endswith(".wav"):
            audio_number = int(f.split("_")[-1].split(".wav")[0])
            file_list.append((audio_number, os.path.join(audio_dir, f)))
    file_list.sort()
    file_list = [t[1] for t in file_list]
    return file_list


def compute_mean_and_confidence_interval(measurements, confidence=0.95):
    mean = np.mean(measurements)
    std_err = stats.sem(measurements)

    confidence_interval = std_err * stats.t.ppf((1 + confidence) / 2, len(measurements) - 1)

    return "{:.4f} +/- {:.4f}".format(mean, confidence_interval), mean, confidence_interval


def main():
    parser = argparse.ArgumentParser(description='Evaluate Squim MOS')
    parser.add_argument('--exp_base_dir', type=str, default="/datap/misc/ContinuousEvalResults/NewTransformerKoelTTS")
    parser.add_argument(
        '--audio_dirs',
        type=str,
        default="svencoder_small_sp_ks3_onlyphoneme_epoch242_Temp0.6_Topk80_Cfg_False_1.0_libri_val",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    squim_mos_model = SQUIM_SUBJECTIVE.get_model().to(device)

    squim_score_list = []
    if args.audio_dirs == "all":
        audio_dirs = [d for d in os.listdir(args.exp_base_dir) if os.path.isdir(os.path.join(args.exp_base_dir, d))]
    else:
        audio_dirs = args.audio_dirs.split(",")
    out_file = os.path.join(args.exp_base_dir, "squim_mos_score.csv")
    for audio_dir in audio_dirs:
        print("Evaluating audio dir: ", audio_dir)
        audio_dir_path = os.path.join(args.exp_base_dir, audio_dir, "audio")
        audio_files = find_sample_audios(audio_dir_path)
        for audio_file in audio_files:
            pred_wav, sr = librosa.load(audio_file, sr=16000)
            pred_wav = torch.tensor(pred_wav).to(device).unsqueeze(0)

            gt_path = audio_file.replace("predicted_audio", "target_audio")
            gt_wav, sr = librosa.load(gt_path, sr=16000)
            gt_wav = torch.tensor(gt_wav).to(device).unsqueeze(0)
            with torch.no_grad():
                squm_mos_score = squim_mos_model(pred_wav, gt_wav)
                squim_score_list.append(squm_mos_score.item())

        mean_with_ci, mean, confidence_interval = compute_mean_and_confidence_interval(squim_score_list)
        # Add to audio_dir,mean_with_ci to csv
        with open(out_file, "a") as f:
            f.write(audio_dir + "," + mean_with_ci + "\n")
            print("Audio dir: ", audio_dir, "Mean with CI: ", mean_with_ci)
            print("Wrote to file: ", out_file)


if __name__ == "__main__":
    main()
