# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shutil

import soundfile as sf
import torch

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.utils import logging

"""
Utilities for logging evaluation results of SpeechLM2 collection models.

This file provides helper functionality for saving audio outputs and structured
metadata during evaluation or inference of Duplex speech-to-speech / TTS models.
It is primarily responsible for:

    - Writing predicted waveforms to disk.
    - Merging user and model audio into multi-channel WAV files for analysis.
    - Exporting metadata (reference text, predictions, ASR output) into JSONL format.
    - Saving auxiliary debug artifacts such as:
        * teacher-forced predictions,
        * reference audio,
        * trimmed outputs,
        * end-of-utterance (EOU) probability signals.

Unlike other files in this directory, which focus on metric evaluation, this module
is dedicated to persisting model outputs — including predicted audio samples and
their associated metadata — for later inspection and analysis.

Key abstraction:
    - `ResultsLogger`: A lightweight utility class that manages audio dumping
      and metadata bookkeeping across inference batches.
"""


def safe_remove_path(path):
    shutil.rmtree(path, ignore_errors=True)


class ResultsLogger:
    """
    Saves audios and a json file with the model outputs.
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.audio_save_path = os.path.join(save_path, "pred_wavs")
        os.makedirs(self.audio_save_path, exist_ok=True)
        self.matadata_save_path = os.path.join(save_path, "metadatas")
        os.makedirs(self.matadata_save_path, exist_ok=True)

    def reset(self):
        # ensures that we are cleaning the metadata files
        metadata_files = os.listdir(self.matadata_save_path)
        for f in metadata_files:
            open(os.path.join(self.matadata_save_path, f), 'w').close()

        # clean out any existing .wav predictions safely
        try:
            audio_files = os.listdir(self.audio_save_path)
            for f in audio_files:
                if f.lower().endswith(".wav"):
                    try:
                        os.remove(os.path.join(self.audio_save_path, f))
                    except FileNotFoundError:
                        pass  # already gone
                    except Exception:
                        logging.warning(f"Failed to remove audio file {f} during reset.", stack_info=False)
        except FileNotFoundError:
            # directory somehow missing: recreate it
            os.makedirs(self.audio_save_path, exist_ok=True)

        return self

    @staticmethod
    def merge_and_save_audio(
        out_audio_path: str, pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor, user_audio_sr: int
    ) -> None:
        # if user_audio is None ignore it
        if user_audio is not None:
            user_audio = resample(user_audio.float(), user_audio_sr, pred_audio_sr)
            T1, T2 = pred_audio.shape[0], user_audio.shape[0]
            max_len = max(T1, T2)
            pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
            user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

            # combine audio in a multichannel audio
            combined_wav = torch.cat(
                [
                    user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                    pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                ],
                dim=0,
            ).squeeze()

        else:
            combined_wav = pred_audio.unsqueeze(0).detach().cpu()

        # save audio
        os.makedirs(os.path.dirname(out_audio_path), exist_ok=True)
        sf.write(out_audio_path, combined_wav.numpy().astype('float32').T, pred_audio_sr)
        logging.info(f"Audio saved at: {out_audio_path}")

    def update(
        self,
        name: str,
        refs: list[str],
        hyps: list[str],
        asr_hyps: list[str],
        samples_id: list[str],
        pred_audio: torch.Tensor,
        pred_audio_sr: int,
        user_audio: torch.Tensor,
        user_audio_sr: int,
        target_audio: torch.Tensor = None,
        pred_audio_tf: torch.Tensor = None,
        pre_audio_trimmed: torch.Tensor = None,
        eou_pred: torch.Tensor = None,
        fps: float = None,
        results=None,
        tokenizer=None,
        reference_audio: torch.Tensor = None,
    ) -> None:

        out_json_path = os.path.join(self.matadata_save_path, f"{name}.json")
        out_dicts = []
        for i in range(len(refs)):
            # save audio
            sample_id = samples_id[i][:150]  # make sure that sample id is not too big
            out_audio_path = os.path.join(self.audio_save_path, f"{name}_{sample_id}.wav")
            self.merge_and_save_audio(
                out_audio_path,
                pred_audio[i],
                pred_audio_sr,
                user_audio[i] if user_audio is not None else None,
                user_audio_sr,
            )

            if pred_audio_tf is not None:
                out_audio_path_tf = out_audio_path.replace(".wav", "_tf.wav")
                self.merge_and_save_audio(
                    out_audio_path_tf,
                    pred_audio_tf[i],
                    pred_audio_sr,
                    user_audio[i] if user_audio is not None else None,
                    user_audio_sr,
                )

            if target_audio is not None:
                out_audio_path_gt = out_audio_path.replace(".wav", "_GT.wav")
                self.merge_and_save_audio(
                    out_audio_path_gt,
                    target_audio[i],
                    pred_audio_sr,
                    user_audio[i] if user_audio is not None else None,
                    user_audio_sr,
                )

            # create a wav with eou prediction for debug purposes
            if eou_pred is not None:
                out_audio_path_eou = os.path.join(self.audio_save_path, f"{name}_{sample_id}_eou.wav")
                repeat_factor = int(pred_audio_sr / fps)
                eou_pred_wav = (
                    eou_pred[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, repeat_factor)
                )  # (B, T, repeat_factor)
                eou_pred_wav = eou_pred_wav.view(1, -1)  # (B, T * repeat_factor)
                eou_pred_wav = eou_pred_wav.float() * 0.8  #  make 1 audible and keep 0 as total silence
                sf.write(
                    out_audio_path_eou,
                    eou_pred_wav.squeeze().unsqueeze(0).detach().cpu().numpy().astype('float32').T,
                    pred_audio_sr,
                )

            if pre_audio_trimmed is not None:
                out_audio_path_trimmed = os.path.join(self.audio_save_path, f"{name}_{sample_id}_pred_trimmed.wav")
                sf.write(
                    out_audio_path_trimmed,
                    pre_audio_trimmed[i].squeeze().unsqueeze(0).detach().cpu().numpy().astype('float32').T,
                    pred_audio_sr,
                )

            if reference_audio is not None:
                out_audio_path_ref = os.path.join(self.audio_save_path, f"{name}_{sample_id}_spk_reference.wav")
                sf.write(
                    out_audio_path_ref,
                    reference_audio[i].squeeze().unsqueeze(0).detach().cpu().numpy().astype('float32').T,
                    pred_audio_sr,
                )

            # cache metadata
            out_dict = {
                "target_text": refs[i],
                "pred_text": hyps[i],
                "speech_pred_transcribed": asr_hyps[i],
                "audio_path": os.path.relpath(out_audio_path, self.save_path),
            }
            if results is not None:
                if tokenizer is not None:
                    out_dict['tokens_text'] = " ".join(tokenizer.ids_to_tokens(results['tokens_text'][i]))
                else:
                    out_dict['tokens_text'] = results['tokens_text'][i].tolist()
            out_dicts.append(out_dict)
        # uses append here to avoid needs to cache
        with open(out_json_path, 'a+', encoding='utf-8') as fout:
            for out_dict in out_dicts:
                fout.write(json.dumps(out_dict, ensure_ascii=False, indent=4) + '\n')
                # json.dump(out_dict, fout)

        logging.info(f"Metadata file for {name} dataset updated at: {out_json_path}")
