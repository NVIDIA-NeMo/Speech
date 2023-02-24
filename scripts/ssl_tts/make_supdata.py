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
# Example Run Command: python make_supdata.py --ssl_model_ckpt_path <PATH TO CKPT> --manifest_path <PATH TO MANIFEST>

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import hydra.utils
import librosa
import numpy as np
import torch
from omegaconf import open_dict
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import ssl_tts
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
from nemo.core.classes import Dataset
from nemo.utils import logging


class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        min_duration=0.5,
        max_duration=16.0,
        pad_multiple=1024,
        sample_rate=22050,
        sup_data_dir=None,
    ):
        self.data = []
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if record['duration'] < min_duration or record['duration'] > max_duration:
                        continue
                    self.data.append(json.loads(line))

        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])
        if sup_data_dir is not None:
            self.sup_data_dir = sup_data_dir
        else:
            self.sup_data_dir = os.path.join(self.base_data_dir, "sup_data")
        if not os.path.exists(self.sup_data_dir):
            os.makedirs(self.sup_data_dir)

        self.pad_multiple = pad_multiple
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath):
        features = AudioSegment.segment_from_file(
            audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
        )
        audio_samples = features.samples
        audio, audio_length = torch.tensor(audio_samples), torch.tensor(audio_samples.shape[0]).long()

        # pad audio to a multiple of self.pad_multiple
        if audio.shape[0] % self.pad_multiple != 0:
            audio = torch.cat(
                [audio, torch.zeros(self.pad_multiple - audio.shape[0] % self.pad_multiple, dtype=torch.float)]
            )
            audio_length = torch.tensor(audio.shape[0]).long()

        return audio, audio_length

    def pad_collate_fn(self, batch):
        final_batch = {}
        for row in batch:
            for key in row:
                if key not in final_batch:
                    final_batch[key] = []
                final_batch[key].append(row[key])

        max_audio_len = max([_audio_len.item() for _audio_len in final_batch["audio_len"]])

        audios_padded = []
        for audio in final_batch["audio"]:
            audio_padded = torch.nn.functional.pad(audio, (0, max_audio_len - audio.size(0)), value=0)
            audios_padded.append(audio_padded)

        final_batch["audio"] = audios_padded
        for key in final_batch:
            if key not in ["rel_audio_path_as_text_id", "wav_path"]:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
        speaker = torch.tensor(sample["speaker"]).long()

        audio, audio_length = self._get_wav_from_filepath(sample["audio_filepath"])

        return {
            "audio": audio,
            "audio_len": audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "wav_path": sample["audio_filepath"],
            "speaker": speaker,
        }


def segment_wav(wav, segment_length, segment_hop_size, min_segment_length):
    if len(wav) < segment_length:
        pad = torch.zeros(segment_length - len(wav))
        segment = torch.cat([wav, pad])
        return [segment]
    else:
        si = 0
        segments = []
        while si < len(wav) - min_segment_length:
            segment = wav[si : si + segment_length]
            if len(segment) < segment_length:
                pad = torch.zeros(segment_length - len(segment))
                segment = torch.cat([segment, pad])
            segments.append(segment)
            si += segment_hop_size
        return segments


def segment_batch(batch, segment_length=44100, segment_hop_size=22050, min_segment_length=22050):
    all_segments = []
    segment_indices = []
    si = 0
    for bidx in range(len(batch['audio'])):
        audio = batch['audio'][bidx]
        audio_length = batch['audio_len'][bidx]
        audio_actual = audio[:audio_length]
        audio_segments = segment_wav(audio_actual, segment_length, segment_hop_size, min_segment_length)
        all_segments += audio_segments
        segment_indices.append((si, si + len(audio_segments) - 1))
        si += len(audio_segments)

    return torch.stack(all_segments), segment_indices


def get_mel_spectrogram(fb, wav, stft_params):
    EPSILON = 1e-9
    window_fn = torch.hann_window

    spec = torch.stft(
        input=wav,
        n_fft=stft_params['n_fft'],  # 1024
        hop_length=stft_params['hop_length'],  # 256
        win_length=stft_params['win_length'],  # 1024
        window=window_fn(stft_params['win_length'], periodic=False).to(torch.float).to('cuda') if window_fn else None,
        return_complex=True,
        center=True,
    )

    if spec.dtype in [torch.cfloat, torch.cdouble]:
        spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + EPSILON)

    mel = torch.matmul(fb.to(spec.dtype), spec)
    log_mel = torch.log(torch.clamp(mel, min=torch.finfo(mel.dtype).tiny))

    return log_mel


def load_wav(wav_path, sample_rate=22050, pad_multiple=1024):
    wav = AudioSegment.segment_from_file(wav_path, target_sr=sample_rate, n_segments=-1, trim=False,).samples

    if wav.shape[0] % pad_multiple != 0:
        wav = np.concatenate([wav, np.zeros(pad_multiple - wav.shape[0] % pad_multiple)])
    wav = wav[:-1]

    return wav


def save_pitch_contour(record):
    wav_path = record['wav_path']
    wav_text_id = record['wav_id']
    sup_data_dir = record['sup_data_dir']
    stft_params = record['stft_params']
    wav = load_wav(wav_path, stft_params['sample_rate'], stft_params['pad_multiple'])
    pitch_contour_fn = f"pitch_contour_{wav_text_id}.pt"
    pitch_contour_fp = os.path.join(sup_data_dir, pitch_contour_fn)

    f0, _, _ = librosa.pyin(
        wav,
        fmin=librosa.note_to_hz('C2'),
        fmax=stft_params['yin_fmax'],
        frame_length=stft_params['win_length'],
        hop_length=stft_params['hop_length'],
        sr=stft_params['sample_rate'],
        center=True,
        fill_na=0.0,
    )

    pitch_contour = torch.tensor(f0, dtype=torch.float32)
    torch.save(pitch_contour, pitch_contour_fp)
    logging.info("saved {}".format(pitch_contour_fp))

    return pitch_contour


def compute_pitch_stats(records):
    def _is_valid_pitch(pitch_mean, pitch_std):
        c1 = pitch_mean > 0 and pitch_mean < 1000
        c2 = pitch_std > 0 and pitch_std < 1000
        return c1 and c2

    speaker_wise_pitch_contours = {}
    for item in records:
        wav_id = item['wav_id']
        speaker = item['speaker']
        sup_data_dir = item['sup_data_dir']
        pitch_contour_fn = f"pitch_contour_{wav_id}.pt"
        pitch_contour_fp = os.path.join(sup_data_dir, pitch_contour_fn)
        if speaker not in speaker_wise_pitch_contours:
            speaker_wise_pitch_contours[speaker] = []
        speaker_wise_pitch_contours[speaker].append(pitch_contour_fp)

    speaker_pitch_stats = {}
    for speaker in speaker_wise_pitch_contours:
        non_zero_pc = []
        for pitch_contour_fp in speaker_wise_pitch_contours[speaker][:50]:
            pitch_contour = torch.load(pitch_contour_fp)
            pitch_contour_nonzero = pitch_contour[pitch_contour != 0]
            if len(pitch_contour_nonzero) > 0:
                non_zero_pc.append(pitch_contour_nonzero)

        if len(non_zero_pc) > 0:
            non_zero_pc = torch.cat(non_zero_pc)
            pitch_mean = non_zero_pc.mean().item()
            pitch_std = non_zero_pc.std().item()
            valid = True

            if not _is_valid_pitch(pitch_mean, pitch_std):
                logging.warning("invalid pitch: {}".format(speaker))
                pitch_mean = 212.0
                pitch_std = 70.0
                valid = "False"
        else:
            logging.warning("could not find pitch contour for speaker {}".format(speaker))
            valid = "False"
            pitch_mean = 212.0
            pitch_std = 70.0

        speaker_pitch_stats[speaker] = {"pitch_mean": pitch_mean, "pitch_std": pitch_std, "valid": valid}

    with open(os.path.join(sup_data_dir, "speaker_pitch_stats.json"), "w") as f:
        json.dump(speaker_pitch_stats, f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument(
        '--ssl_model_ckpt_path', type=str, required=True,
    )
    parser.add_argument('--manifest_paths', type=str, required=True)
    parser.add_argument('--sup_data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ssl_content_emb_type', type=str, default="embedding_and_probs")
    parser.add_argument('--use_unique_tokens', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pool_workers', type=int, default=30)
    parser.add_argument('--compute_pitch_contours', type=int, default=1)
    parser.add_argument('--num_pitch_per_speaker', type=int, default=None)  # saves time.
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--pad_multiple', type=int, default=1024)
    parser.add_argument('--ssl_downsampling_factor', type=int, default=4)
    parser.add_argument('--stft_n_fft', type=int, default=1024)
    parser.add_argument('--stft_hop_length', type=int, default=256)
    parser.add_argument('--stft_win_length', type=int, default=1024)
    parser.add_argument('--stft_n_mel', type=int, default=80)
    parser.add_argument('--stft_fmin', type=int, default=0)
    parser.add_argument('--stft_fmax', type=int, default=8000)
    parser.add_argument('--yin_fmax', type=int, default=500)
    parser.add_argument('--segment_length', type=int, default=44100)
    parser.add_argument('--segment_hop_size', type=int, default=22050)
    parser.add_argument('--min_segment_length', type=int, default=22050)

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    manifest_paths = args.manifest_paths.split(",")
    ssl_model_ckpt_path = args.ssl_model_ckpt_path

    dataset = AudioDataset(
        manifest_paths, pad_multiple=args.pad_multiple, sample_rate=args.sample_rate, sup_data_dir=args.sup_data_dir
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.pad_collate_fn,
        num_workers=args.num_workers,
    )

    ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(ssl_model_ckpt_path, strict=False)
    with open_dict(ssl_model.cfg):
        ssl_model.cfg.preprocessor.exact_pad = True
    ssl_model.preprocessor = hydra.utils.instantiate(ssl_model.cfg.preprocessor)
    ssl_model.preprocessor_disentangler = ssl_model.preprocessor
    ssl_model.eval()
    ssl_model.to(device)

    sample_rate = args.sample_rate
    stft_params = {
        "n_fft": args.stft_n_fft,
        "hop_length": args.stft_hop_length,
        "win_length": args.stft_win_length,
        "n_mel": args.stft_n_mel,
        "sample_rate": sample_rate,
        "pad_multiple": args.pad_multiple,
        "fmin": args.stft_fmin,
        "fmax": args.stft_fmax,
        "yin_fmax": args.yin_fmax,
    }

    fb = (
        torch.tensor(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=stft_params['n_fft'],
                n_mels=stft_params['n_mel'],
                fmin=stft_params['fmin'],
                fmax=stft_params['fmax'],
            ),
            dtype=torch.float,
        )
        .unsqueeze(0)
        .to(device)
    )

    st = time.time()
    bidx = 0
    wav_and_id_list = []

    for batch in tqdm(dataloader):
        bidx += 1
        with torch.no_grad():
            (
                _,
                _,
                batch_content_embedding,
                batch_content_log_probs,
                batch_encoded_len,
            ) = ssl_model.forward_for_export(
                input_signal=batch['audio'].to(device),
                input_signal_length=batch['audio_len'].to(device),
                normalize_content=True,
            )

            batch_mel_specs = get_mel_spectrogram(fb, batch['audio'][:, :-1].to(device), stft_params)
            audio_segmented, segment_indices = segment_batch(
                batch, args.segment_length, args.segment_hop_size, args.min_segment_length
            )
            audio_seg_len = torch.tensor([len(segment) for segment in audio_segmented]).to(device).long()

            _, batch_speaker_embeddings, _, _, _ = ssl_model.forward_for_export(
                input_signal=audio_segmented.to(device), input_signal_length=audio_seg_len, normalize_content=True,
            )

            for idx in range(batch['audio'].shape[0]):
                _speaker = batch['speaker'][idx].item()
                wav_path = batch['wav_path'][idx]

                wav_id = batch['rel_audio_path_as_text_id'][idx]
                wav_and_id_list.append((wav_path, wav_id, _speaker))
                content_embedding = batch_content_embedding[idx].detach()
                content_log_probs = batch_content_log_probs[:, idx, :].detach()  # (content lob prob is (t, b, c))
                encoded_len = batch_encoded_len[idx].detach()
                content_embedding = content_embedding[: encoded_len.item()]
                content_embedding = content_embedding.t()
                content_log_probs = content_log_probs[: encoded_len.item()]
                content_log_probs = content_log_probs.t()
                content_probs = torch.exp(content_log_probs)

                duration = torch.ones(content_embedding.shape[1]) * args.ssl_downsampling_factor

                bsi_start = segment_indices[idx][0]
                bsi_end = segment_indices[idx][1]
                speaker_embedding = torch.mean(batch_speaker_embeddings[bsi_start : bsi_end + 1], dim=0)

                l2_norm = torch.norm(speaker_embedding, p=2)
                speaker_embedding = speaker_embedding / l2_norm

                if args.ssl_content_emb_type == "probs":
                    # content embedding is only character probabilities
                    final_content_embedding = content_probs
                elif args.ssl_content_emb_type == "embedding":
                    # content embedding is only output of content head of SSL backbone
                    final_content_embedding = content_embedding
                elif args.ssl_content_emb_type == "log_probs":
                    # content embedding is only log of character probabilities
                    final_content_embedding = content_log_probs
                elif args.ssl_content_emb_type == "embedding_and_probs":
                    # content embedding is the concatenation of character probabilities and output of content head of SSL backbone
                    final_content_embedding = torch.cat([content_embedding, content_probs], dim=0)

                if args.use_unique_tokens == 1:
                    # group content embeddings with same predicted token (by averaging) and add the durations of the grouped embeddings
                    # Eg. By default each content embedding corresponds to 4 frames of spectrogram (ssl_downsampling_factor)
                    # If we group 3 content embeddings, the duration of the grouped embedding will be 12 frames.
                    # This is useful for adapting the duration during inference based on the speaker.
                    token_predictions = torch.argmax(content_probs, dim=0)
                    content_buffer = [final_content_embedding[:, 0]]
                    unique_content_embeddings = []
                    unique_tokens = []
                    durations = []
                    for _t in range(1, final_content_embedding.shape[1]):
                        if token_predictions[_t] == token_predictions[_t - 1]:
                            content_buffer.append(final_content_embedding[:, _t])
                        else:
                            durations.append(len(content_buffer) * args.ssl_downsampling_factor)
                            unique_content_embeddings.append(torch.mean(torch.stack(content_buffer), dim=0))
                            content_buffer = [final_content_embedding[:, _t]]
                            unique_tokens.append(token_predictions[_t].item())

                    if len(content_buffer) > 0:
                        durations.append(len(content_buffer) * args.ssl_downsampling_factor)
                        unique_content_embeddings.append(torch.mean(torch.stack(content_buffer), dim=0))
                        unique_tokens.append(token_predictions[_t].item())

                    unique_content_embedding = torch.stack(unique_content_embeddings)
                    final_content_embedding = unique_content_embedding.t()
                    duration = torch.tensor(durations).float()

                mel_len = int(batch['audio_len'][idx].item() / stft_params['hop_length'])
                item_mel = batch_mel_specs[idx][:, :mel_len]

                wav_text_id = batch["rel_audio_path_as_text_id"][idx]
                content_emb_fn = f"{args.ssl_content_emb_type}_content_embedding_{wav_text_id}.pt"
                speaker_emb_fn = f"speaker_embedding_{wav_text_id}.pt"
                duration_fn = f"duration_embedding_{wav_text_id}.pt"  # embedding just for namesake
                content_emb_fp = os.path.join(dataset.sup_data_dir, content_emb_fn)
                speaker_emb_fp = os.path.join(dataset.sup_data_dir, speaker_emb_fn)
                duration_fp = os.path.join(dataset.sup_data_dir, duration_fn)

                mel_spec_fn = f"mel_spec_{wav_text_id}.pt"
                mel_spec_fp = os.path.join(dataset.sup_data_dir, mel_spec_fn)

                torch.save(item_mel.cpu(), mel_spec_fp)
                torch.save(final_content_embedding.cpu(), content_emb_fp)
                torch.save(speaker_embedding.cpu(), speaker_emb_fp)
                torch.save(duration.cpu(), duration_fp)

            et = time.time()
            logging.info(
                "Processed Batch {} of {} | Time per batch: {:.4f} s".format(
                    bidx + 1, len(dataloader), (et - st) / bidx
                )
            )

    if args.compute_pitch_contours == 1:
        speaker_wise_records = {}
        for row in wav_and_id_list:
            wav_path, wav_id, speaker = row
            if speaker not in speaker_wise_records:
                speaker_wise_records[speaker] = []
            speaker_wise_records[speaker].append(
                {
                    "wav_path": wav_path,
                    "wav_id": wav_id,
                    "sup_data_dir": dataset.sup_data_dir,
                    "stft_params": stft_params,
                    "speaker": speaker,
                }
            )

        filtered_records = []
        for speaker in speaker_wise_records:
            if args.num_pitch_per_speaker is not None:
                filtered_records += speaker_wise_records[speaker][: args.num_pitch_per_speaker]
            else:
                filtered_records += speaker_wise_records[speaker]

        with Pool(args.pool_workers) as p:
            p.map(save_pitch_contour, filtered_records)

        compute_pitch_stats(filtered_records)


if __name__ == '__main__':
    main()
