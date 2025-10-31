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
# limitations under the License.
import argparse
import json
import os
from pathlib import Path

### from nemo.collections.tts.models import AudioCodecModel
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.audio import RecordingSet
from lhotse.shar.writers import AudioTarWriter
from tqdm import tqdm


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(manifest, audio_root_path, out_shar_dir, shard_size=1000):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    num_shard = int(len(in_manifest) / shard_size)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    user_recordings = []
    answer_list = []
    instructions = []
    source_language = []
    target_language = []
    target_recordings = []
    for i, line in tqdm(enumerate(in_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech

        # User_Speech
        context_audio_path = line["context_audio_filepath"]

        user_recording = Recording.from_file(os.path.join(audio_root_path, context_audio_path))
        user_recordings.append(user_recording)

        # This are the context text, this could be different things like a simple instruction or details about speaker voice
        instructions.append(" ")

        # Language source
        if "lang" in line:
            language = line["lang"]
        elif "language" in line:
            language = line["language"]
        elif "Language:" in str(line["speaker"]):
            language = line["speaker"].split("Language:")[1].split(" ")[0]
        else:
            language = "en"

        source_language.append(language)

        # Loading agent audio and using only the extracted features as nd.array
        target_recordings.append(Recording.from_file(os.path.join(audio_root_path, line["audio_filepath"])))
        # Agent answer transcript
        answer_list.append(line["text"])
        # Language target
        target_language.append(language)

    print("Done extracting data from manifest")
    print(len(user_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))

    # Attach text
    for i, cut in tqdm(enumerate(cuts)):
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=cut.recording.duration,
                text=instructions[i],
                speaker="user",
                language=source_language[i].upper(),
            ),
        )
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=target_recordings[i].duration,
                text=answer_list[i],
                speaker="agent",
                language=target_language[i].upper(),
            ),
        )
        cut.target_audio = target_recordings[i]

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(out_shar_dir, fields={"recording": "wav"}, num_jobs=4, shard_size=shard_size)
    print(f"...share created")

    print(f"...Exporting target_audio to tar files")
    for i, path in tqdm(enumerate(exported["cuts"])):
        path = path[0]
        out_path = path.replace("cuts", "target_audio").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="flac") as writer:
            for cut in CutSet.from_file(path):
                writer.write(cut.id, cut.target_audio.load_audio(), manifest=cut.target_audio, sampling_rate=22050)
    print(f"...Exported target_audio to tar files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=str,
        default="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/manifests/hifitts__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json",
    )
    parser.add_argument(
        '--audio_root_path',
        type=str,
        default="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/hi_fi_tts_v0/",
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        default="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/tts_lhotse_datasets/hifitts/",
    )
    parser.add_argument(
        '--shard_size',
        type=int,
        default=1000,
    )

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"audio_root_path {args.audio_root_path}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.shard_size}")

    create_shar_from_manifest(
        manifest=args.manifest,
        audio_root_path=args.audio_root_path,
        out_shar_dir=args.out_shar_dir,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
