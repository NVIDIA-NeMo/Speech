# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#

import argparse
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from yt_dlp import YoutubeDL


def clean_video_folder(video_folder):
    """Remove non-11-character.mp4 files from the video folder."""
    for video in video_folder.glob('*.mp4'):
        if len(video.stem) != 11:
            video.unlink()


def get_video_duration(video_path):
    """Function to get the duration of a video using ffprobe."""
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        video_path,
    ]
    duration = subprocess.check_output(cmd).strip()
    return float(duration)


def prepare_dataset(source, video_folder, chunk_length=120):
    """Prepare dataset from source JSON and download videos."""

    with open(source, 'r') as f:
        data = json.load(f)

    dataset = {}
    video_missing_counter = 0
    total_videos = len(data.keys())

    for key, value in tqdm(data.items()):
        if not os.path.exists(os.path.join(video_folder, key + ".mp4")):
            print(f"Video if {key} does not exist")
            video_missing_counter += 1
            continue

        duration = value['duration']
        timestamps = value['timestamps']
        sentences = value['sentences']

        # Videos are too long, sliding window of 2 minutes
        video_begin = 0
        video_end = duration
        new_data = {}
        counter = 0

        # We need some negative samples, where timestamps doesn't have any events
        last_end = 0
        empty_timestamps = []
        new_timestamps = []
        new_sentences = []
        for start, end in timestamps:
            if start - last_end > 25 and start - last_end < 35:
                empty_timestamps.append([last_end, start])
            last_end = end

        empty = 0.1 * len(timestamps)
        empty_timestamps = random.choices(empty_timestamps, k=min(int(empty), len(empty_timestamps)))

        for start, end in empty_timestamps:
            time_range = random.randint(20, end - start)
            new_start_time = random.randint(start, end - time_range)
            new_end_time = new_start_time + time_range
            new_data[f"{key}_{counter}"] = {
                "video_begin": new_start_time,
                "video_end": new_end_time,
                "timestamps": [],
                "sentences": [],
            }
            counter += 1

        # Normal samples
        for idx, ((start, end), sentence) in enumerate(zip(timestamps, sentences)):
            if idx == 0:
                video_begin = max(0, start - 5)
                video_end = end
            if end - video_begin > chunk_length:  # Use 2 minute chunks
                since_last_end = start - new_timestamps[-1][1] if new_timestamps else 0
                since_last_end = min(since_last_end, 10)
                pad = since_last_end // 2
                video_end = video_end + pad  # pad the end

                new_data[f"{key}_{counter}"] = {
                    "video_begin": video_begin,
                    "video_end": video_end,
                    "timestamps": new_timestamps,
                    "sentences": new_sentences,
                }
                counter += 1

                new_timestamps = []
                new_sentences = []
                video_begin = max(0, start - pad)  # pad the start

            new_timestamps.append([int(start), int(end)])
            new_sentences.append(sentence)
            video_end = end

            if idx == len(timestamps) - 1:
                video_end = min(duration, end + 5)

        if len(new_timestamps) > 0:
            new_data[f"{key}_{counter}"] = {
                "video_begin": video_begin,
                "video_end": video_end,
                "timestamps": new_timestamps,
                "sentences": new_sentences,
            }
            counter += 1

        dataset.update(new_data)

    print(f"Got {len(dataset)} videos")
    print(f"Total videos missing {video_missing_counter} out of total videos {total_videos}")
    return dataset


def crop_video(input_file, output_file, start_seconds, end_seconds):
    """Crop a video."""
    video = VideoFileClip(input_file)
    cropped_video = video.subclip(start_seconds, end_seconds)
    cropped_video = cropped_video.without_audio()
    cropped_video.write_videofile(output_file, codec='libx264', audio_codec='aac')


def process_video(key, value, video_folder, ignore=False):
    """Process a video."""
    video_begin = value['video_begin']
    video_end = value['video_end']
    timestamps = value['timestamps']
    sentences = value['sentences']
    key_orig = key.rsplit('_', 1)[0]
    video_chunk_dir = os.path.join(Path(video_folder), 'videos')
    os.makedirs(video_chunk_dir, exist_ok=True)
    video_path = os.path.join(Path(video_folder), 'videos_original', key_orig + ".mp4")
    save_video_path = os.path.join(video_chunk_dir, key + ".mp4")
    if ignore == False:
        crop_video(video_path, save_video_path, video_begin, video_end)

    try:
        # vr = decord.VideoReader(str(save_video_path))
        # duration = vr._num_frame / vr.get_avg_fps()
        duration = get_video_duration(save_video_path)
    except Exception as e:
        duration = video_end - video_begin
        print(f"Fallback to {duration} for {save_video_path} because {e}")

    timestamps = [[start - video_begin, end - video_begin] for start, end in timestamps]
    return key, {
        "duration": duration,
        "timestamps": timestamps,
        "sentences": sentences,
    }


def convert_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def download_video(video_id_output_folder):

    video_id, output_folder = video_id_output_folder
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    output_path = os.path.join(output_folder, f'{video_id}.mp4')

    # Check if the video has already been downloaded
    if os.path.exists(output_path):
        print(f"Video {video_id} already exists. Skipping...")
        return

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video {video_id} from {video_url}...")
            ydl.download([video_url])
            print(f"Video {video_id} downloaded successfully!")
    except Exception as e:
        print(f"Error downloading video {video_id} from {video_url}: {str(e)}")


def download_videos(json_file, output_folder):
    # Create the output folder if it doesn't exist
    output_folder = os.path.join(output_folder, 'videos_original')
    os.makedirs(output_folder, exist_ok=True)
    # List to store video ids
    video_ids = []
    # Load video ids from the JSON file
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_id = data['video_id']
            video_ids.append(video_id)

    # Create a pool of processes
    with Pool(10) as executor:
        # Map each video_id to the download_video function
        executor.map(download_video, [(video_id, output_folder) for video_id in video_ids])


def parse_dense_video_captions(original_file_path, output_dir):

    output_file_path = os.path.join(output_dir, 'train_original.json')

    data_dict = {}
    with open(original_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_id = data['video_id']
            step_data = data['step']
            duration = convert_to_seconds(step_data[str(len(step_data))]['endtime'])

            timestamps = []
            sentences = []

            for step in step_data.values():
                start_time = convert_to_seconds(step['startime'])
                end_time = convert_to_seconds(step['endtime'])
                timestamps.append([start_time, end_time])
                sentences.append(step['caption'])

            new_data = {'duration': duration, 'timestamps': timestamps, 'sentences': sentences}

            data_dict[video_id] = new_data

    with open(output_file_path, 'w') as nf:
        json.dump(data_dict, nf, indent=2)
    return output_file_path


def chunk(args, original_json):

    video_folder = os.path.join(args.output_dir, 'videos_original')
    video_folder = Path(video_folder)
    output_json = os.path.join(args.output_dir, 'train.json')
    clean_video_folder(video_folder)
    dataset = prepare_dataset(original_json, video_folder, args.chunk_length)
    fixed_data = {}

    # Function to process a single video
    def process_single_video(key, value):
        video_chunk_dir = os.path.join(args.output_dir, 'videos')
        os.makedirs(video_chunk_dir, exist_ok=True)
        save_video_path = os.path.join(video_chunk_dir, f"{key}.mp4")
        ignore = False
        if os.path.exists(save_video_path):
            print(f"Chunk for video {key} already exists. Skipping...")
            ignore = True

        key, value = process_video(key, value, args.output_dir, ignore=ignore)

        return key, value

    max_threads = 10  # Change this value to adjust the number of threads
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_single_video, key, value) for key, value in tqdm(dataset.items())]

        for future in tqdm(futures):
            result = future.result()
            if result[1]:  # If result is not None
                fixed_data[result[0]] = result[1]
    with open(output_json, 'w') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and create JSON file.")
    parser.add_argument("-d", "--download", type=bool, help="Whether to download videos.", default=False)
    parser.add_argument("-i", "--input_json", help="Path to the input JSON file.")
    parser.add_argument("-o", "--output_dir", help="Path to the output_dir.")
    parser.add_argument(
        "-l",
        "--chunk_length",
        type=int,
        help="Length of each chunked video in seconds (Default=120).",
        default=120,
        required=False,
    )

    args = parser.parse_args()

    if args.download:
        download_videos(args.input_json, args.output_dir)
        print(f"Videos have been downloaded to {args.output_dir}")
    else:
        original_json = parse_dense_video_captions(args.input_json, args.output_dir)
        chunk(args, original_json)
        print(f"Dataset has been prepared at {args.output_dir}")
