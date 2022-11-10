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

import os
from pathlib import Path

import sox
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive
from sox import Transformer

MLS_URL = "https://dl.fbaipublicfiles.com/mls/mls_{language}.tar.gz"
TEST_DATA_PATH = str(Path(__file__).parents[2] / "tests" / "test_data" / "mls_{language}" / "data.tar.gz")


class CreateInitialManifestMLS(BaseParallelProcessor):
    """
    Downloads and unzips raw MLS data for the specified language, and creates an initial manifest using
    the transcripts provided in the raw data. 

    Args:
        language: the language of the data you wish to be downloaded. This will be used to format the 
            URL from which we attempt to download the data.
        download_dir: the directory where the downloaded data will be saved.
        data_split: the data split for which the initial manifest will be created.
        resampled_audio_dir: the directory where the resampled (16kHz) wav files will be stored.
        use_test_data: if `True`, will use the test data manifest located at `TEST_DATA_PATH` to carry out tests.
    """

    def __init__(
        self,
        language: str,
        download_dir: str,
        resampled_audio_dir: str,
        data_split: str,
        use_test_data: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language = language
        self.download_dir = Path(download_dir)
        self.data_split = data_split
        self.resampled_audio_dir = resampled_audio_dir
        self.use_test_data = use_test_data

        # will be initialized in self.prepare method
        self.audio_path_prefix = None
        self.transcription_file = None

    def prepare(self):
        """Downloading and extracting data (unless already done).

        If use_test_data is True, then will not download data, instead will
        copy the included test data (mainly useful for quick development or
        CI pipeline).
        """
        if self.use_test_data:
            test_data_path = TEST_DATA_PATH.format(language=self.language)
            data_folder = extract_archive(str(test_data_path), str(self.download_dir))
        else:
            url = MLS_URL.format(language=self.language)
            download_file(url, str(self.download_dir))
            data_folder = extract_archive(str(self.download_dir / os.path.basename(url)), str(self.download_dir))
        self.audio_path_prefix = str(Path(data_folder) / self.data_split / "audio")
        self.transcription_file = str(Path(data_folder) / self.data_split / "transcripts.txt")

    def read_manifest(self):
        if self.transcription_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        with open(self.transcription_file, "rt", encoding="utf8") as fin:
            dataset_entries = fin.readlines()

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        if len(data_entry.split("\t")) != 2:
            raise RuntimeError(f"have more than one tab in line {data_entry}")

        utt_id, text = data_entry.split("\t")
        transcript_text = text.strip()

        src_flac_path = os.path.join(self.audio_path_prefix, *utt_id.split("_")[:2], utt_id + ".flac")
        tgt_wav_path = os.path.join(self.resampled_audio_dir, *utt_id.split("_")[:2], utt_id + ".wav")

        if not os.path.exists(os.path.dirname(tgt_wav_path)):
            os.makedirs(os.path.dirname(tgt_wav_path), exist_ok=True)
        if not os.path.exists(tgt_wav_path):
            Transformer().build(src_flac_path, tgt_wav_path)

        data = {
            "audio_filepath": tgt_wav_path,
            "duration": float(sox.file_info.duration(tgt_wav_path)),
            "text": transcript_text,
        }

        return [DataEntry(data=data)]
