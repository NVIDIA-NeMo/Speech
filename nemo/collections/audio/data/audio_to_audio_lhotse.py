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
import os

import numpy as np
import torch
from lhotse import AudioSource, CutSet, Recording
from lhotse.array import Array
from lhotse.audio import info
from lhotse.cut import MixedCut
from lhotse.dataset.collation import collate_audio, collate_custom_field
from lhotse.serialization import load_jsonl

from nemo.collections.common.parts.preprocessing.manifest import get_full_path

INPUT_CHANNEL_SELECTOR = "input_channel_selector"
TARGET_CHANNEL_SELECTOR = "target_channel_selector"
REFERENCE_CHANNEL_SELECTOR = "reference_channel_selector"
LHOTSE_TARGET_CHANNEL_SELECTOR = "target_recording_channel_selector"
LHOTSE_REFERENCE_CHANNEL_SELECTOR = "reference_recording_channel_selector"


class LhotseAudioToTargetDataset(torch.utils.data.Dataset):
    """
    A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal.

    .. note:: This is a Lhotse variant of :class:`nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`.
    """

    TARGET_KEY = "target_recording"
    REFERENCE_KEY = "reference_recording"
    EMBEDDING_KEY = "embedding_vector"

    def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor]:
        # In the rare case, the collate_audio function would raise the FileSeek error when loading .flac (https://github.com/bastibe/python-soundfile/issues/274)
        # A workaround is to use fault_tolerant and skip failed data, resulting in a smaller batch size for the few problematic cases.
        src_audio, src_audio_lens, retained_padded_cuts = collate_audio(cuts, fault_tolerant=True)
        ans = {
            "input_signal": src_audio,
            "input_length": src_audio_lens,
        }
        # keep only the first non-padding cuts
        retained_cuts = [
            cut._first_non_padding_cut if isinstance(cut, MixedCut) else cut for cut in retained_padded_cuts
        ]
        retained_cuts = CutSet.from_cuts(retained_cuts)

        if _key_available(retained_cuts, self.TARGET_KEY):
            # TODO: use fault_tolerant=True for robust loading of target
            tgt_audio, tgt_audio_lens = collate_audio(retained_cuts, recording_field=self.TARGET_KEY)
            ans.update(target_signal=tgt_audio, target_length=tgt_audio_lens)
        if _key_available(retained_cuts, self.REFERENCE_KEY):
            # TODO: use fault_tolerant=True for robust loading of target
            ref_audio, ref_audio_lens = collate_audio(retained_cuts, recording_field=self.REFERENCE_KEY)
            ans.update(reference_signal=ref_audio, reference_length=ref_audio_lens)
        if _key_available(cuts, self.EMBEDDING_KEY):
            emb = collate_custom_field(retained_cuts, field=self.EMBEDDING_KEY)
            ans.update(embedding_signal=emb)
        return ans


def _key_available(cuts: CutSet, key: str) -> bool:
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        if cut.custom is not None and key in cut.custom:
            continue
        else:
            return False
    return True


def create_recording(path_or_paths: str | list[str]) -> Recording:
    if isinstance(path_or_paths, list):
        cur_channel_idx = 0
        sources = []
        infos = []
        for p in path_or_paths:
            i = info(p)
            infos.append(i)
            sources.append(
                AudioSource(type="file", channels=list(range(cur_channel_idx, cur_channel_idx + i.channels)), source=p)
            )
            cur_channel_idx += i.channels
        assert all(
            i.samplerate == infos[0].samplerate for i in infos[1:]
        ), f"Mismatched sampling rates for individual audio files in: {path_or_paths}"
        recording = Recording(
            id=path_or_paths[0],
            sources=sources,
            sampling_rate=infos[0].samplerate,
            num_samples=infos[0].frames,
            duration=infos[0].duration,
            channel_ids=list(range(0, cur_channel_idx)),
        )
    else:
        recording = Recording.from_file(path_or_paths)
    return recording


def create_array(path: str) -> Array:
    assert path.endswith(".npy"), f"Currently only conversion of numpy files is supported (got: {path})"
    arr = np.load(path)
    parent, path = os.path.split(path)
    return Array(
        storage_type="numpy_files",
        storage_path=parent,
        storage_key=path,
        shape=list(arr.shape),
    )


def convert_manifest_nemo_to_lhotse(
    input_manifest: str,
    output_manifest: str,
    input_key: str = 'input_filepath',
    target_key: str = 'target_filepath',
    reference_key: str = 'reference_filepath',
    embedding_key: str = 'embedding_filepath',
    force_absolute_paths: bool = False,
):
    """
    Convert an audio-to-audio manifest from NeMo format to Lhotse format.

    Args:
        input_manifest: Path to the input NeMo manifest.
        output_manifest: Path where we'll write the output Lhotse manifest (supported extensions: .jsonl.gz and .jsonl).
        input_key: Key of the input recording, mapped to Lhotse's 'Cut.recording'.
        target_key: Key of the target recording, mapped to Lhotse's 'Cut.target_recording'.
        reference_key: Key of the reference recording, mapped to Lhotse's 'Cut.reference_recording'.
        embedding_key: Key of the embedding, mapped to Lhotse's 'Cut.embedding_vector'.
        force_absolute_paths: If True, the paths in the output manifest will be absolute.
    """
    with CutSet.open_writer(output_manifest) as writer:
        for item in load_jsonl(input_manifest):

            # Create Lhotse recording and cut object, apply offset and duration slicing if present.
            item_input_key = item.pop(input_key)
            recording = create_recording(get_full_path(audio_file=item_input_key, manifest_file=input_manifest))
            cut = recording.to_cut().truncate(duration=item.pop("duration"), offset=item.pop("offset", 0.0))

            _as_relative(cut.recording, item_input_key, enabled=not force_absolute_paths)

            if (channels := item.pop(INPUT_CHANNEL_SELECTOR, None)) is not None:
                if cut.num_channels == 1:
                    assert (
                        len(channels) == 1 and channels[0] == 0
                    ), f"The input recording has only a single channel, but manifest specified {INPUT_CHANNEL_SELECTOR}={channels}"
                else:
                    cut = cut.with_channels(channels)

            if target_key in item:
                item_target_key = item.pop(target_key)
                cut.target_recording = create_recording(
                    get_full_path(audio_file=item_target_key, manifest_file=input_manifest)
                )

                _as_relative(cut.target_recording, item_target_key, enabled=not force_absolute_paths)

                if (channels := item.pop(TARGET_CHANNEL_SELECTOR, None)) is not None:
                    if cut.target_recording.num_channels == 1:
                        assert (
                            len(channels) == 1 and channels[0] == 0
                        ), f"The target recording has only a single channel, but manifest specified {TARGET_CHANNEL_SELECTOR}={channels}"
                    else:
                        cut = cut.with_custom(LHOTSE_TARGET_CHANNEL_SELECTOR, channels)

            if reference_key in item:
                item_reference_key = item.pop(reference_key)
                cut.reference_recording = create_recording(
                    get_full_path(audio_file=item_reference_key, manifest_file=input_manifest)
                )

                _as_relative(cut.reference_recording, item_target_key, enabled=not force_absolute_paths)

                if (channels := item.pop(REFERENCE_CHANNEL_SELECTOR, None)) is not None:
                    if cut.reference_recording.num_channels == 1:
                        assert (
                            len(channels) == 1 and channels[0] == 0
                        ), f"The reference recording has only a single channel, but manifest specified {REFERENCE_CHANNEL_SELECTOR}={channels}"
                    else:
                        cut = cut.with_custom(LHOTSE_REFERENCE_CHANNEL_SELECTOR, channels)

            if embedding_key in item:
                item_embedding_key = item.pop(embedding_key)
                cut.embedding_vector = create_array(
                    get_full_path(audio_file=item_embedding_key, manifest_file=input_manifest)
                )

                if not force_absolute_paths:
                    # Use the same format for paths as in the original manifest
                    cut.embedding_vector.storage_path = ""
                    cut.embedding_vector.storage_key = item_embedding_key

            if item:
                cut.custom.update(item)  # any field that's still left goes to custom fields

            writer.write(cut)


def _as_relative(recording: Recording, paths: list[str] | str, enabled: bool) -> None:
    if not enabled:
        return
    if isinstance(paths, str):
        paths = [paths]
    assert len(recording.sources) == len(
        paths
    ), f"Mismatched number of sources for lhotse Recording and the override list. Got {recording=} and {paths=}"
    for source, path in zip(recording.sources, paths):
        source.source = path
