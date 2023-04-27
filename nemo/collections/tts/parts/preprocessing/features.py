# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from torch import Tensor

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_audio_filepaths
from nemo.utils.decorators import experimental


@experimental
class Featurizer(ABC):
    @abstractmethod
    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        """
        Save feature value to disk for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features will be stored.
        """

    @abstractmethod
    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> Dict[str, Tensor]:
        """
        Read saved feature value for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features were stored by save().

        Returns:
            Dictionary of feature names to Tensors
        """


def _get_feature_filepath(manifest_entry: dict, audio_dir: Path, feature_dir: Path, feature_name: str) -> Path:
    """
    Get the absolute path for the feature file corresponding to the input manifest entry

    Example: audio_filepath "<audio_dir>/speaker1/audio1.wav" becomes
        feature_filepath "<feature_dir>/<feature_name>/speaker1/audio1.pt"
    """
    _, audio_filepath_rel = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
    feature_filepath = feature_dir / feature_name / audio_filepath_rel.with_suffix(".pt")
    return feature_filepath


def _save_pt_feature(
    feature_name: Optional[str], feature_tensor: Tensor, manifest_entry: Dict, audio_dir: Path, feature_dir: Path,
) -> None:
    """
    If feature_name is provided, save feature as .pt file.
    """
    if feature_name is None:
        return

    feature_filepath = _get_feature_filepath(
        manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir, feature_name=feature_name
    )
    feature_filepath.parent.mkdir(exist_ok=True, parents=True)
    torch.save(feature_tensor, feature_filepath)


def _load_pt_feature(
    feature_dict: Dict, feature_name: Optional[str], manifest_entry: Dict, audio_dir: Path, feature_dir: Path,
) -> None:
    """
    If feature_name is provided, load feature into feature_dict from .pt file.
    """

    if feature_name is None:
        return

    feature_filepath = _get_feature_filepath(
        manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir, feature_name=feature_name
    )
    feature_tensor = torch.load(feature_filepath)
    feature_dict[feature_name] = feature_tensor


class MelSpectrogramFeaturizer:
    def __init__(
        self,
        feature_name: str = "mel_spec",
        sample_rate: int = 22050,
        mel_dim: int = 80,
        win_length: int = 1024,
        hop_length: int = 256,
        lowfreq: int = 0,
        highfreq: int = 8000,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 1.0,
        mel_norm: Optional[Union[str, int]] = None,
    ) -> None:
        self.feature_name = feature_name
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length

        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            features=mel_dim,
            pad_to=1,
            n_window_size=win_length,
            n_window_stride=hop_length,
            window_size=False,
            window_stride=False,
            n_fft=win_length,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            mel_norm=mel_norm,
        )

    def compute_mel_spec(self, manifest_entry: dict, audio_dir: Path) -> Tensor:
        """
        Computes mel spectrogram for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            [spec_dim, T_spec] float tensor containing spectrogram features.
        """

        audio_filepath, _ = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(path=audio_filepath, sr=self.sample_rate)

        # [1, T_audio]
        audio_tensor = torch.tensor(audio[np.newaxis, :], dtype=torch.float32)
        # [1]
        audio_len_tensor = torch.tensor([audio.shape[0]], dtype=torch.int32)

        # [1, spec_dim, T_spec]
        spec_tensor, _ = self.preprocessor(input_signal=audio_tensor, length=audio_len_tensor)
        # [spec_dim, T_spec]
        spec_tensor = spec_tensor.detach()[0]

        return spec_tensor

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        spec_tensor = self.compute_mel_spec(manifest_entry=manifest_entry, audio_dir=audio_dir)
        _save_pt_feature(
            feature_name=self.feature_name,
            feature_tensor=spec_tensor,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> Dict[str, Tensor]:
        feature_dict = {}
        _load_pt_feature(
            feature_dict=feature_dict,
            feature_name=self.feature_name,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        return feature_dict


class EnergyFeaturizer:
    def __init__(self, spec_featurizer: MelSpectrogramFeaturizer, feature_name: str = "energy") -> None:
        self.feature_name = feature_name
        self.spec_featurizer = spec_featurizer

    def compute_energy(self, manifest_entry: dict, audio_dir: Path) -> Tensor:
        """
        Computes energy for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            [T_spec] float tensor containing energy features.
        """
        # [1, T_audio]
        spec = self.spec_featurizer.compute_mel_spec(manifest_entry=manifest_entry, audio_dir=audio_dir)
        # [T_audio]
        energy = torch.linalg.norm(spec, axis=0)

        return energy

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        energy_tensor = self.compute_energy(manifest_entry=manifest_entry, audio_dir=audio_dir)
        _save_pt_feature(
            feature_name=self.feature_name,
            feature_tensor=energy_tensor,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> Dict[str, Tensor]:
        feature_dict = {}
        _load_pt_feature(
            feature_dict=feature_dict,
            feature_name=self.feature_name,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        return feature_dict


class PitchFeaturizer:
    def __init__(
        self,
        pitch_name: Optional[str] = "pitch",
        voiced_mask_name: Optional[str] = "voiced_mask",
        voiced_prob_name: Optional[str] = None,
        sample_rate: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        pitch_fmin: int = librosa.note_to_hz('C2'),
        pitch_fmax: int = librosa.note_to_hz('C7'),
    ) -> None:
        self.pitch_name = pitch_name
        self.voiced_mask_name = voiced_mask_name
        self.voiced_prob_name = voiced_prob_name
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax

    def compute_pitch(self, manifest_entry: dict, audio_dir: Path) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes pitch and optional voiced mask for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            pitch: [T_spec] float tensor containing pitch for each audio frame.
            voiced_mask: [T_spec] bool tensor indicating whether each audio frame is voiced.
            voiced_prob: [T_spec] float array with [0, 1] probability that each audio frame is voiced.
        """
        audio_filepath, _ = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(path=audio_filepath, sr=self.sample_rate)

        pitch, voiced_mask, voiced_prob = librosa.pyin(
            audio,
            fmin=self.pitch_fmin,
            fmax=self.pitch_fmax,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            sr=self.sample_rate,
            fill_na=0.0,
        )
        pitch_tensor = torch.tensor(pitch, dtype=torch.float32)
        voiced_mask_tensor = torch.tensor(voiced_mask, dtype=torch.bool)
        voiced_prob_tensor = torch.tensor(voiced_prob, dtype=torch.float32)

        return pitch_tensor, voiced_mask_tensor, voiced_prob_tensor

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        pitch_tensor, voiced_mask_tensor, voiced_prob_tensor = self.compute_pitch(
            manifest_entry=manifest_entry, audio_dir=audio_dir
        )
        _save_pt_feature(
            feature_name=self.pitch_name,
            feature_tensor=pitch_tensor,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        _save_pt_feature(
            feature_name=self.voiced_mask_name,
            feature_tensor=voiced_mask_tensor,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        _save_pt_feature(
            feature_name=self.voiced_prob_name,
            feature_tensor=voiced_prob_tensor,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> Dict[str, Tensor]:
        feature_dict = {}
        _load_pt_feature(
            feature_dict=feature_dict,
            feature_name=self.pitch_name,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        _load_pt_feature(
            feature_dict=feature_dict,
            feature_name=self.voiced_mask_name,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        _load_pt_feature(
            feature_dict=feature_dict,
            feature_name=self.voiced_prob_name,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        return feature_dict
