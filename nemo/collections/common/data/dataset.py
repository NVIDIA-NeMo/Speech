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

import io
import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as pt_data
from torch.utils.data import Dataset, IterableDataset

__all__ = ['ConcatDataset', 'ConcatMapDataset', 'CodeSwitchedDataset']


class ConcatDataset(IterableDataset):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified 
    sampling technique.

    Args:
        datasets (list): A list of datasets to sample from.
        shuffle (bool): Whether to shuffle individual datasets. Only works with non-iterable datasets. 
            Defaults to True.
        sampling_technique (str): Sampling technique to choose which dataset to draw a sample from.
            Defaults to 'temperature'. Currently supports 'temperature', 'random' and 'round-robin'.
        sampling_temperature (int): Temperature value for sampling. Only used when sampling_technique = 'temperature'.
            Defaults to 5.
        sampling_scale: Gives you the ability to upsample / downsample the dataset. Defaults to 1.
        sampling_probabilities (list): Probability values for sampling. Only used when sampling_technique = 'random'.
        seed: Optional value to seed the numpy RNG.
        global_rank (int): Worker rank, used for partitioning map style datasets. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning map style datasets. Defaults to 1.
    """

    def __init__(
        self,
        datasets: List[Any],
        shuffle: bool = True,
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_scale: int = 1,
        sampling_probabilities: List[float] = None,
        seed: Optional[int] = None,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        supported_sampling_techniques = ['temperature', 'random', 'round-robin']
        self.datasets = datasets
        self.iterables = [None] * len(datasets)
        self.shuffle = shuffle
        self.global_rank = global_rank
        self.world_size = world_size
        self.sampling_kwargs = {}
        self.sampling_scale = sampling_scale

        if sampling_technique == 'temperature':
            self.index_generator = ConcatDataset.temperature_generator
            self.sampling_kwargs['temperature'] = sampling_temperature
            self.sampling_kwargs['seed'] = seed
        elif sampling_technique == 'random':
            self.index_generator = ConcatDataset.random_generator
            self.sampling_kwargs['p'] = sampling_probabilities
            self.sampling_kwargs['seed'] = seed
        elif sampling_technique == 'round-robin':
            self.index_generator = ConcatDataset.round_robin_generator
        else:
            raise ValueError(f"Currently we only support sampling techniques in {supported_sampling_techniques}.")
        self.length = 0

        if isinstance(datasets[0], IterableDataset):
            self.kind = 'iterable'
        else:
            self.kind = 'map'

        for idx, dataset in enumerate(datasets):
            isiterable = isinstance(dataset, IterableDataset)
            if (isiterable and not self.kind == 'iterable') or (not isiterable and self.kind == 'iterable'):
                raise ValueError("All datasets in ConcatDataset must be of the same kind (Iterable or Map).")

            if self.kind == 'map':
                self.length += len(dataset) // world_size
            else:
                self.length += len(dataset)

        if self.sampling_scale != 1:
            self.length = int(self.length * self.sampling_scale)
            logging.info(f'applying {sampling_scale} sampling scale, concat ds len: {self.length}')

    def get_iterable(self, dataset):
        if isinstance(dataset, IterableDataset):
            return dataset.__iter__()
        else:
            indices = np.arange(len(dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            return iter(indices)

    def __iter__(self):
        worker_info = pt_data.get_worker_info()
        if worker_info is None:
            max_elements = self.length
            wid = 0
            wnum = 1
        else:
            wid = worker_info.id
            wnum = worker_info.num_workers
            max_elements = len(range(wid, self.length, wnum))

        if self.kind == 'map':
            for idx in range(len(self.datasets)):
                start_idx = (len(self.datasets[idx]) // self.world_size) * self.global_rank
                end_idx = start_idx + (len(self.datasets[idx]) // self.world_size)
                if self.global_rank == self.world_size - 1:
                    end_idx = len(self.datasets[idx])
                indices = range(start_idx + wid, end_idx, wnum)
                self.datasets[idx] = pt_data.Subset(self.datasets[idx], indices)

        for idx, dataset in enumerate(self.datasets):
            iterable = self.get_iterable(dataset)
            self.iterables[idx] = iterable

        n = 0
        ind_gen = self.index_generator(self.datasets, **self.sampling_kwargs)
        while n < max_elements:
            n += 1
            try:
                ind = next(ind_gen)
            except StopIteration:
                return
            try:
                val = next(self.iterables[ind])
                if self.kind == 'map':
                    val = self.datasets[ind][val]
                yield val
            except StopIteration:
                self.iterables[ind] = self.get_iterable(self.datasets[ind])
                n -= 1

    def __len__(self):
        return self.length

    @staticmethod
    def temperature_generator(datasets, **kwargs):
        temp = kwargs.get('temperature')
        if not temp:
            raise ValueError("Temperature generator expects a 'temperature' keyword argument.")

        seed = kwargs.get('seed', None)
        np_rng = np.random.RandomState(seed)
        lengths = []
        num = len(datasets)
        for dataset in datasets:
            lengths.append(len(dataset))

        p = np.array(lengths) / np.sum(lengths)
        p = np.power(p, 1 / temp)
        p = p / np.sum(p)

        while True:
            ind = np_rng.choice(np.arange(num), p=p)
            yield ind

    @staticmethod
    def round_robin_generator(datasets, **kwargs):
        num = len(datasets)
        while True:
            for i in range(num):
                yield i

    @staticmethod
    def random_generator(datasets, **kwargs):
        p = kwargs.get('p')
        if not p:
            raise ValueError("Random generator expects a 'p' keyowrd argument for sampling probabilities.")

        seed = kwargs.get('seed', None)
        np_rng = np.random.RandomState(seed)
        num = len(datasets)
        if len(p) != num:
            raise ValueError("Length of probabilities list must be equal to the number of datasets.")

        while True:
            ind = np_rng.choice(np.arange(num), p=p)
            yield ind


class ConcatMapDataset(Dataset):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified 
    sampling technique.

    Args:
        datasets (list): A list of datasets to sample from.
        sampling_technique (str): Sampling technique to choose which dataset to draw a sample from.
            Defaults to 'temperature'. Currently supports 'temperature', 'random' and 'round-robin'.
        sampling_temperature (int): Temperature value for sampling. Only used when sampling_technique = 'temperature'.
            Defaults to 5.
        sampling_probabilities (list): Probability values for sampling. Only used when sampling_technique = 'random'.
        seed: Optional value to seed the numpy RNG.
    """

    def __init__(
        self,
        datasets: List[Any],
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.lengths = [len(x) for x in self.datasets]
        self.sampling_technique = sampling_technique
        self.sampling_temperature = sampling_temperature
        self.sampling_probabilities = sampling_probabilities
        self.np_rng = np.random.RandomState(seed)

        # Build a list of size `len(self)`. Each tuple contains (dataset_id, dataset_index)
        self.indices: List[Tuple[int, int]] = []
        # Current position as we consume indices from each data set
        dataset_positions = [0] * len(self.datasets)
        # Random permutation of each dataset. Will be regenerated when exhausted.
        shuffled_indices = [self.np_rng.permutation(len(x)) for x in self.datasets]
        # Build the list of randomly-chosen datasets spanning the entire length, adhering to sampling technique
        if self.sampling_technique == "round-robin":
            # To exhaust longest dataset, need to draw `num_datasets * max_dataset_len` samples
            total_length = max(self.lengths) * len(self.lengths)
            # For round robin, iterate through each dataset
            dataset_ids = np.arange(total_length) % len(self.datasets)
            for dataset_id in dataset_ids:
                position = dataset_positions[dataset_id]
                index = shuffled_indices[dataset_id][position]
                self.indices.append((dataset_id, index))
                dataset_positions[dataset_id] += 1
                if dataset_positions[dataset_id] == len(shuffled_indices[dataset_id]):
                    dataset_positions[dataset_id] = 0
                    shuffled_indices[dataset_id] = self.np_rng.permutation(len(self.datasets[dataset_id]))
        else:
            # Resolve probabilities of drawing from each data set
            if self.sampling_technique == "random":
                if sampling_probabilities is None or len(sampling_probabilities) != len(self.datasets):
                    raise ValueError(
                        f"Need {len(self.datasets)} probabilities; got "
                        f"{len(sampling_probabilities) if sampling_probabilities is not None else 'None'}"
                    )
                p = np.array(self.sampling_probabilities)
            elif self.sampling_technique == "temperature":
                p = np.array([len(x) for x in self.datasets])
                p = np.power(p, 1 / self.sampling_temperature)
            else:
                raise ValueError(f"Couldn't interpret sampling technique: {sampling_technique}")
            # Normalize probabilities
            p = p / np.sum(p)
            # Will randomly choose from datasets
            choices = np.arange(len(self.datasets))
            # Keep going until largest dataset is exhausted.
            exhausted_datasets = set()
            while len(exhausted_datasets) < len(self.datasets):
                # Randomly choose a dataset for each position in accordance with p
                dataset_id = self.np_rng.choice(a=choices, p=p)
                dataset = self.datasets[dataset_id]
                # Pick next index from dataset
                position = dataset_positions[dataset_id]
                index = shuffled_indices[dataset_id][position]
                self.indices.append((dataset_id, index))
                # Maybe reset this dataset's permutation
                dataset_positions[dataset_id] += 1
                if dataset_positions[dataset_id] >= len(dataset):
                    shuffled_indices[dataset_id] = self.np_rng.permutation(len(dataset))
                    dataset_positions[dataset_id] = 0
                    exhausted_datasets.add(dataset_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_id, dataset_index = self.indices[idx]
        return self.datasets[dataset_id][dataset_index]


class CodeSwitchedDataset(IterableDataset):
    """
    A dataset that accepts as argument multiple sub-datasets (usually from different languages, but that's not required) and then
    samples from them in order to create synthetic code-switched samples of up to N different sub-datasets

    Args:
        datasets (list): A list of datasets
        lang_probs (list): A list of probabilities (which must sum to 1) corresponding to the sampling probability for each dataset
        shuffle (bool): Whether to shuffle individual datasets. Only works with non-iterable datasets. 
            Defaults to True.
        min_duration (int): the minimum duration (secs) of each synthetic code-switched sample. Will draw randomly until this is hit.
            Defaults to 4
        max_duration (int): the maximum duration (secs) of each synthetic code-switched sample.
            Defaults to 20
        min_monolingual (float): this percentage of the dataset will be original monolingual samples
            Defaults to 0.3 - means 30%
        db_norm (float): will normalise the composite CS sample to this DB level
            Defaults to -25.0
        pause_start (int): inserts silence equal to this value (msecs) at the start of each CS sample
            Defaults to 0
        pause_join (int): inserts silence equal to this value (msecs) between all language changes in the CS sample
            Defaults to 0
        pause_end (int): terminates all CS samples with silence equal to this value (msecs)
            Defaults to 0
        sampling_scales (list or float): gives you the ability to upsample/downsample each individual dataset
        seed: Optional value to seed the numpy RNG.
        global_rank (int): Worker rank, used for partitioning map style datasets. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning map style datasets. Defaults to 1.
        pure_random (bool): If true, then always draw random sample from lang_probs. If false, you only draw from those datasets
            which you haven't sampled from yet for the composite sample
        force_monochannel (bool): If true, then all output audio will be mono-channel
        infinity_mode (bool): If true, then the dataset iterable will generate an infinite amount of samples
        sample_rate (int): the sample rate of all audio being sent to this Dataset
        augmentor (AudioAugmentor): The any perturbations you wish to have applied on the CS samples
    """

    def __init__(
        self,
        datasets: List[Any],
        lang_probs: Optional[List[float]] = None,
        shuffle: bool = True,
        min_duration: int = 4,
        max_duration: int = 20,
        min_monolingual: float = 0.3,
        db_norm: float = -25.0,
        pause_start: int = 0,
        pause_join: int = 0,
        pause_end: int = 0,
        sampling_scales: Optional[Union[float, List[float]]] = None,
        seed: Optional[int] = None,
        global_rank: int = 0,
        world_size: int = 1,
        pure_random: bool = False,
        force_monochannel: bool = True,
        infinity_mode: bool = False,
        sample_rate: int = 16000,
        augmentor: Optional['AudioAugmentor'] = None,
    ):
        super().__init__()

        if len(datasets) == 0:
            raise ValueError("CodeSwitchedDataset must receive a non-zero length datasets dict object")

        self.datasets = datasets
        self.langs = list(range(len(datasets)))
        self.langs_set = set(self.langs)
        self.lang_iterables = {k: None for k in self.langs}
        self.lang_kind = {k: None for k in self.langs}
        self.shuffle = shuffle
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_monolingual = min_monolingual
        self.db_norm = db_norm
        self.pause_start = pause_start
        self.pause_join = pause_join
        self.pause_end = pause_end
        self.pure_random = pure_random
        self.force_monochannel = force_monochannel
        self.infinity_mode = infinity_mode
        self.global_rank = global_rank
        self.world_size = world_size
        self.augmentor = augmentor
        self.sample_rate = sample_rate
        self.length = 0
        if lang_probs is None:
            self.prob_dict = {l: 1.0 / len(self.langs) for l in self.langs}
        else:
            assert len(self.langs) == len(
                lang_probs
            ), "Size mismatch between languages and respective probs in CodeSwitchedDataset"
            self.prob_dict = {l: lang_probs[l] for l in self.langs}
        self.lang_probs = np.array(list(self.prob_dict.values()))
        if sampling_scales is not None and not isinstance(sampling_scales, list):
            self.sampling_scales = {k: sampling_scales for k in self.langs}
        elif (
            sampling_scales is not None
            and isinstance(sampling_scales, list)
            and len(sampling_scales) == len(self.langs)
        ):
            self.sampling_scales = {k: v for k, v in zip(self.langs, sampling_scales)}
        else:
            self.sampling_scales = {k: 1 for k in self.langs}

        for lang, dataset in enumerate(self.datasets):
            isiterable = isinstance(dataset, IterableDataset)

            if isiterable:
                self.lang_kind[lang] = 'iterable'
                self.length += int(len(dataset) * self.sampling_scales[lang])
            else:
                self.lang_kind[lang] = 'map'
                self.length += int((len(dataset) // world_size) * self.sampling_scales[lang])

        if seed is not None:
            np.random.seed(seed)

        # set this to ensure compatibility with models searching for the collate_fn
        # since this class stores datasets as a dict, not list
        # self.collate_fn = self.datasets[self.langs[0]].collate_fn
        if hasattr(self.datasets[self.langs[0]], 'collate_fn'):
            self.collate_fn = self.datasets[self.langs[0]].collate_fn
        elif (
            hasattr(self.datasets[self.langs[0]], 'datasets')
            and isinstance(self.datasets[self.langs[0]].datasets, list)
            and len(self.datasets[self.langs[0]].datasets) > 0
            and hasattr(self.datasets[self.langs[0]].datasets[0], 'collate_fn')
        ):
            # support datasets that are lists of entries
            self.collate_fn = self.datasets[self.langs[0]].datasets[0].collate_fn
        elif (
            hasattr(self.datasets[self.langs[0]], 'datasets')
            and isinstance(self.datasets[self.langs[0]].datasets, list)
            and len(self.datasets[self.langs[0]].datasets) > 0
            and hasattr(self.datasets[self.langs[0]].datasets[0], 'datasets')
            and isinstance(self.datasets[self.langs[0]].datasets[0].datasets, list)
            and len(self.datasets[self.langs[0]].datasets[0].datasets) > 0
            and hasattr(self.datasets[self.langs[0]].datasets[0].datasets[0], 'collate_fn')
        ):
            # support datasets that are lists of lists
            self.collate_fn = self.datasets[self.langs[0]].datasets[0].datasets[0].collate_fn
        else:
            raise RuntimeError("CodeSwitchedDataset could not locate a valid dataset collate_fn to bind to")

    # this method returns an iterator object for a given language ID
    # it correctly handles whether the underlying dataset is IterableDataset or mappable
    def get_iterable_by_lang(self, lang):
        dataset = self.datasets[lang]

        if isinstance(dataset, IterableDataset):
            return dataset.__iter__()
        else:
            indices = np.arange(len(dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            return iter(indices)

    # this method is the main function which builds and returns a composite, synthetic code-switched
    # utterance on the fly. It automatically works with all of the class-based variables stored to create
    # the synthetic utterance
    def build_single_CS_sample(self):
        # get_sample_from_language returns a LongTensor for the transcripts so we create a LongTensor to hold
        # all returned transcripts
        comp_text = torch.LongTensor([])
        created_sample_duration_sec = 0
        created_sample_langs = []
        created_sample_audios = []

        # if min_monolingual fires, it means we will just return a single, original monolingual utterance
        # from one of our languages based on that language's probability
        pure_mono = np.random.rand() <= self.min_monolingual

        # we continue to add to the composite utterance until we hit the min_duration
        while created_sample_duration_sec < self.min_duration:
            # we sample from only those languages which haven't already been sampled for this particular
            # synthetic utterance, unless pure_random=True, in which case, you just sample with replacement
            # every time
            if (self.pure_random and not pure_mono) or (
                len(set(created_sample_langs)) == 0 or len(set(created_sample_langs)) == len(self.langs)
            ):
                lang_id = np.random.choice(self.langs, p=self.lang_probs)
            # elif pure_mono:
            #    use this approach if you want synthetic utterances which are all monolingual
            #    lang_id = created_sample_langs[0]
            else:
                # this code is for when we need to sample from only those languages which haven't been sampled
                # yet for this utterance
                p = np.array(list(map(self.prob_dict.get, list(self.langs_set - set(created_sample_langs)))))
                p = p / p.sum()
                lang_id = np.random.choice(list(self.langs_set - set(created_sample_langs)), p=p)

            audio, audio_len, labels, labels_len, *_ = self.get_sample_from_language(lang_id)

            # in case you get an audio which is all silence we keep sampling
            if audio.count_nonzero().item() == 0:
                continue

            sample_duration = len(audio) / self.sample_rate
            if (created_sample_duration_sec + sample_duration) > self.max_duration:
                continue

            if comp_text.device != labels.device:
                comp_text = comp_text.to(labels.device)

            if audio.ndim > 1 and self.force_monochannel:
                audio = audio.mean(dim=-1)

            created_sample_duration_sec += sample_duration
            created_sample_langs.append(lang_id)
            # need to use numpy instead of torch here because we need numpy's trim_zeros function
            created_sample_audios.append(audio.cpu().numpy())
            comp_text = torch.cat([comp_text, labels], dim=0)

            # we want a real, non-synth pure_mono sample so we break soon as we have one
            if pure_mono:
                break

        # check that all samples have the same number of channels
        sample_channels = list(set([s.ndim for s in created_sample_audios]))
        if len(sample_channels) > 1:
            raise RuntimeError(
                "Mixture of audios with different number of channels in CodeSwitchedDataset. All sources must be same number of channels."
            )

        multichannel = sample_channels[0] > 1

        # we start with pause_start amount of silence (zero array) which needs the correct shape for multi/mono channel
        if multichannel:
            comp_audio = np.zeros(
                shape=(int(self.pause_start * self.sample_rate / 1000.0), created_sample_audios[0].shape[-1]),
                dtype=created_sample_audios[0].dtype,
            )
        else:
            comp_audio = np.zeros(
                shape=(int(self.pause_start * self.sample_rate / 1000.0),), dtype=created_sample_audios[0].dtype
            )

        # iterate over all mono-lingual samples to build the final composite
        for idx, wav in enumerate(created_sample_audios):
            if not multichannel:
                # this function only works if mono-channel
                wav = np.trim_zeros(wav)

            # normalise to provided DB level
            wav_norm = wav * (10.0 ** (self.db_norm / 20.0) / np.maximum(0.01, (wav ** 2).mean(axis=0) ** 0.5))

            # this part appends the normed waveform to the existing waveform, and inserts pause_join amount of silence
            # if necessary, otherwise just a straight append
            if idx < len(created_sample_audios) - 1:
                if multichannel:
                    wav_norm = np.append(
                        wav_norm,
                        np.zeros(
                            shape=(
                                int(self.pause_join * self.sample_rate / 1000.0),
                                created_sample_audios[0].shape[-1],
                            ),
                            dtype=comp_audio.dtype,
                        ),
                        axis=0,
                    )
                else:
                    wav_norm = np.append(
                        wav_norm,
                        np.zeros(shape=(int(self.pause_join * self.sample_rate / 1000.0),), dtype=comp_audio.dtype),
                        axis=0,
                    )

            # this is the penultimate composite wavform, just need to add pause_end silence
            comp_audio = np.append(comp_audio, wav_norm, axis=0)

        # here we add the pause_end amount of silence, in correct channel shape
        if multichannel:
            comp_audio = np.append(
                comp_audio,
                np.zeros(
                    shape=(int(self.pause_end * self.sample_rate / 1000.0), created_sample_audios[0].shape[-1]),
                    dtype=comp_audio.dtype,
                ),
                axis=0,
            )
        else:
            comp_audio = np.append(
                comp_audio,
                np.zeros(shape=(int(self.pause_end * self.sample_rate / 1000.0),), dtype=comp_audio.dtype),
                axis=0,
            )

        # we only want augmentation to happen on the final, synthetic utterance, and not on any of the individual
        # languages, which is why we set augmentor=None when building the individual language datasets in audio_to_text_dataset.get_code_switched_dataset
        # here we now apply augmentation to the final, synthetic utterance only
        # all of this logic here happens in-memory, nothing is written to disk
        if self.augmentor is not None:
            # import here to avoid circular import error
            # import here because otherwise CI test-nlp-imports fails since soundfile is only in requirements_asr and not in requirements_common
            import soundfile as sf

            from nemo.collections.asr.parts.preprocessing import AudioSegment

            mb = io.BytesIO()
            sf.write(mb, comp_audio, self.sample_rate, format='WAV')
            mb.seek(0)
            comp_audio_as = AudioSegment.from_file(mb, target_sr=self.sample_rate)
            self.augmentor.perturb(comp_audio_as)
            comp_audio = comp_audio_as.samples

        return (
            torch.tensor(comp_audio, dtype=audio.dtype, device=audio.device),
            torch.tensor(len(comp_audio), device=audio_len.device).long(),
            comp_text,
            torch.tensor(len(comp_text), device=labels_len.device).long(),
        )

    # this is a helper method which prepares all of the iterator objects for all languages
    # based on whether that language's underlying dataset is a map or an IterableDataset
    def prep_underlying_datasets(self):
        worker_info = pt_data.get_worker_info()
        if worker_info is None:
            max_elements = self.length
            wid = 0
            wnum = 1
        else:
            wid = worker_info.id
            wnum = worker_info.num_workers
            max_elements = len(range(wid, self.length, wnum))

        for lang in self.langs:
            if self.lang_kind[lang] == 'map':
                start_idx = (len(self.datasets[lang]) // self.world_size) * self.global_rank
                end_idx = start_idx + (len(self.datasets[lang]) // self.world_size)
                if self.global_rank == self.world_size - 1:
                    end_idx = len(self.datasets[lang])
                indices = range(start_idx + wid, end_idx, wnum)
                self.datasets[lang] = pt_data.Subset(self.datasets[lang], indices)

            self.lang_iterables[lang] = self.get_iterable_by_lang(lang)

        return max_elements

    # returns a sample (audio and transcript) from any underlying language stored by the class on instantiation
    # the sample returned is a tensor for the audio and a tensor of ints for the transcript
    # this method automatically handles StopIteration errors for the underyling language and rebuilds
    # the iterator if necessary
    def get_sample_from_language(self, lang):
        while True:
            try:
                val = next(self.lang_iterables[lang])
                if self.lang_kind[lang] == 'map':
                    val = self.datasets[lang][val]
                return val
            except StopIteration:
                self.lang_iterables[lang] = self.get_iterable_by_lang(lang)

    def __iter__(self):
        # we create primed iterators for all languages and return the grand total of samples for each
        # underlying language as a sum
        max_elements = self.prep_underlying_datasets()

        if self.infinity_mode:
            while True:
                yield self.build_single_CS_sample()
        else:
            n = 0
            while n < max_elements:
                yield self.build_single_CS_sample()
                n += 1

    def __len__(self):
        return self.length
