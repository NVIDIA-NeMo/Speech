from itertools import islice
from pathlib import Path

import hydra
import lhotse
import numpy as np
import soundfile as sf
from lhotse import CutSet, MonoCut, Recording
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nemo.collections.audio.data.audio_to_audio_lhotse import LhotseAudioToTargetDataset
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

"""
The purpose of this script is to save online-aumented data as provided by NeMo Lhotse dataloader.
The script piggybacks on a train_ds section of an existing model configuration file.

Intended use cases are: 1) preparing a validation set, 2) debugging.

Usage example:
$ python examples/audio/save_augmented.py \
    +input_cuts=some_path/cuts.jsonl \
    +output_cuts=some_other_path/cuts.gsm_and_clipping_augmented.jsonl \
    +keep_directory_structure=true \
    model.sample_rate=48000 \
    ++model.train_ds.rir_enabled=true \
    ++model.train_ds.rir_path=path/to/rir_manifest.jsonl

Assumptions:
- input data are described as a Lhotse CutSet in a JSONL file
   - consists of simple MonoCuts with Recording paths relative to the Cuts manifest
- the parent directory of the output cuts must exist

Requires additional config parameters `input_cuts` and `output_cuts`.
Produces:
- %output_cuts_parent_dir%/audio/
- %output_cuts_parent_dir%/%output_cuts_filename%.jsonl
where the audio folder contains the augmented and clean signals, respectively, with `.dirty.flac` and `.clean.flac` suffixes.

If `keep_directory_structure` provided and is True, the script will preserve the directory structure of the input cuts.

Text is preserved from the input cuts if possible. 

Optional config parameter `num_samples` can be used to limit the number of samples to save (but not more than input dataloader size). 
If not specified, the dataloader is used until exhausted.
"""


def check_input_cuts(input_cuts_path: Path) -> None:
    assert input_cuts_path.exists(), "input_cuts must exist"
    assert input_cuts_path.suffix == '.jsonl', "input_cuts must be a .jsonl file"
    assert input_cuts_path.parent.exists(), "input_cuts parent directory must exist"
    cuts = lhotse.CutSet.from_file(input_cuts_path)
    for i, cut in enumerate(cuts):
        assert isinstance(cut, MonoCut), f"{i}th cut is a {type(cut)}, not a MonoCut"
        assert len(cut.recording.sources) == 1, f"{i}th cut has {len(cut.recording.sources)} sources"
        assert cut.recording.sources[0].source is not None, "{i}th cut has no audio source specified"

        recording_path = Path(cut.recording.sources[0].source)
        assert not recording_path.is_absolute(), f"{i}th cut's recording source is an absolute path: {recording_path}"

        recording_path_full = input_cuts_path.parent / recording_path
        assert recording_path_full.exists(), f"{i}th cut's recording source file does not exist: {recording_path_full}"


@hydra.main(config_path="conf", config_name="flow_matching_generative_finetuning.yaml")
def main(cfg: DictConfig):
    assert (
        cfg.get("input_cuts", None) is not None
    ), "input_cuts is required, please override (for example, +input_cuts=some_path/cuts.jsonl)"
    assert (
        cfg.get("output_cuts", None) is not None
    ), "output_cuts is required, please override (for example, +output_cuts=some_path/cuts.augmented.jsonl)"
    num_samples = cfg.get("num_samples", None)
    sample_rate = cfg.model.sample_rate
    keep_directory_structure = cfg.get("keep_directory_structure", False)

    input_cuts_path = Path(cfg.input_cuts)
    output_cuts_path = Path(cfg.output_cuts)
    check_input_cuts(input_cuts_path)  # throws an exception if they aren't ok

    assert output_cuts_path.parent.exists(), "output_cuts parent directory must exist"

    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "model.train_ds.cuts_path", str(input_cuts_path), force_add=True)
    OmegaConf.update(cfg, "model.train_ds.shuffle", False)  # ensure deterministic behavior
    OmegaConf.update(cfg, "model.train_ds.batch_size", 1)
    OmegaConf.update(cfg, "model.train_ds.shard_seed", 0, force_add=True)  # ensure deterministic behavior
    if cfg.model.train_ds.get("sample_rate", None) != sample_rate:
        OmegaConf.update(cfg, "model.train_ds.sample_rate", sample_rate, force_add=True)

    dataloader = get_lhotse_dataloader_from_config(
        OmegaConf.create(cfg.model.train_ds), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
    )

    cuts = lhotse.CutSet.from_file(input_cuts_path)
    if num_samples is None:
        num_samples = len(cuts)

    with CutSet.open_writer(output_cuts_path) as writer:
        for i, (sample, original_cut) in enumerate(
            tqdm(zip(islice(dataloader, num_samples), cuts), total=num_samples)
        ):
            # batch_size is 1, so we can access the first element
            dirty = sample['input_signal'][0].numpy()
            clean = sample['target_signal'][0].numpy()

            # if necessary, apply negative gain to avoid clipping
            if (coeff := max(np.max(np.abs(dirty)), np.max(np.abs(clean)))) > 1.0:
                dirty = dirty / coeff
                clean = clean / coeff

            if keep_directory_structure:
                # definitely a relative path because we checked for that earlier
                input_relative_path = Path(original_cut.recording.sources[0].source)

                dirty_path = output_cuts_path.parent / input_relative_path.with_suffix('.dirty.flac')
                clean_path = output_cuts_path.parent / input_relative_path.with_suffix('.clean.flac')

                # we know that `audio_dir` exists, but we need to create the parent directories
                dirty_path.parent.mkdir(exist_ok=True, parents=True)
                clean_path.parent.mkdir(exist_ok=True, parents=True)
            else:
                (output_cuts_path.parent / 'audio').mkdir(exist_ok=True, parents=True)
                dirty_path = output_cuts_path.parent / 'audio' / f"{i:06}.dirty.flac"
                clean_path = output_cuts_path.parent / 'audio' / f"{i:06}.clean.flac"

            sf.write(dirty_path, dirty, sample_rate, format='FLAC', subtype='PCM_24')
            sf.write(clean_path, clean, sample_rate, format='FLAC', subtype='PCM_24')

            dirty_recording = Recording.from_file(dirty_path)
            dirty_recording.sources[0].source = str(dirty_path.relative_to(output_cuts_path.parent))
            clean_recording = Recording.from_file(clean_path)
            clean_recording.sources[0].source = str(clean_path.relative_to(output_cuts_path.parent))

            cut = MonoCut(
                id=dirty_recording.id, start=0, channel=0, duration=dirty_recording.duration, recording=dirty_recording
            )
            cut.target_recording = clean_recording

            for optional_field_name in (
                'text',
                'original_text',
                'language',
            ):
                if (
                    hasattr(original_cut, optional_field_name)
                    and getattr(original_cut, optional_field_name) is not None
                ):
                    setattr(cut, optional_field_name, getattr(original_cut, optional_field_name))

            writer.write(cut)


if __name__ == "__main__":
    main()
