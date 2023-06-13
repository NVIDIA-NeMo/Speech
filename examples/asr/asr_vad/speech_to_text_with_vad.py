# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
This file provides the ASR+VAD inference pipeline, with the option to perform only ASR or VAD alone.

There are two types of input, the first one is a manifest passed to `manifest_filepath`, 
and the other one is to pass a directory containing audios to `audio_dir` and specify `audio_type`.

The input manifest must be a manifest json file, where each line is a Python dictionary. The fields ["audio_filepath", "offset", "duration",  "text"] are required. An example of a manifest file is:
```
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000,  "text": "a b c d e"}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000,  "text": "f g h i j"}
```

To run the code with ASR+VAD default settings:

```bash
python speech_to_text_with_vad.py \
    manifest_filepath=/PATH/TO/MANIFEST.json \
    vad_model=vad_multilingual_frame_marblenet\
    asr_model=stt_en_conformer_ctc_large \
    vad_config=../conf/vad/frame_vad_inference_postprocess.yaml
```

To use only ASR and disable VAD, set `vad_model=None` and `use_rttm=False`.

To use only VAD, set `asr_model=None` and specify both `vad_model` and `vad_config`.

To enable profiling, set `profiling=True`, but this will significantly slow down the program.

To use or disable feature masking/droping based on RTTM files, set `use_rttm` to `True` or `False`. 
There are two ways to use RTTM files, either by masking the features (`rttm_mode=mask`) or by dropping the features (`rttm_mode=drop`).
For audios that have long non-speech audios between speech segments, dropping frames is recommended.

To normalize feature before masking, set `normalize=pre_norm`, 
and set `normalize=post_norm` for masking before normalization.

To use a specific value for feature masking, set `feat_mask_val` to the desired value. 
Default is `feat_mask_val=None`, where -16.635 will be used for `post_norm` and 0 will be used for `pre_norm`.

See more options in the `InferenceConfig` class.
"""


import contextlib
import json
import os
import time
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from nemo.collections.asr.data import feature_to_text_dataset
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig, word_error_rate
from nemo.collections.asr.models import ASRModel, EncDecClassificationModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    init_frame_vad_model,
    init_vad_model,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:

    @contextlib.contextmanager
    def autocast(enabled=None):
        yield


@dataclass
class InferenceConfig:
    # Required configs
    asr_model: Optional[str] = None  # Path to a .nemo file or a pretrained NeMo model on NGC
    vad_model: Optional[str] = None  # Path to a .nemo file or a pretrained NeMo model on NGC
    vad_config: Optional[str] = None  # Path to a yaml file containing VAD post-processing configs
    manifest_filepath: Optional[str] = None  # Path to dataset's JSON manifest
    audio_dir: Optional[str] = None  # Path to a directory containing audio files, use this if no manifest is provided

    use_rttm: bool = True  # whether to use RTTM
    rttm_mode: str = "mask"  # how to use RTTM files, choices=[`mask`, `drop`]
    feat_mask_val: Optional[float] = None  # value used to mask features based on RTTM, set None to use defaults
    normalize: Optional[
        str
    ] = "post_norm"  # whether and where to normalize audio feature, choices=[None, `pre_norm`, `post_norm`]
    normalize_type: str = "per_feature"  # how to determine mean and std used for normalization
    normalize_audio_db: Optional[float] = None  # set to normalize RMS DB of audio before extracting audio features

    profiling: bool = False  # whether to enable pytorch profiling

    # General configs
    batch_size: int = 1  # batch size for ASR. Feature extraction and VAD only support single sample per batch.
    num_workers: int = 8
    sample_rate: int = 16000
    frame_unit_time_secs: float = 0.01  # unit time per frame in seconds, equal to `window_stride` in ASR configs, typically 10ms.
    audio_type: str = "wav"

    # Output settings, no need to change
    output_dir: Optional[str] = None  # will be automatically set by the program
    output_filename: Optional[str] = None  # will be automatically set by the program
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, other than the standard one.

    # Set to True to output language ID information
    compute_langs: bool = False

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)

    # VAD model type
    vad_type: str = "frame"  # which type of VAD to use, choices=[`frame`, `segment`]


@hydra_runner(config_name="InferenceConfig", schema=InferenceConfig)
def main(cfg):

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.output_dir is None:
        cfg.output_dir = "./outputs"
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # setup profiling, note that profiling will significantly increast the total runtime
    if cfg.profiling:
        logging.info("Profiling enabled")
        profile_fn = profile
        record_fn = record_function
    else:
        logging.info("Profiling disabled")

        @contextlib.contextmanager
        def profile_fn(*args, **kwargs):
            yield

        @contextlib.contextmanager
        def record_fn(*args, **kwargs):
            yield

    input_manifest_file = prepare_inference_manifest(cfg)

    if cfg.manifest_filepath is None:
        cfg.manifest_filepath = str(input_manifest_file)

    with profile_fn(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:

        input_manifest_file = extract_audio_features(input_manifest_file, cfg, record_fn)

        if cfg.vad_model is not None:
            logging.info(f"Running VAD with model: {cfg.vad_model}")
            input_manifest_file = run_vad_inference(input_manifest_file, cfg, record_fn)

        if cfg.asr_model is not None:
            logging.info(f"Running ASR with model: {cfg.asr_model}")
            run_asr_inference(input_manifest_file, cfg, record_fn)

    if cfg.profiling:
        print("--------------------------------------------------------------------\n")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print("--------------------------------------------------------------------\n")
    logging.info("Done.")


def prepare_inference_manifest(cfg: DictConfig) -> str:

    if cfg.audio_dir is not None and cfg.manifest_filepath is None:
        manifest_data = []
        for audio_file in Path(cfg.audio_dir).glob(f"**/*.{cfg.audio_type}"):
            item = {"audio_filepath": str(audio_file.absolute()), "duration": 1000000, "offset": 0}
            manifest_data.append(item)
        parent_dir = Path(cfg.audio_dir)
    else:
        manifest_data = read_manifest(cfg.manifest_filepath)
        parent_dir = Path(cfg.manifest_filepath).parent

    new_manifest_data = []

    for item in manifest_data:
        audio_file = Path(item["audio_filepath"])
        if len(str(audio_file)) < 255 and not audio_file.is_file() and not audio_file.is_absolute():
            new_audio_file = parent_dir / audio_file
            if new_audio_file.is_file():
                item["audio_filepath"] = str(new_audio_file.absolute())
            else:
                item["audio_filepath"] = os.path.expanduser(str(audio_file))
        else:
            item["audio_filepath"] = os.path.expanduser(str(audio_file))
        item["label"] = "infer"
        item["text"] = "-"
        new_manifest_data.append(item)

    new_manifest_filepath = str(Path(cfg.output_dir) / Path("temp_manifest_input.json"))
    write_manifest(new_manifest_filepath, new_manifest_data)
    return new_manifest_filepath


def extract_audio_features(manifest_filepath: str, cfg: DictConfig, record_fn: Callable) -> str:
    file_list = []
    manifest_data = []
    out_dir = Path(cfg.output_dir) / Path("features")
    new_manifest_filepath = str(Path(cfg.output_dir) / Path("temp_manifest_input_feature.json"))

    if Path(new_manifest_filepath).is_file():
        logging.info("Features already exist in output_dir, skipping feature extraction.")
        return new_manifest_filepath

    has_feat = False
    with open(manifest_filepath, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            item = json.loads(line.strip())
            manifest_data.append(item)
            file_list.append(Path(item['audio_filepath']).stem)
            if "feature_file" in item:
                has_feat = True
    if has_feat:
        logging.info("Features already exist in manifest, skipping feature extraction.")
        return manifest_filepath

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    if cfg.vad_model:
        vad_model = init_frame_vad_model(cfg.vad_model)
    else:
        vad_model = EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'vad_stream': False,
            'sample_rate': cfg.sample_rate,
            'manifest_filepath': manifest_filepath,
            'labels': ['infer',],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'normalize_audio_db': cfg.normalize_audio_db,
        }
    )

    logging.info(f"Extracting features on {len(file_list)} audio files...")
    with record_fn("feat_extract_loop"):
        for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
            test_batch = [x.to(vad_model.device) for x in test_batch]
            with autocast():
                with record_fn("feat_extract_infer"):
                    processed_signal, processed_signal_length = vad_model.preprocessor(
                        input_signal=test_batch[0], length=test_batch[1],
                    )
                with record_fn("feat_extract_other"):
                    processed_signal = processed_signal.squeeze(0)[:, :processed_signal_length]
                    processed_signal = processed_signal.cpu()
                    outpath = os.path.join(out_dir, file_list[i] + ".pt")
                    outpath = str(Path(outpath).absolute())
                    torch.save(processed_signal, outpath)
                    manifest_data[i]["feature_file"] = outpath
                    del test_batch

    logging.info(f"Features saved at: {out_dir}")
    write_manifest(new_manifest_filepath, manifest_data)
    return new_manifest_filepath


def run_vad_inference(manifest_filepath: str, cfg: DictConfig, record_fn: Callable) -> str:
    logging.info("Start VAD inference pipeline...")
    if cfg.vad_type == "segment":
        vad_model = init_vad_model(cfg.vad_model)
    elif cfg.vad_type == "frame":
        vad_model = init_frame_vad_model(cfg.vad_model)
    else:
        raise ValueError(f"Unknown VAD type: {cfg.vad_type}, supported types: ['segment', 'frame']")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()

    vad_yaml = Path(cfg.vad_config)
    if not vad_yaml.is_file():
        raise ValueError(f"VAD config file not found: {cfg.vad_config}")

    with vad_yaml.open("r") as fp:
        vad_cfg = yaml.safe_load(fp)
    vad_cfg = DictConfig(vad_cfg)

    test_data_config = {
        'vad_stream': True,
        'manifest_filepath': manifest_filepath,
        'labels': ['infer',],
        'num_workers': cfg.num_workers,
        'shuffle': False,
        'window_length_in_sec': vad_cfg.vad.parameters.window_length_in_sec,
        'shift_length_in_sec': vad_cfg.vad.parameters.shift_length_in_sec,
    }
    vad_model.setup_test_data(test_data_config=test_data_config, use_feat=True)

    pred_dir = Path(cfg.output_dir) / Path("vad_frame_pred")
    if pred_dir.is_dir():
        logging.info(f"VAD frame-level prediction already exists: {pred_dir}, skipped")
    else:
        logging.info("Generating VAD frame-level prediction")
        pred_dir.mkdir(parents=True)
        t0 = time.time()
        pred_dir = generate_vad_frame_pred(
            vad_model=vad_model,
            window_length_in_sec=vad_cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=vad_cfg.vad.parameters.shift_length_in_sec,
            manifest_vad_input=manifest_filepath,
            out_dir=str(pred_dir),
            use_feat=True,
            record_fn=record_fn,
        )
        t1 = time.time()
        logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
        logging.info(
            f"Finished generating VAD frame level prediction with window_length_in_sec={vad_cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={vad_cfg.vad.parameters.shift_length_in_sec}"
        )

    frame_length_in_sec = vad_cfg.vad.parameters.shift_length_in_sec
    # overlap smoothing filter
    if vad_cfg.vad.parameters.smoothing:
        # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
        # smoothing_method would be either in majority vote (median) or average (mean)
        logging.info("Generating predictions with overlapping input segments")
        t0 = time.time()
        smoothing_pred_dir = generate_overlap_vad_seq(
            frame_pred_dir=pred_dir,
            smoothing_method=vad_cfg.vad.parameters.smoothing,
            overlap=vad_cfg.vad.parameters.overlap,
            window_length_in_sec=vad_cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=vad_cfg.vad.parameters.shift_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=vad_cfg.smoothing_out_dir,
        )
        logging.info(
            f"Finish generating predictions with overlapping input segments with smoothing_method={vad_cfg.vad.parameters.smoothing} and overlap={vad_cfg.vad.parameters.overlap}"
        )
        t1 = time.time()
        logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
        pred_dir = smoothing_pred_dir
        frame_length_in_sec = 0.01

    # Turn frame-wise prediction into speech intervals
    logging.info(f"Generating segment tables with postprocessing params: {vad_cfg.vad.parameters.postprocessing}")
    segment_dir_name = "vad_rttm"
    for key, val in vad_cfg.vad.parameters.postprocessing.items():
        segment_dir_name = segment_dir_name + "-" + str(key) + str(val)

    segment_dir = Path(cfg.output_dir) / Path(segment_dir_name)
    if segment_dir.is_dir():
        logging.info(f"VAD speech segments already exists: {segment_dir}, skipped")
    else:
        segment_dir.mkdir(parents=True)
        t0 = time.time()
        segment_dir = generate_vad_segment_table(
            vad_pred_dir=pred_dir,
            postprocessing_params=vad_cfg.vad.parameters.postprocessing,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=segment_dir,
            use_rttm=True,
        )
        t1 = time.time()
        logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
        logging.info("Finished generating RTTM files from VAD predictions.")

    rttm_map = {}
    for filepath in Path(segment_dir).glob("*.rttm"):
        rttm_map[filepath.stem] = str(filepath.absolute())

    manifest_data = read_manifest(manifest_filepath)
    for i in range(len(manifest_data)):
        key = Path(manifest_data[i]["audio_filepath"]).stem
        manifest_data[i]["rttm_file"] = rttm_map[key]

    new_manifest_filepath = str(Path(cfg.output_dir) / Path(f"temp_manifest_{segment_dir_name}.json"))
    write_manifest(new_manifest_filepath, manifest_data)
    return new_manifest_filepath


def generate_vad_frame_pred(
    vad_model: EncDecClassificationModel,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    manifest_vad_input: str,
    out_dir: str,
    use_feat: bool = False,
    record_fn: Callable = None,
) -> str:
    """
    Generate VAD frame level prediction and write to out_dir
    """
    time_unit = int(window_length_in_sec / shift_length_in_sec)
    trunc = int(time_unit / 2)
    trunc_l = time_unit - trunc
    all_len = 0

    data = []
    with open(manifest_vad_input, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            file = json.loads(line)['audio_filepath'].split("/")[-1]
            data.append(file.split(".wav")[0])
    logging.info(f"Inference on {len(data)} audio files/json lines!")

    status = get_vad_stream_status(data)

    with record_fn("vad_infer_loop"):
        for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
            test_batch = [x.to(vad_model.device) for x in test_batch]
            with autocast():
                with record_fn("vad_infer_model"):
                    if use_feat:
                        log_probs = vad_model(processed_signal=test_batch[0], processed_signal_length=test_batch[1])
                    else:
                        log_probs = vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])

                with record_fn("vad_infer_other"):
                    probs = torch.softmax(log_probs, dim=-1)
                    if len(probs.shape) == 3:
                        # squeeze the batch dimension, since batch size is 1
                        probs = probs.squeeze(0)  # [1,T,C] -> [T,C]
                    pred = probs[:, 1]

                    if window_length_in_sec == 0:
                        to_save = pred
                    elif status[i] == 'start':
                        to_save = pred[:-trunc]
                    elif status[i] == 'next':
                        to_save = pred[trunc:-trunc_l]
                    elif status[i] == 'end':
                        to_save = pred[trunc_l:]
                    else:
                        to_save = pred

                    to_save = to_save.cpu().tolist()
                    all_len += len(to_save)

                    outpath = os.path.join(out_dir, data[i] + ".frame")
                    with open(outpath, "a", encoding='utf-8') as fout:
                        for p in to_save:
                            fout.write(f'{p:0.4f}\n')

                    del test_batch
                    if status[i] == 'end' or status[i] == 'single':
                        all_len = 0
    return out_dir


def init_asr_model(model_path: str) -> ASRModel:
    if model_path.endswith('.nemo'):
        logging.info(f"Using local ASR model from {model_path}")
        asr_model = ASRModel.restore_from(restore_path=model_path)
    elif model_path.endswith('.ckpt'):
        asr_model = ASRModel.load_from_checkpoint(checkpoint_path=model_path)
    else:
        logging.info(f"Using NGC ASR model {model_path}")
        asr_model = ASRModel.from_pretrained(model_name=model_path)
    return asr_model


def run_asr_inference(manifest_filepath, cfg, record_fn) -> str:
    logging.info("Start ASR inference pipeline...")
    asr_model = init_asr_model(cfg.asr_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = asr_model.to(device)
    asr_model.eval()

    # Setup decoding strategy
    decode_function = None
    decoder_type = cfg.get("decoder_type", None)
    if not hasattr(asr_model, 'change_decoding_strategy'):
        raise ValueError(f"ASR model {cfg.asr_model} does not support decoding strategy.")
    if decoder_type is not None:  # Hybrid model
        if decoder_type == 'rnnt':
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            asr_model.change_decoding_strategy(cfg.rnnt_decoding, decoder_type=decoder_type)
            decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
        elif decoder_type == 'ctc':
            asr_model.change_decoding_strategy(cfg.ctc_decoding, decoder_type=decoder_type)
            decode_function = asr_model.decoding.ctc_decoder_predictions_tensor
        else:
            raise ValueError(
                f"Unknown decoder type for hybrid model: {decoder_type}, supported types: ['rnnt', 'ctc']"
            )
    elif hasattr(asr_model, 'joint'):  # RNNT model
        cfg.rnnt_decoding.fused_batch_size = -1
        cfg.rnnt_decoding.compute_langs = cfg.compute_langs
        asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
    else:
        asr_model.change_decoding_strategy(cfg.ctc_decoding)
        decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.pred_name_postfix is not None:
            cfg.output_filename = cfg.manifest_filepath.replace('.json', f'_{cfg.pred_name_postfix}.json')
        else:
            tag = f"{cfg.normalize}_{cfg.normalize_type}"
            if cfg.use_rttm:
                vad_tag = Path(manifest_filepath).stem
                vad_tag = vad_tag[len("temp_manifest_vad_rttm_") :]
                if cfg.rttm_mode == "mask":
                    tag += f"-mask{cfg.feat_mask_val}-{vad_tag}"
                else:
                    tag += f"-dropframe-{vad_tag}"
            cfg.output_filename = cfg.manifest_filepath.replace('.json', f'-{Path(cfg.asr_model).stem}-{tag}.json')
        cfg.output_filename = Path(cfg.output_dir) / Path(cfg.output_filename).name

    logging.info("Setting up dataloader for ASR...")
    data_config = {
        "manifest_filepath": manifest_filepath,
        "normalize": cfg.normalize,
        "normalize_type": cfg.normalize_type,
        "use_rttm": cfg.use_rttm,
        "rttm_mode": cfg.rttm_mode,
        "feat_mask_val": cfg.feat_mask_val,
        "frame_unit_time_secs": cfg.frame_unit_time_secs,
    }
    logging.info(f"use_rttm = {cfg.use_rttm}, rttm_mode = {cfg.rttm_mode}, feat_mask_val = {cfg.feat_mask_val}")

    if hasattr(asr_model, "tokenizer"):
        dataset = feature_to_text_dataset.get_bpe_dataset(config=data_config, tokenizer=asr_model.tokenizer)
    else:
        data_config["labels"] = asr_model.decoder.vocabulary
        dataset = feature_to_text_dataset.get_char_dataset(config=data_config)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset._collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
    )

    logging.info("Start transcribing...")
    hypotheses = []
    all_hypotheses = []
    t0 = time.time()
    with autocast():
        with torch.no_grad():
            with record_fn("asr_infer_loop"):
                for test_batch in tqdm(dataloader, desc="Transcribing"):
                    with record_fn("asr_infer_model"):
                        outputs = asr_model.forward(
                            processed_signal=test_batch[0].to(device),
                            processed_signal_length=test_batch[1].to(device),
                        )

                    with record_fn("asr_infer_other"):
                        logits, logits_len = outputs[0], outputs[1]

                        current_hypotheses, all_hyp = decode_function(logits, logits_len, return_hypotheses=False,)
                        if isinstance(current_hypotheses, tuple) and len(current_hypotheses) == 2:
                            current_hypotheses = current_hypotheses[0]  # handle RNNT output

                        hypotheses += current_hypotheses
                        if all_hyp is not None:
                            all_hypotheses += all_hyp
                        else:
                            all_hypotheses += current_hypotheses

                        del logits
                        del test_batch
    t1 = time.time()
    logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")

    logging.info("Finished transcribing.")
    # Save output to manifest
    input_manifest_data = read_manifest(manifest_filepath)
    manifest_data = read_manifest(cfg.manifest_filepath)

    if "text" not in manifest_data[0]:
        has_groundtruth = False
    else:
        has_groundtruth = True

    groundtruth = []
    for i in range(len(manifest_data)):
        if has_groundtruth:
            groundtruth.append(manifest_data[i]["text"])
        manifest_data[i]["pred_text"] = hypotheses[i]
        manifest_data[i]["feature_file"] = input_manifest_data[i]["feature_file"]
        if "rttm_file" in input_manifest_data[i]:
            manifest_data[i]["feature_file"] = input_manifest_data[i]["feature_file"]

    write_manifest(cfg.output_filename, manifest_data)

    if not has_groundtruth:
        hypotheses = " ".join(hypotheses)
        words = hypotheses.split()
        chars = "".join(words)
        logging.info("-----------------------------------------")
        logging.info(f"Number of generated characters={len(chars)}")
        logging.info(f"Number of generated words={len(words)}")
        logging.info("-----------------------------------------")
    else:
        wer_score = word_error_rate(hypotheses=hypotheses, references=groundtruth)
        cer_score = word_error_rate(hypotheses=hypotheses, references=groundtruth, use_cer=True)
        logging.info("-----------------------------------------")
        logging.info(f"WER={wer_score:.4f}, CER={cer_score:.4f}")
        logging.info("-----------------------------------------")

    logging.info(f"ASR output saved at {cfg.output_filename}")
    return cfg.output_filename


if __name__ == "__main__":
    main()
