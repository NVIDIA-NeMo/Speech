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
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Union, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scripts.magpietts.evalset_config as evalset_config
import scripts.magpietts.evaluate_generated_audio as evaluate_generated_audio
import soundfile as sf
import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from PIL import Image
from scripts.magpietts.infer_and_evaluate import (
    compute_mean_and_confidence_interval,
    create_combined_violin_plots,
    create_violin_plots,
    delete_old_generated_files,
    setup_argument_parser,
    update_ckpt,
    update_config,
)
from torch.utils.data import Dataset, DataLoader

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers
from nemo.collections.tts.models import MagpieTTSStreamingInference
from nemo.collections.tts.parts.utils.tts_dataset_utils import _read_audio, stack_tensors

# EVALUATION_DATASETS is the full list of datasets for evaluation of a new model.
EVALUATION_DATASETS = (
    "riva_hard_digits,riva_hard_letters,riva_hard_money,riva_hard_short,vctk,libritts_seen,libritts_test_clean"
)


def split_by_sentence(
    paragraph: str,
    sentence_separators: Union[str, List[str]] = ['.', '?', '!', '...']
) -> List[str]:
    """
    Splits a paragraph into sentences based on sentence-ending punctuation.
    
    This method handles edge cases like abbreviations (e.g., "Dr.", "Mr.", "a.m.") by checking
    if the separator is followed by a space before splitting. Sentence-ending punctuation is
    preserved with each sentence.
    
    Args:
        paragraph (str): The input text paragraph to split into sentences.
        sentence_separators (Union[str, List[str]]): A string or list of strings representing
            sentence-ending punctuation marks. Defaults to ['.', '?', '!', '...'].
    
    Returns:
        List[str]: A list of sentence strings with punctuation preserved.
    
    Examples:
        >>> model.chunk_and_tokenize_text_sentence("Hello world. How are you?")
        ["Hello world.", "How are you?"]
        
        >>> model.chunk_and_tokenize_text_sentence("Dr. Smith is here. Good morning!")
        ["Dr. Smith is here.", "Good morning!"]
        
        >>> model.chunk_and_tokenize_text_sentence("Really? Yes! Amazing.")
        ["Really?", "Yes!", "Amazing."]
    """
    if not paragraph or not paragraph.strip():
        return []
    paragraph = paragraph.replace('-', ' ')
    paragraph = paragraph.replace('*', '')
    
    # Normalize to list if single string is provided
    if isinstance(sentence_separators, str):
        sentence_separators = [sentence_separators]
    
    sentences = []
    last_sep_idx = -1
    
    for i, char in enumerate(paragraph):
        # Check if the current character is a separator and the next character is a space
        # The additional space check is done to avoid splitting words like "a.m." or "Dr." into sentences
        next_char = paragraph[i + 1] if i + 1 < len(paragraph) else ""
        if char in sentence_separators and next_char == " ":
            sentences.append(paragraph[last_sep_idx + 1 : i + 1].strip())
            last_sep_idx = i + 1
    
    # Add the remaining text as the last sentence
    if last_sep_idx < len(paragraph):
        sentences.append(paragraph[last_sep_idx + 1 :].strip())
    
    # Remove any empty sentences
    sentences = [sent for sent in sentences if len(sent) > 0]
    
    # Capitalize the first letter of each sentence if it's not already capitalized
    sentences = [sent if sent[0].isupper() else sent[0].upper() + sent[1:] for sent in sentences]
    
    # Add '.' to the beginning of each sentence
    sentences = [': ' + sent for sent in sentences]
    
    return sentences


def chunk_and_tokenize_text_sentence(
    text, text_chunk_size, num_chunk_per_window, tokenizer_name, text_tokenizer, eos_token_id, start_of_generation=True
):
    split_sentences = split_by_sentence(text)
    chunked_tokens = []
    chunked_tokens_len = []
    chunked_text = []
    for idx, sentence in enumerate(split_sentences):
        # Add a space betweenthe end of the sentence and the punctuation mark.
        sentence = sentence[:-1] + " " + sentence[-1]
        chunked_text.append(sentence)
        print(f"sentence {sentence}")
        tokens = text_tokenizer.encode(text=sentence, tokenizer_name=tokenizer_name)
        # TODO(sugh): Add EOS token to every sentence.
        if idx == len(split_sentences) - 1:
            tokens = tokens + [eos_token_id]
        tokens = torch.tensor(tokens, dtype=torch.int32)
        print(f"i {idx} tokens {tokens.shape}")
        tokens_len = tokens.shape[0]
        chunked_tokens.append(tokens)
        chunked_tokens_len.append(tokens_len)
    print(f"chunked_tokens {sum(chunked_tokens_len)}")
    return chunked_tokens, chunked_tokens_len, chunked_text


def chunk_and_tokenize_text(
    text, text_chunk_size, num_chunk_per_window, tokenizer_name, text_tokenizer, eos_token_id, start_of_generation=True
):
    split_text = text.split()
    print(f"split_text {len(split_text)}")  # []
    chunked_tokens = []
    chunked_tokens_len = []
    chunked_text = []
    start = num_chunk_per_window * text_chunk_size if start_of_generation else text_chunk_size
    print(f"text len {len(split_text[:start])}")
    current_text = " ".join(split_text[:start])

    chunked_text.append(current_text)
    tokens = text_tokenizer.encode(text=current_text, tokenizer_name=tokenizer_name)
    tokens = torch.tensor(tokens, dtype=torch.int32)
    print(f"tokens {tokens.shape}")

    tokens_len = tokens.shape[0]
    chunked_tokens.append(tokens)
    chunked_tokens_len.append(tokens_len)

    for i in range(start, len(split_text), text_chunk_size):
        print(f"i text len {len(split_text[i : min(i + text_chunk_size, len(split_text))])}")
        current_text = " ".join(split_text[i : min(i + text_chunk_size, len(split_text))])
        chunked_text.append(current_text)
        tokens = text_tokenizer.encode(text=current_text, tokenizer_name=tokenizer_name)
        if i + text_chunk_size >= len(split_text):
            tokens = tokens + [eos_token_id]
        tokens = torch.tensor(tokens, dtype=torch.int32)
        print(f"i {i} tokens {tokens.shape}")
        tokens_len = tokens.shape[0]
        chunked_tokens.append(tokens)
        chunked_tokens_len.append(tokens_len)
    print(f"chunked_tokens {sum(chunked_tokens_len)}")

    return chunked_tokens, chunked_tokens_len, chunked_text


class LongFormTTSDataset(Dataset):
    """
    Dataset class for long-form TTS inference with batching support.
    
    This dataset handles:
    - Loading manifest entries
    - Text tokenization and chunking
    - Context audio code loading and preprocessing
    
    Args:
        manifest_records: List of manifest entries
        text_tokenizer: Text tokenizer instance
        tokenizer_name: Name of the tokenizer to use
        eos_id: End-of-sequence token ID
        text_chunk_size: Number of words per text chunk
        num_chunk_per_window: Number of chunks per window
        dataset_meta: Dataset metadata dictionary
        dataset_name: Name of the dataset
        model: Model instance for audio encoding
        context_duration_min: Minimum context duration in seconds
        context_duration_max: Maximum context duration in seconds
        sample_rate: Audio sample rate
        codec_model_downsample_factor: Codec model downsampling factor
        context_audio_bos_id: Context audio BOS token ID
        context_audio_eos_id: Context audio EOS token ID
    """
    
    def __init__(
        self,
        manifest_records: List[Dict[str, Any]],
        text_tokenizer: Any,
        tokenizer_name: str,
        eos_id: int,
        text_chunk_size: int,
        num_chunk_per_window: int,
        dataset_meta: Dict[str, Any],
        dataset_name: str,
        model: Any,
        context_duration_min: float,
        context_duration_max: float,
        sample_rate: int,
        codec_model_downsample_factor: int,
        context_audio_bos_id: int,
        context_audio_eos_id: int,
        use_text_conditioning_encoder: bool,
        pad_context_text_to_max_duration: bool,
        codec_model_samples_per_frame: int,
    ):
        self.manifest_records = manifest_records
        self.text_tokenizer = text_tokenizer
        self.tokenizer_name = tokenizer_name
        self.eos_id = eos_id
        self.text_chunk_size = text_chunk_size
        self.num_chunk_per_window = num_chunk_per_window
        self.dataset_meta = dataset_meta
        self.dataset_name = dataset_name
        self.model = model
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max
        self.sample_rate = sample_rate
        self.codec_model_downsample_factor = codec_model_downsample_factor
        self.context_audio_bos_id = context_audio_bos_id
        self.context_audio_eos_id = context_audio_eos_id
        self.pad_context_text_to_max_duration = pad_context_text_to_max_duration
        self.use_text_conditioning_encoder = use_text_conditioning_encoder
        self.codec_model_samples_per_frame = codec_model_samples_per_frame
    
    def __len__(self) -> int:
        return len(self.manifest_records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and preprocess a single sample.
        
        Returns:
            Dictionary containing:
                - idx: Sample index
                - chunked_tokens: List of tokenized text chunks
                - chunked_tokens_len: List of token lengths
                - chunked_text_list: List of text strings
                - context_audio_codes: Context audio codes tensor
                - entry: Original manifest entry
        """
        entry = self.manifest_records[idx]
        
        # Get text
        if "normalized_text" in entry:
            text = entry["normalized_text"]
        else:
            text = entry["text"]
        
        # Tokenize and chunk text
        chunked_tokens, chunked_tokens_len, chunked_text_list = chunk_and_tokenize_text_sentence(
            text,
            self.text_chunk_size,
            self.num_chunk_per_window,
            self.tokenizer_name,
            self.text_tokenizer,
            self.eos_id,
        )
        
        # Load context audio codes
        if 'context_audio_codes_path' in entry:
            context_audio_codes_path = entry['context_audio_codes_path']
            context_audio_codes = torch.load(context_audio_codes_path).long()  # (8, T)
        elif 'context_audio_filepath' in entry:
            context_audio_filepath = entry['context_audio_filepath']
            if self.dataset_meta[self.dataset_name]['audio_dir'] is not None:
                context_audio_filepath = os.path.join(
                    self.dataset_meta[self.dataset_name]['audio_dir'], context_audio_filepath
                )
            context_audio_duration = entry['context_audio_duration']
            context_audio_array = _read_audio(
                audio_filepath=context_audio_filepath,
                sample_rate=self.sample_rate,
                offset=0,
                duration=context_audio_duration,
            )
            context_audio_array = context_audio_array.samples
            context_audio_array = torch.tensor(context_audio_array).unsqueeze(0).cuda()
            context_audio_len = torch.tensor(context_audio_array.shape[1]).unsqueeze(0).cuda()
            context_audio_codes, _ = self.model.audio_to_codes(
                context_audio_array, context_audio_len, audio_type='context'
            )
            context_audio_codes = torch.tensor(context_audio_codes).squeeze(0).cpu()
        else:
            raise ValueError(f"Context audio codes path or filepath not found in manifest entry: {entry}")
        
        # Randomly slice context audio to desired duration
        _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
        _num_frames_to_slice = int(_context_duration_to_slice * self.sample_rate / self.codec_model_downsample_factor)
        
        if _num_frames_to_slice < context_audio_codes.shape[1]:
            start_idx = random.randint(0, context_audio_codes.shape[1] - _num_frames_to_slice)
            context_audio_codes = context_audio_codes[:, start_idx : start_idx + _num_frames_to_slice]
        else:
            # Repeat the audio if it is shorter than the desired duration
            _num_repeats = int(np.ceil(_num_frames_to_slice / context_audio_codes.shape[1]))
            context_audio_codes_repeated = context_audio_codes.repeat(1, _num_repeats)
            context_audio_codes = context_audio_codes_repeated[:, :_num_frames_to_slice]
        
        # Add BOS and EOS tokens
        context_bos_tensor = torch.full(
            (context_audio_codes.shape[0], 1), self.context_audio_bos_id, dtype=context_audio_codes.dtype
        )
        context_eos_tensor = torch.full(
            (context_audio_codes.shape[0], 1), self.context_audio_eos_id, dtype=context_audio_codes.dtype
        )
        context_audio_codes = torch.cat([context_bos_tensor, context_audio_codes, context_eos_tensor], dim=1)
        
        return {
            'idx': idx,
            'chunked_tokens': chunked_tokens,
            'chunked_tokens_len': chunked_tokens_len,
            'context_audio_codes': context_audio_codes,
            'entry': entry,
        }


    def collate_fn(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collate function for batching.
        
        Since each sample may have different sequence lengths and the streaming
        inference processes samples independently, we simply return the batch as a list.
        
        Args:
            batch: List of sample dictionaries from __getitem__
        
        Returns:
            List of sample dictionaries (no padding/stacking needed)
        """
        s_idx = []
        context_audio_codes_list = []
        context_audio_codes_lens_list = []
        has_text_context_list = []
        context_text_tokens_list = []
        context_text_tokens_lens_list = []
        current_tokens = [] # List of lists of token tensors for each sample in the batch [[t,t,t,...], [t,t,t,...], ...]
        current_tokens_lens = [] # List of lists of token lengths for each sample in the batch [[l,l,l,...], [l,l,l,...], ...]

        max_num_chunks = max([len(sample['chunked_tokens']) for sample in batch])
        for sample in batch:
            s_idx.append(sample['idx'])
            num_padding = max_num_chunks - len(sample['chunked_tokens'])
            sample['chunked_tokens'] = sample['chunked_tokens'] + [
                torch.tensor(self.eos_id, dtype=torch.int32).unsqueeze(0) for _ in range(num_padding)
            ]
            current_tokens.append(sample['chunked_tokens'])
            sample['chunked_tokens_len'] = sample['chunked_tokens_len'] + [
                torch.tensor(1, dtype=torch.int32) for _ in range(num_padding)
            ]
            current_tokens_lens.append(sample['chunked_tokens_len'])

            context_audio_codes_list.append(sample['context_audio_codes'])

            context_audio_codes_lens_list.append(torch.tensor([sample['context_audio_codes'].shape[1]]))
            if self.use_text_conditioning_encoder:
                text_context_tokens = self.text_tokenizer.encode(
                    "[NO TEXT CONTEXT]",
                    "english_phoneme" if self.tokenizer_name is None else self.tokenizer_name
                )
                if self.pad_context_text_to_max_duration:
                    _required_len = (
                        int(self.context_duration_max * self.sample_rate / self.codec_model_samples_per_frame) + 2
                    )  # +2 for BOS and EOS
                    if len(text_context_tokens) < _required_len:
                        _pad_id = self.text_tokenizer.tokenizer_pad_ids["english_phoneme"]
                        text_context_tokens += [_pad_id] * (_required_len - len(text_context_tokens))
                    else:
                        text_context_tokens = text_context_tokens[:_required_len]
                text_context_tokens = torch.tensor(text_context_tokens, dtype=torch.int32).cuda()
                context_text_len = torch.tensor([text_context_tokens.shape[0]]).cuda()

                context_text_tokens_list.append(text_context_tokens)
                context_text_tokens_lens_list.append(context_text_len)
                has_text_context_list.append(torch.BoolTensor([False]))

        context_text_max_len = max([ct.shape[0] for ct in context_text_tokens_list])
        context_audio_codes_max_len = max([ca.shape[1] for ca in context_audio_codes_list])
        batch_dict = {
            'idx': s_idx,
            'chunked_tokens': current_tokens,
            'chunked_tokens_lens': current_tokens_lens,
            'context_audio_codes': stack_tensors(context_audio_codes_list, max_lens=[context_audio_codes_max_len]).cuda(),
            'context_audio_codes_lens': torch.IntTensor(context_audio_codes_lens_list).cuda(),
            'has_text_context': torch.BoolTensor(has_text_context_list).cuda(),
            'context_text_tokens': stack_tensors(context_text_tokens_list, max_lens=[context_text_max_len]).cuda(),
            'context_text_tokens_lens': torch.IntTensor(context_text_tokens_lens_list).cuda(),
        }

        return batch_dict


def run_inference_streaming(
    hparams_file,
    checkpoint_file,
    nemo_file,
    datasets,
    out_dir,
    temperature,
    topk,
    codecmodel_path,
    use_cfg,
    cfg_scale,
    batch_size,
    sv_model,
    asr_model_name,
    num_repeats=1,
    apply_attention_prior=False,
    attention_prior_epsilon=1e-3,
    attention_prior_lookahead_window=10,
    estimate_alignment_from_layers=None,
    apply_prior_to_layers=None,
    start_prior_after_n_audio_steps=10,
    confidence_level=0.95,
    use_exponential_weight=False,
    use_local_transformer=False,
    maskgit_n_steps=3,
    legacy_codebooks=False,
    legacy_text_conditioning=False,
    clean_up_disk=False,
    hparams_file_from_wandb=False,
    log_exp_name=False,
    compute_fcd=False,
    violin_plot_metrics=['cer', 'pred_context_ssim'],
    tokenizer_name=None,
    eos_detection_method='argmax_or_multinomial_any',
):
    num_chunk_per_window = 2
    true_window_size = 300 # Unit is number of text tokens
    text_chunk_size = 20 # Unit is number of words
    # Load model
    if hparams_file is not None and checkpoint_file is not None:
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        if hparams_file_from_wandb:
            model_cfg = model_cfg.value

        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(
                model_cfg, codecmodel_path, legacy_codebooks, legacy_text_conditioning
            )

        model = MagpieTTSStreamingInference(cfg=model_cfg)
        # use_kv_cache_for_inference is not enabled for streaming inference
        model.use_kv_cache_for_inference = False

        # Load weights from checkpoint file
        print("Loading weights from checkpoint")
        ckpt = torch.load(checkpoint_file, weights_only=False)
        state_dict = update_ckpt(ckpt['state_dict'])
        model.load_state_dict(state_dict)
        checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    elif nemo_file is not None:
        model_cfg = MagpieTTSStreamingInference.restore_from(nemo_file, return_config=True)
        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(
                model_cfg, codecmodel_path, legacy_codebooks, legacy_text_conditioning
            )
        model = MagpieTTSStreamingInference.restore_from(nemo_file, override_config_path=model_cfg)
        # use_kv_cache_for_inference is not enabled for streaming inference
        model.use_kv_cache_for_inference = False
        checkpoint_name = nemo_file.split("/")[-1].split(".nemo")[0]
    else:
        raise ValueError("Need either a checkpoint and hparams file, or a nemo file.")

    if cfg_sample_rate is not None and cfg_sample_rate != model.sample_rate:
        raise ValueError("Sample rate in config and model do not match")

    print("Loaded weights.")
    model.cuda()
    model.eval()

    text_tokenizer = setup_tokenizers(model.cfg.text_tokenizers, mode='test')

    if log_exp_name:
        # the experiment name is the name of the directory two above the checkpoint path,
        # since training produces directories of the form `exp_name/checkpoints/checkpoint_name.ckpt`.
        exp_name = f"{os.path.basename(os.path.dirname(os.path.dirname(checkpoint_file)))}__"
    else:
        exp_name = ""

    checkpoint_name = "{}{}_Temp{}_Topk{}_Cfg_{}_{}_Prior_{}_LT_{}_MGsteps_{}_ST_{}_sched_{}_EOS_{}".format(
        exp_name,
        checkpoint_name,
        temperature,
        topk,
        use_cfg,
        cfg_scale,
        apply_attention_prior,
        attention_prior_epsilon,
        attention_prior_lookahead_window,
        start_prior_after_n_audio_steps,
        (
            "".join([str(l) for l in estimate_alignment_from_layers])
            if estimate_alignment_from_layers is not None
            else "None"
        ),
        "".join([str(l) for l in apply_prior_to_layers]) if apply_prior_to_layers is not None else "None",
        use_local_transformer,
        maskgit_n_steps,
        sv_model,
        eos_detection_method,
    )

    dataset_meta_info = evalset_config.dataset_meta_info
    ssim_per_dataset = []
    cer_per_dataset = []
    all_datasets_filewise_metrics = {}  # Store filewise metrics for all datasets for combined violin plot
    for dataset in datasets:
        print(f"Evaluating dataset {dataset}")
        metrics_n_repeated = []
        manifest_records = read_manifest(dataset_meta_info[dataset]['manifest_path'])
        language = dataset_meta_info[dataset].get('whisper_language', 'en')
        dataset_meta_for_dl = copy.deepcopy(dataset_meta_info[dataset])
        for key in ["whisper_language", "load_cached_codes_if_available"]:
            if key in dataset_meta_for_dl:
                del dataset_meta_for_dl[key]

        dataset_meta = {dataset: dataset_meta_for_dl}

        eval_dir = os.path.join(out_dir, f"{checkpoint_name}_{dataset}")
        audio_dir = os.path.join(eval_dir, "audio")
        pred_audio_dir = os.path.join(audio_dir, f"repeat_0")

        os.makedirs(eval_dir, exist_ok=True)
        all_experiment_csv = os.path.join(eval_dir, "all_experiment_metrics.csv")
        os.makedirs(pred_audio_dir, exist_ok=True)
        delete_old_generated_files(pred_audio_dir)

        if not os.path.exists(all_experiment_csv):
            with open(all_experiment_csv, "w") as f:
                header = "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative"
                if compute_fcd:
                    header += ",frechet_codec_distance"
                header += "\n"
                f.write(header)

        context_duration_min = model.cfg.get('context_duration_min', 5.0)
        context_duration_max = model.cfg.get('context_duration_max', 5.0)
        codec_model_downsample_factor = (
            model_cfg.codec_model_downsample_factor
            if "codec_model_downsample_factor" in model_cfg
            else model._codec_model.samples_per_frame
        )
        sample_rate = model_cfg.sample_rate if "sample_rate" in model_cfg else model.sample_rate
        if context_duration_min < 5.0 and context_duration_max > 5.0:
            context_duration_min = 5.0
            context_duration_max = 5.0
        context_audio_bos_id = model.context_audio_bos_id
        context_audio_eos_id = model.context_audio_eos_id
        audio_bos_id = model.audio_bos_id
        audio_eos_id = model.audio_eos_id

        metrics_n_repeated = []
        dataset_filewise_metrics_all_repeats = []

        # Create dataset and dataloader
        print(f"manifest_records {len(manifest_records)}")
        inference_dataset = LongFormTTSDataset(
            manifest_records=manifest_records,
            text_tokenizer=text_tokenizer,
            tokenizer_name="english_phoneme" if tokenizer_name is None else tokenizer_name,
            eos_id=model.eos_id,
            text_chunk_size=text_chunk_size,
            num_chunk_per_window=num_chunk_per_window,
            dataset_meta=dataset_meta,
            dataset_name=dataset,
            model=model,
            context_duration_min=context_duration_min,
            context_duration_max=context_duration_max,
            sample_rate=sample_rate,
            codec_model_downsample_factor=codec_model_downsample_factor,
            context_audio_bos_id=context_audio_bos_id,
            context_audio_eos_id=context_audio_eos_id,
            use_text_conditioning_encoder=model.use_text_conditioning_encoder,
            pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
            codec_model_samples_per_frame=model.codec_model_samples_per_frame,
        )
        
        dataloader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with CUDA
            collate_fn=inference_dataset.collate_fn,
        )
        
        # Iterate over batches
        for batch in dataloader:
            batch_size = len(batch['chunked_tokens'])
            model.set_streaming_inference_variables(batch_size=batch_size, true_window_size=true_window_size)
            max_num_chunks = max([len(tokens) for tokens in batch['chunked_tokens']])
            # Process each sample in the batch
            # Prepare batch dictionary for model
            # batch = {}
            # context_audio_codes_list = []
            # context_audio_codes_lens_list = []
            # has_text_context_list = []
            # context_text_tokens_list = []
            # context_text_tokens_lens_list = []
            # for sample in batch_samples:
            #     context_audio_codes_list.append(sample['context_audio_codes'])
            #     context_audio_codes_lens_list.append(torch.tensor([sample['context_audio_codes'].shape[1]]))
                
            #     if model.use_text_conditioning_encoder:
            #         text_context_tokens = text_tokenizer.encode(
            #             "[NO TEXT CONTEXT]",
            #             "english_phoneme" if tokenizer_name is None else tokenizer_name
            #         )

            #         if model.pad_context_text_to_max_duration:
            #             _required_len = (
            #                 int(model.cfg.context_duration_max * sample_rate / model.codec_model_samples_per_frame) + 2
            #             )  # +2 for BOS and EOS
            #             if len(text_context_tokens) < _required_len:
            #                 _pad_id = text_tokenizer.tokenizer_pad_ids["english_phoneme"]
            #                 text_context_tokens += [_pad_id] * (_required_len - len(text_context_tokens))
            #             else:
            #                 text_context_tokens = text_context_tokens[:_required_len]

            #         text_context_tokens = torch.tensor(text_context_tokens, dtype=torch.int32).cuda()
            #         context_text_len = torch.tensor([text_context_tokens.shape[0]]).cuda()

            #         context_text_tokens_list.append(text_context_tokens)
            #         context_text_tokens_lens_list.append(context_text_len)
            #         has_text_context_list.append(torch.BoolTensor([False]))

            # text_max_len = max([ct.shape[0] for ct in context_text_tokens_list])
            # audio_max_len = max([ca.shape[1] for ca in context_audio_codes_list])
            # batch['context_audio_codes'] = stack_tensors(context_audio_codes_list, max_lens=[audio_max_len]).cuda()
            # batch['context_audio_codes_lens'] = torch.IntTensor(context_audio_codes_lens_list).cuda()
            # batch['has_text_context'] = torch.BoolTensor(has_text_context_list).cuda()
            # batch['context_text_tokens'] = stack_tensors(context_text_tokens_list, max_lens=[text_max_len]).cuda()
            # batch['context_text_tokens_lens'] = torch.IntTensor(context_text_tokens_lens_list).cuda()

            predicted_codes = [[] for _ in range(batch_size)]
            predicted_codes_lens = [0 for _ in range(batch_size)]
            torch.cuda.empty_cache()

            st = time.time()

            for token_idx in range(max_num_chunks):

                # Extract and pad current chunk of text tokens.
                current_tokens = []
                current_tokens_lens = []
                for b_idx in range(batch_size):
                    current_tokens.append(batch['chunked_tokens'][b_idx][token_idx])
                    current_tokens_lens.append(batch['chunked_tokens_lens'][b_idx][token_idx])

                max_len = max(current_tokens_lens)
                batch['text'] = stack_tensors(current_tokens, max_lens=[max_len]).cuda()
                batch['text_lens'] = torch.IntTensor(current_tokens_lens).cuda()
                print(f"batch['text_lens'] {batch['text_lens']}")

                # Compute is_end_of_text flags for each sample in the batch.
                is_end_of_text = []
                for b_idx in range(batch_size):
                    if token_idx == (max_num_chunks - 1):
                        # This sample goes till the end (maximum number of chunks).
                        is_end_of_text.append(True)
                    elif current_tokens_lens[b_idx] == 1:
                        # Text chunks have ended for this sample.
                        is_end_of_text.append(True)
                    elif batch['chunked_tokens_lens'][b_idx][token_idx + 1] == 1:
                        # This sample ends in this iteration, Hence next chunk of text tokens len is 1.
                        is_end_of_text.append(True)
                    else:
                        is_end_of_text.append(False)
                print(f"is_end_of_text {is_end_of_text}")

                beginning_of_text = token_idx == 0
                current_predicted_codes, current_predicted_codes_lens, _, _ = (
                    model.generate_long_form_speech(
                        batch,
                        is_end_of_text,
                        beginning_of_text,
                        max_decoder_steps=50000,
                        temperature=temperature,
                        topk=topk,
                        use_cfg=use_cfg,
                        cfg_scale=cfg_scale,
                        return_cross_attn_probs=False,
                        apply_attention_prior=apply_attention_prior,
                        prior_epsilon=attention_prior_epsilon,
                        lookahead_window_size=attention_prior_lookahead_window,
                        estimate_alignment_from_layers=estimate_alignment_from_layers,
                        apply_prior_to_layers=apply_prior_to_layers,
                        start_prior_after_n_audio_steps=start_prior_after_n_audio_steps,
                        use_exponential_weight=use_exponential_weight,
                        eos_detection_method=eos_detection_method,
                        ignore_finished_sentence_tracking=False,
                    )
                )
                for b_idx in range(batch_size):
                    if is_end_of_text[b_idx] and current_tokens_lens[b_idx] == 1:
                        continue
                    predicted_codes[b_idx].append(current_predicted_codes[b_idx][:, :current_predicted_codes_lens[b_idx]])
                    predicted_codes_lens[b_idx] += current_predicted_codes_lens[b_idx]

            et = time.time()
            print(f"Magpie Time taken for inference: {et - st} seconds")
            torch.cuda.empty_cache()

            predicted_codes_list = []
            for b_idx, predicted_code in enumerate(predicted_codes):
                predicted_codes_list.append(torch.cat(predicted_code, dim=1).cuda())

            predicted_codes = stack_tensors(predicted_codes_list, max_lens=[max(predicted_codes_lens)]).cuda()
            predicted_codes_lens = torch.tensor(predicted_codes_lens).long().cuda()
            predicted_audio, _ = model.codes_to_audio(predicted_codes, predicted_codes_lens)
            predicted_audio_np = predicted_audio.squeeze(0).float().detach().cpu().numpy()

            print(f"Total Time taken for inference: {time.time() - st} seconds")
            for b_idx, s_idx in enumerate(batch['idx']):
                audio_path = os.path.join(pred_audio_dir, f"predicted_audio_{s_idx}.wav")
                sf.write(audio_path, predicted_audio_np[b_idx], sample_rate)

        metrics, filewise_metrics = evaluate_generated_audio.evaluate(
            dataset_meta[dataset]['manifest_path'],
            dataset_meta[dataset]['audio_dir'],
            pred_audio_dir,
            language=language,
            sv_model_type=sv_model,
            asr_model_name=asr_model_name,
            codecmodel_path=codecmodel_path if compute_fcd else None,
        )
        metrics_n_repeated.append(metrics)
        dataset_filewise_metrics_all_repeats.extend(filewise_metrics)  # Collect all filewise metrics for combined plot

        with open(os.path.join(eval_dir, f"{dataset}_metrics_0.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(os.path.join(eval_dir, f"{dataset}_filewise_metrics_0.json"), "w") as f:
            # Indent for better readability
            json.dump(filewise_metrics, f, indent=4)

        with open(all_experiment_csv, "a") as f:
            data = f"{checkpoint_name},{dataset},{metrics['cer_filewise_avg']},{metrics['wer_filewise_avg']},{metrics['cer_cumulative']},{metrics['wer_cumulative']},{metrics['ssim_pred_gt_avg']},{metrics['ssim_pred_context_avg']},{metrics['ssim_gt_context_avg']},{metrics['ssim_pred_gt_avg_alternate']},{metrics['ssim_pred_context_avg_alternate']},{metrics['ssim_gt_context_avg_alternate']},{metrics['cer_gt_audio_cumulative']},{metrics['wer_gt_audio_cumulative']}"
            if compute_fcd:
                data += f",{metrics['frechet_codec_distance']}"
            data += "\n"
            f.write(data)
            print(f"Wrote metrics for {checkpoint_name} and {dataset} to {all_experiment_csv}")

        output_png_file = Path(eval_dir) / f"{dataset}_violin_0.png"
        create_violin_plots(filewise_metrics, violin_plot_metrics, output_png_file)

        # Store filewise metrics for this dataset for combined plotting
        all_datasets_filewise_metrics[dataset] = dataset_filewise_metrics_all_repeats

        metric_keys = [
            'cer_filewise_avg',
            'wer_filewise_avg',
            'cer_cumulative',
            'wer_cumulative',
            'ssim_pred_gt_avg',
            'ssim_pred_context_avg',
            'ssim_gt_context_avg',
            'ssim_pred_gt_avg_alternate',
            'ssim_pred_context_avg_alternate',
            'ssim_gt_context_avg_alternate',
            'cer_gt_audio_cumulative',
            'wer_gt_audio_cumulative',
        ]
        if compute_fcd:
            metric_keys.append('frechet_codec_distance')
        metrics_mean_ci = compute_mean_and_confidence_interval(
            metrics_n_repeated, metric_keys, confidence=confidence_level
        )
        all_experiment_csv_with_ci = os.path.join(out_dir, "all_experiment_metrics_with_ci.csv")
        if not os.path.exists(all_experiment_csv_with_ci):
            with open(all_experiment_csv_with_ci, "w") as f:
                header = "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative"
                if compute_fcd:
                    header += ",frechet_codec_distance"
                header += "\n"
                f.write(header)
        with open(all_experiment_csv_with_ci, "a") as f:
            data = f"{checkpoint_name},{dataset},{metrics_mean_ci['cer_filewise_avg']},{metrics_mean_ci['wer_filewise_avg']},{metrics_mean_ci['cer_cumulative']},{metrics_mean_ci['wer_cumulative']},{metrics_mean_ci['ssim_pred_gt_avg']},{metrics_mean_ci['ssim_pred_context_avg']},{metrics_mean_ci['ssim_gt_context_avg']},{metrics_mean_ci['ssim_pred_gt_avg_alternate']},{metrics_mean_ci['ssim_pred_context_avg_alternate']},{metrics_mean_ci['ssim_gt_context_avg_alternate']},{metrics_mean_ci['cer_gt_audio_cumulative']},{metrics_mean_ci['wer_gt_audio_cumulative']}"
            if compute_fcd:
                data += f",{metrics_mean_ci['frechet_codec_distance']}"
            data += "\n"
            f.write(data)
            print(f"Wrote metrics with CI for {checkpoint_name} and {dataset} to {all_experiment_csv_with_ci}")

        measurements = [m['ssim_pred_context_avg'] for m in metrics_n_repeated]
        ssim_current = np.mean(measurements)
        ssim_per_dataset.append(ssim_current)
        measurements = [m['cer_cumulative'] for m in metrics_n_repeated]
        cer_current = np.mean(measurements)
        cer_per_dataset.append(cer_current)

    # Create combined violin plot for all datasets
    if len(all_datasets_filewise_metrics) > 1:  # Only create combined plot if we have multiple datasets
        combined_output_png = os.path.join(out_dir, f"{checkpoint_name}_combined_violin_plot.png")
        create_combined_violin_plots(all_datasets_filewise_metrics, violin_plot_metrics, combined_output_png)

    # Average across datasets
    ssim = np.mean(ssim_per_dataset)
    cer = np.mean(cer_per_dataset)
    if clean_up_disk:
        shutil.rmtree(out_dir)
    return cer, ssim


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = EVALUATION_DATASETS

    # FCD computation is enabled by default, disabled only when --disable_fcd is specified
    compute_fcd = not args.disable_fcd

    estimate_alignment_from_layers = None
    if args.estimate_alignment_from_layers is not None:
        estimate_alignment_from_layers = [int(l.strip()) for l in args.estimate_alignment_from_layers.split(",")]
    apply_prior_to_layers = None
    if args.apply_prior_to_layers is not None:
        apply_prior_to_layers = [int(l.strip()) for l in args.apply_prior_to_layers.split(",")]

    run_inference_w_args = partial(
        run_inference_streaming,
        datasets=args.datasets.split(","),
        out_dir=args.out_dir,
        temperature=args.temperature,
        topk=args.topk,
        codecmodel_path=args.codecmodel_path,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        sv_model=args.sv_model,
        asr_model_name=args.asr_model_name,
        num_repeats=args.num_repeats,
        apply_attention_prior=args.apply_attention_prior,
        attention_prior_epsilon=args.attention_prior_epsilon,
        attention_prior_lookahead_window=args.attention_prior_lookahead_window,
        estimate_alignment_from_layers=estimate_alignment_from_layers,
        apply_prior_to_layers=apply_prior_to_layers,
        start_prior_after_n_audio_steps=args.start_prior_after_n_audio_steps,
        confidence_level=args.confidence_level,
        use_local_transformer=args.use_local_transformer,
        maskgit_n_steps=args.maskgit_n_steps,
        legacy_codebooks=args.legacy_codebooks,
        legacy_text_conditioning=args.legacy_text_conditioning,
        clean_up_disk=args.clean_up_disk,
        hparams_file_from_wandb=args.hparams_file_from_wandb,
        log_exp_name=args.log_exp_name,
        compute_fcd=compute_fcd,
        violin_plot_metrics=args.violin_plot_metrics,
        eos_detection_method=args.eos_detection_method,
    )

    # Mode 1: Run inference from provided hparams and checkpoint files
    if (
        (args.hparams_files is not None)
        and (args.checkpoint_files is not None)
        and (args.hparams_files != "null")
        and (args.checkpoint_files != "null")
    ):
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")
        print("Running inference for hparams files: ", hparam_files)
        print("Running inference for checkpoint files: ", checkpoint_files)
        assert len(hparam_files) == len(
            checkpoint_files
        ), "Number of hparams files and checkpoint files should be the same."
        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            cer, ssim = run_inference_w_args(
                hparams_file=hparams_file,
                checkpoint_file=checkpoint_file,
                nemo_file=None,
            )
        return
    # Mode 2: Run inference from a .nemo file
    elif args.nemo_files:
        print(f"Running inference for nemo file: {args.nemo_files}")
        for nemo_file in args.nemo_files.split(","):
            cer, ssim = run_inference_w_args(
                hparams_file=None,
                checkpoint_file=None,
                nemo_file=nemo_file,
            )
    else:
        parser.error(
            "You must provide a model to run. Please specify either:\n"
            "1. --hparams_files and --checkpoint_files\n"
            "2. --nemo_file\n"
        )
    if args.cer_target is not None and cer > float(args.cer_target):
        raise ValueError()
    if args.ssim_target is not None and ssim < float(args.ssim_target):
        raise ValueError()


if __name__ == '__main__':
    main()
